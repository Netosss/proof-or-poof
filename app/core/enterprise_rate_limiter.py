"""
Per-credential rate limiting for the enterprise API.

Fixed-window counter approach:
    INCR rate:ent:{credential_id}            (atomic with EXPIRE-on-create)
    EXPIRE 60s only when count == 1          (set via Lua so the pair is atomic)

Returns a header bundle that the route includes on EVERY response (200, 4xx,
5xx). On 429 we additionally surface Retry-After so partner SDKs can back off
correctly. The limit is per-credential (not per-api_key) so rotating a key
doesn't reset budget for a partner under attack.

Falls back to per-process in-memory state if Redis is unavailable. The
in-memory limiter is shared with no other code so it's safe to keep simple.

Atomicity note: A naive `pipeline.incr + pipeline.expire` is NOT atomic across
Redis crashes — if the server dies between INCR and EXPIRE the key persists
with no TTL and rate limiting is permanently disabled for that credential.
We run both commands inside a Lua script via EVAL so the pair is one
all-or-nothing operation as seen by Redis's command log.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException

from app.config import settings
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitInfo:
    limit: int
    remaining: int
    reset_seconds: int  # seconds until window rolls
    retry_after: int | None = None  # populated only when limit exceeded


# --- In-memory fallback state -------------------------------------------------
_WINDOW_SEC = 60
_memory_buckets: dict[str, deque] = defaultdict(deque)

# Atomically INCR + (EXPIRE only on first hit). Returns the new counter.
# Running this as one Lua script is the only way to guarantee the EXPIRE
# happens even if the Redis server is killed between commands — a plain
# pipeline can leak a TTL-less key permanently.
_INCR_WITH_TTL_LUA = """
local v = redis.call('INCR', KEYS[1])
if v == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return v
"""


def _to_headers(info: RateLimitInfo) -> dict[str, str]:
    h = {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(time.time()) + info.reset_seconds),
    }
    if info.retry_after is not None:
        h["Retry-After"] = str(info.retry_after)
    return h


async def check_and_track(credential_id: str, limit_per_min: int | None = None) -> RateLimitInfo:
    """
    Increment counter and raise HTTPException(429) if over budget.
    Returns rate-limit metadata for the response headers on success.
    """
    limit = int(limit_per_min or settings.enterprise_default_rate_limit_per_min)
    rc = redis_module.client
    if rc:
        return await _check_redis(rc, credential_id, limit)
    return _check_memory(credential_id, limit)


async def _check_redis(rc, credential_id: str, limit: int) -> RateLimitInfo:
    key = f"rate:ent:{credential_id}"
    try:
        count = int(await rc.eval(_INCR_WITH_TTL_LUA, 1, key, _WINDOW_SEC))
    except Exception as e:
        logger.warning(
            "enterprise_rate_limit_redis_error",
            extra={"action": "enterprise_rate_limit_redis_error", "error": str(e)},
        )
        return _check_memory(credential_id, limit)

    ttl: int
    try:
        ttl_raw = await rc.ttl(key)
        ttl = int(ttl_raw) if ttl_raw and int(ttl_raw) > 0 else _WINDOW_SEC
    except Exception:
        ttl = _WINDOW_SEC

    remaining = max(0, limit - count)
    if count > limit:
        info = RateLimitInfo(limit=limit, remaining=0, reset_seconds=ttl, retry_after=ttl)
        raise HTTPException(
            status_code=429,
            detail={
                "type": "rate_limit_error",
                "code": "rate_limited",
                "message": "Per-minute request budget exhausted.",
            },
            headers=_to_headers(info),
        )
    return RateLimitInfo(limit=limit, remaining=remaining, reset_seconds=ttl)


def _check_memory(credential_id: str, limit: int) -> RateLimitInfo:
    now = time.time()
    bucket = _memory_buckets[credential_id]
    while bucket and now - bucket[0] >= _WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= limit:
        oldest = bucket[0]
        ttl = max(1, int(_WINDOW_SEC - (now - oldest)))
        info = RateLimitInfo(limit=limit, remaining=0, reset_seconds=ttl, retry_after=ttl)
        raise HTTPException(
            status_code=429,
            detail={
                "type": "rate_limit_error",
                "code": "rate_limited",
                "message": "Per-minute request budget exhausted.",
            },
            headers=_to_headers(info),
        )
    bucket.append(now)
    return RateLimitInfo(
        limit=limit,
        remaining=max(0, limit - len(bucket)),
        reset_seconds=_WINDOW_SEC,
    )


def headers_for(info: RateLimitInfo) -> dict[str, str]:
    """Public helper so routes can attach the bundle to successful responses."""
    return _to_headers(info)
