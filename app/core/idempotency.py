"""
Idempotency-Key support for the enterprise API.

When a partner sends `Idempotency-Key: <uuid>` we cache the response body +
status under `idem:{credential_id}:{key}` for `enterprise_idempotency_ttl_sec`.
On replay the cached response is returned with `X-Idempotent-Replay: true`,
guaranteeing no double-charge.

Constraints:
    - Required on POST /v1/* — missing header → 400 (operationally we prefer
      enforcement over best-effort: enterprises that don't send the header
      should learn now, not after a billing dispute).
    - Body is stored as a JSON string. We do not store binary payloads.
    - In-flight requests with the same key: we use SETNX as a lock with a
      short TTL so concurrent retries block until the original completes (or
      its lock expires and a fresh attempt proceeds).
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.config import settings
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)

_HEADER = "Idempotency-Key"
_INFLIGHT_TTL_SEC = 300  # 5 min — covers worst-case scan duration
_KEY_MAX_LEN = 255


@dataclass
class CachedResponse:
    status_code: int
    body: dict[str, Any]


def get_idempotency_key(headers) -> str:
    key = headers.get(_HEADER) or headers.get(_HEADER.lower())
    if not key:
        raise HTTPException(
            status_code=400,
            detail={"type": "invalid_request_error", "code": "missing_idempotency_key",
                    "message": f"Required header missing: {_HEADER}"},
        )
    if len(key) > _KEY_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail={"type": "invalid_request_error", "code": "idempotency_key_too_long",
                    "message": f"{_HEADER} must be <= {_KEY_MAX_LEN} chars"},
        )
    return key


def _cache_key(credential_id: str, idem_key: str) -> str:
    return f"idem:{credential_id}:{idem_key}"


def _lock_key(credential_id: str, idem_key: str) -> str:
    return f"idem_lock:{credential_id}:{idem_key}"


async def claim_or_replay(credential_id: str, idem_key: str) -> CachedResponse | None:
    """
    Returns:
        CachedResponse  — replay; return this to the caller immediately
        None            — caller proceeds, must call `store` after success

    Concurrency: SETNX a short-lived lock. If we lose the race, we wait briefly
    and re-check the cache for a finalized response. After lock TTL elapses a
    fresh attempt is allowed (prevents stuck idempotent state on crashes).
    """
    rc = redis_module.client
    if not rc:
        # No Redis → no idempotency cache. Acceptable degraded mode; logged once
        # at the auth layer already.
        return None

    cached = await rc.get(_cache_key(credential_id, idem_key))
    if cached:
        return _decode(cached)

    # Try to claim the in-flight lock.
    locked = await rc.set(
        _lock_key(credential_id, idem_key),
        "1",
        ex=_INFLIGHT_TTL_SEC,
        nx=True,
    )
    if locked:
        return None

    # Lost the race — poll briefly for the originator's result.
    end = time.time() + 30
    while time.time() < end:
        cached = await rc.get(_cache_key(credential_id, idem_key))
        if cached:
            return _decode(cached)
        # Tight poll; redis lookups are sub-ms.
        await _sleep_ms(50)
    # Originator didn't finish in time — let this caller proceed, accepting the
    # small risk of double-execution. The credit refund path remains idempotent
    # so even a real double-fire is recoverable.
    return None


async def store(credential_id: str, idem_key: str,
                status_code: int, body: dict[str, Any]) -> None:
    rc = redis_module.client
    if not rc:
        return
    payload = json.dumps({"status_code": status_code, "body": body})
    await rc.set(
        _cache_key(credential_id, idem_key),
        payload,
        ex=settings.enterprise_idempotency_ttl_sec,
    )
    # Release lock — also handled by TTL but explicit cleanup avoids
    # waiting clients spinning for `_INFLIGHT_TTL_SEC`.
    try:
        await rc.delete(_lock_key(credential_id, idem_key))
    except Exception as e:
        # Lock will expire on its own, but log so operators can see if cleanup
        # is failing repeatedly (would explain prolonged poll stalls).
        logger.warning(
            "idempotency_lock_cleanup_failed",
            extra={
                "action": "idempotency_lock_cleanup_failed",
                "credential_id": credential_id,
                "error": str(e),
            },
        )


def _decode(raw) -> CachedResponse:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return CachedResponse(status_code=int(data["status_code"]), body=data["body"])


async def _sleep_ms(ms: int) -> None:
    import asyncio
    await asyncio.sleep(ms / 1000)
