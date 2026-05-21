"""
Two-tier result cache: Redis (preferred) → Local Memory (fallback).

The Redis client is accessed at call-time via the integration module so that
it picks up the instance initialized during the FastAPI lifespan.

All Redis operations are async — no thread pool needed.

Cache key version: the prefix is bumped whenever the prompt, model, or scoring
scale changes in a way that invalidates prior verdicts. Old-prefix keys
self-expire on the existing 24h TTL; see scripts/redis_clear_forensic_cache.py
for an opt-in immediate cleanup.
"""

import json
import time
import logging
from collections import OrderedDict
from typing import Optional

from app.config import settings
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)

# Bumped from "forensic:" → "forensic_v2:" alongside the Gemini 3.5 Flash upgrade,
# dead-zone scoring scale, and softened quality context. Old "forensic:*" keys
# remain valid TTL-wise but will never be read; they self-expire within 24h.
CACHE_PREFIX = "forensic_v2:"

local_cache: OrderedDict = OrderedDict()


async def get_cached_result(key: str) -> Optional[dict]:
    """Retrieve result from Redis (preferred) or Local Memory (fallback)."""
    rc = redis_module.client
    if rc:
        try:
            data = await rc.get(f"{CACHE_PREFIX}{key}")
            if data:
                logger.debug("cache_redis_hit", extra={"action": "cache_redis_hit"})
                return json.loads(data)
            else:
                logger.debug("cache_redis_miss", extra={"action": "cache_redis_miss"})
                return None
        except Exception as e:
            logger.warning("cache_redis_error", extra={"action": "cache_redis_error", "error": str(e)})
            return None

    entry = local_cache.get(key)
    if entry:
        val, timestamp = entry
        if time.time() - timestamp < settings.local_cache_ttl_sec:
            logger.debug("cache_memory_hit", extra={"action": "cache_memory_hit"})
            local_cache.move_to_end(key)
            return val
        else:
            logger.debug("cache_memory_expired", extra={"action": "cache_memory_expired"})
            del local_cache[key]
            return None

    logger.debug("cache_memory_miss", extra={"action": "cache_memory_miss"})
    return None


async def set_cached_result(key: str, value: dict) -> None:
    """Store result in Redis (24h TTL) or Local Memory (fallback)."""
    rc = redis_module.client
    if rc:
        try:
            await rc.set(f"{CACHE_PREFIX}{key}", json.dumps(value), ex=settings.deepfake_cache_ttl_sec)
        except Exception as e:
            logger.warning("cache_redis_set_error", extra={"action": "cache_redis_set_error", "error": str(e)})
    else:
        if key in local_cache:
            local_cache.move_to_end(key)
        local_cache[key] = (value, time.time())
        if len(local_cache) > settings.local_cache_max_size:
            local_cache.popitem(last=False)
