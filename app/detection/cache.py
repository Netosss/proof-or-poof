"""
Two-tier result cache: Redis (preferred) → Local Memory (fallback).

The Redis client is accessed at call-time via the integration module so that
it picks up the instance initialized during the FastAPI lifespan.
"""

import json
import time
import logging
from collections import OrderedDict
from typing import Optional

from app.config import settings
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)

local_cache: OrderedDict = OrderedDict()


def get_cached_result(key: str) -> Optional[dict]:
    """Retrieve result from Redis (preferred) or Local Memory (fallback)."""
    rc = redis_module.client
    if rc:
        try:
            data = rc.get(f"forensic:{key}")
            if data:
                logger.info(f"[CACHE] Redis HIT for key: {key}")
                return json.loads(data)
            else:
                logger.info(f"[CACHE] Redis MISS for key: {key}")
                return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    entry = local_cache.get(key)
    if entry:
        val, timestamp = entry
        if time.time() - timestamp < settings.local_cache_ttl_sec:
            logger.info(f"[CACHE] Local Memory HIT for key: {key}")
            local_cache.move_to_end(key)
            return val
        else:
            logger.info(f"[CACHE] Local Memory EXPIRED for key: {key}")
            del local_cache[key]
            return None

    logger.info(f"[CACHE] Local Memory MISS for key: {key}")
    return None


def set_cached_result(key: str, value: dict) -> None:
    """Store result in Redis (24h TTL) or Local Memory (fallback)."""
    rc = redis_module.client
    if rc:
        try:
            rc.set(f"forensic:{key}", json.dumps(value), ex=settings.deepfake_cache_ttl_sec)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
    else:
        if key in local_cache:
            local_cache.move_to_end(key)
        local_cache[key] = (value, time.time())
        if len(local_cache) > settings.local_cache_max_size:
            local_cache.popitem(last=False)
