"""
Rate limiting: Redis-backed (preferred) with in-memory fallback.

The Redis client is accessed at call-time via the integration module
so it picks up the instance initialized during the FastAPI lifespan.
"""

import time
import logging
from typing import Dict

from fastapi import HTTPException

from app.config import settings
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)

# In-memory store: {identifier: [timestamp, ...]}
_rate_limits: Dict[str, list] = {}

RATE_LIMIT_WINDOW = settings.rate_limit_request_window_sec
MAX_REQUESTS_PER_WINDOW = settings.rate_limit_max_requests


def check_rate_limit(identifier: str) -> None:
    """Rate limiting using Redis (preferred) or Memory (fallback)."""
    rc = redis_module.client
    if rc:
        _check_rate_limit_redis(rc, identifier)
    else:
        _check_rate_limit_memory(identifier)


def _check_rate_limit_redis(rc, identifier: str) -> None:
    key = f"rate_limit:{identifier}"
    try:
        current_count = rc.incr(key)
        if current_count == 1:
            rc.expire(key, RATE_LIMIT_WINDOW)

        if current_count > MAX_REQUESTS_PER_WINDOW:
            logger.warning(f"Redis Rate limit exceeded for {identifier}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again in a minute."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Redis rate limit error: {e}. Falling back to memory.")
        _check_rate_limit_memory(identifier)


def _check_rate_limit_memory(identifier: str) -> None:
    """Simple sliding-window in-memory rate limiting."""
    now = time.time()

    if len(_rate_limits) > settings.rate_limit_memory_limit:
        _cleanup_all_limits(now)

    if identifier not in _rate_limits:
        _rate_limits[identifier] = []

    _rate_limits[identifier] = [
        t for t in _rate_limits[identifier]
        if now - t < RATE_LIMIT_WINDOW
    ]

    if len(_rate_limits[identifier]) >= MAX_REQUESTS_PER_WINDOW:
        logger.warning(f"Memory Rate limit exceeded for {identifier}")
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again in a minute."
        )

    _rate_limits[identifier].append(now)


def _cleanup_all_limits(now: float) -> None:
    """Remove all identifiers that have been idle for the full window."""
    expired_keys = [
        k for k, v in _rate_limits.items()
        if not v or now - v[-1] > RATE_LIMIT_WINDOW
    ]
    for k in expired_keys:
        del _rate_limits[k]
    logger.info(f"Rate limit cleanup: removed {len(expired_keys)} inactive sessions.")
