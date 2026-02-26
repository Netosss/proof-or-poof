"""
Upstash Redis integration.

`client` starts as None. Call `initialize()` inside the FastAPI lifespan
context manager. Consuming modules reference `redis_client.client` at
call time rather than importing the variable directly.
"""

import os
import logging
from upstash_redis import Redis

logger = logging.getLogger(__name__)

# Set by initialize(). None when Redis credentials are absent or init fails.
client = None  # Redis | None


def initialize() -> None:
    """Create the Upstash Redis client and bind it to the module-level `client`."""
    global client

    redis_url = os.getenv("UPSTASH_REDIS_HOST")
    redis_token = os.getenv("UPSTASH_REDIS_PASSWORD")

    if redis_url and redis_token:
        try:
            client = Redis(url=redis_url, token=redis_token)
            logger.info("[STARTUP] Upstash Redis client initialized successfully")
        except Exception as e:
            logger.error(f"[STARTUP] Failed to initialize Upstash Redis client: {e}")
    else:
        logger.warning(
            "[STARTUP] Redis credentials not found. Rate limiting will fallback to memory."
        )
