"""
Upstash Redis integration — native async via redis.asyncio.

`client` starts as None. Call `await initialize()` inside the FastAPI lifespan
context manager. Consuming modules reference `redis_client.client` at call time.

Connection uses the standard Redis wire protocol over TLS (rediss://) rather
than Upstash's HTTP REST API, enabling true async I/O with no thread overhead
and support for Pub/Sub (used by the RunPod inpainting job flow).

Required env var: UPSTASH_REDIS_URL  (rediss://default:<password>@<host>:<port>)
"""

import os
import logging
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Set by initialize(). None when Redis credentials are absent or init fails.
client: aioredis.Redis | None = None


async def initialize() -> None:
    """Create the async Redis client and validate the connection with PING."""
    global client

    url = os.getenv("UPSTASH_REDIS_URL")
    if not url:
        logger.warning("startup_redis_missing_credentials", extra={
            "action": "startup_redis_missing_credentials",
            "fallback": "memory_rate_limiting",
        })
        return

    try:
        # max_connections caps the pool so Pub/Sub subscriptions (one dedicated
        # connection per active inpainting job, held for up to 180 s) cannot
        # exhaust Upstash's concurrent-connection limit at scale.
        # 200 = 100 simultaneous Pub/Sub + 100 slots for regular commands.
        client = aioredis.from_url(url, decode_responses=True, max_connections=200)
        await client.ping()
        logger.info("startup_redis", extra={"action": "startup_redis"})
    except Exception as e:
        logger.error("startup_redis_failed", extra={
            "action": "startup_redis_failed",
            "error": str(e),
        })
        client = None


async def close() -> None:
    """Close the Redis connection pool gracefully on shutdown."""
    global client
    if client:
        await client.aclose()
        client = None
        logger.info("shutdown_redis", extra={"action": "shutdown_redis"})
