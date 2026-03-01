"""
Shared aiohttp ClientSession — initialized once during FastAPI lifespan.

Reusing a single session avoids per-request TCP handshake overhead, saving
10–50 ms on every external HTTP call (Turnstile, RunPod, URL downloads).

Usage:
    async with http_client.request_session() as sess:
        async with sess.post(url, data=payload) as response:
            ...

The context manager yields the shared session when available, otherwise
creates and closes a temporary one (covers tests and pre-init calls).
"""

import logging
from contextlib import asynccontextmanager

import aiohttp

logger = logging.getLogger(__name__)

session: aiohttp.ClientSession | None = None


async def initialize() -> None:
    global session
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    )
    logger.info("[STARTUP] Shared HTTP session initialized")


async def close() -> None:
    global session
    if session and not session.closed:
        await session.close()
        session = None
        logger.info("[SHUTDOWN] Shared HTTP session closed")


@asynccontextmanager
async def request_session():
    """
    Async context manager that yields the shared session if available,
    otherwise creates and closes a temporary one.

    Never closes the shared session — http_client.close() handles that.
    """
    global session
    if session and not session.closed:
        yield session
    else:
        tmp = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        try:
            yield tmp
        finally:
            await tmp.close()
