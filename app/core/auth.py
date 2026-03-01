"""
Authentication utilities: Turnstile verification, IP extraction, and device-limit enforcement.

All functions that need a Redis client access it at call-time via the integration module
so they pick up the instance initialized during the FastAPI lifespan.
"""

import os
import logging
import re
from typing import Optional

from fastapi import HTTPException, Request

from app.config import settings
from app.integrations import http_client as http_module
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)

IP_DEVICE_WINDOW = settings.rate_limit_window_sec

_DEVICE_ID_MAX_LEN = 128
_DEVICE_ID_RE = re.compile(r"^[a-zA-Z0-9\-_.]+$")


def validate_device_id(device_id: str) -> None:
    """
    Raises HTTP 400 if device_id is longer than 128 characters or contains
    characters outside [a-zA-Z0-9-_.].  Prevents Firestore key injection and
    Redis key-prefix abuse.
    """
    if len(device_id) > _DEVICE_ID_MAX_LEN or not _DEVICE_ID_RE.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid X-Device-ID")


async def verify_turnstile(token: str) -> bool:
    """Verifies a Cloudflare Turnstile token using the shared HTTP session."""
    secret = os.getenv("TURNSTILE_SECRET_KEY")
    if not secret:
        logger.warning("TURNSTILE_SECRET_KEY not set. Skipping validation (DEV MODE).")
        return True

    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    payload = {"secret": secret, "response": token}

    try:
        async with http_module.request_session() as sess:
            async with sess.post(url, data=payload) as response:
                result = await response.json()
                if not result.get("success"):
                    logger.warning(f"Turnstile validation failed: {result}")
                    return False
                return True
    except Exception as e:
        logger.error(f"Turnstile connection error: {e}")
        return False


def get_client_ip(request: Request) -> str:
    """Extracts the real client IP from headers, falling back to host."""
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip

    x_forwarded = request.headers.get("x-forwarded-for")
    if x_forwarded:
        return x_forwarded.split(",")[0].strip()

    return request.client.host if request.client else "127.0.0.1"


async def check_ip_device_limit(
    ip_address: str,
    device_id: str,
    turnstile_token: Optional[str] = None,
    token_already_verified: bool = False
):
    """
    Checks if this IP has created too many unique device IDs.
    If the limit is exceeded, a valid Turnstile token is required.
    """
    rc = redis_module.client
    if not rc:
        return  # Fail open if Redis is down

    ip_key = f"ip_devices:{ip_address}"

    pipeline = rc.pipeline()
    pipeline.sismember(ip_key, device_id)
    pipeline.scard(ip_key)
    results = pipeline.exec()

    is_known = results[0]
    current_count = results[1]

    if is_known:
        return

    is_fallback = device_id.startswith("mobile-fallback")
    limit = 1 if is_fallback else settings.max_new_devices_per_ip

    if current_count >= limit:
        if not token_already_verified:
            if not turnstile_token:
                logger.warning(f"IP {ip_address} reached device limit. STRICT CAPTCHA required.")
                raise HTTPException(
                    status_code=403,
                    detail={"code": "STRICT_CAPTCHA_REQUIRED", "message": "High activity detected. Strict verification needed."}
                )
            is_human = await verify_turnstile(turnstile_token)
            if not is_human:
                raise HTTPException(status_code=403, detail="Invalid CAPTCHA")

    write_pipeline = rc.pipeline()
    write_pipeline.sadd(ip_key, device_id)
    write_pipeline.expire(ip_key, IP_DEVICE_WINDOW)
    write_pipeline.exec()
