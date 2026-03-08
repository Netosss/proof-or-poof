"""
Authentication utilities: Turnstile verification, IP extraction, and device-limit enforcement.

All functions that need a Redis client access it at call-time via the integration module
so they pick up the instance initialized during the FastAPI lifespan.

All Redis operations are async — no thread pool needed.
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
        if os.getenv("APP_ENV") != "dev":
            # In production, a missing key is a misconfiguration — fail loudly.
            raise HTTPException(
                status_code=500,
                detail="Verification service misconfigured"
            )
        logger.warning("turnstile_config_missing", extra={
            "action": "turnstile_config_missing",
            "detail": "TURNSTILE_SECRET_KEY not set, skipping validation (DEV MODE)",
        })
        return True

    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    payload = {"secret": secret, "response": token}

    try:
        async with http_module.request_session() as sess:
            async with sess.post(url, data=payload) as response:
                result = await response.json()
                if not result.get("success"):
                    logger.warning("turnstile_validation_failed", extra={
                        "action": "turnstile_validation_failed",
                        "error_codes": result.get("error-codes", []),
                    })
                    return False
                return True
    except HTTPException:
        raise
    except Exception as e:
        logger.error("turnstile_connection_error", extra={
            "action": "turnstile_connection_error",
            "error": str(e),
        })
        return False


import ipaddress as _ipaddress

# Cloudflare's published egress ranges — https://www.cloudflare.com/ips/
# These are stable and rarely change; update when Cloudflare publishes new ones.
_CF_NETWORKS: tuple[_ipaddress.IPv4Network | _ipaddress.IPv6Network, ...] = (
    # IPv4
    _ipaddress.ip_network("103.21.244.0/22"),
    _ipaddress.ip_network("103.22.200.0/22"),
    _ipaddress.ip_network("103.31.4.0/22"),
    _ipaddress.ip_network("104.16.0.0/13"),
    _ipaddress.ip_network("104.24.0.0/14"),
    _ipaddress.ip_network("108.162.192.0/18"),
    _ipaddress.ip_network("131.0.72.0/22"),
    _ipaddress.ip_network("141.101.64.0/18"),
    _ipaddress.ip_network("162.158.0.0/15"),
    _ipaddress.ip_network("172.64.0.0/13"),
    _ipaddress.ip_network("173.245.48.0/20"),
    _ipaddress.ip_network("188.114.96.0/20"),
    _ipaddress.ip_network("190.93.240.0/20"),
    _ipaddress.ip_network("197.234.240.0/22"),
    _ipaddress.ip_network("198.41.128.0/17"),
    # IPv6
    _ipaddress.ip_network("2400:cb00::/32"),
    _ipaddress.ip_network("2405:8100::/32"),
    _ipaddress.ip_network("2405:b500::/32"),
    _ipaddress.ip_network("2606:4700::/32"),
    _ipaddress.ip_network("2803:f800::/32"),
    _ipaddress.ip_network("2c0f:f248::/32"),
    _ipaddress.ip_network("2a06:98c0::/29"),
)


def _is_cloudflare_ip(ip_str: str) -> bool:
    """Return True if ip_str belongs to a known Cloudflare egress network."""
    try:
        ip = _ipaddress.ip_address(ip_str)
        return any(ip in net for net in _CF_NETWORKS)
    except ValueError:
        return False


def get_client_ip(request: Request) -> str:
    """
    Extract the real client IP, trusting CF-Connecting-IP only when the
    connection actually comes from a Cloudflare edge node.

    If the TCP peer is not a Cloudflare IP, we ignore CF-Connecting-IP
    (it could be forged) and fall back to X-Forwarded-For / the raw host.
    This prevents attackers from spoofing their IP for rate-limiting purposes
    by hitting the Railway URL directly and setting a fake CF-Connecting-IP.
    """
    peer_ip = request.client.host if request.client else ""

    if _is_cloudflare_ip(peer_ip):
        # Traffic arrived through Cloudflare — trust the real-IP header.
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip

    # Direct hit (bypassing Cloudflare) or Railway's own load balancer.
    x_forwarded = request.headers.get("x-forwarded-for")
    if x_forwarded:
        return x_forwarded.split(",")[0].strip()

    return peer_ip or "127.0.0.1"


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

    pipe = rc.pipeline()
    pipe.sismember(ip_key, device_id)
    pipe.scard(ip_key)
    results = await pipe.execute()

    is_known = results[0]
    current_count = results[1]

    if is_known:
        return

    is_fallback = device_id.startswith("mobile-fallback")
    limit = 1 if is_fallback else settings.max_new_devices_per_ip

    if current_count >= limit:
        if not token_already_verified:
            if not turnstile_token:
                logger.warning("ip_device_limit_reached", extra={
                    "action": "ip_device_limit_reached",
                    "ip": ip_address,
                    "device_count": current_count,
                    "limit": limit,
                })
                raise HTTPException(
                    status_code=403,
                    detail={"code": "STRICT_CAPTCHA_REQUIRED", "message": "High activity detected. Strict verification needed."}
                )
            is_human = await verify_turnstile(turnstile_token)
            if not is_human:
                raise HTTPException(status_code=403, detail="Invalid CAPTCHA")

    write_pipe = rc.pipeline()
    write_pipe.sadd(ip_key, device_id)
    write_pipe.expire(ip_key, IP_DEVICE_WINDOW)
    await write_pipe.execute()
