"""
Firebase App Check verification (mobile anti-abuse).

Mobile clients attach a short-lived App Check token in the `X-Firebase-AppCheck`
header (Play Integrity provider on Android; App Attest on iOS later). Verifying
that token with the Firebase Admin SDK proves the request came from a genuine,
unmodified build of our app running on a genuine device — strictly stronger than
a WebView CAPTCHA.

Web is untouched: the website never sends this header and keeps using Cloudflare
Turnstile. The OR-gate lives in app.api.detection / app.api.inpainting.

Rollout is controlled by `settings.app_check_mode` (env APP_CHECK_MODE):
  - "monitor" (default): tokens are verified and logged for metrics, but a valid
    App Check token does NOT by itself satisfy the guest gate — Turnstile is
    still required. Use this to measure real-world success rates before enforcing.
  - "enforce": a valid App Check token alone satisfies the mobile guest gate
    (mobile drops Turnstile). An invalid/expired token falls back to Turnstile
    so a transient failure on a real device is never a hard lockout.

Replay hardening: the Python Admin SDK cannot mint/consume limited-use tokens,
so we cap how many times a single token may be presented within its TTL using a
Redis counter keyed by a hash of the token. This bounds the value of a leaked
token without hurting real heavy users (who reuse one cached token for ~1 h).
Fails open if Redis is down.
"""

import asyncio
import hashlib
import logging

from firebase_admin import app_check as firebase_app_check

from app.config import settings
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)


def _token_fingerprint(token: str) -> str:
    """Short, non-reversible id for a token — safe to use as a Redis key/log field."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]


async def _replay_within_cap(fingerprint: str, endpoint: str) -> bool:
    """Increment the per-token use counter; return False once the cap is exceeded.

    Fails open (returns True) when Redis is unavailable — App Check verification
    itself remains the primary gate.
    """
    rc = redis_module.client
    if not rc:
        return True

    key = f"app_check_uses:{fingerprint}"
    try:
        uses = await rc.incr(key)
        if uses == 1:
            await rc.expire(key, settings.app_check_replay_ttl_sec)
    except Exception as e:  # noqa: BLE001 - Redis hiccup must not block a real device
        logger.warning(
            "app_check_replay_counter_error",
            extra={"action": "app_check_replay_counter_error", "error": str(e)},
        )
        return True

    if uses > settings.app_check_replay_max:
        logger.warning(
            "app_check_replay_exceeded",
            extra={
                "action": "app_check_replay_exceeded",
                "endpoint": endpoint,
                "token_fp": fingerprint,
                "uses": uses,
                "cap": settings.app_check_replay_max,
            },
        )
        return False
    return True


async def verify_app_check(token: str, *, endpoint: str, enforce_replay: bool = True) -> bool:
    """Verify a Firebase App Check token.

    Returns True only if the token is cryptographically valid AND (when
    `enforce_replay`) still under the per-token replay cap. Never raises — any
    failure is logged and returns False so the caller can fall back to Turnstile.

    `verify_token` is synchronous and performs I/O (fetches Google's public keys
    on first use, then caches them), so it runs in a thread pool.
    """
    if not token:
        return False

    fingerprint = _token_fingerprint(token)
    try:
        await asyncio.to_thread(firebase_app_check.verify_token, token)
    except Exception as e:  # noqa: BLE001 - SDK raises a variety of token/HTTP errors
        logger.warning(
            "app_check_verify_failed",
            extra={
                "action": "app_check_verify_failed",
                "endpoint": endpoint,
                "token_fp": fingerprint,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        return False

    if enforce_replay and not await _replay_within_cap(fingerprint, endpoint):
        return False

    logger.info(
        "app_check_verified",
        extra={"action": "app_check_verified", "endpoint": endpoint, "token_fp": fingerprint},
    )
    return True


async def passes_app_check_gate(app_check_token: str | None, *, endpoint: str) -> bool:
    """Guest-gate helper: does App Check alone satisfy the challenge for this request?

    - No token (the web path, and mobile before App Check ships): False — the
      caller then requires Turnstile exactly as before.
    - `monitor` mode: verify + log the token for metrics, but return False so
      Turnstile is still enforced.
    - `enforce` mode: return whether the token verified (replay-capped). On
      failure the caller falls back to Turnstile.
    """
    if not app_check_token:
        return False

    if settings.app_check_mode == "monitor":
        # Verify for metrics only; do not let it substitute for Turnstile yet.
        await verify_app_check(app_check_token, endpoint=endpoint)
        return False

    # enforce
    return await verify_app_check(app_check_token, endpoint=endpoint)
