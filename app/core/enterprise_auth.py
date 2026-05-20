"""
HMAC authentication for the enterprise S2S API.

Verification order (fail-fast — cheapest checks first):
    1. All four required headers present + format-validated
    2. Timestamp drift within `enterprise_timestamp_drift_sec`
    3. Reverse-lookup api_key → credential record (Firestore single doc read)
    4. Replay-nonce SETNX in Redis (`replay:{credential_id}:{signature}`)
       — namespaced by credential so a leaked signature from one partner
       cannot pre-claim another partner's nonce slot.
    5. IP allowlist (if configured on credential)
    6. Constant-time HMAC signature comparison

Signature canonical form (AWS SigV4 inspired):
    payload = f"{timestamp}\\n{method}\\n{path_with_query}\\n{content_sha256}"
    signature = HMAC-SHA256(secret_key, payload).hexdigest()

`path_with_query` is `request.url.path` plus `?{request.url.query}` when a
query string is present — required so a signature for `/v1/analyze` cannot be
replayed against `/v1/analyze?inject=anything`.

The X-FauxLens-Content-SHA256 header MUST equal the SHA-256 hex digest of the
raw request body. The route handler verifies this AFTER streaming the body to
disk (compute-and-compare) — until then, this middleware can fast-reject
clearly bogus signatures without touching the payload.

Why we don't sign the raw body inline:
    A 200 MB enterprise video would force buffer-before-auth, creating a DoS
    surface where attackers without valid keys still consume backend RAM/CPU
    on body parsing before rejection. Hashing the body separately lets us
    verify the signature against a 64-char digest, reject if invalid, and
    only THEN stream the body — same security guarantee, no buffer trap.
"""

import hashlib
import hmac
import logging
import re
import time
from dataclasses import dataclass
from ipaddress import ip_address, ip_network

from fastapi import HTTPException, Request

from app.config import settings
from app.integrations import redis_client as redis_module
from app.services.api_credentials import resolve_credential, touch_credential

logger = logging.getLogger(__name__)


HEADER_KEY = "X-FauxLens-Key"
HEADER_TIMESTAMP = "X-FauxLens-Timestamp"
HEADER_SIGNATURE = "X-FauxLens-Signature"
HEADER_CONTENT_SHA = "X-FauxLens-Content-SHA256"

# Strict format guards: reject malformed header values BEFORE they get
# interpolated into the canonical signing payload, eliminating any chance
# of header smuggling via CR/LF/etc. in attacker-controlled strings.
# NOTE: 10-digit timestamp covers 2001-09-09 through 2286-11-20. Pre/post
# those dates are reject-by-design; the operational concern is only the
# upper bound (2286), which is acceptable.
_TIMESTAMP_RE = re.compile(r"^\d{10}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")
# API key format: fxl_(live|test)_ + 32 alnum chars (matches issuance in
# api_credentials.create_credential). Pre-validate so attacker-controlled
# keys don't waste a Firestore round-trip or trigger encoding surprises.
_API_KEY_RE = re.compile(r"^fxl_(live|test)_[A-Za-z0-9_-]{20,64}$")

# Pre-auth IP rate limit — runs BEFORE credential resolution and HMAC compute
# to make brute-force probing economically prohibitive.
_PRE_AUTH_LIMIT_PER_MIN = 120
_PRE_AUTH_WINDOW_SEC = 60


@dataclass(frozen=True)
class EnterprisePrincipal:
    partner_id: str
    credential_id: str
    api_key_prefix: str
    content_sha256: str  # echoed from header; route MUST re-verify against streamed body


def _auth_error(code: str, message: str, status: int = 401) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"type": "authentication_error", "code": code, "message": message},
    )


def _canonical_signing_payload(
    timestamp: str, method: str, path: str, content_sha256: str
) -> bytes:
    """
    Path MUST be `request.url.path + ("?" + query if query)` when computed
    by callers (see `authenticate`). Including the query string binds the
    signature to the exact URL target, preventing a valid signature from
    being replayed against a query-parameterised variant of the same path.
    """
    return f"{timestamp}\n{method.upper()}\n{path}\n{content_sha256.lower()}".encode()


def _path_with_query(request: Request) -> str:
    """Return `/path?query` when a query string is present, else `/path`."""
    q = request.url.query
    return f"{request.url.path}?{q}" if q else request.url.path


def _extract_client_ip(request: Request) -> str:
    """
    Resolve the real partner IP for IP allowlist + pre-auth rate limiting.

    Behind Railway's edge proxy `request.client.host` is always the internal
    hop address (~`100.64.0.x`), which would make the IP allowlist trivially
    misbehave (either always-pass or always-block) and collapse pre-auth
    rate limiting into a single shared bucket.

    Railway populates `X-Forwarded-For` with the original client IP. We take
    the RIGHTMOST entry because each trusted proxy appends to the right; with
    a single trusted hop (Railway edge) the rightmost is the only entry that
    cannot have been spoofed by the client. If the header is missing (local
    dev, direct connection) we fall back to `request.client.host`.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        # Rightmost non-empty token, stripped.
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else ""


def _ip_in_allowlist(client_ip: str, cidrs: list[str]) -> bool:
    if not cidrs:
        return True
    try:
        addr = ip_address(client_ip)
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            if addr in ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


async def _check_replay(credential_id: str, signature: str) -> None:
    """Reject same-signature replays within the timestamp window.

    Keyed by `(credential_id, signature)` so a leaked signature from one
    partner cannot be used by another principal to pre-claim a nonce slot
    and deny service to the original owner. This MUST be called AFTER
    credential resolution.

    Billable mutations FAIL CLOSED when Redis is unavailable: a credit-deducting
    endpoint must never accept a replayed signature just because the nonce
    backend is down. Less-strict callers (none today) could override this
    behaviour by passing through.
    """
    rc = redis_module.client
    if not rc:
        logger.critical(
            "enterprise_replay_check_unavailable",
            extra={
                "action": "enterprise_replay_check_unavailable",
                "reason": "redis_unavailable",
                "policy": "fail_closed",
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "type": "service_unavailable_error",
                "code": "replay_check_unavailable",
                "message": "Replay protection is temporarily unavailable. Please retry.",
            },
        )
    key = f"replay:{credential_id}:{signature}"
    try:
        ok = await rc.set(key, "1", ex=settings.enterprise_replay_window_sec, nx=True)
    except Exception as e:
        logger.critical(
            "enterprise_replay_check_error",
            extra={
                "action": "enterprise_replay_check_error",
                "error": str(e),
                "policy": "fail_closed",
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "type": "service_unavailable_error",
                "code": "replay_check_unavailable",
                "message": "Replay protection is temporarily unavailable. Please retry.",
            },
        )
    if not ok:
        logger.warning(
            "enterprise_replay_detected",
            extra={
                "action": "enterprise_replay_detected",
                "credential_id": credential_id,
                "sig_prefix": signature[:16],
            },
        )
        raise _auth_error("replay_detected", "Request signature replayed within window.")


# Same atomic INCR + (EXPIRE-only-on-create) used by the per-credential
# limiter. A naive pipeline can leak a TTL-less key if Redis dies between
# commands — which on the pre-auth limiter would permanently lock out the
# legitimate partner's egress IP at the first rate-limit window roll. See
# `enterprise_rate_limiter._INCR_WITH_TTL_LUA` for the rationale.
_PRE_AUTH_INCR_WITH_TTL_LUA = """
local v = redis.call('INCR', KEYS[1])
if v == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return v
"""


async def _pre_auth_rate_limit(client_ip: str) -> None:
    """Coarse pre-auth rate limit by source IP to throttle HMAC brute-forcers.

    Sits BEFORE credential lookup / HMAC compute so a bot blasting bogus
    signatures can't burn Firestore reads or CPU. Bypassed silently if Redis
    is unavailable — this is defense-in-depth, not a primary control, and the
    request-level guards downstream already exist.
    """
    rc = redis_module.client
    if not rc or not client_ip:
        return
    key = f"rate:ent:preauth:{client_ip}"
    try:
        count = int(await rc.eval(_PRE_AUTH_INCR_WITH_TTL_LUA, 1, key, _PRE_AUTH_WINDOW_SEC))
    except Exception:
        return
    if count > _PRE_AUTH_LIMIT_PER_MIN:
        logger.warning(
            "enterprise_pre_auth_rate_limited",
            extra={
                "action": "enterprise_pre_auth_rate_limited",
                "client_ip": client_ip,
                "count": count,
            },
        )
        raise HTTPException(
            status_code=429,
            detail={
                "type": "rate_limit_error",
                "code": "rate_limited",
                "message": "Too many authentication attempts. Please slow down.",
            },
        )


async def authenticate(request: Request) -> EnterprisePrincipal:
    """
    Run the full auth pipeline on an inbound request. Returns an
    EnterprisePrincipal on success; raises HTTPException otherwise.

    Does NOT consume the request body — only headers.
    """
    client_ip = _extract_client_ip(request)

    # --- 1a. Pre-auth IP rate limit (cheap, before any crypto/db work) ---
    await _pre_auth_rate_limit(client_ip)

    api_key = request.headers.get(HEADER_KEY)
    timestamp = request.headers.get(HEADER_TIMESTAMP)
    signature = request.headers.get(HEADER_SIGNATURE)
    content_sha = request.headers.get(HEADER_CONTENT_SHA)

    if not api_key or not timestamp or not signature or not content_sha:
        raise _auth_error(
            "missing_auth_headers",
            f"Required headers: {HEADER_KEY}, {HEADER_TIMESTAMP}, "
            f"{HEADER_SIGNATURE}, {HEADER_CONTENT_SHA}",
        )

    # --- 1b. Strict header value validation BEFORE interpolating into the
    #         canonical signing string. Prevents header-smuggling attacks
    #         where an attacker stuffs CR/LF into a header value to forge
    #         an alternate canonical payload, and short-circuits malformed
    #         api_key values before they cost a Firestore round-trip.
    if not _TIMESTAMP_RE.fullmatch(timestamp):
        raise _auth_error(
            "invalid_timestamp", f"{HEADER_TIMESTAMP} must be exactly 10 digits (UNIX seconds)."
        )
    if not _SHA256_HEX_RE.fullmatch(content_sha.lower()):
        raise _auth_error(
            "invalid_content_hash", f"{HEADER_CONTENT_SHA} must be 64 lowercase hex characters."
        )
    if not _SIGNATURE_RE.fullmatch(signature.lower()):
        raise _auth_error(
            "invalid_signature_format", f"{HEADER_SIGNATURE} must be 64 lowercase hex characters."
        )
    if not _API_KEY_RE.fullmatch(api_key):
        raise _auth_error("invalid_api_key", "API key is invalid, revoked, or expired.")

    # --- 2. Timestamp drift. Include `server_time` in the error so a partner
    #         with clock skew can self-diagnose direction + magnitude without
    #         needing operator support.
    ts_int = int(timestamp)
    server_now = int(time.time())
    drift = abs(server_now - ts_int)
    if drift > settings.enterprise_timestamp_drift_sec:
        raise _auth_error(
            "timestamp_expired",
            f"Request timestamp drift {drift}s exceeds max "
            f"{settings.enterprise_timestamp_drift_sec}s. "
            f"Server time: {server_now}. Send UNIX seconds (10 digits) and "
            "synchronize your host clock via NTP.",
        )

    # --- 3. Credential resolution — runs BEFORE the replay nonce check so we
    #         can scope the nonce to (credential_id, signature). This costs a
    #         Firestore round-trip on every request, mitigated by trivially
    #         rejecting malformed api_key formats above.
    cred = await resolve_credential(api_key)
    if not cred:
        raise _auth_error("invalid_api_key", "API key is invalid, revoked, or expired.")

    # --- 4. Replay protection (fails closed on Redis outage) ---
    await _check_replay(cred["credential_id"], signature)

    # --- 5. IP allowlist (if any) ---
    if cred["allowed_ips"] and not _ip_in_allowlist(client_ip, cred["allowed_ips"]):
        logger.warning(
            "enterprise_ip_blocked",
            extra={
                "action": "enterprise_ip_blocked",
                "partner_id": cred["partner_id"],
                "credential_id": cred["credential_id"],
                "client_ip": client_ip,
            },
        )
        raise _auth_error(
            "ip_not_allowed", "Source IP is not on the credential allowlist.", status=403
        )

    # --- 6. Signature comparison ---
    signed_path = _path_with_query(request)
    expected = hmac.new(
        cred["secret_key"].encode("utf-8"),
        _canonical_signing_payload(timestamp, request.method, signed_path, content_sha),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        logger.warning(
            "enterprise_signature_mismatch",
            extra={
                "action": "enterprise_signature_mismatch",
                "partner_id": cred["partner_id"],
                "credential_id": cred["credential_id"],
                "api_key_prefix": cred["api_key_prefix"],
            },
        )
        # Surface the canonical payload shape so partners can self-diagnose
        # the common failures (wrong method case, trailing slash on path,
        # missing query string in the signed payload).
        raise _auth_error(
            "invalid_signature",
            "HMAC signature did not match canonical payload. Expected "
            "payload format: '{timestamp}\\n{METHOD}\\n{path_with_query}\\n"
            "{content_sha256_hex}' — method uppercase, path includes query "
            "string when present, SHA-256 over raw request body.",
        )

    # Fire-and-forget housekeeping — never blocks the request.
    try:
        await touch_credential(cred["partner_id"], cred["credential_id"])
    except Exception:
        pass

    return EnterprisePrincipal(
        partner_id=cred["partner_id"],
        credential_id=cred["credential_id"],
        api_key_prefix=cred["api_key_prefix"],
        content_sha256=content_sha.lower(),
    )


def verify_body_hash(body_bytes_or_path: bytes | str, expected_sha256: str) -> bool:
    """
    Recompute SHA-256 over the streamed body and compare to the header.

    Accepts either an in-memory bytes object (small JSON requests) or a path to
    the streamed-to-disk temp file (multipart media). For path inputs the file
    is hashed in 64 KB chunks so a 200 MB video never blows RAM.
    """
    h = hashlib.sha256()
    if isinstance(body_bytes_or_path, (bytes, bytearray)):
        h.update(body_bytes_or_path)
    else:
        with open(body_bytes_or_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(65_536), b""):
                h.update(chunk)
    return hmac.compare_digest(h.hexdigest(), (expected_sha256 or "").lower())
