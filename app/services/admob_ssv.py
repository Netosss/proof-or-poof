"""
AdMob rewarded-ad Server-Side Verification (SSV).

Verifies the ECDSA signature Google attaches to every rewarded-ad SSV callback so
credits are granted ONLY for real, Google-confirmed ad views — closing the
"client can call the reward endpoint without watching an ad" gap.

Google appends `signature` and `key_id` as the last two query parameters, in
that order. The signed message is the raw query string up to (not including)
`&signature=`. Public keys are published at VERIFIER_KEYS_URL and cached
in-memory; a cache miss (first call or key rotation) triggers a refresh.

Docs: https://developers.google.com/admob/android/rewarded-video-ssv
"""

import asyncio
import base64
import logging
import time
from urllib.parse import parse_qsl

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_public_key

logger = logging.getLogger(__name__)

# Google's documented URL. The `www.` host is required: the bare `gstatic.com`
# 301-redirects here, and with follow_redirects off httpx would hand back the
# redirect body, breaking the keys fetch and silently disabling all SSV.
VERIFIER_KEYS_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"
_KEYS_FETCH_TIMEOUT_SEC = 10.0

# key_id (str) -> PEM public key (str). Populated lazily and on cache miss.
_keys_cache: dict[str, str] = {}
# Coalesces concurrent cache misses so a burst of requests (or unknown key_ids)
# triggers at most one outbound fetch instead of one per request.
_keys_fetch_lock = asyncio.Lock()

# --- unknown-key_id stall defence -------------------------------------------
# /api/ads/ssv is unauthenticated, and an unknown key_id can never be in
# _keys_cache — so before this, every request carrying a random key_id forced a
# fresh outbound fetch (10s timeout) while holding the global lock above. The
# coalescing lock made it worse rather than better: the fetches serialised, so a
# trickle of junk key_ids stalled SSV for everyone, including real callbacks.
#
# Two independent bounds now:
#   1. a negative cache, so a repeat of the SAME unknown key_id costs nothing;
#   2. a floor on how often the key set may be refetched AT ALL, so a stream of
#      DISTINCT unknown key_ids (which defeats a negative cache) cannot drive
#      more than one fetch per window.
# Legitimate key rotation is unaffected: Google rotates on the order of months,
# far slower than this window.
_KEYS_MIN_REFETCH_SEC = 300.0
_UNKNOWN_KEY_TTL_SEC = 600.0
# Bounded so the negative cache is not itself a memory-growth vector.
_UNKNOWN_KEY_MAX_ENTRIES = 1024

_last_keys_fetch_at: float = 0.0          # time.monotonic() of the last attempt
_unknown_key_ids: dict[str, float] = {}   # key_id -> monotonic expiry


def _note_unknown_key_id(key_id: str, now: float) -> None:
    """Remember a key_id as unknown, pruning expired entries first."""
    if len(_unknown_key_ids) >= _UNKNOWN_KEY_MAX_ENTRIES:
        for k in [k for k, exp in _unknown_key_ids.items() if exp <= now]:
            del _unknown_key_ids[k]
        # Still full => every entry is live, i.e. an active flood of distinct
        # ids. Drop the whole thing rather than grow without bound; the refetch
        # floor is what actually caps outbound work in that case.
        if len(_unknown_key_ids) >= _UNKNOWN_KEY_MAX_ENTRIES:
            _unknown_key_ids.clear()
    _unknown_key_ids[key_id] = now + _UNKNOWN_KEY_TTL_SEC


async def _fetch_keys() -> dict[str, str]:
    async with httpx.AsyncClient(timeout=_KEYS_FETCH_TIMEOUT_SEC, follow_redirects=True) as http:
        resp = await http.get(VERIFIER_KEYS_URL)
        resp.raise_for_status()
        data = resp.json()
    keys: dict[str, str] = {}
    for k in data.get("keys", []):
        kid = k.get("keyId")
        pem = k.get("pem")
        if kid is not None and pem:
            keys[str(kid)] = pem
    return keys


async def _get_public_key(key_id: str):
    global _last_keys_fetch_at

    pem = _keys_cache.get(key_id)
    if pem is None:
        now = time.monotonic()

        # Known-unknown: answered without touching the lock or the network.
        expiry = _unknown_key_ids.get(key_id)
        if expiry is not None and expiry > now:
            return None

        # Miss -> refresh (covers first use and Google's periodic key rotation).
        # Serialize concurrent misses and re-check under the lock so only the
        # first waiter actually performs the network fetch.
        async with _keys_fetch_lock:
            pem = _keys_cache.get(key_id)
            if pem is None:
                now = time.monotonic()
                # Refetch floor. Skipped entirely when the cache is empty, so a
                # cold start still fetches immediately rather than failing every
                # callback for the first window.
                if _keys_cache and (now - _last_keys_fetch_at) < _KEYS_MIN_REFETCH_SEC:
                    logger.warning(
                        "admob_ssv_key_refetch_throttled",
                        extra={
                            "action": "admob_ssv_key_refetch_throttled",
                            "key_id": key_id,
                        },
                    )
                    _note_unknown_key_id(key_id, now)
                    return None

                _last_keys_fetch_at = now
                try:
                    _keys_cache.update(await _fetch_keys())
                except Exception as e:
                    logger.error(
                        "admob_ssv_keys_fetch_failed",
                        extra={"action": "admob_ssv_keys_fetch_failed", "error": str(e)},
                    )
                    return None
                pem = _keys_cache.get(key_id)
                if pem is None:
                    _note_unknown_key_id(key_id, time.monotonic())
    if pem is None:
        # Fetch succeeded but Google's key set has no such key_id (stale/rotated
        # or spoofed). Log it — otherwise it looks identical to a bad signature.
        logger.warning(
            "admob_ssv_key_id_unknown",
            extra={"action": "admob_ssv_key_id_unknown", "key_id": key_id},
        )
        return None
    try:
        return load_pem_public_key(pem.encode("utf-8"))
    except Exception as e:
        logger.error(
            "admob_ssv_pem_parse_failed",
            extra={"action": "admob_ssv_pem_parse_failed", "key_id": key_id, "error": str(e)},
        )
        return None


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + ("=" * (-len(s) % 4)))


def _signed_content(raw_query: str) -> str | None:
    """The exact substring Google signs: everything before the `&signature=` marker."""
    idx = raw_query.find("&signature=")
    if idx == -1:
        return None
    return raw_query[:idx]


def verified_params(raw_query: str) -> dict[str, str]:
    """
    Parse ONLY the signed portion of the SSV query string into a dict.

    Callers MUST read acted-upon fields (custom_data, transaction_id) from here,
    never from request.query_params: Starlette's QueryParams is last-wins on
    duplicate keys, so a replayed callback with `&transaction_id=forged` appended
    AFTER the signature would keep a valid signature yet smuggle a forged value
    past verification. Anything after `&signature=` is outside the signed content
    and is therefore excluded here.
    """
    content = _signed_content(raw_query)
    if content is None:
        return {}
    return dict(parse_qsl(content, keep_blank_values=True))


async def verify_ssv(raw_query: str, signature: str, key_id: str) -> bool:
    """
    Verify an AdMob rewarded SSV callback.

    Args:
        raw_query: the full request query string (request.url.query).
        signature: the `signature` query param (base64url, ECDSA DER).
        key_id:    the `key_id` query param selecting the verifier key.

    Returns True only if the signature validates against Google's public key.
    """
    if not signature or not key_id:
        return False
    content = _signed_content(raw_query)
    if content is None:
        return False
    message = content.encode("utf-8")

    public_key = await _get_public_key(key_id)
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        return False
    try:
        public_key.verify(_b64url_decode(signature), message, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False
    except Exception as e:
        logger.error(
            "admob_ssv_verify_error",
            extra={"action": "admob_ssv_verify_error", "error": str(e)},
        )
        return False
