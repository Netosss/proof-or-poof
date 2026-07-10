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
    pem = _keys_cache.get(key_id)
    if pem is None:
        # Miss -> refresh (covers first use and Google's periodic key rotation).
        # Serialize concurrent misses and re-check under the lock so only the
        # first waiter actually performs the network fetch.
        async with _keys_fetch_lock:
            pem = _keys_cache.get(key_id)
            if pem is None:
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
