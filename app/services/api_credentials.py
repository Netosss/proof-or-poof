"""
API credential management for enterprise partners.

Key format:    fxl_live_<32 base62>  (or fxl_test_ in dev)
Secret format: fxs_live_<48 base62>  (returned ONCE at creation, never stored plaintext)

Storage:
    enterprise_partners/{partner_id}/api_credentials/{credential_id}
        api_key_prefix:   public display string ("fxl_live_abc...xyz")
        api_key_lookup:   SHA-256(full api_key) — indexed for O(1) reverse lookup
        secret_key:       raw HMAC signing key (required for verification)
        allowed_ips:      list[str] (CIDR strings) | None
        status:           "active" | "revoked"
        created_at, expires_at, last_used_at

    api_keys_index/{api_key_lookup}
        partner_id, credential_id

Why the secret_key is stored verbatim (not hashed):
    HMAC verification REQUIRES the server to recompute the signature with the
    same key the client used. This is the standard pattern at AWS, Stripe,
    Twilio, etc. — and is fundamentally different from password storage.
    Firestore encrypts all documents at rest with Google-managed keys; access
    is gated by IAM. Hashing the secret would make verification impossible.

The top-level index lets us resolve an incoming X-FauxLens-Key in a single
document read without requiring a Firestore collection-group index.
"""

import hashlib
import logging
import secrets
import string
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

from app.config import settings
from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)

_KEY_ALPHABET = string.ascii_letters + string.digits  # base62
_KEY_BODY_LEN = 32
_SECRET_BODY_LEN = 48


def _get_db():
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    return db


def _random_b62(length: int) -> str:
    return "".join(secrets.choice(_KEY_ALPHABET) for _ in range(length))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _key_prefix_display(api_key: str) -> str:
    """Returns a display-safe truncation: first 12 + ... + last 4 chars."""
    if len(api_key) <= 16:
        return api_key
    return f"{api_key[:12]}...{api_key[-4:]}"


async def create_credential(
    partner_id: str,
    allowed_ips: list[str] | None = None,
    expires_in_days: int | None = None,
) -> dict:
    """
    Provision a new credential pair for `partner_id`.

    Returns a dict containing:
      - api_key:        plaintext public key (show ONCE to operator)
      - secret_key:     plaintext secret (show ONCE to operator)
      - credential_id
      - api_key_prefix (safe to log)
    """
    db = _get_db()

    env_tag = "test" if settings.is_dev else "live"
    api_key = f"fxl_{env_tag}_{_random_b62(_KEY_BODY_LEN)}"
    secret_key = f"fxs_{env_tag}_{_random_b62(_SECRET_BODY_LEN)}"

    api_key_lookup = _hash(api_key)
    credential_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    expires_at = (now + timedelta(days=expires_in_days)) if expires_in_days else None

    partner_ref = db.collection("enterprise_partners").document(partner_id)
    if not (await partner_ref.get()).exists:
        raise HTTPException(status_code=404, detail=f"Partner {partner_id} not found")

    credential_data = {
        "api_key_prefix": _key_prefix_display(api_key),
        "api_key_lookup": api_key_lookup,
        "secret_key": secret_key,
        "allowed_ips": allowed_ips or [],
        "status": "active",
        "created_at": now,
        "expires_at": expires_at,
        "last_used_at": None,
    }

    await partner_ref.collection("api_credentials").document(credential_id).set(credential_data)

    # Top-level index for O(1) reverse lookup at request time.
    await db.collection("api_keys_index").document(api_key_lookup).set({
        "partner_id": partner_id,
        "credential_id": credential_id,
        "status": "active",
        "created_at": SERVER_TIMESTAMP,
    })

    logger.info(
        "enterprise_credential_created",
        extra={
            "action": "enterprise_credential_created",
            "partner_id": partner_id,
            "credential_id": credential_id,
            "api_key_prefix": _key_prefix_display(api_key),
        },
    )

    return {
        "credential_id": credential_id,
        "api_key": api_key,
        "secret_key": secret_key,
        "api_key_prefix": _key_prefix_display(api_key),
    }


async def resolve_credential(api_key: str) -> dict | None:
    """
    Resolve an incoming X-FauxLens-Key to its credential + partner record.

    Returns None on miss (caller raises 401). Performs two reads:
      1. api_keys_index/{lookup_hash}      — find (partner_id, credential_id)
      2. enterprise_partners/{p}/api_credentials/{c} — load credential

    Partner status is NOT loaded here — the credit engine transaction checks it
    inside the atomic deduct to avoid TOCTOU between auth and billing.
    """
    db = _get_db()
    lookup_hash = _hash(api_key)
    idx_snap = await db.collection("api_keys_index").document(lookup_hash).get()
    if not idx_snap.exists:
        return None
    idx = idx_snap.to_dict()
    if idx.get("status") != "active":
        return None

    partner_id = idx["partner_id"]
    credential_id = idx["credential_id"]

    cred_ref = (
        db.collection("enterprise_partners")
        .document(partner_id)
        .collection("api_credentials")
        .document(credential_id)
    )
    cred_snap = await cred_ref.get()
    if not cred_snap.exists:
        return None
    cred = cred_snap.to_dict()
    if cred.get("status") != "active":
        return None
    expires_at = cred.get("expires_at")
    if expires_at and isinstance(expires_at, datetime) and expires_at < datetime.now(UTC):
        return None

    return {
        "partner_id": partner_id,
        "credential_id": credential_id,
        "api_key_prefix": cred.get("api_key_prefix"),
        "secret_key": cred.get("secret_key"),
        "allowed_ips": cred.get("allowed_ips") or [],
    }


async def revoke_credential(partner_id: str, credential_id: str) -> None:
    """Mark a credential and its index entry revoked. Existing keys stop working immediately."""
    db = _get_db()
    cred_ref = (
        db.collection("enterprise_partners")
        .document(partner_id)
        .collection("api_credentials")
        .document(credential_id)
    )
    snap = await cred_ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Credential not found")
    cred = snap.to_dict()
    lookup_hash = cred.get("api_key_lookup")

    await cred_ref.update({"status": "revoked"})
    if lookup_hash:
        await db.collection("api_keys_index").document(lookup_hash).update({"status": "revoked"})

    logger.info(
        "enterprise_credential_revoked",
        extra={
            "action": "enterprise_credential_revoked",
            "partner_id": partner_id,
            "credential_id": credential_id,
        },
    )


async def touch_credential(partner_id: str, credential_id: str) -> None:
    """Update last_used_at — fire-and-forget after successful request."""
    db = _get_db()
    cred_ref = (
        db.collection("enterprise_partners")
        .document(partner_id)
        .collection("api_credentials")
        .document(credential_id)
    )
    try:
        await cred_ref.update({"last_used_at": SERVER_TIMESTAMP})
    except Exception as e:
        # Non-critical; don't fail the request if this write fails.
        logger.warning(
            "enterprise_credential_touch_failed",
            extra={
                "action": "enterprise_credential_touch_failed",
                "credential_id": credential_id,
                "error": str(e),
            },
        )


