"""
Enterprise application intake — pre-provisioning pipeline.

Anyone with a signed-in Firebase user can submit an application. Sandbox
applications land in `status: 'pending'` and require operator approval via the
CLI (`enterprise_admin.py approve-application`). Paid applications are routed
straight to Lemon Squeezy — payment is the trust signal, no human gate needed.

Anti-abuse layers:
    1. Disposable-email domain blocklist (rejected at submission)
    2. One non-revoked application per Firebase UID (idempotency by UID)
    3. Free-email domains flagged (visible to operator, not auto-rejected)
    4. Per-IP rate limit applied at the route layer

Firestore layout:
    enterprise_applications/{auto_id}
      firebase_uid:    str
      contact_email:   str
      company_name:    str
      use_case:        str        # newsroom / trust_safety / insurance / marketplace / other
      expected_volume: str        # under_2k / 2k_10k / 10k_25k / over_25k
      tier:            str        # sandbox / starter / pro / scale
      notes:           str | None
      status:          str        # pending / approved / rejected / provisioned
      free_email:      bool       # flagged if email domain is a personal-mail provider
      partner_id:      str | None # populated after approval
      created_at, updated_at, approved_at
"""

import logging
from datetime import UTC, datetime

from fastapi import HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

from app.integrations import firebase as firebase_module


def _is_sentinel(value) -> bool:
    """Detect Firestore sentinel objects so they don't leak into JSON responses."""
    return type(value).__name__ in ("ServerTimestamp", "Sentinel", "_UNSET_SENTINEL", "Increment")


logger = logging.getLogger(__name__)


# Curated list — small enough to maintain by hand, large enough to block the
# common offenders. Expand from production logs as new patterns appear.
_DISPOSABLE_EMAIL_DOMAINS: frozenset[str] = frozenset(
    {
        "mailinator.com",
        "guerrillamail.com",
        "tempmail.com",
        "10minutemail.com",
        "trashmail.com",
        "throwaway.email",
        "yopmail.com",
        "fakeinbox.com",
        "maildrop.cc",
        "sharklasers.com",
        "getairmail.com",
        "dispostable.com",
        "mintemail.com",
        "mytrashmail.com",
        "spambox.us",
        "tempinbox.com",
        "tempr.email",
        "33mail.com",
        "spamgourmet.com",
        "mail-temporaire.fr",
        "anonbox.net",
        "byom.de",
        "tmpmail.org",
        "tmpmail.net",
        "burnermail.io",
        "mohmal.com",
        "easytrashmail.com",
        "linshiyouxiang.net",
        "fakemail.net",
        "emailondeck.com",
        "moakt.com",
        "harakirimail.com",
        "mintmail.cc",
        "mail.tm",
        "mail7.io",
    }
)

# Personal email providers — flagged but not blocked.
_FREE_EMAIL_DOMAINS: frozenset[str] = frozenset(
    {
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "icloud.com",
        "live.com",
        "msn.com",
        "aol.com",
        "protonmail.com",
        "proton.me",
        "yandex.com",
        "yandex.ru",
        "gmx.com",
        "gmx.net",
        "mail.com",
        "zoho.com",
        "tutanota.com",
        "fastmail.com",
        "qq.com",
        "163.com",
        "126.com",
        "naver.com",
    }
)

ALLOWED_USE_CASES = ("newsroom", "trust_safety", "insurance", "marketplace", "research", "other")
ALLOWED_VOLUMES = ("under_2k", "2k_10k", "10k_25k", "over_25k")
ALLOWED_TIERS = ("sandbox", "starter", "pro", "scale")


def _get_db():
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    return db


def _email_domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].strip().lower() if "@" in email else ""


def is_disposable_email(email: str) -> bool:
    return _email_domain(email) in _DISPOSABLE_EMAIL_DOMAINS


def is_free_email(email: str) -> bool:
    return _email_domain(email) in _FREE_EMAIL_DOMAINS


async def find_application_for_uid(firebase_uid: str) -> dict | None:
    """Return the most recent application for this Firebase user (any status)."""
    db = _get_db()
    q = db.collection("enterprise_applications").where("firebase_uid", "==", firebase_uid).limit(1)
    async for snap in q.stream():
        return {"id": snap.id, **snap.to_dict()}
    return None


async def create_application(
    *,
    firebase_uid: str,
    contact_email: str,
    company_name: str,
    use_case: str,
    expected_volume: str,
    tier: str,
    notes: str | None = None,
) -> dict:
    """Validate + write a new application doc. Idempotent per Firebase UID."""

    if not company_name or len(company_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="company_name is required")
    if not contact_email or "@" not in contact_email:
        raise HTTPException(status_code=400, detail="valid contact_email is required")
    if use_case not in ALLOWED_USE_CASES:
        raise HTTPException(status_code=400, detail=f"use_case must be one of {ALLOWED_USE_CASES}")
    if expected_volume not in ALLOWED_VOLUMES:
        raise HTTPException(
            status_code=400, detail=f"expected_volume must be one of {ALLOWED_VOLUMES}"
        )
    if tier not in ALLOWED_TIERS:
        raise HTTPException(status_code=400, detail=f"tier must be one of {ALLOWED_TIERS}")
    if is_disposable_email(contact_email):
        raise HTTPException(
            status_code=400,
            detail="disposable email addresses are not accepted — please use a work email",
        )

    existing = await find_application_for_uid(firebase_uid)
    if existing and existing.get("status") in ("pending", "approved", "provisioned"):
        # Return existing record so the UI can show "you already applied" gracefully.
        # Strip Firestore sentinel objects from the response for JSON safety.
        out = {k: v for k, v in existing.items() if not _is_sentinel(v)}
        return out

    db = _get_db()
    data = {
        "firebase_uid": firebase_uid,
        "contact_email": contact_email.strip().lower(),
        "company_name": company_name.strip(),
        "use_case": use_case,
        "expected_volume": expected_volume,
        "tier": tier,
        "notes": (notes or "").strip()[:500] or None,
        "status": "pending",
        "free_email": is_free_email(contact_email),
        "partner_id": None,
        "created_at": SERVER_TIMESTAMP,
        "updated_at": SERVER_TIMESTAMP,
        "approved_at": None,
    }
    ref = db.collection("enterprise_applications").document()
    await ref.set(data)

    logger.info(
        "enterprise_application_created",
        extra={
            "action": "enterprise_application_created",
            "application_id": ref.id,
            "firebase_uid": firebase_uid,
            "tier": tier,
            "free_email": data["free_email"],
            "company_name": data["company_name"],
            "use_case": use_case,
        },
    )
    # Replace SERVER_TIMESTAMP sentinels with python datetimes for the response.
    now = datetime.now(UTC)
    data["created_at"] = now
    data["updated_at"] = now
    return {"id": ref.id, **data}


async def mark_application_approved(application_id: str, partner_id: str) -> None:
    db = _get_db()
    await (
        db.collection("enterprise_applications")
        .document(application_id)
        .update(
            {
                "status": "provisioned",
                "partner_id": partner_id,
                "approved_at": SERVER_TIMESTAMP,
                "updated_at": SERVER_TIMESTAMP,
            }
        )
    )


async def mark_application_rejected(application_id: str, reason: str = "") -> None:
    """Reject a pending application. Used by the operator CLI when the
    review concludes the application isn't a good fit (volume mismatch,
    abuse risk, off-use-case, etc.). The dashboard's RejectedView renders
    based on `status=='rejected'` — that's the only user-visible effect."""
    db = _get_db()
    await (
        db.collection("enterprise_applications")
        .document(application_id)
        .update(
            {
                "status": "rejected",
                "rejection_reason": reason or None,
                "rejected_at": SERVER_TIMESTAMP,
                "updated_at": SERVER_TIMESTAMP,
            }
        )
    )


async def get_application(application_id: str) -> dict | None:
    db = _get_db()
    snap = await db.collection("enterprise_applications").document(application_id).get()
    if not snap.exists:
        return None
    return {"id": application_id, **snap.to_dict()}


async def list_pending_applications(limit: int = 50) -> list[dict]:
    """Operator helper — list pending sandbox applications awaiting approval."""
    db = _get_db()
    q = db.collection("enterprise_applications").where("status", "==", "pending").limit(limit)
    out = []
    async for snap in q.stream():
        out.append({"id": snap.id, **snap.to_dict()})
    return out
