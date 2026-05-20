"""
Enterprise partner registry — Firestore-backed.

A partner is a top-level corporate entity holding a prepaid credit balance.
Credentials live in a subcollection (see api_credentials.py) so a single
partner can rotate keys without losing balance or ledger history.

Firestore layout:
    enterprise_partners/{partner_id}
      ├─ company_name, contact_email, status, credit_balance, credits_version
      ├─ rate_limit_per_min  (optional override)
      └─ /credit_ledger/{auto_id}       — append-only ledger
      └─ /api_credentials/{cred_id}     — see api_credentials.py
"""

import logging
import uuid
from datetime import UTC, datetime

from fastapi import HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)

ALLOWED_STATUSES = ("active", "suspended", "frozen")


def _get_db():
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    return db


async def create_partner(
    company_name: str,
    contact_email: str,
    initial_credits: int = 0,
    rate_limit_per_min: int | None = None,
    firebase_uid: str | None = None,
) -> dict:
    """Create a new enterprise partner. Returns the partner record (incl. id).

    `firebase_uid` links the partner record to a signed-in Firebase user so the
    browser dashboard can resolve "which partner am I?" from the auth token.
    """
    db = _get_db()
    partner_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    data = {
        "company_name": company_name,
        "contact_email": contact_email,
        "credit_balance": int(initial_credits),
        "credits_version": 1,
        "status": "active",
        "rate_limit_per_min": rate_limit_per_min,
        "firebase_uid": firebase_uid,
        "created_at": now,
        "updated_at": now,
    }
    await db.collection("enterprise_partners").document(partner_id).set(data)
    logger.info(
        "enterprise_partner_created",
        extra={
            "action": "enterprise_partner_created",
            "partner_id": partner_id,
            "contact_email": contact_email,
        },
    )
    return {"id": partner_id, **data}


async def get_partner(partner_id: str) -> dict | None:
    db = _get_db()
    snap = await db.collection("enterprise_partners").document(partner_id).get()
    if not snap.exists:
        return None
    return {"id": partner_id, **snap.to_dict()}


async def maybe_raise_rate_limit(partner_id: str, new_limit: int) -> int | None:
    """Raise a partner's per-minute rate limit if `new_limit` exceeds the
    current ceiling. Returns the limit after the operation (None if the
    partner doesn't exist).

    Semantics:
      - `partner.rate_limit_per_min == None` is treated as the system default
        (`settings.enterprise_default_rate_limit_per_min`). A Starter customer
        with no override still gets bumped to Pro's 120 when they upgrade.
      - We never LOWER the limit. A Scale customer who buys a small Starter
        top-up keeps Scale's 300 — the field is a ceiling, not the tier.
      - Idempotent: a webhook retry that sees the same or lower `new_limit`
        is a no-op.

    Returns the effective `rate_limit_per_min` after the write (or after the
    no-op if no write was needed).
    """
    from app.config import settings  # local import to avoid circular config wiring

    if new_limit <= 0:
        return None

    db = _get_db()
    ref = db.collection("enterprise_partners").document(partner_id)
    snap = await ref.get()
    if not snap.exists:
        logger.warning(
            "enterprise_rate_limit_partner_not_found",
            extra={
                "action": "enterprise_rate_limit_partner_not_found",
                "partner_id": partner_id,
                "new_limit": new_limit,
            },
        )
        return None

    partner = snap.to_dict() or {}
    current = partner.get("rate_limit_per_min")
    effective_current = (
        int(current) if isinstance(current, int) else settings.enterprise_default_rate_limit_per_min
    )

    if new_limit <= effective_current:
        logger.info(
            "enterprise_rate_limit_unchanged",
            extra={
                "action": "enterprise_rate_limit_unchanged",
                "partner_id": partner_id,
                "current": effective_current,
                "proposed": new_limit,
                "reason": "proposed_not_higher",
            },
        )
        return effective_current

    await ref.update(
        {
            "rate_limit_per_min": int(new_limit),
            "updated_at": SERVER_TIMESTAMP,
        }
    )
    logger.info(
        "enterprise_rate_limit_raised",
        extra={
            "action": "enterprise_rate_limit_raised",
            "partner_id": partner_id,
            "from": effective_current,
            "to": int(new_limit),
        },
    )
    return int(new_limit)


async def set_partner_status(partner_id: str, status: str) -> None:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid status {status!r}; expected one of {ALLOWED_STATUSES}")
    db = _get_db()
    await (
        db.collection("enterprise_partners")
        .document(partner_id)
        .update({"status": status, "updated_at": SERVER_TIMESTAMP})
    )
    logger.info(
        "enterprise_partner_status_changed",
        extra={
            "action": "enterprise_partner_status_changed",
            "partner_id": partner_id,
            "status": status,
        },
    )


async def list_partners(limit: int = 100) -> list[dict]:
    db = _get_db()
    out = []
    async for snap in db.collection("enterprise_partners").limit(limit).stream():
        out.append({"id": snap.id, **snap.to_dict()})
    return out


async def find_partner_for_uid(firebase_uid: str) -> dict | None:
    """Look up the partner record linked to a signed-in Firebase user."""
    db = _get_db()
    q = db.collection("enterprise_partners").where("firebase_uid", "==", firebase_uid).limit(1)
    async for snap in q.stream():
        return {"id": snap.id, **snap.to_dict()}
    return None


async def list_partner_credentials(partner_id: str) -> list[dict]:
    """List active + revoked credentials for a partner (no secrets returned)."""
    db = _get_db()
    out = []
    creds_ref = (
        db.collection("enterprise_partners").document(partner_id).collection("api_credentials")
    )
    async for snap in creds_ref.stream():
        d = snap.to_dict() or {}
        # Strip the raw secret — never return it after creation.
        d.pop("secret_key", None)
        out.append({"id": snap.id, **d})
    return out


async def list_partner_ledger(partner_id: str, limit: int = 50) -> list[dict]:
    """Return the most-recent `limit` ledger entries for a partner.

    Uses server-side `order_by(created_at desc) + limit(N)` so a partner with
    a long history never causes us to stream the entire collection. Falls
    back to a Python-side sort by `created_at` (or balance_after if that's
    missing — only in dev/test where the mock doesn't honour order_by).
    """
    from google.cloud.firestore_v1 import Query

    db = _get_db()
    ledger_ref = (
        db.collection("enterprise_partners").document(partner_id).collection("credit_ledger")
    )

    out: list[dict] = []
    try:
        q = ledger_ref.order_by("created_at", direction=Query.DESCENDING).limit(limit)
        async for snap in q.stream():
            out.append({"id": snap.id, **snap.to_dict()})
        if out:
            return out
    except Exception as e:
        # Mock or legacy stores without the index fall through to streaming.
        logger.debug(
            "list_ledger_order_by_unavailable",
            extra={"action": "list_ledger_order_by_unavailable", "error": str(e)},
        )

    out = []
    async for snap in ledger_ref.stream():
        out.append({"id": snap.id, **snap.to_dict()})

    def _sort_key(entry: dict):
        ts = entry.get("created_at")
        if ts is not None:
            return ts
        return entry.get("balance_after", 0)

    out.sort(key=_sort_key, reverse=True)
    return out[:limit]
