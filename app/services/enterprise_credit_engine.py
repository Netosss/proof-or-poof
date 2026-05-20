"""
Enterprise credit engine — atomic balance mutations for partner accounts.

Mirrors the design of `services/credit_engine.py` but operates on the separate
`enterprise_partners/{id}` collection. Ledger entries live in the same-named
subcollection. All mutations are transactional and append a ledger row.

Refund-on-failure is idempotent: before refunding, we query the ledger for an
existing refund entry tagged with the same `reference_id` (= request_id) and
no-op if one is found. This makes retry-safe automated refunds possible.

Critical invariant — credits are NEVER deducted on scan failure:
    The route layer calls `reserve_credit` BEFORE running the pipeline, and
    `refund_credit` from a finally/except path if the pipeline raises or returns
    a synthetic "Analysis Failed" verdict. The pipeline crash → automatic refund
    path is the recommended industry pattern (deduct-then-refund-on-failure)
    over check-then-deduct because the former is race-free under concurrent
    requests.
"""

import logging

from fastapi import HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP, Increment
from google.cloud.firestore_v1.async_transaction import async_transactional

from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)


def _get_db():
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    return db


_LEDGER_ID_SAFE_RE = __import__("re").compile(r"[^A-Za-z0-9_\-:.]")


def _ledger_doc_id(prefix: str, reason: str, reference_id: str) -> str:
    """Build a deterministic, Firestore-safe ledger entry doc ID.

    Used as an atomic idempotency key inside a transaction — calling the same
    mutation twice with the same (reason, reference_id) collides on the doc
    create() and is detected before any balance write.
    """
    safe = _LEDGER_ID_SAFE_RE.sub("_", f"{prefix}:{reason}:{reference_id}")
    # Firestore doc IDs are capped at ~1500 bytes; keep well under that.
    return safe[:200]


def _append_ledger(transaction, partner_ref, *, delta: int, reason: str,
                   reference_id: str | None, balance_after: int,
                   doc_id: str | None = None) -> None:
    entry_ref = (
        partner_ref.collection("credit_ledger").document(doc_id)
        if doc_id
        else partner_ref.collection("credit_ledger").document()
    )
    transaction.set(entry_ref, {
        "delta": delta,
        "reason": reason,
        "reference_id": reference_id,
        "balance_after": balance_after,
        "created_at": SERVER_TIMESTAMP,
    })


async def reserve_credit(partner_id: str, cost: int, reason: str,
                         reference_id: str) -> int:
    """
    Atomically check partner is active, deduct `cost` credits, and append ledger.

    Raises:
        HTTPException(402, "insufficient_credits") if balance < cost
        HTTPException(403, "partner_suspended") if status != "active"
        HTTPException(404, "partner_not_found")
    Returns the new balance.
    """
    db = _get_db()
    partner_ref = db.collection("enterprise_partners").document(partner_id)

    @async_transactional
    async def _run(transaction, ref):
        snap = await ref.get(transaction=transaction)
        if not snap.exists:
            raise HTTPException(status_code=404, detail="partner_not_found")
        data = snap.to_dict() or {}
        status = data.get("status", "active")
        if status != "active":
            raise HTTPException(status_code=403, detail=f"partner_{status}")
        balance = int(data.get("credit_balance", 0))
        if balance < cost:
            raise HTTPException(status_code=402, detail="insufficient_credits")

        new_balance = balance - cost
        transaction.update(ref, {
            "credit_balance": new_balance,
            "credits_version": Increment(1),
            "updated_at": SERVER_TIMESTAMP,
        })
        _append_ledger(
            transaction, ref,
            delta=-cost,
            reason=reason,
            reference_id=reference_id,
            balance_after=new_balance,
        )
        return new_balance

    try:
        txn = db.transaction()
        new_balance = await _run(txn, partner_ref)
        logger.info(
            "enterprise_credit_reserved",
            extra={
                "action": "enterprise_credit_reserved",
                "partner_id": partner_id,
                "amount": cost,
                "reason": reason,
                "reference_id": reference_id,
                "new_balance": new_balance,
            },
        )
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "enterprise_credit_reserve_failed",
            extra={
                "action": "enterprise_credit_reserve_failed",
                "partner_id": partner_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="credit_transaction_failed")


async def refund_credit(partner_id: str, amount: int, reason: str,
                        reference_id: str) -> int | None:
    """
    Idempotently refund `amount` credits to `partner_id`.

    Idempotency is enforced ATOMICALLY inside the Firestore transaction by
    using a deterministic ledger doc id derived from (reason, reference_id).
    A concurrent second call sees the ledger doc inside the transaction and
    no-ops without touching the balance — no race window vs. the previous
    "query-outside-transaction" approach.

    Returns the new balance after refund, or None if already refunded.
    """
    db = _get_db()
    partner_ref = db.collection("enterprise_partners").document(partner_id)
    ledger_id = _ledger_doc_id("refund", reason, reference_id)
    ledger_ref = partner_ref.collection("credit_ledger").document(ledger_id)

    @async_transactional
    async def _run(transaction, ref, lref):
        # READS (must precede writes inside a Firestore transaction)
        ledger_snap = await lref.get(transaction=transaction)
        if ledger_snap.exists:
            return None
        snap = await ref.get(transaction=transaction)
        if not snap.exists:
            raise HTTPException(status_code=404, detail="partner_not_found")
        data = snap.to_dict() or {}
        balance = int(data.get("credit_balance", 0))
        new_balance = balance + amount
        # WRITES
        transaction.update(ref, {
            "credit_balance": new_balance,
            "credits_version": Increment(1),
            "updated_at": SERVER_TIMESTAMP,
        })
        _append_ledger(
            transaction, ref,
            delta=amount,
            reason=reason,
            reference_id=reference_id,
            balance_after=new_balance,
            doc_id=ledger_id,
        )
        return new_balance

    try:
        txn = db.transaction()
        new_balance = await _run(txn, partner_ref, ledger_ref)
        if new_balance is None:
            logger.info(
                "enterprise_credit_refund_already_applied",
                extra={
                    "action": "enterprise_credit_refund_already_applied",
                    "partner_id": partner_id,
                    "reference_id": reference_id,
                    "reason": reason,
                },
            )
            return None
        logger.info(
            "enterprise_credit_refunded",
            extra={
                "action": "enterprise_credit_refunded",
                "partner_id": partner_id,
                "amount": amount,
                "reason": reason,
                "reference_id": reference_id,
                "new_balance": new_balance,
            },
        )
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "enterprise_credit_refund_failed",
            extra={
                "action": "enterprise_credit_refund_failed",
                "partner_id": partner_id,
                "reference_id": reference_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="credit_refund_failed")


async def grant_credit(partner_id: str, amount: int, reason: str,
                       reference_id: str) -> int:
    """
    Add `amount` credits to a partner — used by the Lemon Squeezy webhook on
    `order_paid`. Idempotent on (reason, reference_id) via a deterministic
    ledger doc id, so a retried webhook does NOT double-grant credits even
    if the purchase-doc status update failed between attempts.

    Returns the new balance (or the unchanged balance if this grant was
    already applied).
    """
    db = _get_db()
    partner_ref = db.collection("enterprise_partners").document(partner_id)
    ledger_id = _ledger_doc_id("grant", reason, reference_id)
    ledger_ref = partner_ref.collection("credit_ledger").document(ledger_id)

    @async_transactional
    async def _run(transaction, ref, lref):
        # READS
        ledger_snap = await lref.get(transaction=transaction)
        partner_snap = await ref.get(transaction=transaction)
        if not partner_snap.exists:
            raise HTTPException(status_code=404, detail="partner_not_found")
        data = partner_snap.to_dict() or {}
        balance = int(data.get("credit_balance", 0))
        if ledger_snap.exists:
            return balance  # already granted — no-op, return current balance
        new_balance = balance + amount
        # WRITES
        transaction.update(ref, {
            "credit_balance": new_balance,
            "credits_version": Increment(1),
            "updated_at": SERVER_TIMESTAMP,
        })
        _append_ledger(
            transaction, ref,
            delta=amount,
            reason=reason,
            reference_id=reference_id,
            balance_after=new_balance,
            doc_id=ledger_id,
        )
        return new_balance

    try:
        txn = db.transaction()
        new_balance = await _run(txn, partner_ref, ledger_ref)
        logger.info(
            "enterprise_credit_granted",
            extra={
                "action": "enterprise_credit_granted",
                "partner_id": partner_id,
                "amount": amount,
                "reason": reason,
                "reference_id": reference_id,
                "new_balance": new_balance,
            },
        )
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "enterprise_credit_grant_failed",
            extra={
                "action": "enterprise_credit_grant_failed",
                "partner_id": partner_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="credit_grant_failed")


async def get_partner_balance(partner_id: str) -> int:
    db = _get_db()
    snap = await db.collection("enterprise_partners").document(partner_id).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="partner_not_found")
    return int((snap.to_dict() or {}).get("credit_balance", 0))
