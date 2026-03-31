"""
Credit engine for authenticated users.

All credit mutations go through consume_credits() or grant_credits().
Both functions operate inside a Firestore transaction so balance and
ledger are always consistent.

Ledger entries are append-only — they are never updated after creation.

Firestore collections used:
  users/{uid}                          — balance + metadata
  users/{uid}/credit_ledger/{auto_id}  — immutable audit trail

All Firestore operations are async (native AsyncClient) — no thread pool needed.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_ledger(transaction, user_ref, delta: int, reason: str,
                   reference_id: Optional[str], balance_after: int) -> None:
    """Appends one immutable ledger entry inside an existing transaction."""
    entry_ref = user_ref.collection("credit_ledger").document()
    transaction.set(entry_ref, {
        "delta": delta,
        "reason": reason,
        "reference_id": reference_id,
        "balance_after": balance_after,
        "created_at": SERVER_TIMESTAMP,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def consume_credits(user_id: str, cost: int, reason: str,
                          reference_id: Optional[str] = None) -> int:
    """
    Deducts `cost` credits from the authenticated user atomically.

    Raises HTTPException(402) if the balance is insufficient.
    Returns the new balance.
    """
    db = _get_db()
    user_ref = db.collection("users").document(user_id)

    @async_transactional
    async def _run(transaction, ref):
        snapshot = await ref.get(transaction=transaction)
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="User account not found")

        data = snapshot.to_dict() or {}
        current = data.get("credits_balance")
        if current is None:
            current = data.get("credits", 0)
        if current < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        new_balance = current - cost
        transaction.update(ref, {
            "credits_balance": new_balance,
            "credits_version": Increment(1),
        })
        _append_ledger(transaction, ref, -cost, reason, reference_id, new_balance)
        return new_balance

    try:
        txn = db.transaction()
        new_balance = await _run(txn, user_ref)
        logger.info("credits_consumed", extra={
            "action": "credits_consumed",
            "amount": cost,
            "reason": reason,
            "new_balance": new_balance,
        })
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error("credits_consume_failed", extra={
            "action": "credits_consume_failed",
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail="Credit transaction failed")


async def grant_credits(user_id: str, amount: int, reason: str,
                        reference_id: Optional[str] = None) -> int:
    """
    Adds `amount` credits to the authenticated user atomically.
    Negative amounts are allowed (refunds, chargebacks).
    Returns the new balance.
    """
    db = _get_db()
    user_ref = db.collection("users").document(user_id)

    @async_transactional
    async def _run(transaction, ref):
        snapshot = await ref.get(transaction=transaction)
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="User account not found")

        data = snapshot.to_dict() or {}
        current = data.get("credits_balance")
        if current is None:
            current = data.get("credits", 0)
        new_balance = current + amount
        transaction.update(ref, {
            "credits_balance": new_balance,
            "credits_version": Increment(1),
        })
        _append_ledger(transaction, ref, amount, reason, reference_id, new_balance)
        return new_balance

    try:
        txn = db.transaction()
        new_balance = await _run(txn, user_ref)
        logger.info("credits_granted", extra={
            "action": "credits_granted",
            "amount": amount,
            "reason": reason,
            "new_balance": new_balance,
        })
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error("credits_grant_failed", extra={
            "action": "credits_grant_failed",
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail="Credit transaction failed")


async def get_or_create_user(uid: str, email: Optional[str]) -> dict:
    """
    Returns the user doc for `uid`, creating it with a signup bonus if it does not exist.

    Guest and authenticated wallets are kept entirely separate — no credit migration.
    """
    db = _get_db()
    user_ref = db.collection("users").document(uid)

    # Fast path: user already exists.
    snapshot = await user_ref.get()
    if snapshot.exists:
        data = snapshot.to_dict()
        balance = data.get("credits_balance")
        if balance is None:
            balance = data.get("credits", 0)
        return {
            "uid": uid,
            "email": data.get("email"),
            "credits_balance": balance,
            "is_new_user": False,
        }

    @async_transactional
    async def _create_user(transaction, u_ref):
        u_snap = await u_ref.get(transaction=transaction)
        if u_snap.exists:
            data = u_snap.to_dict()
            balance = data.get("credits_balance")
            if balance is None:
                balance = data.get("credits", 0)
            return data, balance, False

        starting_balance = 40
        now = datetime.now(timezone.utc)
        user_data = {
            "email": email,
            "credits_balance": starting_balance,
            "credits_version": 1,
            "created_at": now,
        }
        transaction.set(u_ref, user_data)
        _append_ledger(
            transaction, u_ref,
            delta=starting_balance,
            reason="signup_bonus",
            reference_id=None,
            balance_after=starting_balance,
        )
        return user_data, starting_balance, True

    try:
        txn = db.transaction()
        data, balance, is_new = await _create_user(txn, user_ref)
        return {
            "uid": uid,
            "email": data.get("email") or email,
            "credits_balance": balance,
            "is_new_user": is_new,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("auth_get_or_create_failed", extra={
            "action": "auth_get_or_create_failed",
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail="User initialization failed")


async def get_user_balance(user_id: str) -> int:
    """Returns the current credit balance for an authenticated user."""
    db = _get_db()
    doc = await db.collection("users").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User account not found")
    data = doc.to_dict()
    balance = data.get("credits_balance")
    if balance is None:
        balance = data.get("credits", 0)
    return balance
