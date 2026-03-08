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
        current = data.get("credits_balance") or 0
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
        current = data.get("credits_balance") or 0
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


async def get_or_create_user(uid: str, email: Optional[str],
                             device_id: Optional[str] = None) -> dict:
    """
    Returns the user doc for `uid`, creating it if it does not exist.

    On creation, applies the strict migration rule — fully inside one
    Firestore transaction to prevent TOCTOU race conditions.
    """
    db = _get_db()
    user_ref = db.collection("users").document(uid)
    guest_ref = db.collection("guest_wallets").document(device_id) if device_id else None

    # Fast path: user already exists (non-transactional read is fine here —
    # the transaction below double-checks atomically before writing).
    snapshot = await user_ref.get()
    if snapshot.exists:
        data = snapshot.to_dict()
        return {
            "uid": uid,
            "email": data.get("email"),
            "credits_balance": data.get("credits_balance", 0),
            "is_new_user": False,
        }

    @async_transactional
    async def _create_user(transaction, u_ref, g_ref):
        u_snap = await u_ref.get(transaction=transaction)
        if u_snap.exists:
            return u_snap.to_dict(), False

        starting_balance = 0
        bonus_reason = "signup_bonus"
        bonus_ref_id = None
        was_hijack_attempt = False

        if g_ref is not None:
            g_snap = await g_ref.get(transaction=transaction)
            if g_snap.exists:
                wallet = g_snap.to_dict()
                if wallet.get("is_migrated"):
                    was_hijack_attempt = True
                    starting_balance = 0
                    bonus_reason = "none"
                else:
                    guest_credits = wallet.get("credits", 0)
                    if guest_credits > 0:
                        starting_balance = guest_credits
                        bonus_reason = "guest_migration"
                        bonus_ref_id = device_id
                        transaction.update(g_ref, {
                            "credits": 0,
                            "is_migrated": True,
                            "is_banned": True,
                        })
                    else:
                        starting_balance = 40
                        bonus_reason = "signup_bonus"
            else:
                starting_balance = 40
                bonus_reason = "signup_bonus"
        else:
            starting_balance = 40
            bonus_reason = "signup_bonus"

        now = datetime.now(timezone.utc)
        user_data = {
            "email": email,
            "credits_balance": starting_balance,
            "credits_version": 1,
            "created_at": now,
        }
        transaction.set(u_ref, user_data)

        if bonus_reason != "none":
            _append_ledger(
                transaction, u_ref,
                delta=starting_balance,
                reason=bonus_reason,
                reference_id=bonus_ref_id,
                balance_after=starting_balance,
            )

        if was_hijack_attempt:
            logger.warning("auth_hijack_blocked", extra={
                "action": "auth_hijack_blocked",
                "device_id": device_id,
            })

        return user_data, True

    try:
        txn = db.transaction()
        data, is_new = await _create_user(txn, user_ref, guest_ref)
        balance = data.get("credits_balance", 0)
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
    return doc.to_dict().get("credits_balance", 0)
