"""
Credit engine for authenticated users.

All credit mutations go through consume_credits() or grant_credits().
Both functions operate inside a Firestore transaction so balance and
ledger are always consistent.

Ledger entries are append-only — they are never updated after creation.

Firestore collections used:
  users/{uid}                          — balance + metadata
  users/{uid}/credit_ledger/{auto_id}  — immutable audit trail
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from firebase_admin import firestore

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
        "created_at": firestore.SERVER_TIMESTAMP,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def consume_credits(user_id: str, cost: int, reason: str,
                    reference_id: Optional[str] = None) -> int:
    """
    Deducts `cost` credits from the authenticated user atomically.

    Raises HTTPException(402) if the balance is insufficient.
    Returns the new balance.
    """
    db = _get_db()
    user_ref = db.collection("users").document(user_id)

    @firestore.transactional
    def _run(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="User account not found")

        current = snapshot.get("credits_balance") or 0
        if current < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        new_balance = current - cost
        transaction.update(ref, {
            "credits_balance": new_balance,
            "credits_version": firestore.Increment(1),
        })
        _append_ledger(transaction, ref, -cost, reason, reference_id, new_balance)
        return new_balance

    try:
        txn = db.transaction()
        new_balance = _run(txn, user_ref)
        logger.info(
            f"[CREDITS] Consumed {cost} from {user_id} "
            f"(reason={reason}). New balance: {new_balance}"
        )
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CREDITS] consume_credits failed for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Credit transaction failed")


def grant_credits(user_id: str, amount: int, reason: str,
                  reference_id: Optional[str] = None) -> int:
    """
    Adds `amount` credits to the authenticated user atomically.
    Negative amounts are allowed (refunds, chargebacks).
    Returns the new balance.
    """
    db = _get_db()
    user_ref = db.collection("users").document(user_id)

    @firestore.transactional
    def _run(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="User account not found")

        current = snapshot.get("credits_balance") or 0
        new_balance = current + amount
        transaction.update(ref, {
            "credits_balance": new_balance,
            "credits_version": firestore.Increment(1),
        })
        _append_ledger(transaction, ref, amount, reason, reference_id, new_balance)
        return new_balance

    try:
        txn = db.transaction()
        new_balance = _run(txn, user_ref)
        logger.info(
            f"[CREDITS] Granted {amount} to {user_id} "
            f"(reason={reason}). New balance: {new_balance}"
        )
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CREDITS] grant_credits failed for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Credit transaction failed")


def get_or_create_user(uid: str, email: Optional[str],
                       device_id: Optional[str] = None) -> dict:
    """
    Returns the user doc for `uid`, creating it if it does not exist.

    On creation, applies the strict migration rule — fully inside one
    Firestore transaction to prevent TOCTOU race conditions:

    Scenario A — newbie (no device_id, or guest wallet has 0 credits,
                          or wallet is already migrated):
        Grant 40 welcome credits via signup_bonus ledger entry.
        If the wallet is already migrated (hijack attempt), the user
        gets 0 credits — not the 40 signup bonus.

    Scenario B — converting guest (device_id has credits > 0,
                                    wallet is NOT migrated):
        Migrate the exact guest balance, lock the guest wallet
        (is_migrated=True, is_banned=True, credits=0).
        The 40 welcome bonus is NOT added on top.

    The guest wallet is re-read INSIDE the transaction to prevent the
    TOCTOU window where two concurrent callers both see is_migrated=False
    before either commits.

    Returns dict: {uid, email, credits_balance, is_new_user}
    """
    db = _get_db()
    user_ref = db.collection("users").document(uid)
    guest_ref = db.collection("guest_wallets").document(device_id) if device_id else None

    # Fast path: user already exists (non-transactional read is fine here —
    # the transaction below double-checks atomically before writing).
    snapshot = user_ref.get()
    if snapshot.exists:
        data = snapshot.to_dict()
        return {
            "uid": uid,
            "email": data.get("email"),
            "credits_balance": data.get("credits_balance", 0),
            "is_new_user": False,
        }

    @firestore.transactional
    def _create_user(transaction, u_ref, g_ref):
        # Transactional double-check: guards against concurrent first-login.
        u_snap = u_ref.get(transaction=transaction)
        if u_snap.exists:
            return u_snap.to_dict(), False

        starting_balance = 0
        bonus_reason = "signup_bonus"
        bonus_ref_id = None
        was_hijack_attempt = False

        if g_ref is not None:
            # Re-read the guest wallet INSIDE the transaction — this is the
            # key fix for the TOCTOU race: two concurrent callers with
            # different UIDs can't both migrate the same wallet.
            g_snap = g_ref.get(transaction=transaction)
            if g_snap.exists:
                wallet = g_snap.to_dict()
                if wallet.get("is_migrated"):
                    # Wallet already claimed — hijack attempt or race lost.
                    # Grant 0 credits (not the 40 signup bonus) as a
                    # deterrent: don't reward the attacker with a free account.
                    was_hijack_attempt = True
                    starting_balance = 0
                    bonus_reason = "none"
                else:
                    guest_credits = wallet.get("credits", 0)
                    if guest_credits > 0:
                        # Scenario B: migrate exact guest balance
                        starting_balance = guest_credits
                        bonus_reason = "guest_migration"
                        bonus_ref_id = device_id
                        transaction.update(g_ref, {
                            "credits": 0,
                            "is_migrated": True,
                            "is_banned": True,
                        })
                    else:
                        # Guest wallet exists but has 0 credits → Scenario A
                        starting_balance = 40
                        bonus_reason = "signup_bonus"
            else:
                # device_id provided but no wallet found → Scenario A
                starting_balance = 40
                bonus_reason = "signup_bonus"
        else:
            # No device_id → Scenario A
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

        # Always append a ledger entry, even for 0-credit accounts,
        # so the audit trail is complete.
        if bonus_reason != "none":
            _append_ledger(
                transaction, u_ref,
                delta=starting_balance,
                reason=bonus_reason,
                reference_id=bonus_ref_id,
                balance_after=starting_balance,
            )

        if was_hijack_attempt:
            logger.warning(
                f"[AUTH] Hijack attempt blocked: device_id={device_id} "
                f"already migrated. uid={uid} created with 0 credits."
            )

        return user_data, True

    try:
        txn = db.transaction()
        data, is_new = _create_user(txn, user_ref, guest_ref)
        balance = data.get("credits_balance", 0)
        logger.info(
            f"[AUTH] {'Created' if is_new else 'Found existing'} user {uid} "
            f"(balance={balance}, device_id={device_id})"
        )
        return {
            "uid": uid,
            "email": data.get("email") or email,
            "credits_balance": balance,
            "is_new_user": is_new,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTH] get_or_create_user failed for {uid}: {e}")
        raise HTTPException(status_code=500, detail="User initialization failed")


def get_user_balance(user_id: str) -> int:
    """Returns the current credit balance for an authenticated user."""
    db = _get_db()
    doc = db.collection("users").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User account not found")
    return doc.to_dict().get("credits_balance", 0)
