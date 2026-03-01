"""
Guest wallet management: creation, balance queries, credit deductions, and top-ups.

The Firebase `db` client is accessed at call-time via the integration module
so it picks up the instance initialized during the FastAPI lifespan.
"""

import logging
import os
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException
from firebase_admin import firestore

from app.config import settings
from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)

RECHARGE_SECRET_KEY = os.getenv("RECHARGE_SECRET_KEY", "")


def _get_db():
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    return db


def get_guest_wallet(device_id: str) -> dict:
    """Retrieves or creates a guest wallet for the device."""
    db = _get_db()
    doc_ref = db.collection('guest_wallets').document(device_id)
    doc = doc_ref.get()

    if doc.exists:
        return doc.to_dict()

    new_wallet = {
        "credits": settings.welcome_credits,
        "last_active": firestore.SERVER_TIMESTAMP,
        "is_banned": False,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
    }
    doc_ref.set(new_wallet)
    return new_wallet


def check_ban_status(device_id: str) -> bool:
    """Checks if the device is banned."""
    wallet = get_guest_wallet(device_id)
    return wallet.get("is_banned", False)


def deduct_guest_credits(device_id: str, cost: int = 5) -> int:
    """
    Deducts credits from the guest wallet atomically.
    Raises HTTPException(402) if insufficient funds.
    Returns the new balance.
    """
    db = _get_db()
    doc_ref = db.collection('guest_wallets').document(device_id)

    @firestore.transactional
    def update_in_transaction(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            transaction.set(ref, {
                "credits": settings.welcome_credits,
                "last_active": firestore.SERVER_TIMESTAMP,
                "is_banned": False,
                "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
            })
            current_credits = settings.welcome_credits
        else:
            current_credits = snapshot.get("credits")
            if current_credits is None:
                current_credits = 0

        if current_credits < cost:
            raise HTTPException(status_code=402, detail="Insufficient credits")

        transaction.update(ref, {
            "credits": current_credits - cost,
            "last_active": firestore.SERVER_TIMESTAMP,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
        })
        return current_credits - cost

    transaction = db.transaction()
    try:
        new_balance = update_in_transaction(transaction, doc_ref)
        logger.info(f"Deducted {cost} credits from device {device_id}. New balance: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Guest credit deduction failed for {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Wallet transaction failed")


def perform_recharge(device_id: str, amount: int, secret_key: str) -> dict:
    """
    Top up a guest wallet.
    Validates `secret_key` against RECHARGE_SECRET_KEY env var.
    """
    if not RECHARGE_SECRET_KEY or secret_key != RECHARGE_SECRET_KEY:
        logger.warning(f"Invalid recharge attempt for {device_id}")
        raise HTTPException(status_code=403, detail="Invalid secret key")

    db = _get_db()

    try:
        doc_ref = db.collection('guest_wallets').document(device_id)

        @firestore.transactional
        def recharge_transaction(transaction, ref):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                transaction.set(ref, {
                    "credits": settings.welcome_credits + amount,
                    "last_active": firestore.SERVER_TIMESTAMP,
                    "is_banned": False,
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
                })
                return settings.welcome_credits + amount
            else:
                current = snapshot.get("credits") or 0
                transaction.update(ref, {
                    "credits": current + amount,
                    "last_active": firestore.SERVER_TIMESTAMP,
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
                })
                return current + amount

        transaction = db.transaction()
        new_balance = recharge_transaction(transaction, doc_ref)
        logger.info(f"Recharged {amount} credits for {device_id}. New balance: {new_balance}")
        return {"status": "success", "new_balance": new_balance}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recharge failed: {e}")
        raise HTTPException(status_code=500, detail="Recharge failed")
