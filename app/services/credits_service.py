"""
Guest wallet management: creation, balance queries, credit deductions, and top-ups.

The Firebase `db` client is accessed at call-time via the integration module
so it picks up the instance initialized during the FastAPI lifespan.

All Firestore operations are async (native AsyncClient) — no thread pool needed.
"""

import hmac
import logging
import os
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.cloud.firestore_v1.async_transaction import async_transactional

from app.config import settings
from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)

RECHARGE_SECRET_KEY = os.getenv("RECHARGE_SECRET_KEY", "")


def _get_db():
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    return db


async def get_guest_wallet(device_id: str) -> dict:
    """Retrieves or creates a guest wallet for the device."""
    db = _get_db()
    doc_ref = db.collection("guest_wallets").document(device_id)
    doc = await doc_ref.get()

    if doc.exists:
        return doc.to_dict()

    new_wallet = {
        "credits": settings.welcome_credits,
        "last_active": SERVER_TIMESTAMP,
        "is_banned": False,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
    }
    await doc_ref.set(new_wallet)
    return new_wallet


async def check_ban_status(device_id: str) -> bool:
    """Checks if the device is banned."""
    wallet = await get_guest_wallet(device_id)
    return wallet.get("is_banned", False)


async def deduct_guest_credits(device_id: str, cost: int = 5) -> int:
    """
    Deducts credits from the guest wallet atomically.
    Raises HTTPException(402) if insufficient funds.
    Returns the new balance.
    """
    db = _get_db()
    doc_ref = db.collection("guest_wallets").document(device_id)

    @async_transactional
    async def update_in_transaction(transaction, ref):
        snapshot = await ref.get(transaction=transaction)
        if not snapshot.exists:
            transaction.set(ref, {
                "credits": settings.welcome_credits,
                "last_active": SERVER_TIMESTAMP,
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
            "last_active": SERVER_TIMESTAMP,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
        })
        return current_credits - cost

    transaction = db.transaction()
    try:
        new_balance = await update_in_transaction(transaction, doc_ref)
        logger.info("guest_credits_deducted", extra={
            "action": "guest_credits_deducted",
            "amount": cost,
            "new_balance": new_balance,
        })
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error("guest_credits_deduct_failed", extra={
            "action": "guest_credits_deduct_failed",
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail="Wallet transaction failed")


async def perform_recharge(device_id: str, amount: int, secret_key: str) -> dict:
    """
    Top up a guest wallet.
    Validates `secret_key` against RECHARGE_SECRET_KEY env var.
    """
    if not RECHARGE_SECRET_KEY or not hmac.compare_digest(secret_key, RECHARGE_SECRET_KEY):
        logger.warning("recharge_invalid_attempt", extra={"action": "recharge_invalid_attempt"})
        raise HTTPException(status_code=403, detail="Invalid secret key")

    db = _get_db()
    doc_ref = db.collection("guest_wallets").document(device_id)

    @async_transactional
    async def recharge_transaction(transaction, ref):
        snapshot = await ref.get(transaction=transaction)
        if not snapshot.exists:
            transaction.set(ref, {
                "credits": settings.welcome_credits + amount,
                "last_active": SERVER_TIMESTAMP,
                "is_banned": False,
                "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
            })
            return settings.welcome_credits + amount
        else:
            current = snapshot.get("credits") or 0
            transaction.update(ref, {
                "credits": current + amount,
                "last_active": SERVER_TIMESTAMP,
                "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
            })
            return current + amount

    transaction = db.transaction()
    try:
        new_balance = await recharge_transaction(transaction, doc_ref)
        logger.info("guest_credits_recharged", extra={
            "action": "guest_credits_recharged",
            "amount": amount,
            "new_balance": new_balance,
        })
        return {"status": "success", "new_balance": new_balance}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("recharge_failed", extra={"action": "recharge_failed", "error": str(e)})
        raise HTTPException(status_code=500, detail="Recharge failed")
