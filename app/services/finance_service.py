"""
Financial transaction logging to Firestore.

The Firebase `db` client is accessed at call-time via the integration module
so it picks up the instance initialized during the FastAPI lifespan.
"""

import datetime
import logging

from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)


def log_transaction(category: str, cost: float, meta: dict = None) -> None:
    """
    Logs a financial event to Firestore.

    Args:
        category: One of "GPU", "GEMINI", "CACHE", "CPU", "AD_REWARD", "INPAINT", "LEMONSQUEEZY".
        cost: Dollar value (positive for income, negative for expense).
        meta: Additional metadata (request_id, user_id, file, etc.)
    """
    if meta is None:
        meta = {}
    db = firebase_module.db
    if not db:
        logger.warning("[FINANCE] Firebase not initialized; skipping transaction log.")
        return
    try:
        transaction_type = "INCOME" if cost >= 0 else "EXPENSE"
        event = {
            "timestamp": datetime.datetime.utcnow(),
            "type": transaction_type,
            "category": category,
            "amount": float(cost),
            "meta": meta
        }
        db.collection("financial_events").add(event)
        logger.info(f"[FINANCE] {category}: ${cost:.4f} ({transaction_type})")
    except Exception as e:
        logger.error(f"[FINANCE LOG ERROR] Failed to log transaction: {e}")
