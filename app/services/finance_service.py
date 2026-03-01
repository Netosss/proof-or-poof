"""
Financial transaction logging to Firestore.

The Firebase `db` client is accessed at call-time via the integration module
so it picks up the instance initialized during the FastAPI lifespan.

`log_transaction` is fire-and-forget: when called from an async context it
spawns a background thread so the Firestore write (~100 ms) does not block
the HTTP response.  In sync contexts (tests, scripts) it runs inline.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)


def _sync_log(category: str, cost: float, meta: dict) -> None:
    db = firebase_module.db
    if not db:
        logger.warning("[FINANCE] Firebase not initialized; skipping transaction log.")
        return
    try:
        transaction_type = "INCOME" if cost >= 0 else "EXPENSE"
        event = {
            "timestamp": datetime.now(timezone.utc),
            "type": transaction_type,
            "category": category,
            "amount": float(cost),
            "meta": meta,
        }
        db.collection("financial_events").add(event)
        logger.info(f"[FINANCE] {category}: ${cost:.4f} ({transaction_type})")
    except Exception as e:
        logger.error(f"[FINANCE LOG ERROR] Failed to log transaction: {e}")


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
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(asyncio.to_thread(_sync_log, category, cost, meta))
    except RuntimeError:
        # No running event loop — sync context (tests, CLI). Run inline.
        _sync_log(category, cost, meta)
