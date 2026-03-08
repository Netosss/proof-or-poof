"""
Financial transaction logging to Firestore.

The Firebase `db` client is accessed at call-time via the integration module
so it picks up the instance initialized during the FastAPI lifespan.

`log_transaction` is fire-and-forget: when called from an async context it
spawns a background task so the Firestore write (~100 ms) does not block
the HTTP response. The AsyncClient is used directly — no thread pool needed.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.integrations import firebase as firebase_module

logger = logging.getLogger(__name__)

# Strong references to in-flight background tasks.
# asyncio's event loop holds tasks in a WeakSet in Python 3.12+; without a
# strong reference the GC can collect and silently cancel a task mid-flight.
_background_tasks: set = set()


async def _async_log(category: str, cost: float, meta: dict) -> None:
    db = firebase_module.db
    if not db:
        logger.warning("finance_skip_no_firebase", extra={
            "action": "finance_skip_no_firebase",
            "category": category,
        })
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
        await db.collection("financial_events").add(event)
        logger.info("finance_transaction", extra={
            "action": "finance_transaction",
            "category": category,
            "amount": float(cost),
            "transaction_type": transaction_type,
            **meta,
        })
    except Exception as e:
        logger.error("finance_log_error", extra={
            "action": "finance_log_error",
            "category": category,
            "error": str(e),
        })


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
        task = loop.create_task(_async_log(category, cost, meta))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError:
        # No running event loop — sync context (tests, CLI). Skip gracefully.
        logger.debug("finance_log_skipped_no_loop", extra={
            "action": "finance_log_skipped_no_loop",
            "category": category,
        })
