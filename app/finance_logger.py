import datetime
import logging
from app.firebase_config import db

logger = logging.getLogger(__name__)

def log_transaction(category: str, cost: float, meta: dict = {}):
    """
    Logs a financial event to Firestore.
    
    Args:
        category: "GPU", "GEMINI", "AD_REWARD", "LEMONSQUEEZY"
        cost: The dollar value (positive for income, negative for expense)
        meta: Additional metadata (request_id, user_id, etc.)
    """
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
        logger.info(f"💰 [FINANCE] {category}: ${cost:.4f} ({transaction_type})")
    except Exception as e:
        logger.error(f"❌ [FINANCE LOG ERROR] Failed to log transaction: {e}")
