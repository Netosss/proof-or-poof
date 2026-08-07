"""
Voided-purchase reconciliation (refunds, chargebacks, developer revokes).

Closes the buy -> spend -> refund -> keep-the-credits loop. Nothing here is
triggered by the user: it reads Google's voided-purchases list and reverses the
credits we granted for orders that no longer stand.

Depends on `processed_play_orders/{order_id}`, written by the billing route when
it grants. That doc carries `owner_id`, `credits` and `is_guest` precisely so a
reversal never has to re-derive them — `owner_id` alone is ambiguous, since the
same string shape could be a Firebase uid or a device id, and guessing wrong
debits a stranger.

Google retains voided purchases for **30 days only**, so this must run more
often than that or refunds are lost with no trace. A daily run is the intent.
"""

import logging
from datetime import datetime, timedelta, timezone

from google.cloud.firestore_v1 import SERVER_TIMESTAMP

from app.integrations import firebase as firebase_module
from app.services.credit_engine import grant_credits
from app.services.credits_service import grant_guest_credits
from app.services.google_play_billing import (
    GooglePlayVerificationError,
    is_configured,
    list_voided_purchases,
)

logger = logging.getLogger(__name__)

ORDERS_COLLECTION = "processed_play_orders"

# Google keeps 30 days. Default well inside that so a couple of missed runs are
# still covered, while staying cheap enough to run daily.
DEFAULT_LOOKBACK_DAYS = 14


async def reconcile_voided_purchases(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> dict:
    """
    Reverse credits for every purchase Google reports as voided in the window.

    Idempotent: each order doc carries a `reversed` flag, so re-running reverses
    nothing twice. Returns a summary suitable for logging or an admin response.
    """
    if not is_configured():
        logger.error(
            "refund_reconcile_not_configured",
            extra={"action": "refund_reconcile_not_configured"},
        )
        return {"status": "skipped", "reason": "GOOGLE_PLAY_SERVICE_ACCOUNT_JSON not set"}

    db = firebase_module.db
    if not db:
        return {"status": "skipped", "reason": "firestore unavailable"}

    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    start_time_ms = int(start.timestamp() * 1000)

    try:
        voided = await list_voided_purchases(start_time_ms)
    except GooglePlayVerificationError as e:
        logger.error(
            "refund_reconcile_list_failed",
            extra={"action": "refund_reconcile_list_failed", "error": str(e)},
        )
        return {"status": "error", "reason": "voided-purchases list failed"}

    reversed_count = 0
    already = 0
    unknown = 0
    failed = 0

    for entry in voided:
        order_id = entry.get("orderId")
        if not order_id:
            continue

        ref = db.collection(ORDERS_COLLECTION).document(order_id)
        snapshot = await ref.get()
        if not snapshot.exists:
            # Voided, but we never granted for it — a purchase that failed
            # verification, or predates durable claims. Nothing to reverse.
            unknown += 1
            logger.info(
                "refund_reconcile_unknown_order",
                extra={"action": "refund_reconcile_unknown_order", "order_id": order_id},
            )
            continue

        data = snapshot.to_dict() or {}
        if data.get("reversed"):
            already += 1
            continue

        owner_id = data.get("owner_id")
        credits = data.get("credits")
        if not owner_id or not isinstance(credits, int):
            failed += 1
            logger.error(
                "refund_reconcile_bad_order_doc",
                extra={"action": "refund_reconcile_bad_order_doc", "order_id": order_id},
            )
            continue

        # Claim BEFORE debiting, then undo the claim if the debit fails. The
        # other order — debit then mark — double-debits whenever the mark fails,
        # and this job retries on a schedule, so that failure mode compounds.
        await ref.update({
            "reversed": True,
            "reversed_at": SERVER_TIMESTAMP,
            "voided_reason": entry.get("voidedReason"),
        })

        try:
            if data.get("is_guest"):
                await grant_guest_credits(owner_id, -credits)
            else:
                await grant_credits(owner_id, -credits, "google_play_refund", order_id)
        except Exception as e:
            failed += 1
            logger.error(
                "refund_reconcile_debit_failed",
                extra={
                    "action": "refund_reconcile_debit_failed",
                    "order_id": order_id,
                    "error": str(e),
                },
            )
            try:
                await ref.update({"reversed": False, "reversed_at": None})
            except Exception:
                logger.error(
                    "refund_reconcile_unclaim_failed",
                    extra={
                        "action": "refund_reconcile_unclaim_failed",
                        "order_id": order_id,
                    },
                )
            continue

        reversed_count += 1
        logger.warning(
            "refund_reconcile_reversed",
            extra={
                "action": "refund_reconcile_reversed",
                "order_id": order_id,
                "credits": credits,
                "voided_reason": entry.get("voidedReason"),
            },
        )

    summary = {
        "status": "ok",
        "voided_seen": len(voided),
        "reversed": reversed_count,
        "already_reversed": already,
        "unknown_orders": unknown,
        "failed": failed,
    }
    logger.info("refund_reconcile_done", extra={"action": "refund_reconcile_done", **summary})
    return summary
