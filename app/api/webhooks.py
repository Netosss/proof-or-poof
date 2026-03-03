"""
Webhook routes: RunPod job callbacks and Lemon Squeezy payment events.
"""

import hashlib
import hmac
import json
import logging
import os
import time

from fastapi import APIRouter, Header, HTTPException, Request
from firebase_admin import firestore

from app.config import settings
from app.integrations import firebase as firebase_module
from app.integrations.runpod import pending_jobs, webhook_result_buffer
from app.services.credit_engine import grant_credits
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Webhooks"])

RUNPOD_WEBHOOK_SECRET = os.getenv("RUNPOD_WEBHOOK_SECRET", "")
LEMONSQUEEZY_WEBHOOK_SECRET = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET")


@router.post("/webhook/runpod")
async def runpod_webhook(request: Request):
    """
    Receives webhook callbacks from RunPod when jobs complete.

    Auth: RunPod does not send custom headers, so the secret is embedded in
    RUNPOD_WEBHOOK_URL as a query parameter, e.g.:
        https://your-app.railway.app/webhook/runpod?secret=<RUNPOD_WEBHOOK_SECRET>
    If RUNPOD_WEBHOOK_SECRET is set, requests without a matching ?secret= are rejected.
    """
    try:
        if RUNPOD_WEBHOOK_SECRET:
            secret = request.query_params.get("secret", "")
            if not secret or not hmac.compare_digest(secret, RUNPOD_WEBHOOK_SECRET):
                logger.warning("[WEBHOOK] RunPod secret mismatch — rejecting request.")
                raise HTTPException(status_code=401, detail="Invalid webhook signature")

        payload = await request.json()
        job_id = payload.get("id")
        status = payload.get("status")
        output = payload.get("output")

        logger.info(f"[WEBHOOK] Received callback for job {job_id}, status: {status}")

        if job_id and job_id in pending_jobs:
            future, start_time = pending_jobs[job_id]
            elapsed_ms = (time.time() - start_time) * 1000

            if status == "COMPLETED" and output:
                if not isinstance(output, dict):
                    logger.warning(f"[WEBHOOK] Job {job_id} output is not a dict: {type(output)}")
                    output = {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": "Invalid output format"}

                if not future.done():
                    future.set_result(output)

                if "results" in output:
                    batch_size = len(output.get("results", []))
                    gpu_time = output.get("timing_ms", {}).get("total", 0)
                    logger.info(
                        f"[WEBHOOK] Job {job_id} completed in {elapsed_ms:.2f}ms "
                        f"(batch={batch_size}, gpu={gpu_time:.1f}ms)"
                    )
                else:
                    logger.info(
                        f"[WEBHOOK] Job {job_id} completed in {elapsed_ms:.2f}ms, "
                        f"ai_score={output.get('ai_score', 'N/A')}"
                    )
            elif status == "FAILED":
                error_output = {
                    "error": "Job failed",
                    "details": payload.get("error"),
                    "ai_score": 0.0,
                    "gpu_time_ms": 0.0
                }
                if not future.done():
                    future.set_result(error_output)
                logger.error(f"[WEBHOOK] Job {job_id} failed: {payload.get('error')}")
            else:
                logger.info(f"[WEBHOOK] Job {job_id} status update: {status}")
                return {"status": "acknowledged"}

        elif job_id and status == "COMPLETED" and output:
            # Race condition: webhook arrived before pending_jobs was set
            if isinstance(output, dict) and "ai_score" in output:
                webhook_result_buffer[job_id] = (output, time.time())
                logger.info(
                    f"[WEBHOOK] Job {job_id} buffered (arrived before pending_jobs set), "
                    f"ai_score={output.get('ai_score')}"
                )
            else:
                logger.warning(f"[WEBHOOK] Job {job_id} has invalid output, not buffering")
        else:
            logger.warning(f"[WEBHOOK] Unknown or incomplete job_id: {job_id}")

        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WEBHOOK] Error processing callback: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/webhooks/lemonsqueezy")
async def lemonsqueezy_webhook(
    request: Request,
    x_signature: str = Header(None, alias="X-Signature")
):
    """Verifies and processes Lemon Squeezy order webhooks for payment tracking."""
    if not LEMONSQUEEZY_WEBHOOK_SECRET:
        # A missing secret means every webhook is silently swallowed.
        # Fail loudly so a deployment misconfiguration is immediately visible.
        logger.critical(
            "[LEMONSQUEEZY] LEMONSQUEEZY_WEBHOOK_SECRET is not set — "
            "all webhooks rejected. Fix this immediately."
        )
        raise HTTPException(
            status_code=500,
            detail="Webhook secret not configured on server"
        )

    payload_bytes = await request.body()

    expected_signature = hmac.new(
        LEMONSQUEEZY_WEBHOOK_SECRET.encode(),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

    if not x_signature or not hmac.compare_digest(x_signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(payload_bytes)
    meta = payload.get("meta", {})
    event_name = meta.get("event_name")
    data = payload.get("data", {})
    attributes = data.get("attributes", {})
    order_id = str(data.get("id", ""))
    custom_data = meta.get("custom_data", {})

    if event_name == "order_created":
        # In Lemon Squeezy, order_created is the payment-confirmed event for
        # one-time purchases. There is no separate order_paid event.
        # We check attributes.status == "paid" to guard against edge cases
        # where the webhook fires for a pending/failed order.
        order_status = attributes.get("status", "")
        if order_status != "paid":
            logger.info(
                f"[LEMONSQUEEZY] order_created with status={order_status} — "
                f"not paid yet, skipping. order={order_id}"
            )
            return {"status": "ok", "skipped": f"status={order_status}"}
        return await _handle_order_paid(payload, meta, data, attributes, order_id, custom_data)

    if event_name == "order_paid":
        # Kept for forward-compatibility in case LS adds this event later.
        return await _handle_order_paid(payload, meta, data, attributes, order_id, custom_data)

    if event_name == "order_refunded":
        return await _handle_order_refunded(order_id)

    logger.info(f"[LEMONSQUEEZY] Unhandled event: {event_name}")
    return {"status": "ok"}


async def _handle_order_paid(payload, meta, data, attributes, order_id, custom_data):
    """Grants credits exactly once when a Lemon Squeezy payment is confirmed."""

    # 1. Test mode guard — must be first, before any DB work
    if meta.get("test_mode") is True:
        logger.info(f"[LEMONSQUEEZY] Skipping test-mode order_paid: order={order_id}")
        return {"status": "ok", "skipped": "test_mode"}

    env = custom_data.get("env", "")
    if env != "prod":
        logger.info(f"[LEMONSQUEEZY] Skipping non-prod order_paid (env={env}): order={order_id}")
        return {"status": "ok", "skipped": f"env={env}"}

    # 2. Validate required fields
    user_id = custom_data.get("user_id")
    if not user_id:
        logger.error(f"[LEMONSQUEEZY] order_paid missing user_id in custom_data: order={order_id}")
        return {"status": "error", "reason": "missing user_id"}

    # Look up credit amount from backend config — the backend is the single
    # source of truth. We never trust the webhook payload for credit amounts.
    first_item = (attributes.get("first_order_item") or {})
    variant_id = str(first_item.get("variant_id", ""))

    credits = settings.lemon_squeezy_variants.get(variant_id)
    if not credits:
        logger.error(
            f"[LEMONSQUEEZY] order_paid: variant_id={variant_id} not found in "
            f"LEMON_SQUEEZY_VARIANTS config. order={order_id}. "
            f"Known variants: {list(settings.lemon_squeezy_variants.keys())}"
        )
        # Return 500 so Lemon Squeezy retries the webhook automatically.
        # Once you update LEMON_SQUEEZY_VARIANTS in config, the retry succeeds.
        raise HTTPException(
            status_code=500,
            detail=f"Unknown variant {variant_id} — update LEMON_SQUEEZY_VARIANTS in config"
        )

    # 3. Idempotency guard — use lemon_order_id as document ID.
    #    Start with status="pending" so a failed grant (step 4) is detectable
    #    on Lemon Squeezy's retry: if the doc exists with status="pending" or
    #    "grant_failed", Lemon Squeezy retries → we re-attempt the grant.
    #    If status="paid", the webhook already succeeded → skip silently.
    db = firebase_module.db
    if not db:
        logger.error("[LEMONSQUEEZY] Firestore unavailable")
        raise HTTPException(status_code=503, detail="Database unavailable")

    purchase_ref = db.collection("purchases").document(order_id)
    variant_id = str(
        (attributes.get("first_order_item") or {}).get("variant_id", "")
    )
    total_usd = attributes.get("total", 0) / 100.0

    try:
        purchase_ref.create({
            "lemon_order_id": order_id,
            "lemon_variant_id": variant_id,
            "user_id": user_id,
            "credits_granted": credits,
            "status": "pending",   # not "paid" yet — grant hasn't run
            "test_mode": False,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        if "AlreadyExists" in type(e).__name__ or "ALREADY_EXISTS" in str(e).upper():
            # Doc already exists — check its status before deciding what to do.
            existing = purchase_ref.get()
            existing_status = existing.to_dict().get("status") if existing.exists else None

            if existing_status == "paid":
                # Already successfully processed — safe to skip.
                logger.info(f"[LEMONSQUEEZY] Duplicate order_paid ignored (already paid): order={order_id}")
                return {"status": "ok", "skipped": "duplicate"}

            # status is "pending" or "grant_failed" → a prior attempt started
            # but the grant never completed. Fall through to re-attempt the grant.
            logger.warning(
                f"[LEMONSQUEEZY] Retrying grant for order={order_id} "
                f"(prior status={existing_status})"
            )
        else:
            logger.error(f"[LEMONSQUEEZY] Failed to create purchase doc for order={order_id}: {e}")
            raise HTTPException(status_code=500, detail="Purchase record failed")

    # 4. Grant credits — if this fails, Lemon Squeezy will retry the webhook
    #    and we'll re-attempt the grant (doc exists with status != "paid").
    try:
        new_balance = grant_credits(user_id, credits, "purchase", order_id)
    except Exception as e:
        purchase_ref.update({"status": "grant_failed"})
        logger.error(f"[LEMONSQUEEZY] grant_credits failed for order={order_id}: {e}")
        # Return 500 so Lemon Squeezy retries this webhook.
        raise HTTPException(status_code=500, detail="Credit grant failed — will retry")

    # 5. Mark purchase as fully completed.
    purchase_ref.update({"status": "paid"})

    log_transaction("LEMONSQUEEZY", total_usd, {
        "user_id": user_id,
        "order_id": order_id,
        "credits": credits,
    })
    logger.info(
        f"[LEMONSQUEEZY] order_paid: granted {credits} credits to "
        f"uid={user_id} (order={order_id}). New balance: {new_balance}"
    )
    return {"status": "ok"}


async def _handle_order_refunded(order_id: str):
    """Deducts credits when a Lemon Squeezy order is refunded."""
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")

    purchase_ref = db.collection("purchases").document(order_id)

    # Idempotency check + status update are atomic in one transaction,
    # preventing two concurrent refund webhooks from double-deducting credits.
    user_id = None
    credits_granted = 0

    @firestore.transactional
    def _claim_refund(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            return None, None, "not_found"
        purchase = snap.to_dict()
        status = purchase.get("status")
        if status == "refunded":
            return None, None, "already_refunded"
        transaction.update(ref, {"status": "refunded"})
        return purchase.get("user_id"), purchase.get("credits_granted", 0), "ok"

    try:
        txn = db.transaction()
        user_id, credits_granted, outcome = _claim_refund(txn, purchase_ref)
    except Exception as e:
        logger.error(f"[LEMONSQUEEZY] Refund transaction failed for order={order_id}: {e}")
        raise HTTPException(status_code=500, detail="Refund processing failed")

    if outcome == "not_found":
        logger.warning(f"[LEMONSQUEEZY] order_refunded: purchase not found for order={order_id}")
        return {"status": "ok", "skipped": "purchase_not_found"}

    if outcome == "already_refunded":
        logger.info(f"[LEMONSQUEEZY] Duplicate order_refunded ignored: order={order_id}")
        return {"status": "ok", "skipped": "already_refunded"}

    # Deduct credits after atomically claiming the refund.
    try:
        new_balance = grant_credits(user_id, -credits_granted, "refund", order_id)
        logger.info(
            f"[LEMONSQUEEZY] order_refunded: deducted {credits_granted} credits from "
            f"uid={user_id} (order={order_id}). New balance: {new_balance}"
        )
    except Exception as e:
        # Status is already "refunded" in the DB — log for manual reconciliation.
        logger.error(
            f"[LEMONSQUEEZY] Refund credit deduction failed for order={order_id}. "
            f"Purchase marked refunded but credits NOT deducted. Manual fix needed. Error: {e}"
        )
        raise HTTPException(status_code=500, detail="Refund credit deduction failed")

    return {"status": "ok"}
