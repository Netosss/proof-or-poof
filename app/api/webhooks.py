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
                logger.warning("webhook_runpod_secret_mismatch", extra={
                    "action": "webhook_runpod_secret_mismatch",
                })
                raise HTTPException(status_code=401, detail="Invalid webhook signature")

        payload = await request.json()
        job_id = payload.get("id")
        status = payload.get("status")
        output = payload.get("output")

        logger.info("webhook_runpod_received", extra={
            "action": "webhook_runpod_received",
            "job_id": job_id,
            "status": status,
        })

        if job_id and job_id in pending_jobs:
            future, start_time = pending_jobs[job_id]
            elapsed_ms = (time.time() - start_time) * 1000

            if status == "COMPLETED" and output:
                if not isinstance(output, dict):
                    logger.warning("webhook_runpod_invalid_output", extra={
                        "action": "webhook_runpod_invalid_output",
                        "job_id": job_id,
                        "output_type": type(output).__name__,
                    })
                    output = {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": "Invalid output format"}

                if not future.done():
                    future.set_result(output)

                if "results" in output:
                    logger.info("webhook_runpod_completed", extra={
                        "action": "webhook_runpod_completed",
                        "job_id": job_id,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "batch_size": len(output.get("results", [])),
                        "gpu_time_ms": output.get("timing_ms", {}).get("total", 0),
                    })
                else:
                    logger.info("webhook_runpod_completed", extra={
                        "action": "webhook_runpod_completed",
                        "job_id": job_id,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "ai_score": output.get("ai_score"),
                    })

            elif status == "FAILED":
                error_output = {
                    "error": "Job failed",
                    "details": payload.get("error"),
                    "ai_score": 0.0,
                    "gpu_time_ms": 0.0
                }
                if not future.done():
                    future.set_result(error_output)
                logger.error("webhook_runpod_failed", extra={
                    "action": "webhook_runpod_failed",
                    "job_id": job_id,
                    "error": payload.get("error"),
                })
            else:
                logger.info("webhook_runpod_status", extra={
                    "action": "webhook_runpod_status",
                    "job_id": job_id,
                    "status": status,
                })
                return {"status": "acknowledged"}

        elif job_id and status == "COMPLETED" and output:
            if isinstance(output, dict) and "ai_score" in output:
                webhook_result_buffer[job_id] = (output, time.time())
                logger.info("webhook_runpod_buffered", extra={
                    "action": "webhook_runpod_buffered",
                    "job_id": job_id,
                    "ai_score": output.get("ai_score"),
                })
            else:
                logger.warning("webhook_runpod_invalid_output", extra={
                    "action": "webhook_runpod_invalid_output",
                    "job_id": job_id,
                    "reason": "invalid output, not buffering",
                })
        else:
            logger.warning("webhook_runpod_unknown_job", extra={
                "action": "webhook_runpod_unknown_job",
                "job_id": job_id,
            })

        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("webhook_runpod_callback_error", extra={
            "action": "webhook_runpod_callback_error",
            "error": str(e),
        })
        return {"status": "error", "message": str(e)}


@router.post("/webhooks/lemonsqueezy")
async def lemonsqueezy_webhook(
    request: Request,
    x_signature: str = Header(None, alias="X-Signature")
):
    """Verifies and processes Lemon Squeezy order webhooks for payment tracking."""
    if not LEMONSQUEEZY_WEBHOOK_SECRET:
        logger.critical("lemonsqueezy_config_critical", extra={
            "action": "lemonsqueezy_config_critical",
            "detail": "LEMONSQUEEZY_WEBHOOK_SECRET is not set — all webhooks rejected",
        })
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
        order_status = attributes.get("status", "")
        if order_status != "paid":
            logger.info("lemonsqueezy_order_created_skipped", extra={
                "action": "lemonsqueezy_order_created_skipped",
                "order_id": order_id,
                "order_status": order_status,
            })
            return {"status": "ok", "skipped": f"status={order_status}"}
        return await _handle_order_paid(payload, meta, data, attributes, order_id, custom_data)

    if event_name == "order_paid":
        return await _handle_order_paid(payload, meta, data, attributes, order_id, custom_data)

    if event_name == "order_refunded":
        return await _handle_order_refunded(order_id)

    logger.info("lemonsqueezy_unhandled_event", extra={
        "action": "lemonsqueezy_unhandled_event",
        "event_name": event_name,
    })
    return {"status": "ok"}


async def _handle_order_paid(payload, meta, data, attributes, order_id, custom_data):
    """Grants credits exactly once when a Lemon Squeezy payment is confirmed."""

    payload_env = custom_data.get("env", "")
    is_test_payload = meta.get("test_mode") is True

    if settings.is_dev:
        if not is_test_payload or payload_env != "dev":
            logger.info("lemonsqueezy_payload_skipped", extra={
                "action": "lemonsqueezy_payload_skipped",
                "order_id": order_id,
                "reason": "wrong_env_for_dev_server",
                "test_mode": is_test_payload,
                "payload_env": payload_env,
            })
            return {"status": "ok", "skipped": "wrong_env_for_dev_server"}
    else:
        if is_test_payload or payload_env != "prod":
            logger.info("lemonsqueezy_payload_skipped", extra={
                "action": "lemonsqueezy_payload_skipped",
                "order_id": order_id,
                "reason": "wrong_env_for_prod_server",
                "test_mode": is_test_payload,
                "payload_env": payload_env,
            })
            return {"status": "ok", "skipped": "wrong_env_for_prod_server"}

    user_id = custom_data.get("user_id")
    if not user_id:
        logger.error("lemonsqueezy_order_paid_missing_user", extra={
            "action": "lemonsqueezy_order_paid_missing_user",
            "order_id": order_id,
        })
        return {"status": "error", "reason": "missing user_id"}

    first_item = (attributes.get("first_order_item") or {})
    variant_id = str(first_item.get("variant_id", ""))

    credits = settings.active_ls_variants.get(variant_id)
    if not credits:
        logger.error("lemonsqueezy_variant_not_found", extra={
            "action": "lemonsqueezy_variant_not_found",
            "order_id": order_id,
            "variant_id": variant_id,
            "app_env": settings.app_env,
            "known_variants": list(settings.active_ls_variants.keys()),
        })
        raise HTTPException(
            status_code=500,
            detail=f"Unknown variant {variant_id} — update LEMON_SQUEEZY_VARIANTS in config"
        )

    db = firebase_module.db
    if not db:
        logger.error("lemonsqueezy_firestore_unavailable", extra={
            "action": "lemonsqueezy_firestore_unavailable",
            "order_id": order_id,
        })
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
            "status": "pending",
            "test_mode": False,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        if "AlreadyExists" in type(e).__name__ or "ALREADY_EXISTS" in str(e).upper():
            existing = purchase_ref.get()
            existing_status = existing.to_dict().get("status") if existing.exists else None

            if existing_status == "paid":
                logger.info("lemonsqueezy_duplicate_order", extra={
                    "action": "lemonsqueezy_duplicate_order",
                    "order_id": order_id,
                    "user_id": user_id,
                })
                return {"status": "ok", "skipped": "duplicate"}

            logger.warning("lemonsqueezy_retry_grant", extra={
                "action": "lemonsqueezy_retry_grant",
                "order_id": order_id,
                "user_id": user_id,
                "prior_status": existing_status,
            })
        else:
            logger.error("lemonsqueezy_purchase_failed", extra={
                "action": "lemonsqueezy_purchase_failed",
                "order_id": order_id,
                "user_id": user_id,
                "error": str(e),
            })
            raise HTTPException(status_code=500, detail="Purchase record failed")

    try:
        new_balance = grant_credits(user_id, credits, "purchase", order_id)
    except Exception as e:
        purchase_ref.update({"status": "grant_failed"})
        logger.error("lemonsqueezy_grant_failed", extra={
            "action": "lemonsqueezy_grant_failed",
            "order_id": order_id,
            "user_id": user_id,
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail="Credit grant failed — will retry")

    purchase_ref.update({"status": "paid"})

    log_transaction("LEMONSQUEEZY", total_usd, {
        "user_id": user_id,
        "order_id": order_id,
        "credits": credits,
    })
    logger.info("lemonsqueezy_order_paid", extra={
        "action": "lemonsqueezy_order_paid",
        "order_id": order_id,
        "user_id": user_id,
        "credits_granted": credits,
        "total_usd": total_usd,
        "new_balance": new_balance,
    })
    return {"status": "ok"}


async def _handle_order_refunded(order_id: str):
    """Deducts credits when a Lemon Squeezy order is refunded."""
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")

    purchase_ref = db.collection("purchases").document(order_id)

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
        logger.error("lemonsqueezy_refund_failed", extra={
            "action": "lemonsqueezy_refund_failed",
            "order_id": order_id,
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail="Refund processing failed")

    if outcome == "not_found":
        logger.warning("lemonsqueezy_refund_not_found", extra={
            "action": "lemonsqueezy_refund_not_found",
            "order_id": order_id,
        })
        return {"status": "ok", "skipped": "purchase_not_found"}

    if outcome == "already_refunded":
        logger.info("lemonsqueezy_duplicate_refund", extra={
            "action": "lemonsqueezy_duplicate_refund",
            "order_id": order_id,
        })
        return {"status": "ok", "skipped": "already_refunded"}

    try:
        new_balance = grant_credits(user_id, -credits_granted, "refund", order_id)
        logger.info("lemonsqueezy_order_refunded", extra={
            "action": "lemonsqueezy_order_refunded",
            "order_id": order_id,
            "user_id": user_id,
            "credits_deducted": credits_granted,
            "new_balance": new_balance,
        })
    except Exception as e:
        logger.error("lemonsqueezy_refund_deduction_failed", extra={
            "action": "lemonsqueezy_refund_deduction_failed",
            "order_id": order_id,
            "user_id": user_id,
            "credits_granted": credits_granted,
            "error": str(e),
            "note": "Purchase marked refunded but credits NOT deducted. Manual fix needed.",
        })
        raise HTTPException(status_code=500, detail="Refund credit deduction failed")

    return {"status": "ok"}
