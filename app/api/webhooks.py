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

from app.integrations.runpod import pending_jobs, webhook_result_buffer
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
        return {"status": "ignored", "reason": "No secret set"}

    payload_bytes = await request.body()

    expected_signature = hmac.new(
        LEMONSQUEEZY_WEBHOOK_SECRET.encode(),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

    if not x_signature or not hmac.compare_digest(x_signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(payload_bytes)
    event_name = payload.get("meta", {}).get("event_name")

    if event_name == "order_created":
        data = payload.get("data", {})
        attributes = data.get("attributes", {})
        total_cents = attributes.get("total", 0)
        total_usd = total_cents / 100.0
        custom_data = payload.get("meta", {}).get("custom_data", {})
        user_id = custom_data.get("user_id", "unknown")
        log_transaction("LEMONSQUEEZY", total_usd, {"user_id": user_id, "order_id": data.get("id")})
        logger.info(f"[LEMONSQUEEZY] Processed order {data.get('id')}: ${total_usd}")

    return {"status": "ok"}
