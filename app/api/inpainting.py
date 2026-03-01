"""
Inpainting route: /inpaint/image

Removes objects from an image using the RunPod GPU worker (LaMa).
Verifies Turnstile, checks credits, and only deducts on success.
Grants a free retry token for 10 minutes so users can re-download without being charged.
"""

import hashlib
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.core.auth import check_ip_device_limit, get_client_ip, validate_device_id, verify_turnstile
from app.integrations import redis_client as redis_module
from app.integrations.runpod import run_gpu_inpainting
from app.services.credits_service import deduct_guest_credits, get_guest_wallet
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inpainting"])


@router.post("/inpaint/image")
async def inpaint_image(
    request: Request,
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token")
):
    """
    Remove objects from an image using the RunPod GPU worker (LaMa).
    Verifies Turnstile, checks credits, and only deducts on success.
    Grants a free retry token for 10 minutes so the user can re-download without being charged.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[INPAINT] Request {request_id} started. Image: {image.filename}, Device: {device_id}")

    validate_device_id(device_id)
    ip = get_client_ip(request)

    if not turnstile_token:
        raise HTTPException(
            status_code=403,
            detail={"code": "CAPTCHA_REQUIRED", "message": "Verification needed"}
        )
    is_human = await verify_turnstile(turnstile_token)
    if not is_human:
        raise HTTPException(status_code=403, detail="Invalid CAPTCHA")

    # token_already_verified=True prevents check_ip_device_limit from calling
    # verify_turnstile a second time when the IP hits its device limit.
    await check_ip_device_limit(ip, device_id, token_already_verified=True)

    # Single Firestore read — covers both ban check and credit check below.
    wallet = get_guest_wallet(device_id)
    if wallet.get("is_banned"):
        raise HTTPException(status_code=403, detail="Device is banned")

    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        image_size_mb = len(image_bytes) / (1024 * 1024)
        logger.info(f"[INPAINT] Request {request_id}: Files read. Image size: {image_size_mb:.2f}MB")
    except Exception as e:
        logger.error(f"[INPAINT] File read error: {e}")
        raise HTTPException(status_code=400, detail="Error reading upload files")

    img_hash = hashlib.sha256(image_bytes).hexdigest()
    cache_key = f"paid_image:{device_id}:{img_hash}"
    rc = redis_module.client

    is_free_retry = bool(rc and rc.get(cache_key))
    if is_free_retry:
        logger.info(f"[INPAINT] Free retry available for {device_id} on image {img_hash[:8]}")

    # Reuse the wallet fetched at the top — avoids a second Firestore round-trip.
    current_credits = wallet.get("credits", 0)

    if not is_free_retry and current_credits < settings.inpaint_credit_cost:
        logger.info(
            f"[BILLING] Insufficient credits for {device_id} "
            f"(Has: {current_credits}, Need: {settings.inpaint_credit_cost})"
        )
        raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        start_time = time.time()
        logger.info(f"[INPAINT] Request {request_id} sending to GPU worker...")
        result_bytes = await run_gpu_inpainting(image_bytes, mask_bytes)
        duration = time.time() - start_time
        logger.info(f"[INPAINT] GPU Request {request_id} COMPLETED in {duration:.3f}s")

        usd_cost = duration * settings.inpaint_rate_per_sec
        log_transaction("INPAINT", -usd_cost, {"device_id": device_id, "duration": duration, "request_id": request_id})

        if is_free_retry:
            if rc:
                rc.delete(cache_key)
            new_balance = current_credits
        else:
            new_balance = deduct_guest_credits(device_id, cost=settings.inpaint_credit_cost)
            if rc:
                rc.set(cache_key, "1", ex=settings.deepfake_dedupe_ttl_sec)

        headers = {"X-User-Balance": str(new_balance)}
        return Response(content=result_bytes, media_type="image/png", headers=headers)

    except Exception as e:
        logger.error(f"[INPAINT] GPU Worker failed for {request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Inpainting service unavailable")
