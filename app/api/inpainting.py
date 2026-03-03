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

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.core.auth import check_ip_device_limit, get_client_ip, validate_device_id, verify_turnstile
from app.core.firebase_auth import get_optional_user
from app.core.rate_limiter import check_rate_limit
from app.integrations import redis_client as redis_module
from app.integrations.runpod import run_gpu_inpainting
from app.services.credit_engine import consume_credits, get_user_balance
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
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token"),
    auth_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Remove objects from an image using the RunPod GPU worker (LaMa).

    Supports two billing paths:
      - Authenticated (Authorization: Bearer token present): deducts from
        the user's account via credit_engine. device_id / guest_wallets
        are ignored entirely.
      - Guest (no Authorization header): deducts from the guest wallet
        identified by X-Device-ID as before.
    """
    request_id = str(uuid.uuid4())
    ip = get_client_ip(request)

    if auth_user:
        # --- Authenticated path ---
        uid = auth_user["uid"]
        logger.info(f"[INPAINT] Request {request_id} started. Image: {image.filename}, UID: {uid}")

        # Rate-limit check fires BEFORE any Firestore transaction
        check_rate_limit(f"uid:{uid}")

        if not turnstile_token:
            raise HTTPException(
                status_code=403,
                detail={"code": "CAPTCHA_REQUIRED", "message": "Verification needed"}
            )
        is_human = await verify_turnstile(turnstile_token)
        if not is_human:
            raise HTTPException(status_code=403, detail="Invalid CAPTCHA")

    else:
        # --- Guest path ---
        validate_device_id(device_id)
        logger.info(f"[INPAINT] Request {request_id} started. Image: {image.filename}, Device: {device_id}")

        if not turnstile_token:
            raise HTTPException(
                status_code=403,
                detail={"code": "CAPTCHA_REQUIRED", "message": "Verification needed"}
            )
        is_human = await verify_turnstile(turnstile_token)
        if not is_human:
            raise HTTPException(status_code=403, detail="Invalid CAPTCHA")

        await check_ip_device_limit(ip, device_id, token_already_verified=True)

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
    rc = redis_module.client

    if auth_user:
        uid = auth_user["uid"]
        cache_key = f"paid_image:uid:{uid}:{img_hash}"
        is_free_retry = bool(rc and rc.get(cache_key))
        if is_free_retry:
            logger.info(f"[INPAINT] Free retry available for uid={uid} on image {img_hash[:8]}")

        if not is_free_retry:
            # Pre-check balance BEFORE starting the GPU job — prevents burning
            # RunPod compute for users who can't pay.
            current_balance = get_user_balance(uid)
            if current_balance < settings.inpaint_credit_cost:
                logger.info(
                    f"[BILLING] Insufficient credits for uid={uid} "
                    f"(Has: {current_balance}, Need: {settings.inpaint_credit_cost})"
                )
                raise HTTPException(status_code=402, detail="Insufficient credits")

        try:
            start_time = time.time()
            logger.info(f"[INPAINT] Request {request_id} sending to GPU worker...")
            result_bytes = await run_gpu_inpainting(image_bytes, mask_bytes)
            duration = time.time() - start_time
            logger.info(f"[INPAINT] GPU Request {request_id} COMPLETED in {duration:.3f}s")

            usd_cost = duration * settings.inpaint_rate_per_sec
            log_transaction("INPAINT", -usd_cost, {"uid": uid, "duration": duration, "request_id": request_id})

            if is_free_retry:
                if rc:
                    rc.delete(cache_key)
                new_balance = get_user_balance(uid)
            else:
                new_balance = consume_credits(uid, settings.inpaint_credit_cost, "inpaint", request_id)
                if rc:
                    rc.set(cache_key, "1", ex=settings.deepfake_dedupe_ttl_sec)

            headers = {"X-User-Balance": str(new_balance)}
            return Response(content=result_bytes, media_type="image/png", headers=headers)

        except Exception as e:
            logger.error(f"[INPAINT] GPU Worker failed for {request_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Inpainting service unavailable")

    else:
        # --- Guest billing ---
        cache_key = f"paid_image:{device_id}:{img_hash}"
        is_free_retry = bool(rc and rc.get(cache_key))
        if is_free_retry:
            logger.info(f"[INPAINT] Free retry available for {device_id} on image {img_hash[:8]}")

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
