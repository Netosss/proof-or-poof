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
from app.logging_config import user_id_var
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

    # Set user_id context var so all downstream logs carry it automatically
    user_id_var.set(auth_user["uid"] if auth_user else "")

    if auth_user:
        uid = auth_user["uid"]
        logger.info("inpaint_request_started", extra={
            "action": "inpaint_request_started",
            "inpaint_request_id": request_id,
            "filename": image.filename,
            "user_type": "authenticated",
        })

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
        validate_device_id(device_id)
        logger.info("inpaint_request_started", extra={
            "action": "inpaint_request_started",
            "inpaint_request_id": request_id,
            "filename": image.filename,
            "user_type": "guest",
        })

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
        image_size_mb = round(len(image_bytes) / (1024 * 1024), 2)
    except Exception as e:
        logger.error("inpaint_file_read_error", extra={
            "action": "inpaint_file_read_error",
            "inpaint_request_id": request_id,
            "error": str(e),
        })
        raise HTTPException(status_code=400, detail="Error reading upload files")

    img_hash = hashlib.sha256(image_bytes).hexdigest()
    rc = redis_module.client

    if auth_user:
        uid = auth_user["uid"]
        cache_key = f"paid_image:uid:{uid}:{img_hash}"
        is_free_retry = bool(rc and rc.get(cache_key))
        if is_free_retry:
            logger.info("inpaint_free_retry_used", extra={
                "action": "inpaint_free_retry_used",
                "inpaint_request_id": request_id,
                "img_hash_prefix": img_hash[:8],
                "user_type": "authenticated",
            })

        if not is_free_retry:
            current_balance = get_user_balance(uid)
            if current_balance < settings.inpaint_credit_cost:
                logger.warning("insufficient_credits", extra={
                    "action": "insufficient_credits",
                    "endpoint": "inpaint",
                    "has": current_balance,
                    "need": settings.inpaint_credit_cost,
                    "user_type": "authenticated",
                })
                raise HTTPException(status_code=402, detail="Insufficient credits")

        try:
            start_time = time.time()
            logger.info("inpaint_gpu_dispatched", extra={
                "action": "inpaint_gpu_dispatched",
                "inpaint_request_id": request_id,
                "user_type": "authenticated",
            })
            result_bytes = await run_gpu_inpainting(image_bytes, mask_bytes)
            duration = time.time() - start_time
            usd_cost = duration * settings.inpaint_rate_per_sec

            logger.info("inpaint_completed", extra={
                "action": "inpaint_completed",
                "inpaint_request_id": request_id,
                "duration_ms": round(duration * 1000, 1),
                "cost_usd": round(usd_cost, 6),
                "user_type": "authenticated",
                "is_free_retry": is_free_retry,
                "image_size_mb": image_size_mb,
            })

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
            logger.error("inpaint_gpu_failed", extra={
                "action": "inpaint_gpu_failed",
                "inpaint_request_id": request_id,
                "error": str(e),
                "user_type": "authenticated",
            }, exc_info=True)
            raise HTTPException(status_code=500, detail="Inpainting service unavailable")

    else:
        cache_key = f"paid_image:{device_id}:{img_hash}"
        is_free_retry = bool(rc and rc.get(cache_key))
        if is_free_retry:
            logger.info("inpaint_free_retry_used", extra={
                "action": "inpaint_free_retry_used",
                "inpaint_request_id": request_id,
                "img_hash_prefix": img_hash[:8],
                "user_type": "guest",
            })

        current_credits = wallet.get("credits", 0)

        if not is_free_retry and current_credits < settings.inpaint_credit_cost:
                logger.warning("insufficient_credits", extra={
                    "action": "insufficient_credits",
                    "endpoint": "inpaint",
                    "has": current_credits,
                    "need": settings.inpaint_credit_cost,
                    "user_type": "guest",
                })
                raise HTTPException(status_code=402, detail="Insufficient credits")

        try:
            start_time = time.time()
            logger.info("inpaint_gpu_dispatched", extra={
                "action": "inpaint_gpu_dispatched",
                "inpaint_request_id": request_id,
                "user_type": "guest",
            })
            result_bytes = await run_gpu_inpainting(image_bytes, mask_bytes)
            duration = time.time() - start_time
            usd_cost = duration * settings.inpaint_rate_per_sec

            logger.info("inpaint_completed", extra={
                "action": "inpaint_completed",
                "inpaint_request_id": request_id,
                "duration_ms": round(duration * 1000, 1),
                "cost_usd": round(usd_cost, 6),
                "user_type": "guest",
                "is_free_retry": is_free_retry,
                "image_size_mb": image_size_mb,
            })

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
            logger.error("inpaint_gpu_failed", extra={
                "action": "inpaint_gpu_failed",
                "inpaint_request_id": request_id,
                "error": str(e),
                "user_type": "guest",
            }, exc_info=True)
            raise HTTPException(status_code=500, detail="Inpainting service unavailable")
