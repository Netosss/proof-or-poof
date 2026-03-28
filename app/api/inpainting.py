"""
Inpainting route: /inpaint/image

Removes objects from an image using a Modal GPU worker (LaMa).
Verifies Turnstile, checks credits, and only deducts on success.

After a paid removal the response includes an `X-Op-Ref` header containing a
UUID token.  Sending that token back as `X-Op-Ref` on the *next* request grants
one free refinement (works on any image — the result, an undo, a re-crop, etc.).
The token is single-use (Redis GETDEL) and expires after 10 minutes.
"""

import io
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.config import settings
from app.core.auth import check_ip_device_limit, get_client_ip, validate_device_id, verify_turnstile
from app.core.dependencies import security_manager
from app.core.firebase_auth import get_optional_user
from app.integrations import redis_client as redis_module
from app.integrations.gpu_worker import run_gpu_inpainting
from app.logging_config import user_id_var
from app.services.credit_engine import consume_credits, get_user_balance
from app.services.credits_service import deduct_guest_credits, get_guest_wallet
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inpainting"])

# Formats the Modal / LaMa GPU worker accepts directly.  Any other format
# PIL can open is converted to JPEG before dispatch.  Defined at module level
# to avoid recreating the set literal on every request.
_INPAINT_PASSTHROUGH_FORMATS: frozenset[str] = frozenset({"jpeg", "png"})


@router.post("/inpaint/image")
async def inpaint_image(
    request: Request,
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token"),
    op_ref: Optional[str] = Header(None, alias="X-Op-Ref"),
    auth_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Remove objects from an image using the Modal GPU worker (LaMa).

    Supports two billing paths:
      - Authenticated (Authorization: Bearer token present): deducts from
        the user's account via credit_engine. device_id / guest_wallets
        are ignored entirely.
      - Guest (no Authorization header): deducts from the guest wallet
        identified by X-Device-ID as before.
    """
    request_id = str(uuid.uuid4())
    ip = get_client_ip(request)

    user_id_var.set(auth_user["uid"] if auth_user else "")

    if auth_user:
        uid = auth_user["uid"]
        logger.info("inpaint_request_started", extra={
            "action": "inpaint_request_started",
            "inpaint_request_id": request_id,
            "media_file": image.filename,
            "user_type": "authenticated",
        })

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
            "media_file": image.filename,
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

        wallet = await get_guest_wallet(device_id)
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

    # Validate MIME type and size before touching billing or the GPU.
    # file_path is omitted — magic-bytes PIL check is skipped here because
    # run_gpu_inpainting will reject corrupt images naturally, and writing
    # to disk solely to verify is unnecessary for inpainting.
    await security_manager.validate_file(
        image.filename or "uploaded_image.jpg",
        len(image_bytes),
        content_type=image.content_type or None,
        mode="inpaint",
    )

    # Normalize unusual image formats to JPEG before sending to the GPU worker.
    # The LaMa GPU worker expects standard JPEG or PNG.  Formats like MPO, HEIC, AVIF,
    # TIFF, PSD, TGA, ICO, DDS etc. must be converted first.
    # JPEG and PNG are passed through unchanged to avoid an unnecessary re-encode.
    try:
        buf = io.BytesIO(image_bytes)
        with Image.open(buf) as probe_img:
            detected_fmt = (probe_img.format or "unknown").lower()
            needs_conversion = detected_fmt not in _INPAINT_PASSTHROUGH_FORMATS

        if needs_conversion:
            buf.seek(0)
            original_size = len(image_bytes)
            with Image.open(buf) as src_img:
                out = io.BytesIO()
                src_img.convert("RGB").save(out, format="JPEG", quality=95)
            image_bytes = out.getvalue()
            image_size_mb = round(len(image_bytes) / (1024 * 1024), 2)
            logger.info("inpaint_image_normalized", extra={
                "action": "inpaint_image_normalized",
                "inpaint_request_id": request_id,
                "original_format": detected_fmt,
                "original_size_bytes": original_size,
                "converted_size_bytes": len(image_bytes),
            })
    except HTTPException:
        raise
    except Exception as conv_err:
        logger.warning("inpaint_image_normalization_failed", extra={
            "action": "inpaint_image_normalization_failed",
            "inpaint_request_id": request_id,
            "media_file": image.filename,
            "error": str(conv_err),
            "error_type": type(conv_err).__name__,
        })
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process the uploaded image file "
                f"({image.filename or 'unknown'}). "
                "Please convert it to JPEG or PNG and try again."
            ),
        )

    rc = redis_module.client

    if auth_user:
        uid = auth_user["uid"]
        is_free_retry = False
        if op_ref and rc:
            is_free_retry = bool(await rc.getdel(f"op_ref:{uid}:{op_ref}"))
            if is_free_retry:
                logger.info("inpaint_free_retry_used", extra={
                    "action": "inpaint_free_retry_used",
                    "inpaint_request_id": request_id,
                    "user_type": "authenticated",
                })

        if not is_free_retry:
            current_balance = await get_user_balance(uid)
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
                "credits_consumed": 0 if is_free_retry else settings.inpaint_credit_cost,
                "user_type": "authenticated",
                "is_free_retry": is_free_retry,
                "image_size_mb": image_size_mb,
            })

            log_transaction("INPAINT", -usd_cost, {"uid": uid, "duration": duration, "request_id": request_id})

            if is_free_retry:
                new_balance = await get_user_balance(uid)
                headers = {"X-User-Balance": str(new_balance)}
            else:
                new_balance = await consume_credits(uid, settings.inpaint_credit_cost, "inpaint", request_id)
                new_op_ref = str(uuid.uuid4())
                if rc:
                    await rc.set(f"op_ref:{uid}:{new_op_ref}", "1", ex=600)
                headers = {"X-User-Balance": str(new_balance), "X-Op-Ref": new_op_ref}
            return Response(content=result_bytes, media_type="image/png", headers=headers)

        except Exception as e:
            if is_free_retry and rc:
                await rc.set(f"op_ref:{uid}:{op_ref}", "1", ex=600)
            logger.error("inpaint_gpu_failed", extra={
                "action": "inpaint_gpu_failed",
                "inpaint_request_id": request_id,
                "error": str(e),
                "user_type": "authenticated",
            }, exc_info=True)
            raise HTTPException(status_code=500, detail="Inpainting service unavailable")

    else:
        is_free_retry = False
        if op_ref and rc:
            is_free_retry = bool(await rc.getdel(f"op_ref:{device_id}:{op_ref}"))
            if is_free_retry:
                logger.info("inpaint_free_retry_used", extra={
                    "action": "inpaint_free_retry_used",
                    "inpaint_request_id": request_id,
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
                "credits_consumed": 0 if is_free_retry else settings.inpaint_credit_cost,
                "user_type": "guest",
                "is_free_retry": is_free_retry,
                "image_size_mb": image_size_mb,
            })

            log_transaction("INPAINT", -usd_cost, {"device_id": device_id, "duration": duration, "request_id": request_id})

            if is_free_retry:
                new_balance = current_credits
                headers = {"X-User-Balance": str(new_balance)}
            else:
                new_balance = await deduct_guest_credits(device_id, cost=settings.inpaint_credit_cost)
                new_op_ref = str(uuid.uuid4())
                if rc:
                    await rc.set(f"op_ref:{device_id}:{new_op_ref}", "1", ex=600)
                headers = {"X-User-Balance": str(new_balance), "X-Op-Ref": new_op_ref}
            return Response(content=result_bytes, media_type="image/png", headers=headers)

        except Exception as e:
            if is_free_retry and rc:
                await rc.set(f"op_ref:{device_id}:{op_ref}", "1", ex=600)
            logger.error("inpaint_gpu_failed", extra={
                "action": "inpaint_gpu_failed",
                "inpaint_request_id": request_id,
                "error": str(e),
                "user_type": "guest",
            }, exc_info=True)
            raise HTTPException(status_code=500, detail="Inpainting service unavailable")
