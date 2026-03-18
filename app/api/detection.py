"""
Detection route: /detect

Accepts multipart/form-data with 'file' or 'url' field (plus optional 'trusted_metadata'),
or JSON payload { "url": "https://..." }.

Validates Turnstile, checks bans, and deducts guest credits only on success.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile

from app.config import settings
from app.core.auth import check_ip_device_limit, get_client_ip, validate_device_id, verify_turnstile
from app.core.dependencies import security_manager
from app.core.firebase_auth import get_optional_user
from app.detection.pipeline import detect_ai_media
from app.integrations import redis_client as redis_module
from app.logging_config import user_id_var
from app.schemas.detection import DetectionResponse
from app.services.credit_engine import consume_credits, get_user_balance
from app.services.credits_service import deduct_guest_credits, get_guest_wallet
from app.services.detection_service import _generate_short_id, download_media_to_disk, log_memory
from app.services.finance_service import log_transaction
from app.core.file_validator import ALLOWED_VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Detection"])

# Kept in sync with file_validator so the media_type field in scan_completed
# accurately reflects what was validated as a video.
VIDEO_SUFFIXES = ALLOWED_VIDEO_EXTENSIONS


_UPLOAD_CHUNK = 65_536  # 64 KB — matches the HTTPS streaming path


def _stream_upload_to_disk(upload_file: UploadFile, dest_path: str) -> None:
    """Copy a multipart upload to *dest_path* in chunks without loading it into RAM.

    Runs inside asyncio.to_thread to keep the event loop unblocked.

    Guards:
      - seek(0) resets the file pointer in case Starlette's SpooledTemporaryFile
        has been partially consumed before this call.
      - Per-chunk size accumulation aborts early if the upload exceeds the
        maximum allowed video size, preventing disk exhaustion before
        validate_file gets a chance to run its own size check.
    """
    upload_file.file.seek(0)
    max_bytes = settings.max_video_upload_bytes
    total = 0
    with open(dest_path, "wb") as fp:
        while True:
            chunk = upload_file.file.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"Upload too large (max {max_bytes // (1024 * 1024)} MB)"
                )
            fp.write(chunk)


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token"),
    auth_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Detect AI-generated content in images/videos.

    Supports two billing paths:
      - Authenticated (Authorization: Bearer token present): deducts from
        the user's account via credit_engine. device_id / guest_wallets
        are ignored entirely.
      - Guest (no Authorization header): deducts from the guest wallet
        identified by X-Device-ID as before.
    """
    ip = get_client_ip(request)

    user_id_var.set(auth_user["uid"] if auth_user else "")

    if auth_user:
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

    filename = "unknown"
    upload_content_type: str | None = None   # MIME from the multipart boundary
    sidecar_metadata = None

    # Create the temp file up front so all ingestion paths stream directly to
    # disk — no in-memory buffer for the file payload.  The try/finally below
    # covers every exit path (ingestion errors, validation failures, and normal
    # completion) so the file is always cleaned up.
    fd, temp_path = tempfile.mkstemp(suffix=".tmp")
    os.close(fd)

    try:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                payload = await request.json()
                url = payload.get("url")
                if not url:
                    raise HTTPException(status_code=400, detail="Missing 'url' in JSON body")
                filename = await download_media_to_disk(url, temp_path)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON body")

        elif "multipart/form-data" in content_type:
            form = await request.form()
            file_obj = form.get("file")
            url_obj = form.get("url")
            trusted_metadata_obj = form.get("trusted_metadata")

            if trusted_metadata_obj and isinstance(trusted_metadata_obj, str):
                try:
                    sidecar_metadata = json.loads(trusted_metadata_obj)
                    logger.info("sidecar_metadata_received", extra={
                        "action": "sidecar_metadata_received",
                        "keys": list(sidecar_metadata.keys()),
                    })
                except json.JSONDecodeError as e:
                    logger.warning("sidecar_metadata_invalid", extra={
                        "action": "sidecar_metadata_invalid",
                        "error": str(e),
                    })

            if file_obj:
                if not isinstance(file_obj, UploadFile):
                    if isinstance(file_obj, str):
                        raise HTTPException(status_code=400, detail="Invalid file upload format")
                filename = file_obj.filename or "uploaded_file"
                upload_content_type = file_obj.content_type or None
                try:
                    await asyncio.to_thread(_stream_upload_to_disk, file_obj, temp_path)
                except ValueError as exc:
                    raise HTTPException(status_code=413, detail=str(exc))
            elif url_obj:
                if isinstance(url_obj, str):
                    filename = await download_media_to_disk(url_obj, temp_path)
                else:
                    raise HTTPException(status_code=400, detail="Invalid url field")
            else:
                raise HTTPException(status_code=400, detail="Must provide 'file' or 'url' in form data")

        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type. Use multipart/form-data or application/json"
            )

        # Measure size from disk — never trust a client-supplied Content-Length.
        filesize = os.path.getsize(temp_path)
        suffix = os.path.splitext(filename)[1].lower() or ".jpg"
        await security_manager.validate_file(filename, filesize, temp_path, upload_content_type)

        if auth_user:
            uid = auth_user["uid"]

            current_balance = await get_user_balance(uid)
            if current_balance < settings.detect_credit_cost:
                logger.warning("insufficient_credits", extra={
                    "action": "insufficient_credits",
                    "endpoint": "detect",
                    "has": current_balance,
                    "need": settings.detect_credit_cost,
                    "user_type": "authenticated",
                })
                raise HTTPException(status_code=402, detail="Insufficient credits")

            start_time = time.time()

            result = await security_manager.secure_execute(
                request, filename, filesize, temp_path,
                lambda path: detect_ai_media(path, trusted_metadata=sidecar_metadata),
                uid=uid
            )

            if result.get("summary") in ["Analysis Failed", "File too large to scan"]:
                new_balance = await get_user_balance(uid)
            else:
                new_balance = await consume_credits(uid, settings.detect_credit_cost, "detect", filename)

        else:
            current_credits = wallet.get("credits", 0)

            if current_credits < settings.detect_credit_cost:
                logger.warning("insufficient_credits", extra={
                    "action": "insufficient_credits",
                    "endpoint": "detect",
                    "has": current_credits,
                    "need": settings.detect_credit_cost,
                    "user_type": "guest",
                })
                raise HTTPException(status_code=402, detail="Insufficient credits")

            start_time = time.time()

            result = await security_manager.secure_execute(
                request, filename, filesize, temp_path,
                lambda path: detect_ai_media(path, trusted_metadata=sidecar_metadata),
                uid=device_id
            )

            if result.get("summary") in ["Analysis Failed", "File too large to scan"]:
                wallet = await get_guest_wallet(device_id)
                new_balance = wallet.get("credits", 0)
            else:
                new_balance = await deduct_guest_credits(device_id, cost=settings.detect_credit_cost)

        duration = time.time() - start_time

        is_gemini_used = result.get("is_gemini_used", False)
        is_cached = result.get("is_cached", False)
        actual_gpu_time_ms = result.get("gpu_time_ms", 0.0)
        actual_gpu_sec = actual_gpu_time_ms / 1000.0

        if is_cached:
            cost = 0.0
            log_transaction("CACHE", 0.0, {"file": filename, "device_id": device_id})
        elif is_gemini_used:
            cost = settings.gemini_fixed_cost
            gemini_usage = result.get("usage", {})
            log_transaction("GEMINI", -cost, {"file": filename, "device_id": device_id, "usage": gemini_usage})
        elif actual_gpu_sec > 0:
            cost = actual_gpu_sec * settings.gpu_rate_per_sec
            log_transaction("GPU", -cost, {"file": filename, "device_id": device_id, "duration": actual_gpu_sec})
        else:
            cost = duration * settings.cpu_rate_per_sec
            log_transaction("CPU", -cost, {"file": filename, "device_id": device_id, "duration": duration})

        result.pop("gpu_time_ms", None)
        result.pop("is_gemini_used", None)
        result.pop("is_cached", None)

        result["new_balance"] = new_balance

        short_id = _generate_short_id()
        rc = redis_module.client
        if rc:
            shareable = {k: v for k, v in result.items() if k != "new_balance"}
            await rc.setex(f"report:{short_id}", settings.report_cache_ttl_sec, json.dumps(shareable))
        result["short_id"] = short_id

        failed = result.get("summary") in ("Analysis Failed", "File too large to scan")
        credits_consumed = 0 if failed else settings.detect_credit_cost

        logger.info("scan_completed", extra={
            "action": "scan_completed",
            "outcome": result.get("summary"),
            "confidence_score": result.get("confidence_score"),
            "is_gemini_used": is_gemini_used,
            "is_cached": is_cached,
            "is_short_circuited": result.get("is_short_circuited", False),
            "media_type": "video" if suffix in VIDEO_SUFFIXES else "image",
            "user_type": "authenticated" if auth_user else "guest",
            "duration_ms": round(duration * 1000, 1),
            "cost_usd": cost,
            "credits_consumed": credits_consumed,
            "media_file": filename,
        })

        return result

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
