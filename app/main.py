import os
import tempfile
import logging
import time
import uuid
import hashlib
import json
import base64
import string
import random
from datetime import datetime, timezone, timedelta
from typing import Optional
import hmac
import aiohttp
from app.finance_logger import log_transaction
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, Query, BackgroundTasks
from firebase_admin import firestore
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
import psutil

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.detector import detect_ai_media
from app.schemas import DetectionResponse, ShareRequest, ShareResponse
from app.config import settings
from app.runpod_client import pending_jobs, webhook_result_buffer, cleanup_stale_jobs, run_gpu_inpainting
from app.security import (
    security_manager,
    verify_turnstile,
    check_ban_status,
    deduct_guest_credits,
    get_guest_wallet,
    check_ip_device_limit,
    get_client_ip,
    db,
    redis_client
)
from contextlib import asynccontextmanager
from fastapi.responses import Response, PlainTextResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

RUNPOD_WEBHOOK_SECRET = os.getenv("RUNPOD_WEBHOOK_SECRET", "")
RECHARGE_SECRET_KEY = os.getenv("RECHARGE_SECRET_KEY", "")

cleanup_task = None

def _generate_short_id(length: int = settings.short_id_length) -> str:
    alphabet = string.ascii_letters + string.digits
    return ''.join(random.choices(alphabet, k=length))


def log_memory(stage: str):
    """Log current process and system memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()
    logger.info(
        f"[MEMORY] {stage} | "
        f"PID: {os.getpid()} | "
        f"Process RSS: {mem_info.rss / 1024 / 1024:.2f} MB | "
        f"System Available: {sys_mem.available / 1024 / 1024:.2f} MB / {sys_mem.total / 1024 / 1024:.2f} MB"
    )

async def periodic_cleanup():
    """Background task that removes stale RunPod jobs every 30 seconds."""
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_sec)
            cleanup_stale_jobs()
            logger.debug("[CLEANUP] Periodic cleanup completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[CLEANUP] Error in periodic cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cleanup_task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("[STARTUP] Background cleanup task started")
    yield
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    logger.info("[SHUTDOWN] Background cleanup task stopped")

app = FastAPI(title="AI Provenance & Cleansing API", lifespan=lifespan)

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Forces CORS headers onto every HTTP error response so browsers can read the JSON body.
    Also drains the request body to prevent ERR_HTTP2_PROTOCOL_ERROR on early-rejected uploads.
    """
    headers = getattr(exc, "headers", None) or {}
    headers["Access-Control-Allow-Origin"] = "*"
    headers["Access-Control-Allow-Credentials"] = "true"
    headers["Access-Control-Allow-Methods"] = "*"
    headers["Access-Control-Allow-Headers"] = "*"

    try:
        async for _ in request.stream():
            pass
    except Exception as e:
        logger.warning(f"Error draining request stream in exception handler: {e}")

    response_data = {"detail": exc.detail}
    logger.info(f"[ERROR HANDLER] Returning {exc.status_code} to client. Body: {response_data}, Headers: {headers}")

    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=headers
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nDisallow: /"

@app.post("/webhook/runpod")
async def runpod_webhook(request: Request):
    """
    Receives webhook callbacks from RunPod when jobs complete.
    Validates X-Runpod-Signature header if RUNPOD_WEBHOOK_SECRET is set.
    """
    try:
        logger.info(f"[WEBHOOK] Headers received: {dict(request.headers)}")

        if RUNPOD_WEBHOOK_SECRET:
            signature = (
                request.headers.get("X-Runpod-Signature", "") or
                request.headers.get("Authorization", "").replace("Bearer ", "") or
                request.headers.get("X-Webhook-Secret", "")
            )
            if signature != RUNPOD_WEBHOOK_SECRET:
                logger.warning(f"[WEBHOOK] Signature mismatch. Got: '{signature[:20]}...' Expected: '{RUNPOD_WEBHOOK_SECRET[:20]}...'")
                # TODO: Enable strict auth once RunPod's header format is confirmed

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
                    logger.info(f"[WEBHOOK] Job {job_id} completed in {elapsed_ms:.2f}ms (batch={batch_size}, gpu={gpu_time:.1f}ms)")
                else:
                    logger.info(f"[WEBHOOK] Job {job_id} completed in {elapsed_ms:.2f}ms, ai_score={output.get('ai_score', 'N/A')}")
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
                logger.info(f"[WEBHOOK] Job {job_id} buffered (arrived before pending_jobs set), ai_score={output.get('ai_score')}")
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


async def download_image(url: str, max_size: int = settings.max_image_download_bytes) -> tuple[bytes, str]:
    """Downloads an image from a URL or decodes a base64 data URI."""
    if url.startswith("data:"):
        try:
            header, data_str = url.split(",", 1)
            if ";base64" not in header:
                raise HTTPException(status_code=400, detail="Only base64 data URIs are supported")
            content = base64.b64decode(data_str)
            if len(content) > max_size:
                raise HTTPException(status_code=400, detail=f"Image too large (max {max_size // (1024*1024)}MB)")
            mime_type = header.split(":")[1].split(";")[0]
            suffix = ".jpg"
            if "png" in mime_type: suffix = ".png"
            elif "jpeg" in mime_type or "jpg" in mime_type: suffix = ".jpg"
            elif "webp" in mime_type: suffix = ".webp"
            elif "gif" in mime_type: suffix = ".gif"
            elif "heic" in mime_type: suffix = ".heic"
            elif "heif" in mime_type: suffix = ".heif"
            elif "tiff" in mime_type: suffix = ".tiff"
            elif "bmp" in mime_type: suffix = ".bmp"
            return content, f"pasted_image{suffix}"
        except Exception as e:
            logger.error(f"Error decoding data URI: {e}")
            raise HTTPException(status_code=400, detail="Invalid data URI")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: Status {response.status}")
                content = await response.read()
                if len(content) > max_size:
                    raise HTTPException(status_code=400, detail=f"Image too large (max {max_size // (1024*1024)}MB)")
                content_type = response.headers.get("Content-Type", "")
                suffix = ".jpg"
                if "png" in content_type: suffix = ".png"
                elif "jpeg" in content_type or "jpg" in content_type: suffix = ".jpg"
                elif "webp" in content_type: suffix = ".webp"
                elif "gif" in content_type: suffix = ".gif"
                elif "heic" in content_type: suffix = ".heic"
                elif "heif" in content_type: suffix = ".heif"
                elif "tiff" in content_type: suffix = ".tiff"
                elif "bmp" in content_type: suffix = ".bmp"
                elif "mp4" in content_type: suffix = ".mp4"
                elif "quicktime" in content_type or "mov" in content_type: suffix = ".mov"
                if not content_type or "application" in content_type or "octet-stream" in content_type:
                    lower_url = url.lower()
                    if lower_url.endswith(".png"): suffix = ".png"
                    elif lower_url.endswith(".webp"): suffix = ".webp"
                    elif lower_url.endswith(".gif"): suffix = ".gif"
                    elif lower_url.endswith(".heic"): suffix = ".heic"
                    elif lower_url.endswith(".heif"): suffix = ".heif"
                    elif lower_url.endswith(".tiff") or lower_url.endswith(".tif"): suffix = ".tiff"
                    elif lower_url.endswith(".bmp"): suffix = ".bmp"
                    elif lower_url.endswith(".mp4"): suffix = ".mp4"
                    elif lower_url.endswith(".mov"): suffix = ".mov"
                return content, f"downloaded_media{suffix}"
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token")
):
    """
    Detect AI-generated content in images/videos.

    Accepts multipart/form-data with 'file' or 'url' field (plus optional 'trusted_metadata'),
    or JSON payload { "url": "https://..." }.

    Validates Turnstile, checks bans, and deducts guest credits only on success.
    """
    ip = get_client_ip(request)

    if not turnstile_token:
        raise HTTPException(
            status_code=403,
            detail={"code": "CAPTCHA_REQUIRED", "message": "Verification needed"}
        )

    is_human = await verify_turnstile(turnstile_token)
    if not is_human:
        raise HTTPException(status_code=403, detail="Invalid CAPTCHA")

    await check_ip_device_limit(ip, device_id, token_already_verified=True)

    if check_ban_status(device_id):
        raise HTTPException(status_code=403, detail="Device is banned")

    file_content = None
    filename = "unknown"
    sidecar_metadata = None

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            payload = await request.json()
            url = payload.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="Missing 'url' in JSON body")
            file_content, filename = await download_image(url)
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
                logger.info(f"[SIDECAR] Device {device_id} provided trusted metadata: {list(sidecar_metadata.keys())}")
            except json.JSONDecodeError as e:
                logger.warning(f"[SIDECAR] Invalid JSON in trusted_metadata from {device_id}: {e}")

        if file_obj:
            if not isinstance(file_obj, UploadFile):
                if isinstance(file_obj, str):
                    raise HTTPException(status_code=400, detail="Invalid file upload format")
            file_content = await file_obj.read()
            filename = file_obj.filename or "uploaded_file"
        elif url_obj:
            if isinstance(url_obj, str):
                file_content, filename = await download_image(url_obj)
            else:
                raise HTTPException(status_code=400, detail="Invalid url field")
        else:
            raise HTTPException(status_code=400, detail="Must provide 'file' or 'url' in form data")

    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Use multipart/form-data or application/json")

    filesize = len(file_content)
    suffix = os.path.splitext(filename)[1].lower() or ".jpg"

    log_memory(f"Pre-Detect: {filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    try:
        security_manager.validate_file(filename, filesize, temp_path)

        wallet = get_guest_wallet(device_id)
        current_credits = wallet.get("credits", 0)

        if current_credits < settings.detect_credit_cost:
            logger.info(f"[BILLING] Insufficient credits for {device_id} (Has: {current_credits}, Need: {settings.detect_credit_cost})")
            raise HTTPException(status_code=402, detail="Insufficient credits")

        start_time = time.time()

        result = await security_manager.secure_execute(
            request, filename, filesize, temp_path,
            lambda path: detect_ai_media(path, trusted_metadata=sidecar_metadata),
            uid=device_id
        )

        if result.get("summary") in ["Analysis Failed", "File too large to scan"]:
            logger.info(f"[BILLING] Skipped deduction for {device_id} due to soft failure: {result.get('summary')}")
            wallet = get_guest_wallet(device_id)
            new_balance = wallet.get("credits", 0)
        else:
            new_balance = deduct_guest_credits(device_id, cost=settings.detect_credit_cost)

        duration = time.time() - start_time
        log_memory(f"Post-Detect: {filename}")

        is_gemini_used = result.get("is_gemini_used", False)
        is_cached = result.get("is_cached", False)
        actual_gpu_time_ms = result.get("gpu_time_ms", 0.0)
        actual_gpu_sec = actual_gpu_time_ms / 1000.0

        if is_cached:
            cost = 0.0
            logger.info(f"[COST] Cache hit: $0.00")
            log_transaction("CACHE", 0.0, {"file": filename, "device_id": device_id})
        elif is_gemini_used:
            cost = settings.gemini_fixed_cost
            logger.info(f"[COST] Gemini analysis: ${cost:.6f}")
            gemini_usage = result.get("usage", {})
            log_transaction("GEMINI", -cost, {"file": filename, "device_id": device_id, "usage": gemini_usage})
        elif actual_gpu_sec > 0:
            cost = actual_gpu_sec * settings.gpu_rate_per_sec
            logger.info(f"[COST] GPU: {actual_gpu_sec:.3f}s (actual) vs {duration:.3f}s (round-trip) | Cost: ${cost:.6f}")
            log_transaction("GPU", -cost, {"file": filename, "device_id": device_id, "duration": actual_gpu_sec})
        else:
            cost = duration * settings.cpu_rate_per_sec
            log_transaction("CPU", -cost, {"file": filename, "device_id": device_id, "duration": duration})

        result.pop("gpu_time_ms", None)
        result.pop("is_gemini_used", None)
        result.pop("is_cached", None)

        result["new_balance"] = new_balance

        short_id = _generate_short_id()
        if redis_client:
            shareable = {k: v for k, v in result.items() if k != "new_balance"}
            redis_client.setex(f"report:{short_id}", settings.report_cache_ttl_sec, json.dumps(shareable))
        result["short_id"] = short_id

        logger.info(f"[ROUTE] Final Response for {filename}: {json.dumps(result)}")
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/user/balance")
async def get_balance(
    request: Request,
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token")
):
    """
    Returns the current credit balance for a guest device.
    Auto-creates a wallet with welcome credits if one does not exist.
    """
    ip = get_client_ip(request)
    await check_ip_device_limit(ip, device_id, turnstile_token)
    wallet = get_guest_wallet(device_id)
    balance = wallet.get("credits", 0)
    logger.info(f"[BALANCE] Device: {device_id} | Credits: {balance}")
    return {"balance": balance}


class RechargeRequest(BaseModel):
    device_id: str
    amount: int = settings.default_recharge_amount
    secret_key: str

def perform_recharge(device_id: str, amount: int, secret_key: str):
    if not RECHARGE_SECRET_KEY or secret_key != RECHARGE_SECRET_KEY:
        logger.warning(f"⚠️ Invalid recharge attempt for {device_id}")
        raise HTTPException(status_code=403, detail="Invalid secret key")

    try:
        doc_ref = db.collection('guest_wallets').document(device_id)

        @firestore.transactional
        def recharge_transaction(transaction, ref):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                transaction.set(ref, {
                    "credits": settings.welcome_credits + amount,
                    "last_active": firestore.SERVER_TIMESTAMP,
                    "is_banned": False,
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
                })
                return settings.welcome_credits + amount
            else:
                current = snapshot.get("credits") or 0
                transaction.update(ref, {
                    "credits": current + amount,
                    "last_active": firestore.SERVER_TIMESTAMP,
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.wallet_ttl_days),
                })
                return current + amount

        transaction = db.transaction()
        new_balance = recharge_transaction(transaction, doc_ref)
        logger.info(f"💰 Recharged {amount} credits for {device_id}. New balance: {new_balance}")
        log_transaction("AD_REWARD", settings.ad_revenue_per_reward, {"device_id": device_id, "credits": amount})
        return {"status": "success", "new_balance": new_balance}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔥 Recharge failed: {e}")
        raise HTTPException(status_code=500, detail="Recharge failed")

@app.post("/api/credits/add")
async def add_credits_post(payload: RechargeRequest):
    return perform_recharge(payload.device_id, payload.amount, payload.secret_key)

@app.get("/api/credits/webhook")
async def add_credits_get(
    user_id: str = Query(..., alias="device_id"),
    amount: int = settings.default_recharge_amount,
    key: str = Query(..., alias="secret_key")
):
    return perform_recharge(user_id, amount, key)


@app.post("/inpaint/image")
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

    ip = get_client_ip(request)
    await check_ip_device_limit(ip, device_id, turnstile_token)

    if check_ban_status(device_id):
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
    from app.security import redis_client

    is_free_retry = bool(redis_client and redis_client.get(cache_key))
    if is_free_retry:
        logger.info(f"[INPAINT] Free retry available for {device_id} on image {img_hash[:8]}")

    wallet = get_guest_wallet(device_id)
    current_credits = wallet.get("credits", 0)

    if not is_free_retry and current_credits < settings.inpaint_credit_cost:
        logger.info(f"[BILLING] Insufficient credits for {device_id} (Has: {current_credits}, Need: {settings.inpaint_credit_cost})")
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
            if redis_client:
                redis_client.delete(cache_key)
            new_balance = current_credits
        else:
            new_balance = deduct_guest_credits(device_id, cost=settings.inpaint_credit_cost)
            if redis_client:
                redis_client.set(cache_key, "1", ex=settings.deepfake_dedupe_ttl_sec)

        headers = {"X-User-Balance": str(new_balance)}
        return Response(content=result_bytes, media_type="image/png", headers=headers)

    except Exception as e:
        logger.error(f"[INPAINT] GPU Worker failed for {request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Inpainting service unavailable")


LEMONSQUEEZY_WEBHOOK_SECRET = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET")

@app.post("/webhooks/lemonsqueezy")
async def lemonsqueezy_webhook(request: Request, x_signature: str = Header(None, alias="X-Signature")):
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


@app.post("/api/v1/reports/share", response_model=ShareResponse, status_code=201)
async def create_share_link(request: ShareRequest):
    """
    Publishes a cached scan result as a permanent shared report.
    Idempotent: re-publishing the same short_id returns the existing report_id.
    """
    if redis_client and redis_client.get(f"is_shared:{request.short_id}"):
        return {"report_id": request.short_id}

    raw = redis_client.get(f"report:{request.short_id}") if redis_client else None
    if not raw:
        raise HTTPException(status_code=404, detail="Share link expired or invalid.")

    payload = json.loads(raw) if isinstance(raw, str) else raw
    now = datetime.now(timezone.utc)
    payload["created_at"] = now
    payload["expires_at"] = now + timedelta(days=settings.report_ttl_days)

    db.collection("shared_reports").document(request.short_id).set(payload)

    if redis_client:
        redis_client.setex(f"is_shared:{request.short_id}", settings.share_lock_ttl_sec, "1")

    return {"report_id": request.short_id}


def _extend_report_ttl(report_id: str, new_expiry: datetime):
    """
    Background task: extends a viral report's Firestore TTL.
    Uses Redis nx lock to prevent duplicate writes under concurrent traffic.
    """
    if redis_client and redis_client.set(f"extending:{report_id}", "1", nx=True, ex=settings.extend_lock_ttl_sec):
        db.collection("shared_reports").document(report_id).update({
            "expires_at": new_expiry
        })


@app.get("/api/v1/reports/share/{report_id}")
async def get_shared_report(report_id: str, background_tasks: BackgroundTasks):
    """
    Fetches a public shared report. No auth required.
    Auto-extends TTL by report_extend_days if fewer than report_extend_threshold_days remain.
    """
    doc = db.collection("shared_reports").document(report_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    data = doc.to_dict()
    expires_at = data.get("expires_at")

    if expires_at:
        now = datetime.now(timezone.utc)
        time_left = expires_at - now
        if time_left.days < settings.report_extend_threshold_days:
            background_tasks.add_task(
                _extend_report_ttl,
                report_id,
                now + timedelta(days=settings.report_extend_days)
            )

    data.pop("created_at", None)
    data.pop("expires_at", None)
    return data
