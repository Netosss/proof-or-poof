import os
import tempfile
import logging
import time
import uuid
import hashlib
import json
from typing import Optional, Union
import hmac
import aiohttp
from app.finance_logger import log_transaction
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, Query
from firebase_admin import firestore
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
import psutil

# Load environment variables at the very beginning
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.detector import detect_ai_media
from app.schemas import DetectionResponse
from app.runpod_client import run_video_removal, pending_jobs, webhook_result_buffer, cleanup_stale_jobs
from app.security import (
    security_manager, 
    check_and_deduct_credits, 
    verify_turnstile, 
    check_ban_status, 
    deduct_guest_credits,
    get_guest_wallet,
    check_ip_device_limit,
    get_client_ip,
    db
)
from fastapi import Depends
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from fastapi.responses import Response, PlainTextResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Faux Lens Remover
from app.remover import FauxLensRemover

# Webhook authentication secret (set via environment variable)
RUNPOD_WEBHOOK_SECRET = os.getenv("RUNPOD_WEBHOOK_SECRET", "")

# Recharge Webhook Secret
RECHARGE_SECRET_KEY = os.getenv("RECHARGE_SECRET_KEY", "")

# Background cleanup task
cleanup_task = None

def log_memory(stage: str):
    """Log current memory usage."""
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
    """Background task to clean up stale pending jobs every 30 seconds."""
    while True:
        try:
            await asyncio.sleep(30)
            cleanup_stale_jobs()
            logger.debug("[CLEANUP] Periodic cleanup completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[CLEANUP] Error in periodic cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global cleanup_task
    # Startup: start background cleanup
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("[STARTUP] Background cleanup task started")

    # Initialize FauxLensRemover
    try:
        app.state.remover = FauxLensRemover()
        logger.info("[STARTUP] FauxLensRemover initialized")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize FauxLensRemover: {e}")
        # We don't raise here so the app can still start, but inpaint endpoint will fail
        app.state.remover = None

    yield
    # Shutdown: cancel cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    logger.info("[SHUTDOWN] Background cleanup task stopped")

app = FastAPI(title="AI Provenance & Cleansing API", lifespan=lifespan)

# ---- Global Exception Handler for CORS ----
# Ensures that HTTP exceptions (like our 403 CAPTCHA block) always return CORS headers
# so the frontend can read the JSON body instead of getting a generic Network Error.
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    headers = getattr(exc, "headers", None)
    if headers is None:
        headers = {}
    
    # Force CORS headers onto every single HTTP error response!
    # Without this, the browser will block the 403 response.
    headers["Access-Control-Allow-Origin"] = "*"
    headers["Access-Control-Allow-Credentials"] = "true"
    headers["Access-Control-Allow-Methods"] = "*"
    headers["Access-Control-Allow-Headers"] = "*"
    
    # 🔴 CRITICAL FIX for ERR_HTTP2_PROTOCOL_ERROR on file uploads 🔴
    # Drain the incoming request body so the server doesn't forcefully 
    # drop the TCP connection when rejecting uploads early.
    try:
        # We use stream() to safely discard bytes without loading large files into memory
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

# ---- Pricing Constants (USD per unit) ----
GPU_RATE_PER_SEC = 0.0019  # RunPod A5000/L4 rate
CPU_RATE_PER_SEC = 0.0001  # Estimated Railway CPU rate
GEMINI_FIXED_COST = 0.0024  # Cost per Gemini 3.0 Pro analysis
AD_REVENUE_PER_REWARD = 0.015  # Avg eCPM for verified view

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Healthcheck ----
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    # "User-agent: *" means all bots
    # "Disallow: /" means don't crawl anything here
    return "User-agent: *\nDisallow: /"

# ---- RunPod Webhook Endpoint ----
@app.post("/webhook/runpod")
async def runpod_webhook(request: Request):
    """
    Receives webhook callbacks from RunPod when jobs complete.
    This eliminates polling and provides instant results.
    
    Security: Validates X-Runpod-Signature header if RUNPOD_WEBHOOK_SECRET is set.
    """
    try:
        # ---- Authentication ----
        # Log headers for debugging (remove in production)
        logger.info(f"[WEBHOOK] Headers received: {dict(request.headers)}")
        
        if RUNPOD_WEBHOOK_SECRET:
            # Try multiple header names that RunPod might use
            signature = (
                request.headers.get("X-Runpod-Signature", "") or
                request.headers.get("Authorization", "").replace("Bearer ", "") or
                request.headers.get("X-Webhook-Secret", "")
            )
            
            if signature != RUNPOD_WEBHOOK_SECRET:
                logger.warning(f"[WEBHOOK] Signature mismatch. Got: '{signature[:20]}...' Expected: '{RUNPOD_WEBHOOK_SECRET[:20]}...'")
                # For now, log but don't block - RunPod might not send signature
                # TODO: Enable strict auth once we confirm RunPod's header format
                # raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        payload = await request.json()
        job_id = payload.get("id")
        status = payload.get("status")
        output = payload.get("output")
        
        logger.info(f"[WEBHOOK] Received callback for job {job_id}, status: {status}")
        
        if job_id and job_id in pending_jobs:
            future, start_time = pending_jobs[job_id]
            elapsed_ms = (time.time() - start_time) * 1000
            
            if status == "COMPLETED" and output:
                # Validate output has required fields
                if not isinstance(output, dict):
                    logger.warning(f"[WEBHOOK] Job {job_id} output is not a dict: {type(output)}")
                    output = {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": "Invalid output format"}
                
                if not future.done():
                    future.set_result(output)
                
                # Log based on response type (single vs batch)
                if "results" in output:
                    # Batch response
                    batch_size = len(output.get("results", []))
                    gpu_time = output.get("timing_ms", {}).get("total", 0)
                    logger.info(f"[WEBHOOK] Job {job_id} completed in {elapsed_ms:.2f}ms (batch={batch_size}, gpu={gpu_time:.1f}ms)")
                else:
                    # Single image response
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
                # IN_PROGRESS or other status - don't resolve yet
                logger.info(f"[WEBHOOK] Job {job_id} status update: {status}")
                return {"status": "acknowledged"}
        elif job_id and status == "COMPLETED" and output:
            # Race condition: webhook arrived before pending_jobs was set
            # Validate before buffering
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


async def download_image(url: str, max_size: int = 50 * 1024 * 1024) -> tuple[bytes, str]:
    """Helper to download image from URL."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: Status {response.status}")
                
                content = await response.read()
                if len(content) > max_size:
                    raise HTTPException(status_code=400, detail="Image too large (max 50MB)")
                
                # Determine filename/extension
                content_type = response.headers.get("Content-Type", "")
                suffix = ".jpg" # Default
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
                
                # Try to guess from url if content-type is generic
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

# ---- Detect endpoint ----
@app.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request, 
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token")
):
    """
    Detect AI-generated content in images/videos.
    
    Accepts:
    1. Multipart/form-data with 'file' field (and optional 'trusted_metadata')
    2. Multipart/form-data with 'url' field (and optional 'trusted_metadata')
    3. JSON payload: { "url": "https://..." }
    
    Guest Wallet Flow: Validates Turnstile, checks bans, and deducts credits from device wallet.
    """
    
    # 1. Security & Wallet Check
    # Verify IP Device Limit and CAPTCHA
    ip = get_client_ip(request)
    
    # TEMPORARY FOR FRONTEND TESTING: Force CAPTCHA on every request
    if not turnstile_token:
        raise HTTPException(
            status_code=403, 
            detail={"code": "CAPTCHA_REQUIRED", "message": "Verification needed (Testing Mode)"}
        )
    
    is_human = await verify_turnstile(turnstile_token)
    if not is_human:
        raise HTTPException(status_code=403, detail="Invalid CAPTCHA")
        
    # Standard logic (bypassed by the forced check above for now)
    # await check_ip_device_limit(ip, device_id, turnstile_token)

    # Check Ban Status
    if check_ban_status(device_id):
        raise HTTPException(status_code=403, detail="Device is banned")

    # 2. Extract Content (File or URL)
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
        
        # Parse trusted metadata sidecar if provided
        if trusted_metadata_obj and isinstance(trusted_metadata_obj, str):
            try:
                sidecar_metadata = json.loads(trusted_metadata_obj)
                logger.info(f"[SIDECAR] Device {device_id} provided trusted metadata: {list(sidecar_metadata.keys())}")
            except json.JSONDecodeError as e:
                logger.warning(f"[SIDECAR] Invalid JSON in trusted_metadata from {device_id}: {e}")
        
        if file_obj:
            # Handle standard file upload
            if not isinstance(file_obj, UploadFile):
                 # Should theoretically not happen with correct multipart parsing, but safety check
                 if isinstance(file_obj, str): # Could happen if client sends string in file field?
                     raise HTTPException(status_code=400, detail="Invalid file upload format")
            
            # Use Starlette's UploadFile methods
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

    # 3. Process Content
    # Use Secure Wrapper for primary security checks
    # (Rate limiting + Type/Size validation + Content Deep Check)
    filesize = len(file_content)
    suffix = os.path.splitext(filename)[1].lower()
    
    if not suffix:
        # Fallback if no suffix found
        suffix = ".jpg"

    log_memory(f"Pre-Detect: {filename}")

    # Explicit Validation (Check before charging)
    # This validates extension, size, and magic bytes (if possible)
    # We pass None for file_path here since we haven't saved it yet, 
    # but validate_file will check extension/size.
    # To check magic bytes properly, we'd need the file on disk or a BytesIO wrapper.
    # However, creating the temp file is cheap, so we'll do that first, THEN validate, THEN charge.
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    try:
        # Validate BEFORE charging
        # Pass temp_path so it can check magic bytes (deep validation)
        security_manager.validate_file(filename, filesize, temp_path)
        
        # 1. Check Balance First (Fast, Cheap)
        # Deduct Credits (Atomic) - Raises 402 if insufficient
        wallet = get_guest_wallet(device_id)
        current_credits = wallet.get("credits", 0)
        
        if current_credits < 5:
            logger.info(f"[BILLING] Insufficient credits for {device_id} (Has: {current_credits}, Need: 5)")
            raise HTTPException(status_code=402, detail="Insufficient credits")

        start_time = time.time()
        
        # The wrapper handles security logic; we pass detect_ai_media as the worker function
        # Pass device_id for rate limiting
        # Note: secure_execute calls validate_file again, which is fine (redundant safety)
        result = await security_manager.secure_execute(
            request, filename, filesize, temp_path, 
            lambda path: detect_ai_media(path, trusted_metadata=sidecar_metadata),
            uid=device_id
        )

        # 2. Deduct Credits ONLY on Success
        # If secure_execute raises an error, this line is never reached.
        # Also skip deduction if the result indicates a soft failure (Analysis Failed, File too large)
        if result.get("summary") in ["Analysis Failed", "File too large to scan"]:
            logger.info(f"[BILLING] Skipped deduction for {device_id} due to soft failure: {result.get('summary')}")
            # Get current balance without deduction
            wallet = get_guest_wallet(device_id)
            new_balance = wallet.get("credits", 0)
        else:
            new_balance = deduct_guest_credits(device_id, cost=5)
        
        duration = time.time() - start_time
        
        log_memory(f"Post-Detect: {filename}")
        
        # Check explicit flags for Gemini and Cache usage
        is_gemini_used = result.get("is_gemini_used", False)
        is_cached = result.get("is_cached", False)
        
        # Use actual GPU time for cost calculation (not round-trip time)
        actual_gpu_time_ms = result.get("gpu_time_ms", 0.0)
        actual_gpu_sec = actual_gpu_time_ms / 1000.0
        
        if is_cached:
            cost = 0.0
            method = "cached_result"
            gpu_sec, cpu_sec = 0, 0
            logger.info(f"[COST] Cache hit: $0.00")
            log_transaction("CACHE", 0.0, {"file": filename, "device_id": device_id})
        elif is_gemini_used:
            cost = GEMINI_FIXED_COST
            method = "detect_with_gemini"
            gpu_sec, cpu_sec = 0, duration
            logger.info(f"[COST] Gemini analysis: ${cost:.6f}")
            gemini_usage = result.get("usage", {})
            log_transaction("GEMINI", -cost, {"file": filename, "device_id": device_id, "usage": gemini_usage})
        elif actual_gpu_sec > 0:
            # Cost based on actual GPU utilization, not network round-trip
            cost = actual_gpu_sec * GPU_RATE_PER_SEC
            method = "detect_with_gpu"
            gpu_sec, cpu_sec = actual_gpu_sec, duration - actual_gpu_sec
            logger.info(f"[COST] GPU: {actual_gpu_sec:.3f}s (actual) vs {duration:.3f}s (round-trip) | Cost: ${cost:.6f}")
            log_transaction("GPU", -cost, {"file": filename, "device_id": device_id, "duration": actual_gpu_sec})
        else:
            cost = duration * CPU_RATE_PER_SEC
            method = "detect_metadata_only"
            gpu_sec, cpu_sec = 0, duration
            log_transaction("CPU", -cost, {"file": filename, "device_id": device_id, "duration": duration})
        # Remove internal fields before returning response
        result.pop("gpu_time_ms", None)
        result.pop("is_gemini_used", None)
        result.pop("is_cached", None) 
        
        # Attach new balance
        result["new_balance"] = new_balance

        logger.info(f"[ROUTE] Final Response for {filename}: {json.dumps(result)}")
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---- Guest Balance Endpoint ----
@app.get("/api/user/balance")
async def get_balance(
    request: Request,
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token")
):
    """
    Get current credit balance for a guest device.
    Auto-creates wallet with 10 credits if it doesn't exist.
    """
    ip = get_client_ip(request)
    await check_ip_device_limit(ip, device_id, turnstile_token)

    wallet = get_guest_wallet(device_id)
    balance = wallet.get("credits", 0)
    logger.info(f"[BALANCE] Device: {device_id} | Credits: {balance}")
    return {"balance": balance}

# ---- Recharge Webhook ----
class RechargeRequest(BaseModel):
    device_id: str
    amount: int = 5
    secret_key: str

def perform_recharge(device_id: str, amount: int, secret_key: str):
    # Security Check
    if not RECHARGE_SECRET_KEY or secret_key != RECHARGE_SECRET_KEY:
        logger.warning(f"⚠️ Invalid recharge attempt for {device_id}")
        raise HTTPException(status_code=403, detail="Invalid secret key")

    try:
        doc_ref = db.collection('guest_wallets').document(device_id)

        @firestore.transactional
        def recharge_transaction(transaction, ref):
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                # If user doesn't exist yet, give them Welcome Bonus (10) + Reward (amount)
                transaction.set(ref, {
                    "credits": 10 + amount,
                    "last_active": firestore.SERVER_TIMESTAMP,
                    "is_banned": False
                })
                return 10 + amount
            else:
                current = snapshot.get("credits") or 0
                transaction.update(ref, {
                    "credits": current + amount,
                    "last_active": firestore.SERVER_TIMESTAMP
                })
                return current + amount

        transaction = db.transaction()
        new_balance = recharge_transaction(transaction, doc_ref)
        
        logger.info(f"💰 Recharged {amount} credits for {device_id}. New balance: {new_balance}")
        log_transaction("AD_REWARD", AD_REVENUE_PER_REWARD, {"device_id": device_id, "credits": amount})
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
    user_id: str = Query(..., alias="device_id"), # Supports ?user_id= or ?device_id=
    amount: int = 5,
    key: str = Query(..., alias="secret_key")
):
    return perform_recharge(user_id, amount, key)

# ---- Inpaint Endpoint (Sync/CPU Optimized) ----
@app.post("/inpaint/image")
async def inpaint_image(
    request: Request,
    image: UploadFile = File(...), 
    mask: UploadFile = File(...),
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token")
):
    """
    Remove objects from an image using the LaMA model (CPU optimized).
    This endpoint is async to verify Turnstile, and offloads heavy work to threadpool.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[INPAINT] Request {request_id} started. Image: {image.filename}, Device: {device_id}")

    # 1. Security & Wallet Check
    # Verify IP Device Limit and CAPTCHA
    ip = get_client_ip(request)
    await check_ip_device_limit(ip, device_id, turnstile_token)

    if check_ban_status(device_id):
        raise HTTPException(status_code=403, detail="Device is banned")

    # [NEW] Free Retry Logic (Hash-Based)
    # Check if this exact image was just paid for.
    is_free_retry = False
    image_bytes = await image.read()
    image.file.seek(0) # Reset file pointer for later use
    
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    # Cache key: paid_image:{device_id}:{img_hash}
    cache_key = f"paid_image:{device_id}:{img_hash}"
    
    # We need to access redis directly. Using security_manager logic style.
    # Ideally import redis_client from app.security
    from app.security import redis_client

    if redis_client:
        if redis_client.get(cache_key):
            is_free_retry = True
            logger.info(f"[INPAINT] Free retry for {device_id} on image {img_hash[:8]}")
            # Delete key to consume the free retry (One free retry per payment)
            redis_client.delete(cache_key)
    
    new_balance = -1 # Placeholder

    if not is_free_retry:
        # Deduct 2 Credits (Atomic) - Raises 402 if insufficient
        try:
            new_balance = deduct_guest_credits(device_id, cost=2)
            
            # Set flag for next time
            if redis_client:
                # Expire in 10 minutes (plenty of time to retry)
                redis_client.set(cache_key, "1", ex=600) 
        except HTTPException as e:
            logger.warning(f"[INPAINT] Request {request_id} Insufficient funds for {device_id}")
            raise e
    else:
        # Just get current balance for header
        wallet = get_guest_wallet(device_id)
        new_balance = wallet.get("credits", 0)

    if not hasattr(app.state, "remover") or app.state.remover is None:
        logger.error(f"[INPAINT] Service unavailable for request {request_id}")
        raise HTTPException(status_code=503, detail="Inpainting service unavailable")

    try:
        start_time = time.time()
        
        # 1. READ (Async friendly)
        read_start = time.time()
        # image_bytes already read above for hashing
        mask_bytes = await mask.read()
        read_duration = time.time() - read_start
        
        log_memory(f"Pre-Inpaint: {image.filename}")
        
        image_size_mb = len(image_bytes) / (1024 * 1024)
        logger.info(f"[INPAINT] Request {request_id}: Files read in {read_duration:.3f}s. Image size: {image_size_mb:.2f}MB")
        
        # 2. PROCESS (Offload CPU work to threadpool)
        process_start = time.time()
        # Run CPU-heavy task in threadpool so we don't block the async event loop
        result = await run_in_threadpool(app.state.remover.process_image, image_bytes, mask_bytes)
        process_duration = time.time() - process_start
        
        total_duration = time.time() - start_time
        logger.info(f"[INPAINT] Request {request_id} COMPLETED in {total_duration:.3f}s (Processing: {process_duration:.3f}s)")
        
        log_memory(f"Post-Inpaint: {image.filename}")
        
        # 3. RETURN
        # Add X-User-Balance header
        headers = {"X-User-Balance": str(new_balance)}
        return Response(content=result, media_type="image/png", headers=headers)
        
    except ValueError as e:
        logger.warning(f"[INPAINT] Request {request_id} Validation Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[INPAINT] Request {request_id} FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


# ---- Lemon Squeezy Webhook ----
LEMONSQUEEZY_WEBHOOK_SECRET = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET")

@app.post("/webhooks/lemonsqueezy")
async def lemonsqueezy_webhook(request: Request, x_signature: str = Header(None, alias="X-Signature")):
    if not LEMONSQUEEZY_WEBHOOK_SECRET:
        return {"status": "ignored", "reason": "No secret set"}

    payload_bytes = await request.body()
    
    # Verify Signature
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
        
        # Financials
        total_cents = attributes.get("total", 0)
        total_usd = total_cents / 100.0
        
        # Metadata
        custom_data = payload.get("meta", {}).get("custom_data", {})
        user_id = custom_data.get("user_id", "unknown")
        
        log_transaction("LEMONSQUEEZY", total_usd, {"user_id": user_id, "order_id": data.get("id")})
        logger.info(f"💰 [LEMONSQUEEZY] Processed order {data.get('id')}: ${total_usd}")
        
    return {"status": "ok"}
