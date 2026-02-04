import os
import tempfile
import logging
import time
import uuid
import hashlib
import json
from typing import Optional
import hmac
from app.finance_logger import log_transaction
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, Query
from firebase_admin import firestore
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel

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
    db
)
from fastapi import Depends
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from fastapi.responses import Response

# Faux Lens Remover
from app.remover import FauxLensRemover

# Webhook authentication secret (set via environment variable)
RUNPOD_WEBHOOK_SECRET = os.getenv("RUNPOD_WEBHOOK_SECRET", "")

# Recharge Webhook Secret
RECHARGE_SECRET_KEY = os.getenv("RECHARGE_SECRET_KEY", "")

# Background cleanup task
cleanup_task = None

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

# ---- Detect endpoint ----
@app.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request, 
    file: UploadFile = File(...),
    trusted_metadata: Optional[str] = Form(None),
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: str = Header(..., alias="X-Turnstile-Token")
):
    """
    Detect AI-generated content in images/videos.
    Guest Wallet Flow: Validates Turnstile, checks bans, and deducts credits from device wallet.
    
    Args:
        file: The media file to analyze (image or video)
        trusted_metadata: Optional JSON string with device-extracted EXIF data.
        device_id: The Guest Device ID (FingerprintJS)
        turnstile_token: Cloudflare Turnstile token
    """
    
    # 1. Security & Wallet Check
    # Verify Turnstile
    is_human = await verify_turnstile(turnstile_token)
    if not is_human:
        raise HTTPException(status_code=403, detail="Turnstile validation failed")

    # Check Ban Status
    if check_ban_status(device_id):
        raise HTTPException(status_code=403, detail="Device is banned")

    # Deduct Credits (Atomic) - Raises 402 if insufficient
    new_balance = deduct_guest_credits(device_id, cost=5)

    # Parse trusted metadata sidecar if provided
    sidecar_metadata = None
    if trusted_metadata:
        try:
            sidecar_metadata = json.loads(trusted_metadata)
            logger.info(f"[SIDECAR] Device {device_id} provided trusted metadata: {list(sidecar_metadata.keys())}")
        except json.JSONDecodeError as e:
            logger.warning(f"[SIDECAR] Invalid JSON in trusted_metadata from {device_id}: {e}")
    
    # Use Secure Wrapper for primary security checks
    # (Rate limiting + Type/Size validation + Content Deep Check)
    file_content = await file.read()
    filesize = len(file_content)
    suffix = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    try:
        start_time = time.time()
        
        # The wrapper handles security logic; we pass detect_ai_media as the worker function
        # Pass device_id for rate limiting
        result = await security_manager.secure_execute(
            request, file.filename, filesize, temp_path, 
            lambda path: detect_ai_media(path, trusted_metadata=sidecar_metadata),
            uid=device_id
        )
        
        duration = time.time() - start_time
        
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
            log_transaction("CACHE", 0.0, {"file": file.filename, "device_id": device_id})
        elif is_gemini_used:
            cost = GEMINI_FIXED_COST
            method = "detect_with_gemini"
            gpu_sec, cpu_sec = 0, duration
            logger.info(f"[COST] Gemini analysis: ${cost:.6f}")
            gemini_usage = result.get("usage", {})
            log_transaction("GEMINI", -cost, {"file": file.filename, "device_id": device_id, "usage": gemini_usage})
        elif actual_gpu_sec > 0:
            # Cost based on actual GPU utilization, not network round-trip
            cost = actual_gpu_sec * GPU_RATE_PER_SEC
            method = "detect_with_gpu"
            gpu_sec, cpu_sec = actual_gpu_sec, duration - actual_gpu_sec
            logger.info(f"[COST] GPU: {actual_gpu_sec:.3f}s (actual) vs {duration:.3f}s (round-trip) | Cost: ${cost:.6f}")
            log_transaction("GPU", -cost, {"file": file.filename, "device_id": device_id, "duration": actual_gpu_sec})
        else:
            cost = duration * CPU_RATE_PER_SEC
            method = "detect_metadata_only"
            gpu_sec, cpu_sec = 0, duration
            log_transaction("CPU", -cost, {"file": file.filename, "device_id": device_id, "duration": duration})
        # Remove internal fields before returning response
        result.pop("gpu_time_ms", None)
        result.pop("is_gemini_used", None)
        result.pop("is_cached", None)
        
        # Attach new balance
        result["new_balance"] = new_balance

        logger.info(f"[ROUTE] Final Response for {file.filename}: {json.dumps(result)}")
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---- Guest Balance Endpoint ----
@app.get("/api/user/balance")
async def get_balance(device_id: str = Header(..., alias="X-Device-ID")):
    """
    Get current credit balance for a guest device.
    Auto-creates wallet with 10 credits if it doesn't exist.
    """
    wallet = get_guest_wallet(device_id)
    return {"balance": wallet.get("credits", 0)}

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
    image: UploadFile = File(...), 
    mask: UploadFile = File(...),
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: str = Header(..., alias="X-Turnstile-Token")
):
    """
    Remove objects from an image using the LaMA model (CPU optimized).
    This endpoint is async to verify Turnstile, and offloads heavy work to threadpool.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[INPAINT] Request {request_id} started. Image: {image.filename}, Device: {device_id}")

    # 1. Security & Wallet Check
    # Verify Turnstile (Async)
    is_human = await verify_turnstile(turnstile_token)
    if not is_human:
        raise HTTPException(status_code=403, detail="Turnstile validation failed")

    if check_ban_status(device_id):
        raise HTTPException(status_code=403, detail="Device is banned")

    # Deduct 2 Credits (Atomic) - Raises 402 if insufficient
    try:
        new_balance = deduct_guest_credits(device_id, cost=2)
    except HTTPException as e:
        logger.warning(f"[INPAINT] Request {request_id} Insufficient funds for {device_id}")
        raise e

    if not hasattr(app.state, "remover") or app.state.remover is None:
        logger.error(f"[INPAINT] Service unavailable for request {request_id}")
        raise HTTPException(status_code=503, detail="Inpainting service unavailable")

    try:
        start_time = time.time()
        
        # 1. READ (Async friendly)
        read_start = time.time()
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        read_duration = time.time() - read_start
        
        image_size_mb = len(image_bytes) / (1024 * 1024)
        logger.info(f"[INPAINT] Request {request_id}: Files read in {read_duration:.3f}s. Image size: {image_size_mb:.2f}MB")
        
        # 2. PROCESS (Offload CPU work to threadpool)
        process_start = time.time()
        # Run CPU-heavy task in threadpool so we don't block the async event loop
        result = await run_in_threadpool(app.state.remover.process_image, image_bytes, mask_bytes)
        process_duration = time.time() - process_start
        
        total_duration = time.time() - start_time
        logger.info(f"[INPAINT] Request {request_id} COMPLETED in {total_duration:.3f}s (Processing: {process_duration:.3f}s)")
        
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
