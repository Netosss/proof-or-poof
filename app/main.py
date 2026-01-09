import os
import tempfile
import logging
import csv
import time
import uuid
import hashlib
import json
import pandas as pd
import plotly.express as px
import asyncio
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load environment variables at the very beginning
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from new modular structure
from app.detectors import detect_ai_media
from app.schemas import DetectionResponse
from app.runpod_client import run_video_removal, pending_jobs, webhook_result_buffer, cleanup_stale_jobs
from app.security import security_manager
from contextlib import asynccontextmanager

# Webhook authentication secret (set via environment variable)
RUNPOD_WEBHOOK_SECRET = os.getenv("RUNPOD_WEBHOOK_SECRET", "")

# Dashboard authentication (optional but recommended)
DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET", "")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory="app/templates")

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

# ---- RunPod Webhook Storage ----
# This is imported from runpod_client and shared

USAGE_LOG = "usage_log.csv"

# ---- Pricing Constants (USD per second) ----
GPU_RATE_PER_SEC = 0.0019  # RunPod A5000/L4 rate
CPU_RATE_PER_SEC = 0.0001  # Estimated Railway CPU rate

def log_usage(filename: str, filesize: int, method: str, cost_usd: float, gpu_seconds: float = 0, cpu_seconds: float = 0):
    """Append a row to the usage log with a unique ID and hashed filename."""
    file_exists = os.path.exists(USAGE_LOG)
    fieldnames = ["timestamp", "request_id", "filename", "filesize", "method", "cost_usd", "gpu_seconds", "cpu_seconds"]
    
    # Hash filename for privacy
    hashed_filename = hashlib.sha256(filename.encode()).hexdigest()[:8]
    
    with open(USAGE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.time(),
            "request_id": str(uuid.uuid4())[:8], # Short unique ID
            "filename": f"file_{hashed_filename}",
            "filesize": filesize,
            "method": method,
            "cost_usd": cost_usd,
            "gpu_seconds": gpu_seconds,
            "cpu_seconds": cpu_seconds
        })

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
        if RUNPOD_WEBHOOK_SECRET:
            signature = (
                request.headers.get("X-Runpod-Signature", "") or
                request.headers.get("Authorization", "").replace("Bearer ", "") or
                request.headers.get("X-Webhook-Secret", "")
            )
            
            if signature != RUNPOD_WEBHOOK_SECRET:
                logger.warning(f"[WEBHOOK] Signature mismatch.")
                # For now, log but don't block
        
        payload = await request.json()
        job_id = payload.get("id")
        status = payload.get("status")
        output = payload.get("output")
        
        logger.info(f"[WEBHOOK] Received callback for job {job_id}, status: {status}")
        
        if job_id and job_id in pending_jobs:
            future, start_time = pending_jobs[job_id]
            
            if status == "COMPLETED" and output:
                if not isinstance(output, dict):
                    output = {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": "Invalid output format"}
                
                if not future.done():
                    future.set_result(output)
            elif status == "FAILED":
                error_output = {
                    "error": "Job failed", 
                    "details": payload.get("error"),
                    "ai_score": 0.0,
                    "gpu_time_ms": 0.0
                }
                if not future.done():
                    future.set_result(error_output)
            else:
                return {"status": "acknowledged"}
        elif job_id and status == "COMPLETED" and output:
            if isinstance(output, dict) and "ai_score" in output:
                webhook_result_buffer[job_id] = (output, time.time())
                logger.info(f"[WEBHOOK] Job {job_id} buffered")
            else:
                logger.warning(f"[WEBHOOK] Job {job_id} has invalid output, not buffering")
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[WEBHOOK] Error processing callback: {e}")
        return {"status": "error", "message": str(e)}

# ---- Dashboard endpoint ----
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, secret: str = ""):
    # ---- Authentication (optional) ----
    if DASHBOARD_SECRET:
        provided_secret = secret or request.headers.get("X-Dashboard-Secret", "")
        if provided_secret != DASHBOARD_SECRET:
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "error": "Access Denied. Add ?secret=YOUR_SECRET to the URL."
            }, status_code=401)
    
    # 1. Check if log exists
    if not os.path.exists(USAGE_LOG):
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "is_empty": True
        })

    try:
        # 2. Read and Validate Data
        df = pd.read_csv(USAGE_LOG)
        if df.empty:
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "is_empty": True
            })
            
        # Ensure timestamp is numeric
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Pre-processing
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date_label'] = df['datetime'].dt.strftime('%H:%M:%S')
        df = df.sort_values('timestamp')

        # 3. Create Scatter Plot
        fig = px.scatter(
            df, x="datetime", y="cost_usd", color="method",
            text="request_id",
            hover_data={
                "datetime": "|%Y-%m-%d %H:%M:%S",
                "cost_usd": ":$.4f",
                "filename": True,
                "filesize": True,
                "gpu_seconds": ":.2f",
                "cpu_seconds": ":.2f"
            },
            title="Operational Spending (USD)",
            template="plotly_dark"
        )

        fig.update_traces(
            marker=dict(size=14, line=dict(width=2, color='white')),
            textposition='top center',
            mode='markers+text'
        )
        
        fig.update_layout(
            paper_bgcolor="#111217",
            plot_bgcolor="#111217",
            font_color="#d8d9da",
            xaxis=dict(gridcolor="#24272e", title="Time"),
            yaxis=dict(gridcolor="#24272e", title="Cost", tickformat="$.4f"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#24272e", borderwidth=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Convert DF to list of dicts for Jinja2
        rows = df.tail(10).iloc[::-1].to_dict(orient="records")
        
        total_cost = df['cost_usd'].sum()
        total_requests = len(df)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "is_empty": False,
            "plot_html": plot_html,
            "rows": rows,
            "total_cost": total_cost,
            "total_requests": total_requests
        })
    except Exception as e:
        logger.error(f"Dashboard render error: {e}", exc_info=True)
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "error": f"Could not load usage data: {str(e)}"
        })

# ---- WebSocket endpoint ----
connected_clients = []

@app.websocket("/ws/usage")
async def ws_usage(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(5)
            if os.path.exists(USAGE_LOG):
                df = pd.read_csv(USAGE_LOG)
                data = df.to_dict(orient="records")
                for client in connected_clients:
                    try:
                        await client.send_json(data)
                    except:
                        pass
    except:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

# ---- Detect endpoint ----
@app.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request, 
    file: UploadFile = File(...),
    trusted_metadata: Optional[str] = Form(None),
    # Capture-time sidecar fields (sent by mobile app)
    captured_in_app: Optional[bool] = Form(False),
    capture_session_id: Optional[str] = Form(None),
    capture_timestamp_ms: Optional[int] = Form(None),
    capture_path: Optional[str] = Form(None),
    capture_signature: Optional[str] = Form(None),
):
    """
    Detect AI-generated content in images/videos.
    """
    # Parse trusted metadata sidecar if provided
    sidecar_metadata = None
    if trusted_metadata:
        try:
            sidecar_metadata = json.loads(trusted_metadata)
            logger.info(f"[SIDECAR] Received trusted metadata: {list(sidecar_metadata.keys())}")
        except json.JSONDecodeError as e:
            logger.warning(f"[SIDECAR] Invalid JSON in trusted_metadata: {e}")

    # Merge explicit capture-time fields into sidecar_metadata (so detector has a single dict)
    if sidecar_metadata is None:
        sidecar_metadata = {}
    if captured_in_app:
        sidecar_metadata["captured_in_app"] = True
    if capture_session_id:
        sidecar_metadata["capture_session_id"] = capture_session_id
    if capture_timestamp_ms:
        sidecar_metadata["capture_timestamp_ms"] = int(capture_timestamp_ms)
    if capture_path:
        sidecar_metadata["capture_path"] = capture_path
    if capture_signature:
        sidecar_metadata["capture_signature"] = capture_signature
    
    file_content = await file.read()
    filesize = len(file_content)
    suffix = os.path.splitext(file.filename)[1].lower()

    # [LOGGING] Request Input
    log_meta_keys = list(sidecar_metadata.keys()) if sidecar_metadata else []
    logger.info(f"[REQUEST] Processing: {file.filename} | Size: {filesize} bytes | Metadata: {log_meta_keys}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    try:
        start_time = time.time()
        
        # The wrapper handles security logic; we pass detect_ai_media as the worker function
        result = await security_manager.secure_execute(
            request, file.filename, filesize, temp_path, 
            lambda path: detect_ai_media(path, trusted_metadata=sidecar_metadata, original_filename=file.filename)
        )
        
        duration = time.time() - start_time
        gpu_used = result.get("layers", {}).get("layer2_forensics", {}).get("status") != "skipped"
        
        actual_gpu_time_ms = result.get("gpu_time_ms", 0.0)
        actual_gpu_sec = actual_gpu_time_ms / 1000.0
        
        if gpu_used and actual_gpu_sec > 0:
            cost = actual_gpu_sec * GPU_RATE_PER_SEC
            method = "detect_with_gpu"
            gpu_sec, cpu_sec = actual_gpu_sec, duration - actual_gpu_sec
            logger.info(f"[COST] GPU: {actual_gpu_sec:.3f}s (actual) | Cost: ${cost:.6f}")
        elif gpu_used:
            cost = duration * GPU_RATE_PER_SEC
            method = "detect_with_gpu"
            gpu_sec, cpu_sec = duration, 0
        else:
            cost = duration * CPU_RATE_PER_SEC
            method = "detect_metadata_only"
            gpu_sec, cpu_sec = 0, duration
            
        log_usage(file.filename, filesize, method, cost, gpu_seconds=gpu_sec, cpu_seconds=cpu_sec)

        # Keep gpu_time_ms in the API response (useful for clients + debugging/cost dashboards)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---- Watermark removal ----
@app.post("/remove-watermark")
async def remove_watermark(request: Request, file: UploadFile = File(...)):
    file_content = await file.read()
    filesize = len(file_content)
    suffix = os.path.splitext(file.filename)[1].lower()
    content_type = file.content_type or ""
    is_video = content_type.startswith("video/") or suffix in {".mp4", ".mov", ".avi"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    try:
        start_time = time.time()
        
        async def removal_worker(path):
            if is_video:
                return await run_video_removal(path)
            else:
                return {"status": "success", "method": "local_cheap", "filename": file.filename}

        result = await security_manager.secure_execute(
            request, file.filename, filesize, temp_path, removal_worker
        )
        
        duration = time.time() - start_time
        
        if is_video:
            cost = duration * GPU_RATE_PER_SEC
            log_usage(file.filename, filesize, "remove-watermark-video", cost, gpu_seconds=duration)
            return {
                "status": "success", "method": "runpod_gpu",
                "cost_usd": round(cost, 5), "gpu_seconds": round(duration, 2), "result": result,
            }
        else:
            cost = duration * CPU_RATE_PER_SEC
            log_usage(file.filename, filesize, "remove-watermark-image", cost, cpu_seconds=duration)
            return {
                "status": "success", "method": "local_cheap",
                "cost_usd": round(cost, 5), "filename": file.filename,
            }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
