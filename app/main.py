import os
import shutil
import tempfile
import logging
import csv
import time
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import plotly.express as px
import asyncio

# Load environment variables at the very beginning
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.detector import detect_ai_media
from app.schemas import DetectionResponse
from app.runpod_client import run_image_removal, run_video_removal

app = FastAPI(title="AI Provenance & Cleansing API")

USAGE_LOG = "usage_log.csv"

def log_usage(filename: str, filesize: int, method: str, cost_usd: float, gpu_seconds: float = 0, cpu_seconds: float = 0):
    """Append a row to the usage log."""
    file_exists = os.path.exists(USAGE_LOG)
    fieldnames = ["timestamp", "filename", "filesize", "method", "cost_usd", "gpu_seconds", "cpu_seconds"]
    
    with open(USAGE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.time(),
            "filename": filename,
            "filesize": filesize,
            "method": method,
            "cost_usd": cost_usd,
            "gpu_seconds": gpu_seconds,
            "cpu_seconds": cpu_seconds
        })

# ---- CORS (lock this down later) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Healthcheck (Railway uses this) ----
@app.get("/health")
async def health():
    return {"status": "healthy"}

# ---- Dashboard endpoint ----
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if not os.path.exists(USAGE_LOG):
        return "<h2>No usage data yet</h2>"

    try:
        df = pd.read_csv(USAGE_LOG)
    except:
        return "<h2>No usage data yet (CSV empty)</h2>"
        
    if df.empty:
        return "<h2>No usage data yet</h2>"
        
    fig = px.bar(df, x="filename", y="cost_usd", color="method",
                 hover_data=["filesize", "timestamp", "gpu_seconds", "cpu_seconds"],
                 title="API Usage Costs (USD)")
    
    html = f"""
    <html>
        <head>
            <meta http-equiv="refresh" content="10">  <!-- Auto-refresh -->
            <title>Usage Dashboard</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
            </style>
        </head>
        <body>
            <h2>Live Usage Dashboard</h2>
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
            <br>
            <p>Total Estimated Cost: ${round(df['cost_usd'].sum(), 4)}</p>
        </body>
    </html>
    """
    return HTMLResponse(html)

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
async def detect(file: UploadFile = File(...)):
    """
    Cheap detection only (C2PA metadata).
    No GPU, no RunPod.
    """
    logger.info(f"--- Incoming detection request: {file.filename} ---")
    suffix = os.path.splitext(file.filename)[1].lower()
    
    # Read file content
    file_content = await file.read()
    filesize = len(file_content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    try:
        start_time = time.time()
        result = await detect_ai_media(temp_path)
        end_time = time.time()
        
        cpu_seconds = end_time - start_time
        cpu_cost = cpu_seconds * 0.0001 # Rough estimate for Railway CPU
        
        logger.info(f"Result for {file.filename}: {result}")
        
        # Log usage
        log_usage(file.filename, filesize, "detect", cpu_cost, cpu_seconds=cpu_seconds)
        
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---- Watermark removal ----
@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    """
    Images → local cheap placeholder
    Videos → RunPod GPU
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    content_type = file.content_type or ""
    is_video = content_type.startswith("video/") or suffix in {".mp4", ".mov", ".avi"}

    # Read file content
    file_content = await file.read()
    filesize = len(file_content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    try:
        if is_video:
            start_time = time.time()
            result = await run_video_removal(temp_path)
            end_time = time.time()
            
            gpu_seconds = end_time - start_time
            gpu_cost = gpu_seconds * 0.0019 # USD based on $0.0019/sec
            
            log_usage(file.filename, filesize, "remove-watermark-video", gpu_cost, gpu_seconds=gpu_seconds)
            
            return {
                "status": "success",
                "method": "runpod_gpu",
                "cost_usd": round(gpu_cost, 5),
                "gpu_seconds": round(gpu_seconds, 2),
                "result": result,
            }

        # Local cheap for images
        cost = 0.0
        log_usage(file.filename, filesize, "remove-watermark-image", cost)
        return {
            "status": "success",
            "method": "local_cheap",
            "cost_usd": cost,
            "filename": file.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
