import os
import shutil
import tempfile
import logging
import csv
import time
import uuid
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, Request
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
from app.security import security_manager

app = FastAPI(title="AI Provenance & Cleansing API")

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

# ---- Dashboard endpoint ----
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if not os.path.exists(USAGE_LOG):
        return """
        <body style="background-color: #111217; color: white; font-family: sans-serif; padding: 50px; text-align: center;">
            <h2>No usage data yet. Start scanning images! 🚀</h2>
        </body>
        """

    try:
        df = pd.read_csv(USAGE_LOG)
    except:
        return "<body style='background-color: #111217; color: white;'><h2>Log file error.</h2></body>"
        
    if df.empty:
        return "<body style='background-color: #111217; color: white; text-align: center;'><h2>No data records found.</h2></body>"
    
    # Pre-processing
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date_label'] = df['datetime'].dt.strftime('%H:%M:%S')
    df = df.sort_values('timestamp')

    # Create Grafana-like Scatter Plot
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

    # Style the graph to look like Grafana
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

    # Build HTML table for "Logs" look
    table_rows = ""
    for _, row in df.tail(10).iloc[::-1].iterrows(): # Last 10, newest first
        table_rows += f"""
        <tr style="border-bottom: 1px solid #24272e;">
            <td style="padding: 10px; color: #32d1df;">{row['request_id']}</td>
            <td style="padding: 10px;">{row['date_label']}</td>
            <td style="padding: 10px;">{row['filename']}</td>
            <td style="padding: 10px;">{row['method']}</td>
            <td style="padding: 10px; color: #73bf69;">${row['cost_usd']:.4f}</td>
            <td style="padding: 10px;">{row['gpu_seconds']:.2f}s</td>
        </tr>
        """

    html = f"""
    <html>
        <head>
            <meta http-equiv="refresh" content="10">
            <title>AI Ops Dashboard</title>
            <style>
                body {{ font-family: 'Inter', sans-serif; background-color: #111217; color: #d8d9da; margin: 0; padding: 20px; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #24272e; padding-bottom: 10px; margin-bottom: 20px; }}
                .card {{ background-color: #181b1f; border: 1px solid #24272e; border-radius: 4px; padding: 20px; margin-bottom: 20px; }}
                .stat-box {{ display: flex; gap: 20px; }}
                .stat {{ background: #21262d; padding: 10px 20px; border-radius: 4px; border-left: 4px solid #32d1df; }}
                .stat-val {{ font-size: 1.5em; font-weight: bold; color: #ffffff; }}
                .stat-label {{ font-size: 0.8em; color: #8e8e8e; text-transform: uppercase; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
                th {{ text-align: left; background: #21262d; padding: 10px; color: #8e8e8e; text-transform: uppercase; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div style="font-size: 1.2em; font-weight: bold;">AI OPS / <span style="color: #32d1df;">USAGE_TRACKER</span></div>
                <div class="stat-box">
                    <div class="stat">
                        <div class="stat-label">Total Burn</div>
                        <div class="stat-val">${df['cost_usd'].sum():.4f}</div>
                    </div>
                    <div class="stat" style="border-left-color: #73bf69;">
                        <div class="stat-label">Requests</div>
                        <div class="stat-val">{len(df)}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                {fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>

            <div class="card">
                <div style="font-weight: bold; margin-bottom: 15px; color: #32d1df;">LIVE REQUEST LOGS</div>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th><th>Time</th><th>File</th><th>Method</th><th>Cost</th><th>GPU Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
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
async def detect(request: Request, file: UploadFile = File(...)):
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
        result = await security_manager.secure_execute(
            request, file.filename, filesize, temp_path, detect_ai_media
        )
        
        duration = time.time() - start_time
        gpu_used = result.get("layers", {}).get("layer2_forensics", {}).get("status") != "skipped"
        
        if gpu_used:
            cost = duration * GPU_RATE_PER_SEC
            method = "detect_with_gpu"
            gpu_sec, cpu_sec = duration, 0
        else:
            cost = duration * CPU_RATE_PER_SEC
            method = "detect_metadata_only"
            gpu_sec, cpu_sec = 0, duration
            
        log_usage(file.filename, filesize, method, cost, gpu_seconds=gpu_sec, cpu_seconds=cpu_sec)
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
        # The wrapper might have already raised an HTTPException, but if not:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
