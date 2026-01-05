import runpod
import os
import asyncio
import base64
import logging
import io
import time
from PIL import Image
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def get_config():
    return {
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID")
    }

def optimize_image(source: Union[str, Image.Image], max_size: int = 512) -> tuple:
    """
    Optimizes image for transfer. Accepts file path or PIL Image.
    Returns: (base64_string, width, height)
    """
    try:
        if isinstance(source, str):
            img = Image.open(source)
        else:
            img = source

        orig_w, orig_h = img.size
        # Resize if larger than max_size
        if max(orig_w, orig_h) > max_size:
            img.thumbnail((max_size, max_size))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        
        # Close handle if we opened it from path
        if isinstance(source, str):
            img.close()

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, orig_w, orig_h
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return "", 0, 0

async def run_deep_forensics(source: Union[str, Image.Image], width: int = 0, height: int = 0) -> Dict[str, Any]:
    """
    Offloads forensic scan to RunPod. Now supports direct PIL Images.
    Returns: dict with 'ai_score' and 'gpu_time_ms' for accurate cost calculation.
    """
    config = get_config()
    if not config["endpoint_id"]:
        return {"ai_score": 0.0, "gpu_time_ms": 0.0}

    try:
        total_start = time.perf_counter()
        
        runpod.api_key = config["api_key"]
        endpoint = runpod.Endpoint(config["endpoint_id"])
        
        # In-memory optimization (No Disk!)
        t_opt = time.perf_counter()
        image_base64, w, h = optimize_image(source, max_size=512)
        opt_time_ms = (time.perf_counter() - t_opt) * 1000
        payload_size_kb = len(image_base64) / 1024
        logger.info(f"[TIMING] Image optimization: {opt_time_ms:.2f}ms | Payload: {payload_size_kb:.1f}KB")
        
        # Priority: use passed dimensions, else use detected
        final_w = width if width > 0 else w
        final_h = height if height > 0 else h

        # RunPod API call with fast async polling
        t_api = time.perf_counter()
        job = endpoint.run({
            "image": image_base64,
            "original_width": final_w,
            "original_height": final_h,
            "task": "deep_forensic"
        })
        
        # Fast polling (100ms intervals instead of default 3s)
        poll_count = 0
        timeout_seconds = 90
        while True:
            status = job.status()
            poll_count += 1
            if status == "COMPLETED":
                job_result = job.output()
                break
            if status == "FAILED":
                logger.error(f"RunPod job failed after {poll_count} polls")
                return {"ai_score": 0.0, "gpu_time_ms": 0.0}
            if (time.perf_counter() - t_api) > timeout_seconds:
                logger.error(f"RunPod job timed out after {timeout_seconds}s")
                return {"ai_score": 0.0, "gpu_time_ms": 0.0}
            await asyncio.sleep(0.1)  # 100ms polling - much faster than default 3s
        
        api_time_ms = (time.perf_counter() - t_api) * 1000
        total_time_ms = (time.perf_counter() - total_start) * 1000
        logger.info(f"[TIMING] RunPod API call: {api_time_ms:.2f}ms ({poll_count} polls) | Total: {total_time_ms:.2f}ms")
        
        # Extract actual GPU time from worker response
        gpu_time_ms = 0.0
        if job_result and "timing_ms" in job_result:
            worker_timing = job_result["timing_ms"]
            gpu_time_ms = worker_timing.get("total", 0.0)
            logger.info(f"[TIMING] Worker breakdown: {worker_timing}")
            overhead_ms = api_time_ms - gpu_time_ms
            logger.info(f"[TIMING] Network/Queue overhead: {overhead_ms:.2f}ms")

        ai_score = float(job_result.get("ai_score", 0.0)) if job_result else 0.0
        
        return {
            "ai_score": ai_score,
            "gpu_time_ms": gpu_time_ms,
            "model_score": job_result.get("model_score", ai_score) if job_result else ai_score
        }
    except Exception as e:
        logger.error(f"RunPod Call Failed: {e}")
        return {"ai_score": 0.0, "gpu_time_ms": 0.0}

# ... (rest of the file remains the same) ...
async def run_image_removal(image_path: str) -> Dict[str, Any]:
    config = get_config()
    if not config["endpoint_id"]: return {"error": "Missing endpoint"}
    runpod.api_key = config["api_key"]
    endpoint = runpod.Endpoint(config["endpoint_id"])
    image_base64, w, h = optimize_image(image_path)
    return endpoint.run_sync({"image": image_base64, "task": "image_removal"}, timeout=60)

async def run_video_removal(video_path: str) -> Dict[str, Any]:
    config = get_config()
    if not config["endpoint_id"]: return {"error": "Missing endpoint"}
    runpod.api_key = config["api_key"]
    endpoint = runpod.Endpoint(config["endpoint_id"])
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    job = endpoint.run({"video": video_base64, "task": "video_removal"})
    while True:
        status = job.status()
        if status == "COMPLETED": return job.output()
        if status == "FAILED": return {"error": "Job failed"}
        await asyncio.sleep(1)
