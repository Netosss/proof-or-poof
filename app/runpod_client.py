import runpod
import os
import asyncio
import base64
import logging
import io
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

async def run_deep_forensics(source: Union[str, Image.Image], width: int = 0, height: int = 0) -> float:
    """
    Offloads forensic scan to RunPod. Now supports direct PIL Images.
    """
    config = get_config()
    if not config["endpoint_id"]:
        return 0.0

    try:
        runpod.api_key = config["api_key"]
        endpoint = runpod.Endpoint(config["endpoint_id"])
        
        # In-memory optimization (No Disk!)
        image_base64, w, h = optimize_image(source, max_size=512)
        
        # Priority: use passed dimensions, else use detected
        final_w = width if width > 0 else w
        final_h = height if height > 0 else h

        job_result = endpoint.run_sync({
            "image": image_base64,
            "original_width": final_w,
            "original_height": final_h,
            "task": "deep_forensic"
        }, timeout=90)

        if job_result and "ai_score" in job_result:
            return float(job_result["ai_score"])
        
        return 0.0
    except Exception as e:
        logger.error(f"RunPod Call Failed: {e}")
        return 0.0

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
