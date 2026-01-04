import runpod
import os
import asyncio
import base64
import logging
import io
from PIL import Image
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_config():
    return {
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID")
    }

def optimize_image(image_path: str, max_size: int = 512) -> tuple:
    """
    Resizes image to max_size while keeping aspect ratio and converts to JPEG.
    Reduces 2MB files to ~50KB for fast RunPod transfer.
    Returns: (base64_string, original_width, original_height)
    """
    try:
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
            # Resize
            img.thumbnail((max_size, max_size))
            
            # Convert to RGB (drops alpha channel if any)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save to buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            
            # Encode
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.info(f"Image optimized: {len(encoded)//1024}KB (Max Dim: {max_size}px)")
            return encoded, orig_w, orig_h
    except Exception as e:
        logger.error(f"Failed to optimize image: {e}")
        # Fallback to original bytes if optimization fails
        with open(image_path, "rb") as f:
            data = f.read()
            # Still try to get size for fallback
            try:
                with Image.open(io.BytesIO(data)) as fimg:
                    return base64.b64encode(data).decode("utf-8"), fimg.width, fimg.height
            except:
                return base64.b64encode(data).decode("utf-8"), 0, 0

async def run_image_removal(image_path: str) -> Dict[str, Any]:
    """
    Calls RunPod Serverless synchronously for image cleansing.
    """
    config = get_config()
    if not config["endpoint_id"]:
        return {"error": "RUNPOD_ENDPOINT_ID not configured"}

    try:
        runpod.api_key = config["api_key"]
        endpoint = runpod.Endpoint(config["endpoint_id"])
        
        # Optimize image for transfer
        image_base64, w, h = optimize_image(image_path)

        # Run synchronously
        job_result = endpoint.run_sync({
            "image": image_base64,
            "orig_w": w,
            "orig_h": h,
            "task": "image_removal"
        }, timeout=60)

        return job_result
    except Exception as e:
        return {"error": str(e)}

async def run_deep_forensics(image_path: str) -> float:
    """
    Offloads the expensive SigLIP forensic scan to a RunPod GPU worker.
    Returns the AI score (0.0 to 1.0).
    """
    config = get_config()
    if not config["endpoint_id"]:
        logger.error("RUNPOD_ENDPOINT_ID not configured")
        return 0.0

    try:
        runpod.api_key = config["api_key"]
        endpoint = runpod.Endpoint(config["endpoint_id"])
        
        # Optimize image for transfer (SigLIP only needs 224px, 512px is plenty)
        image_base64, w, h = optimize_image(image_path, max_size=512)

        # Ensure API key is set just before the call
        runpod.api_key = config["api_key"]
        
        logger.info(f"Calling RunPod {config['endpoint_id']} for deep_forensic...")
        job_result = endpoint.run_sync({
            "image": image_base64,
            "orig_w": w,
            "orig_h": h,
            "task": "deep_forensic"
        }, timeout=90)

        if job_result is None:
            logger.error("RunPod returned None. Check credits/API key.")
            return 0.0

        if "ai_score" in job_result:
            score = float(job_result["ai_score"])
            logger.info(f"RunPod result: ai_score={score}")
            return score
        
        logger.error(f"RunPod Error response: {job_result}")
        return 0.0
    except Exception as e:
        logger.error(f"Failed to call RunPod: {e}", exc_info=True)
        return 0.0

async def run_video_removal(video_path: str) -> Dict[str, Any]:
    """
    Calls RunPod Serverless asynchronously and polls for video cleansing.
    """
    config = get_config()
    if not config["endpoint_id"]:
        return {"error": "RUNPOD_ENDPOINT_ID not configured"}

    try:
        runpod.api_key = config["api_key"]
        endpoint = runpod.Endpoint(config["endpoint_id"])
        
        # ⚠️ CRITICAL WARNING: For production, do NOT use base64 for videos.
        # Upload to S3/GCS first and send the 'video_url' instead.
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

        # Start the job asynchronously
        job = endpoint.run({
            "video": video_base64,
            "task": "video_removal"
        })

        # Poll for completion
        while True:
            status = job.status()
            if status == "COMPLETED":
                return job.output()
            elif status == "FAILED":
                return {"error": "RunPod job failed"}
            
            # Wait before polling again
            await asyncio.sleep(1)
            
    except Exception as e:
        return {"error": str(e)}
