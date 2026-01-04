import runpod
import os
import asyncio
import base64
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_config():
    return {
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID")
    }

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
        
        # Read image and convert to base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Run synchronously (blocks until job is done or timeout)
        # run_sync handles the wait internally
        job_result = endpoint.run_sync({
            "image": image_base64,
            "task": "image_removal"
        }, timeout=60) # 60s timeout to allow for warm-up

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
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Use run_sync for a fast 1-2 second GPU scan
        logger.info(f"Calling RunPod endpoint {config['endpoint_id']} for deep_forensic task...")
        
        # Ensure API key is set just before the call
        runpod.api_key = config["api_key"]
        
        job_result = endpoint.run_sync({
            "image": image_base64,
            "task": "deep_forensic"
        }, timeout=90)

        if job_result is None:
            logger.error("RunPod returned None. Check if RUNPOD_API_KEY is valid and has credits.")
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
