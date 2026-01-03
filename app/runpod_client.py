import runpod
import os
import asyncio
import base64
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Configure RunPod
runpod.api_key = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

async def run_image_removal(image_path: str) -> Dict[str, Any]:
    """
    Calls RunPod Serverless synchronously for image cleansing.
    """
    if not ENDPOINT_ID:
        return {"error": "RUNPOD_ENDPOINT_ID not configured"}

    try:
        endpoint = runpod.Endpoint(ENDPOINT_ID)
        
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
    if not ENDPOINT_ID:
        print("Error: RUNPOD_ENDPOINT_ID not configured")
        return 0.0

    try:
        endpoint = runpod.Endpoint(ENDPOINT_ID)
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Use run_sync for a fast 1-2 second GPU scan
        job_result = endpoint.run_sync({
            "image": image_base64,
            "task": "deep_forensic"
        }, timeout=30)

        if "ai_score" in job_result:
            return float(job_result["ai_score"])
        
        print(f"RunPod Error: {job_result.get('error', 'Unknown error')}")
        return 0.0
    except Exception as e:
        print(f"Failed to call RunPod: {e}")
        return 0.0

async def run_video_removal(video_path: str) -> Dict[str, Any]:
    """
    Calls RunPod Serverless asynchronously and polls for video cleansing.
    """
    if not ENDPOINT_ID:
        return {"error": "RUNPOD_ENDPOINT_ID not configured"}

    try:
        endpoint = runpod.Endpoint(ENDPOINT_ID)
        
        # In a real scenario, you'd likely upload the video to S3 and send the URL
        # For this POC, we'll assume the worker can handle the size or we send a placeholder
        # For now, let's assume we send a path or small video as base64
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


