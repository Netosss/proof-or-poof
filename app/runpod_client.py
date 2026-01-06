import runpod
import os
import asyncio
import base64
import logging
import io
import time
from PIL import Image
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

# ---- Pending Jobs for Webhook Support ----
# Format: {job_id: (asyncio.Future, start_time)}
# ⚠️ WARNING: This is in-memory and NOT safe for multi-instance deployments!
# For multi-replica setups (K8s, Railway scale-out), use Redis/Upstash instead.
pending_jobs: Dict[str, tuple] = {}

# Buffer for webhook results that arrive before pending_jobs is set (race condition fix)
# Format: {job_id: (result, timestamp)}
webhook_result_buffer: Dict[str, tuple] = {}

# TTL for pending jobs (seconds) - prevents memory leaks
PENDING_JOB_TTL = 120  # 2 minutes
WEBHOOK_BUFFER_TTL = 30  # 30 seconds for buffered results


def cleanup_stale_jobs():
    """Remove stale pending jobs and buffered results to prevent memory leaks."""
    now = time.time()
    
    # Cleanup pending jobs
    stale_jobs = [
        job_id for job_id, (_, start_time) in pending_jobs.items()
        if now - start_time > PENDING_JOB_TTL
    ]
    for job_id in stale_jobs:
        future, _ = pending_jobs.pop(job_id, (None, None))
        if future and not future.done():
            future.set_exception(TimeoutError(f"Job {job_id} expired after {PENDING_JOB_TTL}s"))
        logger.warning(f"[CLEANUP] Removed stale pending job: {job_id}")
    
    # Cleanup webhook buffer
    stale_buffer = [
        job_id for job_id, (_, timestamp) in webhook_result_buffer.items()
        if now - timestamp > WEBHOOK_BUFFER_TTL
    ]
    for job_id in stale_buffer:
        webhook_result_buffer.pop(job_id, None)
        logger.warning(f"[CLEANUP] Removed stale buffered result: {job_id}")

def get_config():
    return {
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID"),
        "webhook_url": os.getenv("RUNPOD_WEBHOOK_URL"),  # e.g., https://your-api.com/webhook/runpod
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
    Offloads forensic scan to RunPod. Supports webhooks (fast) or polling (fallback).
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

        payload = {
            "image": image_base64,
            "original_width": final_w,
            "original_height": final_h,
            "task": "deep_forensic"
        }

        t_api = time.perf_counter()
        webhook_url = config.get("webhook_url")
        
        if webhook_url:
            # ---- WEBHOOK MODE (Fast!) ----
            # Webhook URL goes inside the payload per RunPod docs:
            # https://docs.runpod.io/serverless/endpoints/send-requests#webhook-notifications
            job_result = await _run_with_webhook(endpoint, payload, webhook_url, timeout_seconds=90)
            mode = "webhook"
        else:
            # ---- POLLING MODE (Fallback) ----
            job_result = await _run_with_polling(endpoint, payload, timeout_seconds=90)
            mode = "polling"
        
        api_time_ms = (time.perf_counter() - t_api) * 1000
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        # Extract actual GPU time from worker response
        gpu_time_ms = 0.0
        if job_result and "timing_ms" in job_result:
            worker_timing = job_result["timing_ms"]
            gpu_time_ms = worker_timing.get("total", 0.0)
            logger.info(f"[TIMING] Worker breakdown: {worker_timing}")
            overhead_ms = api_time_ms - gpu_time_ms
            logger.info(f"[TIMING] Network overhead ({mode}): {overhead_ms:.2f}ms")

        logger.info(f"[TIMING] RunPod API call ({mode}): {api_time_ms:.2f}ms | Total: {total_time_ms:.2f}ms")

        # Check for errors in result
        if job_result and "error" in job_result:
            logger.warning(f"[RUNPOD] Job returned error: {job_result['error']}")
        
        ai_score = float(job_result.get("ai_score", 0.0)) if job_result else 0.0
        
        return {
            "ai_score": ai_score,
            "gpu_time_ms": gpu_time_ms,
            "model_score": job_result.get("model_score", ai_score) if job_result else ai_score,
            "error": job_result.get("error") if job_result else None
        }
    except Exception as e:
        error_msg = f"RunPod call failed: {str(e)}"
        logger.error(f"[RUNPOD] {error_msg}", exc_info=True)
        return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": error_msg}


def _batch_encode_images(frames: list) -> list:
    """CPU-bound: Encode frames to base64 (runs in thread pool)."""
    encoded = []
    for frame in frames:
        img = frame.copy()
        # Resize for efficiency (512px max)
        if max(img.size) > 512:
            img.thumbnail((512, 512))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return encoded


async def run_batch_forensics(frames: list) -> Dict[str, Any]:
    """
    Batch process multiple frames in a SINGLE RunPod request.
    Tri-Frame Strategy: GPU processes 3 images in ~same time as 1.
    Returns: dict with 'results' list and 'gpu_time_ms'.
    """
    config = get_config()
    if not config["endpoint_id"]:
        return {"results": [], "gpu_time_ms": 0.0, "error": "No endpoint configured"}
    
    if not frames:
        return {"results": [], "gpu_time_ms": 0.0}

    try:
        total_start = time.perf_counter()
        
        runpod.api_key = config["api_key"]
        endpoint = runpod.Endpoint(config["endpoint_id"])
        
        # Offload CPU-bound encoding to thread pool (avoids blocking event loop)
        t_opt = time.perf_counter()
        loop = asyncio.get_running_loop()
        images_b64 = await loop.run_in_executor(None, _batch_encode_images, frames)
        
        opt_time_ms = (time.perf_counter() - t_opt) * 1000
        total_payload_kb = sum(len(b) for b in images_b64) / 1024
        logger.info(f"[TIMING] Batch encode ({len(frames)} frames): {opt_time_ms:.2f}ms | Total: {total_payload_kb:.1f}KB")

        payload = {
            "images": images_b64,  # Batch list
            "task": "deep_forensic"
        }

        t_api = time.perf_counter()
        webhook_url = config.get("webhook_url")
        
        if webhook_url:
            job_result = await _run_with_webhook(endpoint, payload, webhook_url, timeout_seconds=90)
            mode = "webhook"
        else:
            job_result = await _run_with_polling(endpoint, payload, timeout_seconds=90)
            mode = "polling"
        
        api_time_ms = (time.perf_counter() - t_api) * 1000
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        # Extract results
        gpu_time_ms = 0.0
        results = []
        
        if job_result:
            if "results" in job_result:
                # Batch response
                results = job_result["results"]
            elif "ai_score" in job_result:
                # Single result (fallback)
                results = [job_result]
            
            if "timing_ms" in job_result:
                gpu_time_ms = job_result["timing_ms"].get("total", 0.0)
                logger.info(f"[TIMING] Batch worker: {job_result['timing_ms']}")
        
        logger.info(f"[TIMING] Batch RunPod ({mode}): {api_time_ms:.2f}ms | Total: {total_time_ms:.2f}ms")
        
        return {
            "results": results,
            "gpu_time_ms": gpu_time_ms,
            "error": job_result.get("error") if job_result else None
        }
        
    except Exception as e:
        error_msg = f"Batch RunPod call failed: {str(e)}"
        logger.error(f"[RUNPOD] {error_msg}", exc_info=True)
        return {"results": [], "gpu_time_ms": 0.0, "error": error_msg}


async def _run_with_webhook(endpoint, payload: dict, webhook_url: str, timeout_seconds: int = 90) -> Dict[str, Any]:
    """
    Submit job with webhook and wait for callback. Much faster than polling!
    Includes buffer polling to handle race conditions.
    
    Uses raw HTTP request to include webhook in payload per RunPod docs:
    https://docs.runpod.io/serverless/endpoints/send-requests#webhook-notifications
    """
    import aiohttp
    
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    start_time = time.time()
    
    # Periodic cleanup to prevent memory leaks
    cleanup_stale_jobs()
    
    # Build the full request payload with webhook at top level
    request_payload = {
        "input": payload,
        "webhook": webhook_url
    }
    
    config = get_config()
    endpoint_id = config["endpoint_id"]
    api_key = config["api_key"]
    
    # Use raw HTTP request to include webhook (SDK doesn't support it directly)
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=request_payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"[WEBHOOK] RunPod API error: {response.status} - {error_text}")
                return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": f"API error: {response.status}"}
            
            result = await response.json()
            job_id = result.get("id")
    
    if not job_id:
        logger.error("[WEBHOOK] No job_id returned from RunPod")
        return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": "No job_id returned"}
    
    logger.info(f"[WEBHOOK] Submitted job {job_id}, waiting for callback...")
    
    # Store future so webhook handler can resolve it
    pending_jobs[job_id] = (future, start_time)
    
    # Check if webhook already arrived (race condition fix)
    if job_id in webhook_result_buffer:
        result, _ = webhook_result_buffer.pop(job_id)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[WEBHOOK] Job {job_id} found in buffer (arrived early) in {elapsed_ms:.2f}ms")
        return result
    
    try:
        # Wait for webhook with periodic buffer checks (race condition fix)
        result = await _wait_with_buffer_check(future, job_id, timeout_seconds)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[WEBHOOK] Job {job_id} completed via webhook in {elapsed_ms:.2f}ms")
        return result
    except asyncio.TimeoutError:
        logger.error(f"[WEBHOOK] Job {job_id} timed out after {timeout_seconds}s, falling back to status check")
        # Fallback: check job status via API
        try:
            status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(status_url, headers=headers) as response:
                    if response.status == 200:
                        status_result = await response.json()
                        if status_result.get("status") == "COMPLETED":
                            return status_result.get("output", {})
        except Exception as e:
            logger.error(f"[WEBHOOK] Fallback status check failed: {e}")
        return {"error": "Webhook timeout", "ai_score": 0.0, "gpu_time_ms": 0.0}
    finally:
        # Cleanup
        pending_jobs.pop(job_id, None)
        webhook_result_buffer.pop(job_id, None)


async def _wait_with_buffer_check(future: asyncio.Future, job_id: str, timeout_seconds: int):
    """
    Wait for future with periodic buffer checks to handle race conditions.
    Checks buffer every 100ms in case webhook arrived but future wasn't set.
    """
    deadline = time.time() + timeout_seconds
    
    while time.time() < deadline:
        # Check if future is resolved
        if future.done():
            return future.result()
        
        # Check buffer (race condition: webhook arrived before pending_jobs was checked)
        if job_id in webhook_result_buffer:
            result, _ = webhook_result_buffer.pop(job_id)
            logger.info(f"[WEBHOOK] Job {job_id} found in buffer during wait")
            return result
        
        # Wait a bit before next check
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=0.1)
        except asyncio.TimeoutError:
            continue  # Keep checking
    
    raise asyncio.TimeoutError(f"Job {job_id} timed out")


async def _run_with_polling(endpoint, payload: dict, timeout_seconds: int = 90) -> Dict[str, Any]:
    """
    Fallback polling mode when webhooks are not configured.
    """
    t_api = time.perf_counter()
    job = endpoint.run(payload)
    job_id = job.job_id
    
    # Fast polling (100ms intervals)
    poll_count = 0
    while True:
        status = job.status()
        poll_count += 1
        if status == "COMPLETED":
            job_result = job.output()
            logger.info(f"[POLLING] Completed after {poll_count} polls")
            return job_result
        if status == "FAILED":
            error_msg = f"RunPod job {job_id} failed after {poll_count} polls"
            logger.error(f"[POLLING] {error_msg}")
            return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": error_msg}
        if (time.perf_counter() - t_api) > timeout_seconds:
            error_msg = f"RunPod job {job_id} timed out after {timeout_seconds}s"
            logger.error(f"[POLLING] {error_msg}")
            return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": error_msg}
        await asyncio.sleep(0.1)  # 100ms polling

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
