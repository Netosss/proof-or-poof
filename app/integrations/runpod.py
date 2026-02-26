"""
RunPod GPU worker integration.

Supports both webhook-driven (fast) and polling (fallback) modes.

Note: `pending_jobs` and `webhook_result_buffer` are in-memory only.
      NOT safe for multi-instance deployments — use Redis for scale-out.
"""

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

# In-memory job tracking — {job_id: (asyncio.Future, start_time)}
pending_jobs: Dict[str, tuple] = {}

# Buffers webhook results arriving before pending_jobs is registered (race-condition fix)
# {job_id: (result, timestamp)}
webhook_result_buffer: Dict[str, tuple] = {}

PENDING_JOB_TTL = 120
WEBHOOK_BUFFER_TTL = 30


def cleanup_stale_jobs():
    """Remove stale pending jobs and buffered results to prevent memory leaks."""
    now = time.time()

    stale_jobs = [
        job_id for job_id, (_, start_time) in pending_jobs.items()
        if now - start_time > PENDING_JOB_TTL
    ]
    for job_id in stale_jobs:
        future, _ = pending_jobs.pop(job_id, (None, None))
        if future and not future.done():
            future.set_exception(TimeoutError(f"Job {job_id} expired after {PENDING_JOB_TTL}s"))
        logger.warning(f"[CLEANUP] Removed stale pending job: {job_id}")

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
        "webhook_url": os.getenv("RUNPOD_WEBHOOK_URL"),
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
        if max(orig_w, orig_h) > max_size:
            img.thumbnail((max_size, max_size))

        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)

        if isinstance(source, str):
            img.close()

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, orig_w, orig_h
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return "", 0, 0


async def _run_with_webhook(
    endpoint, payload: dict, webhook_url: str, timeout_seconds: int = 90
) -> Dict[str, Any]:
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

    cleanup_stale_jobs()

    request_payload = {
        "input": payload,
        "webhook": webhook_url
    }

    config = get_config()
    endpoint_id = config["endpoint_id"]
    api_key = config["api_key"]

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

    pending_jobs[job_id] = (future, start_time)

    if job_id in webhook_result_buffer:
        result, _ = webhook_result_buffer.pop(job_id)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[WEBHOOK] Job {job_id} found in buffer (arrived early) in {elapsed_ms:.2f}ms")
        return result

    try:
        result = await _wait_with_buffer_check(future, job_id, timeout_seconds)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[WEBHOOK] Job {job_id} completed via webhook in {elapsed_ms:.2f}ms")
        return result
    except asyncio.TimeoutError:
        logger.error(f"[WEBHOOK] Job {job_id} timed out after {timeout_seconds}s, falling back to status check")
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
        pending_jobs.pop(job_id, None)
        webhook_result_buffer.pop(job_id, None)


async def _wait_with_buffer_check(future: asyncio.Future, job_id: str, timeout_seconds: int):
    """
    Wait for future with periodic buffer checks to handle race conditions.
    Checks buffer every 100ms in case webhook arrived but future wasn't set.
    """
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if future.done():
            return future.result()

        if job_id in webhook_result_buffer:
            result, _ = webhook_result_buffer.pop(job_id)
            logger.info(f"[WEBHOOK] Job {job_id} found in buffer during wait")
            return result

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=0.1)
        except asyncio.TimeoutError:
            continue

    raise asyncio.TimeoutError(f"Job {job_id} timed out")


async def _run_with_polling(endpoint, payload: dict, timeout_seconds: int = 90) -> Dict[str, Any]:
    """Fallback polling mode when webhooks are not configured."""
    t_api = time.perf_counter()
    job = endpoint.run(payload)
    job_id = job.job_id
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
        await asyncio.sleep(0.1)


async def run_gpu_inpainting(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    """
    Run inpainting on RunPod GPU worker.
    Returns: processed image bytes (PNG).
    """
    config = get_config()
    if not config["endpoint_id"]:
        raise ValueError("RunPod configuration missing (RUNPOD_ENDPOINT_ID)")

    try:
        runpod.api_key = config["api_key"]
        endpoint = runpod.Endpoint(config["endpoint_id"])

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")

        payload = {
            "image": img_b64,
            "mask": mask_b64
        }

        webhook_url = config.get("webhook_url")
        timeout = 180  # longer timeout for potential cold start

        if webhook_url:
            job_result = await _run_with_webhook(endpoint, payload, webhook_url, timeout_seconds=timeout)
        else:
            job_result = await _run_with_polling(endpoint, payload, timeout_seconds=timeout)

        if not job_result:
            raise RuntimeError("Empty response from worker")

        if "error" in job_result:
            raise RuntimeError(f"Worker error: {job_result['error']}")

        if "image_base64" not in job_result:
            if "results" in job_result:
                first = job_result["results"][0]
                if "image_base64" in first:
                    return base64.b64decode(first["image_base64"])

            logger.error(f"[RUNPOD] Invalid response keys: {job_result.keys()}")
            raise RuntimeError("Worker did not return image_base64")

        return base64.b64decode(job_result["image_base64"])

    except Exception as e:
        logger.error(f"[RUNPOD] Inpainting failed: {e}")
        raise e
