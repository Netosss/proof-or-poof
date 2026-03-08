"""
RunPod GPU worker integration.

Supports webhook-driven (fast, default) and polling (fallback) modes.

Webhook mode uses Redis Pub/Sub for job delivery:
  - Route subscribes to runpod:result:{job_id} before submitting
  - Webhook handler writes result to key + publishes to channel
  - Subscriber wakes immediately — zero polling, 5 Redis ops per job total
  - Race condition handled by a GET check after subscribing

Job state lives in Redis, not in memory — safe across Railway redeploys
and horizontally scaled instances.
"""

import runpod
import os
import asyncio
import base64
import json
import logging
import io
import time
from PIL import Image
from typing import Dict, Any, Union

from app.config import settings
from app.integrations import redis_client as redis_module
from app.integrations import http_client as http_module

logger = logging.getLogger(__name__)


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
        logger.error("runpod_optimization_failed", extra={
            "action": "runpod_optimization_failed",
            "error": str(e),
        })
        return "", 0, 0


async def _run_with_webhook(
    endpoint, payload: dict, webhook_url: str, timeout_seconds: int = 90
) -> Dict[str, Any]:
    """
    Submit job with webhook and wait for result via Redis Pub/Sub.

    Flow:
      1. Subscribe to runpod:result:{job_id}
      2. Submit job to RunPod API (using shared aiohttp session)
      3. GET check — if webhook already arrived before subscribe, use it
      4. Await Pub/Sub message — coroutine yields; event loop free for others
      5. Webhook handler writes SET + PUBLISH; subscriber wakes immediately

    Redis ops per job: 1 SUBSCRIBE + 1 GET + 1 UNSUBSCRIBE = 3 reads
    Webhook handler:   1 SET + 1 PUBLISH                   = 2 writes
    Total: 5 ops regardless of GPU duration (vs ~450 GETs with 200ms polling).
    """
    config = get_config()
    endpoint_id = config["endpoint_id"]
    api_key = config["api_key"]

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_payload = {
        "input": payload,
        "webhook": webhook_url,
    }

    # Submit the job using the shared session (avoids per-request TCP overhead)
    async with http_module.request_session() as sess:
        async with sess.post(url, json=request_payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error("runpod_api_error", extra={
                    "action": "runpod_api_error",
                    "status_code": response.status,
                    "error": error_text[:200],
                })
                return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": f"API error: {response.status}"}

            result_data = await response.json()
            job_id = result_data.get("id")

    if not job_id:
        logger.error("runpod_no_job_id", extra={"action": "runpod_no_job_id"})
        return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": "No job_id returned"}

    logger.info("runpod_job_submitted", extra={
        "action": "runpod_job_submitted",
        "job_id": job_id,
        "mode": "webhook_pubsub",
    })

    rc = redis_module.client
    if not rc:
        logger.warning("runpod_no_redis_fallback_polling", extra={
            "action": "runpod_no_redis_fallback_polling",
            "job_id": job_id,
        })
        runpod.api_key = api_key
        ep = runpod.Endpoint(endpoint_id)
        return await _run_with_polling(ep, payload, timeout_seconds)

    channel = f"runpod:result:{job_id}"
    start_time = time.time()

    # Create a dedicated pubsub connection for this job
    pubsub = rc.pubsub()
    await pubsub.subscribe(channel)

    try:
        # Race check: webhook may have arrived in the tiny window before subscribe
        early = await rc.get(channel)
        if early:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info("runpod_job_completed", extra={
                "action": "runpod_job_completed",
                "job_id": job_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "mode": "webhook_race_hit",
            })
            return json.loads(early)

        # Await Pub/Sub message — coroutine yields; event loop free for others
        async with asyncio.timeout(timeout_seconds):
            async for message in pubsub.listen():
                if message["type"] == "message":
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.info("runpod_job_completed", extra={
                        "action": "runpod_job_completed",
                        "job_id": job_id,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "mode": "webhook_pubsub",
                    })
                    return json.loads(message["data"])

    except asyncio.TimeoutError:
        logger.error("runpod_job_timeout", extra={
            "action": "runpod_job_timeout",
            "job_id": job_id,
            "timeout_seconds": timeout_seconds,
        })
        return {"error": "Webhook timeout", "ai_score": 0.0, "gpu_time_ms": 0.0}

    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
        except Exception:
            pass

    return {"error": "Pub/Sub exhausted", "ai_score": 0.0, "gpu_time_ms": 0.0}


async def _run_with_polling(endpoint, payload: dict, timeout_seconds: int = 90) -> Dict[str, Any]:
    """Fallback polling mode when webhooks are not configured.

    Uses exponential backoff (0.1 s → 2 s) to avoid hammering the RunPod API.
    job.status() is synchronous (RunPod SDK) so it runs in a thread.
    """
    job = await asyncio.to_thread(endpoint.run, payload)
    job_id = job.job_id
    poll_count = 0
    poll_delay = 0.1
    t_start = time.perf_counter()

    while True:
        status = await asyncio.to_thread(job.status)
        poll_count += 1
        if status == "COMPLETED":
            job_result = await asyncio.to_thread(job.output)
            logger.info("runpod_polling_completed", extra={
                "action": "runpod_polling_completed",
                "job_id": job_id,
                "poll_count": poll_count,
            })
            return job_result
        if status == "FAILED":
            logger.error("runpod_polling_error", extra={
                "action": "runpod_polling_error",
                "job_id": job_id,
                "status": "FAILED",
                "poll_count": poll_count,
            })
            return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": f"RunPod job {job_id} failed"}
        if (time.perf_counter() - t_start) > timeout_seconds:
            logger.error("runpod_polling_error", extra={
                "action": "runpod_polling_error",
                "job_id": job_id,
                "status": "TIMEOUT",
                "poll_count": poll_count,
            })
            return {"ai_score": 0.0, "gpu_time_ms": 0.0, "error": f"RunPod job {job_id} timed out"}
        await asyncio.sleep(poll_delay)
        poll_delay = min(poll_delay * 2, 2.0)


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
            "mask": mask_b64,
        }

        webhook_url = config.get("webhook_url")
        timeout = 180
        t_wall = time.time()

        if webhook_url:
            job_result = await _run_with_webhook(endpoint, payload, webhook_url, timeout_seconds=timeout)
        else:
            job_result = await _run_with_polling(endpoint, payload, timeout_seconds=timeout)

        wall_elapsed_sec = time.time() - t_wall

        if not job_result:
            raise RuntimeError("Empty response from worker")

        if "error" in job_result:
            raise RuntimeError(f"Worker error: {job_result['error']}")

        gpu_execution_sec = job_result.get("executionTime")
        if gpu_execution_sec is None:
            gpu_execution_sec = wall_elapsed_sec

        logger.info("runpod_call_completed", extra={
            "action": "runpod_call_completed",
            "gpu_execution_sec": round(gpu_execution_sec, 3),
            "wall_elapsed_sec": round(wall_elapsed_sec, 3),
            "cost_usd": round(gpu_execution_sec * settings.inpaint_rate_per_sec, 6),
            "used_reported_time": job_result.get("executionTime") is not None,
        })

        if "image_base64" not in job_result:
            if "results" in job_result:
                first = job_result["results"][0]
                if "image_base64" in first:
                    return base64.b64decode(first["image_base64"])

            logger.error("runpod_invalid_response", extra={
                "action": "runpod_invalid_response",
                "keys": list(job_result.keys()),
            })
            raise RuntimeError("Worker did not return image_base64")

        return base64.b64decode(job_result["image_base64"])

    except Exception as e:
        logger.error("runpod_inpainting_failed", extra={
            "action": "runpod_inpainting_failed",
            "error": str(e),
        })
        raise e
