"""
GPU worker integration — Modal primary, RunPod fallback.

Tries Modal first.  If Modal raises any error the call is retried once
via the existing RunPod integration so the user never sees a failure
during the migration window.  Remove the fallback once Modal is proven
stable in production.
"""

import logging
import os
import time

import modal
import modal.exception

from app.config import settings

logger = logging.getLogger(__name__)

MODAL_APP_NAME = "proof-or-poof-inpainting"
MODAL_ENV = os.getenv("MODAL_ENVIRONMENT", "main")
_MAX_PAYLOAD_BYTES = 95 * 1024 * 1024  # 95 MB guard (Modal gRPC limit is 100 MB)

Inpainter = modal.Cls.from_name(
    MODAL_APP_NAME, "Inpainter", environment_name=MODAL_ENV
)


async def _run_via_modal(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    """Call Modal GPU worker.  Raises on any failure."""
    combined = len(image_bytes) + len(mask_bytes)
    if combined > _MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Payload too large ({combined / 1024 / 1024:.1f} MB), max 95 MB"
        )
    if combined > 50 * 1024 * 1024:
        logger.warning("modal_large_payload", extra={
            "action": "modal_large_payload",
            "size_mb": round(combined / 1024 / 1024, 2),
        })

    inpainter = Inpainter()
    return await inpainter.process.remote.aio(image_bytes, mask_bytes)


async def run_gpu_inpainting(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    """
    Run inpainting on GPU — Modal primary, RunPod fallback.
    Returns: processed image bytes (PNG).
    """
    t_start = time.time()

    try:
        result_bytes = await _run_via_modal(image_bytes, mask_bytes)
        elapsed = time.time() - t_start
        logger.info("modal_inpainting_completed", extra={
            "action": "modal_inpainting_completed",
            "wall_elapsed_sec": round(elapsed, 3),
            "cost_usd": round(elapsed * settings.inpaint_rate_per_sec, 6),
            "provider": "modal",
        })
        return result_bytes

    except Exception as modal_err:
        elapsed = time.time() - t_start
        logger.error("modal_failed_falling_back_to_runpod", extra={
            "action": "modal_failed_falling_back_to_runpod",
            "error": str(modal_err),
            "error_type": type(modal_err).__name__,
            "elapsed_sec": round(elapsed, 3),
        })

    # ---- RunPod fallback (temporary, remove after Modal is stable) ----
    from app.integrations.runpod import run_gpu_inpainting as runpod_inpaint

    t_fallback = time.time()
    result_bytes = await runpod_inpaint(image_bytes, mask_bytes)
    elapsed_fb = time.time() - t_fallback

    logger.info("runpod_fallback_inpainting_completed", extra={
        "action": "runpod_fallback_inpainting_completed",
        "wall_elapsed_sec": round(elapsed_fb, 3),
        "cost_usd": round(elapsed_fb * settings.inpaint_rate_per_sec, 6),
        "provider": "runpod_fallback",
    })
    return result_bytes
