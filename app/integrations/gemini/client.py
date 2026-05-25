"""
Gemini API client — shared client instance + image-prep helpers + the
video batch entry point.

Image-detection callers use the combined engine
(`app/integrations/gemini/client_combined.py`); this module no longer
exposes a single-image sync analyzer. Video frame analysis still flows
through `analyze_batch_images_pro_turbo` below because the video pipeline
sends multiple frames in one Gemini call with its own per-frame quality
context — that path is independent of the combined engine.
"""

import os
import io
import time
import logging
from PIL import Image
from google import genai
from google.genai import types
from typing import Union

from app.config import settings
from app.schemas.detection import DetectionResult
from app.integrations.gemini.quality import get_quality_context
from app.integrations.gemini.prompts import get_system_instruction

# Cap decompression to 20 MP (from config) to prevent decompression-bomb DoS.
# A crafted PNG/TIFF can have a tiny file size but expand to multiple GB.
Image.MAX_IMAGE_PIXELS = settings.pil_max_image_pixels

logger = logging.getLogger(__name__)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=settings.gemini_http_timeout_ms,
        retry_options=types.HttpRetryOptions(
            attempts=settings.gemini_max_retries,
            initial_delay=settings.gemini_retry_initial_delay,
            max_delay=settings.gemini_retry_max_delay,
            exp_base=settings.gemini_retry_exp_base,
            http_status_codes=[408, 429, 500, 502, 503, 504]
        )
    )
)


def _prepare_pil_for_gemini(
    image_source: Union[str, Image.Image],
) -> tuple[Image.Image, list[Image.Image]]:
    """
    Shared image-prep used by every Gemini analyzer (v1, v2, ensemble).

    Opens the image if given a path, resizes to gemini_max_pixels, ensures
    RGB mode. Returns the WORKING PIL image (callers can compute noise_cv,
    encode JPEG, etc.) plus a list of intermediate PIL handles the caller
    must close after they're done — done this way so the same prep can be
    reused without forcing every caller into a context manager pattern.
    """
    img_to_close: list[Image.Image] = []
    if isinstance(image_source, str):
        img_original = Image.open(image_source)
        img_to_close.append(img_original)
    else:
        img_original = image_source

    img_working = _resize_if_needed(img_original)
    if img_working is not img_original:
        img_to_close.append(img_working)

    if img_working.mode != "RGB":
        img_rgb = img_working.convert("RGB")
        img_to_close.append(img_rgb)
        img_working = img_rgb

    return img_working, img_to_close


def _encode_pil_as_jpeg(img: Image.Image, quality: int) -> bytes:
    """Encode a PIL image as JPEG bytes at the given quality."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _resize_if_needed(img: Image.Image) -> Image.Image:
    """
    Resizes image if it exceeds 4MP (~2048x2048) to limit token usage and avoid payload errors.
    Keeps aspect ratio.

    Resampler note: BICUBIC instead of LANCZOS. Lanczos aggressively smooths high-frequency
    pixel noise and micro-textures — the exact mathematical anomalies the vision model uses
    to spot diffusion/GAN signatures. Bicubic preserves a cleaner representation of those
    structural artifacts while still producing acceptable visual quality.
    """
    w, h = img.size
    pixels = w * h

    if pixels > settings.gemini_max_pixels:
        scale = (settings.gemini_max_pixels / pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    return img


def analyze_batch_images_pro_turbo(image_sources: list[Union[str, Image.Image, bytes]]) -> dict:
    """
    Analyzes a batch of images for synthetic generation artifacts using the configured Gemini model.
    Returns a single aggregated result based on a per-frame vote.

    Per-frame quality context: each image's quality is analyzed independently and tagged
    inline in the execution query (e.g. "[FRAME 1 QUALITY: ...]"). The system prompt is built
    once with a neutral global context. This prevents the quality of one frame from
    contaminating the analysis of others in a heterogeneous batch.
    """
    try:
        image_parts: list[types.Part] = []
        per_frame_quality: list[str] = []

        for src in image_sources:
            # --- Per-frame quality probe ---
            try:
                frame_qc, _ = get_quality_context(src)
            except Exception as e:
                logger.error("gemini_quality_context_error", extra={
                    "action": "gemini_quality_context_error",
                    "error": str(e),
                })
                frame_qc = "**CONTEXT: QUALITY UNKNOWN.**"
            per_frame_quality.append(frame_qc)

            # --- Bytes path: no re-encoding, ship as-is ---
            if isinstance(src, bytes):
                image_parts.append(
                    types.Part.from_bytes(data=src, mime_type="image/jpeg")
                )
                continue

            # --- PIL / path path ---
            img_to_close: list[Image.Image] = []

            if isinstance(src, str):
                img_original = Image.open(src)
                img_to_close.append(img_original)
            else:
                img_original = src

            img_working = _resize_if_needed(img_original)
            if img_working is not img_original:
                img_to_close.append(img_working)

            if img_working.mode != "RGB":
                img_rgb = img_working.convert("RGB")
                img_to_close.append(img_rgb)
                img_working = img_rgb

            img_byte_arr = io.BytesIO()
            img_working.save(img_byte_arr, format='JPEG', quality=settings.gemini_batch_jpeg_quality)

            image_parts.append(
                types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg")
            )

            for img_obj in img_to_close:
                img_obj.close()

        # Neutral global context — the per-frame tags below carry the real quality signal.
        global_quality_context = (
            "**CONTEXT: PER-FRAME QUALITY VARIES.** Each frame in this batch carries its "
            "own quality tag in the execution query — apply the matching tag's guidance to "
            "the correspondingly indexed image."
        )

        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(global_quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level=settings.gemini_thinking_level),
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=list[DetectionResult],
        )

        quality_block = "\n".join(
            f"[FRAME {i + 1} QUALITY] {qc}" for i, qc in enumerate(per_frame_quality)
        )
        execution_query = (
            "Analyze EACH of the attached images for SYNTHETIC GENERATION ARTIFACTS, "
            "strictly following the system instructions. Apply the matching quality tag "
            "below to each correspondingly indexed image (image 1 → FRAME 1, etc.). "
            "Return one DetectionResult per image in the same order.\n\n"
            f"{quality_block}"
        )
        request_contents = image_parts + [execution_query]

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=request_contents,
            config=config
        )
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        raw_results = response.parsed

        if hasattr(response, "usage_metadata"):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            logger.info("gemini_call_completed", extra={
                "action": "gemini_call_completed",
                "model": settings.gemini_model,
                "call_type": "batch_images",
                "frame_count": len(image_sources),
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": settings.gemini_fixed_cost,
            })
        else:
            logger.info("gemini_call_completed", extra={
                "action": "gemini_call_completed",
                "model": settings.gemini_model,
                "call_type": "batch_images",
                "frame_count": len(image_sources),
                "duration_ms": duration_ms,
                "cost_usd": settings.gemini_fixed_cost,
            })

        # Representative quality context for downstream display/cache. For homogeneous
        # video frames these are effectively identical; for heterogeneous batches we
        # surface the first frame's tag — the model already saw all of them inline.
        representative_quality_context = per_frame_quality[0] if per_frame_quality else "**CONTEXT: QUALITY UNKNOWN.**"

        if not raw_results:
            # Gemini returned an empty/unparseable result (schema mismatch, safety
            # block, or empty candidates). Surface as a hard failure (-1.0) — the
            # caller must NOT cache this as a confident "clean" verdict.
            logger.error("gemini_batch_empty_parsed", extra={
                "action": "gemini_batch_empty_parsed",
                "frame_count": len(image_sources),
            })
            return {
                "confidence": -1.0,
                "signal_category": "multiple_subtle_ai_artifacts_present",
                "quality_context": representative_quality_context,
            }

        ai_votes = [r for r in raw_results if r.confidence > settings.gemini_ai_vote_threshold]
        not_ai_votes = [r for r in raw_results if r.confidence <= settings.gemini_ai_vote_threshold]

        final_result: dict = {}

        if len(ai_votes) > len(not_ai_votes):
            avg_conf = sum(r.confidence for r in ai_votes) / len(ai_votes)
            best = max(ai_votes, key=lambda x: x.confidence)
            final_result = {
                "confidence": round(avg_conf, 2),
                "signal_category": best.signal_category,
                "quality_context": representative_quality_context,
            }
        else:
            if not_ai_votes:
                avg_conf = sum(r.confidence for r in not_ai_votes) / len(not_ai_votes)
                best = min(not_ai_votes, key=lambda x: x.confidence)
                signal = best.signal_category
            else:
                avg_conf = 0.0
                signal = "no_visual_anomalies_detected"

            final_result = {
                "confidence": round(avg_conf, 2),
                "signal_category": signal,
                "quality_context": representative_quality_context,
            }

        if hasattr(response, "usage_metadata"):
            final_result["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }

        return final_result

    except Exception as e:
        logger.error("gemini_batch_error", extra={
            "action": "gemini_batch_error",
            "call_type": "batch_images",
            "frame_count": len(image_sources),
            "error": str(e),
            "error_type": type(e).__name__,
        }, exc_info=True)
        return {"confidence": -1.0}


