"""
Gemini API client — initialization and core inference functions.

The module-level `client` is created once on import using env vars.
`analyze_image_pro_turbo` and `analyze_batch_images_pro_turbo` are the
public entry points used by the detection pipeline.
"""

import os
import io
import json
import time
import logging
from PIL import Image, ImageFilter, ImageStat, ImageMath
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


def _compute_noise_cv(img: Image.Image) -> float:
    """
    Computes noise spatial CV (coefficient of variation across quadrants).

    Uses grayscale luminance + 32-bit float signed residual (I - B) to preserve
    the true statistical distribution of the noise. The previous implementation
    used ImageChops.difference (absolute |I-B|) in 8-bit uint, which:
      - Folds the signed noise distribution → artificially crushes variance
      - Loses sub-integer noise shifts (1-5px) to rounding/clipping

    Low CV = unnaturally uniform noise across all image regions = AI diffusion signal.
    High CV = spatially-varied organic noise = real camera sensor noise.

    Returns 0.0 (neutral / hint suppressed) if the image is too small to split
    into four meaningful quadrants (< 8px in either dimension).
    """
    w, h = img.size
    if w < 8 or h < 8:
        return 0.0

    img_gray  = img.convert("L")
    blurred   = img_gray.filter(ImageFilter.GaussianBlur(radius=settings.forensic_noise_radius))
    img_f     = img_gray.convert("F")
    blurred_f = blurred.convert("F")
    # Signed float32 residual — lambda_eval preserves negatives; avoids 8-bit clipping/folding
    noise = ImageMath.lambda_eval(lambda args: args["a"] - args["b"], a=img_f, b=blurred_f)

    mx, my = w // 2, h // 2
    quadrants = [
        noise.crop((0,  0,  mx, my)),
        noise.crop((mx, 0,  w,  my)),
        noise.crop((0,  my, mx, h)),
        noise.crop((mx, my, w,  h)),
    ]
    q_vars = [ImageStat.Stat(q).var[0] for q in quadrants]
    mean_var = sum(q_vars) / 4
    std_var  = (sum((v - mean_var) ** 2 for v in q_vars) / 4) ** 0.5
    return std_var / (mean_var + 1e-6)


def _resize_if_needed(img: Image.Image) -> Image.Image:
    """
    Resizes image if it exceeds 4MP (~2048x2048) to limit token usage and avoid payload errors.
    Keeps aspect ratio.
    """
    w, h = img.size
    pixels = w * h

    if pixels > settings.gemini_max_pixels:
        scale = (settings.gemini_max_pixels / pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def analyze_image_pro_turbo(image_source: Union[str, Image.Image], pre_calculated_quality_context: str = None) -> dict:
    """
    GEMINI 3.0 FLASH - OPTIMIZED FOR FORENSIC DETECTION
    """
    img_to_close = []

    try:
        if isinstance(image_source, str):
            img_original = Image.open(image_source)
            img_to_close.append(img_original)
        else:
            img_original = image_source

        quality_score = 0
        if pre_calculated_quality_context:
            quality_context = pre_calculated_quality_context
        else:
            quality_context, quality_score = get_quality_context(img_original)

        img_working = _resize_if_needed(img_original)
        if img_working is not img_original:
            img_to_close.append(img_working)

        if img_working.mode != "RGB":
            img_rgb = img_working.convert("RGB")
            img_to_close.append(img_rgb)
            img_working = img_rgb

        # Compute noise_cv cheaply (~20ms). Only injects a hint when the value
        # falls in [forensic_noise_cv_floor, forensic_noise_cv_ceil) — a range
        # that contains only AI images and zero real images in the gold dataset.
        # Outside that range the signal is too ambiguous to be directional.
        # Wrapped in try-except so any PIL edge-case never fails the full inference call.
        noise_hint = ""
        try:
            noise_cv = _compute_noise_cv(img_working)
            if settings.forensic_noise_cv_floor <= noise_cv < settings.forensic_noise_cv_ceil:
                noise_hint = (
                    f" [Forensic note: noise spatial uniformity score {noise_cv:.3f} is in a "
                    f"range consistent with AI diffusion output — examine micro-texture "
                    f"consistency across flat regions and edges especially carefully.]"
                )
                logger.info("forensic_noise_hint_fired", extra={
                    "action": "forensic_noise_hint_fired",
                    "noise_cv": round(noise_cv, 4),
                    "image_size": f"{img_working.width}x{img_working.height}",
                })
        except Exception as noise_err:
            logger.warning("forensic_noise_cv_failed", extra={
                "action": "forensic_noise_cv_failed",
                "error": str(noise_err),
                "image_size": f"{img_working.width}x{img_working.height}" if img_working else "unknown",
            })

        img_byte_arr = io.BytesIO()
        img_working.save(img_byte_arr, format='JPEG', quality=settings.gemini_jpeg_quality)
        image_bytes = img_byte_arr.getvalue()

        for img_obj in img_to_close:
            img_obj.close()

        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=DetectionResult,
        )

        execution_query = (
            "Carefully analyze this image for generative AI manipulation, "
            f"strictly following the system instructions.{noise_hint}"
        )

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                execution_query,
                execution_query,
            ],
            config=config
        )
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        parsed_result = response.parsed

        result = {
            "confidence": parsed_result.confidence,
            "signal_category": parsed_result.signal_category,
            "quality_score": quality_score
        }

        if hasattr(response, "usage_metadata"):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            result["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": response.usage_metadata.total_token_count
            }
            logger.info("gemini_call_completed", extra={
                "action": "gemini_call_completed",
                "model": "gemini-3-flash-preview",
                "call_type": "single_image",
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": settings.gemini_fixed_cost,
            })
        else:
            logger.info("gemini_call_completed", extra={
                "action": "gemini_call_completed",
                "model": "gemini-3-flash-preview",
                "call_type": "single_image",
                "duration_ms": duration_ms,
                "cost_usd": settings.gemini_fixed_cost,
            })

        result["quality_context"] = quality_context
        return result

    except Exception as e:
        logger.error("gemini_analyze_error", extra={
            "action": "gemini_analyze_error",
            "call_type": "single_image",
            "error": str(e),
            "error_type": type(e).__name__,
            "image_source_type": "file_path" if isinstance(image_source, str) else "pil_image",
        }, exc_info=True)
        # confidence=-1.0 signals a hard failure to image_detector.py
        return {"confidence": -1.0, "signal_category": "multiple_subtle_ai_artifacts_present"}


def analyze_batch_images_pro_turbo(image_sources: list[Union[str, Image.Image, bytes]]) -> dict:
    """
    Analyzes a batch of images for synthetic generation artifacts using Gemini 3.0 Flash.
    Returns a single aggregated result based on the median decision.
    """
    try:
        image_parts = []
        quality_context = None

        idx_to_scan = 1 if len(image_sources) > 1 else 0

        for i, src in enumerate(image_sources):
            if i == idx_to_scan:
                try:
                    quality_context, _ = get_quality_context(src)
                except Exception as e:
                    logger.error("gemini_quality_context_error", extra={
                        "action": "gemini_quality_context_error",
                        "error": str(e),
                    })

            if isinstance(src, bytes):
                image_parts.append(
                    types.Part.from_bytes(data=src, mime_type="image/jpeg")
                )
                continue

            img_to_close = []

            if isinstance(src, str):
                img_original = Image.open(src)
                img_to_close.append(img_original)
            else:
                img_original = src

            if i == idx_to_scan and not quality_context:
                quality_context, _ = get_quality_context(img_original)

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

        if not quality_context:
            quality_context = "**CONTEXT: QUALITY UNKNOWN.**"

        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=list[DetectionResult],
        )

        execution_query = "Analyze EACH of the attached images for SYNTHETIC GENERATION ARTIFACTS, strictly following the system instructions."
        request_contents = image_parts + [execution_query]

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
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
                "model": "gemini-3-flash-preview",
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
                "model": "gemini-3-flash-preview",
                "call_type": "batch_images",
                "frame_count": len(image_sources),
                "duration_ms": duration_ms,
                "cost_usd": settings.gemini_fixed_cost,
            })

        if not raw_results:
            return {"confidence": 0.5, "signal_category": "multiple_subtle_ai_artifacts_present"}

        ai_votes = [r for r in raw_results if r.confidence > settings.gemini_ai_vote_threshold]
        not_ai_votes = [r for r in raw_results if r.confidence <= settings.gemini_ai_vote_threshold]

        final_result = {}

        if len(ai_votes) > len(not_ai_votes):
            avg_conf = sum(r.confidence for r in ai_votes) / len(ai_votes)
            best = max(ai_votes, key=lambda x: x.confidence)
            final_result = {
                "confidence": round(avg_conf, 2),
                "signal_category": best.signal_category,
                "quality_context": quality_context
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
                "quality_context": quality_context
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Analyzing: {path}...")
        start = time.perf_counter()
        result = analyze_image_pro_turbo(path)
        end = time.perf_counter()
        print(f"Result: {json.dumps(result, indent=2)}")
        print(f"Latency: {end - start:.4f}s")
