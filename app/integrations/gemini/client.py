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
from PIL import Image
from google import genai
from google.genai import types
from typing import Union

from app.config import settings
from app.schemas.detection import DetectionResult
from app.integrations.gemini.quality import get_quality_context
from app.integrations.gemini.prompts import get_system_instruction

# Allow loading truncated / oversized images (override PIL safety cap)
Image.MAX_IMAGE_PIXELS = None

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
    img_to_close = []  # Keep a list of intermediate images to close

    try:
        if isinstance(image_source, str):
            img_original = Image.open(image_source)
            img_to_close.append(img_original)
        else:
            img_original = image_source

        # 1. Analyze Quality on ORIGINAL resolution (if not pre-calculated)
        quality_score = 0
        if pre_calculated_quality_context:
            quality_context = pre_calculated_quality_context
        else:
            quality_context, quality_score = get_quality_context(img_original)

        # 2. Resize for Upload
        img_working = _resize_if_needed(img_original)
        if img_working is not img_original:
            img_to_close.append(img_working)

        # 3. Ensure RGB
        if img_working.mode != "RGB":
            img_rgb = img_working.convert("RGB")
            img_to_close.append(img_rgb)
            img_working = img_rgb

        # 4. Save to bytes
        img_byte_arr = io.BytesIO()
        img_working.save(img_byte_arr, format='JPEG', quality=settings.gemini_jpeg_quality)
        image_bytes = img_byte_arr.getvalue()

        # 5. Clean up ALL intermediate objects immediately
        for img_obj in img_to_close:
            img_obj.close()

        # --- CONFIGURATION ---
        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=DetectionResult,
        )

        # --- THE PAYLOAD (With x2 Prompt Repetition Hack) ---
        execution_query = "Carefully analyze this image for generative AI manipulation, strictly following the system instructions."

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                execution_query,
                execution_query  # Duplicated for 360-degree mathematical context
            ],
            config=config
        )

        parsed_result = response.parsed

        result = {
            "confidence": parsed_result.confidence,
            "explanation": parsed_result.explanation,
            "quality_score": quality_score
        }

        if hasattr(response, "usage_metadata"):
            result["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }

        result["quality_context"] = quality_context
        return result

    except Exception as e:
        logger.error(f"[GEMINI] analyze_image_pro_turbo error: {e}")
        return {"confidence": -1.0}


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
                    logger.error(f"[GEMINI] Failed to get quality context for video frame: {e}")

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

        # --- CONFIGURATION ---
        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=list[DetectionResult],
        )

        # --- THE PAYLOAD (With x2 Prompt Repetition Hack) ---
        execution_query = "Analyze EACH of the attached images for SYNTHETIC GENERATION ARTIFACTS, strictly following the system instructions."
        request_contents = image_parts + [execution_query, execution_query]

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=request_contents,
            config=config
        )

        raw_results = response.parsed

        if not raw_results:
            return {"confidence": 0.5, "explanation": "Suspicious: No clear analysis returned."}

        ai_votes = [r for r in raw_results if r.confidence > settings.gemini_ai_vote_threshold]
        not_ai_votes = [r for r in raw_results if r.confidence <= settings.gemini_ai_vote_threshold]

        final_result = {}

        if len(ai_votes) > len(not_ai_votes):
            avg_conf = sum(r.confidence for r in ai_votes) / len(ai_votes)
            best_explanation_item = max(ai_votes, key=lambda x: x.confidence)
            final_result = {
                "confidence": round(avg_conf, 2),
                "explanation": best_explanation_item.explanation,
                "quality_context": quality_context
            }
        else:
            if not_ai_votes:
                avg_conf = sum(r.confidence for r in not_ai_votes) / len(not_ai_votes)
                best_explanation_item = min(not_ai_votes, key=lambda x: x.confidence)
                explanation = best_explanation_item.explanation
            else:
                avg_conf = 0.0
                explanation = "No visual anomalies detected"

            final_result = {
                "confidence": round(avg_conf, 2),
                "explanation": explanation,
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
        logger.error(f"[GEMINI] analyze_batch_images_pro_turbo error: {e}")
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
