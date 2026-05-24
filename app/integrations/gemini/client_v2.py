"""
V2 forensic inference client.

Reuses the shared `client`, `_compute_noise_cv`, and `_resize_if_needed` from v1
so we don't duplicate image-preprocessing logic. Differs from v1 in three ways:

1. Uses the V2 XML prompt (`get_system_instruction_v2`) and `DetectionResultV2`
   schema (forced 2-step CoT + 5 macro signal buckets).
2. Locks deterministic decoding: temperature=0.0, top_k=1, top_p=0.1, applied
   only when the corresponding settings are set to v2's locked values.
3. Maps the V2 macro signal_category back to the legacy 19-category taxonomy
   used by `image_detector._label_for` and downstream UX, so the response shape
   stays identical to v1.
"""

import io
import time
import logging

from PIL import Image
from google.genai import types
from typing import Optional, Union

from app.config import settings
from app.schemas.detection import DetectionResultV2
from app.integrations.gemini.client import (
    client,
    _compute_noise_cv,
    _resize_if_needed,
)
from app.integrations.gemini.quality import get_quality_context
from app.integrations.gemini.prompts_v2 import get_system_instruction_v2


logger = logging.getLogger(__name__)


# V2 macro buckets → legacy SIGNAL_CATEGORIES so the downstream response shape
# and `_label_for` (image_detector.py) stay backward-compatible without forcing
# the frontend to ship a new taxonomy at the same time.
V2_TO_LEGACY_CATEGORY: dict[str, str] = {
    "peripheral_or_background_structural_collapse": "facial_detail_inconsistency_detected",
    "objects_merge_or_dissolve_at_boundaries": "objects_merge_or_dissolve_at_boundaries",
    "geometry_or_perspective_is_physically_impossible": "geometry_or_perspective_is_physically_impossible",
    "in_scene_text_is_melted_or_gibberish": "text_or_signage_contains_gibberish_characters",
    "multiple_subtle_ai_artifacts_present": "multiple_subtle_ai_artifacts_present",
    "no_visual_anomalies_detected": "no_visual_anomalies_detected",
}


def _build_v2_config(quality_context: str) -> types.GenerateContentConfig:
    """Builds the v2 GenerateContentConfig with deterministic-decoding overrides."""
    config_kwargs: dict = dict(
        system_instruction=get_system_instruction_v2(quality_context),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        temperature=settings.gemini_temperature,
        response_mime_type="application/json",
        response_schema=DetectionResultV2,
    )
    # thinking_config is only supported on gemini-3-* models. Older 2.x flash
    # families reject the field with INVALID_ARGUMENT.
    if "gemini-3" in settings.gemini_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=settings.gemini_thinking_level
        )
    # Only forward top_k/top_p when explicitly set — keeps v1 behavior unchanged
    # for any caller that pins gemini_temperature without flipping the v2 flag.
    if settings.gemini_top_k > 0:
        config_kwargs["top_k"] = settings.gemini_top_k
    if settings.gemini_top_p > 0:
        config_kwargs["top_p"] = settings.gemini_top_p
    return types.GenerateContentConfig(**config_kwargs)


def analyze_image_pro_turbo_v2(
    image_source: Union[str, Image.Image],
    pre_calculated_quality_context: Optional[str] = None,
) -> dict:
    """Single-image v2 forensic inference. Returns the same dict shape as v1."""
    img_to_close: list[Image.Image] = []

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

        noise_hint = ""
        try:
            noise_cv = _compute_noise_cv(img_working)
            if settings.forensic_noise_cv_floor <= noise_cv < settings.forensic_noise_cv_ceil:
                noise_hint = (
                    f" [Forensic note: noise spatial uniformity {noise_cv:.3f} is in a "
                    f"range consistent with AI diffusion output — scrutinise micro-texture "
                    f"consistency across flat regions and edges.]"
                )
                logger.info("forensic_noise_hint_fired_v2", extra={
                    "action": "forensic_noise_hint_fired_v2",
                    "noise_cv": round(noise_cv, 4),
                })
        except Exception as noise_err:
            logger.warning("forensic_noise_cv_failed_v2", extra={
                "action": "forensic_noise_cv_failed_v2",
                "error": str(noise_err),
            })

        img_byte_arr = io.BytesIO()
        img_working.save(img_byte_arr, format="JPEG", quality=settings.gemini_jpeg_quality)
        image_bytes = img_byte_arr.getvalue()

        for img_obj in img_to_close:
            img_obj.close()

        config = _build_v2_config(quality_context)
        execution_query = (
            "Carefully analyse this image for generative AI manipulation, strictly "
            "following the system instructions. Fill step_1 (edges/background) and "
            f"step_2 (physics/boundaries) BEFORE choosing confidence.{noise_hint}"
        )

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                execution_query,
            ],
            config=config,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        parsed: Optional[DetectionResultV2] = response.parsed
        if parsed is None:
            raise ValueError("Gemini v2 returned no parsed DetectionResultV2 (schema or safety block)")

        legacy_category = V2_TO_LEGACY_CATEGORY.get(
            parsed.signal_category,
            "multiple_subtle_ai_artifacts_present",
        )

        result: dict = {
            "visual_scan": parsed.visual_scan,
            "confidence": parsed.confidence,
            "signal_category": legacy_category,
            "quality_score": quality_score,
            "quality_context": quality_context,
            # V2-only diagnostic fields (do NOT remove — used by eval harness)
            "v2_signal_category": parsed.signal_category,
            "v2_step_1": parsed.scan_hands_and_boundaries,
            "v2_step_2": parsed.scan_background_and_physics,
        }

        if hasattr(response, "usage_metadata"):
            result["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
            logger.info("gemini_call_completed_v2", extra={
                "action": "gemini_call_completed_v2",
                "model": settings.gemini_model,
                "duration_ms": duration_ms,
                "input_tokens": result["usage"]["prompt_tokens"],
                "output_tokens": result["usage"]["completion_tokens"],
                "cost_usd": settings.gemini_fixed_cost,
            })
        else:
            logger.info("gemini_call_completed_v2", extra={
                "action": "gemini_call_completed_v2",
                "model": settings.gemini_model,
                "duration_ms": duration_ms,
                "cost_usd": settings.gemini_fixed_cost,
            })

        return result

    except Exception as e:
        logger.error("gemini_analyze_error_v2", extra={
            "action": "gemini_analyze_error_v2",
            "error": str(e),
            "error_type": type(e).__name__,
        }, exc_info=True)
        return {"confidence": -1.0, "signal_category": "multiple_subtle_ai_artifacts_present"}
