"""
Combined-engine Gemini analyzer — sole image-detection client.

Single Gemini API call per image with all three forensic perspectives
(anatomy / physics / composition) merged into one system instruction.
Uses the native async client surface (`client.aio.models.generate_content`)
so the Gemini I/O wait doesn't occupy a thread, and CPU-bound image prep
runs in `asyncio.to_thread` so the event loop stays free.

Empirical accuracy vs alternates measured on the 25-case gold set
(scripts/eval_cost_accuracy.py): 96% combined vs 92% parallel ensemble
vs 67% anatomy-solo. See the PR description for the full breakdown.

Image preprocessing helpers (`_prepare_pil_for_gemini`, `_encode_pil_as_jpeg`,
`_resize_if_needed`) and the shared `client` instance live in
`app/integrations/gemini/client.py` — reused by video detection too.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Union

from PIL import Image
from google.genai import types

from app.config import settings
from app.schemas.detection import CombinedDetectionResult, V2_TO_LEGACY_CATEGORY
from app.integrations.gemini.client import (
    client,
    _prepare_pil_for_gemini,
    _encode_pil_as_jpeg,
)
from app.integrations.gemini.prompts_combined import get_combined_prompt
from app.integrations.gemini.quality import get_quality_context

logger = logging.getLogger(__name__)


def _build_combined_config(system_prompt: str) -> types.GenerateContentConfig:
    """Builds the GenerateContentConfig for the combined sub-call."""
    config_kwargs: dict = dict(
        system_instruction=system_prompt,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        temperature=settings.gemini_temperature,
        response_mime_type="application/json",
        response_schema=CombinedDetectionResult,
    )
    if "gemini-3" in settings.gemini_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=settings.gemini_thinking_level
        )
    return types.GenerateContentConfig(**config_kwargs)


def _encode_image_for_gemini(image_source: Union[str, Image.Image, bytes]) -> bytes:
    """Resize-if-needed + JPEG encode. CPU-bound — caller should `asyncio.to_thread` this."""
    if isinstance(image_source, bytes):
        return image_source
    img_working, img_to_close = _prepare_pil_for_gemini(image_source)
    try:
        return _encode_pil_as_jpeg(img_working, settings.gemini_jpeg_quality)
    finally:
        for img_obj in img_to_close:
            try:
                img_obj.close()
            except Exception:
                pass


async def analyze_image_combined_async(
    image_source: Union[str, Image.Image],
    pre_calculated_quality_context: Optional[str] = None,
    pre_calculated_quality_score: Optional[int] = None,
) -> dict:
    """
    Single Gemini call with the combined 3-perspective prompt.

    Returns the standard detection-response dict shape used by image_detector:
        {
          "visual_scan": str,
          "confidence": float | -1.0 on failure,
          "signal_category": <legacy 19-category string>,
          "quality_score": int,
          "quality_context": str,
          # Diagnostic fields (used by eval scripts + structured logs)
          "v2_signal_category": <macro v2 category>,
          "v2_step_1": <findings text>,
          "v2_step_2": <"region_anchor: ..." string>,
          "region_anchor": str,
          "ok": bool,
        }
    """
    quality_score = pre_calculated_quality_score or 0
    if pre_calculated_quality_context:
        quality_context = pre_calculated_quality_context
    else:
        # cv2.Laplacian + cv2.cvtColor are sync CPU — keep off the event loop.
        try:
            quality_context, quality_score = await asyncio.to_thread(
                get_quality_context, image_source
            )
        except Exception as exc:
            logger.warning("combined_quality_context_failed", extra={
                "action": "combined_quality_context_failed",
                "error": str(exc),
            })
            quality_context = "**CONTEXT: QUALITY UNKNOWN.**"

    try:
        # PIL resize + JPEG encode is ~25-65ms of pure CPU. Must run in a
        # thread so concurrent requests don't serialise on the event loop.
        image_bytes = await asyncio.to_thread(_encode_image_for_gemini, image_source)
        config = _build_combined_config(get_combined_prompt(quality_context))

        t0 = time.perf_counter()
        # Native-async client surface — Gemini I/O wait does NOT consume a
        # thread. Cancellations propagate to the HTTP socket cleanly.
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Analyse this image strictly within the focus area defined by the system instructions. Return ONLY the JSON.",
            ],
            config=config,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000)

        parsed: Optional[CombinedDetectionResult] = response.parsed
        if parsed is None:
            logger.warning("combined_subcall_unparseable", extra={
                "action": "combined_subcall_unparseable",
                "duration_ms": duration_ms,
            })
            return _failure_response(quality_context, quality_score, "(unparseable response)")

        # Enforce structural anti-anchoring: an AI verdict (conf >= 0.5) must
        # cite a specific named region. If the model emitted an AI confidence
        # but left region_anchor == "none", it confabulated — coerce to REAL.
        anchor = (parsed.region_anchor or "").strip().lower()
        unanchored = anchor in ("", "none", "n/a", "na")
        if parsed.confidence >= 0.5 and unanchored:
            logger.info("combined_unanchored_ai_demoted", extra={
                "action": "combined_unanchored_ai_demoted",
                "original_confidence": parsed.confidence,
                "original_signal_category": parsed.signal_category,
                "findings": parsed.findings,
            })
            confidence_out = 0.0
            signal_v2 = "no_visual_anomalies_detected"
            findings_out = f"[demoted: AI claim without region anchor] {parsed.findings}"
        else:
            confidence_out = float(parsed.confidence)
            signal_v2 = parsed.signal_category
            findings_out = parsed.findings

        legacy_cat = V2_TO_LEGACY_CATEGORY.get(signal_v2, "multiple_subtle_ai_artifacts_present")

        logger.info("combined_call_completed", extra={
            "action": "combined_call_completed",
            "model": settings.gemini_model,
            "duration_ms": duration_ms,
            "confidence": confidence_out,
            "signal_category": signal_v2,
            "region_anchor": parsed.region_anchor,
            "cost_usd": settings.gemini_fixed_cost,
        })

        return {
            "visual_scan": findings_out,
            "confidence": confidence_out,
            "signal_category": legacy_cat,
            "quality_score": quality_score,
            "quality_context": quality_context,
            "v2_signal_category": signal_v2,
            "v2_step_1": findings_out,
            "v2_step_2": f"region_anchor: {parsed.region_anchor}",
            "region_anchor": parsed.region_anchor,
            "ok": True,
        }

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        from google.genai import errors as _genai_errors
        is_api_error = isinstance(exc, _genai_errors.APIError)
        log_event = "combined_call_api_error" if is_api_error else "combined_call_code_error"
        log_method = logger.warning if is_api_error else logger.error
        log_method(log_event, extra={
            "action": log_event,
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        return _failure_response(
            quality_context, quality_score, f"(error: {type(exc).__name__})"
        )


def _failure_response(quality_context: str, quality_score: int, reason: str) -> dict:
    """Common failure-sentinel shape — confidence -1.0 signals the caller."""
    return {
        "visual_scan": reason,
        "confidence": -1.0,
        "signal_category": "multiple_subtle_ai_artifacts_present",
        "quality_score": quality_score,
        "quality_context": quality_context,
        "v2_signal_category": "no_visual_anomalies_detected",
        "v2_step_1": reason,
        "v2_step_2": "",
        "region_anchor": "none",
        "ok": False,
    }
