"""
Ensemble Gemini sub-call helper.

Each sub-call sends the image with a SINGLE specialised system prompt
(anatomy / physics / composition) and returns an EnsembleSubResult. The
orchestrator (app/detection/ensemble_engine.py) fans these out in parallel
and applies asymmetric voting.

**Native-async path**: this module uses `client.aio.models.generate_content`,
not the sync `client.models.generate_content` wrapped in `asyncio.to_thread`.
The reason: race-to-AI cancellation needs to propagate to the underlying
HTTP socket. With the sync-in-a-thread pattern, `t.cancel()` cancels the
asyncio Task but the OS thread keeps running until the HTTP round-trip
completes, leaking thread-pool capacity and burning the 30-slot pool we
set in main.py lifespan under concurrent load. With `client.aio` the
cancellation propagates to the aiohttp/httpx socket directly.

Image preprocessing (resize, JPEG encode) still runs synchronously — it's
CPU-bound and fast (≤30ms) so threading it adds overhead without benefit.
We do it inline on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Union

from PIL import Image
from google.genai import types

from app.config import settings
from app.schemas.detection import EnsembleSubResult
from app.integrations.gemini.client import (
    client,
    _prepare_pil_for_gemini,
    _encode_pil_as_jpeg,
)
from app.integrations.gemini.quality import get_quality_context

logger = logging.getLogger(__name__)


def _build_ensemble_config(system_prompt: str) -> types.GenerateContentConfig:
    """Builds the GenerateContentConfig for a single ensemble sub-call."""
    config_kwargs: dict = dict(
        system_instruction=system_prompt,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        temperature=settings.gemini_temperature,
        response_mime_type="application/json",
        response_schema=EnsembleSubResult,
    )
    if "gemini-3" in settings.gemini_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=settings.gemini_thinking_level
        )
    if settings.gemini_top_k > 0:
        config_kwargs["top_k"] = settings.gemini_top_k
    if settings.gemini_top_p > 0:
        config_kwargs["top_p"] = settings.gemini_top_p
    return types.GenerateContentConfig(**config_kwargs)


def _encode_image_for_gemini(image_source: Union[str, Image.Image, bytes]) -> bytes:
    """Resize-if-needed + JPEG encode. Delegates to the shared client helpers."""
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


async def analyze_with_prompt(
    image_source: Union[str, Image.Image, bytes],
    system_prompt: str,
    label: str,
    pre_calculated_quality_context: Optional[str] = None,
) -> dict:
    """
    Single Gemini sub-call for the ensemble. Async-native — uses `client.aio`
    so cancellation propagates to the HTTP socket cleanly (race-to-AI
    cancels reach the wire, not just the asyncio Task).

    Returns:
        {
          "label": str,                         # which sub-prompt produced this
          "confidence": float | -1.0 on error,
          "signal_category": str,
          "findings": str,
          "region_anchor": str,
          "duration_ms": int,
          "ok": bool,
        }

    Never raises — surface errors via `ok=False` so the ensemble orchestrator
    can apply asymmetric voting across whatever voters did respond. The one
    exception we DO let propagate is `asyncio.CancelledError`: when
    race-to-AI cancels this voter, the exception must propagate so the inner
    `await` unwinds and the underlying HTTP socket actually closes — catching
    it would leak the in-flight request.
    """
    try:
        if pre_calculated_quality_context is None:
            if isinstance(image_source, (str, Image.Image)):
                quality_context, _ = get_quality_context(image_source)
            else:
                quality_context = "**CONTEXT: QUALITY UNKNOWN.**"
        else:
            quality_context = pre_calculated_quality_context

        # The system_prompt is the focused prompt itself; the {quality_context}
        # placeholder inside each get_*_prompt() is already substituted before
        # we receive it. Just pass through.
        # PIL resize + JPEG encode is CPU-bound (~25-65ms for typical uploads).
        # Move it off the event loop so concurrent detections don't serialise
        # on this work — critical under Railway's single-worker uvicorn where
        # the event loop is the throughput bottleneck.
        image_bytes = await asyncio.to_thread(_encode_image_for_gemini, image_source)
        config = _build_ensemble_config(system_prompt)

        t0 = time.perf_counter()
        # Native-async path — client.aio.models is the Google GenAI SDK's
        # async client surface. Importantly, awaiting it yields the event loop
        # back to the orchestrator's race-to-AI loop; a subsequent .cancel()
        # of this Task closes the underlying HTTP socket, not leaving a
        # ghost thread.
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Analyse this image strictly within the focus area defined by the system instructions. Return ONLY the JSON.",
            ],
            config=config,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000)

        parsed: Optional[EnsembleSubResult] = response.parsed
        if parsed is None:
            logger.warning("ensemble_subcall_unparseable", extra={
                "action": "ensemble_subcall_unparseable",
                "label": label,
            })
            return {
                "label": label, "confidence": -1.0,
                "signal_category": "no_visual_anomalies_detected",
                "findings": "(unparseable response)",
                "duration_ms": duration_ms, "ok": False,
            }

        # Structural anti-anchoring enforcement: an AI verdict (confidence
        # >= 0.5) MUST cite a specific named region. If the model emitted an
        # AI confidence but left region_anchor == "none" / "" / etc., it
        # confabulated — coerce the verdict to REAL with confidence 0.0 and
        # log so we can track how often this happens. Real catches always
        # have an anchor; FPs are the case where the model can't name a region.
        anchor = (parsed.region_anchor or "").strip().lower()
        unanchored = anchor in ("", "none", "n/a", "na")
        if parsed.confidence >= 0.5 and unanchored:
            logger.info("ensemble_subcall_unanchored_ai_demoted", extra={
                "action": "ensemble_subcall_unanchored_ai_demoted",
                "label": label,
                "original_confidence": parsed.confidence,
                "original_signal_category": parsed.signal_category,
                "findings": parsed.findings,
            })
            confidence_out = 0.0
            signal_out = "no_visual_anomalies_detected"
            findings_out = f"[demoted: AI claim without region anchor] {parsed.findings}"
        else:
            confidence_out = float(parsed.confidence)
            signal_out = parsed.signal_category
            findings_out = parsed.findings

        logger.info("ensemble_subcall_completed", extra={
            "action": "ensemble_subcall_completed",
            "label": label,
            "duration_ms": duration_ms,
            "confidence": confidence_out,
            "signal_category": signal_out,
            "region_anchor": parsed.region_anchor,
        })
        return {
            "label": label,
            "confidence": confidence_out,
            "signal_category": signal_out,
            "findings": findings_out,
            "region_anchor": parsed.region_anchor,
            "duration_ms": duration_ms,
            "ok": True,
        }
    except asyncio.CancelledError:
        # Race-to-AI cancellation — let it propagate so the inner await unwinds
        # and the underlying HTTP socket closes. Catching this would leak
        # in-flight Gemini requests after early-exit fires.
        raise
    except Exception as exc:
        # Distinguish transient API failures (genuine reason to abstain) from
        # programming errors (schema/import/code bugs that should be loud).
        # Both still return an `ok=False` sentinel so a single broken voter
        # cannot crash the whole ensemble, but programming errors log at
        # ERROR-level so they actually surface in alerting.
        from google.genai import errors as _genai_errors

        is_api_error = isinstance(exc, _genai_errors.APIError)
        log_event = "ensemble_subcall_api_error" if is_api_error else "ensemble_subcall_code_error"
        log_method = logger.warning if is_api_error else logger.error
        log_method(log_event, extra={
            "action": log_event,
            "label": label,
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        return {
            "label": label, "confidence": -1.0,
            "signal_category": "no_visual_anomalies_detected",
            "findings": f"(error: {type(exc).__name__})",
            "duration_ms": 0, "ok": False,
        }
