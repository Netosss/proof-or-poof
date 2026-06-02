"""
Combined Gemini analyzer — sole detection client for both image and video.

Two public entry points:
  * analyze_image_combined_async(image) — one Gemini call per image
  * analyze_video_frames_async(frames)   — one Gemini call per video,
    N frames in the request body, list[CombinedDetectionResult] schema,
    per-frame vote aggregation

Both use the SAME combined system prompt
(`get_combined_prompt` from prompts_combined.py), the SAME schema
(`CombinedDetectionResult`), and the SAME native async client surface
(`client.aio.models.generate_content`) so Gemini I/O waits never burn
threads. CPU-bound image prep runs in `asyncio.to_thread` to keep the
event loop free.
"""

from __future__ import annotations

import asyncio
import io
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


def _encode_full_and_crop(image_source: Union[str, Image.Image, bytes]):
    """Full image + a center crop (25-75%) for high-frequency micro-texture context.
    Returns (full_bytes, crop_bytes|None). Crop skipped for raw bytes input."""
    if isinstance(image_source, bytes):
        return image_source, None
    img_working, img_to_close = _prepare_pil_for_gemini(image_source)
    try:
        full = _encode_pil_as_jpeg(img_working, settings.gemini_jpeg_quality)
        w, h = img_working.size
        box = (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))
        crop_bytes = _encode_pil_as_jpeg(img_working.crop(box), settings.gemini_jpeg_quality)
        return full, crop_bytes
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
        full_bytes, crop_bytes = await asyncio.to_thread(_encode_full_and_crop, image_source)
        config = _build_combined_config(get_combined_prompt(quality_context))

        contents: list = [types.Part.from_bytes(data=full_bytes, mime_type="image/jpeg")]
        if crop_bytes is not None:
            contents.append(
                "The image above is the FULL view. Below is a high-resolution CENTER CROP of the "
                "focal area — inspect it for micro-texture / material-boundary anomalies, then "
                "judge the whole image."
            )
            contents.append(types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg"))
        contents.append(
            "Analyse this image strictly within the focus area defined by the system instructions. Return ONLY the JSON."
        )

        t0 = time.perf_counter()
        # Native-async client surface — Gemini I/O wait does NOT consume a
        # thread. Cancellations propagate to the HTTP socket cleanly.
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
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
            signal_macro = "no_visual_anomalies_detected"
            findings_out = f"[demoted: AI claim without region anchor] {parsed.findings}"
        else:
            confidence_out = float(parsed.confidence)
            signal_macro = parsed.signal_category
            findings_out = parsed.findings

        legacy_cat = V2_TO_LEGACY_CATEGORY.get(signal_macro, "multiple_subtle_ai_artifacts_present")

        logger.info("combined_call_completed", extra={
            "action": "combined_call_completed",
            "model": settings.gemini_model,
            "duration_ms": duration_ms,
            "confidence": confidence_out,
            "signal_category": signal_macro,
            "region_anchor": parsed.region_anchor,
            "cost_usd": settings.gemini_fixed_cost,
        })

        return {
            "visual_scan": findings_out,
            "confidence": confidence_out,
            "signal_category": legacy_cat,
            "quality_score": quality_score,
            "quality_context": quality_context,
            "v2_signal_category": signal_macro,
            "v2_step_1": findings_out,
            "v2_step_2": f"region_anchor: {parsed.region_anchor}",
            "region_anchor": parsed.region_anchor,
            "content_plausibility": parsed.content_plausibility,
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


async def analyze_video_frames_async(
    image_sources: list[Union[str, Image.Image, bytes]],
) -> dict:
    """
    Multi-frame video forensic inference via the SAME combined prompt used
    for single-image detection — just batched across N frames.

    The combined system instruction stays identical (three perspectives,
    StudioException, LandmarkSignageException). The execution query
    annotates each frame with its own quality tag so heterogeneous batches
    don't contaminate each other. Schema is list[CombinedDetectionResult]
    — one verdict per frame, in input order. Aggregation is a per-frame
    vote: if more frames flag AI than don't, return avg AI confidence;
    otherwise return min REAL confidence.

    Returns the dict shape pipeline.py expects:
        {
          "confidence": float | -1.0,
          "signal_category": str,
          "quality_context": str,
        }
    """
    try:
        # ---------- per-frame quality probes + image encoding (in threads) ----------
        def _prepare_one_frame(src) -> tuple[bytes, str]:
            try:
                qc, _ = get_quality_context(src)
            except Exception:
                qc = "**CONTEXT: QUALITY UNKNOWN.**"
            if isinstance(src, bytes):
                return src, qc
            img_working, to_close = _prepare_pil_for_gemini(src)
            try:
                buf = io.BytesIO()
                img_working.save(buf, format="JPEG", quality=settings.gemini_batch_jpeg_quality)
                return buf.getvalue(), qc
            finally:
                for o in to_close:
                    try:
                        o.close()
                    except Exception:
                        pass

        # Run all frame preps concurrently in worker threads so the event
        # loop stays free for incoming requests.
        prepped = await asyncio.gather(
            *(asyncio.to_thread(_prepare_one_frame, s) for s in image_sources)
        )
        image_parts = [
            types.Part.from_bytes(data=b, mime_type="image/jpeg") for b, _ in prepped
        ]
        per_frame_quality = [qc for _, qc in prepped]

        # ---------- system instruction = same combined prompt as image path ----------
        global_quality_context = (
            "**CONTEXT: PER-FRAME QUALITY VARIES.** Each frame in this batch carries its "
            "own quality tag in the execution query — apply the matching tag's guidance to "
            "the correspondingly indexed image."
        )

        config_kwargs: dict = dict(
            system_instruction=get_combined_prompt(global_quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=list[CombinedDetectionResult],
        )
        if "gemini-3" in settings.gemini_model:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=settings.gemini_thinking_level
            )
        config = types.GenerateContentConfig(**config_kwargs)

        quality_block = "\n".join(
            f"[FRAME {i + 1} QUALITY] {qc}" for i, qc in enumerate(per_frame_quality)
        )
        execution_query = (
            "Analyse EACH of the attached frames for synthetic generation artifacts, "
            "strictly following the system instructions. Apply the matching quality tag "
            "below to each correspondingly indexed image (image 1 → FRAME 1, etc.). "
            "Return one CombinedDetectionResult per image in the same order.\n\n"
            f"{quality_block}"
        )
        request_contents = image_parts + [execution_query]

        t0 = time.perf_counter()
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=request_contents,
            config=config,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000)

        raw_results = response.parsed
        representative_qc = per_frame_quality[0] if per_frame_quality else "**CONTEXT: QUALITY UNKNOWN.**"

        if hasattr(response, "usage_metadata"):
            logger.info("combined_video_batch_completed", extra={
                "action": "combined_video_batch_completed",
                "model": settings.gemini_model,
                "frame_count": len(image_sources),
                "duration_ms": duration_ms,
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "cost_usd": settings.gemini_fixed_cost,
            })

        if not raw_results:
            logger.error("combined_video_batch_empty_parsed", extra={
                "action": "combined_video_batch_empty_parsed",
                "frame_count": len(image_sources),
            })
            return {
                "confidence": -1.0,
                "signal_category": "multiple_subtle_ai_artifacts_present",
                "quality_context": representative_qc,
            }

        # ---------- HIGH-CONVICTION single-frame override ----------
        # A visible AI watermark or undeniable structural collapse in even
        # one frame is essentially proof — the majority vote would dismiss
        # it as a 1-vs-2 minority. If any frame crosses the override
        # threshold AND has a real region_anchor (mirrors the image path's
        # unanchored-AI demotion to prevent a confused frame triggering
        # this), return AI immediately with that frame's verdict.
        high_thr = settings.video_high_conviction_threshold
        for r in raw_results:
            anchor = (getattr(r, "region_anchor", "") or "").strip().lower()
            anchored = anchor not in ("", "none", "n/a", "na")
            if r.confidence >= high_thr and anchored:
                v2_cat = r.signal_category
                legacy = V2_TO_LEGACY_CATEGORY.get(v2_cat, "multiple_subtle_ai_artifacts_present")
                logger.info("combined_video_high_conviction_override", extra={
                    "action": "combined_video_high_conviction_override",
                    "frame_confidence": r.confidence,
                    "frame_signal_category": v2_cat,
                    "region_anchor": r.region_anchor,
                })
                return {
                    "confidence": round(r.confidence, 2),
                    "signal_category": legacy,
                    "quality_context": representative_qc,
                }

        # ---------- per-frame vote ----------
        ai_votes = [r for r in raw_results if r.confidence > settings.gemini_ai_vote_threshold]
        not_ai_votes = [r for r in raw_results if r.confidence <= settings.gemini_ai_vote_threshold]

        if len(ai_votes) > len(not_ai_votes):
            avg = sum(r.confidence for r in ai_votes) / len(ai_votes)
            best = max(ai_votes, key=lambda x: x.confidence)
            v2_cat = best.signal_category
            legacy = V2_TO_LEGACY_CATEGORY.get(v2_cat, "multiple_subtle_ai_artifacts_present")
            return {
                "confidence": round(avg, 2),
                "signal_category": legacy,
                "quality_context": representative_qc,
            }

        if not_ai_votes:
            avg = sum(r.confidence for r in not_ai_votes) / len(not_ai_votes)
            best = min(not_ai_votes, key=lambda x: x.confidence)
            v2_cat = best.signal_category
        else:
            avg = 0.0
            v2_cat = "no_visual_anomalies_detected"
        legacy = V2_TO_LEGACY_CATEGORY.get(v2_cat, "no_visual_anomalies_detected")
        return {
            "confidence": round(avg, 2),
            "signal_category": legacy,
            "quality_context": representative_qc,
        }

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        from google.genai import errors as _genai_errors
        is_api_error = isinstance(exc, _genai_errors.APIError)
        log_event = "combined_video_batch_api_error" if is_api_error else "combined_video_batch_code_error"
        log_method = logger.warning if is_api_error else logger.error
        log_method(log_event, extra={
            "action": log_event,
            "frame_count": len(image_sources),
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        return {"confidence": -1.0}


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
