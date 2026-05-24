"""
Core image analysis logic — the consensus engine for individual images/frames.

`detect_ai_media_image_logic` orchestrates metadata scoring, cache lookup,
and Gemini inference for a single image or video frame.

`boost_score` lives here (not in pipeline.py) because it is also called
from cache-hit paths within this module, avoiding a circular import.
"""

import os
import json
import asyncio
import logging
from typing import Optional
from PIL import Image

from app.config import settings
from app.detection.constants import PROVENANCE_WHITELIST
from app.detection.hashing import get_image_hash
from app.detection.cache import get_cached_result, set_cached_result
from app.detection.metadata_scorer import (
    get_exif_data,
    get_tiered_signature_score,
    get_forensic_metadata_score,
    get_ai_suspicion_score,
)
from app.integrations.gemini.client import analyze_image_pro_turbo
from app.integrations.gemini.client_v2 import analyze_image_pro_turbo_v2
from app.detection.ensemble_engine import analyze_image_ensemble
from app.integrations.gemini.quality import get_quality_context


def _select_analyzer():
    """Dispatch to v1, v2, or ensemble engine based on settings.detection_engine."""
    if settings.detection_engine == "ensemble":
        return analyze_image_ensemble
    if settings.detection_engine == "v2":
        return analyze_image_pro_turbo_v2
    return analyze_image_pro_turbo

logger = logging.getLogger(__name__)

# Only hardware provenance tags count; web/app-added tags (JFIF, DPI, XMP) are excluded
HARDWARE_TAGS = {
    "Make", "Model", "ExposureTime", "ISOSpeedRatings",
    "FNumber", "BodySerialNumber", "LensModel", "GPSLatitude"
}

# Sensor-physics signals — fields a real camera writes directly from sensor / lens
# hardware. None of these are present on phone screenshots, web re-downloads, or
# stripped-EXIF AI images. Required (any one) before the human early-exit paths
# in detect_ai_media_image_logic short-circuit a verdict.
#
# DateTime fields are intentionally excluded — they're string-typed and trivially
# spoofable. Optical-physics fields (FocalLength, FocalPlaneXResolution) are
# included alongside the core triad because they're equally hard to forge by
# screenshot but present on more real-world processed photos.
SENSOR_PHYSICS_SIGNALS = frozenset({
    "ExposureTime", "ISOSpeedRatings", "FNumber",
    "FocalLength", "FocalPlaneXResolution",
})

def _label_for(signal_category: str) -> str:
    """Convert a snake_case signal_category key to a human-readable label."""
    return signal_category.replace("_", " ").capitalize()


def boost_score(score: float, is_ai_likely: bool = True) -> float:
    """
    Soft proportional boost for AI-likely results, capped at 0.99.

    Nudges uncertain AI scores (e.g. 0.55 → 0.66) without hard-flooring every
    result at 0.85. Strong signals (e.g. 0.90) are boosted only slightly
    (→ 0.925). Human results are passed through unmodified. The 0.99 cap
    lives here so callers cannot accidentally emit absolute-certainty (1.0)
    verdicts — UX policy never shows 100% certainty either way.
    """
    if is_ai_likely:
        boosted = score + (1.0 - score) * 0.25
        return min(0.99, boosted)
    return min(0.99, score)


def _build_gemini_evidence_response(
    *,
    summary: str,
    confidence: float,
    is_ai_likely: bool,
    visual_detail: str,
    context_quality: str,
    is_gemini_used: bool,
    is_cached: bool,
) -> dict:
    """
    Build the Gemini-path response envelope.

    Single source of truth for the 3 sites that emit a Gemini-derived verdict
    (cache hit with prior Gemini result, fresh Gemini success, and cached GPU
    forensic result). Schema drift bugs from copy-paste evidence_chain edits
    are mechanically prevented.
    """
    return {
        "summary": summary,
        "confidence_score": round(confidence, 2),
        "is_gemini_used": is_gemini_used,
        "is_cached": is_cached,
        "gpu_time_ms": 0,
        "is_short_circuited": False,
        "evidence_chain": [
            {
                "layer": "Metadata Check",
                "status": "warning",
                "label": "Origin Check",
                "detail": "No camera fingerprint found.",
            },
            {
                "layer": "Visual Context",
                "status": "flagged" if is_ai_likely else "passed",
                "label": "Visual Inspection",
                "detail": visual_detail,
                "context_quality": context_quality,
            },
        ],
    }


async def detect_ai_media_image_logic(
    file_path: Optional[str],
    l1_data: dict = None,
    frame: Image.Image = None,
    trusted_metadata: dict = None
) -> dict:
    """
    Core consensus logic for images and video frames.
    """
    if l1_data is None:
        l1_data = {"status": "not_found", "provider": None, "description": "N/A"}

    file_size = 0
    if frame:
        img_for_res = frame
        exif = {}
        source_for_hash = frame
        width, height = img_for_res.size
    else:
        exif = await asyncio.to_thread(get_exif_data, file_path)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                img_format = (img.format or "unknown").lower()
            source_for_hash = file_path
            file_size = os.path.getsize(file_path)
        except Exception as open_err:
            logger.error("image_open_failed", extra={
                "action": "image_open_failed",
                "file_path_hint": os.path.basename(file_path) if file_path else "none",
                "error": str(open_err),
                "error_type": type(open_err).__name__,
            })
            return {
                "summary": "Analysis Failed",
                "confidence_score": 0.0,
                "is_short_circuited": False,
                "evidence_chain": [
                    {
                        "layer": "System",
                        "status": "warning",
                        "label": "File Error",
                        "detail": "Invalid image file - could not open."
                    }
                ]
            }

    # Compute sensor-physics signal BEFORE merging trusted metadata. The mobile
    # sidecar deliberately does NOT carry ExposureTime/ISO/FNumber (those would
    # be a client-spoofable forgery surface), but real EXIF on disk does. The
    # early-exit human paths below need to read this from the on-disk EXIF only.
    has_physical_signals = any(tag in exif for tag in SENSOR_PHYSICS_SIGNALS)

    # --- Merge Trusted Metadata (Sidecar) ---
    # SECURITY BOUNDARY: this allow-list intentionally excludes ExposureTime,
    # ISOSpeedRatings, FNumber, FocalLength, FocalPlaneXResolution. These are
    # the sensor-physics anchors used by `has_physical_signals` above and must
    # ONLY originate from real on-disk EXIF — never from the mobile sidecar —
    # so a malicious client cannot fake "real camera" provenance on an AI image.
    if trusted_metadata:
        logger.info("metadata_sidecar_used", extra={"action": "metadata_sidecar_used"})
        for key in ["Make", "Model", "Software", "DateTime", "LensModel"]:
            if key in trusted_metadata:
                mapped_key = "DateTimeOriginal" if key == "DateTime" else key
                exif[mapped_key] = trusted_metadata[key]

        if "width" in trusted_metadata and "height" in trusted_metadata:
            width, height = trusted_metadata["width"], trusted_metadata["height"]
        if "fileSize" in trusted_metadata:
            file_size = trusted_metadata["fileSize"]

    slim_log = {k: (str(v)[:20] + "..." if len(str(v)) > 20 else str(v)) for k, v in exif.items()}
    try:
        exif_json = json.dumps(slim_log, default=str)
    except Exception:
        exif_json = "{}"
    logger.info("metadata_raw", extra={
        "action": "metadata_raw",
        "exif_slim": exif_json,
        "exif_key_count": len(exif),
        "dimensions": f"{width}x{height}",
        "file_size_bytes": file_size,
    })

    # Stringify keys (PIL returns int EXIF tag IDs / bytes for IFD sub-blocks)
    # AND fall back default=str for values (IFDRational, bytes, nested tuples).
    # Belt-and-suspenders — without both, json.dumps raises TypeError on
    # certain images with unusual EXIF structures.
    try:
        clean_metadata = f" {json.dumps({str(k): v for k, v in exif.items()}, default=str)} "
    except (TypeError, ValueError):
        clean_metadata = " {} "
    full_dump = clean_metadata

    if not frame and file_path and os.path.exists(file_path):
        try:
            def _read_header():
                with open(file_path, 'rb') as f:
                    return f.read(50000)

            raw_header = await asyncio.to_thread(_read_header)
            full_dump += raw_header.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.warning("metadata_raw_scan_failed", extra={"action": "metadata_raw_scan_failed", "error": str(e)})

    tiered_score, tiered_signals = get_tiered_signature_score(full_dump, clean_metadata)

    # --- Metadata Scoring ---
    human_score, human_signals = get_forensic_metadata_score(exif)
    base_ai_score, ai_signals = get_ai_suspicion_score(exif, width, height, file_size)
    ai_score = min(0.99, base_ai_score + tiered_score)
    if tiered_signals:
        ai_signals.extend(tiered_signals)

    logger.info("metadata_scoring", extra={
        "action": "metadata_scoring",
        "human_score": round(human_score, 2),
        "ai_score": round(ai_score, 2),
        "has_physical_signals": has_physical_signals,
        "human_signals": human_signals or [],
        "ai_signals": ai_signals or [],
    })

    # 1. VERIFIED HUMAN (Early Exit)
    # Requires both a high human score AND at least one physical camera signal —
    # device manufacturer + OS strings alone are not enough; a phone SCREENSHOT
    # of an AI image carries Make/Software but no ExposureTime/ISO/FNumber.
    if human_score >= 0.60 and has_physical_signals:
        logger.info("detection_early_exit_human", extra={
            "action": "detection_early_exit_human",
            "human_score": round(human_score, 2),
            "reason": "high_confidence_human_metadata",
        })
        return {
            "summary": "Likely Authentic",
            "confidence_score": 0.99,
            "gpu_time_ms": 0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "passed",
                    "label": "Origin Check",
                    "detail": f"Confirmed authentic — captured by {exif.get('Make', 'your camera')}."
                }
            ]
        }

    # 2. LIKELY HUMAN (Weaker signals but still skip GPU)
    # Same physical-camera-signal guard as #1 — block phone screenshots from
    # short-circuiting just because they carry device/OS metadata.
    if human_score >= 0.40 and ai_score < 0.15 and has_physical_signals:
        logger.info("detection_early_exit_human", extra={
            "action": "detection_early_exit_human",
            "human_score": round(human_score, 2),
            "ai_score": round(ai_score, 2),
            "reason": "likely_human_metadata",
        })
        return {
            "summary": "Likely Authentic",
            "confidence_score": 0.9,
            "gpu_time_ms": 0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "passed",
                    "label": "Origin Check",
                    "detail": f"Origin analysis indicates authentic capture ({exif.get('Make', 'your camera')})."
                }
            ]
        }

    # 3. LIKELY AI (Early Exit) - Strong AI signals in metadata
    if ai_score >= settings.ai_confidence_threshold:
        logger.info("detection_early_exit_ai", extra={
            "action": "detection_early_exit_ai",
            "ai_score": round(ai_score, 2),
            "reason": "high_ai_suspicion_metadata",
        })
        return {
            "summary": "Likely AI-Generated",
            "confidence_score": 0.95,
            "gpu_time_ms": 0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "flagged",
                    "label": "Origin Check",
                    "detail": f"AI creation tool identified ({exif.get('Software', 'Unknown tool')})."
                }
            ]
        }

    # 4. SUSPICIOUS AI (Early Exit) - AI indicators + zero human signals
    if ai_score >= 0.38 and human_score == 0.0:
        logger.info("detection_early_exit_ai", extra={
            "action": "detection_early_exit_ai",
            "ai_score": round(ai_score, 2),
            "human_score": round(human_score, 2),
            "reason": "ai_indicators_no_human_metadata",
        })
        # Deterministic confidence derived from the underlying ai_score so the
        # same image always produces the same verdict (was previously
        # random.uniform(0.80, 0.90) which made the system non-reproducible and
        # broke any downstream consumer relying on confidence stability).
        # Map the [0.38, 0.55) ai_score range linearly into [0.80, 0.90].
        suspicious_confidence = round(
            0.80 + min(1.0, max(0.0, (ai_score - 0.38) / 0.17)) * 0.10,
            2,
        )
        return {
            "summary": "AI-Generated",
            "confidence_score": suspicious_confidence,
            "gpu_time_ms": 0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "warning",
                    "label": "Origin Check",
                    "detail": "No camera fingerprint found."
                },
                {
                    "layer": "Technical Heuristics",
                    "status": "flagged",
                    "label": "Image Analysis",
                    "detail": "Image characteristics consistent with AI generation."
                }
            ]
        }

    # 5. AMBIGUOUS -> Forensic Scan (Gemini)
    total_pixels = width * height

    has_hardware_provenance = any(tag in exif for tag in HARDWARE_TAGS)
    is_stripped = not has_hardware_provenance and tiered_score < settings.ai_confidence_threshold

    if is_stripped:
        logger.info("metadata_stripped", extra={"action": "metadata_stripped"})
    elif tiered_score >= settings.ai_confidence_threshold:
        logger.info("metadata_ai_signatures", extra={
            "action": "metadata_ai_signatures",
            "tiered_score": round(tiered_score, 2),
        })
    else:
        found_tags = [tag for tag in PROVENANCE_WHITELIST if tag in exif]
        logger.info("metadata_provenance_tags", extra={
            "action": "metadata_provenance_tags",
            "found_tags": found_tags,
        })

    img_hash = await asyncio.to_thread(get_image_hash, source_for_hash, fast_mode=(frame is not None))
    cached_result = await get_cached_result(img_hash)

    if cached_result is not None:
        logger.info("cache_hit_image", extra={"action": "cache_hit_image"})
        forensic_probability = cached_result.get("ai_score", 0.0)
        is_gemini_used = cached_result.get("is_gemini_used", False)

        if is_gemini_used:
            is_ai_likely = forensic_probability > settings.ai_confidence_threshold
            raw_conf = forensic_probability if is_ai_likely else (1.0 - forensic_probability)
            final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)

            cached_signal = cached_result.get("signal_category", "multiple_subtle_ai_artifacts_present")
            cached_explanation = _label_for(cached_signal)
            cached_quality_context = cached_result.get("quality_context", "Unknown")

            return _build_gemini_evidence_response(
                summary="Likely AI-Generated" if is_ai_likely else "Likely Authentic",
                confidence=final_conf,
                is_ai_likely=is_ai_likely,
                visual_detail=cached_explanation,
                context_quality=cached_quality_context,
                is_gemini_used=True,
                is_cached=True,
            )
        else:
            logger.info("cache_hit_image", extra={"action": "cache_hit_image", "ai_score": round(forensic_probability, 4)})
            is_ai_likely = forensic_probability > settings.ai_confidence_threshold
            raw_conf = forensic_probability if is_ai_likely else (1.0 - forensic_probability)
            final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)

            summary = "AI-Generated" if forensic_probability > settings.ai_confidence_threshold else "No AI Detected"
            # Deep Forensics path uses a different second-row label/detail; emit
            # it inline rather than overloading the Gemini helper with a flag.
            return {
                "summary": summary,
                "confidence_score": round(final_conf, 2),
                "is_cached": True,
                "gpu_time_ms": 0,
                "is_short_circuited": False,
                "evidence_chain": [
                    {
                        "layer": "Metadata Check",
                        "status": "warning",
                        "label": "Origin Check",
                        "detail": "No camera fingerprint found.",
                    },
                    {
                        "layer": "Deep Forensics",
                        "status": "flagged" if is_ai_likely else "passed",
                        "label": "Structural Analysis",
                        "detail": (
                            "Noise patterns consistent with generative AI."
                            if is_ai_likely
                            else "Sensor noise patterns consistent with optical lenses."
                        ),
                    },
                ],
            }

    # --- GEMINI ---
    logger.info("gemini_triggered", extra={
        "action": "gemini_triggered",
        "total_pixels": total_pixels,
        "dimensions": f"{width}x{height}",
        "file_size_bytes": file_size,
        "tiered_score": round(tiered_score, 2),
        "human_score": round(human_score, 2),
        "ai_score": round(ai_score, 2),
        "source": "frame" if frame else "file",
    })

    pre_calc_context = None
    source_for_gemini = frame or file_path

    try:
        def _get_context_safe():
            if frame:
                return get_quality_context(frame)[0]
            else:
                with Image.open(file_path) as img:
                    return get_quality_context(img)[0]

        pre_calc_context = await asyncio.to_thread(_get_context_safe)
    except Exception as e:
        logger.warning("gemini_precalc_failed", extra={
            "action": "gemini_precalc_failed",
            "error": str(e),
            "error_type": type(e).__name__,
        })

    analyzer = _select_analyzer()
    gemini_res = await asyncio.to_thread(
        analyzer, source_for_gemini,
        pre_calculated_quality_context=pre_calc_context
    )
    logger.info("gemini_response", extra={
        "action": "gemini_response",
        "confidence": gemini_res.get("confidence"),
        "quality_score": gemini_res.get("quality_score"),
    })

    gemini_score = float(gemini_res.get("confidence", -1.0))
    gemini_signal = gemini_res.get("signal_category", "multiple_subtle_ai_artifacts_present")
    gemini_explanation = _label_for(gemini_signal)
    quality_context = gemini_res.get("quality_context", "Unknown")

    if gemini_score >= 0.0:
        await set_cached_result(img_hash, {
            "ai_score": gemini_score,
            "signal_category": gemini_signal,
            "is_gemini_used": True,
            "gpu_time_ms": 0,
            "quality_context": quality_context
        })

        is_ai_likely = gemini_score > settings.ai_confidence_threshold
        raw_conf = gemini_score if is_ai_likely else (1.0 - gemini_score)
        final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)

        return _build_gemini_evidence_response(
            summary="Likely AI-Generated" if is_ai_likely else "Likely Authentic",
            confidence=final_conf,
            is_ai_likely=is_ai_likely,
            visual_detail=gemini_explanation,
            context_quality=quality_context,
            is_gemini_used=True,
            is_cached=False,
        )

    return {
        "summary": "Analysis Failed",
        "confidence_score": 0.0,
        "is_short_circuited": False,
        "evidence_chain": [
            {
                "layer": "System",
                "status": "warning",
                "label": "Service Error",
                "detail": "Advanced analysis service unavailable."
            }
        ]
    }
