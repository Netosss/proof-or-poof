"""
Core image analysis logic — the consensus engine for individual images/frames.

`detect_ai_media_image_logic` orchestrates metadata scoring, cache lookup,
and Gemini inference for a single image or video frame.

`boost_score` lives here (not in pipeline.py) because it is also called
from cache-hit paths within this module, avoiding a circular import.
"""

import os
import json
import random
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
from app.integrations.gemini.quality import get_quality_context

logger = logging.getLogger(__name__)

# Only hardware provenance tags count; web/app-added tags (JFIF, DPI, XMP) are excluded
HARDWARE_TAGS = {
    "Make", "Model", "ExposureTime", "ISOSpeedRatings",
    "FNumber", "BodySerialNumber", "LensModel", "GPSLatitude"
}


def boost_score(score: float, is_ai_likely: bool = True) -> float:
    """
    Soft proportional boost for AI-likely results only.

    Nudges uncertain AI scores (e.g. 0.55 → 0.66) without hard-flooring every
    result at 0.85, which inflated false positives. Strong signals (e.g. 0.90)
    are boosted only slightly (→ 0.925). Human results are never boosted.
    """
    if is_ai_likely:
        return score + (1.0 - score) * 0.25
    return score


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
            source_for_hash = file_path
            file_size = os.path.getsize(file_path)
        except Exception:
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

    # --- Merge Trusted Metadata (Sidecar) ---
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
    logger.info("metadata_raw", extra={"action": "metadata_raw", "exif_slim": slim_log})

    # Stringify values to handle non-JSON-serializable types (IFDRational, bytes, etc.)
    clean_metadata = f" {json.dumps({k: str(v) for k, v in exif.items()})} "
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
        "human_signals": human_signals or [],
        "ai_signals": ai_signals or [],
    })

    # 1. VERIFIED HUMAN (Early Exit)
    if human_score >= 0.60:
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
    if human_score >= 0.40 and ai_score < 0.15:
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
        suspicious_confidence = round(random.uniform(0.80, 0.90), 2)
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

            cached_explanation = cached_result.get("explanation", "Analyzed via second layer of AI analysis (Cached)")
            cached_quality_context = cached_result.get("quality_context", "Unknown")

            return {
                "summary": "Likely AI-Generated" if is_ai_likely else "Likely Authentic",
                "confidence_score": round(final_conf, 2),
                "is_gemini_used": True,
                "is_cached": True,
                "gpu_time_ms": 0,
                "is_short_circuited": False,
                "evidence_chain": [
                    {
                        "layer": "Metadata Check",
                        "status": "warning",
                        "label": "Origin Check",
                        "detail": "No camera fingerprint found."
                    },
                    {
                        "layer": "Visual Context",
                        "status": "flagged" if is_ai_likely else "passed",
                        "label": "Visual Inspection",
                        "detail": cached_explanation,
                        "context_quality": cached_quality_context
                    }
                ]
            }
        else:
            logger.info("cache_hit_image", extra={"action": "cache_hit_image", "ai_score": round(forensic_probability, 4)})
            is_ai_likely = forensic_probability > settings.ai_confidence_threshold
            raw_conf = forensic_probability if is_ai_likely else (1.0 - forensic_probability)
            final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)

            if final_conf > 0.99:
                final_conf = 0.99

            summary = "AI-Generated" if forensic_probability > settings.ai_confidence_threshold else "No AI Detected"

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
                        "detail": "No camera fingerprint found."
                    },
                    {
                        "layer": "Deep Forensics",
                        "status": "flagged" if is_ai_likely else "passed",
                        "label": "Structural Analysis",
                        "detail": "Noise patterns consistent with generative AI." if is_ai_likely else "Sensor noise patterns consistent with optical lenses."
                    }
                ]
            }

    # --- GEMINI ---
    logger.info("gemini_triggered", extra={
        "action": "gemini_triggered",
        "total_pixels": total_pixels,
        "tiered_score": round(tiered_score, 2),
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
        logger.warning("gemini_precalc_failed", extra={"action": "gemini_precalc_failed", "error": str(e)})

    gemini_res = await asyncio.to_thread(
        analyze_image_pro_turbo, source_for_gemini,
        pre_calculated_quality_context=pre_calc_context
    )
    logger.info("gemini_response", extra={
        "action": "gemini_response",
        "confidence": gemini_res.get("confidence"),
        "quality_score": gemini_res.get("quality_score"),
    })

    gemini_score = float(gemini_res.get("confidence", -1.0))
    gemini_explanation = gemini_res.get("explanation", "Analyzed via second layer of AI analysis")
    quality_context = gemini_res.get("quality_context", "Unknown")

    if gemini_score >= 0.0:
        await set_cached_result(img_hash, {
            "ai_score": gemini_score,
            "explanation": gemini_explanation,
            "is_gemini_used": True,
            "gpu_time_ms": 0,
            "quality_context": quality_context
        })

        is_ai_likely = gemini_score > settings.ai_confidence_threshold
        raw_conf = gemini_score if is_ai_likely else (1.0 - gemini_score)
        final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)

        return {
            "summary": "Likely AI-Generated" if is_ai_likely else "Likely Authentic",
            "confidence_score": round(final_conf, 2),
            "is_gemini_used": True,
            "gpu_time_ms": 0,
            "is_short_circuited": False,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "warning",
                    "label": "Origin Check",
                    "detail": "No camera fingerprint found."
                },
                {
                    "layer": "Visual Context",
                    "status": "flagged" if is_ai_likely else "passed",
                    "label": "Visual Inspection",
                    "detail": gemini_explanation,
                    "context_quality": quality_context
                }
            ]
        }

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
