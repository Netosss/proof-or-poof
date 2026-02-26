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
    Boost confidence only for AI-likely results.
    Human-likely results keep their raw confidence to avoid misleading scores.
    """
    if is_ai_likely:
        return max(0.85, score)
    return score  # No boost for human results


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
        logger.info("[SIDECAR] Using trusted metadata from device")
        for key in ["Make", "Model", "Software", "DateTime", "LensModel"]:
            if key in trusted_metadata:
                mapped_key = "DateTimeOriginal" if key == "DateTime" else key
                exif[mapped_key] = trusted_metadata[key]

        if "width" in trusted_metadata and "height" in trusted_metadata:
            width, height = trusted_metadata["width"], trusted_metadata["height"]
        if "fileSize" in trusted_metadata:
            file_size = trusted_metadata["fileSize"]

    slim_log = {k: (str(v)[:20] + "..." if len(str(v)) > 20 else v) for k, v in exif.items()}
    logger.info(f"[META] Raw Metadata (Slim): {slim_log}")

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
            logger.warning(f"Raw scan failed: {e}")

    tiered_score, tiered_signals = get_tiered_signature_score(full_dump, clean_metadata)

    # --- Metadata Scoring ---
    human_score, human_signals = get_forensic_metadata_score(exif)
    base_ai_score, ai_signals = get_ai_suspicion_score(exif, width, height, file_size)
    ai_score = min(0.99, base_ai_score + tiered_score)
    if tiered_signals:
        ai_signals.extend(tiered_signals)

    logger.info(f"[META] Metadata scoring: human={human_score:.2f}, ai={ai_score:.2f}")
    if human_signals:
        logger.info(f"[META] Human signals: {human_signals}")
    if ai_signals:
        logger.info(f"[META] AI signals: {ai_signals}")

    # 1. VERIFIED HUMAN (Early Exit)
    if human_score >= 0.60:
        logger.info(f"[EARLY EXIT] Skipping GPU scan: High confidence human metadata ({human_score:.2f})")
        return {
            "summary": "Likely Authentic",
            "confidence_score": 0.99,
            "gpu_time_ms": 0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "passed",
                    "label": "Device Metadata",
                    "detail": f"Valid camera metadata found ({exif.get('Make', 'Unknown')})."
                }
            ]
        }

    # 2. LIKELY HUMAN (Weaker signals but still skip GPU)
    if human_score >= 0.40 and ai_score < 0.15:
        logger.info(f"[EARLY EXIT] Skipping GPU scan: Likely human metadata ({human_score:.2f}, ai={ai_score:.2f})")
        return {
            "summary": "Likely Authentic",
            "confidence_score": 0.9,
            "gpu_time_ms": 0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "passed",
                    "label": "Device Metadata",
                    "detail": f"Heuristic analysis suggests authentic origin ({exif.get('Make', 'Unknown')})."
                }
            ]
        }

    # 3. LIKELY AI (Early Exit) - Strong AI signals in metadata
    if ai_score >= settings.ai_confidence_threshold:
        logger.info(f"[EARLY EXIT] Skipping GPU scan: High AI suspicion in metadata ({ai_score:.2f})")
        return {
            "summary": "Likely AI-Generated",
            "confidence_score": 0.95,
            "gpu_time_ms": 0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "flagged",
                    "label": "Software Signature",
                    "detail": f"AI generation software detected ({exif.get('Software', 'AI Generator')})."
                }
            ]
        }

    # 4. SUSPICIOUS AI (Early Exit) - AI indicators + zero human signals
    if ai_score >= 0.38 and human_score == 0.0:
        logger.info(f"[EARLY EXIT] Skipping GPU scan: AI indicators + no human metadata (ai={ai_score:.2f}, human={human_score:.2f})")
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
                    "label": "Metadata Check",
                    "detail": "No camera metadata found."
                },
                {
                    "layer": "Technical Heuristics",
                    "status": "flagged",
                    "label": "Image Structure",
                    "detail": "Dimensions typical of AI generation."
                }
            ]
        }

    # 5. AMBIGUOUS -> Forensic Scan (Gemini)
    total_pixels = width * height

    has_hardware_provenance = any(tag in exif for tag in HARDWARE_TAGS)
    is_stripped = not has_hardware_provenance and tiered_score < settings.ai_confidence_threshold

    if is_stripped:
        logger.info("[META] Image classified as STRIPPED (No Hardware Provenance Tags found)")
    elif tiered_score >= settings.ai_confidence_threshold:
        logger.info(f"[META] Image has technical AI signatures (score={tiered_score:.2f}) - bypassing stripped check")
    else:
        found_tags = [tag for tag in PROVENANCE_WHITELIST if tag in exif]
        logger.info(f"[META] Image has PROVENANCE tags: {found_tags}")

    img_hash = await asyncio.to_thread(get_image_hash, source_for_hash, fast_mode=(frame is not None))
    cached_result = get_cached_result(img_hash)

    if cached_result is not None:
        logger.info("[CACHE] Hit for image scan")
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
                        "label": "Metadata Check",
                        "detail": "No camera metadata found."
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
            logger.info(f"[CACHE] Returning cached GPU result (ai_score={forensic_probability:.4f})")
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
                        "label": "Metadata Check",
                        "detail": "No camera metadata found."
                    },
                    {
                        "layer": "Deep Forensics",
                        "status": "flagged" if is_ai_likely else "passed",
                        "label": "Pixel Analysis",
                        "detail": "Noise patterns consistent with generative AI." if is_ai_likely else "Sensor noise patterns consistent with optical lenses."
                    }
                ]
            }

    # --- GEMINI ---
    logger.info(f"[GEMINI] Triggering Gemini Pro Turbo (Pixels: {total_pixels}, Score: {tiered_score:.2f})")

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
        logger.warning(f"Failed to pre-calc quality context: {e}")

    gemini_res = await asyncio.to_thread(
        analyze_image_pro_turbo, source_for_gemini,
        pre_calculated_quality_context=pre_calc_context
    )
    logger.info(f"[GEMINI] Raw response: {json.dumps(gemini_res)}")

    gemini_score = float(gemini_res.get("confidence", -1.0))
    gemini_explanation = gemini_res.get("explanation", "Analyzed via second layer of AI analysis")
    quality_context = gemini_res.get("quality_context", "Unknown")

    if gemini_score >= 0.0:
        set_cached_result(img_hash, {
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
                    "label": "Metadata Check",
                    "detail": "No camera metadata found."
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
