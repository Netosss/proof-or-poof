"""
Top-level detection pipeline — public entry point for the /detect route.

`detect_ai_media` orchestrates:
  1. C2PA manifest check (cryptographic provenance — instant early exit)
  2. Video path  → metadata scoring → tri-frame Gemini batch
  3. Image path  → detect_ai_media_image_logic (metadata → Gemini)
"""

import os
import asyncio
import logging

from app.integrations.c2pa import get_c2pa_manifest
from app.integrations.gemini.client import analyze_batch_images_pro_turbo
from app.detection.hashing import get_smart_file_hash
from app.detection.cache import get_cached_result, set_cached_result
from app.detection.video_detector import (
    extract_video_frames,
    get_video_metadata,
    get_video_metadata_score,
)
from app.detection.image_detector import detect_ai_media_image_logic
from app.config import settings

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif')


async def detect_ai_media(file_path: str, trusted_metadata: dict = None) -> dict:
    """
    Final Optimized Consensus Engine.

    Args:
        file_path: Path to the media file.
        trusted_metadata: Optional sidecar metadata from a mobile device.
            Bypasses OS privacy stripping. Fields: Make, Model, Software,
            DateTime, width, height, fileSize, namingEntropy, isOriginalPath, etc.
    """
    l1_data = {
        "status": "not_found",
        "provider": None,
        "description": "No cryptographic signature found."
    }

    # --- Layer 1: C2PA cryptographic provenance check ---
    manifest = await asyncio.to_thread(get_c2pa_manifest, file_path)

    if manifest:
        gen_info = manifest.get("claim_generator_info", [])
        generator = (
            gen_info[0].get("name", "Unknown AI") if gen_info
            else manifest.get("claim_generator", "Unknown AI")
        )

        is_generative_ai = False
        assertions = manifest.get("assertions", [])
        for assertion in assertions:
            if assertion.get("label") == "c2pa.actions.v2":
                actions = assertion.get("data", {}).get("actions", [])
                for action in actions:
                    source_type = action.get("digitalSourceType", "")
                    if "trainedAlgorithmicMedia" in source_type:
                        is_generative_ai = True
                    desc = action.get("description", "").lower()
                    if any(term in desc for term in ["generative fill", "ai-modified", "edited with ai", "ai generated"]):
                        is_generative_ai = True
            if is_generative_ai:
                break

        l1_data = {
            "status": "verified_ai" if is_generative_ai else "verified_human",
            "provider": generator,
            "description": (
                f"Verified AI signature found ({generator})."
                if is_generative_ai
                else "Verified human-captured content."
            )
        }

        return {
            "summary": "AI-Generated" if is_generative_ai else "No AI Detected",
            "confidence_score": 1.0,
            "is_short_circuited": True,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "flagged" if is_generative_ai else "passed",
                    "label": "Digital Signature",
                    "detail": (
                        f"Content Credentials confirm {'AI origin' if is_generative_ai else 'authentic origin'} ({generator})."
                    )
                }
            ]
        }

    # --- Layer 2: Video vs Image routing ---
    is_video = file_path.lower().endswith(VIDEO_EXTENSIONS)

    if is_video:
        filename = os.path.basename(file_path)
        logger.info(f"[PIPELINE] Detecting AI in video: {filename}")

        video_hash = await asyncio.to_thread(get_smart_file_hash, file_path)
        cached_video_result = get_cached_result(video_hash)

        if cached_video_result:
            logger.info(f"[CACHE] Hit for VIDEO scan: {video_hash[:8]}...")
            return cached_video_result

        video_metadata = await get_video_metadata(file_path)
        human_score, ai_meta_score, meta_signals, early_exit = get_video_metadata_score(
            video_metadata, filename, file_path
        )

        logger.info(f"[VIDEO META] Human={human_score:.2f}, AI={ai_meta_score:.2f}, early_exit={early_exit}")
        logger.info(f"[VIDEO META] Signals: {meta_signals}")

        if early_exit == "human":
            logger.info(f"[VIDEO] Early exit: Verified Human via metadata")
            res = {
                "summary": "No AI Detected",
                "confidence_score": 0.99,
                "is_short_circuited": True,
                "evidence_chain": [
                    {
                        "layer": "Metadata Check",
                        "status": "passed",
                        "label": "Device Metadata",
                        "detail": f"Valid video metadata found ({meta_signals[0] if meta_signals else 'Camera/Phone'})."
                    }
                ]
            }
            cached_version = res.copy()
            cached_version["is_cached"] = True
            set_cached_result(video_hash, cached_version)
            return res

        if early_exit == "ai":
            logger.info("[VIDEO] Early exit: AI Generator detected via metadata")
            res = {
                "summary": "AI-Generated",
                "confidence_score": 0.99,
                "is_short_circuited": True,
                "evidence_chain": [
                    {
                        "layer": "Metadata Check",
                        "status": "flagged",
                        "label": "Software Signature",
                        "detail": f"AI generator signature detected in metadata ({meta_signals[0] if meta_signals else 'Unknown'})."
                    }
                ]
            }
            cached_version = res.copy()
            cached_version["is_cached"] = True
            set_cached_result(video_hash, cached_version)
            return res

        logger.info("[VIDEO] No early exit, proceeding to tri-frame batch analysis...")

        loop = asyncio.get_running_loop()
        frames, quality_rejected = await loop.run_in_executor(
            None, extract_video_frames, file_path
        )

        if not frames:
            return {
                "summary": "Analysis Failed",
                "confidence_score": 0.0,
                "is_short_circuited": False,
                "evidence_chain": [
                    {
                        "layer": "System",
                        "status": "warning",
                        "label": "Video Error",
                        "detail": "Could not extract frames from video."
                    }
                ]
            }

        logger.info(f"[VIDEO] Extracted {len(frames)} frames (rejected {quality_rejected} low-quality)")

        gemini_result = await loop.run_in_executor(
            None, analyze_batch_images_pro_turbo, frames
        )

        confidence = gemini_result.get("confidence", 0.0)
        explanation = gemini_result.get("explanation", "Analysis completed.")
        quality_context = gemini_result.get("quality_context", "Unknown")

        logger.info(
            f"[VIDEO] Gemini Batch: confidence={confidence}, explanation='{explanation}'"
        )

        is_ai_likely = confidence > settings.ai_confidence_threshold
        summary = "Likely AI-Generated" if is_ai_likely else "Likely Authentic"
        final_conf = confidence if is_ai_likely else (1.0 - confidence)

        final_video_result = {
            "summary": summary,
            "confidence_score": final_conf,
            "gpu_time_ms": 0,
            "is_gemini_used": True,
            "is_short_circuited": False,
            "evidence_chain": [
                {
                    "layer": "Metadata Check",
                    "status": "warning",
                    "label": "Metadata Check",
                    "detail": "No definitive camera metadata found."
                },
                {
                    "layer": "Visual Context",
                    "status": "flagged" if is_ai_likely else "passed",
                    "label": "Visual Inspection",
                    "detail": explanation,
                    "context_quality": quality_context
                }
            ]
        }

        cached_version = final_video_result.copy()
        cached_version["is_cached"] = True
        set_cached_result(video_hash, cached_version)

        return final_video_result

    # --- Image path ---
    return await detect_ai_media_image_logic(file_path, l1_data, trusted_metadata=trusted_metadata)
