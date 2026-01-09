
import logging
import time
import os
import random
from PIL import Image
from typing import Optional
import hmac
import hashlib

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    # numpy is only needed for frame-based paths; file-path based detection can run without it.
    np = None  # type: ignore

from app.detectors.utils import LRUCache, get_image_hash, get_exif_data
from app.detectors.video import extract_video_frames, get_video_metadata, get_video_metadata_score
from app.detectors.metadata import get_forensic_metadata_score, get_ai_suspicion_score
from app.c2pa_reader import get_c2pa_manifest
from app.runpod_client import run_deep_forensics, run_batch_forensics  # Lazy import might be better if circular deps arise
from app.security import security_manager
from app.scoring_config import ScoringConfig

logger = logging.getLogger(__name__)

forensic_cache = LRUCache(capacity=1000)

def _benchmark_mode_enabled() -> bool:
    """
    Benchmark mode disables behaviors that distort offline evaluation:
    - conflict-resolution that can override the (mocked) GPU output using metadata
    - caching (benchmark harness also patches forensic_cache, but keep this as a safety net)
    """
    return os.getenv("AI_DETECTOR_BENCHMARK", "0").lower() in {"1", "true", "yes"}

def _log_decision(result: dict, source: str) -> dict:
    """Helper to log the final decision before returning."""
    try:
        summary = result.get("summary", "N/A")
        conf = result.get("confidence_score", 0.0)
        meta = result.get("metadata", {}) or {}
        # Handle cases where metadata might be nested differently or missing
        h_score = meta.get("human_score", 0.0)
        a_score = meta.get("ai_score", 0.0)
        
        # If it's a layer structure (Video), extract from layer1 if metadata is empty
        if not meta and "layers" in result:
            l1 = result["layers"].get("layer1_metadata", {})
            h_score = l1.get("human_score", 0.0)
            a_score = l1.get("ai_score", 0.0)

        logger.info(f"[DECISION] Verdict: {summary} ({conf:.2f}) | Source: {source} | Scores: H={h_score}, A={a_score}")
    except Exception as e:
        logger.error(f"[LOGGING] Error logging decision: {e}")
    return result

def _verify_capture_signature(trusted_metadata: dict) -> bool:
    """
    Optional hardening: if CAPTURE_HMAC_SECRET is set, require a valid signature before
    trusting captured_in_app. If not set, we assume the endpoint is not publicly exposed.
    """
    secret = os.getenv("CAPTURE_HMAC_SECRET", "")
    if not secret:
        return True
    sig = str((trusted_metadata or {}).get("capture_signature") or "")
    if not sig:
        return False
    sid = str((trusted_metadata or {}).get("capture_session_id") or "")
    ts = str((trusted_metadata or {}).get("capture_timestamp_ms") or (trusted_metadata or {}).get("capture_timestamp") or "")
    path = str((trusted_metadata or {}).get("capture_path") or "")
    payload = f"{sid}|{ts}|{path}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)

def boost_score(score: float, is_ai_likely: bool = True) -> float:
    """
    Boost confidence only for AI-likely results.
    Human-likely results keep their raw confidence to avoid misleading scores.
    """
    if is_ai_likely:
        return max(0.85, score)
    return score  # No boost for human results

async def detect_ai_media(file_path: str, trusted_metadata: dict = None, original_filename: str = None) -> dict:
    """
    Final Optimized Consensus Engine.
    """
    total_start = time.perf_counter()
    
    l1_data = {
        "status": "not_found",
        "provider": None,
        "description": "No cryptographic signature found."
    }

    # --- 1️⃣ LAYER 1: C2PA ---
    t_c2pa = time.perf_counter()
    manifest = get_c2pa_manifest(file_path)
    c2pa_time_ms = (time.perf_counter() - t_c2pa) * 1000
    logger.info(f"[TIMING] Layer 1 - C2PA check: {c2pa_time_ms:.2f}ms")
    if manifest:
        gen_info = manifest.get("claim_generator_info", [])
        generator = gen_info[0].get("name", "Unknown AI") if gen_info else manifest.get("claim_generator", "Unknown AI")

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
            if is_generative_ai: break

        l1_data = {
            "status": "verified_ai" if is_generative_ai else "verified_original",
            "provider": generator,
            "description": f"Verified AI signature found ({generator})." if is_generative_ai else "Verified original content."
        }

        return _log_decision({
            "summary": "Verified AI" if is_generative_ai else "Verified Original",
            "confidence_score": 1.0,
            "layers": {
                "layer1_metadata": l1_data,
                "layer2_forensics": {
                    "status": "skipped", 
                    "probability": 1.0 if is_generative_ai else 0.0, 
                    "signals": ["Source verified via cryptographic signature."]
                }
            },
            "metadata": {
                "human_score": 0.0 if is_generative_ai else 1.0,
                "ai_score": 1.0 if is_generative_ai else 0.0,
                "signals": ["C2PA cryptographic signature"],
                "bypass_reason": "c2pa"
            }
        }, "C2PA Signature")

    # --- IN-APP CAPTURE (Strongest Human Signal) ---
    # Only trust this if:
    # - captured_in_app=true AND
    # - capture_session_id + capture timestamp exist AND
    # - (optional) HMAC signature is valid when CAPTURE_HMAC_SECRET is configured.
    if trusted_metadata and trusted_metadata.get("captured_in_app") is True:
        has_sid = bool(trusted_metadata.get("capture_session_id"))
        has_ts = bool(trusted_metadata.get("capture_timestamp_ms") or trusted_metadata.get("capture_timestamp"))
        if has_sid and has_ts and _verify_capture_signature(trusted_metadata):
            return _log_decision({
                "summary": "Verified Original (In-App Capture)",
                "confidence_score": 0.99,
                "layers": {
                    "layer1_metadata": {
                        "status": "verified_original",
                        "provider": "InAppCapture",
                        "description": "Captured inside the app (trusted capture session).",
                        "human_score": 0.70,
                        "ai_score": 0.0,
                        "signals": [
                            "Captured in app (strong provenance)"
                        ]
                    },
                    "layer2_forensics": {
                        "status": "skipped",
                        "probability": 0.0,
                        "signals": [
                            "Captured in app (captured_in_app=true)",
                            f"capture_session_id={trusted_metadata.get('capture_session_id')}",
                            f"capture_timestamp_ms={trusted_metadata.get('capture_timestamp_ms') or trusted_metadata.get('capture_timestamp')}",
                        ],
                    },
                },
                "gpu_time_ms": 0.0,
                "gpu_bypassed": True,
                "metadata": {
                    "human_score": 0.70,
                    "ai_score": 0.0,
                    "signals": [
                        "Captured in app (strong provenance)"
                    ],
                    "extracted": {
                        "capture_session_id": trusted_metadata.get("capture_session_id"),
                        "capture_timestamp_ms": trusted_metadata.get("capture_timestamp_ms") or trusted_metadata.get("capture_timestamp"),
                        "capture_path": trusted_metadata.get("capture_path"),
                    },
                    "bypass_reason": "captured_in_app",
                }
            }, "In-App Capture")

    is_video = file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
    
    if is_video:
        safe_path = security_manager.sanitize_log_message(file_path)
        filename = os.path.basename(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Detecting AI in video: {safe_path} ({file_size_mb:.1f}MB)")
        
        if file_size_mb > 100:
            logger.warning(f"[VIDEO] Large file ({file_size_mb:.0f}MB) - processing may take longer")
        
        # --- VIDEO METADATA EARLY EXIT ---
        t_video_meta = time.perf_counter()
        video_metadata = await get_video_metadata(file_path)
        human_score, ai_meta_score, meta_signals, early_exit = get_video_metadata_score(video_metadata, filename, file_path)
        
        if early_exit == "original":
            return {
                "summary": "Verified Original Video",
                "confidence_score": 0.99,
                "layers": {
                    "layer1_metadata": {
                        "status": "verified_original",
                        "provider": meta_signals[0] if meta_signals else "Camera/Phone",
                        "description": "Video recorded on real device with authentic metadata.",
                        "human_score": human_score,
                        "ai_score": ai_meta_score
                    },
                    "layer2_forensics": {
                        "status": "skipped",
                        "probability": 0.0,
                        "signals": meta_signals
                    }
                }
            }
        
        if early_exit == "ai":
            return {
                "summary": "Verified AI Video",
                "confidence_score": 0.99,
                "layers": {
                    "layer1_metadata": {
                        "status": "verified_ai",
                        "provider": meta_signals[0] if meta_signals else "AI Generator",
                        "description": "Video generated by AI tool detected in metadata.",
                        "human_score": human_score,
                        "ai_score": ai_meta_score
                    },
                    "layer2_forensics": {
                        "status": "skipped",
                        "probability": 1.0,
                        "signals": meta_signals
                    }
                }
            }
        
        # No early exit - proceed to frame analysis
        frames, quality_rejected = await extract_video_frames(file_path)
        if not frames:
            return {
                "summary": "Analysis Failed",
                "confidence_score": 0.0,
                "layers": {
                    "layer1_metadata": l1_data,
                    "layer2_forensics": {
                        "status": "error",
                        "probability": 0.0,
                        "signals": ["Could not extract frames from video"]
                    }
                }
            }
        
        batch_result = await run_batch_forensics(frames)
        
        if batch_result.get("error"):
            return {
                "summary": "Analysis Failed",
                "confidence_score": 0.0,
                "layers": {
                    "layer1_metadata": l1_data,
                    "layer2_forensics": {
                        "status": "error",
                        "probability": 0.0,
                        "signals": [f"Frame analysis failed: {batch_result['error']}"]
                    }
                }
            }
        
        results = batch_result.get("results", [])
        gpu_time_ms = batch_result.get("gpu_time_ms", 0.0)
        
        if not results:
            return {
                "summary": "Analysis Failed",
                "confidence_score": 0.0,
                "layers": {
                    "layer1_metadata": l1_data,
                    "layer2_forensics": {
                        "status": "error",
                        "probability": 0.0,
                        "signals": ["No frame results returned"]
                    }
                }
            }
        
        frame_probs = [r.get("ai_score", 0.0) for r in results if isinstance(r, dict) and "ai_score" in r]
        
        median_prob = float(np.median(frame_probs))
        max_prob = float(np.max(frame_probs))
        mean_prob = float(np.mean(frame_probs))
        
        # Aggregation Logic Refinement:
        # If the median is low, a single high frame is likely a false positive (common in screen recordings)
        if median_prob < 0.20:
            if max_prob > 0.98: 
                # Very strong signal on one frame. If we have few frames (<=3), this makes median < 0.2 possible (0.0, 0.0, 0.99 -> meds 0.0).
                # One frame shouldn't condemn a video unless mean is significant.
                final_prob = max_prob if mean_prob > 0.35 else 0.45 # Cap at suspicious, not detected
            elif max_prob > 0.85:
                # High outlier but not extreme -> suppressed
                final_prob = median_prob
            else:
                final_prob = median_prob
        else:
            # If median is elevated, trust the max more
            final_prob = max_prob if max_prob > 0.85 else median_prob
        
        is_ai_likely = final_prob > 0.5
        if final_prob > 0.85: 
            summary = "Likely AI Video"
        elif final_prob > 0.5: 
            summary = "Suspicious Video"
        else: 
            summary = "Likely Original Video"
        
        raw_conf = final_prob if is_ai_likely else 1.0 - final_prob
        final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)
        if final_conf > 0.99: final_conf = 0.99

        analysis_signals = [
            f"Tri-frame batch analysis ({len(frames)} frames)",
            f"Aggregation: median={median_prob:.2f}, max={max_prob:.2f}"
        ]
        if meta_signals:
            analysis_signals.extend(meta_signals[:2])
        
        return {
            "summary": summary,
            "confidence_score": round(final_conf, 2),
            "layers": {
                "layer1_metadata": l1_data,
                "layer2_forensics": {
                    "status": "detected" if final_prob > 0.5 else "not_detected",
                    "probability": round(final_prob, 2),
                    "signals": analysis_signals
                }
            },
            "gpu_time_ms": gpu_time_ms
        }
    else:
        return await detect_ai_media_image_logic(file_path, l1_data, trusted_metadata=trusted_metadata, original_filename=original_filename)

async def detect_ai_media_image_logic(
    file_path: Optional[str], 
    l1_data: dict = None, 
    frame: Image.Image = None,
    trusted_metadata: dict = None,
    original_filename: str = None
) -> dict:
    """
    Core consensus logic for images and video frames.
    """
    layer_start = time.perf_counter()
    
    if l1_data is None:
        l1_data = {"status": "not_found", "provider": None, "description": "N/A"}

    # --- EXIF Extraction ---
    file_size = 0
    if frame:
        img_for_res = frame
        exif = {} 
        source_for_hash = frame
        width, height = img_for_res.size
    else:
        exif = get_exif_data(file_path)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
            source_for_hash = file_path
            file_size = os.path.getsize(file_path)
        except:
            return {
                "summary": "Analysis Failed",
                "confidence_score": 0.0,
                "layers": {
                    "layer1_metadata": l1_data,
                    "layer2_forensics": {"status": "error", "probability": 0.0, "signals": ["Invalid image file"]}
                }
            }
    
    # --- Merge Trusted Metadata (Sidecar) ---
    if trusted_metadata:
        logger.info(f"[SIDECAR] Using trusted metadata from device")
        if "Make" in trusted_metadata: exif["Make"] = trusted_metadata["Make"]
        if "Model" in trusted_metadata: exif["Model"] = trusted_metadata["Model"]
        if "Software" in trusted_metadata: exif["Software"] = trusted_metadata["Software"]
        if "DateTime" in trusted_metadata: exif["DateTimeOriginal"] = trusted_metadata["DateTime"]
        if "width" in trusted_metadata: width = trusted_metadata["width"]
        if "height" in trusted_metadata: height = trusted_metadata["height"]
        if "fileSize" in trusted_metadata: file_size = trusted_metadata["fileSize"]
        # In-app capture marker (used for scoring/auditing)
        if trusted_metadata.get("captured_in_app") is True:
            exif["CapturedInApp"] = True
            exif["CaptureSessionId"] = trusted_metadata.get("capture_session_id")
            exif["CaptureTimestampMs"] = trusted_metadata.get("capture_timestamp_ms") or trusted_metadata.get("capture_timestamp")
            
    # --- Metadata Scoring ---
    human_score, human_signals = get_forensic_metadata_score(exif)
    effective_filename = original_filename or os.path.basename(file_path if file_path else "")
    ai_score, ai_signals = get_ai_suspicion_score(exif, width, height, file_size, filename=effective_filename)
    
    logger.debug(f"[DEBUG] meta scores - human: {human_score} ({type(human_score)}), ai: {ai_score} ({type(ai_score)})")

    meta_summary = {
        "human_score": float(human_score),
        "ai_score": float(ai_score),
        "signals": [str(s) for s in (human_signals or [])][:10] + [str(s) for s in (ai_signals or [])][:10],
        "extracted": {
            "make": exif.get("Make"),
            "model": exif.get("Model"),
            "software": exif.get("Software"),
            "has_icc": bool(exif.get("HasICCProfile")),
            "has_makernote": bool(exif.get("HasMakerNote") or exif.get("MakerNote")),
            "has_embedded_thumbnail": bool(exif.get("HasEmbeddedThumbnail")),
            "jpeg_qtable_generic": bool(exif.get("JPEGQuantIsGeneric")),
            "dct_midhigh_ratio": exif.get("DCTMidHighRatio"),
            "width": width,
            "height": height,
            "file_size": file_size,
            "filename": effective_filename,
        },
        "bypass_reason": None,
    }

    # 1. VERIFIED ORIGINAL (Early Exit - Save Money)
    if human_score >= ScoringConfig.THRESHOLDS["HUMAN_EXIT_HIGH"]:
        return _log_decision({
            "summary": "Verified Original (Metadata)",
            "confidence_score": 0.99,
            "layers": {
                "layer1_metadata": {
                    "status": "verified_original", 
                    "provider": exif.get("Make", "Unknown"),
                    "description": "Device metadata verified - real camera footprint."
                },
                "layer2_forensics": {
                    "status": "skipped", "probability": 0.0, "signals": human_signals
                 }
            },
            "gpu_time_ms": 0,
            "gpu_bypassed": True,
            "metadata": {**meta_summary, "bypass_reason": "metadata_verified_original"}
        }, "Metadata (Verified)")

    # 2. LIKELY ORIGINAL (Early Exit - Save Money)
    # Be aggressive if ai_score is very low
    if (
        (human_score >= ScoringConfig.THRESHOLDS["HUMAN_EXIT_LOW"] and ai_score < ScoringConfig.THRESHOLDS.get("HUMAN_LOW_AI_MAX", 0.10))
        or (ai_score == 0 and human_score >= ScoringConfig.THRESHOLDS.get("HUMAN_LOW_NO_AI_MIN", 0.25))
    ):
        return _log_decision({
            "summary": "Likely Original (Metadata)",
            "confidence_score": 0.9,
            "layers": {
                "layer1_metadata": {
                    "status": "likely_original", 
                    "provider": exif.get("Make", "Unknown"),
                    "description": "Heuristic analysis suggests original source."
                },
                "layer2_forensics": {
                    "status": "skipped", "probability": 0.1, "signals": human_signals
                 }
            },
            "gpu_time_ms": 0,
            "gpu_bypassed": True,
            "metadata": {**meta_summary, "bypass_reason": "metadata_likely_original"}
        }, "Metadata (Likely Original)")

    # 3. LIKELY AI (Early Exit - Save Money)
    if ai_score >= ScoringConfig.THRESHOLDS["AI_EXIT_META"]:
        return _log_decision({
            "summary": "Likely AI (Metadata Evidence)",
            "confidence_score": 0.95,
            "layers": {
                "layer1_metadata": {
                    "status": "verified_ai", 
                    "provider": exif.get("Software", "AI Generator"),
                    "description": "Image metadata contains known AI signatures."
                },
                "layer2_forensics": {
                    "status": "skipped", "probability": 0.95, "signals": ai_signals
                 }
            },
            "gpu_time_ms": 0,
            "gpu_bypassed": True,
            "metadata": {**meta_summary, "bypass_reason": "metadata_likely_ai"}
        }, "Metadata (Likely AI)")

    # 4. SUSPICIOUS AI (Continue to GPU)
    # We no longer exit early here to let the GPU 'veto' suspicious metadata
    # as requested: "always assume that the gpu is gives u the correct answer"
    is_suspicious_meta = ai_score >= ScoringConfig.THRESHOLDS["AI_SUSPICIOUS"] and human_score == 0.0
    
    # 5. GPU Verification
    img_for_gpu = frame if frame else source_for_hash
    if not frame and os.path.exists(file_path):
        f_size = os.path.getsize(file_path)
        if f_size > 50 * 1024 * 1024:
            return {
                "summary": "File too large to scan", 
                "confidence_score": 0.0, 
                "layers": {
                    "layer1_metadata": {"status": "not_found", "provider": None, "description": "Size limit"},
                    "layer2_forensics": {"status": "skipped", "probability": 0.0, "signals": ["Skipped"]}
                }
            }

    # --- Consensus Preparation ---
    final_signals = ["Multi-layered consensus applied (Deep Learning + FFT)"]
    forensic_probability = 0.0
    actual_gpu_time_ms = 0.0

    # --- Forensic Cache Lookup ---
    file_hash = get_image_hash(file_path) if file_path else None
    if file_hash and not _benchmark_mode_enabled():
        cached_result = forensic_cache.get(file_hash)
        if cached_result:
            logger.info(f"[CACHE] Hit for {original_filename if original_filename else file_path}")
            return cached_result

    # --- GPU Scan (Production) ---
    actual_gpu_time_ms = 0.0
    start_gpu = time.perf_counter()
    
    # Actual GPU call via RunPod client
    if file_path:
        gpu_source = file_path
    elif frame is not None:
        if np is None:
            raise RuntimeError("numpy is required for frame-based GPU scan but is not installed")
        gpu_source = Image.fromarray(np.uint8(frame))
    else:
        gpu_source = ""

    gpu_result = await run_deep_forensics(gpu_source)
    
    forensic_probability = gpu_result.get("ai_score", 0.0)
    actual_gpu_time_ms = gpu_result.get("gpu_time_ms", 0.0)
    
    gpu_signals = [f"Forensic models scanned (Score: {forensic_probability:.2f})"]
    if gpu_result.get("error"):
        gpu_signals.append(f"GPU Error: {gpu_result['error']}")
    
    final_signals.extend(gpu_signals)

    # --- Metadata-Model Conflict Resolution ---
    original_prob = forensic_probability
    
    # HARD AI SIGNALS (Keywords) always push the score up even if GPU is low
    ai_signals_str = " ".join(ai_signals).lower()
    hard_ai_signal = any(k in ai_signals_str for k in ["keyword", "software", "manufacturer", "filename", "marker", "credit"])
    
    if (
        (not _benchmark_mode_enabled())
        and ai_score >= ScoringConfig.THRESHOLDS["CONFLICT_AI_SCORE"]
        and forensic_probability < ScoringConfig.THRESHOLDS["CONFLICT_MODEL_LOW"]
    ):
        if hard_ai_signal:
            # Hard evidence overrides silent GPU
            forensic_probability = max(forensic_probability, ai_score, 0.95)
            final_signals.append(f"Hard AI Metadata evidence verified ({ai_score}) - overriding silent forensics")
        else:
            # Soft suspicion (meta-data absence) blends with GPU
            # Refinement (Round 2): (ai * 0.6) + (gpu * 0.4)
            blended_prob = (float(ai_score) * 0.60) + (float(forensic_probability) * 0.40)
            forensic_probability = blended_prob
            final_signals.append(f"Consensus blend: Suspicious metadata ({ai_score}) + Forensic consensus")
    
    l2_data = {
        "status": "detected" if forensic_probability > 0.85 else "suspicious" if forensic_probability > 0.5 else "not_detected",
        "probability": round(forensic_probability, 4),
        "signals": final_signals
    }
    
    is_ai_likely = forensic_probability > 0.5
    logger.debug(f"[DEBUG] Prob: {forensic_probability}, LIKELY_AI: {ScoringConfig.THRESHOLDS['LIKELY_AI']}, POSSIBLE: {ScoringConfig.THRESHOLDS['POSSIBLE_AI']}")
    
    if forensic_probability > ScoringConfig.THRESHOLDS["LIKELY_AI"]: summary = "Likely AI (High Confidence)"
    elif forensic_probability > ScoringConfig.THRESHOLDS["POSSIBLE_AI"]: summary = "Possible AI (Forensic Match)"
    elif forensic_probability > ScoringConfig.THRESHOLDS["SUSPICIOUS_AI"]: summary = "Suspicious (Inconsistent Pixels)"
    elif forensic_probability > ScoringConfig.THRESHOLDS["LIKELY_HUMAN_NOISE"]: summary = "Likely Original (Minor Noise)"
    else: summary = "Likely Original"

    logger.debug(f"[DEBUG] Summary chosen: {summary}")
    
    raw_conf = forensic_probability if is_ai_likely else (1.0 - forensic_probability)
    final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)
    if final_conf > 0.99: final_conf = 0.99

    final_result = {
        "summary": summary,
        "confidence_score": round(final_conf, 2),
        "layers": {
            "layer1_metadata": l1_data, 
            "layer2_forensics": l2_data
        },
        "gpu_time_ms": actual_gpu_time_ms,
        "gpu_bypassed": actual_gpu_time_ms == 0,
        "metadata": meta_summary
    }

    if file_hash and not _benchmark_mode_enabled():
        forensic_cache.put(file_hash, final_result)

    return _log_decision(final_result, "Final Consensus")
