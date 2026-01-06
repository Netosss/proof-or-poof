import logging
import time
import cv2
import numpy as np
import asyncio
import hashlib
import os
import io
import tempfile
import subprocess
import json
from collections import OrderedDict
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Optional, List, Union
from app.c2pa_reader import get_c2pa_manifest
from app.runpod_client import run_deep_forensics
from app.security import security_manager

logger = logging.getLogger(__name__)

# LRU Cache implementation for forensic results
class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

forensic_cache = LRUCache(capacity=1000)

def get_image_hash(source: Union[str, Image.Image]) -> str:
    """Generate a secure SHA-256 hash of the image source (optimized with grayscale)."""
    if isinstance(source, str):
        with open(source, 'rb') as f:
            # Hash first 2MB for speed, but use secure method
            return security_manager.get_safe_hash(f.read(2048 * 1024))
    else:
        # For PIL Images: small grayscale thumbnail for fast, unique hash
        thumb = source.copy()
        thumb.thumbnail((64, 64))  # Smaller for speed
        thumb = thumb.convert("L")  # Grayscale reduces data while preserving uniqueness
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=50)  # Lower quality = faster
        return security_manager.get_safe_hash(buf.getvalue())

def get_exif_data(file_path: str) -> dict:
    """Extract EXIF metadata from the image. Explicitly closed via 'with'."""
    try:
        with Image.open(file_path) as img:
            exif = img._getexif() or {}
            exif_data = {}
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
            return exif_data
    except Exception:
        return {}

def is_frame_quality_ok(frame: np.ndarray, min_brightness: float = 20, min_sharpness: float = 50) -> tuple:
    """
    Check if frame is not too dark or blurry for reliable AI detection.
    Returns (is_ok, brightness, sharpness) for potential weighted aggregation.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Quick brightness check first (fast, avoids expensive Laplacian for dark frames)
        brightness = np.mean(gray)
        if brightness < min_brightness:
            return False, brightness, 0.0
        
        # Check sharpness via Laplacian variance (avoid blurry frames)
        # Use smaller region for speed (center crop)
        h, w = gray.shape
        center_crop = gray[h//4:3*h//4, w//4:3*w//4]
        laplacian_var = cv2.Laplacian(center_crop, cv2.CV_64F).var()
        if laplacian_var < min_sharpness:
            return False, brightness, laplacian_var
        
        return True, brightness, laplacian_var
    except:
        return True, 128.0, 100.0  # Safe defaults if check fails

def extract_video_frames(video_path: str, num_frames: int = 8) -> list:
    """Extract N evenly-spaced quality frames from a video file."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
        
        # Evenly spaced sampling points (skip first/last 5% to avoid black frames)
        start_frame = int(total_frames * 0.05)
        end_frame = int(total_frames * 0.95)
        sample_points = np.linspace(start_frame, end_frame, num=min(num_frames, total_frames), dtype=int)
        
        quality_rejected = 0
        frame_qualities = []  # Store (brightness, sharpness) for potential weighted aggregation
        for pos in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                # Quality filter: skip dark/blurry frames
                is_ok, brightness, sharpness = is_frame_quality_ok(frame)
                if not is_ok:
                    quality_rejected += 1
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                frame_qualities.append((brightness, sharpness))
        
        cap.release()
        
        if quality_rejected > 0:
            logger.info(f"[VIDEO] Skipped {quality_rejected} low-quality frames (dark/blurry)")
        
        # Ensure we have at least 2 frames
        if len(frames) < 2 and total_frames >= 2:
            # Fallback: just grab first and last readable frames
            cap = cv2.VideoCapture(video_path)
            for pos in [start_frame, end_frame]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            cap.release()
            
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
    return frames

def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timeout")
    except FileNotFoundError:
        logger.warning("ffprobe not installed")
    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}")
    return {}

def get_video_metadata_score(metadata: dict) -> tuple:
    """
    Analyze video metadata for human vs AI signals.
    Returns (human_score, ai_score, signals)
    """
    human_score = 0.0
    ai_score = 0.0
    signals = []
    
    if not metadata:
        return 0.0, 0.0, ["No metadata available"]
    
    format_info = metadata.get("format", {})
    tags = format_info.get("tags", {})
    streams = metadata.get("streams", [])
    
    # Normalize tag keys to lowercase for consistent matching
    tags_lower = {k.lower(): v for k, v in tags.items()}
    
    # === HUMAN SIGNALS (Camera/Phone recordings) ===
    
    # 1. Apple device markers
    apple_markers = [
        "com.apple.quicktime.make",
        "com.apple.quicktime.model", 
        "com.apple.quicktime.software",
        "com.apple.quicktime.creationdate"
    ]
    for marker in apple_markers:
        if marker in tags_lower or marker.replace(".", "_") in tags_lower:
            human_score += 0.25
            signals.append(f"Apple device marker: {marker}")
    
    # 2. Android/Samsung markers
    android_markers = ["com.android.version", "com.samsung", "manufacturer"]
    for marker in android_markers:
        for key in tags_lower:
            if marker in key:
                human_score += 0.25
                signals.append(f"Android device marker: {key}")
                break
    
    # 3. Known camera brands in encoder/handler
    camera_brands = ["iphone", "samsung", "google", "pixel", "gopro", "dji", 
                     "sony", "canon", "nikon", "panasonic", "fujifilm"]
    encoder = tags_lower.get("encoder", "").lower()
    handler = tags_lower.get("handler_name", "").lower()
    make = tags_lower.get("make", "").lower()
    model = tags_lower.get("model", "").lower()
    
    for brand in camera_brands:
        if brand in encoder or brand in handler or brand in make or brand in model:
            human_score += 0.30
            signals.append(f"Camera brand detected: {brand}")
            break
    
    # 4. GPS/Location data (strong human signal)
    gps_markers = ["location", "gps", "coordinates", "com.apple.quicktime.location"]
    for marker in gps_markers:
        for key in tags_lower:
            if marker in key:
                human_score += 0.35
                signals.append("GPS/Location data present")
                break
    
    # 5. Creation time with timezone (phones add this)
    creation_time = tags_lower.get("creation_time", "")
    if creation_time and ("+" in creation_time or "Z" in creation_time):
        human_score += 0.15
        signals.append("Creation time with timezone")
    
    # === AI SIGNALS (Generated video markers) ===
    
    # 1. Known AI video generators
    ai_encoders = ["runway", "pika", "sora", "kling", "luma", "midjourney", 
                   "stable video", "deforum", "animatediff", "svd", "cogvideo"]
    for ai_enc in ai_encoders:
        if ai_enc in encoder.lower():
            ai_score += 0.90
            signals.append(f"AI generator detected: {ai_enc}")
            break
    
    # 2. Generic FFmpeg encoding (suspicious but not conclusive)
    if "lavf" in encoder.lower() or "ffmpeg" in encoder.lower():
        # Not strong signal - humans also use FFmpeg
        ai_score += 0.10
        signals.append("FFmpeg encoder (neutral)")
    
    # 3. No metadata at all (slightly suspicious for modern videos)
    if not tags:
        ai_score += 0.15
        signals.append("No metadata tags")
    
    # 4. Unusual resolutions common in AI (1024x1024, 768x768, etc.)
    for stream in streams:
        if stream.get("codec_type") == "video":
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            # Square resolutions are common in AI
            if width == height and width in [512, 768, 1024]:
                ai_score += 0.20
                signals.append(f"AI-typical square resolution: {width}x{height}")
            # Very short duration is common in AI
            duration = float(format_info.get("duration", 0))
            if 0 < duration <= 4.0:
                ai_score += 0.10
                signals.append(f"Short duration: {duration:.1f}s")
            break
    
    # Cap scores at 1.0
    human_score = min(1.0, human_score)
    ai_score = min(1.0, ai_score)
    
    return human_score, ai_score, signals

def get_forensic_metadata_score(exif: dict) -> tuple:
    """
    Advanced forensic check for human sensor physics using weighted tiers.
    Includes type and range validation to prevent metadata spoofing.
    """
    score = 0.0
    signals = []

    def to_float(val):
        try: return float(val)
        except: return None

    # --- Tier 1: Device Provenance (Max 0.55) ---
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()
    
    if any(m in make for m in ["apple", "google", "samsung", "sony", "canon", "nikon"]):
        score += 0.30
        signals.append("Trusted device manufacturer provenance")
    
    if any(s in software for s in ["hdr+", "ios", "android", "deep fusion", "one ui", "version"]):
        score += 0.25
        signals.append("Validated vendor-specific camera pipeline")

    # --- Tier 2: Physical Camera Consistency (Max 0.30) ---
    # Signal 2.1: Exposure Time (0.10) - Valid range: 0 < exp < 30s
    exp = to_float(exif.get("ExposureTime"))
    if exp is not None and 0 < exp < 30:
        score += 0.10
        signals.append("Physically valid exposure duration")
    
    # Signal 2.2: ISO Speed (0.10) - Valid range: 50 < ISO < 102400
    iso = to_float(exif.get("ISOSpeedRatings"))
    if iso is not None and 50 <= iso <= 102400:
        score += 0.10
        signals.append("Realistic sensor sensitivity (ISO)")
        
    # Signal 2.3: Aperture/F-Number (0.10) - Valid range: 0.95 < f < 32
    f_num = to_float(exif.get("FNumber"))
    if f_num is not None and 0.95 <= f_num <= 32:
        score += 0.10
        signals.append("Valid physical aperture geometry")

    # --- Tier 3: Temporal Authenticity (Max 0.05) ---
    if "DateTimeOriginal" in exif:
        score += 0.03
        signals.append("Temporal capture timestamp present")
        
    subsec = str(exif.get("SubSecTimeOriginal", exif.get("SubSecTimeDigitized", "")))
    if subsec and subsec.isdigit() and subsec not in ["000", "000000"]:
        score += 0.02
        signals.append("High-precision sensor timing")

    # --- Tier 4: JPEG Structure (Max 0.10) ---
    if "JPEGInterchangeFormat" in exif:
        score += 0.05
        signals.append("Firmware-level segment tables")
        
    if exif.get("Compression") in [6, 1]: 
        score += 0.05
        signals.append("Standard camera compression")

    return round(score, 2), signals

def get_ai_suspicion_score(exif: dict) -> tuple:
    """
    Weighted AI suspicion score based on blatant signatures and missing camera metadata.
    """
    score = 0.0
    signals = []
    
    # 1. Hard AI Evidence (Software/Make keywords)
    ai_keywords = ["stable", "diffusion", "midjourney", "dalle", "flux", "sora", "kling", "firefly", "generative", "artificial"]
    software = str(exif.get("Software", "")).lower()
    make = str(exif.get("Make", "")).lower()
    
    if any(k in software for k in ai_keywords):
        score += 0.40
        signals.append(f"AI software signature: {software}")
    elif any(k in make for k in ai_keywords):
        score += 0.40
        signals.append(f"AI manufacturer signature: {make}")

    # 2. Negative Signals (Missing Metadata statistically unlikely for real cameras)
    if not exif.get("Make") and not exif.get("Model"):
        score += 0.10
        signals.append("Missing camera hardware provenance")

    if "DateTimeOriginal" not in exif:
        score += 0.05
        signals.append("Missing capture timestamp")

    if not exif.get("SubSecTimeOriginal") and not exif.get("SubSecTimeDigitized"):
        score += 0.03
        signals.append("Missing high-precision sensor timing")

    if "JPEGInterchangeFormat" not in exif:
        score += 0.05
        signals.append("Non-standard JPEG segment structure")

    return round(min(score, 1.0), 2), signals

def boost_score(score: float, is_ai_likely: bool = True) -> float:
    """
    Boost confidence only for AI-likely results.
    Human-likely results keep their raw confidence to avoid misleading scores.
    """
    if is_ai_likely:
        return max(0.85, score)
    return score  # No boost for human results

async def detect_ai_media(file_path: str) -> dict:
    """Final Optimized Consensus Engine."""
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
            "status": "verified_ai" if is_generative_ai else "verified_human",
            "provider": generator,
            "description": f"Verified AI signature found ({generator})." if is_generative_ai else "Verified human-captured content."
        }

        return {
            "summary": "Verified AI" if is_generative_ai else "Verified Human",
            "confidence_score": 1.0,
            "layers": {
                "layer1_metadata": l1_data,
                "layer2_forensics": {
                    "status": "skipped", 
                    "probability": 1.0 if is_generative_ai else 0.0, 
                    "signals": ["Source verified via cryptographic signature."]
                }
            }
        }

    is_video = file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
    
    if is_video:
        safe_path = security_manager.sanitize_log_message(file_path)
        logger.info(f"Detecting AI in video: {safe_path}")
        
        # --- VIDEO CACHE CHECK (same as images) ---
        video_hash = get_image_hash(file_path)  # Uses first 2MB of file
        cached_video_result = forensic_cache.get(f"video_{video_hash}")
        if cached_video_result is not None:
            logger.info(f"[VIDEO] Cache hit! Returning cached result.")
            return cached_video_result
        
        # --- VIDEO METADATA EARLY EXIT ---
        t_video_meta = time.perf_counter()
        video_metadata = get_video_metadata(file_path)
        human_score, ai_meta_score, meta_signals = get_video_metadata_score(video_metadata)
        video_meta_time_ms = (time.perf_counter() - t_video_meta) * 1000
        logger.info(f"[TIMING] Video metadata extraction: {video_meta_time_ms:.2f}ms")
        logger.info(f"[VIDEO META] Human score: {human_score:.2f}, AI score: {ai_meta_score:.2f}")
        logger.info(f"[VIDEO META] Signals: {meta_signals}")
        
        # Check for strong device markers (Android/Apple/Samsung)
        strong_device_markers = ["android", "apple", "samsung", "iphone", "pixel", "galaxy", "gopro", "dji"]
        has_strong_device = any(
            any(marker in signal.lower() for marker in strong_device_markers)
            for signal in meta_signals
        )
        
        # Early exit: Strong human metadata (camera/phone recording)
        # Lower threshold (0.40) if we have a strong device marker like Android/Apple
        early_exit_threshold = 0.40 if has_strong_device else 0.70
        
        if human_score >= early_exit_threshold and ai_meta_score < 0.20:
            logger.info(f"[VIDEO] Early exit: Verified Human via metadata (score={human_score:.2f}, device_marker={has_strong_device})")
            return {
                "summary": "Verified Human Video",
                "confidence_score": 1.0,
                "layers": {
                    "layer1_metadata": {
                        "status": "verified_human",
                        "provider": meta_signals[0] if meta_signals else "Camera/Phone",
                        "description": "Video recorded on real device with authentic metadata."
                    },
                    "layer2_forensics": {
                        "status": "skipped",
                        "probability": 0.0,
                        "signals": meta_signals
                    }
                }
            }
        
        # Early exit: Strong AI metadata (known AI generator)
        if ai_meta_score >= 0.80:
            logger.info(f"[VIDEO] Early exit: AI Generator detected via metadata (score={ai_meta_score:.2f})")
            return {
                "summary": "Verified AI Video",
                "confidence_score": 1.0,
                "layers": {
                    "layer1_metadata": {
                        "status": "verified_ai",
                        "provider": meta_signals[0] if meta_signals else "AI Generator",
                        "description": "Video generated by AI tool detected in metadata."
                    },
                    "layer2_forensics": {
                        "status": "skipped",
                        "probability": 1.0,
                        "signals": meta_signals
                    }
                }
            }
        
        # No early exit - proceed to frame analysis
        logger.info(f"[VIDEO] No early exit, proceeding to frame analysis...")
        frames = extract_video_frames(file_path)
        if not frames:
            return {"error": "Could not extract frames from video."}
            
        tasks = [detect_ai_media_image_logic(None, frame=f) for f in frames]
        frame_results = await asyncio.gather(*tasks)
        
        # Check for strong human metadata in any frame (early exit)
        for res in frame_results:
            if res.get("summary") == "Verified Human (Forensic Metadata)":
                logger.info(f"[VIDEO] Early exit: Frame scan found trusted human metadata.")
                return res
        
        # Extract probabilities for robust aggregation
        frame_probs = [r['layers']['layer2_forensics']['probability'] for r in frame_results]
        num_frames_analyzed = len(frame_probs)
        
        # Use MEDIAN instead of mean (more robust to outliers)
        median_prob = float(np.median(frame_probs))
        mean_prob = float(np.mean(frame_probs))
        std_prob = float(np.std(frame_probs))
        
        logger.info(f"[VIDEO] Frame analysis: {num_frames_analyzed} frames | "
                   f"Median: {median_prob:.3f}, Mean: {mean_prob:.3f}, Std: {std_prob:.3f}")
        
        # Temporal consistency check
        # If high variance (some frames AI, some human), be conservative
        if std_prob > 0.35:
            logger.info(f"[VIDEO] High variance detected ({std_prob:.3f}) - inconsistent frames")
            # If most frames are human-like, trust the majority
            human_frames = sum(1 for p in frame_probs if p < 0.5)
            if human_frames > len(frame_probs) / 2:
                # Majority human - use trimmed mean of low scores
                low_probs = [p for p in frame_probs if p < 0.5]
                final_prob = float(np.mean(low_probs)) if low_probs else median_prob
                logger.info(f"[VIDEO] Majority human ({human_frames}/{num_frames_analyzed}), using trimmed mean: {final_prob:.3f}")
            else:
                # Mixed or majority AI - use median (conservative)
                final_prob = median_prob
        else:
            # Consistent frames - use median
            final_prob = median_prob
        
        # Determine summary based on final probability
        is_ai_likely = final_prob > 0.5
        if final_prob > 0.85: 
            summary = "Likely AI Video"
        elif final_prob > 0.5: 
            summary = "Suspicious Video"
        else: 
            summary = "Likely Human Video"
        
        # Apply boost to confidence score (only for AI-likely results)
        raw_conf = final_prob if is_ai_likely else 1.0 - final_prob
        final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)
        
        # Cap at 0.99 for probabilistic results
        if final_conf > 0.99:
            final_conf = 0.99

        # Build detailed signals
        analysis_signals = [
            f"Analyzed {num_frames_analyzed} frames (median aggregation)",
            f"Frame scores: median={median_prob:.2f}, std={std_prob:.2f}"
        ]
        if std_prob > 0.35:
            analysis_signals.append(f"High variance detected - used conservative estimate")
        if meta_signals:
            analysis_signals.extend(meta_signals[:2])  # Add top 2 metadata signals
        
        video_result = {
            "summary": summary,
            "confidence_score": round(final_conf, 2),
            "layers": {
                "layer1_metadata": l1_data,
                "layer2_forensics": {
                    "status": "detected" if final_prob > 0.5 else "not_detected",
                    "probability": round(final_prob, 2),
                    "signals": analysis_signals
                }
            }
        }
        
        # Cache the video result for future requests
        forensic_cache.put(f"video_{video_hash}", video_result)
        logger.info(f"[VIDEO] Cached result for future requests")
        
        return video_result
    else:
        return await detect_ai_media_image_logic(file_path, l1_data)

async def detect_ai_media_image_logic(file_path: Optional[str], l1_data: dict = None, frame: Image.Image = None) -> dict:
    """Core consensus logic for images and video frames."""
    layer_start = time.perf_counter()
    
    if l1_data is None:
        l1_data = {"status": "not_found", "provider": None, "description": "N/A"}

    # --- EXIF Extraction ---
    t_exif = time.perf_counter()
    if frame:
        img_for_res = frame
        exif = {} 
        source_for_hash = frame
        source_path = None
        width, height = img_for_res.size
    else:
        exif = get_exif_data(file_path)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
            source_for_hash = file_path
            source_path = file_path
        except:
            return {"error": "Invalid image file"}
    exif_time_ms = (time.perf_counter() - t_exif) * 1000
    logger.info(f"[TIMING] EXIF extraction: {exif_time_ms:.2f}ms")
    
    # --- Metadata Scoring ---
    t_scoring = time.perf_counter()
    human_score, human_signals = get_forensic_metadata_score(exif)
    ai_score, ai_signals = get_ai_suspicion_score(exif)
    scoring_time_ms = (time.perf_counter() - t_scoring) * 1000
    logger.info(f"[TIMING] Metadata scoring: {scoring_time_ms:.2f}ms (human={human_score:.2f}, ai={ai_score:.2f})")
    
    # 1. VERIFIED HUMAN (Early Exit)
    if human_score >= 0.80:
        return {
            "summary": "Verified Human (Forensic Metadata)",
            "confidence_score": 1.0,
            "layers": {
                "layer1_metadata": {
                    "status": "verified_human", 
                    "provider": exif.get("Make", "Unknown"),
                    "description": "Hardware sensor physics confirmed."
                },
                "layer2_forensics": {
                    "status": "skipped", 
                    "probability": 0.0,
                    "signals": human_signals
                }
            }
        }

    # 2. LIKELY HUMAN (Early Exit)
    if human_score >= 0.60 and ai_score < 0.20:
        return {
            "summary": "Likely Human (Strong Heuristics)",
            "confidence_score": 0.9,
            "layers": {
                "layer1_metadata": {
                    "status": "likely_human", 
                    "provider": exif.get("Make", "Unknown"),
                    "description": "Heuristic analysis suggests human origin."
                },
                "layer2_forensics": {
                    "status": "skipped", 
                    "probability": 0.1,
                    "signals": human_signals
                }
            }
        }

    # 3. LIKELY AI (Early Exit)
    if ai_score >= 0.50:
        return {
            "summary": "Likely AI (Metadata Evidence)",
            "confidence_score": 0.95,
            "layers": {
                "layer1_metadata": {
                    "status": "verified_ai", 
                    "provider": exif.get("Software", "AI Generator"),
                    "description": "Image metadata contains known AI signatures."
                },
                "layer2_forensics": {
                    "status": "skipped", 
                    "probability": 0.95,
                    "signals": ai_signals
                }
            }
        }

    # 4. AMBIGUOUS -> GPU Scan
    # Wallet Guard: Prevent multi-minute GPU jobs for huge files
    if not frame and os.path.exists(file_path):
        f_size = os.path.getsize(file_path)
        if f_size > 50 * 1024 * 1024:
            return {
                "summary": "File too large to scan", 
                "confidence_score": 0.0, 
                "layers": {
                    "layer1_metadata": {
                        "status": "not_found", 
                        "provider": None, 
                        "description": "File exceeds size limit."
                    },
                    "layer2_forensics": {
                        "status": "skipped", 
                        "probability": 0.0, 
                        "signals": ["Skipped due to file size"]
                    }
                }
            }

    img_hash = get_image_hash(source_for_hash)
    cached_result = forensic_cache.get(img_hash)
    
    # --- GPU Scan ---
    t_gpu = time.perf_counter()
    if cached_result is not None:
        forensic_probability = cached_result.get("ai_score", cached_result) if isinstance(cached_result, dict) else cached_result
        actual_gpu_time_ms = 0.0  # Cached, no GPU used
        roundtrip_ms = (time.perf_counter() - t_gpu) * 1000
        logger.info(f"[TIMING] Layer 2 - GPU scan (CACHED): {roundtrip_ms:.2f}ms")
    else:
        forensic_result = await run_deep_forensics(source_for_hash, width, height)
        forensic_probability = forensic_result.get("ai_score", 0.0)
        actual_gpu_time_ms = forensic_result.get("gpu_time_ms", 0.0)
        forensic_cache.put(img_hash, forensic_result)
        roundtrip_ms = (time.perf_counter() - t_gpu) * 1000
        logger.info(f"[TIMING] Layer 2 - GPU scan (RunPod): {roundtrip_ms:.2f}ms | Actual GPU: {actual_gpu_time_ms:.2f}ms")
    
    total_layer_time_ms = (time.perf_counter() - layer_start) * 1000
    logger.info(f"[TIMING] Layer 2 - Total: {total_layer_time_ms:.2f}ms | Result: {forensic_probability:.4f}")
    
    l2_data = {
        "status": "detected" if forensic_probability > 0.85 else "suspicious" if forensic_probability > 0.5 else "not_detected",
        "probability": round(forensic_probability, 4),  # RAW probability for video aggregation
        "signals": ["Multi-layered consensus applied (Deep Learning + FFT)"]
    }
    
    is_ai_likely = forensic_probability > 0.5
    if forensic_probability > 0.92: summary = "Likely AI (High Confidence)"
    elif forensic_probability > 0.75: summary = "Possible AI (Forensic Match)"
    elif forensic_probability > 0.5: summary = "Suspicious (Inconsistent Pixels)"
    elif forensic_probability > 0.2: summary = "Likely Human (Minor Noise)"
    else: summary = "Likely Human"
    
    # Boost the overall confidence score (only for AI-likely results)
    raw_conf = forensic_probability if is_ai_likely else (1.0 - forensic_probability)
    final_conf = boost_score(raw_conf, is_ai_likely=is_ai_likely)
    
    # Cap probabilistic scores at 0.99 to avoid "fake" 100% look, unless it's a hard metadata match
    if final_conf > 0.99:
        final_conf = 0.99

    return {
        "summary": summary,
        "confidence_score": round(final_conf, 2),
        "layers": {
            "layer1_metadata": l1_data, 
            "layer2_forensics": l2_data
        },
        "gpu_time_ms": actual_gpu_time_ms  # Actual GPU time for cost calculation
    }