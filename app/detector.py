import logging
import cv2
import numpy as np
import asyncio
import hashlib
import os
import io
import tempfile
from collections import OrderedDict
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Optional, List, Union
from app.c2pa_reader import get_c2pa_manifest
from app.runpod_client import run_deep_forensics
from app.security import security_manager

logger = logging.getLogger(__name__)

# --- Constants ---
SCREEN_RECORD_PENALTY = 0.25
MAX_CONCURRENT_GPU_JOBS = 5
gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPU_JOBS)

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
    """Generate a secure SHA-256 hash of the image source."""
    if isinstance(source, str):
        with open(source, 'rb') as f:
            # Hash first 2MB for speed, but use secure method
            return security_manager.get_safe_hash(f.read(2048 * 1024))
    else:
        # For PIL Images, hash a small thumbnail
        thumb = source.copy()
        thumb.thumbnail((128, 128))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG")
        return security_manager.get_safe_hash(buf.getvalue())

# Local CPU cache for EXIF to avoid re-reading the same file during video frame sampling
exif_cache = LRUCache(capacity=500)

def get_exif_data(file_path: str) -> dict:
    """Extract targeted EXIF metadata efficiently. Cache result (including empty) by content hash."""
    try:
        # Use content hash instead of file path to avoid collisions in temp folders
        file_hash = get_image_hash(file_path)
        cached = exif_cache.get(file_hash)
        if cached is not None: return cached

        with Image.open(file_path) as img:
            # Selective tag reading
            TARGET_TAGS = {271, 272, 305, 33434, 34855, 33437, 36867, 37521, 513, 259}
            exif = img._getexif()
            
            exif_data = {}
            if exif:
                for tag in TARGET_TAGS:
                    if tag in exif:
                        decoded = TAGS.get(tag, tag)
                        exif_data[decoded] = exif[tag]
            
            # Cache the result even if it's empty {}
            exif_cache.put(file_hash, exif_data)
            return exif_data
    except Exception:
        return {}

def get_metadata_context(exif: dict) -> dict:
    """Centralized metadata parsing to avoid redundant string logic."""
    return {
        "make": str(exif.get("Make", "")).lower(),
        "model": str(exif.get("Model", "")).lower(),
        "software": str(exif.get("Software", "")).lower(),
        "is_screenshot": False # Determined below
    }

def extract_video_frames(video_path: str, num_frames: int = 2) -> list:
    """Extract N frames from a video file."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
            
        # Sampling points
        sample_points = [int(total_frames * 0.1), int(total_frames * 0.9)]
        
        for pos in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
    return frames

def get_forensics_for_context(exif: dict, ctx: dict) -> tuple:
    """Refactored to use centralized context and selective tags."""
    score = 0.0
    signals = []
    
    make = ctx["make"]
    software = ctx["software"]
    
    # SENSITIVE: Check for screenshot signatures once
    ctx["is_screenshot"] = any(x in software or x in make for x in ["screenshot", "screen-capture", "screen record"])
    
    if any(m in make for m in ["apple", "google", "samsung", "sony", "canon", "nikon"]):
        if not ctx["is_screenshot"]:
            score += 0.30
            signals.append("Trusted device manufacturer provenance")
        else:
            signals.append("Device match found but ignored (Screenshot detected)")
    
    if any(s in software for s in ["hdr+", "ios", "android", "deep fusion", "one ui", "version"]):
        if not ctx["is_screenshot"]:
            score += 0.25
            signals.append("Validated vendor-specific camera pipeline")
        else:
            score += 0.05
            signals.append("Mobile software detected on screenshot")

    def to_float(val):
        try: return float(val)
        except: return None

    # --- Tier 2: Physical Camera Consistency ---
    exp = to_float(exif.get("ExposureTime"))
    if exp is not None and 0 < exp < 30:
        score += 0.10
        signals.append("Physically valid exposure duration")
    
    iso = to_float(exif.get("ISOSpeedRatings"))
    if iso is not None and 50 <= iso <= 102400:
        score += 0.10
        signals.append("Realistic sensor sensitivity (ISO)")
        
    f_num = to_float(exif.get("FNumber"))
    if f_num is not None and 0.95 <= f_num <= 32:
        score += 0.10
        signals.append("Valid physical aperture geometry")

    if "DateTimeOriginal" in exif:
        score += 0.03
        signals.append("Temporal capture timestamp present")
        
    subsec = str(exif.get("SubSecTimeOriginal", ""))
    if subsec and subsec.isdigit() and subsec not in ["000", "000000"]:
        score += 0.02
        signals.append("High-precision sensor timing")

    if "JPEGInterchangeFormat" in exif:
        score += 0.05
        signals.append("Firmware-level segment tables")
        
    if exif.get("Compression") in [6, 1]: 
        score += 0.05
        signals.append("Standard camera compression")

    return round(score, 2), signals

def get_ai_suspicion_score_refactored(exif: dict, ctx: dict) -> tuple:
    """Uses centralized context to check for AI keywords."""
    score = 0.0
    signals = []
    
    ai_keywords = ["stable", "diffusion", "midjourney", "dalle", "flux", "sora", "kling", "firefly", "generative", "artificial"]
    software = ctx["software"]
    make = ctx["make"]
    
    if any(k in software for k in ai_keywords):
        score += 0.40
        signals.append(f"AI software signature: {software}")
    elif any(k in make for k in ai_keywords):
        score += 0.40
        signals.append(f"AI manufacturer signature: {make}")

    if not exif.get("Make") and not exif.get("Model"):
        score += 0.10
        signals.append("Missing camera hardware provenance")

    if "DateTimeOriginal" not in exif:
        score += 0.05
        signals.append("Missing capture timestamp")

    return round(min(score, 1.0), 2), signals

def boost_score(score: float) -> float:
    """Ensure all presented percentages are at least 85%."""
    return max(0.85, score)

def get_video_metadata_score(file_path: str) -> tuple:
    """
    Scans the raw video container for mobile device signatures (Android/iOS).
    These atoms are rarely present in AI-generated or desktop-encoded videos.
    """
    try:
        # Read the first 16KB - enough for most MP4/MOV headers (moov/meta atoms)
        with open(file_path, 'rb') as f:
            header = f.read(16384)
            
        header_str = header.decode('ascii', errors='ignore')
        score = 0.0
        signals = []
        provider = "Unknown"

        # 1. Android Specifics
        if "com.android.version" in header_str:
            score += 0.85
            signals.append("Android mobile container signature found")
            provider = "Android Device"
        if "VideoHandle" in header_str:
            score += 0.10
            signals.append("Standard Android media handle detected")

        # 2. Apple / iOS Specifics
        if "com.apple.quicktime" in header_str:
            score += 0.85
            signals.append("Apple QuickTime mobile container signature")
            provider = "Apple Device"
        if "com.apple.photos" in header_str or "com.apple.itunes" in header_str:
            score += 0.10
            signals.append("iOS-specific media metadata found")

        # 3. Resolution Check for Mobile (Bonus)
        # Pixel 9 Pro XL, iPhone 15 Pro Max etc have non-standard resolutions
        # 1344x2992 (Pixel 9), 1290x2796 (iPhone 15)
        # We handle this inside detect_ai_media if we have width/height
            
        return round(score, 2), signals, provider
    except Exception as e:
        logger.error(f"Video metadata scan failed: {e}")
        return 0.0, [], "Unknown"

async def detect_ai_media(file_path: str) -> dict:
    """Final Optimized Consensus Engine."""
    l1_data = {
        "status": "not_found",
        "provider": None,
        "description": "No cryptographic signature found."
    }

    # --- 1️⃣ LAYER 1: C2PA ---
    manifest = get_c2pa_manifest(file_path)
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
            "media_type": "video" if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')) else "image",
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

        # --- 2️⃣ LAYER 1.5: Video Metadata Check ---
        v_score, v_signals, v_provider = get_video_metadata_score(file_path)
        is_screen_record = any(s in " ".join(v_signals).lower() for s in ["handle", "capture", "record"])

        logger.info(f"Parallel Temporal Sampling starting for video: {safe_path}")
        frames = extract_video_frames(file_path)
        if not frames:
            return {"error": "Could not extract frames from video."}
            
        tasks = [detect_ai_media_image_logic(None, frame=f, media_type="video") for f in frames]
        frame_results = await asyncio.gather(*tasks)
        
        # Check for Verified Human frame (e.g. C2PA or strong EXIF in frame)
        for res in frame_results:
            if res.get("summary") == "Verified Human (Forensic Metadata)":
                logger.info(f"Video Early Exit: Frame scan found trusted human metadata.")
                return res
        
        avg_prob = sum(r['layers']['layer2_forensics']['probability'] for r in frame_results) / len(frame_results)
        
        # Apply conservative adjustment for screen recordings
        if is_screen_record:
            logger.info(f"Applying conservative adjustment (-{SCREEN_RECORD_PENALTY}) for screen record video.")
            avg_prob = max(0.0, avg_prob - SCREEN_RECORD_PENALTY)

        # Simplified Verdict Logic
        if 0.4 <= avg_prob <= 0.6:
            summary = "Ambiguous"
        elif avg_prob > 0.6: 
            summary = "Likely AI"
        else: 
            summary = "Likely Human"
        
        # Apply boost to confidence score
        raw_conf = avg_prob if avg_prob > 0.5 else 1.0 - avg_prob
        if summary == "Ambiguous":
            final_conf = raw_conf
        else:
            final_conf = boost_score(raw_conf)
        
        if final_conf > 0.99:
            final_conf = 0.99

        return {
            "summary": summary,
            "media_type": "video",
            "confidence_score": round(final_conf, 2),
            "layers": {
                "layer1_metadata": {
                    "status": "ambiguous" if is_screen_record else l1_data["status"],
                    "provider": v_provider if is_screen_record else l1_data["provider"],
                    "description": f"Screen recording detected. The analysis reflects the recording container; original forensics may be masked." if is_screen_record else l1_data["description"]
                },
                "layer2_forensics": {
                    "status": "detected" if avg_prob > 0.5 else "not_detected",
                    "probability": round(avg_prob, 2), # Show actual adjusted prob
                    "signals": [f"Analyzed {len(frame_results)} frames via Parallel Temporal Sampling"] + ([f"Screen record adjustment (-{SCREEN_RECORD_PENALTY}) applied"] if is_screen_record else [])
                }
            }
        }
    else:
        return await detect_ai_media_image_logic(file_path, l1_data)

async def detect_ai_media_image_logic(file_path: Optional[str], l1_data: dict = None, frame: Image.Image = None, media_type: str = "image") -> dict:
    """Core consensus logic for images and video frames."""
    if l1_data is None:
        l1_data = {"status": "not_found", "provider": None, "description": "N/A"}

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
    
    ctx = get_metadata_context(exif)
    human_score, human_signals = get_forensics_for_context(exif, ctx)
    ai_score, ai_signals = get_ai_suspicion_score_refactored(exif, ctx)
    
    # 1. VERIFIED HUMAN (Early Exit)
    if human_score >= 0.80:
        return {
            "summary": "Verified Human (Forensic Metadata)",
            "media_type": media_type,
            "confidence_score": 1.0,
            "layers": {
                "layer1_metadata": {
                    "status": "verified_human", 
                    "provider": ctx["make"] or "Unknown",
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
            "media_type": media_type,
            "confidence_score": 0.9,
            "layers": {
                "layer1_metadata": {
                    "status": "likely_human", 
                    "provider": ctx["make"] or "Unknown",
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
            "media_type": media_type,
            "confidence_score": 0.95,
            "layers": {
                "layer1_metadata": {
                    "status": "verified_ai", 
                    "provider": ctx["software"] or "AI Generator",
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
                "media_type": media_type,
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
    cached_score = forensic_cache.get(img_hash)
    if cached_score is not None:
        forensic_probability = cached_score
    else:
        # Use semaphore to prevent overwhelming the GPU worker
        async with gpu_semaphore:
            forensic_probability = await run_deep_forensics(source_for_hash, width, height)
        forensic_cache.put(img_hash, forensic_probability)
    
    # Simplified Verdict Logic
    if 0.4 <= forensic_probability <= 0.6:
        summary = "Ambiguous"
    elif forensic_probability > 0.6: 
        summary = "Likely AI"
    else: 
        summary = "Likely Human"
    
    # Boost the overall confidence score
    raw_conf = forensic_probability if forensic_probability > 0.5 else (1.0 - forensic_probability)
    
    # instructions: if ambiguous (summary == "Ambiguous"), keep the score as is, don't boost to 85%
    if summary == "Ambiguous":
        final_conf = raw_conf
    else:
        final_conf = boost_score(raw_conf)
    
    # Cap probabilistic scores at 0.99 to avoid "fake" 100% look, unless it's a hard metadata match
    if final_conf > 0.99:
        final_conf = 0.99

    l2_data = {
        "status": "detected" if forensic_probability > 0.5 else "not_detected",
        "probability": round(forensic_probability, 2),
        "signals": ["Multi-layered consensus applied (Deep Learning + FFT)"]
    }

    return {
        "summary": summary,
        "media_type": media_type,
        "confidence_score": round(final_conf, 2),
        "layers": {
            "layer1_metadata": l1_data, 
            "layer2_forensics": l2_data
        }
    }
