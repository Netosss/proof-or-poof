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

def get_image_hash(source: Union[str, Image.Image], fast_mode: bool = False) -> str:
    """
    Generate a secure SHA-256 hash of the image source (optimized with grayscale).
    fast_mode=True uses even smaller thumbnail for video frame caching.
    """
    if isinstance(source, str):
        with open(source, 'rb') as f:
            # Hash first 2MB for speed, but use secure method
            return security_manager.get_safe_hash(f.read(2048 * 1024))
    else:
        # For PIL Images: small grayscale thumbnail for fast, unique hash
        thumb = source.copy()
        # Use 32x32 for video frames (fast_mode), 64x64 for standalone images
        size = (32, 32) if fast_mode else (64, 64)
        thumb.thumbnail(size)
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

def extract_video_frames(video_path: str, num_frames: int = 8) -> tuple:
    """
    Extract N evenly-spaced quality frames from a video file.
    Returns (frames, quality_rejected_count, frame_qualities)
    frame_qualities: list of (brightness, sharpness) tuples for weighted aggregation
    """
    frames = []
    frame_qualities = []  # Store (brightness, sharpness) for weighted aggregation
    quality_rejected = 0
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0, []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return [], 0, []
        
        # Evenly spaced sampling points (skip first/last 5% to avoid black frames)
        start_frame = int(total_frames * 0.05)
        end_frame = int(total_frames * 0.95)
        sample_points = np.linspace(start_frame, end_frame, num=min(num_frames, total_frames), dtype=int)
        
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
                    # Use default quality for fallback frames
                    frame_qualities.append((128.0, 100.0))
            cap.release()
            
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
    
    return frames, quality_rejected, frame_qualities

async def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffprobe (async to avoid blocking event loop)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning("ffprobe timeout")
    except FileNotFoundError:
        logger.warning("ffprobe not installed")
    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}")
    return {}

def get_video_metadata_score(metadata: dict, filename: str = "", file_path: str = "") -> tuple:
    """
    Analyze video metadata for human vs AI signals using soft thresholds.
    Returns (human_score, ai_score, signals, early_exit_label)
    
    early_exit_label: "human", "ai", or None (continue to frame analysis)
    """
    human_score = 0.0
    ai_score = 0.0
    signals = []
    
    if not metadata:
        return 0.0, 0.0, ["No metadata available"], None
    
    format_info = metadata.get("format", {})
    tags = format_info.get("tags", {})
    streams = metadata.get("streams", [])
    
    # Normalize tag keys to lowercase for consistent matching
    tags_lower = {k.lower(): v for k, v in tags.items()}
    
    # Get encoder info
    encoder = str(tags_lower.get("encoder", "")).lower()
    
    # === DEVICE MARKER DETECTION ===
    
    # Check for Android device markers
    has_android_marker = False
    android_markers = ["com.android.version", "com.android.capture", "com.samsung"]
    for marker in android_markers:
        for key in tags_lower:
            if marker in key.lower():
                has_android_marker = True
                signals.append(f"Android device marker: {key}")
                break
        if has_android_marker:
            break
    
    # Check for iOS/Apple device markers  
    has_ios_marker = False
    apple_markers = [
        "com.apple.quicktime.make",
        "com.apple.quicktime.model", 
        "com.apple.quicktime.software",
        "com.apple.quicktime.creationdate"
    ]
    for marker in apple_markers:
        if marker in tags_lower or marker.replace(".", "_") in tags_lower:
            has_ios_marker = True
            signals.append(f"Apple device marker: {marker}")
            break
    
    device_marker = has_android_marker or has_ios_marker
    
    # Check for FFmpeg/x264 encoding
    has_ffmpeg = "lavf" in encoder
    has_x264 = "x264" in encoder
    
    # Also check stream-level encoder tags
    for stream in streams:
        stream_tags = stream.get("tags", {})
        stream_encoder = str(stream_tags.get("encoder", "")).lower()
        if "x264" in stream_encoder:
            has_x264 = True
        if "lavf" in stream_encoder:
            has_ffmpeg = True
    
    # Binary check for x264 in file (it's often in H.264 private data, not in tags)
    # Only do this if we have ffmpeg but not x264, and we have a file path
    if file_path and has_ffmpeg and not has_x264:
        try:
            with open(file_path, 'rb') as f:
                # Read first 1MB (encoder string is usually near the start)
                chunk = f.read(1024 * 1024)
                if b'x264' in chunk:
                    has_x264 = True
        except:
            pass
    
    # === PRE-COMPUTE STREAM INFO (efficiency: single pass) ===
    video_stream = None
    audio_stream = None
    for stream in streams:
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream
        if video_stream and audio_stream:
            break
    
    # Pre-compute common values
    duration = 0.0
    bit_rate = 0
    try:
        duration = float(format_info.get("duration", 0))
        bit_rate = int(format_info.get("bit_rate", 0))
    except:
        pass
    
    # FPS computation
    fps = 0.0
    avg_frame_rate = ""
    r_frame_rate = ""
    if video_stream:
        try:
            avg_frame_rate = video_stream.get("avg_frame_rate", "0/1")
            r_frame_rate = video_stream.get("r_frame_rate", "0/1")
            if "/" in avg_frame_rate:
                num, den = avg_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0
            else:
                fps = float(avg_frame_rate)
        except:
            fps = 0.0
    
    is_exact_round_fps = abs(fps - 30.0) < 0.001 or abs(fps - 60.0) < 0.001 or abs(fps - 24.0) < 0.001
    
    # Resolution
    width = video_stream.get("width", 0) if video_stream else 0
    height = video_stream.get("height", 0) if video_stream else 0
    
    # Handler names
    format_handler = tags_lower.get("handler_name", "").lower()
    make = tags_lower.get("make", "").lower()
    model = tags_lower.get("model", "").lower()
    
    # Stream-level handler
    stream_handler = ""
    if video_stream:
        stream_handler = str(video_stream.get("tags", {}).get("handler_name", "")).lower()
    
    # === HUMAN SIGNALS ===
    
    # 1. Device marker: STRONG human signal (+0.50)
    if device_marker:
        human_score += 0.50
    
    # 2. FPS analysis - Non-round FPS typical of real device
    if fps > 0 and not is_exact_round_fps:
        human_score += 0.10
        signals.append(f"Device-native FPS: {fps:.4f}")
    
    # 3. Known camera brands in encoder/handler
    camera_brands = ["iphone", "samsung", "google", "pixel", "gopro", "dji", 
                     "sony", "canon", "nikon", "panasonic", "fujifilm", "xiaomi", "huawei", "oneplus"]
    for brand in camera_brands:
        if brand in encoder or brand in format_handler or brand in make or brand in model:
            human_score += 0.20
            signals.append(f"Camera brand detected: {brand}")
            break
    
    # 4. GPS/Location data (strong human signal)
    gps_found = False
    gps_markers = ["location", "gps", "coordinates", "com.apple.quicktime.location"]
    for marker in gps_markers:
        for key in tags_lower:
            if marker in key:
                human_score += 0.30
                signals.append("GPS/Location data present")
                gps_found = True
                break
        if gps_found:
            break
    
    # 5. Creation time with timezone (phones add this)
    creation_time = tags_lower.get("creation_time", "")
    if creation_time and ("+" in creation_time or "Z" in creation_time):
        human_score += 0.10
        signals.append("Creation time with timezone")
    
    # 6. Audio channel analysis - mono typical of phone
    if audio_stream:
        channels = audio_stream.get("channels", 0)
        if channels == 1:
            human_score += 0.05
            signals.append("Mono audio (phone typical)")
        elif channels == 2:
            ai_score += 0.05
            signals.append("Stereo audio")
    
    # 7. Rotation metadata - phones often record with rotation
    if video_stream:
        rotation = video_stream.get("tags", {}).get("rotate", "")
        side_data = video_stream.get("side_data_list", [])
        has_rotation = rotation or any(sd.get("rotation") for sd in side_data if isinstance(sd, dict))
        if has_rotation:
            human_score += 0.10
            signals.append("Video rotation metadata (phone typical)")
    
    # 8. Duration analysis - AI videos typically 3-10s, human videos vary
    if duration > 30:
        human_score += 0.05
        signals.append(f"Long duration ({duration:.1f}s) - human typical")
    elif 2 < duration <= 5:
        ai_score += 0.05
        signals.append(f"Short duration ({duration:.1f}s) - AI typical")
    
    # 9. Variable frame rate / VBR - common in phone recordings
    if avg_frame_rate and r_frame_rate and avg_frame_rate != r_frame_rate:
        human_score += 0.05
        signals.append("Variable frame rate detected (phone/screen rec typical)")
    
    # 10. Handler name analysis - can reveal recording device
    handler_to_check = stream_handler or format_handler
    if "core media" in handler_to_check or "apple" in handler_to_check:
        human_score += 0.15
        signals.append("Apple Core Media handler")
    elif "android" in handler_to_check or "media handler" in handler_to_check:
        human_score += 0.10
        signals.append("Android media handler")
    
    # 11. Bitrate analysis - AI videos often have very consistent/high bitrates
    if bit_rate > 0 and duration > 0:
        if duration < 15 and bit_rate > 15_000_000:  # >15 Mbps for short video
            ai_score += 0.05
            signals.append(f"High bitrate short video ({bit_rate//1000}kbps)")
    
    # === AI SIGNALS ===
    
    # 1. FFmpeg + x264 + NO device marker = STRONG AI signal (+0.50)
    if has_ffmpeg and has_x264 and not device_marker:
        ai_score += 0.50
        signals.append("FFmpeg/x264 encoding without device marker (AI typical)")
    elif has_ffmpeg and not device_marker:
        # Just FFmpeg without device marker is weaker signal
        ai_score += 0.15
        signals.append("FFmpeg encoder without device marker")
    
    # 2. Exact round FPS (30/60/24) = synthetic signal (+0.15)
    if is_exact_round_fps:
        ai_score += 0.15
        signals.append(f"Exact round FPS: {fps:.4f} (synthetic typical)")
    
    # 3. Known AI video generators in encoder
    ai_encoders = ["runway", "pika", "sora", "kling", "luma", "midjourney", 
                   "stable video", "deforum", "animatediff", "svd", "cogvideo", "gen-2"]
    for ai_enc in ai_encoders:
        if ai_enc in encoder:
            ai_score += 0.80
            signals.append(f"AI generator in encoder: {ai_enc}")
            break
    
    # 4. Known AI in filename: weak AI signal (+0.10)
    filename_lower = filename.lower()
    ai_filename_markers = ["sora", "runway", "pika", "kling", "luma", "midjourney", 
                           "stablediffusion", "cogvideo", "gen2", "animatediff"]
    for ai_name in ai_filename_markers:
        if ai_name in filename_lower:
            ai_score += 0.10
            signals.append(f"AI keyword in filename: {ai_name}")
            break
    
    # 5. No metadata at all (slightly suspicious for modern videos)
    if not tags:
        ai_score += 0.10
        signals.append("No metadata tags")
    
    # 6. Resolution analysis (using pre-computed width/height)
    if width > 0 and height > 0:
        # Square resolutions are common in AI
        if width == height and width in [512, 768, 1024, 1280]:
            ai_score += 0.15
            signals.append(f"AI-typical square resolution: {width}x{height}")
        
        # AI-typical resolutions (common render sizes)
        elif (width, height) in [(1280, 720), (704, 1280), (1024, 576), (576, 1024)]:
            ai_score += 0.10
            signals.append(f"AI-typical resolution: {width}x{height}")
        
        # Device-native resolutions (phone screens often have odd sizes)
        # These are real phone resolutions that AI generators don't use
        elif width > 1000 and height > 1000:
            aspect = width / height if height > 0 else 0
            # Very tall aspect (phone portrait) with non-standard width
            if 0.4 < aspect < 0.6 and width not in [720, 1080, 1280]:
                human_score += 0.05
                signals.append(f"Device-native resolution: {width}x{height}")
            # Very wide aspect (ultrawide screen recording)
            elif aspect > 2.0:
                human_score += 0.05
                signals.append(f"Ultrawide resolution: {width}x{height}")
    
    # Cap scores at 1.0
    human_score = min(1.0, human_score)
    ai_score = min(1.0, ai_score)
    
    # === EARLY EXIT LOGIC ===
    early_exit = None
    
    # Strong human signal: exit immediately if device marker + other cues
    if human_score >= 0.60 and ai_score < 0.30:
        early_exit = "human"
        signals.append(f"EARLY EXIT: Human (h={human_score:.2f}, ai={ai_score:.2f})")
    
    # Strong AI signal: exit if clear AI markers
    elif ai_score >= 0.70 and human_score < 0.20:
        early_exit = "ai"
        signals.append(f"EARLY EXIT: AI (h={human_score:.2f}, ai={ai_score:.2f})")
    
    return human_score, ai_score, signals, early_exit

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

    # --- Tier 1: Device Provenance (Max 0.60) ---
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()
    
    if any(m in make for m in ["apple", "google", "samsung", "sony", "canon", "nikon", "fujifilm", "panasonic", "olympus", "leica"]):
        score += 0.35  # Increased from 0.30
        signals.append("Trusted device manufacturer provenance")
    
    if any(s in software for s in ["hdr+", "ios", "android", "deep fusion", "one ui", "version", "lightroom", "capture one"]):
        score += 0.25
        signals.append("Validated vendor-specific camera pipeline")

    # --- Tier 2: Physical Camera Consistency (Max 0.45) ---
    # Signal 2.1: Exposure Time - Valid range: 0 < exp < 30s
    exp = to_float(exif.get("ExposureTime"))
    if exp is not None and 0 < exp < 30:
        score += 0.15  # Increased from 0.10
        signals.append("Physically valid exposure duration")
    
    # Signal 2.2: ISO Speed - Valid range: 50 < ISO < 102400
    iso = to_float(exif.get("ISOSpeedRatings"))
    if iso is not None and 50 <= iso <= 102400:
        score += 0.15  # Increased from 0.10
        signals.append("Realistic sensor sensitivity (ISO)")
        
    # Signal 2.3: Aperture/F-Number - Valid range: 0.95 < f < 32
    f_num = to_float(exif.get("FNumber"))
    if f_num is not None and 0.95 <= f_num <= 32:
        score += 0.15  # Increased from 0.10
        signals.append("Valid physical aperture geometry")

    # --- Tier 3: Temporal Authenticity (Max 0.10) ---
    if "DateTimeOriginal" in exif:
        score += 0.08  # Increased from 0.03
        signals.append("Temporal capture timestamp present")
        
    subsec = str(exif.get("SubSecTimeOriginal", exif.get("SubSecTimeDigitized", "")))
    if subsec and subsec.isdigit() and subsec not in ["000", "000000"]:
        score += 0.02
        signals.append("High-precision sensor timing")

    # --- Tier 4: Hardware Serial Numbers (Max 0.10) - Strong provenance ---
    # Camera body serial number - unique hardware ID
    body_serial = exif.get("BodySerialNumber", "")
    if body_serial and len(str(body_serial)) >= 6:
        score += 0.05
        signals.append(f"Camera body serial: {str(body_serial)[:8]}...")
    
    # Lens serial number - confirms physical lens
    lens_serial = exif.get("LensSerialNumber", "")
    if lens_serial and len(str(lens_serial)) >= 4:
        score += 0.05
        signals.append(f"Lens serial number present")

    # --- Tier 5: Lens & Flash Data (Max 0.10) ---
    # Lens model - AI won't have real lens info
    lens_model = exif.get("LensModel", "")
    if lens_model and len(str(lens_model)) > 3:
        score += 0.05
        signals.append(f"Lens model: {str(lens_model)[:30]}")
    
    # Flash data - physical sensor event
    flash = exif.get("Flash")
    if flash is not None and flash > 0:
        score += 0.05
        signals.append("Flash sensor event recorded")

    # --- Tier 6: GPS Data (Max 0.08) - Strong hardware provenance ---
    # GPS coordinates
    if "GPSLatitude" in exif or "GPSLongitude" in exif:
        score += 0.03
        signals.append("GPS coordinates present")
    
    # GPS altitude - very strong human signal (requires actual GPS hardware)
    gps_alt = exif.get("GPSAltitude")
    if gps_alt is not None:
        try:
            alt = float(gps_alt) if not isinstance(gps_alt, tuple) else float(gps_alt[0]) / float(gps_alt[1])
            if -500 < alt < 10000:  # Valid altitude range
                score += 0.03
                signals.append(f"GPS altitude: {alt:.0f}m")
        except:
            pass
    
    # GPS timestamp - separate from photo timestamp, confirms GPS fix
    if "GPSDateStamp" in exif or "GPSTimeStamp" in exif:
        score += 0.02
        signals.append("GPS timestamp present")

    # --- Tier 7: JPEG Structure (Max 0.10) ---
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
        filename = os.path.basename(file_path)
        logger.info(f"Detecting AI in video: {safe_path}")
        
        # --- VIDEO METADATA EARLY EXIT ---
        t_video_meta = time.perf_counter()
        video_metadata = await get_video_metadata(file_path)
        human_score, ai_meta_score, meta_signals, early_exit = get_video_metadata_score(video_metadata, filename, file_path)
        video_meta_time_ms = (time.perf_counter() - t_video_meta) * 1000
        logger.info(f"[TIMING] Video metadata extraction: {video_meta_time_ms:.2f}ms")
        logger.info(f"[VIDEO META] Human score: {human_score:.2f}, AI score: {ai_meta_score:.2f}")
        logger.info(f"[VIDEO META] Signals: {meta_signals}")
        logger.info(f"[VIDEO META] Early exit: {early_exit}")
        
        # Early exit: Strong human metadata (camera/phone recording)
        if early_exit == "human":
            logger.info(f"[VIDEO] Early exit: Verified Human via metadata (h={human_score:.2f}, ai={ai_meta_score:.2f})")
            return {
                "summary": "Verified Human Video",
                "confidence_score": 1.0,
                "layers": {
                    "layer1_metadata": {
                        "status": "verified_human",
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
        
        # Early exit: Strong AI metadata (known AI generator)
        if early_exit == "ai":
            logger.info(f"[VIDEO] Early exit: AI Generator detected via metadata (h={human_score:.2f}, ai={ai_meta_score:.2f})")
            return {
                "summary": "Verified AI Video",
                "confidence_score": 1.0,
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
        logger.info(f"[VIDEO] No early exit, proceeding to frame analysis...")
        frames, quality_rejected, frame_qualities = extract_video_frames(file_path)
        if not frames:
            return {"error": "Could not extract frames from video."}
        
        logger.info(f"[VIDEO] Extracted {len(frames)} frames (rejected {quality_rejected} low-quality)")
            
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
        
        # Calculate quality weights (higher brightness + sharpness = more reliable)
        # Normalize weights so they sum to 1
        if frame_qualities and len(frame_qualities) == len(frame_probs):
            # Quality score = brightness * sharpness (both contribute)
            # Use max(0.01, ...) to avoid zero weights while still allowing differentiation
            # Typical values: brightness ~50-200, sharpness ~50-500
            # Normalized: (b/255) * (s/200) gives ~0.05-0.8 range
            quality_scores = [max(0.01, (b / 255.0) * (s / 200.0)) for b, s in frame_qualities]
            total_quality = sum(quality_scores)
            quality_weights = [q / total_quality for q in quality_scores] if total_quality > 0 else [1.0/len(frame_probs)] * len(frame_probs)
            
            # Weighted mean
            weighted_mean = sum(p * w for p, w in zip(frame_probs, quality_weights))
            logger.info(f"[VIDEO] Quality weights: {[f'{w:.2f}' for w in quality_weights]} (scores: {[f'{q:.3f}' for q in quality_scores]})")
        else:
            quality_weights = None
            weighted_mean = float(np.mean(frame_probs))
        
        # Use MEDIAN instead of mean (more robust to outliers)
        median_prob = float(np.median(frame_probs))
        mean_prob = float(np.mean(frame_probs))
        std_prob = float(np.std(frame_probs))
        
        logger.info(f"[VIDEO] Frame analysis: {num_frames_analyzed} frames | "
                   f"Median: {median_prob:.3f}, Mean: {mean_prob:.3f}, Std: {std_prob:.3f}, "
                   f"Weighted: {weighted_mean:.3f}")
        
        # Temporal consistency check
        # If high variance (some frames AI, some human), be conservative
        # NOTE: Threshold 0.35 may need dataset tuning for borderline cases
        VARIANCE_THRESHOLD = 0.35
        if std_prob > VARIANCE_THRESHOLD:
            logger.info(f"[VIDEO] High variance detected ({std_prob:.3f}) - inconsistent frames")
            # If most frames are human-like, trust the majority
            human_frames = sum(1 for p in frame_probs if p < 0.5)
            if human_frames > len(frame_probs) / 2:
                # Majority human - use quality-weighted mean if available, else trimmed mean
                if quality_weights:
                    # Weight by quality: dark/blurry frames contribute less
                    low_idx = [i for i, p in enumerate(frame_probs) if p < 0.5]
                    if low_idx:
                        low_sum = sum(frame_probs[i] * quality_weights[i] for i in low_idx)
                        low_weight = sum(quality_weights[i] for i in low_idx)
                        final_prob = low_sum / low_weight if low_weight > 0 else median_prob
                    else:
                        final_prob = median_prob
                else:
                    low_probs = [p for p in frame_probs if p < 0.5]
                    final_prob = float(np.mean(low_probs)) if low_probs else median_prob
                logger.info(f"[VIDEO] Majority human ({human_frames}/{num_frames_analyzed}), using quality-weighted: {final_prob:.3f}")
            else:
                # Mixed or majority AI - use quality-weighted mean (conservative)
                final_prob = weighted_mean if quality_weights else median_prob
        else:
            # Consistent frames - use quality-weighted mean
            final_prob = weighted_mean if quality_weights else median_prob
        
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
            f"Analyzed {num_frames_analyzed} frames (quality-weighted aggregation)",
            f"Frame scores: median={median_prob:.2f}, weighted={weighted_mean:.2f}, std={std_prob:.2f}"
        ]
        if std_prob > VARIANCE_THRESHOLD:
            analysis_signals.append(f"High variance ({std_prob:.2f}) - used conservative estimate")
        if meta_signals:
            analysis_signals.extend(meta_signals[:2])  # Add top 2 metadata signals
        
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
            }
        }
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

    # Use fast_mode for video frames (smaller hash thumbnail for speed)
    img_hash = get_image_hash(source_for_hash, fast_mode=(frame is not None))
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