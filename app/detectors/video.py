
import logging
import cv2
import json
import asyncio
import time
import numpy as np
from PIL import Image
from app.scoring_config import ScoringConfig

logger = logging.getLogger(__name__)

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

def _extract_video_frames_sync(video_path: str, max_dimension: int = 720) -> tuple:
    """
    Synchronous frame extraction (runs in thread pool).
    Extracts 3 frames at 20%, 50%, 80% - resizes immediately to save memory.
    Returns (frames, quality_rejected_count)
    """
    frames = []
    quality_rejected = 0
    
    try:
        t_start = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return [], 0
        
        # Tri-Frame: 20%, 50%, 80% (avoids intro/outro black frames)
        sample_points = [
            int(total_frames * 0.20),
            int(total_frames * 0.50),
            int(total_frames * 0.80)
        ]
        
        for pos in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                # OPTIMIZATION: Resize immediately to save memory and speed up quality check
                h, w = frame.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Quality filter: skip dark/blurry frames
                is_ok, brightness, sharpness = is_frame_quality_ok(frame)
                if not is_ok:
                    quality_rejected += 1
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        extract_time_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[VIDEO] Extracted {len(frames)} frames in {extract_time_ms:.0f}ms (rejected {quality_rejected})")
        
        # Fallback if too many frames rejected
        if len(frames) < 1 and total_frames >= 1:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.5))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            cap.release()
            
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
    
    return frames, quality_rejected


async def extract_video_frames(video_path: str) -> tuple:
    """
    Async wrapper for frame extraction (runs in thread pool to avoid blocking).
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _extract_video_frames_sync, video_path)

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
    Analyze video metadata for original vs AI signals using soft thresholds.
    Returns (original_score, ai_score, signals, early_exit_label)
    
    early_exit_label: "original", "ai", or None (continue to frame analysis)
    """
    original_score = 0.0
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
    writing_library = str(tags_lower.get("writing_library", "")).lower()
    major_brand = str(tags_lower.get("major_brand", "")).lower()
    compatible_brands = str(tags_lower.get("compatible_brands", "")).lower()
    format_name = str(format_info.get("format_name", "")).lower()
    
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
    
    # === ORIGINAL SIGNALS ===
    
    # 1. Device marker
    if device_marker:
        original_score += ScoringConfig.ORIGINAL["DEVICE_MARKER"]

    # 1.1 Container provenance (mp4 brands / writing library)
    # These are not cryptographic proof, but they are strong hints of real phone camera pipelines.
    if has_ios_marker and ("qt" in major_brand or "qt" in compatible_brands or "mov" in format_name):
        original_score += ScoringConfig.ORIGINAL.get("MP4_BRAND_APPLE", 0.05)
        signals.append("Apple QuickTime brand/container signature")
    if writing_library and ("apple" in writing_library or "com.apple" in writing_library):
        original_score += ScoringConfig.ORIGINAL.get("WRITING_LIBRARY_APPLE", 0.08)
        signals.append("Apple writing_library signature")
    
    # 2. FPS analysis
    if fps > 0 and not is_exact_round_fps:
        original_score += ScoringConfig.ORIGINAL["NON_ROUND_FPS"]
        signals.append(f"Device-native FPS: {fps:.4f}")
    
    # 3. Known camera brands
    camera_brands = ["iphone", "samsung", "google", "pixel", "gopro", "dji", 
                     "sony", "canon", "nikon", "panasonic", "fujifilm", "xiaomi", "huawei", "oneplus"]
    for brand in camera_brands:
        if brand in encoder or brand in format_handler or brand in make or brand in model:
            original_score += ScoringConfig.ORIGINAL["CAMERA_BRAND"]
            signals.append(f"Camera brand detected: {brand}")
            break
    
    # 4. GPS/Location data
    gps_found = False
    gps_markers = ["location", "gps", "coordinates", "com.apple.quicktime.location"]
    for marker in gps_markers:
        for key in tags_lower:
            if marker in key:
                original_score += ScoringConfig.ORIGINAL["GPS_DATA"]
                signals.append("GPS/Location data present")
                gps_found = True
                break
        if gps_found:
            break
    
    # 5. Creation time with timezone
    creation_time = tags_lower.get("creation_time", "")
    if creation_time and ("+" in creation_time or "Z" in creation_time):
        original_score += ScoringConfig.ORIGINAL["TIMEZONE_CREATION"]
        signals.append("Creation time with timezone")
    
    # 6. Audio channel analysis
    if audio_stream:
        channels = audio_stream.get("channels", 0)
        if channels == 1:
            original_score += ScoringConfig.ORIGINAL["MONO_AUDIO"]
            signals.append("Mono audio (phone typical)")
        elif channels == 2:
            ai_score += ScoringConfig.AI["STEREO_AUDIO"]
            signals.append("Stereo audio")
    
    # 7. Rotation metadata
    if video_stream:
        rotation = video_stream.get("tags", {}).get("rotate", "")
        side_data = video_stream.get("side_data_list", [])
        has_rotation = rotation or any(sd.get("rotation") for sd in side_data if isinstance(sd, dict))
        if has_rotation:
            original_score += ScoringConfig.ORIGINAL["VIDEO_ROTATION"]
            signals.append("Video rotation metadata (phone typical)")
    
    # 8. Duration analysis
    if duration > 30:
        original_score += ScoringConfig.ORIGINAL["LONG_DURATION"]
        signals.append(f"Long duration ({duration:.1f}s) - original typical")
    elif 2 < duration <= 5:
        ai_score += ScoringConfig.AI["SHORT_DURATION"]
        signals.append(f"Short duration ({duration:.1f}s) - AI typical")
    
    # 9. Variable frame rate / VBR
    if avg_frame_rate and r_frame_rate and avg_frame_rate != r_frame_rate:
        original_score += ScoringConfig.ORIGINAL["VARIABLE_FRAMERATE"]
        signals.append("Variable frame rate detected (phone/screen rec typical)")
    
    # 10. Handler name analysis
    handler_to_check = stream_handler or format_handler
    if "core media" in handler_to_check or "apple" in handler_to_check:
        original_score += ScoringConfig.ORIGINAL["CORE_MEDIA_HANDLER"]
        signals.append("Apple Core Media handler")
    elif "android" in handler_to_check or "media handler" in handler_to_check:
        original_score += ScoringConfig.ORIGINAL["ANDROID_HANDLER"]
        signals.append("Android media handler")
    
    # 11. Bitrate analysis
    if bit_rate > 0 and duration > 0:
        if duration < 15 and bit_rate > 15_000_000:  # >15 Mbps for short video
            ai_score += ScoringConfig.AI["HIGH_BITRATE_SHORT"]
            signals.append(f"High bitrate short video ({bit_rate//1000}kbps)")
    
    # === AI SIGNALS ===
    
    # 1. FFmpeg + x264 + NO device marker
    if has_ffmpeg and has_x264 and not device_marker:
        ai_score += ScoringConfig.AI["FFMPEG_X264_NO_DEVICE"]
        signals.append("FFmpeg/x264 encoding without device marker (AI typical)")
    elif has_ffmpeg and not device_marker:
        ai_score += ScoringConfig.AI["FFMPEG_NO_DEVICE"]
        signals.append("FFmpeg encoder without device marker")

    # 1.1 writing_library can reveal ffmpeg even when encoder tag is absent
    if writing_library and ("lavf" in writing_library or "ffmpeg" in writing_library) and not device_marker:
        ai_score += ScoringConfig.AI.get("WRITING_LIBRARY_FFMPEG", 0.10)
        signals.append("FFmpeg writing_library without device marker")
    
    # 2. Exact round FPS
    if is_exact_round_fps:
        ai_score += ScoringConfig.AI["EXACT_ROUND_FPS"]
        signals.append(f"Exact round FPS: {fps:.4f} (synthetic typical)")
    
    # 3. Known AI video generators
    ai_encoders = ["runway", "pika", "sora", "kling", "luma", "midjourney", 
                   "stable video", "deforum", "animatediff", "svd", "cogvideo", "gen-2"]
    for ai_enc in ai_encoders:
        if ai_enc in encoder:
            ai_score += ScoringConfig.AI["GENERATOR_NAME"]
            signals.append(f"AI generator in encoder: {ai_enc}")
            break
    
    # 4. Known AI in filename
    filename_lower = filename.lower()
    ai_filename_markers = ["sora", "runway", "pika", "kling", "luma", "midjourney", 
                           "stablediffusion", "cogvideo", "gen2", "animatediff"]
    for ai_name in ai_filename_markers:
        if ai_name in filename_lower:
            ai_score += ScoringConfig.AI["KEYWORD_FILENAME"]
            signals.append(f"AI keyword in filename: {ai_name}")
            break
    
    # 5. No metadata at all
    if not tags:
        ai_score += ScoringConfig.AI["NO_METADATA"]
        signals.append("No metadata tags")
    
    # 6. Resolution analysis
    if width > 0 and height > 0:
        # Square resolutions are common in AI
        if width == height and width in [512, 768, 1024, 1280]:
            ai_score += ScoringConfig.AI["SQUARE_RES"]
            signals.append(f"AI-typical square resolution: {width}x{height}")
        
        # AI-typical resolutions
        elif (width, height) in [(1280, 720), (704, 1280), (1024, 576), (576, 1024)]:
            ai_score += ScoringConfig.AI["AI_TYPICAL_RES"]
            signals.append(f"AI-typical resolution: {width}x{height}")
        
        # Device-native resolutions
        elif width > 1000 and height > 1000:
            aspect = width / height if height > 0 else 0
            if 0.4 < aspect < 0.6 and width not in [720, 1080, 1280]:
                original_score += ScoringConfig.ORIGINAL["NATIVE_RESOLUTION"]
                signals.append(f"Device-native resolution: {width}x{height}")
            elif aspect > 2.0:
                original_score += ScoringConfig.ORIGINAL["NATIVE_RESOLUTION"]
                signals.append(f"Ultrawide resolution: {width}x{height}")
    
    # Cap scores at 1.0
    original_score = min(1.0, original_score)
    ai_score = min(1.0, ai_score)
    
    # === EARLY EXIT LOGIC ===
    early_exit = None
    
    if original_score >= ScoringConfig.THRESHOLDS["HUMAN_EXIT_HIGH"] and ai_score < 0.30:
        early_exit = "original"
        signals.append(f"EARLY EXIT: Original (h={original_score:.2f}, ai={ai_score:.2f})")
    
    elif ai_score >= ScoringConfig.THRESHOLDS["AI_EXIT_HIGH"] and original_score < 0.20:
        early_exit = "ai"
        signals.append(f"EARLY EXIT: AI (h={original_score:.2f}, ai={ai_score:.2f})")
    
    return original_score, ai_score, signals, early_exit
