"""
Video-specific detection utilities.

Handles frame extraction (Tri-Frame Strategy), frame quality filtering,
ffprobe-based metadata extraction, and video metadata scoring.
"""

import asyncio
import json
import logging
import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def is_frame_quality_ok(
    frame: np.ndarray,
    min_brightness: float = settings.frame_min_brightness,
    min_sharpness: float = settings.frame_min_sharpness
) -> tuple:
    """
    Check if frame is not too dark or blurry for reliable AI detection.
    Returns (is_ok, brightness, sharpness) for potential weighted aggregation.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        if brightness < min_brightness:
            return False, brightness, 0.0

        h, w = gray.shape
        center_crop = gray[h//4:3*h//4, w//4:3*w//4]
        laplacian_var = cv2.Laplacian(center_crop, cv2.CV_64F).var()
        if laplacian_var < min_sharpness:
            return False, brightness, laplacian_var

        return True, brightness, laplacian_var
    except Exception:
        return True, 128.0, 100.0


def extract_video_frames(video_path: str) -> tuple:
    """
    Extract 3 frames at 20%, 50%, 80% of video duration (Tri-Frame Strategy).
    Optimized for batch GPU processing - 3 frames process in ~same time as 1.
    Returns (frames, quality_rejected_count)
    """
    frames = []
    quality_rejected = 0

    try:
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
                is_ok, brightness, sharpness = is_frame_quality_ok(frame)
                if not is_ok:
                    quality_rejected += 1
                    continue

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), settings.video_jpeg_quality]
                success, encoded_image = cv2.imencode('.jpg', frame, encode_param)

                if success:
                    frames.append(encoded_image.tobytes())

        cap.release()

        if quality_rejected > 0:
            logger.info(f"[VIDEO] Skipped {quality_rejected} low-quality frames (dark/blurry)")

        # Fallback if too many frames rejected
        if len(frames) < 1 and total_frames >= 1:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.5))
            ret, frame = cap.read()
            if ret:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), settings.video_jpeg_quality]
                success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
                if success:
                    frames.append(encoded_image.tobytes())
            cap.release()

    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")

    return frames, quality_rejected


async def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffprobe (async to avoid blocking event loop)."""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=settings.ffprobe_timeout_sec)
            if proc.returncode == 0:
                return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            logger.warning(f"ffprobe timeout for {video_path}")
    except FileNotFoundError:
        logger.warning("ffprobe not installed")
    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}")
    finally:
        if proc is not None:
            try:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass

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

    tags_lower = {k.lower(): v for k, v in tags.items()}
    encoder = str(tags_lower.get("encoder", "")).lower()

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

    has_ffmpeg = "lavf" in encoder
    has_x264 = "x264" in encoder

    for stream in streams:
        stream_tags = stream.get("tags", {})
        stream_encoder = str(stream_tags.get("encoder", "")).lower()
        if "x264" in stream_encoder:
            has_x264 = True
        if "lavf" in stream_encoder:
            has_ffmpeg = True

    if file_path and has_ffmpeg and not has_x264:
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(settings.video_header_read_bytes)
                if b'x264' in chunk:
                    has_x264 = True
        except Exception:
            pass

    video_stream = None
    audio_stream = None
    for stream in streams:
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream
        if video_stream and audio_stream:
            break

    duration = 0.0
    bit_rate = 0
    try:
        duration = float(format_info.get("duration", 0))
        bit_rate = int(format_info.get("bit_rate", 0))
    except Exception:
        pass

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
        except Exception:
            fps = 0.0

    is_exact_round_fps = abs(fps - 30.0) < 0.001 or abs(fps - 60.0) < 0.001 or abs(fps - 24.0) < 0.001

    width = video_stream.get("width", 0) if video_stream else 0
    height = video_stream.get("height", 0) if video_stream else 0

    format_handler = tags_lower.get("handler_name", "").lower()
    make = tags_lower.get("make", "").lower()
    model = tags_lower.get("model", "").lower()

    stream_handler = ""
    if video_stream:
        stream_handler = str(video_stream.get("tags", {}).get("handler_name", "")).lower()

    # === HUMAN SIGNALS ===
    if device_marker:
        human_score += 0.50

    if fps > 0 and not is_exact_round_fps:
        human_score += 0.10
        signals.append(f"Device-native FPS: {fps:.4f}")

    if make or model:
        human_score += 0.25
        signals.append("Device manufacturer/model metadata present")

    # Partial matching covers EXIF, XMP, Apple QuickTime, and Android (©xyz) location tags
    location_markers = ["gps", "location", "coordinates", "xyz", "latitud", "longitud"]

    for key, value in tags_lower.items():
        if any(term in key for term in location_markers):
            val_str = str(value).strip()
            if value and val_str not in ["0", "0.0", "+0.0000+000.0000/", "None", "[]"]:
                human_score += 0.35
                signals.append(f"GPS/Location data present: {key}")
                break

    creation_time = tags_lower.get("creation_time", "")
    if creation_time and ("+" in creation_time or "Z" in creation_time):
        human_score += 0.10
        signals.append("Creation time with timezone")

    if audio_stream:
        channels = audio_stream.get("channels", 0)
        if channels == 1:
            human_score += 0.05
            signals.append("Mono audio (phone typical)")
        elif channels == 2:
            ai_score += 0.05
            signals.append("Stereo audio")

    if video_stream:
        rotation = video_stream.get("tags", {}).get("rotate", "")
        side_data = video_stream.get("side_data_list", [])
        has_rotation = rotation or any(sd.get("rotation") for sd in side_data if isinstance(sd, dict))
        if has_rotation:
            human_score += 0.10
            signals.append("Video rotation metadata (phone typical)")

    if duration > 30:
        human_score += 0.05
        signals.append(f"Long duration ({duration:.1f}s) - human typical")
    elif 2 < duration <= 5:
        ai_score += 0.05
        signals.append(f"Short duration ({duration:.1f}s) - AI typical")

    if avg_frame_rate and r_frame_rate and avg_frame_rate != r_frame_rate:
        human_score += 0.05
        signals.append("Variable frame rate detected (phone/screen rec typical)")

    handler_to_check = stream_handler or format_handler
    if "core media" in handler_to_check or "apple" in handler_to_check:
        human_score += 0.15
        signals.append("Apple Core Media handler")
    elif "android" in handler_to_check or "media handler" in handler_to_check:
        human_score += 0.10
        signals.append("Android media handler")

    if bit_rate > 0 and duration > 0:
        if duration < 15 and bit_rate > 15_000_000:  # >15 Mbps for short video
            ai_score += 0.05
            signals.append(f"High bitrate short video ({bit_rate//1000}kbps)")

    # HDR/BT.2020 color profiles — real cameras use wide gamut; AI generators almost always SDR
    if video_stream:
        color_primaries = video_stream.get("color_primaries", "unknown")
        color_transfer = video_stream.get("color_transfer", "unknown")

        if "bt2020" in color_primaries:
            human_score += 0.45
            signals.append(f"Wide Color Gamut detected ({color_primaries})")

        # arib-std-b67 = HLG (iPhone HDR), smpte2084 = PQ (Samsung/Cinema HDR)
        if "arib-std-b67" in color_transfer or "smpte2084" in color_transfer:
            human_score += 0.45
            signals.append(f"HDR Transfer Function detected ({color_transfer})")

    major_brand = tags_lower.get("major_brand", "").lower()

    # "qt  " with spaces is Apple's QuickTime signature
    if "qt  " in major_brand or "qt" == major_brand.strip():
        human_score += 0.18
        signals.append("Apple QuickTime Container (major_brand)")

    if "mp42" in major_brand:
        human_score += 0.14
        signals.append("Android/Camera Container (major_brand: mp42)")

    # === AI SIGNALS ===
    if has_ffmpeg and has_x264 and not device_marker:
        ai_score += 0.50
        signals.append("FFmpeg/x264 encoding without device marker (AI typical)")
    elif has_ffmpeg and not device_marker:
        ai_score += 0.15
        signals.append("FFmpeg encoder without device marker")

    if is_exact_round_fps:
        ai_score += 0.15
        signals.append(f"Exact round FPS: {fps:.4f} (synthetic typical)")

    ai_encoders = ["runway", "pika", "sora", "kling", "luma", "midjourney",
                   "stable video", "deforum", "animatediff", "svd", "cogvideo", "gen-2"]
    for ai_enc in ai_encoders:
        if ai_enc in encoder:
            ai_score += 0.80
            signals.append(f"AI generator in encoder: {ai_enc}")
            break

    filename_lower = filename.lower()
    ai_filename_markers = ["sora", "runway", "pika", "kling", "luma", "midjourney",
                           "stablediffusion", "cogvideo", "gen2", "animatediff"]
    for ai_name in ai_filename_markers:
        if ai_name in filename_lower:
            ai_score += 0.10
            signals.append(f"AI keyword in filename: {ai_name}")
            break

    if not tags:
        ai_score += 0.10
        signals.append("No metadata tags")

    if width > 0 and height > 0:
        if width == height and width in [512, 768, 1024, 1280]:
            ai_score += 0.15
            signals.append(f"AI-typical square resolution: {width}x{height}")

        elif (width, height) in [(1280, 720), (704, 1280), (1024, 576), (576, 1024)]:
            ai_score += 0.10
            signals.append(f"AI-typical resolution: {width}x{height}")

        elif width > 1000 and height > 1000:
            aspect = width / height if height > 0 else 0
            if 0.4 < aspect < 0.6 and width not in [720, 1080, 1280]:
                human_score += 0.05
                signals.append(f"Device-native resolution: {width}x{height}")
            elif aspect > 2.0:
                human_score += 0.05
                signals.append(f"Ultrawide resolution: {width}x{height}")

    human_score = min(1.0, human_score)
    ai_score = min(1.0, ai_score)

    early_exit = None
    if human_score >= 0.60 and ai_score < 0.30:
        early_exit = "human"
        signals.append(f"EARLY EXIT: Human (h={human_score:.2f}, ai={ai_score:.2f})")

    elif ai_score >= 0.70 and human_score < 0.20:
        early_exit = "ai"
        signals.append(f"EARLY EXIT: AI (h={human_score:.2f}, ai={ai_score:.2f})")

    return human_score, ai_score, signals, early_exit
