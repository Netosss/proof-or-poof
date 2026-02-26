"""
Metadata-based scoring for AI vs. human-origin detection.

Functions:
  - get_exif_data: Extracts EXIF/PIL metadata from an image file.
  - get_tiered_signature_score: Scores metadata text for known AI generator signatures.
  - get_forensic_metadata_score: Scores human-camera-provenance evidence.
  - get_ai_suspicion_score: Scores AI-indicator signals from EXIF + dimensions.
"""

import re
import json
import logging
from PIL import Image
from PIL.ExifTags import TAGS

from app.detection.constants import TIER_1_GENERATORS, TIER_2_TECH_TERMS, TIER_3_REGEX

logger = logging.getLogger(__name__)


def get_exif_data(file_path: str) -> dict:
    """
    Extract metadata from the image (EXIF for JPEG/TIFF, 'info' for PNG/WebP).
    Explicitly closed via 'with'.
    """
    try:
        with Image.open(file_path) as img:
            metadata = {}

            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    decoded = TAGS.get(tag, tag)
                    metadata[decoded] = value

            if hasattr(img, 'info') and img.info:
                for key, value in img.info.items():
                    if key == "icc_profile":
                        metadata["icc_profile"] = "present"
                    elif isinstance(key, str) and isinstance(value, (str, int, float)):
                        if key not in metadata:
                            metadata[key] = value

            return metadata
    except Exception:
        return {}


def get_tiered_signature_score(full_dump: str, clean_dump: str) -> tuple:
    """
    Returns (score, signals_list) based on tiered metadata signatures.

    full_dump: Includes raw binary text scan (Tier 1 & 2 only)
    clean_dump: Includes only parsed EXIF/PIL info (Tier 3 only)
    """
    score = 0.0
    signals = []

    full_dump = full_dump.lower()
    clean_dump = clean_dump.lower()

    for word in TIER_1_GENERATORS:
        if word in full_dump:
            signals.append(f"Found definite generator signature: '{word}'")
            return 0.99, signals

    tier_2_hits = 0
    for word in TIER_2_TECH_TERMS:
        if len(word) <= 4:
            if re.search(r'\b' + re.escape(word) + r'\b', clean_dump):
                tier_2_hits += 1
                signals.append(f"Found technical artifact in metadata: '{word}'")
        else:
            if word in full_dump:
                tier_2_hits += 1
                signals.append(f"Found technical artifact: '{word}'")

    if tier_2_hits > 0:
        score += 0.90 + ((tier_2_hits - 1) * 0.40)

    if score < 0.99:
        matches = re.findall(TIER_3_REGEX, clean_dump)
        if matches:
            unique_matches = list(set(matches))
            score += 0.25
            signals.append(f"Found generic AI keywords in metadata: {unique_matches}")

    return min(score, 0.99), signals


def get_forensic_metadata_score(exif: dict) -> tuple:
    """
    Advanced forensic check for human sensor physics using weighted tiers.
    Includes type and range validation to prevent metadata spoofing.
    """
    score = 0.0
    signals = []

    def to_float(val):
        try:
            return float(val)
        except Exception:
            return None

    # --- Tier 1: Device Provenance (Max 0.60) ---
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()

    if any(m in make for m in ["apple", "google", "samsung", "sony", "canon", "nikon", "fujifilm", "panasonic", "olympus", "leica"]):
        score += 0.35
        signals.append("Trusted device manufacturer provenance")

    if any(s in software for s in ["hdr+", "ios", "android", "deep fusion", "one ui", "version", "lightroom", "capture one"]):
        score += 0.25
        signals.append("Validated vendor-specific camera pipeline")

    # --- Tier 2: Physical Camera Consistency ---
    exp = to_float(exif.get("ExposureTime"))
    if exp is not None and 0 < exp < 30:
        score += 0.15
        signals.append("Physically valid exposure duration")

    iso = to_float(exif.get("ISOSpeedRatings"))
    if iso is not None and 50 <= iso <= 102400:
        score += 0.15
        signals.append("Realistic sensor sensitivity (ISO)")

    f_num = to_float(exif.get("FNumber"))
    if f_num is not None and 0.95 <= f_num <= 32:
        score += 0.15
        signals.append("Valid physical aperture geometry")

    # --- Tier 3: Temporal Authenticity ---
    if "DateTimeOriginal" in exif:
        score += 0.08
        signals.append("Temporal capture timestamp present")

    subsec = str(exif.get("SubSecTimeOriginal", exif.get("SubSecTimeDigitized", "")))
    if subsec and subsec.isdigit() and subsec not in ["000", "000000"]:
        score += 0.02
        signals.append("High-precision sensor timing")

    # --- Tier 4: Hardware Serial Numbers ---
    body_serial = exif.get("BodySerialNumber", "")
    if body_serial and len(str(body_serial)) >= 6:
        score += 0.05
        signals.append(f"Camera body serial: {str(body_serial)[:8]}...")

    lens_serial = exif.get("LensSerialNumber", "")
    if lens_serial and len(str(lens_serial)) >= 4:
        score += 0.05
        signals.append("Lens serial number present")

    # --- Tier 5: Lens & Flash Data ---
    lens_model = exif.get("LensModel", "")
    if lens_model and len(str(lens_model)) > 3:
        score += 0.05
        signals.append(f"Lens model: {str(lens_model)[:30]}")

    flash = exif.get("Flash")
    if flash is not None and flash > 0:
        score += 0.05
        signals.append("Flash sensor event recorded")

    # --- Tier 6: GPS Data ---
    if "GPSLatitude" in exif or "GPSLongitude" in exif:
        score += 0.03
        signals.append("GPS coordinates present")

    gps_alt = exif.get("GPSAltitude")
    if gps_alt is not None:
        try:
            alt = float(gps_alt) if not isinstance(gps_alt, tuple) else float(gps_alt[0]) / float(gps_alt[1])
            if -500 < alt < 10000:
                score += 0.03
                signals.append(f"GPS altitude: {alt:.0f}m")
        except Exception:
            pass

    if "GPSDateStamp" in exif or "GPSTimeStamp" in exif:
        score += 0.02
        signals.append("GPS timestamp present")

    # --- Tier 7: JPEG Structure ---
    if "JPEGInterchangeFormat" in exif:
        score += 0.05
        signals.append("Firmware-level segment tables")

    if exif.get("Compression") in [6, 1]:
        score += 0.05
        signals.append("Standard camera compression")

    # --- Tier 8: Color Profiles & Sensor Physics ---
    # ColorSpace 65535 = "Uncalibrated" / Wide Gamut (common in Apple Display P3)
    color_space = exif.get("ColorSpace")
    if color_space == 65535:
        score += 0.20
        signals.append("Wide Gamut / Uncalibrated Color Space (Human Typical)")

    if "icc_profile" in exif:
        score += 0.15
        signals.append("ICC Color Profile detected")

    if "Orientation" in exif:
        score += 0.10
        signals.append("Physical orientation sensor data present")

    # SensingMethod: 2 = One-chip color area sensor
    sensing_method = to_float(exif.get("SensingMethod"))
    if sensing_method == 2:
        score += 0.20
        signals.append("Authenticated sensor sensing method (Digital Camera)")

    if "FocalPlaneXResolution" in exif or "FocalPlaneYResolution" in exif:
        score += 0.15
        signals.append("High-fidelity focal plane resolution calibration")

    # --- Tier 9: Image Origin ---
    # FileSource: 3 = Digital Camera (can be int or bytes b'\x03')
    file_src = exif.get("FileSource")
    if file_src == 3 or file_src == b'\x03' or str(file_src) == '3':
        score += 0.15
        signals.append("Digital camera file source verified")

    # SceneType: 1 = Directly photographed
    scene_type = exif.get("SceneType")
    if scene_type == 1 or scene_type == b'\x01' or str(scene_type) == '1':
        score += 0.15
        signals.append("Directly photographed scene type (Non-synthetic)")

    if "CFAPattern" in exif:
        score += 0.10
        signals.append("Color Filter Array (CFA) pattern fingerprint")

    return round(score, 2), signals


def get_ai_suspicion_score(exif: dict, width: int = 0, height: int = 0, file_size: int = 0) -> tuple:
    """
    Weighted AI suspicion score based on blatant signatures, missing camera metadata,
    and image characteristics (dimensions, file size).
    """
    score = 0.0
    signals = []

    has_camera_info = exif.get("Make") or exif.get("Model")

    ai_keywords = ["stable", "diffusion", "midjourney", "dalle", "flux", "sora", "kling", "firefly", "generative", "artificial"]
    software = str(exif.get("Software", "")).lower()
    make = str(exif.get("Make", "")).lower()

    # AI tools often store signatures in XMP for PNG files
    if not software and "XML:com.adobe.xmp" in exif:
        software = str(exif.get("XML:com.adobe.xmp", "")).lower()

    if any(k in software for k in ai_keywords):
        score += 0.40
        log_software = software[:50] + "..." if len(software) > 50 else software
        signals.append(f"AI software signature detected: {log_software}")
    elif any(k in make for k in ai_keywords):
        score += 0.40
        signals.append(f"AI manufacturer signature: {make}")

    # 2. Missing Metadata (statistically unlikely for real cameras)
    if not has_camera_info:
        score += 0.03
        signals.append("Missing camera hardware provenance")

    if "DateTimeOriginal" not in exif:
        score += 0.02
        signals.append("Missing capture timestamp")

    if not exif.get("SubSecTimeOriginal") and not exif.get("SubSecTimeDigitized"):
        score += 0.03
        signals.append("Missing high-precision sensor timing")

    if "JPEGInterchangeFormat" not in exif:
        score += 0.05
        signals.append("Non-standard JPEG segment structure")

    if width > 0 and height > 0 and not has_camera_info:
        # Exclude 2048 — commonly used for social media uploads, not just AI
        ai_typical_widths = [512, 768, 1024, 1536]
        if width in ai_typical_widths:
            score += 0.15
            signals.append(f"AI-typical width: {width}px")
        elif height in ai_typical_widths:
            score += 0.15
            signals.append(f"AI-typical height: {height}px")

        total_pixels = width * height
        if total_pixels < 500000 and (width in ai_typical_widths or height in ai_typical_widths):
            score += 0.10
            signals.append(f"Small image ({total_pixels//1000}K pixels) with AI-typical dimensions")

        aspect = width / height if height > 0 else 0
        standard_aspects = [1.0, 1.33, 1.5, 1.78, 0.75, 0.67, 0.56, 1.0]
        is_standard = any(abs(aspect - std) < 0.08 for std in standard_aspects)

        # Phone screenshot detection: reduce AI suspicion to route to Gemini instead of early-flagging
        phone_widths = range(640, 1500)
        is_portrait = height > width
        is_phone_width = width in phone_widths
        is_phone_aspect = 0.40 < aspect < 0.60  # 9:16 to 9:22 range

        if is_portrait and is_phone_width and is_phone_aspect:
            score -= 0.10
            signals.append(f"Likely phone screenshot ({width}x{height})")
        elif not is_standard and aspect > 0:
            score += 0.05
            signals.append(f"Non-standard aspect ratio: {aspect:.2f}")

    if width > 0 and height > 0 and file_size > 0 and not has_camera_info:
        pixels = width * height
        bytes_per_pixel = file_size / pixels if pixels > 0 else 0
        if bytes_per_pixel < 0.15 and pixels > 500000:
            score += 0.10
            signals.append(f"Low bytes/pixel: {bytes_per_pixel:.2f} (heavily compressed)")
        elif bytes_per_pixel < 0.3 and pixels > 500000:
            score += 0.05
            signals.append(f"Compressed image: {bytes_per_pixel:.2f} bytes/pixel")

    return round(min(score, 1.0), 2), signals
