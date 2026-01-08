"""
Configuration for AI Detection Scoring Weights and Thresholds.
Centralizes "magic numbers" for easier tuning.
"""

class ScoringConfig:
    # --- Original Signal Weights ---
    ORIGINAL = {
        "DEVICE_MARKER": 0.50,
        "GPS_DATA": 0.30,
        "CAMERA_BRAND": 0.20,
        "CORE_MEDIA_HANDLER": 0.15,
        "ANDROID_HANDLER": 0.10,
        "NON_ROUND_FPS": 0.10,
        "TIMEZONE_CREATION": 0.10,
        "VIDEO_ROTATION": 0.10,
        "MONO_AUDIO": 0.05,
        "LONG_DURATION": 0.05,      # >30s
        "VARIABLE_FRAMERATE": 0.05,
        "NATIVE_RESOLUTION": 0.05,  # Odd phone resolutions
        # Forensic Tier 1
        "TRUSTED_MAKER": 0.35,      # Apple, Google, etc.
        "VALID_PIPELINE": 0.25,     # HDR+, iOS, etc.
        # Forensic Tier 2 (Physics)
        "VALID_EXPOSURE": 0.15,
        "VALID_ISO": 0.15,
        "VALID_APERTURE": 0.15,
        # Forensic Tier 3-7
        "TEMPORAL_TIMESTAMP": 0.08,
        "SUBSEC_TIME": 0.02,
        "BODY_SERIAL": 0.05,
        "LENS_SERIAL": 0.05,
        "LENS_MODEL": 0.05,
        "FLASH_FIRED": 0.05,
        "GPS_COORDS": 0.03,
        "GPS_ALTITUDE": 0.03,
        "GPS_TIMESTAMP": 0.02,
        "JPEG_STRUCTURE": 0.05,
        "STD_COMPRESSION": 0.05,
        # Pro editing software is NOT proof of “original camera capture”.
        # It can appear on both real photos and AI images that were post-processed.
        # Keep it as a weak signal (still useful for dashboards), but do not let it drive early-exit.
        "PRO_SOFTWARE": 0.05,
        "ICC_PROFILE": 0.08,       # Professional/Standard ICC tags
        "SENSOR_DATA": 0.25,       # SensingMethod (2), FocalPlaneRes
        "FILE_SOURCE": 0.15,       # FileSource (3), SceneType (1)
        "CFA_PATTERN": 0.10,       # CFA Pattern existence
        # Video container provenance (ffprobe tags)
        "MP4_BRAND_APPLE": 0.05,
        "WRITING_LIBRARY_APPLE": 0.07,
        # Image forensic (metadata-adjacent)
        "EMBEDDED_THUMBNAIL": 0.08,     # EXIF embedded JPEG thumbnail (very camera-typical)
        "JPEG_QTABLE_CAMERA_LIKE": 0.10, # Non-generic quant tables (camera-like pipeline)
        "NATURAL_DCT_DECAY": 0.10,      # DCT mid/high energy present (not over-smoothed)
        "VALID_MAKERNOTE": 0.12,        # MakerNote looks like real camera blob
    }

    # --- AI Signal Weights ---
    AI = {
        "HARD_GENERATOR": 1.0,      # Blatant AI Software/Generator name
        "GENERATOR_NAME": 0.80,     # e.g., "Sora" in encoder
        "FFMPEG_X264_NO_DEVICE": 0.50,
        "KEYWORD_SOFTWARE": 0.40,   # "Stable Diffusion" in software
        "KEYWORD_MAKE": 0.40,
        "FFMPEG_NO_DEVICE": 0.15,
        "EXACT_ROUND_FPS": 0.15,    # 30.000
        "SQUARE_RES": 0.12,         # [TUNE] Reduced to 0.12 to protect Amazon data
        "AI_TYPICAL_RES": 0.08,     # [TUNE] Reduced to 0.08
        "KEYWORD_FILENAME": 0.15,
        "NO_METADATA": 0.05,          # [SAFETY] Reduced to 0.05
        "TINY_IMAGE_AI_RES": 0.10,
        "LOW_BPP": 0.05,             # [SAFETY] Reduced to 0.05
        "SHORT_DURATION": 0.05,
        "HIGH_BITRATE_SHORT": 0.05,
        "STEREO_AUDIO": 0.05,
        # Suspicion
        "MISSING_CAMERA_INFO": 0.03,  # [SAFETY] Reduced to 0.03
        "MISSING_TIMESTAMP": 0.08,
        "MISSING_SUBSEC": 0.04,
        "NON_STD_JPEG": 0.03,
        "NON_STD_ASPECT": 0.05,      # [SAFETY] Reduced to 0.05
        # Embedded text chunks (PNG/JPEG comments)
        # Parameter blocks are common in Stable Diffusion tooling and are a strong AI marker.
        "AI_TEXT_PARAMS": 0.35,
        "AI_TEXT_PROMPT": 0.15,
        # Video container provenance (ffprobe tags)
        "WRITING_LIBRARY_FFMPEG": 0.10,
        # Image forensic (metadata-adjacent)
        "GENERIC_QTABLE": 0.05,         # libjpeg-like quant tables (weak AI hint; many re-encodes)
        "OVER_SMOOTH_DCT": 0.08,        # very low mid/high DCT energy ratio
        "BROKEN_OR_EMPTY_MAKERNOTE": 0.05,
    }

    # --- Thresholds ---
    THRESHOLDS = {
        # Early Exit
        # Tuned via benchmark sweep to maximize GPU bypass while keeping >=95% accuracy (metadata-only errors).
        # Tuned for higher bypass while keeping >=95% global accuracy across all datasets (GPU mocked).
        "HUMAN_EXIT_HIGH": 0.25,
        "HUMAN_EXIT_LOW": 0.08,
        # If ai_score is below this, we allow a cheaper "Likely Original" bypass.
        # (Tuned via benchmark sweep; keep conservative to avoid AI false negatives.)
        "HUMAN_LOW_AI_MAX": 0.30,
        # If ai_score is exactly 0 (no suspicious signals at all), we can optionally allow
        # a cheaper bypass when human_score is decent. Make it tunable because it can
        # create false negatives on AI that carries strong “camera-like” metadata.
        "HUMAN_LOW_NO_AI_MIN": 0.12,
        "AI_EXIT_HIGH": 0.70,       # For video metadata
        # If ai_score is above this, we allow a cheaper "Likely AI" bypass.
        # (Tuned via benchmark sweep; lower increases bypass but may increase false positives.)
        "AI_EXIT_META": 0.40,
        "AI_SUSPICIOUS": 0.15,       # [TUNE] Lowered to 0.15 - send more to GPU
        
        # Conflict Resolution
        "CONFLICT_AI_SCORE": 0.15,
        "CONFLICT_MODEL_LOW": 0.8,
        
        # Final Confidence
        "LIKELY_AI": 0.92,
        "POSSIBLE_AI": 0.75,
        "SUSPICIOUS_AI": 0.50,
        "LIKELY_HUMAN_NOISE": 0.20,
    }
