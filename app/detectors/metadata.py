
from app.scoring_config import ScoringConfig

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

    # --- Tier 0: In-App Capture (Strongest Signal) ---
    if exif.get("CapturedInApp") is True:
        score += 0.70
        signals.append("Captured in app (strong provenance)")

    # --- Tier 1: Device Provenance ---
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()
    has_camera_provenance = bool(exif.get("Make") or exif.get("Model"))
    
    is_trusted_make = any(m in make for m in ["apple", "google", "samsung", "sony", "canon", "nikon", "fujifilm", "panasonic", "olympus", "leica"])
    if is_trusted_make:
        score += ScoringConfig.ORIGINAL["TRUSTED_MAKER"]
        signals.append("Trusted device manufacturer provenance")
    
    # Expand software list with pro-grade tools
    pro_software = ["photoshop", "lightroom", "capture one", "dxo", "affinity", "darktable", "luminar", "gimp"]
    if any(s in software for s in pro_software):
        signals.append(f"Professional media pipeline: {software}")
        # IMPORTANT (tuning): Pro software is NOT proof of original capture.
        # It's common on both real photos and AI images that were post-processed.
        # Only count it as "human" evidence when corroborated by camera provenance.
        if has_camera_provenance:
            score += ScoringConfig.ORIGINAL["PRO_SOFTWARE"]
    
    if any(s in software for s in ["hdr+", "ios", "android", "deep fusion", "one ui", "version"]):
        score += ScoringConfig.ORIGINAL["VALID_PIPELINE"]
        signals.append("Validated vendor-specific camera pipeline")

    # --- Tier 2: Physical Camera Consistency ---
    exp = to_float(exif.get("ExposureTime"))
    if exp is not None and 0 < exp < 30:
        score += ScoringConfig.ORIGINAL["VALID_EXPOSURE"]
        signals.append("Physically valid exposure duration")
    
    iso = to_float(exif.get("ISOSpeedRatings"))
    if iso is not None and 50 <= iso <= 102400:
        score += ScoringConfig.ORIGINAL["VALID_ISO"]
        signals.append("Realistic sensor sensitivity (ISO)")
        
    f_num = to_float(exif.get("FNumber"))
    if f_num is not None and 0.95 <= f_num <= 32:
        score += ScoringConfig.ORIGINAL["VALID_APERTURE"]
        signals.append("Valid physical aperture geometry")

    # --- Tier 3: Temporal Authenticity ---
    if "DateTimeOriginal" in exif:
        score += ScoringConfig.ORIGINAL["TEMPORAL_TIMESTAMP"]
        signals.append("Temporal capture timestamp present")
        
    subsec = str(exif.get("SubSecTimeOriginal", exif.get("SubSecTimeDigitized", "")))
    if subsec and subsec.isdigit() and subsec not in ["000", "000000"]:
        score += ScoringConfig.ORIGINAL["SUBSEC_TIME"]
        signals.append("High-precision sensor timing")

    # --- Tier 4: Hardware Serial Numbers ---
    body_serial = exif.get("BodySerialNumber", "")
    if body_serial and len(str(body_serial)) >= 6:
        score += ScoringConfig.ORIGINAL["BODY_SERIAL"]
        signals.append(f"Camera body serial: {str(body_serial)[:8]}...")
    
    lens_serial = exif.get("LensSerialNumber", "")
    if lens_serial and len(str(lens_serial)) >= 4:
        score += ScoringConfig.ORIGINAL["LENS_SERIAL"]
        signals.append(f"Lens serial number present")

    # --- Tier 5: Lens & Flash Data ---
    lens_model = exif.get("LensModel", "")
    if lens_model and len(str(lens_model)) > 3:
        score += ScoringConfig.ORIGINAL["LENS_MODEL"]
        signals.append(f"Lens model: {str(lens_model)[:30]}")
    
    flash = exif.get("Flash")
    if flash is not None:
        try:
            # Handle possible tuple/list return for Flash tag
            flash_int = int(flash) if not isinstance(flash, (tuple, list)) else int(flash[0])
            if flash_int > 0:
                score += ScoringConfig.ORIGINAL["FLASH_FIRED"]
                signals.append("Flash sensor event recorded")
        except:
            pass

    # --- Tier 6: GPS Data ---
    if "GPSLatitude" in exif or "GPSLongitude" in exif:
        score += ScoringConfig.ORIGINAL["GPS_COORDS"]
        signals.append("GPS coordinates present")
    
    gps_alt = exif.get("GPSAltitude")
    if gps_alt is not None:
        try:
            alt = float(gps_alt) if not isinstance(gps_alt, tuple) else float(gps_alt[0]) / float(gps_alt[1])
            if -500 < alt < 10000:
                score += ScoringConfig.ORIGINAL["GPS_ALTITUDE"]
                signals.append(f"GPS altitude: {alt:.0f}m")
        except:
            pass
    
    if "GPSDateStamp" in exif or "GPSTimeStamp" in exif:
        score += ScoringConfig.ORIGINAL["GPS_TIMESTAMP"]
        signals.append("GPS timestamp present")

    # --- Tier 7: JPEG Structure ---
    if "JPEGInterchangeFormat" in exif:
        score += ScoringConfig.ORIGINAL["JPEG_STRUCTURE"]
        signals.append("Firmware-level segment tables")

    # Embedded EXIF thumbnail (camera-typical, rare in AI)
    if exif.get("HasEmbeddedThumbnail"):
        score += ScoringConfig.ORIGINAL.get("EMBEDDED_THUMBNAIL", 0.08)
        signals.append("Embedded EXIF thumbnail present")
        
    if exif.get("Compression") in [6, 1]: 
        score += ScoringConfig.ORIGINAL["STD_COMPRESSION"]
        signals.append("Standard camera compression")

    if exif.get("HasICCProfile"):
        signals.append("Standard/Professional ICC color profile")
        # IMPORTANT (tuning): ICC profiles are extremely common on the web and can be
        # present on AI outputs. Never let ICC alone trigger "Likely Original".
        # Only count ICC as "human" evidence when corroborated by strong camera provenance.
        has_serials = bool(exif.get("BodySerialNumber") or exif.get("LensSerialNumber"))
        has_file_source = bool(exif.get("FileSource") or exif.get("SceneType"))
        has_embedded_thumb = bool(exif.get("HasEmbeddedThumbnail"))

        # MakerNote validation (from our extractor)
        has_valid_makernote = False
        if exif.get("HasMakerNote"):
            try:
                ln = int(exif.get("MakerNoteLength", 0))
                ent = float(exif.get("MakerNoteEntropy", 0.0))
                has_valid_makernote = (ln >= 256 and 4.0 <= ent <= 7.95)
            except Exception:
                has_valid_makernote = False

        icc_corroborated = any([
            has_camera_provenance,
            has_valid_makernote,
            has_serials,
            has_file_source,
            has_embedded_thumb,
        ])
        if icc_corroborated:
            score += ScoringConfig.ORIGINAL["ICC_PROFILE"]

    # JPEG quantization tables: non-generic tables are a strong hint of camera pipeline
    # IMPORTANT: camera-like QTables are necessary but not sufficient anymore.
    # Only treat them as human evidence if corroborated by at least one stronger camera-footprint signal.
    if exif.get("HasJPEGQuantTables") and not exif.get("JPEGQuantIsGeneric"):
        # Corroborators: hard-to-fake capture footprints
        has_serials = bool(exif.get("BodySerialNumber") or exif.get("LensSerialNumber"))
        has_file_source = bool(exif.get("FileSource") or exif.get("SceneType"))
        subsec = str(exif.get("SubSecTimeOriginal", exif.get("SubSecTimeDigitized", "")))
        has_subsec = bool(subsec and subsec.isdigit() and subsec not in ["000", "000000"])
        has_gps = bool(exif.get("GPSLatitude") or exif.get("GPSLongitude") or exif.get("GPSDateStamp") or exif.get("GPSTimeStamp"))
        has_sensor_cal = bool(exif.get("CFAPattern") or exif.get("SensingMethod") or exif.get("FocalPlaneXResolution") or exif.get("FocalPlaneYResolution"))

        # MakerNote validation (from our extractor)
        has_valid_makernote = False
        if exif.get("HasMakerNote"):
            try:
                ln = int(exif.get("MakerNoteLength", 0))
                ent = float(exif.get("MakerNoteEntropy", 0.0))
                has_valid_makernote = (ln >= 256 and 4.0 <= ent <= 7.95)
            except Exception:
                has_valid_makernote = False

        corroborated = any([
            has_valid_makernote,
            has_serials,
            has_file_source,
            has_subsec,
            has_gps,
            has_sensor_cal,
        ])

        if corroborated:
            score += ScoringConfig.ORIGINAL.get("JPEG_QTABLE_CAMERA_LIKE", 0.10)
            signals.append("JPEG quantization tables look camera-specific (corroborated)")

    # DCT coefficient statistics (block-sampled): natural images have non-trivial mid/high energy
    ratio = exif.get("DCTMidHighRatio")
    if exif.get("HasDCTStats") and ratio is not None:
        try:
            r = float(ratio)
            # Only use as human-evidence when we have some provenance (avoid helping AI w/ no metadata)
            if (has_camera_provenance or is_trusted_make) and r > 0.30:
                score += ScoringConfig.ORIGINAL.get("NATURAL_DCT_DECAY", 0.10)
                signals.append("Natural DCT energy decay (not over-smoothed)")
        except Exception:
            pass

    # MakerNote structure: length + entropy (hard to fake convincingly)
    if exif.get("HasMakerNote"):
        try:
            ln = int(exif.get("MakerNoteLength", 0))
            ent = float(exif.get("MakerNoteEntropy", 0.0))
            # Typical MakerNotes are non-trivial blobs with mid-high entropy.
            if ln >= 256 and 4.0 <= ent <= 7.95:
                score += ScoringConfig.ORIGINAL.get("VALID_MAKERNOTE", 0.12)
                signals.append("Valid MakerNote structure")
        except Exception:
            pass

    # --- [NEW] Tier 8: Sensor Physics ---
    # SensingMethod: 2 = One-chip color area sensor (Standard Digital Camera)
    if to_float(exif.get("SensingMethod")) == 2:
        score += ScoringConfig.ORIGINAL["SENSOR_DATA"]
        signals.append("Authenticated sensor sensing method (Digital Camera)")
    
    # FocalPlaneResolution: Pro cameras use this
    if "FocalPlaneXResolution" in exif or "FocalPlaneYResolution" in exif:
        score += ScoringConfig.ORIGINAL["SENSOR_DATA"]
        signals.append("High-fidelity focal plane resolution calibration")

    # --- [NEW] Tier 9: Image Origin ---
    # FileSource: 3 = Digital Camera
    file_src = exif.get("FileSource")
    # FileSource is often bytes b'\x03'
    if file_src == b'\x03' or str(file_src) == '\x03' or to_float(file_src) == 3:
         score += ScoringConfig.ORIGINAL["FILE_SOURCE"]
         signals.append("Digital camera file source verified")
    
    # SceneType: 1 = Directly photographed
    scene_type = exif.get("SceneType")
    if scene_type == b'\x01' or str(scene_type) == '\x01' or to_float(scene_type) == 1:
        score += ScoringConfig.ORIGINAL["FILE_SOURCE"]
        signals.append("Directly photographed scene type (Non-synthetic)")

    # --- [NEW] Tier 10: Low-Level Hardware ---
    if "CFAPattern" in exif:
        score += ScoringConfig.ORIGINAL["CFA_PATTERN"]
        signals.append("Color Filter Array (CFA) pattern fingerprint")

    return round(score, 2), signals

def get_ai_suspicion_score(exif: dict, width: int = 0, height: int = 0, file_size: int = 0, filename: str = "") -> tuple:
    """
    Weighted AI suspicion score based on blatant signatures, missing camera metadata,
    and image characteristics.
    """
    score = 0.0
    signals = []
    
    # -1. Filename Keywords (Money Saving Logic)
    if filename:
        import re
        fn_lower = filename.lower()
        
        # High confidence Hard AI suffixes/prefixes
        # NOTE: Avoid stock-photo prefixes like "1000_F_" (common on real images).
        hard_ai_filename_keywords = ["midjourney", "dalle", "lall-e", "lensa", "fotor", "ai-art", "aiart", "generated-ai", "generative-ai", "stability-ai"]
        if any(k in fn_lower for k in hard_ai_filename_keywords):
             score += ScoringConfig.AI["HARD_GENERATOR"] # 1.0 -> Immediate bypass
             signals.append(f"Hard AI signature in filename: {filename}")
        else:
            # Low confidence: match as standalone words or specific patterns
            # NOTE: Do NOT include standalone "ai" (it appears in many real filenames like "ai-technology-free-photo").
            ai_file_keywords = ["stable", "diffusion", "flux", "sora", "kling", "firefly", "generation", "generated", "artwork"]
            # Look for words separated by -, _, . or numbers
            pattern = r"(?:^|[\W_])(" + "|".join(ai_file_keywords) + r")(?:[\W_]|$)"
            match = re.search(pattern, fn_lower)
            if match:
                score += ScoringConfig.AI["KEYWORD_FILENAME"]
                signals.append(f"Potential AI keyword in filename: {match.group(1)}")
    
    has_camera_info = exif.get("Make") or exif.get("Model")
    
    # 0. Base Missing Meta
    if not exif:
        score += ScoringConfig.AI["NO_METADATA"]
        signals.append("No metadata found")

    # 1. Hard AI Evidence
    ai_keywords = ["stable", "diffusion", "midjourney", "dalle", "flux", "sora", "kling", "firefly", "generative", "artificial"]
    software = str(exif.get("Software", "")).lower()
    make = str(exif.get("Make", "")).lower()

    # 1.25 Embedded Text Chunks (PNG/JPEG comments) - often survives where EXIF doesn't
    embedded_raw = str(exif.get("EmbeddedText", "")).lower()
    if embedded_raw:
        hard_text_markers = [
            "automatic1111", "comfyui", "invokeai", "stable diffusion", "sdxl",
            "midjourney", "dall-e", "adobe firefly", "firefly", "runway", "pika",
        ]
        if any(k in embedded_raw for k in hard_text_markers):
            score += ScoringConfig.AI["HARD_GENERATOR"]
            signals.append("Hard AI signature found in embedded text")
        else:
            # Stable Diffusion-style parameter blocks (strong but not cryptographic)
            param_markers = [
                "steps:", "sampler", "cfg scale", "seed", "model:", "negative prompt",
                "clip skip", "denoising strength", "size:", "hires", "vae",
            ]
            hits = sum(1 for m in param_markers if m in embedded_raw)
            if hits >= 3:
                score += ScoringConfig.AI.get("AI_TEXT_PARAMS", 0.25)
                signals.append(f"AI workflow parameters found in embedded text ({hits} markers)")
            elif "prompt" in embedded_raw or "negative prompt" in embedded_raw:
                score += ScoringConfig.AI.get("AI_TEXT_PROMPT", 0.10)
                signals.append("Prompt-like embedded text found")
    
    # Hard generator signals (immediate skip if weight allows)
    hard_ai_software = ["stable diffusion", "midjourney", "dall-e", "flux.1", "sora", "kling", "luma ai", "runway gen"]
    if any(k in software for k in hard_ai_software):
        score += ScoringConfig.AI["HARD_GENERATOR"]
        signals.append(f"Hard AI software signature: {software}")
    elif any(k in software for k in ai_keywords):
        score += ScoringConfig.AI["KEYWORD_SOFTWARE"]
        signals.append(f"AI software signature: {software}")
    elif any(k in make for k in ai_keywords):
        score += ScoringConfig.AI["KEYWORD_MAKE"]
        signals.append(f"AI manufacturer signature: {make}")

    # 1.5 XMP/IPTC High-Confidence Markers (Money Saving Layer)
    xmp_raw = exif.get("XMP", "").lower()
    if xmp_raw:
        if "trainedalgorithmicmedia" in xmp_raw:
            score += ScoringConfig.AI["HARD_GENERATOR"]
            signals.append("IPTC AI provenance marker: TrainedAlgorithmicMedia")
        elif "google ai" in xmp_raw:
            score += ScoringConfig.AI["HARD_GENERATOR"]
            signals.append("Hard AI credit signature: Google AI")
        elif "adobe firefly" in xmp_raw:
            score += ScoringConfig.AI["HARD_GENERATOR"]
            signals.append("Hard AI software signature: Adobe Firefly")
        elif "midjourney" in xmp_raw or "dall-e" in xmp_raw:
             score += ScoringConfig.AI["HARD_GENERATOR"]
             signals.append("Hard AI signature found in XMP metadata")

    # 2. Missing Metadata
    if not has_camera_info:
        score += ScoringConfig.AI["MISSING_CAMERA_INFO"]
        signals.append("Missing camera hardware provenance")

    if "DateTimeOriginal" not in exif:
        score += ScoringConfig.AI["MISSING_TIMESTAMP"]
        signals.append("Missing capture timestamp")

    if not exif.get("SubSecTimeOriginal") and not exif.get("SubSecTimeDigitized"):
        score += ScoringConfig.AI["MISSING_SUBSEC"]
        signals.append("Missing high-precision sensor timing")

    if "JPEGInterchangeFormat" not in exif:
        score += ScoringConfig.AI["NON_STD_JPEG"]
        signals.append("Non-standard JPEG segment structure")

    # 2.5 JPEG quantization tables (weak AI hint if generic AND no camera provenance)
    if exif.get("HasJPEGQuantTables") and exif.get("JPEGQuantIsGeneric") and not has_camera_info:
        score += ScoringConfig.AI.get("GENERIC_QTABLE", 0.05)
        signals.append("Generic JPEG quantization tables (libjpeg-like)")

    # 2.6 DCT stats (over-smoothed images often correlate with AI/denoise pipelines)
    if exif.get("HasDCTStats") and not has_camera_info:
        try:
            r = float(exif.get("DCTMidHighRatio", 1.0))
            if r < 0.22:
                score += ScoringConfig.AI.get("OVER_SMOOTH_DCT", 0.08)
                signals.append("Over-smooth DCT profile (low mid/high energy)")
        except Exception:
            pass

    # 2.7 MakerNote empty/broken when other camera fields exist (soft AI hint)
    if has_camera_info and not exif.get("HasMakerNote") and ("BodySerialNumber" in exif or "LensModel" in exif):
        score += ScoringConfig.AI.get("BROKEN_OR_EMPTY_MAKERNOTE", 0.05)
        signals.append("Missing MakerNote despite other camera fields")

    # 3. AI-typical dimensions
    if width > 0 and height > 0 and not has_camera_info:
        # 1536 is common in AI (1.5k) but also iPad screenshots. We treat it as neutral if alone.
        ai_typical_widths = [512, 768, 1024] 
        
        if width in ai_typical_widths:
            score += ScoringConfig.AI["AI_TYPICAL_RES"]
            signals.append(f"AI-typical width: {width}px")
        elif height in ai_typical_widths:
            score += ScoringConfig.AI["AI_TYPICAL_RES"]
            signals.append(f"AI-typical height: {height}px")
        
        total_pixels = width * height
        if total_pixels < 500000 and (width in ai_typical_widths or height in ai_typical_widths):
            score += ScoringConfig.AI["TINY_IMAGE_AI_RES"]
            signals.append(f"Small image ({total_pixels//1000}K pixels) with AI-typical dimensions")
        
        # 4. Non-standard aspect ratio
        aspect = width / height if height > 0 else 0
        standard_aspects = [1.0, 1.33, 1.5, 1.78, 0.75, 0.67, 0.56, 1.0]
        is_standard = any(abs(aspect - std) < 0.08 for std in standard_aspects)
        
        # 4a. SCREENSHOT PROTECTION
        phone_widths = range(640, 1500)
        is_portrait = height > width
        is_phone_width = width in phone_widths
        is_phone_aspect = 0.40 < aspect < 0.60
        
        if is_portrait and is_phone_width and is_phone_aspect:
            score -= 0.20
            signals.append(f"Likely phone screenshot ({width}x{height}) - forcing GPU analysis")
        if abs(aspect - 1.0) < 0.02:
            score += ScoringConfig.AI["SQUARE_RES"]
            signals.append("Perfectly square aspect ratio (AI typical)")
        elif not is_standard and aspect > 0:
            score += ScoringConfig.AI["NON_STD_ASPECT"]
            signals.append(f"Non-standard aspect ratio: {aspect:.2f}")

    # 6. File size analysis
    if width > 0 and height > 0 and file_size > 0 and not has_camera_info:
        pixels = width * height
        bytes_per_pixel = file_size / pixels if pixels > 0 else 0
        if bytes_per_pixel < 0.15 and pixels > 500000:
            score += ScoringConfig.AI["LOW_BPP"]
            signals.append(f"Low bytes/pixel: {bytes_per_pixel:.2f} (heavily compressed)")
        elif bytes_per_pixel < 0.3 and pixels > 500000:
            score += 0.05
            signals.append(f"Compressed image: {bytes_per_pixel:.2f} bytes/pixel")

    # logger.debug(f"[DEBUG] get_ai_suspicion_score({filename}): {score}")
    return round(score, 2), signals
