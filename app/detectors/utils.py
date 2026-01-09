
import logging
import hashlib
import numpy as np
from collections import OrderedDict
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Union
from app.security import security_manager

logger = logging.getLogger(__name__)

# Standard libjpeg quantization tables (baseline) - used to detect generic re-encodes.
# These are widely reused by "default" pipelines.
_STD_LUMA_QTABLE = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

_STD_CHROMA_QTABLE = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]

def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of bytes (0..8)."""
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    p = hist / float(arr.size)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _safe_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0

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
        # For PIL Images: hash raw pixel bytes (no JPEG artifacts)
        thumb = source.copy()
        # Use 32x32 for video frames (fast_mode), 64x64 for standalone images
        size = (32, 32) if fast_mode else (64, 64)
        thumb.thumbnail(size)
        thumb = thumb.convert("L")  # Grayscale reduces data while preserving uniqueness
        # Hash raw pixel bytes directly (faster, no compression artifacts)
        return security_manager.get_safe_hash(np.array(thumb).tobytes())

def get_exif_data(file_path: str) -> dict:
    """Extract EXIF metadata from the image. Explicitly closed via 'with'."""
    try:
        with Image.open(file_path) as img:
            exif = img._getexif() or {}
            exif_data = {}
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
            
            # Capture ICC Profile presence (pro-grade signal)
            if img.info.get("icc_profile"):
                exif_data["HasICCProfile"] = True
            
            # Capture XMP/IPTC Data (Modern AI Markers)
            xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp")
            if xmp:
                if isinstance(xmp, bytes):
                    exif_data["XMP"] = xmp.decode("utf-8", errors="ignore")
                else:
                    exif_data["XMP"] = str(xmp)

            # Capture embedded text chunks (PNG tEXt/iTXt/zTXt, JPEG comment blocks, etc.)
            # This is high-signal for AI workflows (e.g., "Steps:", "Sampler", "ComfyUI", etc.)
            embedded_parts = []
            embedded_keys = []

            def _add_embedded_text(key, value):
                try:
                    if value is None:
                        return
                    if isinstance(value, bytes):
                        s = value.decode("utf-8", errors="ignore")
                    else:
                        s = str(value)
                    s = s.strip()
                    if not s:
                        return
                    # Avoid huge payloads (some tools dump large JSON blobs)
                    if len(s) > 50000:
                        s = s[:50000]
                    embedded_keys.append(str(key))
                    embedded_parts.append(s)
                except Exception:
                    return

            # Common locations (works across formats depending on decoder)
            info = img.info or {}
            for k, v in info.items():
                kl = str(k).lower()
                if kl in {
                    "parameters", "prompt", "workflow", "comment", "description", "software",
                    "generator", "model", "negative_prompt", "negative prompt",
                    "exif", "usercomment"
                }:
                    _add_embedded_text(k, v)

            # PNG-specific text dict (Pillow exposes for PNG images)
            text_dict = getattr(img, "text", None)
            if isinstance(text_dict, dict):
                for k, v in text_dict.items():
                    _add_embedded_text(k, v)

            if embedded_parts:
                exif_data["HasEmbeddedText"] = True
                exif_data["EmbeddedTextKeys"] = embedded_keys[:25]
                exif_data["EmbeddedText"] = "\n".join(embedded_parts)

            # --- JPEG metadata-adjacent forensic signals (CPU-only, very cheap) ---
            try:
                if (img.format or "").upper() == "JPEG":
                    # 1) Embedded EXIF thumbnail presence (very rare in AI exports)
                    if "JPEGInterchangeFormat" in exif_data:
                        exif_data["HasEmbeddedThumbnail"] = True

                    # 2) Quantization tables (JPEG only)
                    q = getattr(img, "quantization", None)
                    if isinstance(q, dict) and q:
                        q0 = q.get(0) or q.get("0")
                        q1 = q.get(1) or q.get("1")
                        if isinstance(q0, (list, tuple)) and len(q0) >= 64:
                            q0_64 = [int(x) for x in q0[:64]]
                        else:
                            q0_64 = []
                        if isinstance(q1, (list, tuple)) and len(q1) >= 64:
                            q1_64 = [int(x) for x in q1[:64]]
                        else:
                            q1_64 = []

                        if q0_64:
                            diff_l = float(np.mean(np.abs(np.array(q0_64, dtype=np.float32) - np.array(_STD_LUMA_QTABLE, dtype=np.float32))))
                        else:
                            diff_l = 999.0
                        if q1_64:
                            diff_c = float(np.mean(np.abs(np.array(q1_64, dtype=np.float32) - np.array(_STD_CHROMA_QTABLE, dtype=np.float32))))
                        else:
                            diff_c = 999.0

                        exif_data["HasJPEGQuantTables"] = True
                        exif_data["JPEGQuantDiffLumaToStd"] = round(diff_l, 3)
                        exif_data["JPEGQuantDiffChromaToStd"] = round(diff_c, 3)
                        # "Generic" if very close to standard libjpeg tables (common in re-encodes and many AI exports)
                        exif_data["JPEGQuantIsGeneric"] = (diff_l < 3.0 and diff_c < 3.0)

                    # 3) DCT coefficient statistics (fast, block-sampled)
                    # Not GPU / not deep vision; still a container-adjacent forensic signal.
                    try:
                        import cv2  # opencv is already used in this repo
                        # Downsample aggressively for speed
                        gray = img.convert("L")
                        gray.thumbnail((256, 256))
                        arr = np.array(gray, dtype=np.float32)
                        h, w = arr.shape[:2]
                        if h >= 16 and w >= 16:
                            # Sample blocks on a coarse grid (deterministic)
                            step = max(8, min(h, w) // 8)
                            low_energy = []
                            midhigh_energy = []
                            for y in range(0, h - 8, step):
                                for x in range(0, w - 8, step):
                                    block = arr[y:y+8, x:x+8]
                                    if block.shape != (8, 8):
                                        continue
                                    # center & scale
                                    b = block - 128.0
                                    dct = cv2.dct(b)
                                    # Low freq (DC + first neighbors)
                                    low = float(np.sum(np.abs(dct[0:2, 0:2])))
                                    midhigh = float(np.sum(np.abs(dct[2:, 2:])))
                                    low_energy.append(low)
                                    midhigh_energy.append(midhigh)

                            if low_energy and midhigh_energy:
                                low_m = float(np.mean(low_energy))
                                midhigh_m = float(np.mean(midhigh_energy))
                                ratio = midhigh_m / (low_m + 1e-6)
                                exif_data["HasDCTStats"] = True
                                exif_data["DCTMidHighRatio"] = round(ratio, 4)
                    except Exception:
                        pass
            except Exception:
                pass

            # --- MakerNote structure (when present) ---
            try:
                maker = exif_data.get("MakerNote")
                if isinstance(maker, (bytes, bytearray)):
                    maker_b = bytes(maker)
                    exif_data["MakerNoteLength"] = int(len(maker_b))
                    exif_data["MakerNoteEntropy"] = round(_byte_entropy(maker_b), 3)
                    exif_data["HasMakerNote"] = True
            except Exception:
                pass

            return exif_data
    except Exception:
        return {}
