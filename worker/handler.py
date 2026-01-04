import runpod
import base64
import io
import torch
import logging
import hashlib
import numpy as np
from collections import OrderedDict
from PIL import Image, ExifTags
from transformers import pipeline

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Device ----------------
device = 0 if torch.cuda.is_available() else -1
logger.info(f"Initializing worker on device: {'GPU' if device == 0 else 'CPU'}")

# ---------------- Worker Cache ----------------
class WorkerLRUCache:
    def __init__(self, capacity: int = 500):
        self.cache = OrderedDict()
        self.capacity = capacity
    def get(self, key):
        if key not in self.cache: return None
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity: self.cache.popitem(last=False)

worker_cache = WorkerLRUCache(capacity=500)

# ---------------- Load SigLIP2 ----------------
detector = None
try:
    logger.info("Loading SigLIP2 NaFlex model (google/siglip2-base-patch16-naflex)...")
    detector = pipeline(
        task="zero-shot-image-classification",
        model="google/siglip2-base-patch16-naflex",
        device=device,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    logger.info("SigLIP2 NaFlex loaded successfully!")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load SigLIP2: {e}", exc_info=True)
    detector = None

# ---------------- FFT Heuristic ----------------
def get_cpu_fft_score(img: Image.Image) -> float:
    """Lightweight CPU FFT check to help identify real vs AI patterns"""
    try:
        gray_img = np.array(img.convert("L"))
        dft = np.fft.fft2(gray_img)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
        mean_val = np.mean(magnitude_spectrum)
        peaks = np.sum(magnitude_spectrum > (mean_val * 2.0))
        return min(peaks / 10000, 1.0)
    except Exception as e:
        logger.error(f"FFT error: {e}")
        return 0.5  # Neutral if error

# ---------------- Handler ----------------
def handler(job):
    job_input = job.get("input", {})
    task = job_input.get("task")

    if task != "deep_forensic":
        return {"error": f"Invalid task: {task}"}

    if detector is None:
        return {"error": "Model failed to initialize on worker."}

    image_base64 = job_input.get("image")
    if not image_base64:
        return {"error": "No image data provided"}

    # Cache check
    img_bytes = base64.b64decode(image_base64)
    img_hash = hashlib.md5(img_bytes).hexdigest()
    cached = worker_cache.get(img_hash)
    if cached:
        logger.info("Result retrieved from cache.")
        return cached

    try:
        # Decode and load image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        # ---------------- Heuristics ----------------
        # 1. FFT Score with Resolution Normalization
        fft_score = get_cpu_fft_score(img)
        # Normalize peaks by megapixels to avoid high-res bias
        megapixels = (orig_w * orig_h) / 1_000_000
        # If image is very small, megapixels might be < 1, so we floor it at 1.0 for the divisor
        normalized_fft_score = fft_score / max(1.0, megapixels)

        # 2. Dynamic High-resolution Bias (scaling from 0.5 to 1.0)
        high_res_bias = 1.0
        if megapixels > 2.0:
            # Gradually reduce AI confidence for high-res images (max reduction 50%)
            high_res_bias = max(0.5, 1.0 - (megapixels / 20.0))

        # 3. EXIF/Software Metadata Bias (Fast CPU check)
        metadata_bias = 1.0
        try:
            exif_data = img._getexif() or {}
            # Tag 305 is Software, Tag 271 is Make (Camera Manufacturer)
            software = str(exif_data.get(305, "")).lower()
            make = str(exif_data.get(271, "")).lower()
            
            # If it has professional software or a known camera make, give it a human bonus
            if any(s in software for s in ["photoshop", "lightroom", "capture one", "gimp"]):
                metadata_bias *= 0.9
            if any(m in make for m in ["canon", "nikon", "sony", "fujifilm", "leica", "apple", "google"]):
                metadata_bias *= 0.85 # Stronger human signal from a real camera
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

        # ---------------- SigLIP2 ----------------
        # Using descriptive labels from the documentation for better zero-shot accuracy
        candidate_labels = [
            "a natural real-world photograph with camera noise",
            "a synthetic AI-generated image with smooth generative textures"
        ]
        results = detector(img, candidate_labels=candidate_labels)
        
        # Parse results correctly
        ai_score = 0.0
        if isinstance(results, dict) and "labels" in results and "scores" in results:
            for label, score in zip(results["labels"], results["scores"]):
                if "synthetic" in label.lower():
                    ai_score = float(score)
        elif isinstance(results, list):
            for res in results:
                if "synthetic" in res.get("label", "").lower():
                    ai_score = float(res.get("score", 0.0))

        # ---------------- Weighted Combination ----------------
        # Consensus: SigLIP2 (80%) + Normalized FFT (20%)
        final_score = (ai_score * 0.8) + (normalized_fft_score * 0.2)
        
        # Apply the cumulative human biases
        final_score *= high_res_bias
        final_score *= metadata_bias
        
        final_score = max(0.0, min(1.0, final_score))

        # ---------------- Cache & Return ----------------
        result = {
            "ai_score": final_score,
            "siglip2_score": ai_score,
            "fft_score": normalized_fft_score,
            "high_res_bias": high_res_bias,
            "metadata_bias": metadata_bias,
            "image_size": [orig_w, orig_h]
        }
        worker_cache.put(img_hash, result)
        return result

    except Exception as e:
        logger.error(f"Worker processing error: {e}", exc_info=True)
        return {"error": f"Internal scan error: {str(e)}"}

# ---------------- Start RunPod Loop ----------------
runpod.serverless.start({"handler": handler})
