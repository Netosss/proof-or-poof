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
        fft_score = get_cpu_fft_score(img)

        # High-resolution bias
        high_res_bias = 1.0
        if orig_w * orig_h > 2_000_000:  # >2MP
            high_res_bias = 0.7  # Reduce AI confidence by 30%

        # ---------------- SigLIP2 ----------------
        candidate_labels = ["real photo", "AI generated image"]
        results = detector(img, candidate_labels=candidate_labels)
        
        # Parse results correctly
        ai_score = 0.0
        if isinstance(results, dict) and "labels" in results and "scores" in results:
            for label, score in zip(results["labels"], results["scores"]):
                if "ai" in label.lower() or "synthetic" in label.lower():
                    ai_score = float(score)
        elif isinstance(results, list):  # fallback for list output
            for res in results:
                if "ai" in res.get("label", "").lower() or "synthetic" in res.get("label", "").lower():
                    ai_score = float(res.get("score", 0.0))

        # ---------------- Weighted Combination ----------------
        # Simple weighted average: SigLIP2 70%, FFT 20%, high-res bias 10%
        final_score = ai_score * 0.7 + (1 - fft_score) * 0.2
        final_score *= high_res_bias
        final_score = max(0.0, min(1.0, final_score))

        # ---------------- Cache & Return ----------------
        result = {
            "ai_score": final_score,
            "siglip2_score": ai_score,
            "fft_score": fft_score,
            "high_res_bias": high_res_bias,
            "image_size": [orig_w, orig_h]
        }
        worker_cache.put(img_hash, result)
        return result

    except Exception as e:
        logger.error(f"Worker processing error: {e}", exc_info=True)
        return {"error": f"Internal scan error: {str(e)}"}

# ---------------- Start RunPod Loop ----------------
runpod.serverless.start({"handler": handler})
