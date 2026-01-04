import runpod
import base64
import io
import torch
import logging
import numpy as np
import hashlib
from collections import OrderedDict
from PIL import Image
from transformers import pipeline

# Set up logging for RunPod
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Initialization ---
device = 0 if torch.cuda.is_available() else -1
logger.info(f"Initializing worker on device: {'GPU' if device == 0 else 'CPU'}")

detector = None

# Worker-side cache to avoid redundant GPU calls
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

try:
    logger.info("Loading SigLIP2 model (Ateeqq/ai-vs-human-image-detector-2)...")
    detector = pipeline(
        "image-classification", 
        model="Ateeqq/ai-vs-human-image-detector-2",
        device=device,
        torch_dtype=torch.float16,        # Optimization from SigLIP2 docs
        model_kwargs={"attn_implementation": "sdpa"}, # Faster attention
        trust_remote_code=True
    )
    logger.info("SigLIP2 model loaded successfully!")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load SigLIP2: {e}", exc_info=True)
    detector = None

def get_cpu_fft_score(img: Image.Image) -> float:
    """CPU-only FFT check to identify natural vs artificial patterns."""
    try:
        # Convert PIL to grayscale numpy
        gray_img = np.array(img.convert("L"))
        
        # FFT Math (CPU only)
        dft = np.fft.fft2(gray_img)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
        mean_val = np.mean(magnitude_spectrum)
        
        # High strictness peak detection
        peaks = np.sum(magnitude_spectrum > (mean_val * 2.0))
        score = min(peaks / 10000, 1.0)
        return float(score)
    except Exception as e:
        logger.error(f"Worker FFT error: {e}")
        return 0.5 # Neutral if error

def handler(job):
    """
    The main RunPod task handler with heuristics and caching.
    """
    job_input = job["input"]
    task = job_input.get("task")
    
    if task == "deep_forensic":
        if detector is None:
            return {"error": "Model failed to initialize on worker."}

        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image data provided"}
        
        # Metadata for heuristics
        orig_w = job_input.get("orig_w", 0)
        orig_h = job_input.get("orig_h", 0)
        
        # Cache Check
        img_hash = hashlib.md5(image_base64.encode()).hexdigest()
        cached = worker_cache.get(img_hash)
        if cached:
            logger.info("Result found in worker cache.")
            return {"ai_score": cached}

        try:
            # Decode base64 to image
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # --- 1. Consistent Resizing ---
            # Standardize for inference
            inference_img = img.resize((512, 512), Image.LANCZOS)
            
            # --- 2. GPU Inference ---
            results = detector(inference_img)
            ai_score = 0.0
            for res in results:
                if res['label'].lower() == 'ai':
                    ai_score = float(res['score'])
            
            # --- 3. CPU-only Heuristics ---
            
            # A. FFT Heuristic Adjustment
            fft_score = get_cpu_fft_score(inference_img)
            # If FFT is very low (natural textures), slightly lower AI score
            if fft_score < 0.3:
                ai_score *= 0.9 # Small "Benefit of the doubt" bonus
            
            # B. High-Resolution Heuristic (Missing C2PA is assumed if worker is called)
            if orig_w * orig_h > 2_000_000: # > 2MP
                logger.info(f"Applying high-res human bias penalty: {orig_w}x{orig_h}")
                ai_score *= 0.7 # Reduce confidence by 30%
            
            # Final capping
            final_score = max(0.0, min(1.0, ai_score))
            
            # Cache the result
            worker_cache.put(img_hash, final_score)
            
            return {"ai_score": final_score}
        except Exception as e:
            logger.error(f"Worker process error: {e}", exc_info=True)
            return {"error": f"Internal scan error: {str(e)}"}

    return {"error": f"Invalid task: {task}"}

# Start the RunPod serverless loop
runpod.serverless.start({"handler": handler})
