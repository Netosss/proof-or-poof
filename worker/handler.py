import runpod
import base64
import io
import torch
import logging
import hashlib
import numpy as np
from collections import OrderedDict
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------- Optimization Flags ----------------
# Enable TF32 for significantly faster matmuls on Ampere+ GPUs (Safe for classification)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Initializing worker on device: {device}")

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

# ---------------- Model Initialization ----------------
try:
    MODEL_ID = "haywoodsloan/ai-image-detector-dev-deploy"
    logger.info(f"Loading Model and Processor: {MODEL_ID}")
    
    # 1. Load Processor
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True
    )
    
    # 2. Load Model directly in FP16 (Fast & Memory Efficient)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
    
    if device == "cuda":
        model = model.to("cuda")
    
    # 3. Pre-parse Label IDs for AI classes (Remove string logic from hot path)
    id2label = {
        idx: label.lower()
        for idx, label in model.config.id2label.items()
    }
    ai_label_ids = [
        idx for idx, label in id2label.items()
        if any(x in label for x in ("ai", "fake", "generated", "artificial"))
    ]
    logger.info(f"Detected AI labels at indices: {ai_label_ids}")

    # 4. GPU Warm-up (Important for first-request latency)
    if device == "cuda":
        logger.info("Warming up GPU...")
        dummy = torch.zeros(1, 3, 224, 224, device="cuda", dtype=torch.float16)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            model(pixel_values=dummy)
            
    logger.info("Model and Processor loaded successfully!")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load model: {e}", exc_info=True)
    model = None
    processor = None

# ---------------- Optimized FFT Heuristic ----------------
def get_cpu_fft_score(img: Image.Image) -> float:
    """Optimized CPU FFT check: Resizes to 256x256 first for consistent speed."""
    try:
        # Downscale for 4-6x faster FFT - patterns survive downscaling
        fft_img = img.resize((256, 256), Image.BILINEAR)
        gray_img = np.array(fft_img.convert("L"))
        
        dft = np.fft.fft2(gray_img)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
        mean_val = np.mean(magnitude_spectrum)
        peaks = np.sum(magnitude_spectrum > (mean_val * 2.0))
        return min(peaks / 10000, 1.0)
    except Exception as e:
        logger.error(f"FFT error: {e}")
        return 0.5

# ---------------- Inference Wrapper ----------------
@torch.inference_mode()
def run_deep_scan(img: Image.Image, debug: bool = False):
    """Direct model call using FP16 and Autocast for maximum GPU throughput."""
    if model is None or processor is None:
        return None, None
    
    # Let processor handle standard resizing and normalization
    inputs = processor(images=img, return_tensors="pt")
    
    if device == "cuda":
        inputs = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}
        # Use Autocast for mixed precision inference (Tensor Core usage)
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

    # Get probabilities
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    # Extract AI score from pre-parsed indices
    ai_score = float(probs[ai_label_ids].max())
    
    # Construct raw results for API compatibility (Only if debug is requested)
    raw_results = None
    if debug:
        raw_results = []
        for idx, prob in enumerate(probs):
            raw_results.append({"label": id2label[idx], "score": float(prob)})
        
    return ai_score, raw_results

# ---------------- Handler ----------------
def handler(job):
    job_input = job.get("input", {})
    task = job_input.get("task")

    if task != "deep_forensic":
        return {"error": f"Invalid task: {task}"}

    if model is None:
        return {"error": "Model failed to initialize on worker."}

    image_base64 = job_input.get("image")
    if not image_base64:
        return {"error": "No image data provided"}

    # Cache check
    try:
        img_bytes = base64.b64decode(image_base64)
        img_hash = hashlib.md5(img_bytes).hexdigest()
    except Exception as e:
        return {"error": f"Invalid base64: {str(e)}"}

    cached = worker_cache.get(img_hash)
    if cached:
        logger.info("Result retrieved from cache.")
        return cached

    try:
        # Decode and load image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        # 1. FFT Score (Now much faster due to internal resize)
        fft_score = get_cpu_fft_score(img)
        megapixels = (orig_w * orig_h) / 1_000_000
        normalized_fft_score = fft_score / max(1.0, megapixels)

        # 2. Dynamic High-resolution Bias
        high_res_bias = 1.0
        if megapixels > 2.0:
            high_res_bias = max(0.5, 1.0 - (megapixels / 20.0))

        # 3. EXIF/Software Metadata Bias
        metadata_bias = 1.0
        try:
            exif = img.getexif()
            if exif:
                software = str(exif.get(305, "")).lower()
                make = str(exif.get(271, "")).lower()
                if any(s in software for s in ["photoshop", "lightroom", "capture one", "gimp"]):
                    metadata_bias *= 0.9
                if any(m in make for m in ["canon", "nikon", "sony", "fujifilm", "leica", "apple", "google"]):
                    metadata_bias *= 0.85
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

        # 4. Optimized Model Inference
        debug_mode = job_input.get("debug", False)
        ai_score, raw_results = run_deep_scan(img, debug=debug_mode)
        if ai_score is None:
            return {"error": "Inference failed"}

        # 4.5 NEW: Digital UI / Screen Record Detection
        # Check for "Flatness" - real photos and AI photos have noise. 
        # Screen recordings have perfectly flat digital blocks.
        is_ui_recording = False
        try:
            # Convert to small array to check for identical neighbor pixels (flatness)
            small_gray = np.array(img.resize((32, 32)).convert("L"))
            unique_colors = len(np.unique(small_gray))
            # If a 32x32 block has relatively few unique colors, it's likely a UI/Vector graphic
            # Real photos and complex AI textures typically have > 250 unique colors in 32x32
            if unique_colors < 200: 
                is_ui_recording = True
                logger.info(f"UI/Screen Record detected (Unique colors: {unique_colors}). Applying Human bonus.")
        except:
            pass

        # 5. Weighted Combination
        # Reverted FFT weight to 10% (0.1) as requested.
        final_score = (ai_score * 0.9) + (normalized_fft_score * 0.1)
        
        if is_ui_recording:
            # Apply a strong reduction for confirmed digital UI content
            final_score *= 0.4 
        
        final_score *= high_res_bias
        final_score *= metadata_bias
        final_score = max(0.0, min(1.0, final_score))

        result = {
            "ai_score": final_score,
            "model_score": ai_score,
            "fft_score": normalized_fft_score,
            "high_res_bias": high_res_bias,
            "metadata_bias": metadata_bias,
            "image_size": [orig_w, orig_h],
            "raw_results": raw_results
        }
        worker_cache.put(img_hash, result)
        return result

    except Exception as e:
        logger.error(f"Worker processing error: {e}", exc_info=True)
        return {"error": f"Internal scan error: {str(e)}"}

# ---------------- Start RunPod Loop ----------------
runpod.serverless.start({"handler": handler})
