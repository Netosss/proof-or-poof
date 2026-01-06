import runpod
import base64
import io
import time
import torch
import logging
import hashlib
import numpy as np
from collections import OrderedDict
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------- Optimization Flags ----------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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
    
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
    
    if device == "cuda":
        model = model.to("cuda", memory_format=torch.channels_last)
    
    id2label = {idx: label.lower() for idx, label in model.config.id2label.items()}
    ai_label_ids = [
        idx for idx, label in id2label.items()
        if any(x in label for x in ("ai", "fake", "generated", "artificial"))
    ]
    logger.info(f"Detected AI labels at indices: {ai_label_ids}")

    def safe_to_fp16(tensor):
        if tensor.dtype == torch.float32:
            return tensor.to(device="cuda", dtype=torch.float16, non_blocking=True)
        return tensor.to(device="cuda", non_blocking=True)

    # GPU Warm-up with batch dimension
    if device == "cuda":
        logger.info("Warming up GPU (batch pipeline)...")
        dummy_imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        inputs = processor(images=dummy_imgs, return_tensors="pt")
        inputs = {k: safe_to_fp16(v) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.inference_mode():
            _ = model(**inputs)
        torch.cuda.synchronize()
            
    logger.info("Model and Processor loaded successfully!")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load model: {e}", exc_info=True)
    model = None
    processor = None

# ---------------- Utilities ----------------
def get_cpu_fft_score(img: Image.Image) -> float:
    """Optimized CPU FFT check."""
    try:
        fft_img = img.resize((256, 256), Image.BILINEAR)
        gray_img = np.array(fft_img.convert("L"))
        dft = np.fft.fft2(gray_img)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
        mean_val = np.mean(magnitude_spectrum)
        peaks = np.sum(magnitude_spectrum > (mean_val * 2.0))
        return min(peaks / 10000, 1.0)
    except:
        return 0.5

@torch.inference_mode()
def launch_gpu_batch(images: list):
    """Launch batch inference for a list of PIL images."""
    if not images or model is None or processor is None:
        return None
    inputs = processor(images=images, return_tensors="pt")
    if device == "cuda":
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(memory_format=torch.channels_last)
        inputs = {k: safe_to_fp16(v) if torch.is_tensor(v) else v for k, v in inputs.items()}
    return model(**inputs).logits

# ---------------- Main Handler (Batch Support) ----------------
def handler(job):
    job_input = job.get("input", {})
    task = job_input.get("task")
    
    if task != "deep_forensic":
        return {"error": f"Invalid task: {task}"}
    
    if model is None:
        return {"error": "Model failed to initialize on worker."}

    # Normalize input to list (supports both single and batch)
    images_b64 = []
    is_batch = False
    if "images" in job_input and isinstance(job_input["images"], list):
        images_b64 = job_input["images"]
        is_batch = True
    elif "image" in job_input:
        images_b64 = [job_input["image"]]
    else:
        return {"error": "No image data provided"}

    total_start = time.perf_counter()
    
    # Results array (maintain order)
    results = [None] * len(images_b64)
    
    # Images that need GPU processing (cache misses)
    images_to_process = []  # (original_index, PIL_Image, width, height, metadata_bias)
    hashes_to_process = []
    
    # 1. DECODE & CACHE CHECK
    t0 = time.perf_counter()
    for idx, b64_str in enumerate(images_b64):
        try:
            img_bytes = base64.b64decode(b64_str)
            img_hash = hashlib.md5(img_bytes).hexdigest()
            
            # Check cache
            cached = worker_cache.get(img_hash)
            if cached:
                cached_result = cached.copy()
                cached_result["cache_hit"] = True
                results[idx] = cached_result
                continue
            
            # Decode for processing
            original_img = Image.open(io.BytesIO(img_bytes))
            orig_w, orig_h = original_img.size
            
            # Extract EXIF before conversion
            metadata_bias = 1.0
            try:
                exif = original_img.getexif()
                if exif:
                    software = str(exif.get(305, "")).lower()
                    make = str(exif.get(271, "")).lower()
                    if any(s in software for s in ["photoshop", "lightroom", "capture one", "gimp"]):
                        metadata_bias *= 0.9
                    if any(m in make for m in ["canon", "nikon", "sony", "fujifilm", "leica", "apple", "google"]):
                        metadata_bias *= 0.85
            except:
                pass
            
            # Convert for model
            img = original_img.convert("RGB")
            
            images_to_process.append((idx, img, orig_w, orig_h, metadata_bias))
            hashes_to_process.append(img_hash)
            
        except Exception as e:
            results[idx] = {"error": str(e), "ai_score": 0.0}
    
    decode_ms = (time.perf_counter() - t0) * 1000
    cache_hits = len(images_b64) - len(images_to_process)
    logger.info(f"[TIMING] Decode & cache check: {decode_ms:.2f}ms ({cache_hits} hits, {len(images_to_process)} misses)")
    
    # 2. PROCESS CACHE MISSES (Batch GPU)
    if images_to_process:
        pil_images = [x[1] for x in images_to_process]
        
        # 2A. Launch GPU Batch (non-blocking dispatch)
        t1 = time.perf_counter()
        logits_batch = launch_gpu_batch(pil_images)
        
        # 2B. CPU Heuristics (FFT) - runs while GPU is busy
        fft_data = []
        for _, img, w, h, _ in images_to_process:
            fft_raw = get_cpu_fft_score(img)
            megapixels = (w * h) / 1_000_000
            fft_norm = fft_raw / max(1.0, megapixels)
            
            # High-res bias (capped at 15% penalty)
            hr_bias = 1.0
            if megapixels > 2.0:
                hr_bias = max(0.85, 1.0 - (megapixels / 40.0))
            
            fft_data.append((fft_norm, hr_bias))
        
        fft_ms = (time.perf_counter() - t1) * 1000
        
        # 2C. Gather GPU Results (CUDA sync happens here)
        if logits_batch is not None:
            probs_batch = torch.softmax(logits_batch, dim=-1)
            
            for i, (idx, _, w, h, meta_bias) in enumerate(images_to_process):
                probs = probs_batch[i]
                ai_score = float(probs[ai_label_ids].max())
                ai_score = max(0.0, min(1.0, ai_score))
                
                fft_norm, hr_bias = fft_data[i]
                
                # Weighted combination
                if ai_score > 0.85:
                    final_score = ai_score
                else:
                    final_score = (ai_score * 0.9) + (fft_norm * 0.1)
                
                final_score = max(0.0, min(1.0, final_score * hr_bias * meta_bias))
                
                res_obj = {
                    "ai_score": final_score,
                    "model_score": ai_score,
                    "fft_score": fft_norm,
                    "high_res_bias": hr_bias,
                    "metadata_bias": meta_bias,
                    "image_size": [w, h],
                    "cache_hit": False
                }
                
                # Cache result
                worker_cache.put(hashes_to_process[i], res_obj)
                results[idx] = res_obj
        
        gpu_ms = (time.perf_counter() - t1) * 1000
        logger.info(f"[TIMING] Batch GPU + FFT: {gpu_ms:.2f}ms for {len(images_to_process)} images")
    
    total_ms = (time.perf_counter() - total_start) * 1000
    logger.info(f"[TIMING] Total: {total_ms:.2f}ms for {len(images_b64)} images ({cache_hits} cached)")
    
    # Add timing to response
    timing = {
        "decode": round(decode_ms, 2),
        "total": round(total_ms, 2),
        "batch_size": len(images_b64),
        "cache_hits": cache_hits
    }
    
    # Return single result or batch
    if is_batch:
        return {"results": results, "timing_ms": timing}
    else:
        # Single image: return flat result with timing
        single_result = results[0] if results else {"error": "No result"}
        if isinstance(single_result, dict):
            single_result["timing_ms"] = timing
        return single_result

# ---------------- Start RunPod Loop ----------------
runpod.serverless.start({"handler": handler})
