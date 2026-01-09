import runpod
import base64
import io
import time
import torch
import logging
import hashlib
import numpy as np
import cv2
import concurrent.futures
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

# ---------------- Router Classifier ----------------
class RouterClassifier:
    def __init__(self):
        self.device = device
        self.models_loaded = False
        # Thread pool for inference models (3 models max)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.load_models()

    def load_models(self):
        if self.models_loaded: return
        logger.info("Loading Production Models...")
        try:
            # Load in parallel? Usually fine to load seq, but inference is key.
            # Loading seq to avoid memory spike race conditions during init.
            
            # Model A
            self.processor_a = AutoImageProcessor.from_pretrained("haywoodsloan/ai-image-detector-dev-deploy", use_fast=True)
            self.model_a = AutoModelForImageClassification.from_pretrained(
                "haywoodsloan/ai-image-detector-dev-deploy", torch_dtype=torch.float16
            ).to(self.device).eval()
            if self.device == "cuda": self.model_a = torch.compile(self.model_a, mode="reduce-overhead")

            # Model B
            self.processor_b = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector", use_fast=True)
            self.model_b = AutoModelForImageClassification.from_pretrained(
                "Ateeqq/ai-vs-human-image-detector", torch_dtype=torch.float16
            ).to(self.device).eval()
            if self.device == "cuda": self.model_b = torch.compile(self.model_b, mode="reduce-overhead")

            # Model C
            self.processor_c = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection", use_fast=True)
            self.model_c = AutoModelForImageClassification.from_pretrained(
                "dima806/ai_vs_real_image_detection", torch_dtype=torch.float16
            ).to(self.device).eval()
            if self.device == "cuda": self.model_c = torch.compile(self.model_c, mode="reduce-overhead")
            
            # Warmup
            dummy = Image.new('RGB', (224, 224), color='white')
            # Warm all 3
            f1 = self.executor.submit(self._predict_single, self.model_a, self.processor_a, [dummy])
            f2 = self.executor.submit(self._predict_single, self.model_b, self.processor_b, [dummy])
            f3 = self.executor.submit(self._predict_single, self.model_c, self.processor_c, [dummy])
            concurrent.futures.wait([f1, f2, f3])
            
            self.models_loaded = True
            logger.info("Router Models Loaded Successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _predict_single(self, model, processor, images):
        try:
            inputs = processor(images=images, return_tensors="pt").to(self.device)
            # FP16 Cast check
            if self.device == "cuda" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(memory_format=torch.channels_last)
                if getattr(model, "dtype", torch.float32) == torch.float16:
                    inputs["pixel_values"] = inputs["pixel_values"].half()
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs.cpu().numpy()
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def predict_batch(self, images: list):
        results = [None] * len(images)
        
        low_res_indices = []
        high_res_indices = []
        
        for i, img in enumerate(images):
            w, h = img.size
            if (w * h) < 200000:
                low_res_indices.append(i)
            else:
                high_res_indices.append(i)
                
        futures = {}
        
        # --- PATH 1: Low Res (Model C) ---
        if low_res_indices:
            batch_c = [images[i].convert("RGB") for i in low_res_indices]
            f_c = self.executor.submit(self._predict_single, self.model_c, self.processor_c, batch_c)
            futures[f_c] = ("C", low_res_indices)

        # --- PATH 2: High Res (Ensemble A+B) ---
        if high_res_indices:
            batch_ab = []
            for i in high_res_indices:
                img = images[i].convert("RGB")
                w, h = img.size
                if max(w, h) > 1500:
                    ratio = 1024 / max(w, h)
                    img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
                batch_ab.append(img)
            
            # Submit A and B in parallel
            f_a = self.executor.submit(self._predict_single, self.model_a, self.processor_a, batch_ab)
            f_b = self.executor.submit(self._predict_single, self.model_b, self.processor_b, batch_ab)
            futures[f_a] = ("A", high_res_indices)
            futures[f_b] = ("B", high_res_indices)
            
        # Collect Results
        # Use simple dict for A/B result mapping
        high_res_results = {idx: {} for idx in high_res_indices} # {orig_idx: {A: probs, B: probs}}
        
        for f in concurrent.futures.as_completed(futures):
            model_key, indices = futures[f]
            probs = f.result()
            
            if probs is None: # handle error
                continue
                
            if model_key == "C":
                # Process C immediately
                for idx_in_batch, original_idx in enumerate(indices):
                    try:
                        score_real = float(probs[idx_in_batch][0])
                        score_ai = float(probs[idx_in_batch][1])
                        label = "AI" if score_ai > 0.5 else "REAL"
                        score = score_ai if label == "AI" else score_real
                        results[original_idx] = {
                            "ai_score": float(score_ai),
                            "label": label,
                            "confidence": float(score),
                            "router": "LowRes_ModelC",
                            "model_breakdown": {"C": score_ai}
                        }
                    except:
                        pass
            
            else:
                # Store A/B for later merge
                for idx_in_batch, original_idx in enumerate(indices):
                    high_res_results[original_idx][model_key] = probs[idx_in_batch]

        # Merge High Res
        for original_idx in high_res_indices:
            res_parts = high_res_results.get(original_idx, {})
            prob_a = res_parts.get("A")
            prob_b = res_parts.get("B")
            
            if prob_a is not None and prob_b is not None:
                # Merge Logic
                try:
                    val_a_ai = float(prob_a[0])
                    val_a_real = float(prob_a[1])
                    val_b_ai = float(prob_b[0])
                    val_b_real = float(prob_b[1])
                    
                    wA, wB = 0.60, 0.40
                    score_ai = (val_a_ai * wA) + (val_b_ai * wB)
                    score_real = (val_a_real * wA) + (val_b_real * wB)
                    
                    # Sharpness check re-calc (cheap on CPU)
                    # We need the image back.
                    # Since we are inside the loop, we access images[original_idx]
                    try:
                         # Quick sharpness check
                         img = images[original_idx]
                         w, h = img.size
                         if (w*h) >= 200000: # Confirm high res
                             # We can optimize: Only do this if score is borderline or specific condition?
                             # For now, keep original logic.
                             img_np = np.array(img.convert("L"))
                             sharpness = cv2.Laplacian(img_np, cv2.CV_64F).var()
                             if sharpness > 4000 and val_a_real < 0.90:
                                 score_ai = max(score_ai, 0.95)
                                 score_real = 1.0 - score_ai
                    except: pass
                    
                    label = "AI" if score_ai > score_real + 0.1 else "REAL"
                    final_score = score_ai if label == "AI" else score_real
                    if abs(score_ai - score_real) <= 0.1: final_score = 0.5

                    results[original_idx] = {
                        "ai_score": float(score_ai),
                        "label": label,
                        "confidence": float(final_score),
                        "router": "HighRes_Ensemble",
                        "model_breakdown": {"A": val_a_ai, "B": val_b_ai}
                    }
                except Exception as e:
                    logger.error(f"Merge error idx {original_idx}: {e}")
                    results[original_idx] = {"error": "Merge failed", "ai_score": 0.5}
            else:
                 # One model failed
                 results[original_idx] = {"error": "Model A/B mismatch", "ai_score": 0.5}

        return results

# Initialize Classifier Global
classifier = RouterClassifier()
# Decoder Pool
decode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8) # IO/CPU bound

def decode_image(args):
    idx, b64_str = args
    try:
        img_bytes = base64.b64decode(b64_str)
        img_hash = hashlib.md5(img_bytes).hexdigest()
        
        # Cache Check MUST be thread-safe (OrderedDict is not fully thread safe for writes? but read is ok-ish)
        # Using a centralized check is safer, or assuming cache hits are rare in this pool?
        # Actually, let's just decode here. Cache check can happen before or after.
        # Let's decode to bytes/hash here.
        
        return (idx, img_bytes, img_hash, None)
    except Exception as e:
        return (idx, None, None, str(e))

# ---------------- Main Handler ----------------
def handler(job):
    job_input = job.get("input", {})
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
    results = [None] * len(images_b64)
    
    # 1. Parallel Decode & Cache Check
    images_to_process = [] # (original_idx, PIL Image)
    hashes_to_process = []
    
    t0 = time.perf_counter()
    
    # Prepare args
    decode_args = [(i, s) for i, s in enumerate(images_b64)]
    
    # Launch parallel decoding
    futures = decode_pool.map(decode_image, decode_args)
    
    for idx, img_bytes, img_hash, err in futures:
        if err:
            results[idx] = {"error": err, "ai_score": 0.0}
            continue
            
        # Check Cache (Synced access)
        cached = worker_cache.get(img_hash)
        if cached:
            res = cached.copy() # Safe copy?
            res["cache_hit"] = True
            results[idx] = res
            continue
            
        # If not cached, convert to PIL (Fast enough in main thread or do we want parallel?)
        # PIL.Image.open is fast, convert RGB might take ms.
        # Let's do it here.
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images_to_process.append((idx, img))
            hashes_to_process.append(img_hash)
        except Exception as e:
             results[idx] = {"error": str(e), "ai_score": 0.0}

    decode_ms = (time.perf_counter() - t0) * 1000
    cache_hits = len(images_b64) - len(images_to_process)
    
    # 2. Parallel Inference
    if images_to_process:
        pil_images = [x[1] for x in images_to_process]
        t1 = time.perf_counter()
        
        preds = classifier.predict_batch(pil_images)
        
        gpu_ms = (time.perf_counter() - t1) * 1000
        
        for i, pred in enumerate(preds):
            original_idx = images_to_process[i][0]
            if pred:
                pred["cache_hit"] = False
                worker_cache.put(hashes_to_process[i], pred)
                results[original_idx] = pred
            else:
                results[original_idx] = {"error": "Prediction failed", "ai_score": 0.5}
                
    total_ms = (time.perf_counter() - total_start) * 1000
    
    response = {
        "results": results if is_batch else results[0],
        "timing_ms": {
            "decode": round(decode_ms, 2),
            "total": round(total_ms, 2),
            "cache_hits": cache_hits
        }
    }
    
    if not is_batch and isinstance(results[0], dict):
        results[0]["timing_ms"] = response["timing_ms"]
        return results[0]
        
    return response

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
