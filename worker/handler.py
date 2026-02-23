import runpod
import base64
import time
import logging
from remover import FauxLensRemover

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Global Initialization ----------------
# This triggers the JIT warmup immediately on pod boot
remover = None
try:
    remover = FauxLensRemover()
    logger.info("FauxLensRemover initialized successfully.")
except Exception as e:
    logger.error(f"Worker initialization failed: {e}")
    # Don't set remover to None, just leave it as None to fail specific requests
    # but still allow the worker to start and report error


# ---------------- Worker Logic ----------------
def handler(job):
    if remover is None:
        return {"error": "Model engine failed to initialize."}

    job_input = job.get("input", {})
    
    # Support both single object and arrays for multi-image requests
    payloads = job_input.get("payloads", [])
    if not payloads and "image" in job_input and "mask" in job_input:
        payloads = [{"image": job_input["image"], "mask": job_input["mask"]}]
        
    if not payloads:
        return {"error": "Missing image and mask payload."}

    results = []
    start_time = time.perf_counter()

    # Process sequentially to protect VRAM boundaries
    for item in payloads:
        try:
            img_bytes = base64.b64decode(item["image"])
            mask_bytes = base64.b64decode(item["mask"])
            
            t0 = time.perf_counter()
            out_bytes = remover.process(img_bytes, mask_bytes)
            inf_time = (time.perf_counter() - t0) * 1000
            
            results.append({
                "image_base64": base64.b64encode(out_bytes).decode('utf-8'),
                "inference_ms": round(inf_time, 2)
            })
        except Exception as e:
            logger.error(f"Processing error: {e}")
            results.append({"error": str(e)})

    total_time = (time.perf_counter() - start_time) * 1000

    # If the user sent a single object payload, return a flat object
    # If they sent a batch (payloads array), return the batch results structure
    if "payloads" not in job_input and results:
        # Single image request -> Flat response (Backward Compatibility)
        single_res = results[0]
        single_res["timing_ms"] = round(total_time, 2)
        return single_res
    
    # Batch request -> Nested response
    return {
        "results": results,
        "timing_ms": round(total_time, 2)
    }

runpod.serverless.start({"handler": handler})
