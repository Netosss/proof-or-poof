import runpod
import base64
import io
import torch
import logging
import numpy as np
from PIL import Image
from transformers import pipeline

# Set up logging for RunPod
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Initialization ---
device = 0 if torch.cuda.is_available() else -1
logger.info(f"Initializing worker on device: {'GPU' if device == 0 else 'CPU'}")

detector = None

try:
    logger.info("Loading SigLIP model (Ateeqq/ai-vs-human-image-detector)...")
    # SigLIP often requires timm and newer transformers versions.
    # We added 'timm' to requirements.txt and bumped transformers to 4.38.2.
    detector = pipeline(
        "image-classification", 
        model="Ateeqq/ai-vs-human-image-detector",
        device=device,
        trust_remote_code=True
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load model: {e}", exc_info=True)
    detector = None

def handler(job):
    """
    The main RunPod task handler.
    """
    job_input = job["input"]
    task = job_input.get("task")
    
    if task == "deep_forensic":
        if detector is None:
            return {"error": "Model failed to initialize on worker. Check worker logs."}

        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image data provided"}
        
        try:
            # Decode base64 to image
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # --- Normalization / Standardization ---
            # Resize to a standard size for more consistent classification results
            # even though the pipeline handles its own resizing.
            img = img.resize((512, 512), Image.LANCZOS)
            
            # Run detection
            results = detector(img)
            logger.info(f"Scan results: {results}")
            
            # Extract AI score
            ai_score = 0.0
            for res in results:
                # The model labels are usually 'ai' and 'hum' or 'AI' and 'Human'
                if res['label'].lower() == 'ai':
                    ai_score = float(res['score'])
            
            return {"ai_score": ai_score}
        except Exception as e:
            logger.error(f"Scan Error: {e}", exc_info=True)
            return {"error": f"Internal scan error: {str(e)}"}

    elif task == "video_removal":
        return {"status": "mocked_success", "message": "Video removal not yet implemented in worker"}
    
    return {"error": f"Invalid task: {task}"}

# Start the RunPod serverless loop
runpod.serverless.start({"handler": handler})
