import runpod
import base64
import io
import torch
from PIL import Image
from transformers import pipeline

# --- Global Initialization ---
# This runs once when the container starts
device = 0 if torch.cuda.is_available() else -1
print(f"Initializing worker on device: {'GPU' if device == 0 else 'CPU'}")

try:
    print("Loading SigLIP model (Ateeqq/ai-vs-human-image-detector)...")
    detector = pipeline(
        "image-classification", 
        model="Ateeqq/ai-vs-human-image-detector",
        device=device
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL: Failed to load model: {e}")
    detector = None

def handler(job):
    """
    The main RunPod task handler.
    """
    job_input = job["input"]
    task = job_input.get("task")
    
    if task == "deep_forensic":
        if detector is None:
            return {"error": "Model failed to initialize on worker"}

        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image data provided"}
        
        try:
            # Decode base64 to image
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Run detection
            results = detector(img)
            print(f"Scan results: {results}")
            
            # Extract AI score
            ai_score = 0.0
            for res in results:
                # The model labels are 'ai' and 'hum'
                if res['label'].lower() == 'ai':
                    ai_score = float(res['score'])
            
            return {"ai_score": ai_score}
        except Exception as e:
            print(f"Scan Error: {e}")
            return {"error": f"Internal scan error: {str(e)}"}

    elif task == "video_removal":
        # Placeholder for video logic
        return {"status": "mocked_success", "message": "Video removal not yet implemented in worker"}
    
    return {"error": f"Invalid task: {task}"}

# Start the RunPod serverless loop
runpod.serverless.start({"handler": handler})
