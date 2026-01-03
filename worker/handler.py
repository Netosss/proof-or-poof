import runpod
import base64
import io
from PIL import Image
from transformers import pipeline

# Load model globally in the worker so it stays in GPU memory
print("Loading SigLIP model...")
detector = pipeline("image-classification", model="Ateeqq/ai-vs-human-image-detector")

def handler(job):
    job_input = job["input"]
    task = job_input.get("task")
    
    if task == "deep_forensic":
        image_base64 = job_input.get("image")
        if not image_base64:
            return {"error": "No image data provided"}
        
        try:
            # Decode base64 to image
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Run detection
            results = detector(img)
            
            # Extract AI score
            ai_score = 0.0
            for res in results:
                if res['label'].lower() == 'ai':
                    ai_score = float(res['score'])
            
            return {"ai_score": ai_score}
        except Exception as e:
            return {"error": str(e)}

    elif task == "video_removal":
        video_data = job_input.get("video")
        return {"cleansed_video": video_data, "status": "mocked_success"}
    
    return {"error": "Invalid task"}

runpod.serverless.start({"handler": handler})
