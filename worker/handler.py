import runpod
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import base64
import io
import os
import cv2
import numpy as np
import tempfile

# Initialize models (FlashBoot: loaded from local path)
MODEL_PATH = "/models/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
)
pipe = pipe.to(device)

def process_image(image_base64: str) -> str:
    """
    Cleanses a single image using a 0.15 denoise pass.
    """
    image_data = base64.b64decode(image_base64)
    init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 0.15 denoise is enough to break pixel signatures without changing the image much
    with torch.autocast(device):
        result_image = pipe(
            prompt="", 
            image=init_image, 
            strength=0.15, 
            guidance_scale=7.5
        ).images[0]
    
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_video(video_base64: str) -> str:
    """
    Cleanses a video by processing frames in batches.
    """
    # 1. Decode video to temporary file
    video_data = base64.b64decode(video_base64)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_data)
        video_path = tmp_in.name

    # 2. Extract frames and process in batches
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = video_path.replace(".mp4", "_cleansed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    batch_size = 4 # Optimized for RTX 4090
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(pil_img)
        
        if len(frames) == batch_size:
            # Batch process
            with torch.autocast(device):
                results = pipe(
                    prompt="", 
                    image=frames, 
                    strength=0.15, 
                    guidance_scale=7.5
                ).images
            
            # Write results back to video
            for res_img in results:
                res_frame = cv2.cvtColor(np.array(res_img), cv2.COLOR_RGB2BGR)
                out.write(res_frame)
            frames = []

    # Process remaining frames
    if frames:
        with torch.autocast(device):
            results = pipe(prompt="", image=frames, strength=0.15).images
        for res_img in results:
            res_frame = cv2.cvtColor(np.array(res_img), cv2.COLOR_RGB2BGR)
            out.write(res_frame)

    cap.release()
    out.release()
    
    # 3. Read back cleansed video
    with open(out_path, "rb") as f:
        cleansed_video_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    # Cleanup
    os.remove(video_path)
    os.remove(out_path)
    
    return cleansed_video_b64

def handler(job):
    """
    Main handler for RunPod jobs.
    """
    job_input = job["input"]
    task = job_input.get("task")
    
    if task == "image_removal":
        return {"cleansed_image": process_image(job_input["image"])}
    elif task == "video_removal":
        return {"cleansed_video": process_video(job_input["video"])}
    else:
        return {"error": "Invalid task"}

runpod.serverless.start({"handler": handler})

