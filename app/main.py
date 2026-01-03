import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.detector import detect_ai_media
from app.schemas import DetectionResponse
from app.runpod_client import run_image_removal, run_video_removal

app = FastAPI(title="AI Provenance & Cleansing API")

# Print to logs to confirm startup
print("Starting AI Provenance & Cleansing API...")
print(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")

# Add CORS Middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    DETECT API: Always cheap, NO RunPod, NO GPU.
    Exclusively uses C2PA metadata.
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name

    try:
        result = await detect_ai_media(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    """
    WATERMARK REMOVAL API:
    - Videos: Initiates RunPod async processing (Returns 202 while warming up).
    - Images: Uses Cheap local mock.
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    content_type = file.content_type
    is_video = content_type.startswith("video/") or suffix in [".mp4", ".mov", ".avi"]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name

    try:
        if is_video:
            # TRIGGER THE MUSCLE: Async polling for RunPod GPU
            # Using 202 Accepted logic effectively via the polling loop
            cleansing_result = await run_video_removal(temp_path)
            return {
                "status": "success",
                "method": "runpod_gpu",
                "result": cleansing_result
            }
        else:
            return {
                "status": "success",
                "method": "local_cheap",
                "message": "Image watermark removal placeholder (Returning file as-is)",
                "filename": file.filename
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    import os
    # Read port from environment variable for production (Railway/Render)
    # We use a more robust check for the PORT variable
    port_str = os.getenv("PORT")
    if port_str:
        port = int(port_str)
        print(f"Railway PORT detected: {port}")
    else:
        port = 8000
        print("No Railway PORT detected, defaulting to 8000")
        
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
