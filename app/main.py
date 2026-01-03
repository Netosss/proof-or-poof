import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.detector import detect_ai_media
from app.schemas import DetectionResponse
from app.runpod_client import run_image_removal, run_video_removal

app = FastAPI(title="AI Provenance & Cleansing API")

# ---- Startup logs (good for Railway) ----
print("Starting AI Provenance & Cleansing API...")
print(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")

# ---- CORS (lock this down later) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Healthcheck (Railway uses this) ----
@app.get("/health")
async def health():
    return {"status": "healthy"}

# ---- Detect endpoint ----
@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Cheap detection only (C2PA metadata).
    No GPU, no RunPod.
    """
    suffix = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name

    try:
        return await detect_ai_media(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---- Watermark removal ----
@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    """
    Images → local cheap placeholder
    Videos → RunPod GPU
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    content_type = file.content_type or ""
    is_video = content_type.startswith("video/") or suffix in {".mp4", ".mov", ".avi"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name

    try:
        if is_video:
            result = await run_video_removal(temp_path)
            return {
                "status": "success",
                "method": "runpod_gpu",
                "result": result,
            }

        return {
            "status": "success",
            "method": "local_cheap",
            "message": "Image watermark removal placeholder",
            "filename": file.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
