import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.detector import detect_ai_media
from app.schemas import DetectionResponse
from app.runpod_client import run_image_removal, run_video_removal

app = FastAPI(title="AI Provenance & Cleansing API")

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Endpoint to detect AI provenance using C2PA metadata.
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name

    try:
        # detect_ai_media is now async
        result = await detect_ai_media(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/process")
async def process(file: UploadFile = File(...)):
    """
    Triage and cleanse AI media. 
    Checks metadata first (free), then uses GPU if needed.
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    is_video = suffix in [".mp4", ".mov", ".avi"]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name

    try:
        # 1. Triage: Check C2PA metadata
        detection_result = await detect_ai_media(temp_path)
        
        # 2. Early Exit: If C2PA found AI, return immediately
        if detection_result.get("is_ai") is True:
            return {
                "detection": detection_result,
                "cleansing": "skipped_metadata_found",
                "status": "success"
            }
        
        # 3. GPU Muscle: If no metadata, send to RunPod
        if is_video:
            cleansing_result = await run_video_removal(temp_path)
        else:
            cleansing_result = await run_image_removal(temp_path)
            
        return {
            "detection": detection_result,
            "cleansing": cleansing_result,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
