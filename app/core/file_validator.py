"""
File validation and log sanitization utilities.

Sets PIL.Image.MAX_IMAGE_PIXELS to prevent decompression-bomb attacks.
Note: app/integrations/gemini/client.py sets it to None for its own processing;
both settings are intentional and preserve the original behavior.
"""

import os
import re
import logging

import cv2
from fastapi import HTTPException
from PIL import Image

from app.config import settings

# Prevent decompression-bomb attacks for all non-Gemini image operations
Image.MAX_IMAGE_PIXELS = settings.pil_max_image_pixels

logger = logging.getLogger(__name__)


def validate_file(filename: str, filesize: int, file_path: str = None) -> bool:
    """Check file extension, size, and content integrity using magic bytes."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif', '.tiff', '.tif', '.bmp']:
        if filesize > settings.max_image_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Max {settings.max_image_upload_bytes // 1024 // 1024}MB allowed."
            )
    elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        if filesize > settings.max_video_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Video too large. Max {settings.max_video_upload_bytes // 1024 // 1024}MB allowed."
            )
    else:
        raise HTTPException(status_code=415, detail="Unsupported file format.")

    if file_path:
        try:
            if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif', '.tiff', '.tif', '.bmp']:
                with Image.open(file_path) as img:
                    img.verify()
                    with Image.open(file_path) as img2:
                        actual_format = img2.format.lower()
                        if actual_format == 'jpeg':
                            actual_format = 'jpg'
                        if actual_format not in ['jpg', 'jpeg', 'png', 'webp', 'gif', 'heic', 'heif', 'tiff', 'tif', 'bmp']:
                            raise Exception(f"Format mismatch: {actual_format}")
            else:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    raise Exception("Could not open video stream")
                ret, _ = cap.read()
                cap.release()
                if not ret:
                    raise Exception("Could not read video frames")
        except Exception as e:
            logger.error(f"Malicious or corrupted file detected ({filename}): {e}")
            raise HTTPException(status_code=400, detail="Invalid file content or format mismatch.")

    return True


def sanitize_log_message(message: str) -> str:
    """Strip sensitive file paths from log messages."""
    msg = re.sub(r'\/[^\s]+\/tmp[a-zA-Z0-9_]+', '[TEMP_FILE]', message)
    msg = re.sub(r'\/[^\s]+\/([^\/\s]+)', r'.../\1', msg)
    return msg
