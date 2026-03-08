"""
File validation and log sanitization utilities.

Validation runs in three layers (cheapest → most expensive):
  1. MIME type   — from the multipart Content-Type header (O(1) set lookup)
  2. Extension   — from the filename (O(1) set lookup)
  3. Magic bytes — PIL.verify() / cv2.VideoCapture (actual file I/O)

Layer 3 is the real security gate; layers 1 & 2 are cheap early rejects
that keep garbage off disk before we ever open the file.

SVG is intentionally excluded even though the frontend accepts it — SVG is
XML that can embed <script> tags and is a common XSS/SSRF attack vector.

Sets PIL.Image.MAX_IMAGE_PIXELS to prevent decompression-bomb attacks.
Note: app/integrations/gemini/client.py raises it for its own processing;
both settings are intentional.
"""

import os
import re
import logging

import cv2
from fastapi import HTTPException
from PIL import Image

from app.config import settings

# Prevent decompression-bomb attacks for all non-Gemini image operations.
Image.MAX_IMAGE_PIXELS = settings.pil_max_image_pixels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed MIME types — single source of truth, kept in sync with the frontend.
# SVG is deliberately absent (XSS/SSRF risk).
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/avif",
    "image/apng",
    "image/heic",
    "image/heif",
    "image/bmp",
    "image/tiff",
})

ALLOWED_VIDEO_MIME_TYPES: frozenset[str] = frozenset({
    "video/mp4",
    "video/quicktime",   # .mov
    "video/webm",
    "video/ogg",
    "video/x-matroska",  # .mkv
    "video/x-msvideo",   # .avi
    "video/x-ms-wmv",    # .wmv
    "video/mpeg",
    "video/3gpp",
    "video/3gpp2",
})

ALLOWED_DETECT_MIME_TYPES: frozenset[str] = ALLOWED_IMAGE_MIME_TYPES | ALLOWED_VIDEO_MIME_TYPES

# Inpainting only supports raster images (no video, no vector).
ALLOWED_INPAINT_MIME_TYPES: frozenset[str] = ALLOWED_IMAGE_MIME_TYPES

# ---------------------------------------------------------------------------
# Corresponding file extensions
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".avif", ".apng",
    ".heic", ".heif",
    ".bmp", ".tiff", ".tif",
})

ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".webm", ".ogg",
    ".mkv", ".avi", ".wmv", ".mpeg", ".mpg",
    ".3gp", ".3g2",
})

ALLOWED_DETECT_EXTENSIONS: frozenset[str] = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
ALLOWED_INPAINT_EXTENSIONS: frozenset[str] = ALLOWED_IMAGE_EXTENSIONS

# Formats PIL recognises (used in the magic-bytes cross-check).
_PIL_IMAGE_FORMATS: frozenset[str] = frozenset({
    "jpeg", "jpg", "png", "webp", "gif",
    "avif", "apng", "heic", "heif",
    "bmp", "tiff",
})


def validate_file(
    filename: str,
    filesize: int,
    file_path: str | None = None,
    content_type: str | None = None,
    *,
    mode: str = "detect",   # "detect" | "inpaint"
) -> bool:
    """
    Validate a file upload in three layers:
      1. MIME type (if provided by the multipart boundary)
      2. File extension
      3. Magic bytes (PIL / cv2) when file_path is given

    Raises HTTPException (400 / 413 / 415) on failure.
    Returns True on success.
    """
    allowed_mime = ALLOWED_DETECT_MIME_TYPES if mode == "detect" else ALLOWED_INPAINT_MIME_TYPES
    allowed_ext  = ALLOWED_DETECT_EXTENSIONS  if mode == "detect" else ALLOWED_INPAINT_EXTENSIONS

    # --- Layer 1: MIME type (cheap, from multipart header) ---
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if mime not in allowed_mime:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type: {mime}",
            )

    # --- Layer 2: Extension ---
    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in ALLOWED_IMAGE_EXTENSIONS
    is_video = ext in ALLOWED_VIDEO_EXTENSIONS

    if ext not in allowed_ext:
        raise HTTPException(status_code=415, detail="Unsupported file format.")

    # --- Size check (after type is confirmed) ---
    size_limit = settings.max_image_upload_bytes if is_image else settings.max_video_upload_bytes
    if filesize > size_limit:
        limit_mb = size_limit // 1024 // 1024
        kind = "Image" if is_image else "Video"
        raise HTTPException(
            status_code=413,
            detail=f"{kind} too large. Max {limit_mb} MB allowed.",
        )

    # --- Layer 3: Magic bytes (most expensive — only when file is on disk) ---
    if file_path:
        try:
            if is_image:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    actual = (img.format or "").lower()
                    if actual == "jpeg":
                        actual = "jpg"
                    if actual not in _PIL_IMAGE_FORMATS:
                        raise ValueError(f"Unexpected PIL format: {actual!r}")
            else:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    raise ValueError("Could not open video stream")
                ret, _ = cap.read()
                cap.release()
                if not ret:
                    raise ValueError("Could not read video frames")
        except (HTTPException, ValueError) as exc:
            logger.warning("file_validation_failed", extra={
                "action": "file_validation_failed",
                "ext": ext,
                "filesize": filesize,
                "error": str(exc),
            })
            raise HTTPException(
                status_code=400,
                detail="Invalid file content or format mismatch.",
            )
        except Exception as exc:
            logger.error("file_validation_error", extra={
                "action": "file_validation_error",
                "ext": ext,
                "filesize": filesize,
                "error": str(exc),
            })
            raise HTTPException(
                status_code=400,
                detail="Invalid file content or format mismatch.",
            )

    return True


def sanitize_log_message(message: str) -> str:
    """Strip sensitive file paths from log messages."""
    msg = re.sub(r'\/[^\s]+\/tmp[a-zA-Z0-9_]+', '[TEMP_FILE]', message)
    msg = re.sub(r'\/[^\s]+\/([^\/\s]+)', r'.../\1', msg)
    return msg
