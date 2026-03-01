"""
Detection request helpers: URL/base64 image downloading, short ID generation,
and memory usage logging.
"""

import base64
import logging
import os
import secrets
import string

import aiohttp
import psutil
from fastapi import HTTPException

from app.config import settings
from app.integrations import http_client as http_module

logger = logging.getLogger(__name__)


def _generate_short_id(length: int = settings.short_id_length) -> str:
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def log_memory(stage: str) -> None:
    """Log current process and system memory usage. Only runs when DEBUG logging is active."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()
    logger.debug(
        f"[MEMORY] {stage} | "
        f"PID: {os.getpid()} | "
        f"Process RSS: {mem_info.rss / 1024 / 1024:.2f} MB | "
        f"System Available: {sys_mem.available / 1024 / 1024:.2f} MB / {sys_mem.total / 1024 / 1024:.2f} MB"
    )


async def download_image(url: str, max_size: int = settings.max_image_download_bytes) -> tuple[bytes, str]:
    """Downloads an image from a URL or decodes a base64 data URI."""
    if url.startswith("data:"):
        try:
            header, data_str = url.split(",", 1)
            if ";base64" not in header:
                raise HTTPException(status_code=400, detail="Only base64 data URIs are supported")
            content = base64.b64decode(data_str)
            if len(content) > max_size:
                raise HTTPException(status_code=400, detail=f"Image too large (max {max_size // (1024*1024)}MB)")
            mime_type = header.split(":")[1].split(";")[0]
            suffix = ".jpg"
            if "png" in mime_type:
                suffix = ".png"
            elif "jpeg" in mime_type or "jpg" in mime_type:
                suffix = ".jpg"
            elif "webp" in mime_type:
                suffix = ".webp"
            elif "gif" in mime_type:
                suffix = ".gif"
            elif "heic" in mime_type:
                suffix = ".heic"
            elif "heif" in mime_type:
                suffix = ".heif"
            elif "tiff" in mime_type:
                suffix = ".tiff"
            elif "bmp" in mime_type:
                suffix = ".bmp"
            return content, f"pasted_image{suffix}"
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error decoding data URI: {e}")
            raise HTTPException(status_code=400, detail="Invalid data URI")

    async with http_module.request_session() as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch image from URL: Status {response.status}"
                    )
                content = await response.read()
                if len(content) > max_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image too large (max {max_size // (1024*1024)}MB)"
                    )
                content_type = response.headers.get("Content-Type", "")
                suffix = ".jpg"
                if "png" in content_type:
                    suffix = ".png"
                elif "jpeg" in content_type or "jpg" in content_type:
                    suffix = ".jpg"
                elif "webp" in content_type:
                    suffix = ".webp"
                elif "gif" in content_type:
                    suffix = ".gif"
                elif "heic" in content_type:
                    suffix = ".heic"
                elif "heif" in content_type:
                    suffix = ".heif"
                elif "tiff" in content_type:
                    suffix = ".tiff"
                elif "bmp" in content_type:
                    suffix = ".bmp"
                elif "mp4" in content_type:
                    suffix = ".mp4"
                elif "quicktime" in content_type or "mov" in content_type:
                    suffix = ".mov"

                if not content_type or "application" in content_type or "octet-stream" in content_type:
                    lower_url = url.lower()
                    if lower_url.endswith(".png"):
                        suffix = ".png"
                    elif lower_url.endswith(".webp"):
                        suffix = ".webp"
                    elif lower_url.endswith(".gif"):
                        suffix = ".gif"
                    elif lower_url.endswith(".heic"):
                        suffix = ".heic"
                    elif lower_url.endswith(".heif"):
                        suffix = ".heif"
                    elif lower_url.endswith(".tiff") or lower_url.endswith(".tif"):
                        suffix = ".tiff"
                    elif lower_url.endswith(".bmp"):
                        suffix = ".bmp"
                    elif lower_url.endswith(".mp4"):
                        suffix = ".mp4"
                    elif lower_url.endswith(".mov"):
                        suffix = ".mov"

                return content, f"downloaded_media{suffix}"
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
