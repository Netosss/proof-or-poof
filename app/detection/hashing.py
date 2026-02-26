"""
File and image hashing utilities for the detection cache layer.

`get_safe_hash` is intentionally inlined here (not imported from core) to
avoid a circular dependency: detection → core → detection.
"""

import os
import hashlib
import logging
import numpy as np
from PIL import Image
from typing import Union

from app.config import settings

logger = logging.getLogger(__name__)


def get_safe_hash(data: bytes) -> str:
    """Securely hash raw bytes using SHA-256."""
    return hashlib.sha256(data).hexdigest()


def get_smart_file_hash(file_path: str) -> str:
    """
    Smart hashing for large video files.
    Reads Start+Middle+End chunks to create a unique signature without reading 200MB.
    """
    if not os.path.exists(file_path):
        return "missing_file"

    file_size = os.path.getsize(file_path)

    if file_size < settings.hash_chunk_threshold_bytes:
        with open(file_path, 'rb') as f:
            data = f.read()
            h = get_safe_hash(data)
            logger.info(f"[HASH] Full file hash for {file_path} ({file_size} bytes): {h}")
            return h

    chunk_size = settings.hash_chunk_size_bytes
    h = hashlib.sha256()

    with open(file_path, 'rb') as f:
        h.update(f.read(chunk_size))
        f.seek(file_size // 2)
        h.update(f.read(chunk_size))
        f.seek(max(0, file_size - chunk_size), os.SEEK_SET)
        h.update(f.read(chunk_size))

    h.update(str(file_size).encode())
    res = h.hexdigest()
    logger.info(f"[HASH] Smart hash for {file_path} ({file_size} bytes): {res}")
    return res


def get_image_hash(source: Union[str, Image.Image], fast_mode: bool = False) -> str:
    """
    Generate a secure SHA-256 hash of the image source (optimized with grayscale).
    fast_mode=True uses even smaller thumbnail for video frame caching.
    """
    if isinstance(source, str):
        with open(source, 'rb') as f:
            return get_safe_hash(f.read(settings.image_hash_header_bytes))
    else:
        thumb = source.copy()
        size = settings.image_hash_thumb_fast if fast_mode else settings.image_hash_thumb_full
        thumb.thumbnail(size)
        thumb = thumb.convert("L")
        return get_safe_hash(np.array(thumb).tobytes())
