"""
Gemini SDK shim — exposes the shared `client` instance and the image-prep
helpers reused by `client_combined.py` (both the single-image and the
video-frames analyzers live there).

This module deliberately holds no detection logic. It's the bottom of the
import graph so prompt + schema modules can import the prep helpers and
the SDK client without circular dependencies.
"""

import io
import os
import logging
from PIL import Image
from google import genai
from google.genai import types
from typing import Union

from app.config import settings

# Cap decompression to 20 MP (from config) to prevent decompression-bomb DoS.
# A crafted PNG/TIFF can have a tiny file size but expand to multiple GB.
Image.MAX_IMAGE_PIXELS = settings.pil_max_image_pixels

logger = logging.getLogger(__name__)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=settings.gemini_http_timeout_ms,
        retry_options=types.HttpRetryOptions(
            attempts=settings.gemini_max_retries,
            initial_delay=settings.gemini_retry_initial_delay,
            max_delay=settings.gemini_retry_max_delay,
            exp_base=settings.gemini_retry_exp_base,
            http_status_codes=[408, 429, 500, 502, 503, 504]
        )
    )
)


def _prepare_pil_for_gemini(
    image_source: Union[str, Image.Image],
) -> tuple[Image.Image, list[Image.Image]]:
    """
    Shared image-prep used by both image and video analyzers in
    client_combined.py.

    Opens the image if given a path, resizes to gemini_max_pixels, ensures
    RGB mode. Returns the WORKING PIL image (callers can compute noise_cv,
    encode JPEG, etc.) plus a list of intermediate PIL handles the caller
    must close after they're done — done this way so the same prep can be
    reused without forcing every caller into a context manager pattern.
    """
    img_to_close: list[Image.Image] = []
    if isinstance(image_source, str):
        img_original = Image.open(image_source)
        img_to_close.append(img_original)
    else:
        img_original = image_source

    img_working = _resize_if_needed(img_original)
    if img_working is not img_original:
        img_to_close.append(img_working)

    if img_working.mode != "RGB":
        img_rgb = img_working.convert("RGB")
        img_to_close.append(img_rgb)
        img_working = img_rgb

    return img_working, img_to_close


def _encode_pil_as_jpeg(img: Image.Image, quality: int) -> bytes:
    """Encode a PIL image as JPEG bytes at the given quality."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _resize_if_needed(img: Image.Image) -> Image.Image:
    """
    Resizes image if it exceeds 4MP (~2048x2048) to limit token usage and avoid payload errors.
    Keeps aspect ratio.

    Resampler note: BICUBIC instead of LANCZOS. Lanczos aggressively smooths high-frequency
    pixel noise and micro-textures — the exact mathematical anomalies the vision model uses
    to spot diffusion/GAN signatures. Bicubic preserves a cleaner representation of those
    structural artifacts while still producing acceptable visual quality.
    """
    w, h = img.size
    pixels = w * h

    if pixels > settings.gemini_max_pixels:
        scale = (settings.gemini_max_pixels / pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    return img
