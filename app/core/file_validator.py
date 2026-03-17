"""
File validation and log sanitization utilities.

Validation runs in three layers (cheapest → most expensive):
  1. MIME type   — from the multipart Content-Type header
  2. Extension   — from the filename
  3. Magic bytes — PIL.verify() + PIL.load() / ffprobe + cv2 (actual file I/O)

Layer 3 is the real security gate; layers 1 & 2 are cheap early rejects.

Security notes:
  - Layer 3 runs via asyncio.to_thread so PIL/cv2 blocking I/O never stalls
    the event loop.
  - PIL/cv2 error strings are sanitized before being surfaced in HTTP responses
    to prevent temp-file path leaks (e.g. /tmp/tmpXYZ123).
  - File size is measured from disk (os.path.getsize) when a path is available,
    not trusted from the caller's filesize parameter which can be spoofed via a
    crafted Content-Length header.

Security exclusions (intentional):
  - SVG / XML-based images  → can embed <script> / SSRF
  - PostScript / EPS / EMF / WMF → can execute arbitrary code
  - PDF → documents, not images
  - Scientific rasters (FITS, HDF5, GRIB) → not user-facing image files
"""

import asyncio
import json
import os
import re
import subprocess
import logging

import cv2
from fastapi import HTTPException
from PIL import Image

from app.config import settings

# Prevent decompression-bomb attacks for all non-Gemini image operations.
Image.MAX_IMAGE_PIXELS = settings.pil_max_image_pixels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIME type validation
#
# Layer 1 uses a BLOCKLIST (deny by default for dangerous types; allow all
# other image/* and video/*) rather than an explicit allowlist.
#
# Intentional design decision: this shifts Layer 1 from "deny unknown" to
# "allow unknown image/video".  This is safe here because Layer 3 is the real
# gate — PIL verify/load and cv2 will reject anything that is not a genuine
# image or video regardless of what the MIME header claims.  Using a blocklist
# means new formats (e.g. AVIF, HEIF variants, future codecs) work without
# code changes, while the known dangerous types (SVG, WMF, EPS, Flash) are
# always blocked regardless of extension.
# ---------------------------------------------------------------------------

# image/* sub-types that must never be accepted (security / not real images).
_BLOCKED_IMAGE_MIME_SUBTYPES: frozenset[str] = frozenset({
    "image/svg+xml",        # SVG → XSS/SSRF via embedded <script>
    "image/xml+svg",        # alternate SVG MIME
    "image/svg",            # non-standard SVG
    "image/x-wmf",          # Windows Metafile → can execute code
    "image/x-emf",          # Enhanced Metafile → can execute code
    "image/x-eps",          # Encapsulated PostScript → arbitrary code
})

# video/* sub-types that must never be accepted.
_BLOCKED_VIDEO_MIME_SUBTYPES: frozenset[str] = frozenset({
    "video/x-shockwave-flash",  # Flash → outdated, security risk
})

# ---------------------------------------------------------------------------
# File extension lists
#
# Covers every extension PIL (Pillow 10+) can open plus all common video
# container formats that ffmpeg/OpenCV can decode.
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    # JPEG family
    ".jpg", ".jpeg", ".jpe", ".jfif", ".jif",
    # PNG
    ".png",
    # GIF
    ".gif",
    # WebP
    ".webp",
    # BMP / DIB
    ".bmp", ".dib",
    # TIFF
    ".tiff", ".tif",
    # AVIF / AVIFS
    ".avif", ".avifs",
    # APNG
    ".apng",
    # HEIC / HEIF  (Apple Live Photos, iPhone default)
    ".heic", ".heif",
    # MPO  (dual-lens cameras: Samsung, Fujifilm, some Sony)
    ".mpo",
    # ICO / ICNS / CUR
    ".ico", ".icns", ".cur",
    # JPEG 2000 family
    ".jp2", ".j2k", ".j2c", ".jpc", ".jpf", ".jpx",
    # TGA / TARGA
    ".tga",
    # PCX / DCX (multi-page PCX)
    ".pcx", ".dcx",
    # Netpbm family
    ".ppm", ".pgm", ".pbm", ".pnm", ".pfm",
    # SGI / IRIX raster
    ".sgi", ".rgb", ".rgba", ".bw",
    # XBM / XPM  (X11 bitmap formats)
    ".xbm", ".xpm",
    # Photoshop
    ".psd",
    # IM format
    ".im",
    # Microsoft Paint
    ".msp",
    # Quite OK Image
    ".qoi",
    # DirectDraw Surface  (game textures)
    ".dds",
    # Sun Raster
    ".ras",
    # FLIC animation
    ".fli", ".flc",
    # Pixar
    ".pxr",
    # ICNS sub-formats
    ".icb", ".vda", ".vst",
    # Blizzard Mipmap  (game assets)
    ".blp",
})

ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    # MPEG-4 family
    ".mp4", ".m4v", ".m4p",
    # QuickTime / Apple
    ".mov", ".qt",
    # AVI / DivX
    ".avi", ".divx",
    # Matroska
    ".mkv",
    # WebM
    ".webm",
    # Flash Video
    ".flv", ".f4v",
    # Windows Media
    ".wmv", ".asf",
    # MPEG Transport Stream  (cameras, broadcasting, Blu-ray)
    # NOTE: ".ts" is intentionally excluded — it collides with TypeScript source
    # files and would produce a confusing "Could not open video stream" error
    # for any developer who accidentally uploads a .ts file.  Use the longer,
    # unambiguous aliases instead.
    ".mts", ".m2ts", ".m2t",
    # MPEG family
    ".mpeg", ".mpg", ".mpe", ".m2v", ".m2p",
    # 3GPP  (mobile)
    ".3gp", ".3g2", ".3gpp", ".3gpp2",
    # Ogg / Theora
    ".ogg", ".ogv",
    # RealMedia
    ".rm", ".rmvb",
    # DVD
    ".vob",
    # Digital Video  (DV camcorders)
    ".dv",
    # Professional broadcast
    ".mxf",
    # Windows Video
    ".wmx", ".wvx",
    # AMV  (portable MP4 players)
    ".amv",
    # H.264/H.265 raw elementary streams
    ".h264", ".h265", ".264", ".265",
    # HEVC container
    ".hevc",
})

ALLOWED_DETECT_EXTENSIONS: frozenset[str] = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
# Inpainting only: images (no video, no vector).
ALLOWED_INPAINT_EXTENSIONS: frozenset[str] = ALLOWED_IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Video codecs that cannot be decoded on the current platform.
# ffprobe is only called when cv2 fails (failure path), so this dict is
# consulted only for broken/exotic files — not on every video upload.
#
# Expand this dict based on deployment platform capabilities.  For example,
# H.265/HEVC decoding is hardware-dependent; add "hevc" here if your server
# reports cv2 failures with that codec.
# ---------------------------------------------------------------------------
_UNSUPPORTED_VIDEO_CODECS: dict[str, str] = {
    "av1": (
        "AV1 video codec is not currently supported. "
        "Please convert your video to H.264 (MP4) using HandBrake, FFmpeg, or any "
        "online converter and try again."
    ),
    # Example for platforms without HEVC hardware decoding:
    # "hevc": (
    #     "H.265/HEVC video codec is not supported on this server. "
    #     "Please convert to H.264 (MP4) and try again."
    # ),
}


# ---------------------------------------------------------------------------
# Blocking helpers — run via asyncio.to_thread to keep the event loop free.
# All path-containing error messages are sanitized before being returned so
# that PIL/cv2 strings like "/tmp/tmpXYZ123 is truncated" never reach clients.
# ---------------------------------------------------------------------------

def _probe_video_codec(file_path: str) -> str:
    """
    Return the primary video stream codec name (lowercase) via ffprobe.
    Returns an empty string if ffprobe is unavailable or the probe fails.
    Blocking — intended to run inside asyncio.to_thread.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                return streams[0].get("codec_name", "").lower()
    except Exception:
        pass
    return ""


def _check_image_magic_bytes(file_path: str, ext: str) -> None:
    """
    Blocking image validation: PIL verify + full load.

    - verify()  checks the file header / structural integrity
    - load()    forces a complete pixel decode, catching truncated trailing data
                (especially important for JPEG where verify() does not read all bytes)

    Raises ValueError with a sanitized message on failure.
    Blocking — intended to run inside asyncio.to_thread.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(
            f"Image failed header verification (format: {ext}): "
            f"{sanitize_log_message(str(exc))}"
        ) from exc

    # Must re-open after verify() — PIL exhausts the file pointer during verify.
    try:
        with Image.open(file_path) as img:
            detected_format = (img.format or "unknown").lower()
            img.load()
    except Exception as exc:
        raise ValueError(
            f"Image appears to be truncated or corrupted "
            f"(format: {detected_format}): "
            f"{sanitize_log_message(str(exc))}"
        ) from exc


def _check_video_magic_bytes(file_path: str, ext: str) -> None:
    """
    Blocking video validation: cv2 frame read, with ffprobe on failure only.

    Happy path (valid video):
      cv2 opens → reads one frame → returns immediately, no subprocess spawned.

    Failure path (corrupt / unsupported codec):
      cv2 fails → ffprobe runs to identify the codec → returns a codec-specific
      415 (e.g. AV1) or a generic 400 with the codec name for context.

    Deferring ffprobe to the failure path avoids spawning a subprocess on every
    video upload and removes a potential process-pool exhaustion vector under
    high traffic.

    Raises HTTPException(415) for known-unsupported codecs (e.g. AV1).
    Raises ValueError with a sanitized message for other read failures.
    Blocking — intended to run inside asyncio.to_thread.
    """
    cap = cv2.VideoCapture(file_path)
    can_open = cap.isOpened()
    can_read = False
    if can_open:
        ret, _ = cap.read()
        can_read = ret
    cap.release()

    if can_open and can_read:
        # Happy path — cv2 succeeded, no subprocess needed.
        return

    # cv2 failed.  Run ffprobe now to identify the codec and surface a
    # codec-specific 415 instead of a confusing generic 400.
    codec = _probe_video_codec(file_path)

    if codec in _UNSUPPORTED_VIDEO_CODECS:
        raise HTTPException(
            status_code=415,
            detail=_UNSUPPORTED_VIDEO_CODECS[codec],
        )

    if not can_open:
        raise ValueError(
            f"Could not open video stream "
            f"(ext: {ext}, codec: {codec or 'unknown'}). "
            "The file may be corrupted or encoded with an unsupported codec."
        )

    raise ValueError(
        f"Could not read video frames "
        f"(ext: {ext}, codec: {codec or 'unknown'}). "
        "Ensure the file is a valid, non-corrupted video. "
        "H.264 (MP4) is recommended for best compatibility."
    )


# ---------------------------------------------------------------------------
# Public validation entry point
# ---------------------------------------------------------------------------

async def validate_file(
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
      3. Magic bytes (PIL / ffprobe + cv2) — async, runs in thread pool

    Raises HTTPException (400 / 413 / 415) on failure.
    Returns True on success.
    """
    allowed_ext = ALLOWED_DETECT_EXTENSIONS if mode == "detect" else ALLOWED_INPAINT_EXTENSIONS

    # --- Layer 1: MIME type (cheap, from multipart header) ---
    if content_type:
        mime = content_type.split(";")[0].strip().lower()

        is_image_mime = mime.startswith("image/")
        is_video_mime = mime.startswith("video/")
        is_blocked = mime in _BLOCKED_IMAGE_MIME_SUBTYPES or mime in _BLOCKED_VIDEO_MIME_SUBTYPES

        if is_blocked:
            logger.warning("file_validation_failed", extra={
                "action": "file_validation_failed",
                "layer": "mime",
                "mime": mime,
                "media_file": filename,
                "filesize": filesize,
                "mode": mode,
                "error": f"Blocked MIME type: {mime}",
            })
            raise HTTPException(
                status_code=415,
                detail=f"File type '{mime}' is not allowed for security reasons.",
            )

        if mode == "inpaint" and is_video_mime:
            logger.warning("file_validation_failed", extra={
                "action": "file_validation_failed",
                "layer": "mime",
                "mime": mime,
                "media_file": filename,
                "filesize": filesize,
                "mode": mode,
                "error": "Video MIME type sent to inpaint endpoint",
            })
            raise HTTPException(
                status_code=415,
                detail="Object removal only supports images, not videos.",
            )

        if not is_image_mime and not is_video_mime:
            logger.warning("file_validation_failed", extra={
                "action": "file_validation_failed",
                "layer": "mime",
                "mime": mime,
                "media_file": filename,
                "filesize": filesize,
                "mode": mode,
                "error": f"MIME type {mime!r} is not an image or video",
            })
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type: '{mime}'. "
                    "Only image and video files are accepted."
                ),
            )

    # --- Layer 2: Extension ---
    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in ALLOWED_IMAGE_EXTENSIONS
    is_video = ext in ALLOWED_VIDEO_EXTENSIONS

    if ext not in allowed_ext:
        logger.warning("file_validation_failed", extra={
            "action": "file_validation_failed",
            "layer": "extension",
            "ext": ext or "(none)",
            "media_file": filename,
            "filesize": filesize,
            "mode": mode,
            "error": f"Extension {ext!r} not in allowed list",
        })
        if mode == "inpaint":
            detail = (
                f"Unsupported file extension '{ext}'. "
                "Object removal supports images: JPEG, PNG, WebP, GIF, BMP, TIFF, "
                "HEIC, AVIF, PSD, TGA, ICO, JPEG 2000, and many more."
            )
        else:
            detail = (
                f"Unsupported file extension '{ext}'. "
                "Detection supports all common image formats (JPEG, PNG, WebP, GIF, "
                "BMP, TIFF, HEIC, AVIF, PSD, TGA, ICO, JPEG 2000…) and video formats "
                "(MP4, MOV, AVI, MKV, WebM, FLV, TS, MTS, M4V, WMV and many more)."
            )
        raise HTTPException(status_code=415, detail=detail)

    # --- Size check ---
    # When the file is on disk, measure actual size from the filesystem rather
    # than trusting the caller-supplied filesize (which can be spoofed via a
    # crafted Content-Length multipart header).
    if file_path:
        try:
            filesize = os.path.getsize(file_path)
        except OSError:
            pass  # Fall back to caller-supplied value if stat fails

    size_limit = settings.max_image_upload_bytes if is_image else settings.max_video_upload_bytes
    if filesize > size_limit:
        limit_mb = size_limit // 1024 // 1024
        kind = "Image" if is_image else "Video"
        raise HTTPException(
            status_code=413,
            detail=f"{kind} too large. Max {limit_mb} MB allowed.",
        )

    # --- Layer 3: Magic bytes ---
    # Run blocking PIL / ffprobe / cv2 calls in a thread pool so the async
    # event loop is never stalled waiting for file I/O.
    if file_path:
        try:
            if is_image:
                await asyncio.to_thread(_check_image_magic_bytes, file_path, ext)
            else:
                await asyncio.to_thread(_check_video_magic_bytes, file_path, ext)

        except HTTPException:
            # Codec-specific 415 errors — already have clear user-facing messages.
            raise
        except (ValueError, OSError) as exc:
            # sanitize_log_message strips temp-file paths like /tmp/tmpXYZ123
            # before they reach either logs or the HTTP response body.
            safe_msg = sanitize_log_message(str(exc))
            logger.warning("file_validation_failed", extra={
                "action": "file_validation_failed",
                "layer": "magic_bytes",
                "ext": ext,
                "media_file": filename,
                "filesize": filesize,
                "mode": mode,
                "error": safe_msg,
            })
            raise HTTPException(
                status_code=400,
                detail=safe_msg or "Invalid file content or format mismatch.",
            )
        except Exception as exc:
            safe_msg = sanitize_log_message(str(exc))
            logger.error("file_validation_error", extra={
                "action": "file_validation_error",
                "ext": ext,
                "media_file": filename,
                "filesize": filesize,
                "mode": mode,
                "error": safe_msg,
                "error_type": type(exc).__name__,
            })
            raise HTTPException(
                status_code=400,
                detail="Invalid file content or format mismatch.",
            )

    return True


def sanitize_log_message(message: str) -> str:
    """Strip sensitive server-side file paths from log messages and HTTP responses."""
    msg = re.sub(r'\/[^\s]+\/tmp[a-zA-Z0-9_]+', '[TEMP_FILE]', message)
    msg = re.sub(r'\/[^\s]+\/([^\/\s]+)', r'.../\1', msg)
    return msg
