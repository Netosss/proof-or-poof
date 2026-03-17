"""
Pure unit tests for app/core/file_validator.py.

Image content validation uses a real in-memory JPEG written to a temp file.
OpenCV (video) is mocked so no video codec is needed in CI.

All tests that exercise layer-3 (magic-bytes) are async because validate_file
now runs PIL/cv2 I/O inside asyncio.to_thread.  Tests for layers 1+2 are also
async (consistent style) since the function signature is async throughout.
"""

import io
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from app.core.file_validator import sanitize_log_message, validate_file


def _write_tiny_jpg(tmp_path) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(100, 150, 200)).save(buf, format="JPEG")
    p = tmp_path / "valid.jpg"
    p.write_bytes(buf.getvalue())
    return str(p)


# ---------------------------------------------------------------------------
# Extension + size checks (no file_path, so PIL/CV2 not called)
# ---------------------------------------------------------------------------


async def test_valid_image_extension_and_size():
    assert await validate_file("photo.jpg", 100, None) is True


async def test_image_too_large_raises_413():
    from app.config import settings

    oversized = settings.max_image_upload_bytes + 1
    with pytest.raises(HTTPException) as exc:
        await validate_file("photo.jpg", oversized, None)
    assert exc.value.status_code == 413


async def test_video_too_large_raises_413():
    from app.config import settings

    oversized = settings.max_video_upload_bytes + 1
    with pytest.raises(HTTPException) as exc:
        await validate_file("clip.mp4", oversized, None)
    assert exc.value.status_code == 413


async def test_unsupported_extension_raises_415():
    with pytest.raises(HTTPException) as exc:
        await validate_file("malware.exe", 100, None)
    assert exc.value.status_code == 415


# ---------------------------------------------------------------------------
# Content integrity checks (PIL-verified real JPEG)
# ---------------------------------------------------------------------------


async def test_valid_jpeg_content_passes(tmp_path):
    path = _write_tiny_jpg(tmp_path)
    result = await validate_file("valid.jpg", os.path.getsize(path), path)
    assert result is True


async def test_corrupted_image_content_raises_400(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"this is not an image")
    with pytest.raises(HTTPException) as exc:
        await validate_file("bad.jpg", len(p.read_bytes()), str(p))
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Video content check (OpenCV mocked)
# ---------------------------------------------------------------------------


async def test_valid_video_content_passes(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake_mp4_bytes")

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, MagicMock())

    with patch("app.core.file_validator.cv2") as mock_cv2, \
         patch("app.core.file_validator._probe_video_codec", return_value="h264"):
        mock_cv2.VideoCapture.return_value = mock_cap
        result = await validate_file("clip.mp4", len(p.read_bytes()), str(p))

    assert result is True


# ---------------------------------------------------------------------------
# Path-leak prevention: error messages must be sanitized
# ---------------------------------------------------------------------------


async def test_error_message_does_not_leak_temp_path(tmp_path):
    """
    When PIL raises an error containing a temp file path, the HTTP response
    detail must have that path stripped by sanitize_log_message.
    """
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")

    with pytest.raises(HTTPException) as exc:
        await validate_file("bad.jpg", len(p.read_bytes()), str(p))

    assert str(tmp_path) not in exc.value.detail
    assert str(p) not in exc.value.detail


# ---------------------------------------------------------------------------
# Actual file size is measured from disk, not trusted from caller
# ---------------------------------------------------------------------------


async def test_actual_filesize_from_disk_used_for_size_check(tmp_path):
    """
    The size check must use os.path.getsize, not the caller-supplied value.
    Passing a fake small filesize should not bypass the limit.
    """
    from app.config import settings

    path = _write_tiny_jpg(tmp_path)
    # Write a valid image but lie about its size to try to bypass the limit.
    # The real size is tiny, but we fake a massive caller-supplied value.
    # Since validate_file now measures from disk, this should NOT raise 413.
    result = await validate_file("photo.jpg", settings.max_image_upload_bytes + 1, path)
    assert result is True


# ---------------------------------------------------------------------------
# sanitize_log_message
# ---------------------------------------------------------------------------


def test_sanitize_log_message_strips_temp_path():
    msg = "Processing /tmp/tmpABCDEF/uploaded_file.jpg successfully"
    sanitized = sanitize_log_message(msg)
    assert "/tmp/tmpABCDEF" not in sanitized


def test_sanitize_log_message_keeps_non_path_content():
    msg = "No issues found"
    assert sanitize_log_message(msg) == msg
