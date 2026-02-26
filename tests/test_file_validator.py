"""
Pure unit tests for app/core/file_validator.py.

Image content validation uses a real in-memory JPEG written to a temp file.
OpenCV (video) is mocked so no video codec is needed in CI.
"""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch

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


def test_valid_image_extension_and_size():
    assert validate_file("photo.jpg", 100, None) is True


def test_image_too_large_raises_413():
    from app.config import settings

    oversized = settings.max_image_upload_bytes + 1
    with pytest.raises(HTTPException) as exc:
        validate_file("photo.jpg", oversized, None)
    assert exc.value.status_code == 413


def test_video_too_large_raises_413():
    from app.config import settings

    oversized = settings.max_video_upload_bytes + 1
    with pytest.raises(HTTPException) as exc:
        validate_file("clip.mp4", oversized, None)
    assert exc.value.status_code == 413


def test_unsupported_extension_raises_415():
    with pytest.raises(HTTPException) as exc:
        validate_file("malware.exe", 100, None)
    assert exc.value.status_code == 415


# ---------------------------------------------------------------------------
# Content integrity checks (PIL-verified real JPEG)
# ---------------------------------------------------------------------------


def test_valid_jpeg_content_passes(tmp_path):
    path = _write_tiny_jpg(tmp_path)
    result = validate_file("valid.jpg", os.path.getsize(path), path)
    assert result is True


def test_corrupted_image_content_raises_400(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"this is not an image")
    with pytest.raises(HTTPException) as exc:
        validate_file("bad.jpg", len(p.read_bytes()), str(p))
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Video content check (OpenCV mocked)
# ---------------------------------------------------------------------------


def test_valid_video_content_passes(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake_mp4_bytes")

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, MagicMock())

    with patch("app.core.file_validator.cv2") as mock_cv2:
        mock_cv2.VideoCapture.return_value = mock_cap
        result = validate_file("clip.mp4", len(p.read_bytes()), str(p))

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
