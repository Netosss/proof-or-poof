"""
Pure unit tests for app/core/file_validator.py.

Image content validation uses a real in-memory JPEG written to a temp file.
Video validation is tested by mocking _probe_video_codec — no real ffprobe
or video codec is needed in CI.

All tests that exercise layer-3 (magic-bytes) are async because validate_file
runs PIL/ffprobe I/O inside asyncio.to_thread.  Tests for layers 1+2 are also
async (consistent style) since the function signature is async throughout.

Test sections:
  1. Layer 1 — MIME type validation (blocklist + mode checks)
  2. Layer 2 — File extension + size checks
  3. Layer 3 — Image magic bytes (real PIL)
  4. Layer 3 — Video magic bytes (mocked _probe_video_codec)
  5. Security — path-leak prevention & Content-Length spoofing
  6. sanitize_log_message — regex correctness
"""

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from app.core.file_validator import sanitize_log_message, validate_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_jpg(tmp_path) -> str:
    """Write a minimal but valid JPEG to disk and return its path."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(100, 150, 200)).save(buf, format="JPEG")
    p = tmp_path / "valid.jpg"
    p.write_bytes(buf.getvalue())
    return str(p)


def _mock_probe(codec: str = "h264", streams: list | None = None):
    """Return a patch target and return value for _probe_video_codec."""
    if streams is None:
        streams = [{"codec_name": codec}]
    return patch(
        "app.core.file_validator._probe_video_codec",
        return_value=(codec, streams),
    )


# ---------------------------------------------------------------------------
# 1. Layer 1 — MIME type validation
# ---------------------------------------------------------------------------


async def test_blocked_svg_mime_raises_415():
    with pytest.raises(HTTPException) as exc:
        await validate_file("image.svg", 100, None, content_type="image/svg+xml")
    assert exc.value.status_code == 415
    assert "not allowed" in exc.value.detail


async def test_blocked_svg_alternate_mime_raises_415():
    with pytest.raises(HTTPException) as exc:
        await validate_file("image.svg", 100, None, content_type="image/svg")
    assert exc.value.status_code == 415


async def test_blocked_wmf_mime_raises_415():
    with pytest.raises(HTTPException) as exc:
        await validate_file("image.wmf", 100, None, content_type="image/x-wmf")
    assert exc.value.status_code == 415


async def test_blocked_eps_mime_raises_415():
    with pytest.raises(HTTPException) as exc:
        await validate_file("image.eps", 100, None, content_type="image/x-eps")
    assert exc.value.status_code == 415


async def test_blocked_flash_video_mime_raises_415():
    with pytest.raises(HTTPException) as exc:
        await validate_file("video.swf", 100, None, content_type="video/x-shockwave-flash")
    assert exc.value.status_code == 415


async def test_non_image_non_video_mime_raises_415():
    """A plain application/* MIME type should be rejected at layer 1."""
    with pytest.raises(HTTPException) as exc:
        await validate_file("file.pdf", 100, None, content_type="application/pdf")
    assert exc.value.status_code == 415
    assert "Only image and video" in exc.value.detail


async def test_video_mime_in_inpaint_mode_raises_415():
    """Inpainting only accepts images — a video MIME must be rejected."""
    with pytest.raises(HTTPException) as exc:
        await validate_file(
            "clip.mp4", 100, None,
            content_type="video/mp4",
            mode="inpaint",
        )
    assert exc.value.status_code == 415
    assert "images" in exc.value.detail.lower()


async def test_valid_image_mime_passes_layer1():
    """Any non-blocked image/* MIME type should not be rejected at layer 1."""
    # No file_path → layer 3 is skipped; only layers 1+2 run.
    result = await validate_file("photo.jpg", 100, None, content_type="image/jpeg")
    assert result is True


async def test_mime_with_charset_param_is_accepted():
    """Content-Type headers sometimes carry charset params; they should be stripped."""
    result = await validate_file(
        "photo.jpg", 100, None,
        content_type="image/jpeg; charset=utf-8",
    )
    assert result is True


# ---------------------------------------------------------------------------
# 2. Layer 2 — Extension + size checks (no file_path → PIL/ffprobe not called)
# ---------------------------------------------------------------------------


async def test_valid_image_extension_and_size():
    assert await validate_file("photo.jpg", 100, None) is True


async def test_valid_video_extension_and_size():
    assert await validate_file("clip.mp4", 100, None) is True


async def test_ts_extension_rejected_415():
    """.ts is intentionally excluded to avoid collision with TypeScript files."""
    with pytest.raises(HTTPException) as exc:
        await validate_file("script.ts", 100, None)
    assert exc.value.status_code == 415


async def test_unsupported_extension_raises_415():
    with pytest.raises(HTTPException) as exc:
        await validate_file("malware.exe", 100, None)
    assert exc.value.status_code == 415


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


async def test_inpaint_mode_rejects_video_extension():
    """Video extensions must be rejected in inpaint mode regardless of MIME."""
    with pytest.raises(HTTPException) as exc:
        await validate_file("clip.mp4", 100, None, mode="inpaint")
    assert exc.value.status_code == 415


# ---------------------------------------------------------------------------
# 3. Layer 3 — Image magic bytes (real PIL, no mocks)
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


async def test_image_open_fails_before_detected_format_assigned(tmp_path):
    """
    Regression: if Image.open() raises in the second try block (full-load pass)
    before detected_format is assigned, the except clause must NOT raise a
    NameError.  The fallback initialisation (detected_format = ext) covers this.
    The file contains valid-enough bytes to pass verify() but fails on load().
    """
    # A JPEG with a valid SOI marker but truncated body: passes verify()
    # because verify() only checks the header, but PIL.load() fails.
    p = tmp_path / "truncated.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # SOI + truncated APP0

    with pytest.raises(HTTPException) as exc:
        await validate_file("truncated.jpg", len(p.read_bytes()), str(p))
    # Must be a clean 400, not a 500 NameError
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# 4. Layer 3 — Video magic bytes (mocked _probe_video_codec)
# ---------------------------------------------------------------------------


async def test_valid_video_content_passes(tmp_path):
    """Happy path: ffprobe returns a valid video stream."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake_mp4_bytes")

    with _mock_probe("h264"):
        result = await validate_file("clip.mp4", len(p.read_bytes()), str(p))

    assert result is True


async def test_ffprobe_unavailable_raises_400(tmp_path):
    """streams=None signals ffprobe is not installed → clear 400, not a silent fallback."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake_mp4_bytes")

    with patch(
        "app.core.file_validator._probe_video_codec",
        return_value=("", None),
    ):
        with pytest.raises(HTTPException) as exc:
            await validate_file("clip.mp4", len(p.read_bytes()), str(p))

    assert exc.value.status_code == 400
    assert "ffprobe" in exc.value.detail.lower()


async def test_video_no_streams_raises_400(tmp_path):
    """ffprobe ran but found no video stream (e.g. audio-only or corrupted container)."""
    p = tmp_path / "audio_only.mp4"
    p.write_bytes(b"fake_audio_bytes")

    with patch(
        "app.core.file_validator._probe_video_codec",
        return_value=("", []),
    ):
        with pytest.raises(HTTPException) as exc:
            await validate_file("audio_only.mp4", len(p.read_bytes()), str(p))

    assert exc.value.status_code == 400
    assert "no readable video stream" in exc.value.detail.lower()


async def test_av1_codec_raises_415(tmp_path):
    """AV1 is in _UNSUPPORTED_VIDEO_CODECS → should get a clear 415 with a conversion hint."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake_av1_bytes")

    with patch(
        "app.core.file_validator._probe_video_codec",
        return_value=("av1", [{"codec_name": "av1"}]),
    ):
        with pytest.raises(HTTPException) as exc:
            await validate_file("clip.mp4", len(p.read_bytes()), str(p))

    assert exc.value.status_code == 415
    assert "AV1" in exc.value.detail
    assert "H.264" in exc.value.detail  # conversion hint must be present


async def test_unknown_codec_with_no_streams_raises_400(tmp_path):
    """ffprobe returns non-zero exit code → empty streams list → generic 400."""
    p = tmp_path / "weird.mkv"
    p.write_bytes(b"garbage")

    with patch(
        "app.core.file_validator._probe_video_codec",
        return_value=("", []),
    ):
        with pytest.raises(HTTPException) as exc:
            await validate_file("weird.mkv", len(p.read_bytes()), str(p))

    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# 5. Security — path-leak prevention & Content-Length spoofing
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


async def test_actual_filesize_from_disk_used_for_size_check(tmp_path):
    """
    The size check must use os.path.getsize, not the caller-supplied value.
    Passing a huge caller-supplied filesize with a tiny real file should NOT
    raise 413 — the disk measurement overrides the spoofed value.
    """
    from app.config import settings

    path = _write_tiny_jpg(tmp_path)
    result = await validate_file("photo.jpg", settings.max_image_upload_bytes + 1, path)
    assert result is True


# ---------------------------------------------------------------------------
# 6. sanitize_log_message — regex correctness
# ---------------------------------------------------------------------------


def test_sanitize_strips_tmp_tempfile_path():
    msg = "PIL error: /tmp/tmpABCDEF is truncated"
    sanitized = sanitize_log_message(msg)
    assert "[TEMP_FILE]" in sanitized
    assert "/tmp/tmpABCDEF" not in sanitized


def test_sanitize_strips_absolute_server_path():
    msg = "cannot open /app/core/file_validator.py"
    sanitized = sanitize_log_message(msg)
    assert "/app/core/file_validator.py" not in sanitized
    assert "file_validator.py" in sanitized  # filename is preserved


def test_sanitize_strips_deep_absolute_path():
    msg = "error at /home/user/myapp/uploads/photo.jpg line 42"
    sanitized = sanitize_log_message(msg)
    assert "/home/user/myapp/uploads" not in sanitized
    assert "photo.jpg" in sanitized


def test_sanitize_does_not_mangle_mime_types():
    """MIME type slashes must not be touched (slash is preceded by a word char)."""
    msg = "MIME type mismatch: expected application/json but got video/mp4"
    assert sanitize_log_message(msg) == msg


def test_sanitize_does_not_mangle_fractions():
    """Mathematical fractions like 4/3 must not be altered."""
    msg = "Expected aspect ratio 16/9 but got 4/3"
    assert sanitize_log_message(msg) == msg


def test_sanitize_keeps_plain_messages_intact():
    msg = "No issues found in this file"
    assert sanitize_log_message(msg) == msg


def test_sanitize_handles_empty_string():
    assert sanitize_log_message("") == ""
