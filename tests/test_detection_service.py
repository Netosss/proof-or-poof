"""
Pure unit tests for app/services/detection_service.py.

Tests: _generate_short_id, download_media_to_disk (base64 + HTTPS).
aiohttp is mocked so no real network calls are made.
"""

import base64
import io
import os
import string
import tempfile
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from app.config import settings
from app.services.detection_service import (
    _generate_short_id,
    _suffix_from_content_type,
    download_media_to_disk,
)


# ---------------------------------------------------------------------------
# _generate_short_id
# ---------------------------------------------------------------------------


def test_generate_short_id_correct_length():
    sid = _generate_short_id()
    assert len(sid) == settings.short_id_length


def test_generate_short_id_alphanumeric_only():
    allowed = set(string.ascii_letters + string.digits)
    for _ in range(20):
        sid = _generate_short_id()
        assert set(sid).issubset(allowed)


def test_generate_short_id_unique():
    ids = {_generate_short_id() for _ in range(100)}
    # Extremely unlikely to collide in 100 tries (62^8 space)
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_data_uri() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_mock_session(status=200, content=b"image_bytes", content_type="image/jpeg"):
    """
    Build a mock aiohttp session whose .get() returns a context-manager response.

    response.content.iter_chunked() is an async generator that yields the
    content in a single chunk, matching the streaming read path in
    download_media_to_disk.
    """
    async def _iter_chunked(_chunk_size):
        if content:
            yield content

    mock_content = MagicMock()
    mock_content.iter_chunked = _iter_chunked

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)
    mock_resp.status = status
    mock_resp.content = mock_content
    mock_resp.headers = {"Content-Type": content_type}

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    return mock_session


def _patch_session(mock_session):
    """Patch http_client.request_session to yield mock_session directly."""
    @asynccontextmanager
    async def _fake_request_session():
        yield mock_session

    return patch(
        "app.integrations.http_client.request_session",
        side_effect=_fake_request_session,
    )


def _patch_ssrf_check():
    """Bypass the SSRF DNS check — tests use mocked sessions, no real network."""
    return patch(
        "app.services.detection_service._assert_url_safe",
        new_callable=AsyncMock,
    )


# ---------------------------------------------------------------------------
# _suffix_from_content_type — Content-Type header → file extension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content_type,expected", [
    # --- Images ---
    ("image/jpeg",                  ".jpg"),
    ("image/jpg",                   ".jpg"),
    ("image/png",                   ".png"),
    ("image/webp",                  ".webp"),
    ("image/gif",                   ".gif"),
    ("image/heic",                  ".heic"),
    ("image/heif",                  ".heif"),
    ("image/tiff",                  ".tiff"),
    ("image/bmp",                   ".bmp"),
    # --- Videos ---
    ("video/mp4",                   ".mp4"),
    ("video/quicktime",             ".mov"),
    ("video/webm",                  ".webm"),
    # --- Generic / unknown → fallback to .jpg ---
    ("application/octet-stream",    ".jpg"),   # no URL hint
    ("",                            ".jpg"),
    ("text/html",                   ".jpg"),
])
def test_suffix_from_content_type_known_types(content_type, expected):
    assert _suffix_from_content_type(content_type, "") == expected


@pytest.mark.parametrize("url,expected", [
    ("https://cdn.example.com/clip.mp4",    ".mp4"),
    ("https://cdn.example.com/film.mov",    ".mov"),
    ("https://cdn.example.com/image.webp",  ".webp"),
    ("https://cdn.example.com/photo.png",   ".png"),
    # Query-string stripped before extension matching
    ("https://cdn.example.com/video.mp4?token=abc", ".mp4"),
])
def test_suffix_from_content_type_url_path_fallback(url, expected):
    """When Content-Type is generic, the URL path extension is used."""
    assert _suffix_from_content_type("application/octet-stream", url) == expected


def test_suffix_from_content_type_url_fallback_unknown_defaults_to_jpg():
    """URL with an unrecognised extension falls back to .jpg."""
    assert _suffix_from_content_type("application/octet-stream",
                                     "https://cdn.example.com/data.bin") == ".jpg"


# ---------------------------------------------------------------------------
# download_media_to_disk — base64 data URIs
# ---------------------------------------------------------------------------


async def test_download_media_to_disk_base64_png(tmp_path):
    dest = str(tmp_path / "out.tmp")
    uri = _make_png_data_uri()
    name = await download_media_to_disk(uri, dest)

    assert name == "pasted_image.png"
    assert os.path.getsize(dest) > 0


async def test_download_media_to_disk_base64_content_on_disk(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    uri = f"data:image/png;base64,{b64}"
    dest = str(tmp_path / "out.tmp")

    await download_media_to_disk(uri, dest)

    with open(dest, "rb") as f:
        assert f.read() == raw


async def test_download_media_to_disk_base64_too_large(tmp_path):
    dest = str(tmp_path / "out.tmp")
    uri = _make_png_data_uri()
    with pytest.raises(HTTPException) as exc:
        await download_media_to_disk(uri, dest, max_size=1)
    assert exc.value.status_code == 400


async def test_download_media_to_disk_base64_no_base64_header_rejected(tmp_path):
    """Data URIs without ;base64 flag must be rejected."""
    dest = str(tmp_path / "out.tmp")
    with pytest.raises(HTTPException) as exc:
        await download_media_to_disk("data:image/png,notbase64data", dest)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# download_media_to_disk — HTTPS enforcement
# ---------------------------------------------------------------------------


async def test_download_media_to_disk_http_url_rejected(tmp_path):
    """Plain HTTP URLs must be rejected (HTTPS-only enforcement)."""
    dest = str(tmp_path / "out.tmp")
    with pytest.raises(HTTPException) as exc:
        await download_media_to_disk("http://example.com/photo.jpg", dest)
    assert exc.value.status_code == 400
    assert "HTTPS" in exc.value.detail


async def test_download_media_to_disk_ftp_url_rejected(tmp_path):
    dest = str(tmp_path / "out.tmp")
    with pytest.raises(HTTPException) as exc:
        await download_media_to_disk("ftp://example.com/file.jpg", dest)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# download_media_to_disk — SSRF protection
# ---------------------------------------------------------------------------


async def test_download_media_to_disk_ssrf_blocked(tmp_path):
    """Internal / private URLs must be rejected before any HTTP request."""
    dest = str(tmp_path / "out.tmp")
    for bad_url in [
        "http://169.254.169.254/latest/meta-data/",   # AWS metadata — also fails HTTPS check
        "http://localhost/admin",
        "http://127.0.0.1:6379/",
        "ftp://example.com/file.jpg",
    ]:
        with pytest.raises(HTTPException) as exc:
            await download_media_to_disk(bad_url, dest)
        assert exc.value.status_code == 400, f"Expected 400 for {bad_url}"


# ---------------------------------------------------------------------------
# download_media_to_disk — HTTPS URL (aiohttp mocked)
# ---------------------------------------------------------------------------


async def test_download_media_to_disk_url_success(tmp_path):
    dest = str(tmp_path / "out.tmp")
    session = _make_mock_session(status=200, content=b"fake_img", content_type="image/jpeg")
    with _patch_session(session), _patch_ssrf_check():
        name = await download_media_to_disk("https://example.com/photo.jpg", dest)

    assert name.endswith(".jpg")
    with open(dest, "rb") as f:
        assert f.read() == b"fake_img"


async def test_download_media_to_disk_url_video_filename(tmp_path):
    dest = str(tmp_path / "out.tmp")
    session = _make_mock_session(status=200, content=b"fake_vid", content_type="video/mp4")
    with _patch_session(session), _patch_ssrf_check():
        name = await download_media_to_disk("https://example.com/clip.mp4", dest)

    assert name == "downloaded_media.mp4"


async def test_download_media_to_disk_url_not_found(tmp_path):
    dest = str(tmp_path / "out.tmp")
    session = _make_mock_session(status=404)
    with _patch_session(session), _patch_ssrf_check():
        with pytest.raises(HTTPException) as exc:
            await download_media_to_disk("https://example.com/missing.jpg", dest)
    assert exc.value.status_code == 400


async def test_download_media_to_disk_url_too_large(tmp_path):
    dest = str(tmp_path / "out.tmp")
    big = b"X" * 10
    session = _make_mock_session(status=200, content=big, content_type="image/jpeg")
    with _patch_session(session), _patch_ssrf_check():
        with pytest.raises(HTTPException) as exc:
            await download_media_to_disk("https://example.com/big.jpg", dest, max_size=5)
    assert exc.value.status_code == 400


async def test_download_media_to_disk_url_client_error(tmp_path):
    import aiohttp
    dest = str(tmp_path / "out.tmp")
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=aiohttp.ClientError("connection refused"))

    with _patch_session(mock_session), _patch_ssrf_check():
        with pytest.raises(HTTPException) as exc:
            await download_media_to_disk("https://unreachable.example.com/img.jpg", dest)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# download_media_to_disk — redirect blocking
# ---------------------------------------------------------------------------


async def test_download_media_to_disk_redirect_blocked(tmp_path):
    """3xx responses must be rejected to prevent redirect-chain SSRF."""
    dest = str(tmp_path / "out.tmp")
    for redirect_code in (301, 302, 303, 307, 308):
        session = _make_mock_session(status=redirect_code, content=b"")
        with _patch_session(session), _patch_ssrf_check():
            with pytest.raises(HTTPException) as exc:
                await download_media_to_disk("https://example.com/photo.jpg", dest)
        assert exc.value.status_code == 400, (
            f"Expected 400 for redirect {redirect_code}"
        )
        assert "Redirect" in exc.value.detail or "redirect" in exc.value.detail.lower()
