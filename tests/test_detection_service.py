"""
Pure unit tests for app/services/detection_service.py.

Tests: _generate_short_id, download_image (base64 + HTTP).
aiohttp is mocked so no real network calls are made.
"""

import base64
import io
import string
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from app.config import settings
from app.services.detection_service import _generate_short_id, download_image


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
# download_image — base64 data URIs
# ---------------------------------------------------------------------------


def _make_png_data_uri() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


async def test_download_image_base64_png():
    uri = _make_png_data_uri()
    content, name = await download_image(uri)

    assert isinstance(content, bytes)
    assert len(content) > 0
    assert name == "pasted_image.png"


async def test_download_image_base64_too_large():
    # Build a data URI whose decoded size exceeds 1 byte (use max_size=1)
    uri = _make_png_data_uri()
    with pytest.raises(HTTPException) as exc:
        await download_image(uri, max_size=1)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# download_image — HTTP URLs (aiohttp mocked)
# ---------------------------------------------------------------------------


def _make_mock_session(status=200, content=b"image_bytes", content_type="image/jpeg"):
    """Build a mock aiohttp session whose .get() returns a context-manager response."""
    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)
    mock_resp.status = status
    mock_resp.read = AsyncMock(return_value=content)
    mock_resp.headers = {"Content-Type": content_type}

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    return mock_session


def _patch_session(mock_session):
    """
    Patch http_client.request_session to yield mock_session directly,
    bypassing aiohttp.ClientSession construction entirely.
    """
    @asynccontextmanager
    async def _fake_request_session():
        yield mock_session

    return patch(
        "app.integrations.http_client.request_session",
        side_effect=_fake_request_session,
    )


async def test_download_image_url_success():
    session = _make_mock_session(status=200, content=b"fake_img", content_type="image/jpeg")
    with _patch_session(session):
        content, name = await download_image("https://example.com/photo.jpg")

    assert content == b"fake_img"
    assert name.endswith(".jpg")


async def test_download_image_url_not_found():
    session = _make_mock_session(status=404)
    with _patch_session(session):
        with pytest.raises(HTTPException) as exc:
            await download_image("https://example.com/missing.jpg")
    assert exc.value.status_code == 400


async def test_download_image_url_too_large():
    big = b"X" * 10
    session = _make_mock_session(status=200, content=big, content_type="image/jpeg")
    with _patch_session(session):
        with pytest.raises(HTTPException) as exc:
            await download_image("https://example.com/big.jpg", max_size=5)
    assert exc.value.status_code == 400


async def test_download_image_url_client_error():
    import aiohttp

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=aiohttp.ClientError("connection refused"))

    with _patch_session(mock_session):
        with pytest.raises(HTTPException) as exc:
            await download_image("https://unreachable.example.com/img.jpg")
    assert exc.value.status_code == 400
