"""
Shared pytest fixtures for all test modules.

IMPORTANT: TESTING must be set before the app is imported so the lifespan
skips the asyncio background cleanup task.
"""

import io
import os

os.environ["TESTING"] = "true"
# Gemini client is instantiated at module import time; a non-empty stub prevents
# the SDK from raising ValueError before our mocks are in place.
# Real API calls never happen in tests — all Gemini functions are mocked.
os.environ.setdefault("GEMINI_API_KEY", "stub-key-for-tests")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.mocks.firebase_mock import MockFirestore
from tests.mocks.redis_mock import MockRedis

# App import happens AFTER os.environ["TESTING"] is set above.
from app.main import app  # noqa: E402


# ---------------------------------------------------------------------------
# Core infrastructure fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_firebase(monkeypatch):
    """Replace firebase.db with an in-memory MockFirestore."""
    from app.integrations import firebase as fb

    mock_db = MockFirestore()
    monkeypatch.setattr(fb, "db", mock_db)
    return mock_db


@pytest.fixture
def mock_redis(monkeypatch):
    """Replace redis_client.client with an in-memory MockRedis."""
    from app.integrations import redis_client as rc

    mock_rc = MockRedis()
    monkeypatch.setattr(rc, "client", mock_rc)
    return mock_rc


@pytest.fixture
def client(mock_firebase, mock_redis):
    """
    FastAPI TestClient with mocked Firebase and Redis.

    initialize() calls are patched to no-ops so they can't overwrite our mocks
    or attempt real network connections during the lifespan startup.
    """
    with (
        patch("app.integrations.firebase.initialize"),
        patch("app.integrations.redis_client.initialize"),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Shared test-data helpers
# ---------------------------------------------------------------------------


def make_tiny_jpeg() -> bytes:
    """Create a minimal 10×10 JPEG in memory — fast and valid."""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tiny_jpg(tmp_path) -> str:
    """Write a tiny JPEG to a temp file and return the path."""
    p = tmp_path / "test.jpg"
    p.write_bytes(make_tiny_jpeg())
    return str(p)


MOCK_DETECT_RESULT = {
    "summary": "No AI Detected",
    "confidence_score": 0.95,
    "is_short_circuited": True,
    "evidence_chain": [
        {
            "layer": "Metadata Check",
            "status": "passed",
            "label": "Device Metadata",
            "detail": "Valid camera metadata found.",
        }
    ],
    "is_gemini_used": False,
    "is_cached": False,
    "gpu_time_ms": 0,
}
