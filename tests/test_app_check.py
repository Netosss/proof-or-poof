"""
Tests for Firebase App Check (mobile anti-abuse).

Covers app.core.app_check (verify_app_check, replay cap, passes_app_check_gate)
and the OR-gate wired into /detect. The website path (Cloudflare Turnstile) must
stay byte-identical — the web client never sends X-Firebase-AppCheck.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings
from app.core import app_check as ac
from tests.conftest import make_tiny_jpeg
from tests.test_detection_route import DEVICE_ID, _patches

APPCHECK_HEADERS = {"X-Device-ID": DEVICE_ID, "X-Firebase-AppCheck": "appcheck-tok"}


@pytest.fixture
def enforce_mode(monkeypatch):
    monkeypatch.setattr(settings, "app_check_mode", "enforce")


@pytest.fixture
def monitor_mode(monkeypatch):
    monkeypatch.setattr(settings, "app_check_mode", "monitor")


# ---------------------------------------------------------------------------
# verify_app_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_app_check_valid_token(mock_redis):
    with patch.object(
        ac.firebase_app_check, "verify_token", MagicMock(return_value={"app_id": "x"})
    ):
        assert await ac.verify_app_check("tok", endpoint="detect") is True


@pytest.mark.asyncio
async def test_verify_app_check_invalid_token(mock_redis):
    with patch.object(
        ac.firebase_app_check, "verify_token", MagicMock(side_effect=ValueError("bad token"))
    ):
        assert await ac.verify_app_check("tok", endpoint="detect") is False


@pytest.mark.asyncio
async def test_verify_app_check_empty_token():
    assert await ac.verify_app_check("", endpoint="detect") is False


# ---------------------------------------------------------------------------
# Replay cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_cap_blocks_after_max(mock_redis, monkeypatch):
    monkeypatch.setattr(settings, "app_check_replay_max", 3)
    with patch.object(
        ac.firebase_app_check, "verify_token", MagicMock(return_value={"app_id": "x"})
    ):
        results = [await ac.verify_app_check("same-token", endpoint="detect") for _ in range(4)]
    # First 3 uses pass; the 4th trips the cap even though the token is valid.
    assert results == [True, True, True, False]


@pytest.mark.asyncio
async def test_replay_cap_fails_open_without_redis():
    with patch.object(ac.redis_module, "client", None):
        with patch.object(
            ac.firebase_app_check, "verify_token", MagicMock(return_value={"app_id": "x"})
        ):
            # Redis down → replay counter is skipped, verification still gates.
            assert await ac.verify_app_check("tok", endpoint="detect") is True


# ---------------------------------------------------------------------------
# passes_app_check_gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_no_token_returns_false():
    # Web path: no header → gate never satisfied, no firebase/redis touched.
    assert await ac.passes_app_check_gate(None, endpoint="detect") is False


@pytest.mark.asyncio
async def test_gate_monitor_verifies_but_returns_false(monitor_mode):
    with patch.object(ac, "verify_app_check", AsyncMock(return_value=True)) as m:
        assert await ac.passes_app_check_gate("tok", endpoint="detect") is False
        m.assert_awaited_once()  # verified for metrics, but does not substitute


@pytest.mark.asyncio
async def test_gate_enforce_valid_returns_true(enforce_mode):
    with patch.object(ac, "verify_app_check", AsyncMock(return_value=True)):
        assert await ac.passes_app_check_gate("tok", endpoint="detect") is True


@pytest.mark.asyncio
async def test_gate_enforce_invalid_returns_false(enforce_mode):
    with patch.object(ac, "verify_app_check", AsyncMock(return_value=False)):
        assert await ac.passes_app_check_gate("tok", endpoint="detect") is False


# ---------------------------------------------------------------------------
# /detect route integration
# ---------------------------------------------------------------------------


def test_detect_app_check_enforce_skips_turnstile(client, enforce_mode):
    """Enforce mode + valid App Check token → no Turnstile needed (mobile path)."""
    with _patches() as stack:
        stack.enter_context(
            patch("app.api.detection.passes_app_check_gate", AsyncMock(return_value=True))
        )
        response = client.post(
            "/detect",
            headers=APPCHECK_HEADERS,  # note: NO X-Turnstile-Token
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 200


def test_detect_app_check_monitor_still_requires_turnstile(client, monitor_mode):
    """Monitor mode: App Check is logged but Turnstile is still required."""
    with _patches() as stack:
        stack.enter_context(
            patch("app.api.detection.passes_app_check_gate", AsyncMock(return_value=False))
        )
        response = client.post(
            "/detect",
            headers=APPCHECK_HEADERS,  # app check present, no turnstile
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 403
    assert "CAPTCHA_REQUIRED" in str(response.json())


def test_detect_web_path_unchanged_no_appcheck(client):
    """Web client (Turnstile, no App Check header) is unaffected → 200."""
    with _patches():
        response = client.post(
            "/detect",
            headers={"X-Device-ID": DEVICE_ID, "X-Turnstile-Token": "tok_valid"},
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 200
