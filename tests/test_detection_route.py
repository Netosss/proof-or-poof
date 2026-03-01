"""
Tests for POST /detect.

All external calls are mocked: Turnstile, IP-device limit, detection pipeline,
file validation, Firebase (credits), and finance logging.
"""

import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import MOCK_DETECT_RESULT, make_tiny_jpeg

DEVICE_ID = "test-device-001"
HEADERS = {"X-Device-ID": DEVICE_ID, "X-Turnstile-Token": "tok_valid"}

WALLET_ENOUGH = {"credits": 100, "is_banned": False}
WALLET_LOW = {"credits": 2, "is_banned": False}
WALLET_BANNED = {"credits": 100, "is_banned": True}


def _patches(
    verify_result=True,
    ban=False,
    wallet=None,
    detect_result=None,
    deduct_result=90,
):
    """Return an ExitStack with all common detection route patches applied."""
    if wallet is None:
        # When ban=True use WALLET_BANNED so the route's is_banned check fires.
        wallet = WALLET_BANNED if ban else WALLET_ENOUGH
    if detect_result is None:
        detect_result = dict(MOCK_DETECT_RESULT)

    stack = ExitStack()
    stack.enter_context(
        patch("app.api.detection.verify_turnstile", new_callable=AsyncMock, return_value=verify_result)
    )
    stack.enter_context(
        patch("app.api.detection.check_ip_device_limit", new_callable=AsyncMock)
    )
    stack.enter_context(
        patch("app.api.detection.get_guest_wallet", return_value=wallet)
    )
    stack.enter_context(
        patch("app.api.detection.deduct_guest_credits", return_value=deduct_result)
    )
    stack.enter_context(
        patch("app.api.detection.detect_ai_media", new_callable=AsyncMock, return_value=detect_result)
    )
    stack.enter_context(patch("app.api.detection.log_transaction"))
    from app.core.dependencies import security_manager
    stack.enter_context(patch.object(security_manager, "validate_file", return_value=True))
    stack.enter_context(patch.object(security_manager, "check_rate_limit"))
    return stack


# ---------------------------------------------------------------------------
# Auth / access-control cases
# ---------------------------------------------------------------------------


def test_detect_missing_turnstile_token(client):
    response = client.post(
        "/detect",
        headers={"X-Device-ID": DEVICE_ID},  # no X-Turnstile-Token
        json={"url": "https://example.com/img.jpg"},
    )
    assert response.status_code == 403
    assert "CAPTCHA_REQUIRED" in str(response.json())


def test_detect_invalid_turnstile(client):
    with _patches(verify_result=False):
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 403


def test_detect_banned_device(client):
    with _patches(ban=True):
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 403


def test_detect_insufficient_credits(client):
    with _patches(wallet=WALLET_LOW):
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 402


# ---------------------------------------------------------------------------
# Successful detection cases
# ---------------------------------------------------------------------------


def test_detect_multipart_file_upload(client):
    with _patches():
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "short_id" in data
    assert "new_balance" in data
    assert data["summary"] == "No AI Detected"


def test_detect_json_url_upload(client):
    with _patches():
        with patch(
            "app.api.detection.download_image",
            new_callable=AsyncMock,
            return_value=(make_tiny_jpeg(), "photo.jpg"),
        ):
            response = client.post(
                "/detect",
                headers={**HEADERS, "content-type": "application/json"},
                content=json.dumps({"url": "https://example.com/photo.jpg"}),
            )
    assert response.status_code == 200
    assert "short_id" in response.json()


# ---------------------------------------------------------------------------
# Input validation cases
# ---------------------------------------------------------------------------


def test_detect_multipart_missing_file_and_url(client):
    # Use a wrong field name to trigger multipart/form-data without file or url
    with _patches():
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"wrong_field": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 400


def test_detect_unsupported_content_type(client):
    with _patches():
        response = client.post(
            "/detect",
            headers={**HEADERS, "content-type": "text/plain"},
            content=b"raw text",
        )
    assert response.status_code == 415


# ---------------------------------------------------------------------------
# Billing edge-cases: soft failures do NOT deduct credits
# ---------------------------------------------------------------------------


def test_detect_analysis_failed_no_deduction(client):
    failed_result = {
        **MOCK_DETECT_RESULT,
        "summary": "Analysis Failed",
        "confidence_score": 0.0,
    }
    with _patches(detect_result=failed_result) as stack:
        mock_deduct = MagicMock(return_value=90)
        stack.enter_context(
            patch("app.api.detection.deduct_guest_credits", mock_deduct)
        )
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 200
    mock_deduct.assert_not_called()


def test_detect_file_too_large_no_deduction(client):
    too_large_result = {
        **MOCK_DETECT_RESULT,
        "summary": "File too large to scan",
    }
    with _patches(detect_result=too_large_result) as stack:
        mock_deduct = MagicMock(return_value=90)
        stack.enter_context(
            patch("app.api.detection.deduct_guest_credits", mock_deduct)
        )
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 200
    mock_deduct.assert_not_called()


def test_detect_redis_unavailable_still_returns_result(client, monkeypatch):
    """When Redis is None, short_id is still generated but not cached."""
    from app.integrations import redis_client as rc

    monkeypatch.setattr(rc, "client", None)

    with _patches():
        response = client.post(
            "/detect",
            headers=HEADERS,
            files={"file": ("photo.jpg", make_tiny_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json()["summary"] == "No AI Detected"
