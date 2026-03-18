"""
Tests for POST /inpaint/image.

Mocks: check_ip_device_limit, run_gpu_inpainting, Firebase (wallet/ban), Redis (op_ref token).
"""

import uuid
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_tiny_jpeg

DEVICE_ID = "inpaint-device-001"
HEADERS = {"X-Device-ID": DEVICE_ID, "X-Turnstile-Token": "tok_valid"}
WALLET_OK = {"credits": 50, "is_banned": False}
WALLET_LOW = {"credits": 0, "is_banned": False}
WALLET_BANNED = {"credits": 50, "is_banned": True}
FAKE_RESULT = b"PNG_RESULT_BYTES"


def _make_files():
    return {
        "image": ("img.jpg", make_tiny_jpeg(), "image/jpeg"),
        "mask": ("mask.jpg", make_tiny_jpeg(), "image/jpeg"),
    }


def _patches(turnstile_ok=True, wallet=None):
    """Return an ExitStack with the standard inpainting route patches applied."""
    if wallet is None:
        wallet = WALLET_OK
    stack = ExitStack()
    stack.enter_context(
        patch("app.api.inpainting.verify_turnstile", new_callable=AsyncMock, return_value=turnstile_ok)
    )
    stack.enter_context(patch("app.api.inpainting.check_ip_device_limit", new_callable=AsyncMock))
    stack.enter_context(patch("app.api.inpainting.get_guest_wallet", new_callable=AsyncMock, return_value=wallet))
    return stack


# ---------------------------------------------------------------------------
# Turnstile / auth cases
# ---------------------------------------------------------------------------


def test_inpaint_missing_turnstile(client):
    response = client.post(
        "/inpaint/image",
        headers={"X-Device-ID": DEVICE_ID},  # no X-Turnstile-Token
        files=_make_files(),
    )
    assert response.status_code == 403
    assert "CAPTCHA_REQUIRED" in str(response.json())


def test_inpaint_invalid_turnstile(client):
    with _patches(turnstile_ok=False):
        response = client.post("/inpaint/image", headers=HEADERS, files=_make_files())
    assert response.status_code == 403


def test_inpaint_banned_device(client):
    with _patches(wallet=WALLET_BANNED):
        response = client.post("/inpaint/image", headers=HEADERS, files=_make_files())
    assert response.status_code == 403


def test_inpaint_insufficient_credits(client):
    with _patches(wallet=WALLET_LOW):
        response = client.post("/inpaint/image", headers=HEADERS, files=_make_files())
    assert response.status_code == 402


# ---------------------------------------------------------------------------
# Success / billing cases
# ---------------------------------------------------------------------------


def test_inpaint_success_deducts_credits(client):
    with _patches(wallet=WALLET_OK) as stack:
        stack.enter_context(
            patch("app.api.inpainting.run_gpu_inpainting", new_callable=AsyncMock, return_value=FAKE_RESULT)
        )
        stack.enter_context(patch("app.api.inpainting.deduct_guest_credits", new_callable=AsyncMock, return_value=20))
        stack.enter_context(patch("app.api.inpainting.log_transaction"))
        response = client.post("/inpaint/image", headers=HEADERS, files=_make_files())
    assert response.status_code == 200
    assert response.content == FAKE_RESULT
    assert response.headers["x-user-balance"] == "20"


def test_inpaint_paid_success_returns_op_ref_header(client):
    """A paid inpainting response must include X-Op-Ref for the next free retry."""
    with _patches(wallet=WALLET_OK) as stack:
        stack.enter_context(
            patch("app.api.inpainting.run_gpu_inpainting", new_callable=AsyncMock, return_value=FAKE_RESULT)
        )
        stack.enter_context(patch("app.api.inpainting.deduct_guest_credits", new_callable=AsyncMock, return_value=20))
        stack.enter_context(patch("app.api.inpainting.log_transaction"))
        response = client.post("/inpaint/image", headers=HEADERS, files=_make_files())
    assert response.status_code == 200
    assert "x-op-ref" in response.headers
    # Value should be a valid UUID
    uuid.UUID(response.headers["x-op-ref"])


def test_inpaint_free_retry_skips_deduction(client, mock_redis):
    """If a valid op_ref token exists in Redis the user gets a free retry."""
    token = str(uuid.uuid4())
    cache_key = f"op_ref:{DEVICE_ID}:{token}"
    mock_redis.seed(cache_key, "1")

    with _patches(wallet=WALLET_LOW) as stack:
        stack.enter_context(
            patch("app.api.inpainting.run_gpu_inpainting", new_callable=AsyncMock, return_value=FAKE_RESULT)
        )
        mock_deduct = MagicMock()
        stack.enter_context(patch("app.api.inpainting.deduct_guest_credits", mock_deduct))
        stack.enter_context(patch("app.api.inpainting.log_transaction"))
        response = client.post(
            "/inpaint/image",
            headers={**HEADERS, "X-Op-Ref": token},
            files=_make_files(),
        )

    assert response.status_code == 200
    mock_deduct.assert_not_called()
    # Free retry must not return a new X-Op-Ref
    assert "x-op-ref" not in response.headers


def test_inpaint_op_ref_consumed_on_free_retry(client, mock_redis):
    """The Redis op_ref key must be deleted after use (no double-spend)."""
    token = str(uuid.uuid4())
    cache_key = f"op_ref:{DEVICE_ID}:{token}"
    mock_redis.seed(cache_key, "1")

    with _patches(wallet=WALLET_LOW) as stack:
        stack.enter_context(
            patch("app.api.inpainting.run_gpu_inpainting", new_callable=AsyncMock, return_value=FAKE_RESULT)
        )
        stack.enter_context(patch("app.api.inpainting.deduct_guest_credits", new_callable=AsyncMock, return_value=0))
        stack.enter_context(patch("app.api.inpainting.log_transaction"))
        client.post(
            "/inpaint/image",
            headers={**HEADERS, "X-Op-Ref": token},
            files=_make_files(),
        )

    assert mock_redis._store.get(cache_key) is None


def test_inpaint_stale_op_ref_charges_normally(client, mock_redis):
    """An X-Op-Ref with no matching Redis key is treated as a normal paid call."""
    stale_token = str(uuid.uuid4())
    # Do NOT seed the key — simulates an expired or already-used token

    with _patches(wallet=WALLET_OK) as stack:
        stack.enter_context(
            patch("app.api.inpainting.run_gpu_inpainting", new_callable=AsyncMock, return_value=FAKE_RESULT)
        )
        mock_deduct = AsyncMock(return_value=20)
        stack.enter_context(patch("app.api.inpainting.deduct_guest_credits", mock_deduct))
        stack.enter_context(patch("app.api.inpainting.log_transaction"))
        response = client.post(
            "/inpaint/image",
            headers={**HEADERS, "X-Op-Ref": stale_token},
            files=_make_files(),
        )

    assert response.status_code == 200
    mock_deduct.assert_called_once()
    # A new op_ref should be issued for the paid call
    assert "x-op-ref" in response.headers


def test_inpaint_gpu_failure_returns_500(client):
    with _patches(wallet=WALLET_OK) as stack:
        stack.enter_context(
            patch(
                "app.api.inpainting.run_gpu_inpainting",
                new_callable=AsyncMock,
                side_effect=RuntimeError("GPU timeout"),
            )
        )
        stack.enter_context(patch("app.api.inpainting.log_transaction"))
        response = client.post("/inpaint/image", headers=HEADERS, files=_make_files())
    assert response.status_code == 500
