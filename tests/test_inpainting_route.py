"""
Tests for POST /inpaint/image.

Mocks: check_ip_device_limit, run_gpu_inpainting, Firebase (wallet/ban), Redis (retry token).
"""

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_tiny_jpeg

DEVICE_ID = "inpaint-device-001"
HEADERS = {"X-Device-ID": DEVICE_ID}
WALLET_OK = {"credits": 50, "is_banned": False}
WALLET_LOW = {"credits": 0, "is_banned": False}
WALLET_BANNED = {"credits": 50, "is_banned": True}
FAKE_RESULT = b"PNG_RESULT_BYTES"


def test_inpaint_banned_device(client):
    with (
        patch("app.api.inpainting.check_ip_device_limit", new_callable=AsyncMock),
        patch("app.api.inpainting.check_ban_status", return_value=True),
    ):
        response = client.post(
            "/inpaint/image",
            headers=HEADERS,
            files={
                "image": ("img.jpg", make_tiny_jpeg(), "image/jpeg"),
                "mask": ("mask.jpg", make_tiny_jpeg(), "image/jpeg"),
            },
        )
    assert response.status_code == 403


def test_inpaint_insufficient_credits(client):
    with (
        patch("app.api.inpainting.check_ip_device_limit", new_callable=AsyncMock),
        patch("app.api.inpainting.check_ban_status", return_value=False),
        patch("app.api.inpainting.get_guest_wallet", return_value=WALLET_LOW),
    ):
        response = client.post(
            "/inpaint/image",
            headers=HEADERS,
            files={
                "image": ("img.jpg", make_tiny_jpeg(), "image/jpeg"),
                "mask": ("mask.jpg", make_tiny_jpeg(), "image/jpeg"),
            },
        )
    assert response.status_code == 402


def test_inpaint_success_deducts_credits(client):
    with (
        patch("app.api.inpainting.check_ip_device_limit", new_callable=AsyncMock),
        patch("app.api.inpainting.check_ban_status", return_value=False),
        patch("app.api.inpainting.get_guest_wallet", return_value=WALLET_OK),
        patch(
            "app.api.inpainting.run_gpu_inpainting",
            new_callable=AsyncMock,
            return_value=FAKE_RESULT,
        ),
        patch("app.api.inpainting.deduct_guest_credits", return_value=48),
        patch("app.api.inpainting.log_transaction"),
    ):
        response = client.post(
            "/inpaint/image",
            headers=HEADERS,
            files={
                "image": ("img.jpg", make_tiny_jpeg(), "image/jpeg"),
                "mask": ("mask.jpg", make_tiny_jpeg(), "image/jpeg"),
            },
        )
    assert response.status_code == 200
    assert response.content == FAKE_RESULT
    assert response.headers["x-user-balance"] == "48"


def test_inpaint_free_retry_skips_deduction(client, mock_redis):
    """If a paid_image key exists in Redis, the user gets a free retry."""
    image_bytes = make_tiny_jpeg()

    # Pre-compute the cache key that the route will compute
    import hashlib

    img_hash = hashlib.sha256(image_bytes).hexdigest()
    cache_key = f"paid_image:{DEVICE_ID}:{img_hash}"
    mock_redis.set(cache_key, "1")

    with (
        patch("app.api.inpainting.check_ip_device_limit", new_callable=AsyncMock),
        patch("app.api.inpainting.check_ban_status", return_value=False),
        patch("app.api.inpainting.get_guest_wallet", return_value=WALLET_LOW),
        patch(
            "app.api.inpainting.run_gpu_inpainting",
            new_callable=AsyncMock,
            return_value=FAKE_RESULT,
        ),
        patch("app.api.inpainting.deduct_guest_credits") as mock_deduct,
        patch("app.api.inpainting.log_transaction"),
    ):
        response = client.post(
            "/inpaint/image",
            headers=HEADERS,
            files={
                "image": ("img.jpg", image_bytes, "image/jpeg"),
                "mask": ("mask.jpg", make_tiny_jpeg(), "image/jpeg"),
            },
        )

    assert response.status_code == 200
    mock_deduct.assert_not_called()
    # Cache key should be deleted after free retry
    assert mock_redis.get(cache_key) is None


def test_inpaint_gpu_failure_returns_500(client):
    with (
        patch("app.api.inpainting.check_ip_device_limit", new_callable=AsyncMock),
        patch("app.api.inpainting.check_ban_status", return_value=False),
        patch("app.api.inpainting.get_guest_wallet", return_value=WALLET_OK),
        patch(
            "app.api.inpainting.run_gpu_inpainting",
            new_callable=AsyncMock,
            side_effect=RuntimeError("GPU worker timeout"),
        ),
        patch("app.api.inpainting.log_transaction"),
    ):
        response = client.post(
            "/inpaint/image",
            headers=HEADERS,
            files={
                "image": ("img.jpg", make_tiny_jpeg(), "image/jpeg"),
                "mask": ("mask.jpg", make_tiny_jpeg(), "image/jpeg"),
            },
        )
    assert response.status_code == 500
