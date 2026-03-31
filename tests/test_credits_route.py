"""
Tests for:
  GET  /api/user/balance  (guest path AND authenticated path)
  POST /api/credits/add
  GET  /api/credits/webhook
"""

import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

DEVICE_ID = "credit-test-device"
SECRET = "supersecret"
AUTH_UID = "auth-user-001"
AUTH_USER = {"uid": AUTH_UID, "email": "auth@example.com"}


@contextmanager
def _override_auth_user(user: dict | None):
    """Temporarily override get_optional_user dependency on the FastAPI app."""
    from app.main import app
    from app.core.firebase_auth import get_optional_user

    async def _fake():
        return user

    app.dependency_overrides[get_optional_user] = _fake
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_optional_user, None)


# ---------------------------------------------------------------------------
# Balance endpoint — guest path
# ---------------------------------------------------------------------------


def test_get_balance_existing_wallet(client):
    with (
        patch("app.api.credits.check_ip_device_limit", new_callable=AsyncMock),
        patch("app.api.credits.get_guest_wallet", new_callable=AsyncMock, return_value={"credits": 42}),
    ):
        response = client.get(
            "/api/user/balance", headers={"X-Device-ID": DEVICE_ID}
        )
    assert response.status_code == 200
    assert response.json() == {"balance": 42}


def test_get_balance_new_device_gets_welcome_credits(client):
    from app.config import settings

    with (
        patch("app.api.credits.check_ip_device_limit", new_callable=AsyncMock),
        patch(
            "app.api.credits.get_guest_wallet",
            new_callable=AsyncMock,
            return_value={"credits": settings.welcome_credits},
        ),
    ):
        response = client.get(
            "/api/user/balance", headers={"X-Device-ID": "brand-new-device"}
        )
    assert response.status_code == 200
    assert response.json()["balance"] == settings.welcome_credits


def test_get_balance_no_device_id_no_auth_returns_error(client):
    """Guest path with no X-Device-ID and no auth token must be rejected."""
    with _override_auth_user(None):
        response = client.get("/api/user/balance")
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Balance endpoint — authenticated path
# ---------------------------------------------------------------------------


def test_get_balance_authenticated_returns_user_balance(client):
    """Authenticated user gets balance from users/{uid}, not guest_wallets."""
    with _override_auth_user(AUTH_USER):
        with (
            patch("app.api.credits.check_rate_limit", new_callable=AsyncMock),
            patch(
                "app.api.credits.get_user_balance",
                new_callable=AsyncMock,
                return_value=123,
            ),
        ):
            response = client.get(
                "/api/user/balance",
                headers={"Authorization": "Bearer fake_token"},
            )
    assert response.status_code == 200
    assert response.json() == {"balance": 123}


def test_get_balance_authenticated_no_device_id_needed(client):
    """Authenticated user does not need X-Device-ID header."""
    with _override_auth_user(AUTH_USER):
        with (
            patch("app.api.credits.check_rate_limit", new_callable=AsyncMock),
            patch(
                "app.api.credits.get_user_balance",
                new_callable=AsyncMock,
                return_value=50,
            ),
        ):
            # No X-Device-ID header at all
            response = client.get("/api/user/balance")
    assert response.status_code == 200
    assert response.json() == {"balance": 50}


def test_get_balance_authenticated_uses_uid_rate_limit(client):
    """Authenticated path calls check_rate_limit with uid-scoped key."""
    mock_rate_limit = AsyncMock()
    with _override_auth_user(AUTH_USER):
        with (
            patch("app.api.credits.check_rate_limit", mock_rate_limit),
            patch("app.api.credits.get_user_balance", new_callable=AsyncMock, return_value=0),
        ):
            client.get("/api/user/balance")
    mock_rate_limit.assert_called_once_with(f"balance:{AUTH_UID}")


def test_get_balance_authenticated_does_not_call_guest_wallet(client):
    """Authenticated path must never touch the guest wallet service."""
    mock_guest_wallet = AsyncMock()
    with _override_auth_user(AUTH_USER):
        with (
            patch("app.api.credits.check_rate_limit", new_callable=AsyncMock),
            patch("app.api.credits.get_user_balance", new_callable=AsyncMock, return_value=0),
            patch("app.api.credits.get_guest_wallet", mock_guest_wallet),
        ):
            client.get("/api/user/balance")
    mock_guest_wallet.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/credits/add
# ---------------------------------------------------------------------------


def test_add_credits_post_valid(client):
    with (
        patch(
            "app.api.credits.perform_recharge",
            new_callable=AsyncMock,
            return_value={"status": "success", "new_balance": 15},
        ),
        patch("app.api.credits.log_transaction"),
    ):
        response = client.post(
            "/api/credits/add",
            json={"device_id": DEVICE_ID, "amount": 5, "secret_key": SECRET},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["new_balance"] == 15


def test_add_credits_post_wrong_secret(client):
    with patch(
        "app.api.credits.perform_recharge",
        side_effect=__import__("fastapi").HTTPException(
            status_code=403, detail="Invalid secret key"
        ),
    ):
        response = client.post(
            "/api/credits/add",
            json={"device_id": DEVICE_ID, "amount": 5, "secret_key": "wrong"},
        )
    assert response.status_code == 403


def test_add_credits_post_db_unavailable(client):
    with patch(
        "app.api.credits.perform_recharge",
        side_effect=__import__("fastapi").HTTPException(
            status_code=503, detail="Database service unavailable."
        ),
    ):
        response = client.post(
            "/api/credits/add",
            json={"device_id": DEVICE_ID, "amount": 5, "secret_key": SECRET},
        )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/credits/webhook
# ---------------------------------------------------------------------------


def test_add_credits_webhook_valid(client):
    with (
        patch(
            "app.api.credits.perform_recharge",
            new_callable=AsyncMock,
            return_value={"status": "success", "new_balance": 10},
        ),
        patch("app.api.credits.log_transaction"),
    ):
        response = client.get(
            f"/api/credits/webhook?device_id={DEVICE_ID}&secret_key={SECRET}"
        )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_add_credits_webhook_wrong_key(client):
    with patch(
        "app.api.credits.perform_recharge",
        side_effect=__import__("fastapi").HTTPException(
            status_code=403, detail="Invalid secret key"
        ),
    ):
        response = client.get(
            f"/api/credits/webhook?device_id={DEVICE_ID}&secret_key=bad"
        )
    assert response.status_code == 403
