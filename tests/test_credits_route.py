"""
Tests for:
  GET  /api/user/balance
  POST /api/credits/add
  GET  /api/credits/webhook
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

DEVICE_ID = "credit-test-device"
SECRET = "supersecret"


# ---------------------------------------------------------------------------
# Balance endpoint
# ---------------------------------------------------------------------------


def test_get_balance_existing_wallet(client):
    with (
        patch("app.api.credits.check_ip_device_limit", new_callable=AsyncMock),
        patch("app.api.credits.get_guest_wallet", return_value={"credits": 42}),
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
            return_value={"credits": settings.welcome_credits},
        ),
    ):
        response = client.get(
            "/api/user/balance", headers={"X-Device-ID": "brand-new-device"}
        )
    assert response.status_code == 200
    assert response.json()["balance"] == settings.welcome_credits


# ---------------------------------------------------------------------------
# POST /api/credits/add
# ---------------------------------------------------------------------------


def test_add_credits_post_valid(client):
    with (
        patch(
            "app.api.credits.perform_recharge",
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
