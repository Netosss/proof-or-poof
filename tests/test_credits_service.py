"""
Pure unit tests for app/services/credits_service.py.

Firebase transactions are neutralised with a pass-through async_transactional
mock that executes the inner function directly with a MockAsyncTransaction.

All tests are async because the service functions are native async coroutines.
"""

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import HTTPException

os.environ.setdefault("RECHARGE_SECRET_KEY", "test-recharge-secret")

DEVICE = "svc-device-001"
SECRET = "test-recharge-secret"


def mock_async_transactional(func):
    """Pass-through replacement for @async_transactional (applied at call time)."""
    async def wrapper(transaction, *args, **kwargs):
        return await func(transaction, *args, **kwargs)
    return wrapper


@contextmanager
def _tx_patch():
    """Patch async_transactional wherever it is used in credits_service."""
    with patch(
        "app.services.credits_service.async_transactional",
        side_effect=mock_async_transactional,
    ):
        yield


# ---------------------------------------------------------------------------
# get_guest_wallet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_guest_wallet_existing(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("guest_wallets", DEVICE, {"credits": 42, "is_banned": False})

    from app.services.credits_service import get_guest_wallet
    wallet = await get_guest_wallet(DEVICE)

    assert wallet["credits"] == 42
    assert wallet["is_banned"] is False


@pytest.mark.asyncio
async def test_get_guest_wallet_new_creates_with_welcome_credits(mock_firebase, monkeypatch):
    from app.config import settings
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.credits_service import get_guest_wallet
    wallet = await get_guest_wallet("brand-new-device")

    assert wallet["credits"] == settings.welcome_credits
    assert wallet["is_banned"] is False


@pytest.mark.asyncio
async def test_get_guest_wallet_db_none_raises_503(monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", None)

    from app.services.credits_service import get_guest_wallet
    with pytest.raises(HTTPException) as exc:
        await get_guest_wallet(DEVICE)
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# check_ban_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_ban_status_banned(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("guest_wallets", DEVICE, {"credits": 10, "is_banned": True})

    from app.services.credits_service import check_ban_status
    assert await check_ban_status(DEVICE) is True


@pytest.mark.asyncio
async def test_check_ban_status_not_banned(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("guest_wallets", DEVICE, {"credits": 10, "is_banned": False})

    from app.services.credits_service import check_ban_status
    assert await check_ban_status(DEVICE) is False


# ---------------------------------------------------------------------------
# deduct_guest_credits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deduct_guest_credits_sufficient(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("guest_wallets", DEVICE, {"credits": 20, "is_banned": False})

    from app.services.credits_service import deduct_guest_credits
    with _tx_patch():
        new_balance = await deduct_guest_credits(DEVICE, cost=5)

    assert new_balance == 15


@pytest.mark.asyncio
async def test_deduct_guest_credits_insufficient_raises_402(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("guest_wallets", DEVICE, {"credits": 3, "is_banned": False})

    from app.services.credits_service import deduct_guest_credits
    with _tx_patch():
        with pytest.raises(HTTPException) as exc:
            await deduct_guest_credits(DEVICE, cost=5)
    assert exc.value.status_code == 402


# ---------------------------------------------------------------------------
# perform_recharge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_recharge_wrong_secret_raises_403(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.credits_service import perform_recharge
    with _tx_patch():
        with pytest.raises(HTTPException) as exc:
            await perform_recharge(DEVICE, amount=5, secret_key="wrong-key")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_perform_recharge_valid_existing_wallet(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("guest_wallets", DEVICE, {"credits": 10, "is_banned": False})

    from app.services.credits_service import perform_recharge
    with (
        _tx_patch(),
        patch.dict(os.environ, {"RECHARGE_SECRET_KEY": SECRET}),
        patch("app.services.credits_service.RECHARGE_SECRET_KEY", SECRET),
    ):
        result = await perform_recharge(DEVICE, amount=5, secret_key=SECRET)

    assert result["status"] == "success"
    assert result["new_balance"] == 15
