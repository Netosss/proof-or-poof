"""Unit tests for enterprise_credit_engine."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import HTTPException


PARTNER = "partner-1"


def _mock_async_transactional(func):
    async def wrapper(transaction, *args, **kwargs):
        return await func(transaction, *args, **kwargs)
    return wrapper


@contextmanager
def _tx_patch():
    with patch(
        "app.services.enterprise_credit_engine.async_transactional",
        side_effect=_mock_async_transactional,
    ):
        yield


@pytest.mark.asyncio
async def test_reserve_credit_deducts_balance(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", PARTNER,
                       {"credit_balance": 100, "status": "active", "credits_version": 1})

    from app.services.enterprise_credit_engine import reserve_credit
    with _tx_patch():
        new_balance = await reserve_credit(PARTNER, cost=1, reason="api_scan_deduction",
                                           reference_id="req-1")
    assert new_balance == 99


@pytest.mark.asyncio
async def test_reserve_credit_insufficient_raises_402(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", PARTNER,
                       {"credit_balance": 0, "status": "active", "credits_version": 1})

    from app.services.enterprise_credit_engine import reserve_credit
    with _tx_patch():
        with pytest.raises(HTTPException) as exc:
            await reserve_credit(PARTNER, cost=1, reason="api_scan_deduction",
                                 reference_id="req-1")
    assert exc.value.status_code == 402
    assert exc.value.detail == "insufficient_credits"


@pytest.mark.asyncio
async def test_reserve_credit_suspended_raises_403(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", PARTNER,
                       {"credit_balance": 100, "status": "suspended", "credits_version": 1})

    from app.services.enterprise_credit_engine import reserve_credit
    with _tx_patch():
        with pytest.raises(HTTPException) as exc:
            await reserve_credit(PARTNER, cost=1, reason="api_scan_deduction",
                                 reference_id="req-1")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_reserve_credit_partner_not_found_raises_404(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.enterprise_credit_engine import reserve_credit
    with _tx_patch():
        with pytest.raises(HTTPException) as exc:
            await reserve_credit(PARTNER, cost=1, reason="api_scan_deduction",
                                 reference_id="req-1")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_refund_is_idempotent_via_ledger(mock_firebase, monkeypatch):
    """Calling refund twice with the same reference_id must only credit once."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", PARTNER,
                       {"credit_balance": 100, "status": "active", "credits_version": 1})

    from app.services.enterprise_credit_engine import refund_credit
    with _tx_patch():
        first = await refund_credit(PARTNER, amount=1,
                                    reason="api_refund:test", reference_id="req-1")
        second = await refund_credit(PARTNER, amount=1,
                                     reason="api_refund:test", reference_id="req-1")
    assert first == 101
    assert second is None  # idempotent no-op


@pytest.mark.asyncio
async def test_grant_credit_adds_to_balance(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", PARTNER,
                       {"credit_balance": 0, "status": "active", "credits_version": 1})

    from app.services.enterprise_credit_engine import grant_credit
    with _tx_patch():
        new_balance = await grant_credit(PARTNER, amount=5000,
                                         reason="purchase", reference_id="order-123")
    assert new_balance == 5000


@pytest.mark.asyncio
async def test_get_partner_balance(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", PARTNER, {"credit_balance": 42})

    from app.services.enterprise_credit_engine import get_partner_balance
    assert await get_partner_balance(PARTNER) == 42
