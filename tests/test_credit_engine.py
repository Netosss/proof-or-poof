"""
Unit tests for app/services/credit_engine.py.

Covers:
  - consume_credits: old field name, new field name, null field, insufficient, not found
  - grant_credits:   old field name, new field name, null field
  - get_user_balance: old field name, new field name, null field, not found
  - get_or_create_user: existing user (old/new/null field), new user signup bonus,
                        race condition (doc appears inside transaction)

Firebase transactions are neutralised with a pass-through async_transactional
mock that calls the inner function directly, matching the pattern in
test_credits_service.py.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import HTTPException


UID = "test-uid-001"
EMAIL = "test@example.com"


def _mock_async_transactional(func):
    """Pass-through: execute the inner function directly with a MockAsyncTransaction."""
    async def wrapper(transaction, *args, **kwargs):
        return await func(transaction, *args, **kwargs)
    return wrapper


@contextmanager
def _tx_patch():
    """Patch async_transactional in credit_engine so transactions execute inline."""
    with patch(
        "app.services.credit_engine.async_transactional",
        side_effect=_mock_async_transactional,
    ):
        yield


# ---------------------------------------------------------------------------
# get_user_balance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_user_balance_new_field(mock_firebase, monkeypatch):
    """New docs with credits_balance field are read correctly."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": 99, "email": EMAIL})

    from app.services.credit_engine import get_user_balance
    assert await get_user_balance(UID) == 99


@pytest.mark.asyncio
async def test_get_user_balance_old_field(mock_firebase, monkeypatch):
    """Old docs with only the `credits` field fall back correctly."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits": 123, "email": EMAIL})

    from app.services.credit_engine import get_user_balance
    assert await get_user_balance(UID) == 123


@pytest.mark.asyncio
async def test_get_user_balance_null_credits_balance_falls_back(mock_firebase, monkeypatch):
    """Explicit null credits_balance falls back to the legacy credits field."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": None, "credits": 55, "email": EMAIL})

    from app.services.credit_engine import get_user_balance
    assert await get_user_balance(UID) == 55


@pytest.mark.asyncio
async def test_get_user_balance_not_found_raises_404(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.credit_engine import get_user_balance
    with pytest.raises(HTTPException) as exc:
        await get_user_balance(UID)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# consume_credits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_credits_new_field(mock_firebase, monkeypatch):
    """Deduction works correctly when the doc uses credits_balance."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": 50, "credits_version": 1})

    from app.services.credit_engine import consume_credits
    with _tx_patch():
        new_balance = await consume_credits(UID, cost=10, reason="detect")
    assert new_balance == 40


@pytest.mark.asyncio
async def test_consume_credits_old_field(mock_firebase, monkeypatch):
    """Deduction works correctly when the doc uses the legacy credits field."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits": 123, "credits_version": 1})

    from app.services.credit_engine import consume_credits
    with _tx_patch():
        new_balance = await consume_credits(UID, cost=10, reason="detect")
    assert new_balance == 113


@pytest.mark.asyncio
async def test_consume_credits_null_credits_balance_falls_back(mock_firebase, monkeypatch):
    """Explicit null credits_balance falls back to legacy credits field for deduction."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": None, "credits": 80, "credits_version": 1})

    from app.services.credit_engine import consume_credits
    with _tx_patch():
        new_balance = await consume_credits(UID, cost=20, reason="inpaint")
    assert new_balance == 60


@pytest.mark.asyncio
async def test_consume_credits_insufficient_raises_402(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": 5, "credits_version": 1})

    from app.services.credit_engine import consume_credits
    with _tx_patch():
        with pytest.raises(HTTPException) as exc:
            await consume_credits(UID, cost=10, reason="detect")
    assert exc.value.status_code == 402


@pytest.mark.asyncio
async def test_consume_credits_user_not_found_raises_404(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.credit_engine import consume_credits
    with _tx_patch():
        with pytest.raises(HTTPException) as exc:
            await consume_credits(UID, cost=10, reason="detect")
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# grant_credits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grant_credits_new_field(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": 100, "credits_version": 1})

    from app.services.credit_engine import grant_credits
    with _tx_patch():
        new_balance = await grant_credits(UID, amount=50, reason="purchase")
    assert new_balance == 150


@pytest.mark.asyncio
async def test_grant_credits_old_field(mock_firebase, monkeypatch):
    """Grant reads the legacy credits field correctly before adding."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits": 123, "credits_version": 1})

    from app.services.credit_engine import grant_credits
    with _tx_patch():
        new_balance = await grant_credits(UID, amount=7, reason="ad_reward")
    assert new_balance == 130


@pytest.mark.asyncio
async def test_grant_credits_null_credits_balance_falls_back(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": None, "credits": 40, "credits_version": 1})

    from app.services.credit_engine import grant_credits
    with _tx_patch():
        new_balance = await grant_credits(UID, amount=10, reason="ad_reward")
    assert new_balance == 50


@pytest.mark.asyncio
async def test_grant_credits_negative_amount_for_refund(mock_firebase, monkeypatch):
    """Negative grants (refunds) are allowed and reduce the balance."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": 200, "credits_version": 1})

    from app.services.credit_engine import grant_credits
    with _tx_patch():
        new_balance = await grant_credits(UID, amount=-50, reason="refund")
    assert new_balance == 150


# ---------------------------------------------------------------------------
# get_or_create_user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_create_user_existing_new_field(mock_firebase, monkeypatch):
    """Returning user with credits_balance field: fast path returns correct balance."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": 300, "email": EMAIL})

    from app.services.credit_engine import get_or_create_user
    result = await get_or_create_user(UID, EMAIL)

    assert result["uid"] == UID
    assert result["credits_balance"] == 300
    assert result["is_new_user"] is False


@pytest.mark.asyncio
async def test_get_or_create_user_existing_old_field(mock_firebase, monkeypatch):
    """Returning user with legacy credits field: fast path falls back correctly."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits": 123, "email": EMAIL})

    from app.services.credit_engine import get_or_create_user
    result = await get_or_create_user(UID, EMAIL)

    assert result["credits_balance"] == 123
    assert result["is_new_user"] is False


@pytest.mark.asyncio
async def test_get_or_create_user_existing_null_credits_balance(mock_firebase, monkeypatch):
    """Returning user with explicit null credits_balance falls back to credits."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("users", UID, {"credits_balance": None, "credits": 77, "email": EMAIL})

    from app.services.credit_engine import get_or_create_user
    result = await get_or_create_user(UID, EMAIL)

    assert result["credits_balance"] == 77
    assert result["is_new_user"] is False


@pytest.mark.asyncio
async def test_get_or_create_user_new_user_gets_signup_bonus(mock_firebase, monkeypatch):
    """A brand-new user is created with 40 signup bonus credits."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.credit_engine import get_or_create_user
    with _tx_patch():
        result = await get_or_create_user(UID, EMAIL)

    assert result["uid"] == UID
    assert result["email"] == EMAIL
    assert result["credits_balance"] == 40
    assert result["is_new_user"] is True


@pytest.mark.asyncio
async def test_get_or_create_user_no_migration_from_guest(mock_firebase, monkeypatch):
    """Guest wallet credits are NOT migrated — new user always gets the fixed signup bonus."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    # Seed a guest wallet with many credits — should have zero effect on auth account
    mock_firebase.seed("guest_wallets", "some-device-id", {"credits": 999})

    from app.services.credit_engine import get_or_create_user
    with _tx_patch():
        result = await get_or_create_user(UID, EMAIL)

    assert result["credits_balance"] == 40
    assert result["is_new_user"] is True


@pytest.mark.asyncio
async def test_get_or_create_user_race_condition_returns_existing(mock_firebase, monkeypatch):
    """
    If the doc appears between the fast-path read (not found) and the
    transactional write (also not found initially, but doc is created by another
    request inside the transaction), the transactional double-check returns the
    existing doc without overwriting it.
    """
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    # Simulate: fast-path read sees nothing, but by the time the transactional
    # read runs the doc already exists (another concurrent request created it).
    mock_firebase.seed("users", UID, {"credits_balance": 40, "email": EMAIL})

    # Override the non-transactional `snapshot = await user_ref.get()` to
    # return not-exists, forcing the code into the transaction path.
    from tests.mocks.firebase_mock import MockDocumentSnapshot

    original_collection = mock_firebase.collection

    call_count = 0

    def _patched_collection(name):
        coll = original_collection(name)
        if name != "users":
            return coll

        original_document = coll.document

        def _patched_document(doc_id):
            ref = original_document(doc_id)
            original_get = ref.get

            async def _first_get_returns_empty(transaction=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1 and transaction is None:
                    # First call: the non-transactional fast-path read — simulate not found
                    return MockDocumentSnapshot(None, exists=False)
                return await original_get(transaction=transaction)

            ref.get = _first_get_returns_empty
            return ref

        coll.document = _patched_document
        return coll

    monkeypatch.setattr(mock_firebase, "collection", _patched_collection)

    from app.services.credit_engine import get_or_create_user
    with _tx_patch():
        result = await get_or_create_user(UID, EMAIL)

    # Should return the already-existing doc, not create a new one
    assert result["credits_balance"] == 40
    assert result["is_new_user"] is False
