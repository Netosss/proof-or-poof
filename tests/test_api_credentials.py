"""Unit tests for api_credentials.py (key generation, resolution, revocation)."""

import pytest


@pytest.mark.asyncio
async def test_create_and_resolve_credential(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", "p1",
                       {"credit_balance": 0, "status": "active", "credits_version": 1})

    from app.services.api_credentials import create_credential, resolve_credential

    issued = await create_credential("p1")
    assert issued["api_key"].startswith(("fxl_live_", "fxl_test_"))
    assert issued["secret_key"].startswith(("fxs_live_", "fxs_test_"))

    cred = await resolve_credential(issued["api_key"])
    assert cred is not None
    assert cred["partner_id"] == "p1"
    assert cred["credential_id"] == issued["credential_id"]
    assert cred["secret_key"] == issued["secret_key"]


@pytest.mark.asyncio
async def test_resolve_unknown_key_returns_none(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.api_credentials import resolve_credential
    assert await resolve_credential("fxl_test_bogus") is None


@pytest.mark.asyncio
async def test_revoke_credential_blocks_resolution(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("enterprise_partners", "p1",
                       {"credit_balance": 0, "status": "active", "credits_version": 1})

    from app.services.api_credentials import create_credential, resolve_credential, revoke_credential
    issued = await create_credential("p1")
    await revoke_credential("p1", issued["credential_id"])
    assert await resolve_credential(issued["api_key"]) is None


@pytest.mark.asyncio
async def test_create_for_unknown_partner_raises(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    from fastapi import HTTPException
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.api_credentials import create_credential
    with pytest.raises(HTTPException) as exc:
        await create_credential("does-not-exist")
    assert exc.value.status_code == 404
