"""Tests for enterprise_applications service."""

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_create_application_happy_path(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.enterprise_applications import create_application
    app = await create_application(
        firebase_uid="uid-1",
        contact_email="ops@acme.com",
        company_name="Acme Corp",
        use_case="newsroom",
        expected_volume="2k_10k",
        tier="sandbox",
        notes="just checking",
    )
    assert app["status"] == "pending"
    assert app["firebase_uid"] == "uid-1"
    assert app["contact_email"] == "ops@acme.com"
    assert app["free_email"] is False


@pytest.mark.asyncio
async def test_create_application_rejects_disposable_email(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.enterprise_applications import create_application
    with pytest.raises(HTTPException) as exc:
        await create_application(
            firebase_uid="uid-disposable",
            contact_email="test@mailinator.com",
            company_name="Spam Co",
            use_case="other",
            expected_volume="under_2k",
            tier="sandbox",
        )
    assert exc.value.status_code == 400
    assert "disposable" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_create_application_flags_free_email(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.enterprise_applications import create_application
    app = await create_application(
        firebase_uid="uid-gmail",
        contact_email="someone@gmail.com",
        company_name="Indie Co",
        use_case="research",
        expected_volume="under_2k",
        tier="sandbox",
    )
    assert app["free_email"] is True
    # Not rejected — just flagged for operator review
    assert app["status"] == "pending"


@pytest.mark.asyncio
async def test_create_application_validates_enums(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.enterprise_applications import create_application
    with pytest.raises(HTTPException) as exc:
        await create_application(
            firebase_uid="uid-bad",
            contact_email="ops@acme.com",
            company_name="Acme",
            use_case="not_a_real_use_case",
            expected_volume="2k_10k",
            tier="sandbox",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_create_application_is_idempotent_per_uid(mock_firebase, monkeypatch):
    """Second submission from same UID returns existing record, no double-write."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.enterprise_applications import create_application
    a1 = await create_application(
        firebase_uid="uid-once", contact_email="ops@acme.com",
        company_name="Acme", use_case="newsroom", expected_volume="2k_10k",
        tier="sandbox",
    )
    a2 = await create_application(
        firebase_uid="uid-once", contact_email="different@acme.com",
        company_name="Acme Renamed", use_case="research", expected_volume="over_25k",
        tier="pro",
    )
    assert a1["id"] == a2["id"]
    # Original payload is preserved — second call doesn't overwrite
    assert a2["company_name"] == "Acme"
    assert a2["tier"] == "sandbox"


@pytest.mark.asyncio
async def test_disposable_email_detection():
    from app.services.enterprise_applications import is_disposable_email, is_free_email
    assert is_disposable_email("anyone@mailinator.com") is True
    assert is_disposable_email("anyone@guerrillamail.com") is True
    assert is_disposable_email("anyone@gmail.com") is False
    assert is_disposable_email("anyone@acme.com") is False
    assert is_free_email("anyone@gmail.com") is True
    assert is_free_email("anyone@acme.com") is False


@pytest.mark.asyncio
async def test_list_pending_applications(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.enterprise_applications import (
        create_application,
        list_pending_applications,
    )
    await create_application(firebase_uid="u1", contact_email="a@acme.com",
                             company_name="Acme", use_case="newsroom",
                             expected_volume="2k_10k", tier="sandbox")
    await create_application(firebase_uid="u2", contact_email="b@bco.com",
                             company_name="BetaCo", use_case="insurance",
                             expected_volume="10k_25k", tier="sandbox")

    pending = await list_pending_applications(limit=10)
    assert len(pending) == 2
    assert {p["company_name"] for p in pending} == {"Acme", "BetaCo"}
