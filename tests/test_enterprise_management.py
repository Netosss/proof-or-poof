"""Integration tests for the /api/enterprise/* browser-callable endpoints."""

from unittest.mock import patch

import pytest


FAKE_USER = {"uid": "fb-user-1", "email": "founder@acme.com"}


def _override_auth(client):
    """Bypass Firebase ID token verification on the FastAPI TestClient."""
    from app.core.firebase_auth import get_current_user
    client.app.dependency_overrides[get_current_user] = lambda: FAKE_USER


# ---------------------------------------------------------------------------
# GET /api/enterprise/me
# ---------------------------------------------------------------------------

def test_me_returns_nulls_when_no_partner(client, mock_firebase):
    _override_auth(client)
    r = client.get("/api/enterprise/me")
    assert r.status_code == 200
    assert r.json() == {"partner": None, "application": None}


def test_me_returns_partner_after_provisioning(client, mock_firebase):
    _override_auth(client)
    # Seed a partner linked to this Firebase UID
    mock_firebase.seed(
        "enterprise_partners", "partner-xyz",
        {"company_name": "Acme", "contact_email": "ops@acme.com",
         "credit_balance": 50, "status": "active", "credits_version": 1,
         "firebase_uid": "fb-user-1"},
    )
    r = client.get("/api/enterprise/me")
    assert r.status_code == 200
    body = r.json()
    assert body["partner"]["id"] == "partner-xyz"
    assert body["partner"]["company_name"] == "Acme"
    # secret-ish fields are not exposed
    assert "credits_version" not in body["partner"]


# ---------------------------------------------------------------------------
# POST /api/enterprise/apply
# ---------------------------------------------------------------------------

def test_apply_creates_application(client, mock_firebase):
    _override_auth(client)
    payload = {
        "company_name": "Acme Corp",
        "contact_email": "ops@acme.com",
        "use_case": "newsroom",
        "expected_volume": "2k_10k",
        "tier": "sandbox",
        "notes": "evaluating",
    }
    r = client.post("/api/enterprise/apply", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert body["firebase_uid"] == "fb-user-1"


def test_apply_rejects_disposable_email(client, mock_firebase):
    _override_auth(client)
    payload = {
        "company_name": "Spam Co",
        "contact_email": "evil@mailinator.com",
        "use_case": "other",
        "expected_volume": "under_2k",
        "tier": "sandbox",
    }
    r = client.post("/api/enterprise/apply", json=payload)
    assert r.status_code == 400


def test_apply_validates_required_fields(client, mock_firebase):
    _override_auth(client)
    r = client.post("/api/enterprise/apply", json={"company_name": "X"})
    assert r.status_code == 422


def test_apply_idempotent_per_uid(client, mock_firebase):
    _override_auth(client)
    payload = {
        "company_name": "Acme Corp",
        "contact_email": "ops@acme.com",
        "use_case": "newsroom",
        "expected_volume": "2k_10k",
        "tier": "sandbox",
    }
    r1 = client.post("/api/enterprise/apply", json=payload)
    r2 = client.post("/api/enterprise/apply", json=payload)
    assert r1.json()["id"] == r2.json()["id"]


# ---------------------------------------------------------------------------
# Keys + usage — requires a partner record
# ---------------------------------------------------------------------------

def _seed_partner(mock_firebase, partner_id="p1"):
    mock_firebase.seed(
        "enterprise_partners", partner_id,
        {"company_name": "Acme", "contact_email": "ops@acme.com",
         "credit_balance": 100, "status": "active", "credits_version": 1,
         "firebase_uid": "fb-user-1"},
    )


def test_usage_returns_balance_and_ledger(client, mock_firebase):
    _override_auth(client)
    _seed_partner(mock_firebase)
    r = client.get("/api/enterprise/usage")
    assert r.status_code == 200
    body = r.json()
    assert body["credit_balance"] == 100
    assert body["company_name"] == "Acme"
    assert isinstance(body["ledger"], list)


def test_usage_blocks_non_partners(client, mock_firebase):
    _override_auth(client)
    r = client.get("/api/enterprise/usage")
    assert r.status_code == 403


def test_issue_key_returns_plaintext_once(client, mock_firebase):
    _override_auth(client)
    _seed_partner(mock_firebase)
    r = client.post("/api/enterprise/keys")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["api_key"].startswith(("fxl_live_", "fxl_test_"))
    assert body["secret_key"].startswith(("fxs_live_", "fxs_test_"))


def test_list_keys_strips_secret(client, mock_firebase):
    _override_auth(client)
    _seed_partner(mock_firebase)
    # Issue one key
    client.post("/api/enterprise/keys")
    r = client.get("/api/enterprise/keys")
    assert r.status_code == 200
    keys = r.json()["keys"]
    assert len(keys) >= 1
    for k in keys:
        assert "secret_key" not in k


def test_suspended_partner_blocked(client, mock_firebase):
    _override_auth(client)
    mock_firebase.seed(
        "enterprise_partners", "p-suspended",
        {"company_name": "X", "contact_email": "x@x.com",
         "credit_balance": 0, "status": "suspended", "credits_version": 1,
         "firebase_uid": "fb-user-1"},
    )
    r = client.get("/api/enterprise/usage")
    assert r.status_code == 403
