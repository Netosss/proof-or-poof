"""
Tests for account deletion (Google Play "Delete account" compliance):
  - GET  /delete-account   public self-service page
  - DELETE /api/account     authenticated hard-delete of user + associated data
"""

from contextlib import contextmanager
from unittest.mock import patch

from app.main import app


@contextmanager
def _override_auth(uid: str):
    from app.core.firebase_auth import get_current_user

    def _fake():
        return {"uid": uid, "email": "user@example.com"}

    app.dependency_overrides[get_current_user] = _fake
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_delete_account_page_renders(client):
    resp = client.get("/delete-account")
    assert resp.status_code == 200
    body = resp.text
    # Firebase web config is injected, and the self-service delete call is present.
    assert "proof-or-poof" in body
    assert "AIzaSy" in body
    assert "/api/account" in body
    assert "Permanently delete my account" in body


def test_delete_account_removes_all_user_data(client, mock_firebase):
    uid = "uid-del"

    # Seed: user doc + linked guest wallet, credit ledger, ad-reward records.
    users = mock_firebase.collection("users")
    users._docs[uid] = {
        "email": "user@example.com",
        "credits_balance": 40,
        "known_device_ids": ["dev-1"],
    }
    users.document(uid).collection("credit_ledger")._docs["L1"] = {"delta": 40}
    mock_firebase.collection("ad_rewards")._docs[f"{uid}_2026-07-10"] = {
        "user_id": uid,
        "count": 2,
    }
    mock_firebase.collection("ad_ssv_rewards")._docs["tx-1"] = {"user_id": uid}
    mock_firebase.collection("guest_wallets")._docs["dev-1"] = {"credits": 5}
    # A different user's data must survive.
    users._docs["other"] = {"email": "other@example.com"}

    with (
        _override_auth(uid),
        patch("app.api.account.fb_auth.delete_user") as m_del_user,
    ):
        resp = client.delete("/api/account", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted"}

    # Every trace of this user is gone.
    assert uid not in users._docs
    assert users.document(uid).collection("credit_ledger")._docs == {}
    assert mock_firebase.collection("ad_rewards")._docs == {}
    assert mock_firebase.collection("ad_ssv_rewards")._docs == {}
    assert "dev-1" not in mock_firebase.collection("guest_wallets")._docs
    # Firebase Auth user deleted exactly once.
    m_del_user.assert_called_once_with(uid)
    # Another user's data is untouched.
    assert "other" in users._docs


def test_delete_account_idempotent_when_auth_user_missing(client, mock_firebase):
    """If the Firebase Auth user is already gone, deletion still succeeds."""
    from firebase_admin import auth as fb_auth

    uid = "uid-gone"
    mock_firebase.collection("users")._docs[uid] = {"email": "g@example.com"}

    with (
        _override_auth(uid),
        patch(
            "app.api.account.fb_auth.delete_user",
            side_effect=fb_auth.UserNotFoundError("gone"),
        ),
    ):
        resp = client.delete("/api/account", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 200
    assert uid not in mock_firebase.collection("users")._docs
