"""End-to-end tests for the enterprise Lemon Squeezy webhook paths.

Exercises POST /webhooks/lemonsqueezy with account_type=enterprise payloads
through the FastAPI TestClient. Covers the auto-provisioning, duplicate-order,
and refund branches that previously had zero coverage.
"""

import hashlib
import hmac as hmac_mod
import json
from unittest.mock import patch

import pytest


_SECRET = "lemon-enterprise-test-secret"


def _signed(payload: dict) -> tuple[bytes, str]:
    body = json.dumps(payload).encode()
    sig = hmac_mod.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return body, sig


def _seed_partner(mock_firebase, partner_id="p-existing", firebase_uid="fb-existing"):
    mock_firebase.seed(
        "enterprise_partners", partner_id,
        {
            "company_name": "Existing Co",
            "contact_email": "ex@co.com",
            "credit_balance": 0,
            "status": "active",
            "credits_version": 1,
            "firebase_uid": firebase_uid,
        },
    )


def _set_variant_map(monkeypatch, variant_id="vt-1", credits=10000):
    """Force the active variant map to contain our test variant so the webhook
    finds it on lookup."""
    from app.config import settings
    monkeypatch.setattr(settings, "lemon_squeezy_variants", {variant_id: credits})
    monkeypatch.setattr(settings, "enterprise_ls_variants", {variant_id: credits})
    # APP_ENV is "prod" by default in tests — the env_gate accepts non-test_mode payloads.


def _post_webhook(client, payload):
    body, sig = _signed(payload)
    with patch("app.api.webhooks._lemonsqueezy_webhook_secret", return_value=_SECRET):
        return client.post(
            "/webhooks/lemonsqueezy",
            content=body,
            headers={"X-Signature": sig, "Content-Type": "application/json"},
        )


def _ls_payload(*, order_id, variant_id="vt-1", account_type="enterprise",
                custom_data=None, status="paid", event="order_paid",
                user_name="Acme Buyer", user_email="buyer@acme.com",
                total=34999, test_mode=False):
    cd = {"account_type": account_type, "env": "prod"}
    if custom_data:
        cd.update(custom_data)
    return {
        "meta": {"event_name": event, "test_mode": test_mode, "custom_data": cd},
        "data": {
            "id": order_id,
            "attributes": {
                "status": status,
                "total": total,
                "user_name": user_name,
                "user_email": user_email,
                "first_order_item": {"variant_id": variant_id},
            },
        },
    }


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_enterprise_order_paid_with_existing_partner(client, mock_firebase, monkeypatch):
    _set_variant_map(monkeypatch)
    _seed_partner(mock_firebase, partner_id="p-existing", firebase_uid="fb-1")

    payload = _ls_payload(
        order_id="ord-1",
        custom_data={"partner_id": "p-existing", "firebase_uid": "fb-1"},
    )
    r = _post_webhook(client, payload)

    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}
    # Partner balance should have been bumped.
    data = mock_firebase.collection("enterprise_partners")._docs["p-existing"]
    assert data["credit_balance"] == 10000


def test_enterprise_order_paid_auto_provisions_from_ls_payload(client, mock_firebase, monkeypatch):
    """Self-serve paid flow: no prior partner, no application — must create
    partner from LS payload (user_name + user_email)."""
    _set_variant_map(monkeypatch)
    payload = _ls_payload(
        order_id="ord-self-serve",
        custom_data={"firebase_uid": "fb-newcomer"},
        user_name="Newcomer Inc",
        user_email="founder@newcomer.com",
    )
    r = _post_webhook(client, payload)

    assert r.status_code == 200, r.text
    # Find the partner the webhook created.
    partners = mock_firebase.collection("enterprise_partners")._docs
    created = [p for p in partners.values() if p.get("firebase_uid") == "fb-newcomer"]
    assert len(created) == 1
    assert created[0]["company_name"] == "Newcomer Inc"
    assert created[0]["contact_email"] == "founder@newcomer.com"
    assert created[0]["credit_balance"] == 10000


def test_enterprise_order_paid_duplicate_is_idempotent(client, mock_firebase, monkeypatch):
    """Same order_id posted twice → second is skipped, balance unchanged."""
    _set_variant_map(monkeypatch)
    _seed_partner(mock_firebase, partner_id="p-dup", firebase_uid="fb-dup")

    payload = _ls_payload(
        order_id="ord-dup",
        custom_data={"partner_id": "p-dup", "firebase_uid": "fb-dup"},
    )
    first = _post_webhook(client, payload)
    second = _post_webhook(client, payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json().get("skipped") == "duplicate"

    # Balance reflects exactly ONE grant.
    data = mock_firebase.collection("enterprise_partners")._docs["p-dup"]
    assert data["credit_balance"] == 10000


def test_enterprise_order_refunded(client, mock_firebase, monkeypatch):
    """A refund event clawbacks credits from the partner."""
    _set_variant_map(monkeypatch)
    _seed_partner(mock_firebase, partner_id="p-refund", firebase_uid="fb-refund")

    # Step 1: pay
    pay_payload = _ls_payload(
        order_id="ord-refund",
        custom_data={"partner_id": "p-refund", "firebase_uid": "fb-refund"},
    )
    r1 = _post_webhook(client, pay_payload)
    assert r1.status_code == 200
    balance_after_pay = mock_firebase.collection("enterprise_partners")._docs["p-refund"]["credit_balance"]
    assert balance_after_pay == 10000

    # Step 2: refund
    refund_payload = _ls_payload(
        order_id="ord-refund",
        event="order_refunded",
        custom_data={"partner_id": "p-refund", "firebase_uid": "fb-refund"},
    )
    r2 = _post_webhook(client, refund_payload)
    assert r2.status_code == 200, r2.text
    balance_after_refund = mock_firebase.collection("enterprise_partners")._docs["p-refund"]["credit_balance"]
    # Refund clawbacks the 10000 credits.
    assert balance_after_refund == 0


def test_enterprise_order_refunded_unknown_order(client, mock_firebase, monkeypatch):
    """Refund for an order we never recorded → skip cleanly, no error."""
    _set_variant_map(monkeypatch)
    refund_payload = _ls_payload(
        order_id="ord-ghost",
        event="order_refunded",
        custom_data={"partner_id": "anyone", "firebase_uid": "anyone"},
    )
    r = _post_webhook(client, refund_payload)
    assert r.status_code == 200
    assert r.json().get("skipped") == "purchase_not_found"
