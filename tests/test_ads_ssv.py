"""
Tests for AdMob rewarded Server-Side Verification (SSV):
  - app.services.admob_ssv.verify_ssv  (ECDSA signature verification)
  - GET /api/ads/ssv                    (verify -> idempotent grant)
"""

import asyncio
import base64
from unittest.mock import AsyncMock, patch

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from app.services import admob_ssv

# ---------------------------------------------------------------------------
# Signature verification (unit)
# ---------------------------------------------------------------------------


def _make_key_and_pem():
    priv = ec.generate_private_key(ec.SECP256R1())
    pem = (
        priv.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return priv, pem


def _sign(priv, message: bytes) -> str:
    der = priv.sign(message, ec.ECDSA(hashes.SHA256()))
    return base64.urlsafe_b64encode(der).decode().rstrip("=")


def test_verify_ssv_valid(monkeypatch):
    priv, pem = _make_key_and_pem()
    monkeypatch.setattr(admob_ssv, "_keys_cache", {"123": pem})
    content = "ad_network=x&custom_data=uid1&transaction_id=tx1&reward_amount=1&reward_item=credits"
    sig = _sign(priv, content.encode())
    raw = f"{content}&signature={sig}&key_id=123"
    assert asyncio.run(admob_ssv.verify_ssv(raw, sig, "123")) is True


def test_verify_ssv_tampered_content_rejected(monkeypatch):
    """A forged custom_data (different user) must fail — this is the whole point."""
    priv, pem = _make_key_and_pem()
    monkeypatch.setattr(admob_ssv, "_keys_cache", {"123": pem})
    content = "ad_network=x&custom_data=uid1&transaction_id=tx1"
    sig = _sign(priv, content.encode())
    tampered = "ad_network=x&custom_data=ATTACKER&transaction_id=tx1"
    raw = f"{tampered}&signature={sig}&key_id=123"
    assert asyncio.run(admob_ssv.verify_ssv(raw, sig, "123")) is False


def test_verify_ssv_unknown_key_rejected(monkeypatch):
    async def _empty():
        return {}

    monkeypatch.setattr(admob_ssv, "_keys_cache", {})
    monkeypatch.setattr(admob_ssv, "_fetch_keys", _empty)  # no network
    raw = "a=b&signature=xx&key_id=999"
    assert asyncio.run(admob_ssv.verify_ssv(raw, "xx", "999")) is False


def test_verify_ssv_missing_params_rejected(monkeypatch):
    monkeypatch.setattr(admob_ssv, "_keys_cache", {})
    assert asyncio.run(admob_ssv.verify_ssv("a=b", "", "1")) is False
    assert asyncio.run(admob_ssv.verify_ssv("a=b&signature=x", "x", "")) is False
    assert asyncio.run(admob_ssv.verify_ssv("a=b_no_marker", "x", "1")) is False


# ---------------------------------------------------------------------------
# Route: GET /api/ads/ssv
# ---------------------------------------------------------------------------

_PARAMS = (
    "custom_data=uid-ssv&transaction_id=tx-abc"
    "&reward_amount=1&reward_item=credits&signature=SIG&key_id=1"
)


def test_ssv_valid_signature_grants(client):
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            return_value=(20, 1, 60),
        ) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{_PARAMS}")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    m_grant.assert_awaited_once()
    call_args = m_grant.await_args.args
    assert call_args[0] == "uid-ssv"
    assert call_args[1] == "ssv_tx-abc"


def test_ssv_invalid_signature_403_no_grant(client):
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=False),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{_PARAMS}")

    assert resp.status_code == 403
    m_grant.assert_not_awaited()


def test_ssv_missing_custom_data_400_no_grant(client):
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get("/api/ads/ssv?transaction_id=tx1&signature=S&key_id=1")

    assert resp.status_code == 400
    m_grant.assert_not_awaited()


def test_ssv_duplicate_transaction_grants_once(client):
    """Google retries the callback — the same transaction_id must grant only once."""
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            return_value=(20, 1, 60),
        ) as m_grant,
    ):
        first = client.get(f"/api/ads/ssv?{_PARAMS}")
        second = client.get(f"/api/ads/ssv?{_PARAMS}")

    assert first.status_code == 200 and first.json() == {"status": "ok"}
    assert second.status_code == 200 and second.json() == {"status": "duplicate"}
    m_grant.assert_awaited_once()


def test_ssv_appended_params_after_signature_ignored(client):
    """
    Query-param smuggling: params appended AFTER the signature (last-wins in
    Starlette) must NOT be the ones acted on — the signed content wins.
    """
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            return_value=(20, 1, 60),
        ) as m_grant,
    ):
        q = (
            "custom_data=uid1&transaction_id=tx1&reward_amount=1&reward_item=credits"
            "&signature=SIG&key_id=1&transaction_id=FORGED&custom_data=ATTACKER"
        )
        resp = client.get(f"/api/ads/ssv?{q}")

    assert resp.status_code == 200
    m_grant.assert_awaited_once()
    # Must grant to the SIGNED uid/tx, never the appended forged values.
    assert m_grant.await_args.args[0] == "uid1"
    assert m_grant.await_args.args[1] == "ssv_tx1"


def test_ssv_grant_failure_releases_claim_so_retry_regrants(client):
    """
    If the grant fails AFTER the transaction_id claim commits, the claim must be
    released so Google's retry lands on a fresh claim and actually grants —
    otherwise the reward is permanently lost as a false "duplicate".
    """
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("firestore hiccup"), (20, 1, 60)],
        ) as m_grant,
    ):
        first = client.get(f"/api/ads/ssv?{_PARAMS}")
        second = client.get(f"/api/ads/ssv?{_PARAMS}")

    assert first.status_code == 500
    assert second.status_code == 200 and second.json() == {"status": "ok"}
    assert m_grant.await_count == 2
