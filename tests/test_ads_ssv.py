"""
Tests for AdMob rewarded Server-Side Verification (SSV):
  - app.services.admob_ssv.verify_ssv  (ECDSA signature verification)
  - GET /api/ads/ssv                    (verify -> idempotent grant)
"""

import asyncio
import base64
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from app.api.credits import parse_custom_data
from app.config import settings
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

_OUR_UNIT = "ca-app-pub-2844061727637796/6754427834"


@pytest.fixture(autouse=True)
def _configure_ad_unit(monkeypatch):
    """
    Every route test needs our ad unit configured, because the endpoint fails
    CLOSED when it is not — an unconfigured deployment must never grant.
    `test_ssv_unconfigured_unit_grants_nothing` opts out by setting it back.
    """
    monkeypatch.setattr(settings, "admob_rewarded_ad_unit_id", _OUR_UNIT)


_PARAMS = (
    f"ad_unit={_OUR_UNIT}&custom_data=uid:uid-ssv&transaction_id=tx-abc"
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
    # (subject_id, kind, reference_id) — the "uid:" prefix is stripped.
    assert call_args[0] == "uid-ssv"
    assert call_args[1] == "uid"
    assert call_args[2] == "ssv_tx-abc"


def test_ssv_invalid_signature_403_no_grant(client):
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=False),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{_PARAMS}")

    assert resp.status_code == 403
    m_grant.assert_not_awaited()


def test_ssv_missing_custom_data_200_ignored_no_grant(client):
    """AdMob's verify ping (and guest views) are validly signed but carry no
    custom_data — ACK with 200 so AdMob accepts the URL, but grant nothing."""
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get("/api/ads/ssv?transaction_id=tx1&signature=S&key_id=1")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ignored"}
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
            f"ad_unit={_OUR_UNIT}&custom_data=uid:uid1&transaction_id=tx1"
            "&reward_amount=1&reward_item=credits"
            "&signature=SIG&key_id=1&transaction_id=FORGED&custom_data=uid:ATTACKER"
            f"&ad_unit=ca-app-pub-9999999999999999/9999999999"
        )
        resp = client.get(f"/api/ads/ssv?{q}")

    assert resp.status_code == 200
    m_grant.assert_awaited_once()
    # Must grant to the SIGNED uid/tx, never the appended forged values — and the
    # appended foreign ad_unit must not defeat the pin either.
    assert m_grant.await_args.args[0] == "uid1"
    assert m_grant.await_args.args[2] == "ssv_tx1"


# ---------------------------------------------------------------------------
# ad_unit pinning — the callback must be bound to OUR inventory
# ---------------------------------------------------------------------------


def test_ssv_foreign_ad_unit_grants_nothing(client):
    """
    THE critical case. AdMob's SSV verifier keys are a single GLOBAL key set
    shared by every publisher, so a valid signature proves only that *some*
    AdMob server sent the callback — never that our app did.

    Without this pin, an attacker creates their own AdMob account, points its
    SSV callback URL at us, stamps custom_data with any wallet, and watches an
    ad. Google signs it with the same global key and we grant — funded by an
    impression Google pays THEM for.
    """
    q = (
        "ad_unit=ca-app-pub-9999999999999999/9999999999"
        "&custom_data=uid:victim&transaction_id=tx-foreign"
        "&reward_amount=1&reward_item=credits&signature=SIG&key_id=1"
    )
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{q}")

    # 200 (never 4xx — AdMob backs off the whole unit on errors) but NO grant.
    assert resp.status_code == 200
    assert resp.json() == {"status": "ignored"}
    m_grant.assert_not_awaited()


def test_ssv_missing_ad_unit_grants_nothing(client):
    q = (
        "custom_data=uid:someone&transaction_id=tx-nounit"
        "&reward_amount=1&reward_item=credits&signature=SIG&key_id=1"
    )
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{q}")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ignored"}
    m_grant.assert_not_awaited()


def test_ssv_unconfigured_unit_grants_nothing(client, monkeypatch):
    """An unconfigured deployment must fail CLOSED, not grant to everyone."""
    monkeypatch.setattr(settings, "admob_rewarded_ad_unit_id", "")
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{_PARAMS}")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ignored"}
    m_grant.assert_not_awaited()


# ---------------------------------------------------------------------------
# custom_data parsing — attacker-influenced, becomes a Firestore document id
# ---------------------------------------------------------------------------


def test_parse_custom_data_uid_and_device():
    assert parse_custom_data("uid:abc123") == ("uid", "abc123")
    assert parse_custom_data("device:dev-42.x_y") == ("device", "dev-42.x_y")


def test_parse_custom_data_legacy_bare_uid_treated_as_uid():
    """Builds predating the prefix stamp a bare uid — must keep working."""
    assert parse_custom_data("abc123") == ("uid", "abc123")


def test_parse_custom_data_rejects_path_injection():
    """
    A subject containing "/" becomes a NESTED Firestore path, which yields an
    unlimited supply of fresh daily-cap documents and makes the cap moot. The
    signature does not help here: custom_data is attacker-chosen and Google
    signs whatever the client stamped.
    """
    assert parse_custom_data("uid:a/b/c") is None
    assert parse_custom_data("device:../../users/victim") is None
    assert parse_custom_data("a/b") is None


def test_parse_custom_data_rejects_firestore_reserved_shapes():
    """Firestore rejects these as document ids — a write would 500, not ignore."""
    assert parse_custom_data("uid:.") is None
    assert parse_custom_data("uid:..") is None
    assert parse_custom_data("uid:__proto__") is None
    assert parse_custom_data("device:__x__") is None


def test_parse_custom_data_rejects_junk_and_empty():
    assert parse_custom_data(None) is None
    assert parse_custom_data("") is None
    assert parse_custom_data("uid:") is None
    assert parse_custom_data("bogus:abc") is None
    assert parse_custom_data("uid:has spaces") is None
    assert parse_custom_data("uid:" + "x" * 129) is None


def test_ssv_unparseable_custom_data_grants_nothing(client):
    q = (
        f"ad_unit={_OUR_UNIT}&custom_data=uid:bad%2Fpath&transaction_id=tx-bad"
        "&reward_amount=1&reward_item=credits&signature=SIG&key_id=1"
    )
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{q}")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ignored"}
    m_grant.assert_not_awaited()


# ---------------------------------------------------------------------------
# Guest branch
# ---------------------------------------------------------------------------


def test_ssv_guest_grants_to_device_wallet(client):
    """device: subjects credit guest_wallets, not users."""
    q = (
        f"ad_unit={_OUR_UNIT}&custom_data=device:dev-abc&transaction_id=tx-guest"
        "&reward_amount=1&reward_item=credits&signature=SIG&key_id=1"
    )
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            return_value=(20, 1, 60),
        ) as m_grant,
    ):
        resp = client.get(f"/api/ads/ssv?{q}")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert m_grant.await_args.args[0] == "dev-abc"
    assert m_grant.await_args.args[1] == "device"


def test_ssv_device_kind_cannot_burn_a_real_users_cap(client):
    """
    Griefing vector: stamping device:<a-real-firebase-uid> must NOT touch that
    user's wallet or their daily cap. The kind is carried through so the cap
    document is namespaced (`device_<id>_<date>` vs `uid_<id>_<date>`).
    """
    victim_uid = "victim-firebase-uid"
    q = (
        f"ad_unit={_OUR_UNIT}&custom_data=device:{victim_uid}&transaction_id=tx-grief"
        "&reward_amount=1&reward_item=credits&signature=SIG&key_id=1"
    )
    with (
        patch("app.api.credits.verify_ssv", new_callable=AsyncMock, return_value=True),
        patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            return_value=(20, 1, 60),
        ) as m_grant,
    ):
        client.get(f"/api/ads/ssv?{q}")

    # Routed as a GUEST subject — never as the signed-in uid.
    assert m_grant.await_args.args[1] == "device"


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
