"""
Tests for:
  POST /api/billing/google/verify  (authenticated AND guest paths)

The Google Play Developer API call is always mocked — no network I/O.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from app.services.google_play_billing import InvalidPurchaseError

DEVICE_ID = "billing-test-device"
AUTH_UID = "auth-user-billing-001"
AUTH_USER = {"uid": AUTH_UID, "email": "auth@example.com"}
ORDER_ID = "GPA.3312-4081-1234-56789"
SA_JSON = '{"type": "service_account", "project_id": "test"}'

VALID_PURCHASE = {
    "purchaseState": 0,
    "orderId": ORDER_ID,
    "consumptionState": 0,
}


@contextmanager
def _override_auth_user(user: dict | None):
    """Temporarily override get_optional_user dependency on the FastAPI app."""
    from app.core.firebase_auth import get_optional_user
    from app.main import app

    async def _fake():
        return user

    app.dependency_overrides[get_optional_user] = _fake
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_optional_user, None)


def _post(client, product_id="credits_starter", token="purchase-token-abc", headers=None):
    return client.post(
        "/api/billing/google/verify",
        json={"product_id": product_id, "purchase_token": token},
        headers=headers or {},
    )


# ---------------------------------------------------------------------------
# Valid purchase — authenticated path
# ---------------------------------------------------------------------------


def test_valid_purchase_grants_credits_authenticated(client, mock_redis, monkeypatch):
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    mock_grant = AsyncMock(return_value=540)

    with _override_auth_user(AUTH_USER):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value=dict(VALID_PURCHASE),
            ),
            patch("app.api.billing.grant_credits", mock_grant),
            patch("app.api.billing.log_transaction"),
        ):
            response = _post(client)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "credits_granted": 500,
        "new_balance": 540,
        "order_id": ORDER_ID,
    }
    mock_grant.assert_awaited_once_with(AUTH_UID, 500, "google_play_purchase", ORDER_ID)
    # Idempotency key claimed in Redis.
    assert f"billing:google:order:{ORDER_ID}" in mock_redis._store


def test_valid_purchase_grants_credits_guest(client, monkeypatch):
    """Guest path (X-Device-ID, no Bearer) grants to the guest wallet."""
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    mock_grant = AsyncMock(return_value=2040)

    with _override_auth_user(None):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value=dict(VALID_PURCHASE),
            ),
            patch("app.api.billing.grant_guest_credits", mock_grant),
            patch("app.api.billing.log_transaction"),
        ):
            response = _post(client, product_id="credits_pro", headers={"X-Device-ID": DEVICE_ID})

    assert response.status_code == 200
    body = response.json()
    assert body["credits_granted"] == 2000
    assert body["new_balance"] == 2040
    assert body["order_id"] == ORDER_ID
    mock_grant.assert_awaited_once_with(DEVICE_ID, 2000)


# ---------------------------------------------------------------------------
# Idempotency — duplicate orderId never double-grants
# ---------------------------------------------------------------------------


def test_duplicate_order_returns_zero_granted(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    mock_grant = AsyncMock(return_value=540)

    with _override_auth_user(AUTH_USER):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value=dict(VALID_PURCHASE),
            ),
            patch("app.api.billing.grant_credits", mock_grant),
            patch(
                "app.api.billing.get_user_balance",
                new_callable=AsyncMock,
                return_value=540,
            ),
            patch("app.api.billing.log_transaction"),
        ):
            first = _post(client)
            second = _post(client)

    assert first.status_code == 200
    assert first.json()["credits_granted"] == 500

    assert second.status_code == 200
    assert second.json() == {
        "status": "ok",
        "credits_granted": 0,
        "new_balance": 540,
        "order_id": ORDER_ID,
    }
    # Credits granted exactly once despite two identical requests.
    mock_grant.assert_awaited_once()


def test_grant_failure_releases_idempotency_claim(client, mock_redis, monkeypatch):
    """A failed grant must free the orderId so the client's retry can succeed."""
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)

    with _override_auth_user(AUTH_USER):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value=dict(VALID_PURCHASE),
            ),
            patch(
                "app.api.billing.grant_credits",
                new_callable=AsyncMock,
                side_effect=__import__("fastapi").HTTPException(
                    status_code=500, detail="Credit transaction failed"
                ),
            ),
        ):
            response = _post(client)

    assert response.status_code == 500
    assert f"billing:google:order:{ORDER_ID}" not in mock_redis._store


# ---------------------------------------------------------------------------
# Durable idempotency (M4)
#
# Redis held the ONLY claim, under a TTL. A flush, an eviction, or just waiting
# out google_play_order_ttl_sec made every historical order re-grantable by
# replaying its token.
# ---------------------------------------------------------------------------


def test_already_consumed_purchase_grants_nothing(client, monkeypatch):
    """
    consumptionState == 1 means Play already handed this token over and the
    client consumed it, which only happens after a successful grant. Replaying
    it must pay out nothing — and this check must not depend on any cache we
    own, so no Redis state is seeded here.
    """
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    mock_grant = AsyncMock(return_value=540)

    with _override_auth_user(AUTH_USER):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value={**VALID_PURCHASE, "consumptionState": 1},
            ),
            patch("app.api.billing.grant_credits", mock_grant),
            patch(
                "app.api.billing.get_user_balance",
                new_callable=AsyncMock,
                return_value=540,
            ),
            patch("app.api.billing.log_transaction"),
        ):
            response = _post(client)

    assert response.status_code == 200
    assert response.json()["credits_granted"] == 0
    mock_grant.assert_not_awaited()


def test_redis_flush_does_not_reopen_a_granted_order(client, mock_redis, monkeypatch):
    """
    The actual M4 scenario. Grant once, wipe Redis entirely (flush / eviction /
    TTL expiry are indistinguishable here), replay the same token. Firestore is
    the system of record now, so the second call must grant nothing.
    """
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    mock_grant = AsyncMock(return_value=540)

    with _override_auth_user(AUTH_USER):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value=dict(VALID_PURCHASE),
            ),
            patch("app.api.billing.grant_credits", mock_grant),
            patch(
                "app.api.billing.get_user_balance",
                new_callable=AsyncMock,
                return_value=540,
            ),
            patch("app.api.billing.log_transaction"),
        ):
            first = _post(client)
            mock_redis._store.clear()          # <- the whole point
            second = _post(client)

    assert first.json()["credits_granted"] == 500
    assert second.status_code == 200
    assert second.json()["credits_granted"] == 0
    mock_grant.assert_awaited_once()


def test_grant_failure_releases_the_durable_claim_too(client, mock_redis, monkeypatch):
    """
    Releasing only the Redis claim would leave the durable one behind, so every
    retry of a purchase the user PAID for would answer "already granted, 0
    credits" and the credits would never arrive. That is worse than the
    double-grant this mechanism exists to prevent, so it gets its own test:
    fail the grant, then let the retry succeed.
    """
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    failing = AsyncMock(
        side_effect=__import__("fastapi").HTTPException(status_code=500, detail="boom")
    )
    succeeding = AsyncMock(return_value=540)

    with _override_auth_user(AUTH_USER):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value=dict(VALID_PURCHASE),
            ),
            patch("app.api.billing.log_transaction"),
        ):
            with patch("app.api.billing.grant_credits", failing):
                first = _post(client)
            with patch("app.api.billing.grant_credits", succeeding):
                retry = _post(client)

    assert first.status_code == 500
    # The retry must actually pay out — not report a phantom prior grant.
    assert retry.status_code == 200
    assert retry.json()["credits_granted"] == 500
    succeeding.assert_awaited_once()


# ---------------------------------------------------------------------------
# Invalid purchases
# ---------------------------------------------------------------------------


def test_canceled_purchase_returns_invalid(client, monkeypatch):
    """purchaseState != 0 (canceled/pending) → 400 INVALID_PURCHASE, no grant."""
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    mock_grant = AsyncMock()

    with _override_auth_user(AUTH_USER):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value={"purchaseState": 1, "orderId": ORDER_ID},
            ),
            patch("app.api.billing.grant_credits", mock_grant),
        ):
            response = _post(client)

    assert response.status_code == 400
    assert response.json() == {"detail": "INVALID_PURCHASE"}
    mock_grant.assert_not_called()


def test_unverifiable_token_returns_invalid(client, monkeypatch):
    """Google rejecting the token (404/400) → 400 INVALID_PURCHASE."""
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)

    with _override_auth_user(AUTH_USER):
        with patch(
            "app.api.billing.get_product_purchase",
            new_callable=AsyncMock,
            side_effect=InvalidPurchaseError("Google Play returned 404"),
        ):
            response = _post(client)

    assert response.status_code == 400
    assert response.json() == {"detail": "INVALID_PURCHASE"}


# ---------------------------------------------------------------------------
# Unknown SKU
# ---------------------------------------------------------------------------


def test_unknown_product_returns_400(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    mock_verify = AsyncMock()

    with _override_auth_user(AUTH_USER):
        with patch("app.api.billing.get_product_purchase", mock_verify):
            response = _post(client, product_id="credits_mega")

    assert response.status_code == 400
    assert response.json() == {"detail": "UNKNOWN_PRODUCT"}
    # Never hits Google for an unmapped SKU.
    mock_verify.assert_not_called()


# ---------------------------------------------------------------------------
# Missing service-account configuration
# ---------------------------------------------------------------------------


def test_missing_service_account_returns_503(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", raising=False)
    mock_verify = AsyncMock()

    with _override_auth_user(AUTH_USER):
        with patch("app.api.billing.get_product_purchase", mock_verify):
            response = _post(client)

    assert response.status_code == 503
    assert response.json() == {"detail": "BILLING_NOT_CONFIGURED"}
    # Never fake-verifies without credentials.
    mock_verify.assert_not_called()


# ---------------------------------------------------------------------------
# Rate limiting — rotating X-Device-ID must not bypass the per-IP bucket
# ---------------------------------------------------------------------------


def test_ip_rate_limit_blocks_rotating_device_ids(client, monkeypatch):
    """
    An attacker rotating X-Device-ID gets a fresh per-owner bucket every
    request, so only the per-IP bucket can stop unbounded verification calls.
    All requests here share the TestClient IP; the 4th must be rejected 429
    even though every device_id is unique.
    """
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)
    monkeypatch.setattr("app.core.rate_limiter.MAX_REQUESTS_PER_WINDOW", 3)

    with _override_auth_user(None):
        with (
            patch(
                "app.api.billing.get_product_purchase",
                new_callable=AsyncMock,
                return_value=dict(VALID_PURCHASE),
            ),
            patch("app.api.billing.grant_guest_credits", AsyncMock(return_value=100)),
            patch("app.api.billing.get_guest_wallet", AsyncMock(return_value={"credits": 100})),
            patch("app.api.billing.log_transaction"),
        ):
            responses = [
                _post(client, headers={"X-Device-ID": f"rotating-device-{i}"}) for i in range(4)
            ]

    assert [r.status_code for r in responses[:3]] == [200, 200, 200]
    assert responses[3].status_code == 429


# ---------------------------------------------------------------------------
# Auth boundary
# ---------------------------------------------------------------------------


def test_no_auth_and_no_device_id_rejected(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", SA_JSON)

    with _override_auth_user(None):
        response = _post(client)  # no Bearer, no X-Device-ID

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Device-ID"}
