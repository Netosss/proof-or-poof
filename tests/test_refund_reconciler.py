"""
Tests for voided-purchase reconciliation (M5).

Closes buy -> spend -> refund -> keep the credits. The Google Play
voided-purchases call is always mocked — no network I/O.
"""

from unittest.mock import AsyncMock, patch

import pytest

ORDER_ID = "GPA.1111-2222-3333-44444"
UID = "auth-user-refund-001"
DEVICE = "guest-device-refund-001"


def _voided(order_id=ORDER_ID, reason=1):
    return [{"orderId": order_id, "voidedReason": reason, "purchaseToken": "tok"}]


def _patches(voided, grant=None, guest_grant=None):
    """Common patch set: configured + a canned voided list + debit spies."""
    return (
        patch("app.services.refund_reconciler.is_configured", return_value=True),
        patch(
            "app.services.refund_reconciler.list_voided_purchases",
            new_callable=AsyncMock,
            return_value=voided,
        ),
        patch(
            "app.services.refund_reconciler.grant_credits",
            grant or AsyncMock(return_value=0),
        ),
        patch(
            "app.services.refund_reconciler.grant_guest_credits",
            guest_grant or AsyncMock(return_value=0),
        ),
    )


@pytest.mark.asyncio
async def test_voided_order_reverses_credits(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("processed_play_orders", ORDER_ID, {
        "owner_id": UID, "credits": 500, "is_guest": False, "reversed": False,
    })

    grant = AsyncMock(return_value=-500)
    p = _patches(_voided(), grant=grant)
    with p[0], p[1], p[2], p[3]:
        from app.services.refund_reconciler import reconcile_voided_purchases
        result = await reconcile_voided_purchases()

    assert result["reversed"] == 1
    # Debited, not credited — the sign is the whole point.
    grant.assert_awaited_once_with(UID, -500, "google_play_refund", ORDER_ID)


@pytest.mark.asyncio
async def test_guest_order_debits_the_guest_wallet(mock_firebase, monkeypatch):
    """is_guest routes the debit to guest_wallets. Getting this wrong debits a
    stranger, since a uid and a device id are indistinguishable as strings."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("processed_play_orders", ORDER_ID, {
        "owner_id": DEVICE, "credits": 200, "is_guest": True, "reversed": False,
    })

    grant = AsyncMock()
    guest = AsyncMock(return_value=-200)
    p = _patches(_voided(), grant=grant, guest_grant=guest)
    with p[0], p[1], p[2], p[3]:
        from app.services.refund_reconciler import reconcile_voided_purchases
        await reconcile_voided_purchases()

    guest.assert_awaited_once_with(DEVICE, -200)
    grant.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_is_idempotent(mock_firebase, monkeypatch):
    """It runs on a schedule and Google reports the same void for 30 days, so
    re-reversal is the default failure mode, not an edge case."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("processed_play_orders", ORDER_ID, {
        "owner_id": UID, "credits": 500, "is_guest": False, "reversed": False,
    })

    grant = AsyncMock(return_value=-500)
    p = _patches(_voided(), grant=grant)
    with p[0], p[1], p[2], p[3]:
        from app.services.refund_reconciler import reconcile_voided_purchases
        first = await reconcile_voided_purchases()
        second = await reconcile_voided_purchases()

    assert first["reversed"] == 1
    assert second["reversed"] == 0
    assert second["already_reversed"] == 1
    grant.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_order_is_not_debited(mock_firebase, monkeypatch):
    """Voided, but we never granted for it — nothing to claw back."""
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    grant = AsyncMock()
    p = _patches(_voided(order_id="GPA.never-granted"), grant=grant)
    with p[0], p[1], p[2], p[3]:
        from app.services.refund_reconciler import reconcile_voided_purchases
        result = await reconcile_voided_purchases()

    assert result["unknown_orders"] == 1
    assert result["reversed"] == 0
    grant.assert_not_awaited()


@pytest.mark.asyncio
async def test_debit_failure_unmarks_so_a_retry_still_reverses(mock_firebase, monkeypatch):
    """
    The claim is taken BEFORE the debit, so a failed debit must release it —
    otherwise the order reads as already-reversed forever and the refunded
    credits are never clawed back.
    """
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)
    mock_firebase.seed("processed_play_orders", ORDER_ID, {
        "owner_id": UID, "credits": 500, "is_guest": False, "reversed": False,
    })

    failing = AsyncMock(side_effect=RuntimeError("firestore down"))
    p = _patches(_voided(), grant=failing)
    with p[0], p[1], p[2], p[3]:
        from app.services.refund_reconciler import reconcile_voided_purchases
        first = await reconcile_voided_purchases()

    assert first["failed"] == 1
    assert first["reversed"] == 0

    succeeding = AsyncMock(return_value=-500)
    p2 = _patches(_voided(), grant=succeeding)
    with p2[0], p2[1], p2[2], p2[3]:
        from app.services.refund_reconciler import reconcile_voided_purchases
        retry = await reconcile_voided_purchases()

    assert retry["reversed"] == 1
    succeeding.assert_awaited_once_with(UID, -500, "google_play_refund", ORDER_ID)


# ---------------------------------------------------------------------------
# Route guard
# ---------------------------------------------------------------------------


def test_reconcile_route_rejects_wrong_secret(client, monkeypatch):
    monkeypatch.setenv("RECHARGE_SECRET_KEY", "right-secret")
    response = client.post(
        "/api/billing/reconcile-refunds", headers={"X-Admin-Secret": "wrong"}
    )
    assert response.status_code == 403


def test_reconcile_route_rejects_missing_secret(client, monkeypatch):
    monkeypatch.setenv("RECHARGE_SECRET_KEY", "right-secret")
    response = client.post("/api/billing/reconcile-refunds")
    assert response.status_code == 403


def test_reconcile_route_rejects_out_of_range_lookback(client, monkeypatch):
    """Above 30 is pointless — Google retains 30 days — and 0 is a typo."""
    monkeypatch.setenv("RECHARGE_SECRET_KEY", "right-secret")
    response = client.post(
        "/api/billing/reconcile-refunds?lookback_days=365",
        headers={"X-Admin-Secret": "right-secret"},
    )
    assert response.status_code == 422
