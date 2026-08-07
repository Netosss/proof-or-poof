"""
Tests for the shared ad-reward daily cap and the guest wallet top-up.

Covers the paths that previously had NO test at all:
  - `_apply_ad_reward` cap semantics (the cap itself was only ever mocked out)
  - `POST /api/ads/reward` (zero coverage before this file)
  - the guest welcome-bonus double-mint in `_apply_guest_topup`
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.api.credits import _apply_ad_reward
from app.config import settings
from app.services.credits_service import _apply_guest_topup

AUTH_USER = {"uid": "user-1", "email": "u@example.com"}


@contextmanager
def _override_auth_user(user: dict | None):
    """Temporarily override get_current_user on the FastAPI app."""
    from app.core.firebase_auth import get_current_user
    from app.main import app

    async def _fake():
        return user

    app.dependency_overrides[get_current_user] = _fake
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# _apply_ad_reward — cap semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cap_allows_exactly_the_daily_limit_then_refuses(mock_firebase):
    """3 grants succeed; the 4th returns None (capped) and grants nothing."""
    limit = settings.ad_reward_daily_limit
    with patch(
        "app.api.credits.grant_credits", new_callable=AsyncMock, return_value=100
    ) as m_grant:
        for expected_count in range(1, limit + 1):
            credits, count, balance = await _apply_ad_reward("user-1", "uid")
            assert credits == settings.ad_reward_credits
            assert count == expected_count
            assert balance == 100

        credits, count, balance = await _apply_ad_reward("user-1", "uid")

    assert credits is None, "grant past the cap must be refused"
    assert balance is None
    assert count == limit
    assert m_grant.await_count == limit, "no credit engine call past the cap"


@pytest.mark.asyncio
async def test_cap_is_namespaced_by_kind(mock_firebase):
    """
    The griefing vector: `device:<a-real-firebase-uid>` must not consume the
    signed-in user's cap. Same id, different kind -> independent counters.
    """
    with patch("app.api.credits.grant_credits", new_callable=AsyncMock, return_value=1):
        with patch(
            "app.api.credits.grant_guest_credits", new_callable=AsyncMock, return_value=1
        ):
            # Exhaust the signed-in subject.
            for _ in range(settings.ad_reward_daily_limit):
                await _apply_ad_reward("collide", "uid")
            capped, _, _ = await _apply_ad_reward("collide", "uid")
            assert capped is None

            # Same id as a guest subject — must still have its own full budget.
            credits, count, _ = await _apply_ad_reward("collide", "device")

    assert credits == settings.ad_reward_credits
    assert count == 1


@pytest.mark.asyncio
async def test_guest_subject_grants_to_guest_wallet_not_credit_engine(mock_firebase):
    with (
        patch("app.api.credits.grant_credits", new_callable=AsyncMock) as m_engine,
        patch(
            "app.api.credits.grant_guest_credits", new_callable=AsyncMock, return_value=55
        ) as m_guest,
    ):
        credits, _, balance = await _apply_ad_reward("dev-1", "device")

    assert credits == settings.ad_reward_credits
    assert balance == 55
    m_guest.assert_awaited_once()
    m_engine.assert_not_awaited(), "guests have no ledger — must not hit the credit engine"


@pytest.mark.asyncio
async def test_failed_grant_releases_the_cap_slot(mock_firebase):
    """
    The slot is committed before the grant. If the grant throws, it must be
    handed back — otherwise a transient Firestore error permanently burns one of
    the user's daily rewards.
    """
    with patch(
        "app.api.credits.grant_credits",
        new_callable=AsyncMock,
        side_effect=RuntimeError("firestore hiccup"),
    ):
        with pytest.raises(RuntimeError):
            await _apply_ad_reward("user-2", "uid")

    # The slot came back, so a retry starts from count 0 again.
    with patch("app.api.credits.grant_credits", new_callable=AsyncMock, return_value=20):
        credits, count, _ = await _apply_ad_reward("user-2", "uid")

    assert credits == settings.ad_reward_credits
    assert count == 1, "cap slot was not released"


# ---------------------------------------------------------------------------
# POST /api/ads/reward — previously zero coverage
# ---------------------------------------------------------------------------


def test_ads_reward_requires_auth(client):
    """Bearer-only: a guest gets 401, never a grant."""
    with patch("app.api.credits._apply_ad_reward", new_callable=AsyncMock) as m_grant:
        resp = client.post("/api/ads/reward")

    assert resp.status_code in (401, 403)
    m_grant.assert_not_awaited()


def test_ads_reward_grants_and_reports_balance(client):
    with _override_auth_user(AUTH_USER):
        with patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            return_value=(20, 1, 60),
        ):
            resp = client.post("/api/ads/reward", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 200
    assert resp.json() == {"credits_granted": 20, "new_balance": 60, "rewards_today": 1}


def test_ads_reward_capped_returns_429(client):
    """Cap reached -> 429, and the client shows 'daily limit', not a generic error."""
    limit = settings.ad_reward_daily_limit
    with _override_auth_user(AUTH_USER):
        with patch(
            "app.api.credits._apply_ad_reward",
            new_callable=AsyncMock,
            return_value=(None, limit, None),
        ):
            resp = client.post("/api/ads/reward", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Guest welcome-bonus double-mint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guest_topup_does_not_remint_the_welcome_bonus(mock_firebase):
    """
    Regression: `_apply_guest_topup` used to create a missing wallet with
    `welcome_credits + amount`, so the FIRST grant to any device id the server
    had never seen paid the welcome bonus again — on top of the one
    `get_guest_wallet` already grants on first read. An attacker rotating device
    ids harvested a fresh bonus per id, and it silently inflated every genuine
    first guest purchase on the billing path.
    """
    balance = await _apply_guest_topup("brand-new-device", 20)
    assert balance == 20, "welcome bonus must NOT be minted by a top-up"


@pytest.mark.asyncio
async def test_guest_topup_can_still_include_welcome_explicitly(mock_firebase):
    balance = await _apply_guest_topup("another-new-device", 20, include_welcome=True)
    assert balance == settings.welcome_credits + 20


@pytest.mark.asyncio
async def test_guest_topup_accumulates_on_an_existing_wallet(mock_firebase):
    first = await _apply_guest_topup("dev-existing", 20)
    second = await _apply_guest_topup("dev-existing", 20)
    assert first == 20
    assert second == 40
