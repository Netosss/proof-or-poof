"""Unit tests for `maybe_raise_rate_limit` — the helper that bumps a
partner's `rate_limit_per_min` ceiling on tier purchase.

Behaviour under test:
  - Raises when the proposed limit exceeds the current ceiling.
  - Treats `partner.rate_limit_per_min == None` as the system default
    (`settings.enterprise_default_rate_limit_per_min`) for comparison.
  - Never lowers — a Scale customer buying a Starter top-up keeps Scale.
  - No-op when the partner doesn't exist (logs but does not raise).
  - Idempotent: repeated calls with the same proposed limit don't churn.
"""

import pytest

from app.services.enterprise_partners import maybe_raise_rate_limit


def _seed(mock_firebase, partner_id: str = "p-rl", rate_limit_per_min=None):
    mock_firebase.seed(
        "enterprise_partners",
        partner_id,
        {
            "company_name": "Acme",
            "contact_email": "ops@acme.com",
            "credit_balance": 1000,
            "status": "active",
            "credits_version": 1,
            "rate_limit_per_min": rate_limit_per_min,
            "firebase_uid": "fb-acme",
        },
    )


@pytest.mark.asyncio
async def test_raises_from_default_to_pro(mock_firebase):
    """Starter partner (null override → system default 60) buys Pro (120)."""
    _seed(mock_firebase, rate_limit_per_min=None)
    result = await maybe_raise_rate_limit("p-rl", 120)
    assert result == 120
    doc = mock_firebase.collection("enterprise_partners")._docs["p-rl"]
    assert doc["rate_limit_per_min"] == 120


@pytest.mark.asyncio
async def test_raises_from_pro_to_scale(mock_firebase):
    _seed(mock_firebase, rate_limit_per_min=120)
    result = await maybe_raise_rate_limit("p-rl", 300)
    assert result == 300
    doc = mock_firebase.collection("enterprise_partners")._docs["p-rl"]
    assert doc["rate_limit_per_min"] == 300


@pytest.mark.asyncio
async def test_does_not_lower_when_proposed_below_current(mock_firebase):
    """Scale customer (300) buying a Pro top-up (120) keeps Scale's ceiling."""
    _seed(mock_firebase, rate_limit_per_min=300)
    result = await maybe_raise_rate_limit("p-rl", 120)
    assert result == 300
    doc = mock_firebase.collection("enterprise_partners")._docs["p-rl"]
    assert doc["rate_limit_per_min"] == 300


@pytest.mark.asyncio
async def test_idempotent_when_proposed_equals_current(mock_firebase):
    _seed(mock_firebase, rate_limit_per_min=120)
    result = await maybe_raise_rate_limit("p-rl", 120)
    assert result == 120
    doc = mock_firebase.collection("enterprise_partners")._docs["p-rl"]
    assert doc["rate_limit_per_min"] == 120


@pytest.mark.asyncio
async def test_no_op_when_proposed_below_system_default(mock_firebase):
    """A custom variant with a tiny ceiling shouldn't downgrade an
    unmodified partner who's already on the system default."""
    from app.config import settings

    _seed(mock_firebase, rate_limit_per_min=None)
    below_default = settings.enterprise_default_rate_limit_per_min - 1
    result = await maybe_raise_rate_limit("p-rl", max(1, below_default))
    # No write — effective_current was the system default.
    assert result == settings.enterprise_default_rate_limit_per_min
    doc = mock_firebase.collection("enterprise_partners")._docs["p-rl"]
    assert doc["rate_limit_per_min"] is None


@pytest.mark.asyncio
async def test_partner_not_found_returns_none(mock_firebase):
    """Missing partner: log and return None, never throw — billable mutations
    upstream MUST NOT roll back the credit grant because of a rate-limit
    bookkeeping miss."""
    result = await maybe_raise_rate_limit("p-does-not-exist", 120)
    assert result is None


@pytest.mark.asyncio
async def test_zero_or_negative_proposed_is_ignored(mock_firebase):
    """Defensive: zero or negative values from a misconfigured env var must
    never be written to the partner."""
    _seed(mock_firebase, rate_limit_per_min=120)
    assert (await maybe_raise_rate_limit("p-rl", 0)) is None
    assert (await maybe_raise_rate_limit("p-rl", -1)) is None
    doc = mock_firebase.collection("enterprise_partners")._docs["p-rl"]
    assert doc["rate_limit_per_min"] == 120
