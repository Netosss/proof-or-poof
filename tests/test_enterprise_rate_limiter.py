"""Unit tests for enterprise rate limiter."""

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_rate_limit_under_budget_redis(mock_redis):
    from app.core.enterprise_rate_limiter import check_and_track
    info = await check_and_track("cred-1", limit_per_min=3)
    assert info.limit == 3
    assert info.remaining == 2


@pytest.mark.asyncio
async def test_rate_limit_exceeded_redis(mock_redis):
    from app.core.enterprise_rate_limiter import check_and_track

    for _ in range(3):
        await check_and_track("cred-x", limit_per_min=3)

    with pytest.raises(HTTPException) as exc:
        await check_and_track("cred-x", limit_per_min=3)
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


@pytest.mark.asyncio
async def test_rate_limit_in_memory_fallback(monkeypatch):
    """When Redis is None, falls back to in-memory bucket."""
    from app.integrations import redis_client as rc
    monkeypatch.setattr(rc, "client", None)

    from app.core.enterprise_rate_limiter import check_and_track, _memory_buckets
    _memory_buckets.clear()

    info = await check_and_track("cred-mem", limit_per_min=2)
    assert info.remaining == 1
    info = await check_and_track("cred-mem", limit_per_min=2)
    assert info.remaining == 0

    with pytest.raises(HTTPException) as exc:
        await check_and_track("cred-mem", limit_per_min=2)
    assert exc.value.status_code == 429


def test_headers_for():
    from app.core.enterprise_rate_limiter import RateLimitInfo, headers_for
    info = RateLimitInfo(limit=60, remaining=42, reset_seconds=30)
    h = headers_for(info)
    assert h["X-RateLimit-Limit"] == "60"
    assert h["X-RateLimit-Remaining"] == "42"
    assert "X-RateLimit-Reset" in h
    assert "Retry-After" not in h
