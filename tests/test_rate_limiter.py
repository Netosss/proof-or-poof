"""
Pure unit tests for app/core/rate_limiter.py — in-memory fallback path only.

Redis client is set to None so the memory implementation is exercised.
Time is frozen with unittest.mock.patch to test window sliding without sleeping.
"""

import time
import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _uid() -> str:
    return f"rl_test_{uuid.uuid4().hex}"


def _with_null_redis(monkeypatch):
    from app.integrations import redis_client as rc
    monkeypatch.setattr(rc, "client", None)


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_first_request_passes(monkeypatch):
    _with_null_redis(monkeypatch)
    from app.core.rate_limiter import _rate_limits, check_rate_limit

    uid = _uid()
    _rate_limits.pop(uid, None)
    check_rate_limit(uid)  # should not raise


def test_requests_under_limit_pass(monkeypatch):
    _with_null_redis(monkeypatch)
    from app.config import settings
    from app.core.rate_limiter import _rate_limits, check_rate_limit

    uid = _uid()
    _rate_limits.pop(uid, None)
    for _ in range(settings.rate_limit_max_requests):
        check_rate_limit(uid)  # no exception


def test_request_exceeding_limit_raises_429(monkeypatch):
    _with_null_redis(monkeypatch)
    from app.config import settings
    from app.core.rate_limiter import _rate_limits, check_rate_limit

    uid = _uid()
    _rate_limits.pop(uid, None)

    for _ in range(settings.rate_limit_max_requests):
        check_rate_limit(uid)

    with pytest.raises(HTTPException) as exc:
        check_rate_limit(uid)
    assert exc.value.status_code == 429


# ---------------------------------------------------------------------------
# Sliding-window: new window resets the counter
# ---------------------------------------------------------------------------


def test_new_window_allows_requests_again(monkeypatch):
    _with_null_redis(monkeypatch)
    from app.config import settings
    from app.core.rate_limiter import RATE_LIMIT_WINDOW, _rate_limits, check_rate_limit

    uid = _uid()
    _rate_limits.pop(uid, None)

    # Fill the window
    for _ in range(settings.rate_limit_max_requests):
        check_rate_limit(uid)

    # Advance time past the window so all timestamps expire
    future_time = time.time() + RATE_LIMIT_WINDOW + 1
    with patch("app.core.rate_limiter.time") as mock_time:
        mock_time.time.return_value = future_time
        check_rate_limit(uid)  # should not raise in the new window


# ---------------------------------------------------------------------------
# Cleanup removes idle sessions
# ---------------------------------------------------------------------------


def test_cleanup_removes_idle_sessions(monkeypatch):
    _with_null_redis(monkeypatch)
    from app.core.rate_limiter import (
        RATE_LIMIT_WINDOW,
        _cleanup_all_limits,
        _rate_limits,
    )

    uid = _uid()
    _rate_limits[uid] = [time.time() - RATE_LIMIT_WINDOW - 5]  # stale entry

    _cleanup_all_limits(time.time())

    assert uid not in _rate_limits
