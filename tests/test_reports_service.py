"""
Pure unit tests for app/services/reports_service.py.

Redis and Firebase are provided via mock fixtures.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException

SHORT_ID = "rpt12345"
REPORT_ID = "rpt12345"

_REPORT_PAYLOAD = {
    "summary": "No AI Detected",
    "confidence_score": 0.99,
    "evidence_chain": [],
}


def _seed_redis_report(mock_redis, short_id, payload=None):
    if payload is None:
        payload = _REPORT_PAYLOAD
    mock_redis.set(f"report:{short_id}", json.dumps(payload))


# ---------------------------------------------------------------------------
# create_share_link
# ---------------------------------------------------------------------------


def test_create_share_link_valid(mock_firebase, mock_redis, monkeypatch):
    from app.integrations import firebase as fb, redis_client as rc
    monkeypatch.setattr(fb, "db", mock_firebase)
    monkeypatch.setattr(rc, "client", mock_redis)

    _seed_redis_report(mock_redis, SHORT_ID)

    from app.services.reports_service import create_share_link
    result = create_share_link(SHORT_ID)

    assert result["report_id"] == SHORT_ID
    # Firestore doc should have been written
    doc = mock_firebase.collection("shared_reports").document(SHORT_ID).get()
    assert doc.exists


def test_create_share_link_idempotent(mock_firebase, mock_redis, monkeypatch):
    from app.integrations import firebase as fb, redis_client as rc
    monkeypatch.setattr(fb, "db", mock_firebase)
    monkeypatch.setattr(rc, "client", mock_redis)

    # Idempotency key already exists → early return
    mock_redis.set(f"is_shared:{SHORT_ID}", "1")

    from app.services.reports_service import create_share_link
    result = create_share_link(SHORT_ID)

    assert result["report_id"] == SHORT_ID


def test_create_share_link_cache_miss_raises_404(mock_firebase, mock_redis, monkeypatch):
    from app.integrations import firebase as fb, redis_client as rc
    monkeypatch.setattr(fb, "db", mock_firebase)
    monkeypatch.setattr(rc, "client", mock_redis)

    from app.services.reports_service import create_share_link
    with pytest.raises(HTTPException) as exc:
        create_share_link("nope_id")
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# get_shared_report
# ---------------------------------------------------------------------------


def test_get_shared_report_found_plenty_ttl(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    mock_firebase.seed(
        "shared_reports",
        REPORT_ID,
        {
            **_REPORT_PAYLOAD,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(days=30),
        },
    )

    from app.services.reports_service import get_shared_report
    data, should_extend = get_shared_report(REPORT_ID)

    assert data["summary"] == "No AI Detected"
    assert should_extend is False
    assert "created_at" not in data
    assert "expires_at" not in data


def test_get_shared_report_near_expiry_flags_extend(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    mock_firebase.seed(
        "shared_reports",
        REPORT_ID,
        {
            **_REPORT_PAYLOAD,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(days=1),  # < 3-day threshold
        },
    )

    from app.services.reports_service import get_shared_report
    _, should_extend = get_shared_report(REPORT_ID)

    assert should_extend is True


def test_get_shared_report_not_found_raises_404(mock_firebase, monkeypatch):
    from app.integrations import firebase as fb
    monkeypatch.setattr(fb, "db", mock_firebase)

    from app.services.reports_service import get_shared_report
    with pytest.raises(HTTPException) as exc:
        get_shared_report("nonexistent-id")
    assert exc.value.status_code == 404
