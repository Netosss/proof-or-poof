"""
Tests for:
  POST /api/v1/reports/share
  GET  /api/v1/reports/share/{report_id}
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# POST /api/v1/reports/share
# ---------------------------------------------------------------------------


def test_create_share_link_valid(client, mock_redis):
    short_id = "abc12345"
    payload = {"summary": "No AI Detected", "confidence_score": 0.95, "evidence_chain": []}
    mock_redis.set(f"report:{short_id}", json.dumps(payload))

    with patch("app.services.reports_service.firebase_module") as mock_fb_mod:
        mock_db = __import__("tests.mocks.firebase_mock", fromlist=["MockFirestore"]).MockFirestore()
        mock_fb_mod.db = mock_db

        response = client.post(
            "/api/v1/reports/share", json={"short_id": short_id}
        )

    assert response.status_code == 201
    assert response.json()["report_id"] == short_id


def test_create_share_link_idempotent(client, mock_redis):
    short_id = "idem1234"
    mock_redis.set(f"is_shared:{short_id}", "1")

    with patch("app.services.reports_service.firebase_module") as mock_fb_mod:
        mock_db = __import__("tests.mocks.firebase_mock", fromlist=["MockFirestore"]).MockFirestore()
        mock_fb_mod.db = mock_db

        response = client.post(
            "/api/v1/reports/share", json={"short_id": short_id}
        )

    assert response.status_code == 201
    assert response.json()["report_id"] == short_id


def test_create_share_link_cache_miss(client, mock_redis):
    # Nothing set in redis — should return 404
    with patch("app.services.reports_service.firebase_module") as mock_fb_mod:
        mock_db = __import__("tests.mocks.firebase_mock", fromlist=["MockFirestore"]).MockFirestore()
        mock_fb_mod.db = mock_db

        response = client.post(
            "/api/v1/reports/share", json={"short_id": "notexist"}
        )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/reports/share/{report_id}
# ---------------------------------------------------------------------------


def test_get_shared_report_found(client, mock_firebase):
    report_id = "report001"
    mock_firebase.seed(
        "shared_reports",
        report_id,
        {
            "summary": "No AI Detected",
            "confidence_score": 0.99,
            "evidence_chain": [],
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(days=30),
        },
    )

    response = client.get(f"/api/v1/reports/share/{report_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "No AI Detected"
    # created_at and expires_at should be stripped
    assert "created_at" not in data
    assert "expires_at" not in data


def test_get_shared_report_not_found(client, mock_firebase):
    response = client.get("/api/v1/reports/share/doesnotexist")
    assert response.status_code == 404


def test_get_shared_report_near_expiry_triggers_extend(client, mock_firebase, mock_redis):
    """When fewer than report_extend_threshold_days remain, extend_report_ttl is scheduled."""
    report_id = "expiring01"
    mock_firebase.seed(
        "shared_reports",
        report_id,
        {
            "summary": "Likely AI-Generated",
            "confidence_score": 0.9,
            "evidence_chain": [],
            "created_at": datetime.now(timezone.utc),
            # Only 1 day left — below the 3-day threshold
            "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
        },
    )

    with patch("app.api.reports.extend_report_ttl") as mock_extend:
        response = client.get(f"/api/v1/reports/share/{report_id}")

    assert response.status_code == 200
    # Background task should have been registered (called by TestClient synchronously)
    mock_extend.assert_called_once()
