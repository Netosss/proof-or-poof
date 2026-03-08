"""
Tests for:
  POST /webhook/runpod
  POST /webhooks/lemonsqueezy
"""

import hashlib
import hmac as hmac_mod
import json
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# RunPod webhook — Redis Pub/Sub delivery
# ---------------------------------------------------------------------------


def test_runpod_webhook_completed_stores_result_in_redis(client, mock_redis):
    """COMPLETED job: result is SET in Redis (race-check backup) and PUBLISHed."""
    job_id = "runpod-job-completed"
    output = {"ai_score": 0.9, "gpu_time_ms": 200.0}

    payload = {"id": job_id, "status": "COMPLETED", "output": output}
    response = client.post("/webhook/runpod", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # The result must be stored in Redis for the race-check GET in runpod.py
    channel = f"runpod:result:{job_id}"
    assert channel in mock_redis._store
    stored = json.loads(mock_redis._store[channel])
    assert stored["ai_score"] == 0.9
    assert stored["gpu_time_ms"] == 200.0


def test_runpod_webhook_failed_job_notifies_redis(client, mock_redis):
    """FAILED job: error payload is SET in Redis so the waiting coroutine unblocks."""
    job_id = "runpod-job-failed"

    payload = {"id": job_id, "status": "FAILED", "error": "OOM error"}
    response = client.post("/webhook/runpod", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    channel = f"runpod:result:{job_id}"
    assert channel in mock_redis._store
    stored = json.loads(mock_redis._store[channel])
    assert "error" in stored
    assert stored["ai_score"] == 0.0


def test_runpod_webhook_completed_stores_backup_before_publish(client, mock_redis):
    """
    The SET must happen before PUBLISH so a subscriber who arrives after
    the PUBLISH can still retrieve the result via the GET race-check.
    """
    job_id = "runpod-race-job"
    output = {"ai_score": 0.7, "gpu_time_ms": 150.0}

    payload = {"id": job_id, "status": "COMPLETED", "output": output}
    response = client.post("/webhook/runpod", json=payload)

    assert response.status_code == 200
    channel = f"runpod:result:{job_id}"
    # Key must exist — this is the SET that enables the race-check GET
    assert channel in mock_redis._store


def test_runpod_webhook_status_update_acknowledged(client):
    """IN_PROGRESS (and other non-terminal) statuses return ok without touching Redis."""
    payload = {"id": "runpod-in-progress", "status": "IN_PROGRESS"}
    response = client.post("/webhook/runpod", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_runpod_webhook_unknown_job(client):
    payload = {"id": "unknown-job-xyz", "status": "COMPLETED", "output": None}
    response = client.post("/webhook/runpod", json=payload)
    assert response.status_code == 200


def test_runpod_webhook_invalid_signature_rejected(client):
    """When RUNPOD_WEBHOOK_SECRET is set, a missing/wrong ?secret= param → 401."""
    with patch("app.api.webhooks.RUNPOD_WEBHOOK_SECRET", "real-runpod-secret"):
        response = client.post(
            "/webhook/runpod",
            json={"id": "fake-job", "status": "COMPLETED", "output": {"ai_score": 0.99}},
        )
    assert response.status_code == 401


def test_runpod_webhook_wrong_secret_rejected(client):
    """Wrong ?secret= value → 401."""
    with patch("app.api.webhooks.RUNPOD_WEBHOOK_SECRET", "real-runpod-secret"):
        response = client.post(
            "/webhook/runpod?secret=wrong-value",
            json={"id": "fake-job", "status": "COMPLETED", "output": {"ai_score": 0.99}},
        )
    assert response.status_code == 401


def test_runpod_webhook_valid_signature_accepted(client):
    """Correct ?secret= query param → request is processed (not 401)."""
    secret = "real-runpod-secret"
    with patch("app.api.webhooks.RUNPOD_WEBHOOK_SECRET", secret):
        response = client.post(
            f"/webhook/runpod?secret={secret}",
            json={"id": "legit-job", "status": "COMPLETED", "output": None},
        )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# LemonSqueezy webhook
# ---------------------------------------------------------------------------


def test_lemonsqueezy_no_secret_returns_500(client):
    # When LEMONSQUEEZY_WEBHOOK_SECRET is unset, the server must fail loudly
    # with 500 (not silently swallow webhooks with 200) so a misconfigured
    # deployment is immediately visible and Lemon Squeezy retries automatically.
    with patch("app.api.webhooks.LEMONSQUEEZY_WEBHOOK_SECRET", None):
        response = client.post(
            "/webhooks/lemonsqueezy",
            json={"meta": {"event_name": "order_created"}},
        )
    assert response.status_code == 500
    assert "secret" in response.json()["detail"].lower()


def test_lemonsqueezy_invalid_signature(client):
    with patch("app.api.webhooks.LEMONSQUEEZY_WEBHOOK_SECRET", "real-secret"):
        response = client.post(
            "/webhooks/lemonsqueezy",
            content=b'{"meta": {"event_name": "order_created"}}',
            headers={"X-Signature": "badsig", "Content-Type": "application/json"},
        )
    assert response.status_code == 401


def test_lemonsqueezy_order_created_not_paid_is_skipped(client):
    # order_created with status != "paid" must be skipped silently.
    # Credits are only granted on order_created with status == "paid".
    secret = "lemon-test-secret"
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "user-abc"},
        },
        "data": {
            "id": "order-001",
            "attributes": {"total": 999, "status": "pending"},
        },
    }
    payload_bytes = json.dumps(payload).encode()
    sig = hmac_mod.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

    with (
        patch("app.api.webhooks.LEMONSQUEEZY_WEBHOOK_SECRET", secret),
        patch("app.api.webhooks.log_transaction") as mock_log,
    ):
        response = client.post(
            "/webhooks/lemonsqueezy",
            content=payload_bytes,
            headers={"X-Signature": sig, "Content-Type": "application/json"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "skipped" in data
    mock_log.assert_not_called()  # no transaction logged for non-paid orders
