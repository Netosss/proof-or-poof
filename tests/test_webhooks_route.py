"""
Tests for:
  POST /webhook/runpod
  POST /webhooks/lemonsqueezy
"""

import hashlib
import hmac as hmac_mod
import json
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# RunPod webhook
# ---------------------------------------------------------------------------


def _make_runpod_future():
    m = MagicMock()
    m.done.return_value = False
    return m


def test_runpod_webhook_completed_resolves_future(client):
    from app.integrations.runpod import pending_jobs

    job_id = "runpod-job-completed"
    mock_future = _make_runpod_future()
    pending_jobs[job_id] = (mock_future, time.time())

    try:
        payload = {
            "id": job_id,
            "status": "COMPLETED",
            "output": {"ai_score": 0.9, "gpu_time_ms": 200.0},
        }
        response = client.post("/webhook/runpod", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_future.set_result.assert_called_once_with(
            {"ai_score": 0.9, "gpu_time_ms": 200.0}
        )
    finally:
        pending_jobs.pop(job_id, None)


def test_runpod_webhook_failed_job(client):
    from app.integrations.runpod import pending_jobs

    job_id = "runpod-job-failed"
    mock_future = _make_runpod_future()
    pending_jobs[job_id] = (mock_future, time.time())

    try:
        payload = {"id": job_id, "status": "FAILED", "error": "OOM error"}
        response = client.post("/webhook/runpod", json=payload)
        assert response.status_code == 200
        args = mock_future.set_result.call_args[0][0]
        assert args["error"] == "Job failed"
        assert args["ai_score"] == 0.0
    finally:
        pending_jobs.pop(job_id, None)


def test_runpod_webhook_unknown_job(client):
    payload = {"id": "unknown-job-xyz", "status": "COMPLETED", "output": None}
    response = client.post("/webhook/runpod", json=payload)
    assert response.status_code == 200


def test_runpod_webhook_race_condition_buffers_result(client):
    """Webhook arrives before the job is registered in pending_jobs → buffered."""
    from app.integrations.runpod import webhook_result_buffer

    job_id = "runpod-race-job"
    webhook_result_buffer.pop(job_id, None)

    try:
        payload = {
            "id": job_id,
            "status": "COMPLETED",
            "output": {"ai_score": 0.7, "gpu_time_ms": 150.0},
        }
        response = client.post("/webhook/runpod", json=payload)
        assert response.status_code == 200
        assert job_id in webhook_result_buffer
        buffered_output, _ = webhook_result_buffer[job_id]
        assert buffered_output["ai_score"] == 0.7
    finally:
        webhook_result_buffer.pop(job_id, None)


def test_runpod_webhook_status_update_acknowledged(client):
    from app.integrations.runpod import pending_jobs

    job_id = "runpod-in-progress"
    mock_future = _make_runpod_future()
    pending_jobs[job_id] = (mock_future, time.time())

    try:
        payload = {"id": job_id, "status": "IN_PROGRESS"}
        response = client.post("/webhook/runpod", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "acknowledged"
    finally:
        pending_jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# LemonSqueezy webhook
# ---------------------------------------------------------------------------


def test_lemonsqueezy_no_secret_ignored(client):
    with patch("app.api.webhooks.LEMONSQUEEZY_WEBHOOK_SECRET", None):
        response = client.post(
            "/webhooks/lemonsqueezy",
            json={"meta": {"event_name": "order_created"}},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


def test_lemonsqueezy_invalid_signature(client):
    with patch("app.api.webhooks.LEMONSQUEEZY_WEBHOOK_SECRET", "real-secret"):
        response = client.post(
            "/webhooks/lemonsqueezy",
            content=b'{"meta": {"event_name": "order_created"}}',
            headers={"X-Signature": "badsig", "Content-Type": "application/json"},
        )
    assert response.status_code == 401


def test_lemonsqueezy_valid_order_created(client):
    secret = "lemon-test-secret"
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "user-abc"},
        },
        "data": {
            "id": "order-001",
            "attributes": {"total": 999},
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
    assert response.json()["status"] == "ok"
    mock_log.assert_called_once()
    args = mock_log.call_args[0]
    assert args[0] == "LEMONSQUEEZY"
    assert abs(args[1] - 9.99) < 0.01  # 999 cents → $9.99
