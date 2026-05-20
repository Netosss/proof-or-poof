"""End-to-end tests for POST /v1/analyze via the FastAPI TestClient.

These exercise the full pipeline wiring (auth → rate-limit → idempotency →
file validation → reserve_credit → detect → refund-on-failure → store) so any
regression in the order or interaction of those steps fails loudly.
"""

import hashlib
import hmac as hmac_mod
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest


# ─── Helpers ────────────────────────────────────────────────────────────────

# Boundary used in every multipart body — keeps signing deterministic.
_BOUNDARY = "----V1AnalyzeTest"


def _build_multipart(file_bytes: bytes, filename: str = "t.jpg",
                     content_type: str = "image/jpeg") -> bytes:
    parts = [
        f"--{_BOUNDARY}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        file_bytes,
        f"\r\n--{_BOUNDARY}--\r\n".encode(),
    ]
    return b"".join(parts)


def _sign(secret: str, ts: str, method: str, path: str, content_sha: str) -> str:
    payload = f"{ts}\n{method.upper()}\n{path}\n{content_sha.lower()}".encode()
    return hmac_mod.new(secret.encode(), payload, hashlib.sha256).hexdigest()


_FAKE_CRED = {
    "partner_id": "p1",
    "credential_id": "c1",
    "api_key_prefix": "fxl_test_aaa...zzz",
    "secret_key": "fxs_test_supersecret_e2e",
    "allowed_ips": [],
}


def _make_headers(body: bytes, idem_key: str = "idem-e2e-1") -> dict[str, str]:
    ts = str(int(time.time()))
    content_sha = hashlib.sha256(body).hexdigest()
    sig = _sign(_FAKE_CRED["secret_key"], ts, "POST", "/v1/analyze", content_sha)
    return {
        "X-FauxLens-Key": "fxl_test_aaa",
        "X-FauxLens-Timestamp": ts,
        "X-FauxLens-Content-SHA256": content_sha,
        "X-FauxLens-Signature": sig,
        "Idempotency-Key": idem_key,
        "Content-Type": f"multipart/form-data; boundary={_BOUNDARY}",
    }


def _seed_partner(mock_firebase, balance: int = 100):
    mock_firebase.seed(
        "enterprise_partners", _FAKE_CRED["partner_id"],
        {"company_name": "Acme", "contact_email": "ops@acme.com",
         "credit_balance": balance, "status": "active", "credits_version": 1,
         "firebase_uid": "fb-1"},
    )


@contextmanager
def _patch_auth_and_pipeline(detect_return=None, detect_raise: Exception | None = None):
    """Patch HMAC resolution + pipeline so the route runs without real crypto/Gemini."""
    from app.core import enterprise_auth as ea
    from app.api.enterprise import analyze as analyze_mod

    if detect_raise is not None:
        detect_mock = AsyncMock(side_effect=detect_raise)
    else:
        detect_mock = AsyncMock(return_value=detect_return or {
            "summary": "Likely Authentic",
            "confidence_score": 0.95,
            "is_short_circuited": False,
            "evidence_chain": [],
            "is_gemini_used": False,
            "is_cached": False,
            "gpu_time_ms": 0,
        })

    with (
        patch.object(ea, "resolve_credential", new=AsyncMock(return_value=_FAKE_CRED)),
        patch.object(ea, "touch_credential", new=AsyncMock(return_value=None)),
        patch.object(analyze_mod, "detect_ai_media", new=detect_mock),
        patch.object(analyze_mod, "validate_file", new=AsyncMock(return_value=True)),
    ):
        yield detect_mock


# ─── Tests ──────────────────────────────────────────────────────────────────

def test_v1_analyze_happy_path(client, mock_firebase, mock_redis, tiny_jpg):
    _seed_partner(mock_firebase, balance=100)
    body = _build_multipart(open(tiny_jpg, "rb").read())
    headers = _make_headers(body, idem_key="happy-1")

    with _patch_auth_and_pipeline():
        r = client.post("/v1/analyze", content=body, headers=headers)

    assert r.status_code == 200, r.text
    j = r.json()
    assert j["data"]["summary"] == "Likely Authentic"
    assert j["credits_remaining"] == 99
    assert j["idempotent_replay"] is False
    assert r.headers.get("X-RateLimit-Limit")
    assert r.headers.get("X-RateLimit-Remaining")


def test_v1_analyze_body_hash_mismatch(client, mock_firebase, mock_redis, tiny_jpg):
    _seed_partner(mock_firebase, balance=100)
    body = _build_multipart(open(tiny_jpg, "rb").read())
    headers = _make_headers(body, idem_key="mismatch-1")
    # Replace the SHA header AND re-sign so HMAC passes but body hash doesn't.
    forged_sha = "f" * 64
    headers["X-FauxLens-Content-SHA256"] = forged_sha
    headers["X-FauxLens-Signature"] = _sign(
        _FAKE_CRED["secret_key"], headers["X-FauxLens-Timestamp"],
        "POST", "/v1/analyze", forged_sha,
    )

    with _patch_auth_and_pipeline():
        r = client.post("/v1/analyze", content=body, headers=headers)

    assert r.status_code == 400
    # Stripe-style envelope on /v1/* errors
    err = r.json().get("error") or {}
    assert err.get("code") == "content_hash_mismatch"


def test_v1_analyze_refunds_on_pipeline_crash(client, mock_firebase, mock_redis, tiny_jpg):
    """Pipeline crash → 500 + credit refunded via the idempotent ledger."""
    _seed_partner(mock_firebase, balance=100)
    body = _build_multipart(open(tiny_jpg, "rb").read())
    headers = _make_headers(body, idem_key="crash-1")

    from app.api.enterprise import analyze as analyze_mod

    with _patch_auth_and_pipeline(detect_raise=RuntimeError("simulated_crash")), \
         patch.object(analyze_mod, "refund_credit", new=AsyncMock(return_value=100)) as ref_mock:
        r = client.post("/v1/analyze", content=body, headers=headers)

    assert r.status_code == 500
    assert ref_mock.called, "refund_credit MUST be called when the pipeline crashes"


def test_v1_analyze_refunds_on_analysis_failed_verdict(client, mock_firebase, mock_redis, tiny_jpg):
    """Synthetic 'Analysis Failed' verdict → 422 + refund."""
    _seed_partner(mock_firebase, balance=100)
    body = _build_multipart(open(tiny_jpg, "rb").read())
    headers = _make_headers(body, idem_key="failed-1")

    failed_result = {
        "summary": "Analysis Failed",
        "confidence_score": 0.0,
        "is_short_circuited": False,
        "evidence_chain": [],
        "is_gemini_used": False,
        "is_cached": False,
        "gpu_time_ms": 0,
    }
    from app.api.enterprise import analyze as analyze_mod

    with _patch_auth_and_pipeline(detect_return=failed_result), \
         patch.object(analyze_mod, "refund_credit", new=AsyncMock(return_value=100)) as ref_mock:
        r = client.post("/v1/analyze", content=body, headers=headers)

    assert r.status_code == 422
    err = r.json().get("error") or {}
    assert err.get("code") == "analysis_failed"
    assert ref_mock.called


def test_v1_analyze_missing_idempotency_key_rejected(client, mock_firebase, mock_redis, tiny_jpg):
    _seed_partner(mock_firebase, balance=100)
    body = _build_multipart(open(tiny_jpg, "rb").read())
    headers = _make_headers(body)
    del headers["Idempotency-Key"]

    with _patch_auth_and_pipeline():
        r = client.post("/v1/analyze", content=body, headers=headers)

    assert r.status_code == 400
    err = r.json().get("error") or {}
    assert err.get("code") == "missing_idempotency_key"


def test_v1_analyze_insufficient_credits(client, mock_firebase, mock_redis, tiny_jpg):
    _seed_partner(mock_firebase, balance=0)
    body = _build_multipart(open(tiny_jpg, "rb").read())
    headers = _make_headers(body, idem_key="broke-1")

    with _patch_auth_and_pipeline():
        r = client.post("/v1/analyze", content=body, headers=headers)

    assert r.status_code == 402
