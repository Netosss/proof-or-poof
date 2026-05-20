"""Unit tests for HMAC enterprise authentication."""

import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# Fixture API keys MUST satisfy the production format regex
# `fxl_(live|test)_[A-Za-z0-9_-]{20,64}` so the format guard in
# authenticate() doesn't fast-fail before the test reaches the path under
# verification. These are not real credentials — just 32-char alnum filler.
_VALID_API_KEY = "fxl_test_" + "A" * 32
_INVALID_FORMAT_API_KEY = "fxl_test_short"


def _make_request(
    headers: dict,
    method: str = "POST",
    path: str = "/v1/analyze",
    client_host: str = "1.2.3.4",
    query: str = "",
):
    req = MagicMock()
    req.headers = headers
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    req.url.query = query
    req.client = MagicMock()
    req.client.host = client_host
    return req


def _sign(timestamp: str, method: str, path: str, content_sha: str, secret: str) -> str:
    payload = f"{timestamp}\n{method.upper()}\n{path}\n{content_sha.lower()}".encode()
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


@pytest.fixture
def stub_credential():
    return {
        "partner_id": "p1",
        "credential_id": "c1",
        "api_key_prefix": "fxl_test_abc...xyz",
        "secret_key": "fxs_test_supersecret",
        "allowed_ips": [],
    }


@pytest.mark.asyncio
async def test_authenticate_happy_path(mock_redis, stub_credential):
    from app.core import enterprise_auth as ea

    ts = str(int(time.time()))
    content_sha = "a" * 64
    sig = _sign(ts, "POST", "/v1/analyze", content_sha, stub_credential["secret_key"])

    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": ts,
            "X-FauxLens-Signature": sig,
            "X-FauxLens-Content-SHA256": content_sha,
        }
    )

    with (
        patch.object(ea, "resolve_credential", new=AsyncMock(return_value=stub_credential)),
        patch.object(ea, "touch_credential", new=AsyncMock(return_value=None)),
    ):
        principal = await ea.authenticate(req)

    assert principal.partner_id == "p1"
    assert principal.credential_id == "c1"
    assert principal.content_sha256 == content_sha


@pytest.mark.asyncio
async def test_authenticate_missing_headers(mock_redis):
    from app.core import enterprise_auth as ea

    req = _make_request({"X-FauxLens-Key": _VALID_API_KEY})
    with pytest.raises(HTTPException) as exc:
        await ea.authenticate(req)
    assert exc.value.status_code == 401
    assert exc.value.detail["code"] == "missing_auth_headers"


@pytest.mark.asyncio
async def test_authenticate_timestamp_drift_too_large(mock_redis):
    from app.core import enterprise_auth as ea

    stale_ts = str(int(time.time()) - 9999)
    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": stale_ts,
            "X-FauxLens-Signature": "a" * 64,
            "X-FauxLens-Content-SHA256": "b" * 64,
        }
    )
    with pytest.raises(HTTPException) as exc:
        await ea.authenticate(req)
    assert exc.value.status_code == 401
    assert exc.value.detail["code"] == "timestamp_expired"


@pytest.mark.asyncio
async def test_authenticate_timestamp_not_integer(mock_redis):
    from app.core import enterprise_auth as ea

    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": "not-a-number",
            "X-FauxLens-Signature": "a" * 64,
            "X-FauxLens-Content-SHA256": "b" * 64,
        }
    )
    with pytest.raises(HTTPException) as exc:
        await ea.authenticate(req)
    assert exc.value.detail["code"] == "invalid_timestamp"


@pytest.mark.asyncio
async def test_authenticate_replay_detected(mock_redis, stub_credential):
    """Second call with same signature within window must be rejected."""
    from app.core import enterprise_auth as ea

    ts = str(int(time.time()))
    content_sha = "a" * 64
    sig = _sign(ts, "POST", "/v1/analyze", content_sha, stub_credential["secret_key"])

    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": ts,
            "X-FauxLens-Signature": sig,
            "X-FauxLens-Content-SHA256": content_sha,
        }
    )

    with (
        patch.object(ea, "resolve_credential", new=AsyncMock(return_value=stub_credential)),
        patch.object(ea, "touch_credential", new=AsyncMock(return_value=None)),
    ):
        await ea.authenticate(req)  # first call OK
        with pytest.raises(HTTPException) as exc:
            await ea.authenticate(req)  # second call → replay
    assert exc.value.detail["code"] == "replay_detected"


@pytest.mark.asyncio
async def test_authenticate_invalid_api_key(mock_redis):
    from app.core import enterprise_auth as ea

    ts = str(int(time.time()))
    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": ts,
            "X-FauxLens-Signature": "a" * 64,
            "X-FauxLens-Content-SHA256": "b" * 64,
        }
    )
    with patch.object(ea, "resolve_credential", new=AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await ea.authenticate(req)
    assert exc.value.detail["code"] == "invalid_api_key"


@pytest.mark.asyncio
async def test_authenticate_ip_not_allowed(mock_redis, stub_credential):
    from app.core import enterprise_auth as ea

    stub_credential = {**stub_credential, "allowed_ips": ["10.0.0.0/8"]}
    ts = str(int(time.time()))
    content_sha = "a" * 64
    sig = _sign(ts, "POST", "/v1/analyze", content_sha, stub_credential["secret_key"])

    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": ts,
            "X-FauxLens-Signature": sig,
            "X-FauxLens-Content-SHA256": content_sha,
        },
        client_host="8.8.8.8",
    )

    with patch.object(ea, "resolve_credential", new=AsyncMock(return_value=stub_credential)):
        with pytest.raises(HTTPException) as exc:
            await ea.authenticate(req)
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "ip_not_allowed"


@pytest.mark.asyncio
async def test_authenticate_ip_allowed_in_cidr(mock_redis, stub_credential):
    from app.core import enterprise_auth as ea

    stub_credential = {**stub_credential, "allowed_ips": ["10.0.0.0/8"]}
    ts = str(int(time.time()))
    content_sha = "a" * 64
    sig = _sign(ts, "POST", "/v1/analyze", content_sha, stub_credential["secret_key"])

    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": ts,
            "X-FauxLens-Signature": sig,
            "X-FauxLens-Content-SHA256": content_sha,
        },
        client_host="10.1.2.3",
    )

    with (
        patch.object(ea, "resolve_credential", new=AsyncMock(return_value=stub_credential)),
        patch.object(ea, "touch_credential", new=AsyncMock(return_value=None)),
    ):
        principal = await ea.authenticate(req)
    assert principal.partner_id == "p1"


@pytest.mark.asyncio
async def test_authenticate_signature_mismatch(mock_redis, stub_credential):
    from app.core import enterprise_auth as ea

    ts = str(int(time.time()))
    content_sha = "a" * 64
    req = _make_request(
        {
            "X-FauxLens-Key": _VALID_API_KEY,
            "X-FauxLens-Timestamp": ts,
            "X-FauxLens-Signature": "f" * 64,  # wrong
            "X-FauxLens-Content-SHA256": content_sha,
        }
    )

    with patch.object(ea, "resolve_credential", new=AsyncMock(return_value=stub_credential)):
        with pytest.raises(HTTPException) as exc:
            await ea.authenticate(req)
    assert exc.value.detail["code"] == "invalid_signature"


def test_verify_body_hash_bytes(tmp_path):
    from app.core.enterprise_auth import verify_body_hash

    data = b"hello world"
    digest = hashlib.sha256(data).hexdigest()
    assert verify_body_hash(data, digest)
    assert not verify_body_hash(data, "0" * 64)


def test_verify_body_hash_file(tmp_path):
    from app.core.enterprise_auth import verify_body_hash

    f = tmp_path / "body.bin"
    f.write_bytes(b"hello world")
    digest = hashlib.sha256(b"hello world").hexdigest()
    assert verify_body_hash(str(f), digest)
    assert not verify_body_hash(str(f), "0" * 64)
