"""Unit tests for Idempotency-Key handling."""

import pytest
from fastapi import HTTPException


def test_get_idempotency_key_missing():
    from app.core.idempotency import get_idempotency_key
    with pytest.raises(HTTPException) as exc:
        get_idempotency_key({})
    assert exc.value.status_code == 400


def test_get_idempotency_key_too_long():
    from app.core.idempotency import get_idempotency_key
    with pytest.raises(HTTPException) as exc:
        get_idempotency_key({"Idempotency-Key": "x" * 1000})
    assert exc.value.status_code == 400


def test_get_idempotency_key_present():
    from app.core.idempotency import get_idempotency_key
    assert get_idempotency_key({"Idempotency-Key": "abc-123"}) == "abc-123"


@pytest.mark.asyncio
async def test_claim_returns_none_on_first_call(mock_redis):
    from app.core.idempotency import claim_or_replay
    result = await claim_or_replay("cred-1", "key-1")
    assert result is None


@pytest.mark.asyncio
async def test_replay_returns_cached_response(mock_redis):
    from app.core.idempotency import claim_or_replay, store
    await store("cred-1", "key-1", status_code=200, body={"hello": "world"})

    cached = await claim_or_replay("cred-1", "key-1")
    assert cached is not None
    assert cached.status_code == 200
    assert cached.body == {"hello": "world"}
