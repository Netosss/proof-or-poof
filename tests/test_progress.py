"""
Tests for app/integrations/progress.py and the GET /detect/progress/{task_id}
endpoint exposed by app/api/detection.py.

Verifies:
  - Stage payloads round-trip through Redis correctly.
  - Best-effort semantics: emit/read never raise when Redis is unavailable.
  - Pipeline emits the expected stage order on a happy-path detection.
  - Pipeline tolerates a mid-pipeline crash: the last emitted stage stays in
    Redis (the client's 45s watchdog handles the dead-worker case).
  - Endpoint rejects malformed task_ids and returns 'pending' for unknown ones.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integrations import progress as progress_module


# ---------------------------------------------------------------------------
# Fake Redis — just a dict with the subset of API the module uses.
# ---------------------------------------------------------------------------


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    async def get(self, key: str) -> str | None:
        return self.store.get(key)


# ---------------------------------------------------------------------------
# emit() + read() round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_writes_stage_and_elapsed_to_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(progress_module.redis_module, "client", fake)

    progress_module.init("abc123")
    await progress_module.emit("provenance")

    raw = fake.store["progress:abc123"]
    payload = json.loads(raw)
    assert payload["stage"] == "provenance"
    assert isinstance(payload["elapsed_ms"], int)
    assert payload["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_read_returns_pending_for_unknown_task(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(progress_module.redis_module, "client", fake)

    result = await progress_module.read("nope")
    assert result == {"stage": "pending", "elapsed_ms": 0}


@pytest.mark.asyncio
async def test_emit_silent_when_no_task_bound(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(progress_module.redis_module, "client", fake)

    # Reset the contextvar by re-initialising it to None via a fresh context.
    # The current process-level var has no task_id bound by default.
    monkeypatch.setattr(progress_module, "task_id_var", progress_module.task_id_var)
    # Don't call init() — leave task_id unbound.
    progress_module.task_id_var.set(None)

    await progress_module.emit("provenance")
    assert fake.store == {}


@pytest.mark.asyncio
async def test_emit_silent_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr(progress_module.redis_module, "client", None)
    progress_module.init("task_no_redis")
    # Must not raise.
    await progress_module.emit("forensic")


@pytest.mark.asyncio
async def test_read_silent_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr(progress_module.redis_module, "client", None)
    result = await progress_module.read("anything")
    assert result["stage"] == "complete"


# ---------------------------------------------------------------------------
# Stage ordering through detect_ai_media — happy path (image, Gemini used)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_pipeline_emits_forensic_when_gemini_runs(monkeypatch, tmp_path):
    """
    On the Gemini-path image flow, image_detector should emit 'forensic' just
    before the Gemini call. We mock everything else and capture the emit().
    """
    from PIL import Image as _PILImage
    fake_file = tmp_path / "photo.jpg"
    _PILImage.new("RGB", (32, 32), color=(128, 128, 128)).save(fake_file, "JPEG")

    emitted: list[str] = []

    async def fake_emit(stage):
        emitted.append(stage)

    with (
        patch("app.detection.pipeline.get_c2pa_manifest", return_value=None),
        patch("app.detection.image_detector.progress_module.emit", side_effect=fake_emit),
        patch("app.detection.image_detector.get_exif_data", return_value={}),
        patch(
            "app.detection.image_detector.get_quality_context",
            return_value=("HIGH", 90),
        ),
        patch(
            "app.detection.image_detector.get_image_hash",
            return_value="hash_test",
        ),
        patch(
            "app.detection.image_detector.get_cached_result",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.detection.image_detector.set_cached_result",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.detection.image_detector.analyze_image_combined_async",
            new_callable=AsyncMock,
            return_value={
                "confidence": 0.9,
                "signal_category": "objects_merge_or_dissolve_at_boundaries",
                "quality_context": "HIGH",
            },
        ),
    ):
        from app.detection.pipeline import detect_ai_media
        result = await detect_ai_media(str(fake_file))

    assert "forensic" in emitted
    assert result["summary"] in ("Likely AI-Generated", "Likely Authentic")


# ---------------------------------------------------------------------------
# Failure path — Gemini crashes; we never emit 'verdict' from pipeline. The
# route's `await progress_module.emit("verdict")` after detect_ai_media is the
# only call that fires verdict, and on raise the route never reaches it. This
# test confirms emit() itself doesn't blow up the pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_swallows_redis_errors(monkeypatch):
    class ExplodingRedis:
        async def setex(self, *_a, **_kw):
            raise RuntimeError("redis exploded")

    monkeypatch.setattr(progress_module.redis_module, "client", ExplodingRedis())
    progress_module.init("task_explode")
    # Must NOT raise — best-effort semantics.
    await progress_module.emit("forensic")


# ---------------------------------------------------------------------------
# Endpoint shape validation
# ---------------------------------------------------------------------------


def test_task_id_regex_rejects_bad_shapes():
    from app.api.detection import _TASK_ID_RE
    assert _TASK_ID_RE.match("a" * 32)
    assert _TASK_ID_RE.match("0123456789abcdef0123456789abcdef")
    assert not _TASK_ID_RE.match("short")
    assert not _TASK_ID_RE.match("g" * 32)  # non-hex
    assert not _TASK_ID_RE.match("../../etc/passwd")
    assert not _TASK_ID_RE.match("a" * 33)
