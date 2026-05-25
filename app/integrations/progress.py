"""
Lightweight progress channel for the /detect wait-state UI.

The detect endpoint binds a UUID task_id to a contextvar at request start.
Pipeline stages call `emit(stage)` at real boundaries; the value is written
to Redis under `progress:{task_id}` with a 60s TTL. The wait-state UI polls
`GET /detect/progress/{task_id}` every ~800ms to drive stage transitions.

Stages — 3 user-facing + 1 terminal sentinel:
  provenance — C2PA + metadata + file integrity
  forensic   — Gemini combined-batch call (image or video frames)
  verdict    — evidence chain assembly, cache write
  complete   — terminal; client stops polling

Failure semantics: NO `failed` stage emitted. The main /detect POST's error
response is the single source of truth for failure. The client's 45s
watchdog handles the case where the worker dies mid-pipeline.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from typing import Literal

from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)

Stage = Literal["provenance", "forensic", "verdict", "complete"]

REDIS_KEY_PREFIX = "progress:"
REDIS_TTL_SEC = 60

task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "task_id_var", default=None
)

_start_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "progress_start_var", default=None
)


def init(task_id: str) -> None:
    """Bind a task_id + start anchor to the current async context."""
    task_id_var.set(task_id)
    _start_var.set(time.time())


async def emit(stage: Stage) -> None:
    """
    Best-effort progress emit. Silent no-op if no task_id is bound or Redis
    is unavailable — never blocks or fails the detection itself.
    """
    task_id = task_id_var.get()
    if not task_id:
        return

    start = _start_var.get()
    elapsed_ms = int((time.time() - start) * 1000) if start else 0

    rc = redis_module.client
    if not rc:
        return

    try:
        payload = json.dumps({"stage": stage, "elapsed_ms": elapsed_ms})
        await rc.setex(f"{REDIS_KEY_PREFIX}{task_id}", REDIS_TTL_SEC, payload)
    except Exception as e:
        logger.warning(
            "progress_emit_failed",
            extra={
                "action": "progress_emit_failed",
                "stage": stage,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )


async def read(task_id: str) -> dict:
    """
    Read the current progress payload for a task_id.

    Returns:
      {stage: "pending"|"provenance"|"forensic"|"verdict"|"complete", elapsed_ms: int}

    `pending` means "task_id is unknown / hasn't started emitting yet" — the
    client just keeps polling. `complete` means the client should stop.
    """
    rc = redis_module.client
    if not rc:
        return {"stage": "complete", "elapsed_ms": 0}

    try:
        raw = await rc.get(f"{REDIS_KEY_PREFIX}{task_id}")
    except Exception as e:
        logger.warning(
            "progress_read_failed",
            extra={
                "action": "progress_read_failed",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        return {"stage": "pending", "elapsed_ms": 0}

    if not raw:
        return {"stage": "pending", "elapsed_ms": 0}

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"stage": "pending", "elapsed_ms": 0}
