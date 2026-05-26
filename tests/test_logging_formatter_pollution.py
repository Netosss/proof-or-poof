"""
Integration tests for the StreamHandler's JsonFormatter pipeline.

These lock in the format-string fix:
  - Context fields populated by RequestContextFilter (request_id/device_id/
    user_id/severity) MUST NOT appear in the rendered JSON when they are
    empty strings or None. The pre-fix format string explicitly referenced
    `%(device_id)s` etc., which made pythonjsonlogger emit them as `null`
    even after StripEmptyFieldsFilter had deleted them from `record.__dict__`.
  - When a context field DOES have a real value, it must still appear in
    the JSON output (no regression on the populated case).

This file complements test_logging_strip_empty.py — that one tests the strip
filter in isolation (record-level), this one tests the full
ContextFilter → StripFilter → JsonFormatter chain (json-output-level).
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest

from app.logging_config import (
    RequestContextFilter,
    StripEmptyFieldsFilter,
    device_id_var,
    request_id_var,
    user_id_var,
)
from pythonjsonlogger import jsonlogger


def _build_handler() -> tuple[logging.Handler, io.StringIO]:
    """Wire a StreamHandler with the same formatter + filters as production."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler.addFilter(RequestContextFilter())
    handler.addFilter(StripEmptyFieldsFilter())
    return handler, buf


def _emit(handler: logging.Handler, level: int, msg: str, **extras: Any) -> dict:
    """Push a record through the handler and return the parsed JSON output."""
    logger = logging.getLogger("test_pollution")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.log(level, msg, extra=extras)
    raw = handler.stream.getvalue().splitlines()[-1]
    return json.loads(raw)


def test_empty_context_fields_omitted_from_json_output():
    """
    The bug fix proof: when ContextVars are unset (e.g. startup logs, OPTIONS
    preflight, /health pings), the rendered JSON must NOT include
    request_id/device_id/user_id as `null`.
    """
    request_id_var.set("")
    device_id_var.set("")
    user_id_var.set("")

    handler, _ = _build_handler()
    payload = _emit(handler, logging.INFO, "startup_event", action="startup")

    assert "request_id" not in payload, f"request_id leaked as {payload.get('request_id')!r}"
    assert "device_id" not in payload, f"device_id leaked as {payload.get('device_id')!r}"
    assert "user_id" not in payload, f"user_id leaked as {payload.get('user_id')!r}"
    # severity is always populated by RequestContextFilter (never empty), so it
    # legitimately stays.
    assert payload["severity"] == "info"
    assert payload["action"] == "startup"
    assert payload["message"] == "startup_event"


def test_populated_context_fields_still_appear_in_json_output():
    """
    No-regression check: when ContextVars are set (real authenticated request),
    the context fields MUST still appear in the JSON. Removing them from the
    format string only suppresses the null-placeholder case — not the
    populated case.
    """
    request_id_var.set("req-abc-123")
    device_id_var.set("dev-xyz-789")
    user_id_var.set("user-firebase-uid")

    handler, _ = _build_handler()
    payload = _emit(handler, logging.INFO, "request_completed", path="/detect")

    assert payload["request_id"] == "req-abc-123"
    assert payload["device_id"] == "dev-xyz-789"
    assert payload["user_id"] == "user-firebase-uid"
    assert payload["severity"] == "info"
    assert payload["path"] == "/detect"


def test_partial_context_only_populated_fields_appear():
    """
    The /api/auth/me case: user_id is populated by Firebase auth, but
    device_id arrives in the POST body (not the X-Device-ID header) so it
    stays empty in the ContextVar. The JSON should include user_id and omit
    device_id.
    """
    request_id_var.set("req-auth-me-1")
    device_id_var.set("")  # header-less endpoint
    user_id_var.set("ifjIk2G4lyXqF1NemqfxfWUiHFg1")

    handler, _ = _build_handler()
    payload = _emit(handler, logging.INFO, "auth_me_success", balance=1760)

    assert payload["request_id"] == "req-auth-me-1"
    assert payload["user_id"] == "ifjIk2G4lyXqF1NemqfxfWUiHFg1"
    assert "device_id" not in payload
    assert payload["balance"] == 1760


def test_semantic_falsy_extras_still_emitted():
    """
    The strip filter must not drop False / 0 / 0.0 / [] / {} extras. Confirm
    end-to-end through the formatter (not just at filter level).
    """
    request_id_var.set("")
    device_id_var.set("")
    user_id_var.set("")

    handler, _ = _build_handler()
    payload = _emit(
        handler,
        logging.INFO,
        "metadata_scoring",
        human_score=0,
        ai_score=0.0,
        has_physical_signals=False,
        human_signals=[],
        usage={},
    )

    assert payload["human_score"] == 0
    assert payload["ai_score"] == 0.0
    assert payload["has_physical_signals"] is False
    assert payload["human_signals"] == []
    assert payload["usage"] == {}


@pytest.fixture(autouse=True)
def _reset_context_vars():
    """Each test gets a clean ContextVar slate."""
    yield
    request_id_var.set("")
    device_id_var.set("")
    user_id_var.set("")
