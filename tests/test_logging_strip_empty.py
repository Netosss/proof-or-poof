"""
Tests for StripEmptyFieldsFilter — the universal empty-field stripper attached
to every logging handler (stdout + default Axiom + enterprise Axiom).

These tests lock in:
  1. None and "" are stripped.
  2. False, 0, 0.0, [], {} are PRESERVED — semantic falsy values carry meaning.
  3. Reserved LogRecord attributes (message, levelname, ...) are never touched.
  4. The filter mutates record.__dict__ in place and returns True
     (it never drops a record entirely — that's the router's job).
"""

from __future__ import annotations

import logging

import pytest

from app.logging_config import StripEmptyFieldsFilter, _LOG_RECORD_RESERVED


def _make_record(**extras) -> logging.LogRecord:
    """Build a LogRecord with arbitrary extra fields, like logger.info('m', extra={...})."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="msg",
        args=(),
        exc_info=None,
    )
    for k, v in extras.items():
        setattr(record, k, v)
    return record


# ---------------------------------------------------------------------------
# Strip cases — None and "" must be removed.
# ---------------------------------------------------------------------------


def test_strips_none():
    record = _make_record(partner_id=None)
    StripEmptyFieldsFilter().filter(record)
    assert "partner_id" not in record.__dict__


def test_strips_empty_string():
    record = _make_record(partner_id="")
    StripEmptyFieldsFilter().filter(record)
    assert "partner_id" not in record.__dict__


def test_strips_multiple_empty_fields_in_one_pass():
    record = _make_record(
        partner_id="",
        email=None,
        short_id="",
        media_file="image.jpg",  # this one survives
    )
    StripEmptyFieldsFilter().filter(record)
    assert "partner_id" not in record.__dict__
    assert "email" not in record.__dict__
    assert "short_id" not in record.__dict__
    assert record.__dict__["media_file"] == "image.jpg"


# ---------------------------------------------------------------------------
# Preservation cases — semantic falsy values must SURVIVE.
# Regressing any of these would silently lose information from logs.
# ---------------------------------------------------------------------------


def test_preserves_false_boolean():
    record = _make_record(is_cached=False)
    StripEmptyFieldsFilter().filter(record)
    assert record.__dict__["is_cached"] is False


def test_preserves_integer_zero():
    record = _make_record(credits_consumed=0)
    StripEmptyFieldsFilter().filter(record)
    assert record.__dict__["credits_consumed"] == 0


def test_preserves_float_zero():
    record = _make_record(cost_usd=0.0)
    StripEmptyFieldsFilter().filter(record)
    assert record.__dict__["cost_usd"] == 0.0


def test_preserves_empty_list():
    # `human_signals: []` means "we scored and found no signals" — meaningful.
    record = _make_record(human_signals=[])
    StripEmptyFieldsFilter().filter(record)
    assert record.__dict__["human_signals"] == []


def test_preserves_empty_dict():
    record = _make_record(usage={})
    StripEmptyFieldsFilter().filter(record)
    assert record.__dict__["usage"] == {}


def test_preserves_truthy_values():
    record = _make_record(
        action="scan_completed",
        confidence_score=0.95,
        duration_ms=2451,
        is_gemini_used=True,
    )
    StripEmptyFieldsFilter().filter(record)
    assert record.__dict__["action"] == "scan_completed"
    assert record.__dict__["confidence_score"] == 0.95
    assert record.__dict__["duration_ms"] == 2451
    assert record.__dict__["is_gemini_used"] is True


# ---------------------------------------------------------------------------
# Reserved-attribute safety — the framework owns these.
# ---------------------------------------------------------------------------


def test_does_not_touch_reserved_logrecord_attributes():
    """Even if a reserved attr is somehow empty, the filter must leave it alone."""
    record = _make_record()
    # Sanity: standard reserved attributes are present on every LogRecord.
    assert "message" in _LOG_RECORD_RESERVED
    assert "levelname" in _LOG_RECORD_RESERVED
    assert "exc_info" in _LOG_RECORD_RESERVED

    StripEmptyFieldsFilter().filter(record)
    # All reserved keys still resolve via getattr (logging framework guarantees).
    assert record.levelname == "INFO"
    assert record.name == "test"


def test_does_not_touch_dunder_prefixed_keys():
    """Custom dunder/private keys (rare but possible) must be left alone."""
    record = _make_record()
    record.__dict__["_internal"] = ""
    StripEmptyFieldsFilter().filter(record)
    # Underscore-prefixed key survives even with empty value.
    assert record.__dict__["_internal"] == ""


# ---------------------------------------------------------------------------
# Filter contract — must always return True (never drop the record).
# ---------------------------------------------------------------------------


def test_filter_always_returns_true_even_with_no_extras():
    record = _make_record()
    assert StripEmptyFieldsFilter().filter(record) is True


def test_filter_returns_true_when_everything_gets_stripped():
    record = _make_record(a="", b=None, c="")
    assert StripEmptyFieldsFilter().filter(record) is True


# ---------------------------------------------------------------------------
# The classic Python footgun — proves we don't use `if not v`.
# `not 0` is True; `not False` is True; `not []` is True. A naive
# implementation would strip all three. Ours preserves them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [False, 0, 0.0, [], {}, (), "0", "False"],
    ids=["False", "int_0", "float_0", "empty_list", "empty_dict", "empty_tuple", "str_zero", "str_False"],
)
def test_falsy_but_meaningful_values_survive(value):
    """Every value in this list is falsy under bool() but carries information."""
    record = _make_record(my_field=value)
    StripEmptyFieldsFilter().filter(record)
    assert "my_field" in record.__dict__, (
        f"Filter incorrectly stripped {value!r} — likely regressed to `if not v` logic"
    )
    assert record.__dict__["my_field"] == value
