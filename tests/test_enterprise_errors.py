"""Unit tests for the Stripe-style enterprise error envelope."""


def test_envelope_from_structured_detail():
    from app.core.enterprise_errors import build_envelope
    env = build_envelope(402, {"type": "payment_required_error", "code": "insufficient_credits",
                               "message": "out of credits"})
    assert env["error"]["type"] == "payment_required_error"
    assert env["error"]["code"] == "insufficient_credits"
    assert env["error"]["message"] == "out of credits"
    assert "request_id" in env["error"]


def test_envelope_from_bare_string():
    from app.core.enterprise_errors import build_envelope
    env = build_envelope(404, "missing thing")
    assert env["error"]["type"] == "not_found_error"
    assert env["error"]["code"] == "not_found"
    assert env["error"]["message"] == "missing thing"


def test_envelope_fallback_message_for_unknown_status():
    from app.core.enterprise_errors import build_envelope
    env = build_envelope(418, None)
    assert env["error"]["type"] == "api_error"
    assert env["error"]["code"] == "api_error"
    # Default message used when none supplied
    assert env["error"]["message"]
