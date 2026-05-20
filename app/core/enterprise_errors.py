"""
Stripe-style error envelope for /v1/* responses.

`build_envelope` accepts either a flat string or the structured dict that
enterprise modules raise inside `HTTPException.detail`. The output shape is:

    {"error": {"type": str, "code": str, "message": str, "request_id": str}}

Why Stripe-style and not RFC 7807:
    - More common in modern enterprise SDKs (Stripe, OpenAI, Twilio, Cohere)
    - Always includes `request_id` for support correlation, which RFC 7807
      omits by design
    - Smaller payload; no `instance` URI noise
"""

from typing import Any

from app.logging_config import request_id_var

_DEFAULT_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "payment_required_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "request_too_large_error",
    415: "unsupported_media_type_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


def build_envelope(status_code: int, detail: Any) -> dict[str, dict[str, str]]:
    """Translate an HTTPException.detail into the public error envelope."""
    request_id = request_id_var.get("") or ""

    if isinstance(detail, dict):
        error = {
            "type": str(detail.get("type") or _DEFAULT_TYPE_BY_STATUS.get(status_code, "api_error")),
            "code": str(detail.get("code") or _code_from_status(status_code)),
            "message": str(detail.get("message") or _message_from_status(status_code)),
            "request_id": request_id,
        }
    else:
        # Bare string from generic FastAPI HTTPException (or 500).
        error = {
            "type": _DEFAULT_TYPE_BY_STATUS.get(status_code, "api_error"),
            "code": _code_from_status(status_code),
            "message": str(detail) if detail else _message_from_status(status_code),
            "request_id": request_id,
        }

    return {"error": error}


def _code_from_status(status_code: int) -> str:
    return {
        400: "invalid_request",
        401: "unauthorized",
        402: "insufficient_credits",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "invalid_request",
        429: "rate_limited",
        500: "internal_error",
        503: "service_unavailable",
    }.get(status_code, "api_error")


def _message_from_status(status_code: int) -> str:
    return {
        400: "Invalid request.",
        401: "Authentication failed.",
        402: "Insufficient credits.",
        403: "Access denied.",
        404: "Not found.",
        413: "Payload too large.",
        415: "Unsupported media type.",
        429: "Too many requests.",
        500: "An internal error occurred.",
        503: "Service temporarily unavailable.",
    }.get(status_code, "An error occurred.")
