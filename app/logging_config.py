"""
Centralised logging configuration.

Call configure_json_logging() once at app startup (before any other imports
that might emit logs). After that, every module's standard
    logger = logging.getLogger(__name__)
automatically emits structured JSON to stdout AND ships directly to Axiom.

Context vars (request_id, device_id, user_id) are set once per HTTP request
by the middleware in main.py and are injected into every log record by
RequestContextFilter — no manual threading through function signatures needed.
"""

import contextvars
import logging
import os

from pythonjsonlogger import jsonlogger

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
device_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("device_id", default="")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="")


class RequestContextFilter(logging.Filter):
    """Injects per-request context into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        record.device_id = device_id_var.get("")
        record.user_id = user_id_var.get("")
        return True


def configure_json_logging() -> None:
    """
    Replace the root logger's handlers with:
      1. StreamHandler  — JSON to stdout (always active)
      2. AxiomHandler   — direct SDK delivery to dataset 'backend-logs'
                          (only when AXIOM_TOKEN env var is present)

    Both handlers share the same RequestContextFilter so every record
    automatically carries request_id, device_id, and user_id.
    """
    root = logging.getLogger()
    ctx_filter = RequestContextFilter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s "
            "%(request_id)s %(device_id)s %(user_id)s"
        )
    )
    stream_handler.addFilter(ctx_filter)

    handlers: list[logging.Handler] = [stream_handler]

    axiom_token = os.getenv("AXIOM_TOKEN")
    if axiom_token:
        try:
            from axiom_py import Client
            from axiom_py.logging import AxiomHandler

            axiom_client = Client(token=axiom_token)
            axiom_handler = AxiomHandler(axiom_client, "backend-logs")
            axiom_handler.addFilter(ctx_filter)
            handlers.append(axiom_handler)
        except Exception as e:
            # Don't crash startup if Axiom SDK fails to initialise
            stream_handler.stream.write(
                f'{{"level":"WARNING","message":"axiom_handler_init_failed","error":"{e}"}}\n'
            )

    root.handlers = handlers
    root.setLevel(logging.INFO)
