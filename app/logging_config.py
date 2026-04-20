"""
Centralised logging configuration.

Call configure_json_logging() once at app startup (before any other imports
that might emit logs). After that, every module's standard
    logger = logging.getLogger(__name__)
automatically emits structured JSON to stdout AND ships directly to Axiom.

Context vars (request_id, device_id, user_id) are set once per HTTP request
by the middleware in main.py and are injected into every log record by
RequestContextFilter — no manual threading through function signatures needed.

Axiom setup
-----------
Set two environment variables in Railway:
  AXIOM_TOKEN   — your Axiom API token  (xaat-xxxx…)
  AXIOM_DATASET — dataset name (defaults to "backend-logs" if not set)

If AXIOM_TOKEN is absent, only stdout logging is active.
If Axiom initialisation fails for any reason, a WARNING is written to stdout
and the app continues without Axiom rather than crashing at startup.
"""

import contextvars
import logging
import os
import sys

from pythonjsonlogger import jsonlogger

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
device_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("device_id", default="")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="")
user_email_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_email", default="")

# Map Python log levels to Axiom/standard severity strings (lowercase).
# Without this, Axiom sees no "severity" field and defaults everything to "error".
_LEVEL_TO_SEVERITY: dict[int, str] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}


class RequestContextFilter(logging.Filter):
    """
    Injects per-request context and severity into every LogRecord.

    Added to each handler so it runs for every record that reaches the handler,
    whether the record originates from the root logger or a child logger.
    Fields injected:
      - request_id, device_id, user_id  — from ContextVars set by middleware
      - severity                         — lowercase level name ("info", "warning", …)
      - level                            — alias for severity (Axiom accepts both)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        record.device_id = device_id_var.get("")
        record.user_id = user_id_var.get("")
        record.user_email = user_email_var.get("")
        sev = _LEVEL_TO_SEVERITY.get(record.levelno, "info")
        record.severity = sev
        record.level = sev
        return True


class _SafeAxiomHandler(logging.Handler):
    """
    Thin wrapper around AxiomHandler that:
      - Catches and logs exceptions from flush() so they never die silently
        inside a threading.Timer callback (Python swallows those errors).
      - Marks the internal timer as a daemon so it never blocks process exit.
    """

    def __init__(self, axiom_handler):
        super().__init__()
        self._inner = axiom_handler
        # Make the repeating timer a daemon so Railway can terminate cleanly.
        self._inner.timer.daemon = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._inner.emit(record)
        except Exception as exc:
            # Write directly to stderr — can't use the logger (infinite loop).
            sys.stderr.write(f'{{"level":"ERROR","message":"axiom_emit_error","error":"{exc}"}}\n')

    def flush(self) -> None:
        try:
            self._inner.flush()
        except Exception as exc:
            sys.stderr.write(f'{{"level":"ERROR","message":"axiom_flush_error","error":"{exc}"}}\n')


def configure_json_logging() -> None:
    """
    Replace the root logger's handlers with:
      1. StreamHandler   — JSON to stdout (always active, captured by Railway)
      2. _SafeAxiomHandler — direct SDK delivery to Axiom
                             (only when AXIOM_TOKEN env var is present)

    RequestContextFilter is added to each handler (not the root logger).
    root.addFilter() only runs when root.handle() is called directly — child
    logger records propagate via callHandlers() which invokes handler.handle(),
    which runs handler-level filters only.  Adding the filter to each handler
    guarantees request_id / device_id / user_id / severity appear in every record.
    """
    root = logging.getLogger()
    ctx_filter = RequestContextFilter()

    # --- 1. Stdout / Railway handler (always active) ---
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s "
            "%(request_id)s %(device_id)s %(user_id)s %(user_email)s %(severity)s"
        )
    )
    stream_handler.addFilter(ctx_filter)
    handlers: list[logging.Handler] = [stream_handler]

    # --- 2. Axiom handler (only when AXIOM_TOKEN is configured) ---
    axiom_token = os.getenv("AXIOM_TOKEN")
    axiom_dataset = os.getenv("AXIOM_DATASET", "backend-logs")

    if axiom_token:
        try:
            from axiom_py import Client
            from axiom_py.logging import AxiomHandler

            axiom_client = Client(token=axiom_token)
            raw_handler = AxiomHandler(axiom_client, axiom_dataset)

            # Wrap in a safe handler that surfaces flush errors to stderr
            # and marks the internal timer as daemon.
            safe_handler = _SafeAxiomHandler(raw_handler)
            safe_handler.addFilter(ctx_filter)
            handlers.append(safe_handler)

            # Confirm Axiom is wired up — visible in Railway and in Axiom.
            stream_handler.stream.write(
                f'{{"level":"INFO","severity":"info","message":"axiom_handler_configured",'
                f'"dataset":"{axiom_dataset}"}}\n'
            )
        except Exception as exc:
            stream_handler.stream.write(
                f'{{"level":"WARNING","severity":"warning","message":"axiom_handler_init_failed","error":"{exc}"}}\n'
            )
    else:
        stream_handler.stream.write(
            '{"level":"WARNING","severity":"warning","message":"axiom_handler_skipped",'
            '"reason":"AXIOM_TOKEN not set — logs go to stdout only"}\n'
        )

    root.handlers = handlers
    root.setLevel(logging.INFO)
