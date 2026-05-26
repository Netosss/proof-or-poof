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
Set these environment variables in Railway:
  AXIOM_TOKEN              — your Axiom API token  (xaat-xxxx…)
  AXIOM_DATASET            — default dataset (defaults to "backend-logs")
  AXIOM_DATASET_ENTERPRISE — optional separate dataset for /v1/* + enterprise_*
                             events. When set, those records ship there INSTEAD
                             of the default dataset. Lets enterprise audit logs
                             live in a clean schema separate from the bloated
                             consumer dataset (Axiom free tier has a 257-column
                             ceiling per dataset).

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
        sev = _LEVEL_TO_SEVERITY.get(record.levelno, "info")
        record.severity = sev
        record.level = sev
        return True


# LogRecord attributes that the logging framework owns — never touch these.
# Stripping any of these breaks formatters, traceback rendering, or correlation.
_LOG_RECORD_RESERVED: frozenset[str] = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


class StripEmptyFieldsFilter(logging.Filter):
    """
    Removes fields with value `None` or `""` from every LogRecord before the
    formatter and Axiom serialiser see them. Applied universally — any log,
    any dataset, any handler.

    Rationale: Axiom (and any structured-log backend) bills by ingested bytes,
    indexes a finite number of distinct field names per dataset (Axiom's free
    tier hard-caps at 257 columns), and clogs query results with `partner_id=""`
    style noise for every consumer request. Stripping at emit time keeps the
    schema honest: a field's presence in a log entry means the field had a
    real value at that moment, never an empty placeholder.

    Preserves semantic falsy values — these carry meaning and must NOT be dropped:
      - `False`          (boolean — "this thing was checked and was false")
      - `0` / `0.0`      (numeric zero — "count was zero", "cost was free")
      - `[]` / `{}`      (empty container — "we scored, found no signals")

    The check `v is None or v == ""` is deliberately narrow:
      - `None == ""`     → False  (so `None` is caught by the `is None` branch)
      - `False == ""`    → False  (preserved)
      - `0 == ""`        → False  (preserved)
      - `0.0 == ""`      → False  (preserved)
      - `"" == ""`       → True   (stripped)
    Empty lists/dicts compare equal to `""` only via duck typing if you write
    `not v`, which is exactly the lazy mistake to avoid — we don't use that.

    Reserved LogRecord attributes (`message`, `levelname`, `exc_info`, etc.) are
    always skipped; touching them would break the framework. Application
    extras and ContextVar-injected fields (`request_id`, `partner_id`, …) are
    fair game.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # `list()` snapshot — we mutate __dict__ while iterating.
        for key in list(record.__dict__.keys()):
            if key in _LOG_RECORD_RESERVED or key.startswith("_"):
                continue
            v = record.__dict__[key]
            if v is None or v == "":
                del record.__dict__[key]
        return True


class _EnterpriseRouterFilter(logging.Filter):
    """
    Routes log records between the default and enterprise Axiom datasets.

    A record is "enterprise" when EITHER:
      - its `action` starts with `enterprise_` / `lemonsqueezy_enterprise_`, or
      - its `path` starts with `/v1/` (covers `request_completed` and
        `http_exception_response` on enterprise routes — those don't carry an
        `enterprise_*` action but still belong with the enterprise schema).

    Instantiate twice:
      _EnterpriseRouterFilter(target="enterprise") → attach to enterprise handler
      _EnterpriseRouterFilter(target="default")    → attach to default handler

    Without this split, enterprise events would duplicate into both datasets
    (double cost) and the consumer dataset's 257-column ceiling would keep
    blocking new enterprise fields from landing in Axiom.
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        if target not in ("enterprise", "default"):
            raise ValueError(f"target must be 'enterprise' or 'default', got {target!r}")
        self.target = target

    def filter(self, record: logging.LogRecord) -> bool:
        action = getattr(record, "action", "") or ""
        if action.startswith("enterprise_") or action.startswith("lemonsqueezy_enterprise_"):
            is_enterprise = True
        else:
            path = getattr(record, "path", "") or ""
            is_enterprise = isinstance(path, str) and path.startswith("/v1/")
        return is_enterprise if self.target == "enterprise" else not is_enterprise


class _SafeAxiomHandler(logging.Handler):
    """
    Thin wrapper around AxiomHandler that:
      - Catches and logs exceptions from emit/flush so they never die silently
        inside a threading.Timer callback (Python swallows those errors).
      - Surfaces WHICH action failed when Axiom drops a batch — without this,
        column-limit / serialization errors are invisible.
      - Marks the internal timer as a daemon so it never blocks process exit.
    """

    def __init__(self, axiom_handler, dataset_label: str = "default"):
        super().__init__()
        self._inner = axiom_handler
        self._dataset_label = dataset_label  # surfaced in error logs for debugging
        # Make the repeating timer a daemon so Railway can terminate cleanly.
        self._inner.timer.daemon = True
        # axiom_py.AxiomHandler runs a `threading.Timer` that calls
        # `self.flush()` on ITSELF, bypassing this wrapper's `flush()`. When
        # that periodic flush hits an HTTP error (e.g. 400 from a bloated
        # dataset hitting the 257-column ceiling), the exception bubbles up
        # through the Timer thread and Sentry's threading integration prints
        # the full Python stack trace to stderr. Patch the inner method so
        # those periodic failures degrade to a single-line stderr message
        # instead of multi-line tracebacks that drown legitimate signal.
        self._patch_inner_flush()

    def _patch_inner_flush(self) -> None:
        original_flush = self._inner.flush
        label = self._dataset_label

        def safe_inner_flush() -> None:
            try:
                original_flush()
            except Exception as exc:
                sys.stderr.write(
                    f'{{"level":"ERROR","message":"axiom_periodic_flush_error",'
                    f'"dataset":"{label}",'
                    f'"error_type":"{type(exc).__name__}",'
                    f'"error":"{str(exc)[:300]}"}}\n'
                )

        self._inner.flush = safe_inner_flush  # type: ignore[method-assign]

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._inner.emit(record)
        except Exception as exc:
            # Write directly to stderr — can't use the logger (infinite loop).
            # Include the failing action + dataset so column-limit / serialization
            # errors point straight at the offending log call.
            action = getattr(record, "action", "") or record.getMessage()[:80]
            sys.stderr.write(
                f'{{"level":"ERROR","message":"axiom_emit_error",'
                f'"dataset":"{self._dataset_label}",'
                f'"failed_action":"{action}",'
                f'"error_type":"{type(exc).__name__}",'
                f'"error":"{str(exc)[:300]}"}}\n'
            )

    def flush(self) -> None:
        try:
            self._inner.flush()
        except Exception as exc:
            sys.stderr.write(
                f'{{"level":"ERROR","message":"axiom_flush_error",'
                f'"dataset":"{self._dataset_label}",'
                f'"error_type":"{type(exc).__name__}",'
                f'"error":"{str(exc)[:300]}"}}\n'
            )


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
    strip_filter = StripEmptyFieldsFilter()

    # Filter ORDER matters and is preserved by Python's logging (filters run in
    # addFilter order, short-circuiting on the first that returns False):
    #   1. ctx_filter   — populates request_id/device_id/user_id/severity
    #   2. strip_filter — removes any field whose value is None or ""
    #   3. router       — accepts/rejects records for default vs enterprise
    # Strip must come AFTER context (otherwise it strips empty context that the
    # context filter is about to set) and BEFORE router (so both datasets see
    # the same clean record shape).

    # --- 1. Stdout / Railway handler (always active) ---
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s "
            "%(request_id)s %(device_id)s %(user_id)s %(severity)s"
        )
    )
    stream_handler.addFilter(ctx_filter)
    stream_handler.addFilter(strip_filter)
    handlers: list[logging.Handler] = [stream_handler]

    # --- 2. Axiom handlers (only when AXIOM_TOKEN is configured) ---
    # Two handlers when AXIOM_DATASET_ENTERPRISE is set: enterprise records go
    # to the dedicated dataset, everything else stays on the default. Without
    # the enterprise env var, the default handler receives ALL records
    # (backward-compatible with the single-dataset setup).
    axiom_token = os.getenv("AXIOM_TOKEN")
    axiom_dataset_default = os.getenv("AXIOM_DATASET", "backend-logs")
    axiom_dataset_enterprise = os.getenv("AXIOM_DATASET_ENTERPRISE", "").strip() or None

    if axiom_token:
        try:
            from axiom_py import Client
            from axiom_py.logging import AxiomHandler

            axiom_client = Client(token=axiom_token)

            # Default dataset handler — receives non-enterprise records (or all
            # records when no enterprise dataset is configured).
            default_axiom = AxiomHandler(axiom_client, axiom_dataset_default)
            default_safe = _SafeAxiomHandler(default_axiom, dataset_label=axiom_dataset_default)
            default_safe.addFilter(ctx_filter)
            default_safe.addFilter(strip_filter)
            if axiom_dataset_enterprise:
                default_safe.addFilter(_EnterpriseRouterFilter(target="default"))
            handlers.append(default_safe)

            # Enterprise dataset handler (optional) — receives /v1/* +
            # enterprise_* records only.
            if axiom_dataset_enterprise:
                ent_axiom = AxiomHandler(axiom_client, axiom_dataset_enterprise)
                ent_safe = _SafeAxiomHandler(ent_axiom, dataset_label=axiom_dataset_enterprise)
                ent_safe.addFilter(ctx_filter)
                ent_safe.addFilter(strip_filter)
                ent_safe.addFilter(_EnterpriseRouterFilter(target="enterprise"))
                handlers.append(ent_safe)

            # Confirm Axiom is wired up — visible in Railway and in Axiom.
            datasets_str = axiom_dataset_default + (
                f" + {axiom_dataset_enterprise}" if axiom_dataset_enterprise else ""
            )
            stream_handler.stream.write(
                f'{{"level":"INFO","severity":"info","message":"axiom_handler_configured",'
                f'"datasets":"{datasets_str}"}}\n'
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
