"""
Application entry point.

Responsibilities (only):
  - Configure structured JSON logging + Sentry before anything else
  - Create the FastAPI application instance
  - Register the lifespan context manager (integration init + background tasks)
  - Mount CORS middleware and request-logging middleware
  - Register the global HTTP-exception handler
  - Include all APIRouters
"""

import asyncio
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api import auth, checkout, credits, detection, inpainting, reports, system, webhooks
from app.api.enterprise import analyze as enterprise_analyze
from app.api.enterprise import management as enterprise_management
from app.core.enterprise_errors import build_envelope
from app.integrations import firebase as firebase_module
from app.integrations import http_client as http_module
from app.integrations import redis_client as redis_module
from app.logging_config import configure_json_logging, device_id_var, request_id_var, user_id_var

load_dotenv()

# --- Logging + Sentry must be configured before any other module emits logs ---
configure_json_logging()

sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        send_default_pii=True,
        enable_logs=True,
        traces_sample_rate=0.1,  # 10% sampled — accurate averages within free tier
    )

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    # Increase the default thread pool from ~12 (os.cpu_count()+4) to 30 so up
    # to 30 Gemini calls can run in parallel.  Gemini is I/O-bound (network
    # wait), so threads stay idle most of the time and cost <200 KB RSS each.
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=30))

    try:
        firebase_module.initialize()
        await redis_module.initialize()
        await http_module.initialize()
    except Exception as e:
        logger.critical(
            "startup_failed",
            extra={
                "action": "startup_failed",
                "error": str(e),
            },
        )
        raise

    logger.info(
        "startup_complete",
        extra={
            "action": "startup_complete",
            "services": ["firebase", "redis", "http_session"],
        },
    )

    yield

    # --- Shutdown ---
    await http_module.close()
    await redis_module.close()
    logger.info("shutdown_complete", extra={"action": "shutdown_complete"})


app = FastAPI(title="AI Provenance & Cleansing API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request lifecycle middleware — sets context vars and logs every request.
# Pure ASGI (no BaseHTTPMiddleware) so the inner app runs in the SAME asyncio
# task: ContextVar mutations in route handlers (e.g. user_id_var.set()) are
# visible here after `await self.app(...)` returns.
# ---------------------------------------------------------------------------
class _RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        req_id = str(uuid.uuid4())
        device_id = request.headers.get("X-Device-ID", "")
        request_id_var.set(req_id)
        device_id_var.set(device_id)

        sentry_sdk.set_user(
            {
                "id": device_id or "anonymous",
                "ip_address": request.client.host if request.client else None,
            }
        )

        t0 = time.perf_counter()
        status_code = 500

        async def send_wrapper(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", req_id)
            await send(message)

        await self.app(scope, receive, send_wrapper)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Enterprise auth sets user_id to "ent:{partner_id}". Promote the
        # partner UUID to its own field so per-partner SLA dashboards in
        # the enterprise Axiom dataset can `where partner_id == "..."`
        # without parsing a prefix off `user_id`.
        user_id_value = user_id_var.get("")
        partner_id_value = user_id_value[4:] if user_id_value.startswith("ent:") else ""

        logger.info(
            "request_completed",
            extra={
                "action": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "ip": request.client.host if request.client else "",
                "user_agent": request.headers.get("user-agent", ""),
                "user_id": user_id_value,
                "partner_id": partner_id_value,
            },
        )


# ---------------------------------------------------------------------------
# Global exception handler — forces CORS headers onto every error response
# so browsers can read the JSON body.  Also drains the request body to
# prevent ERR_HTTP2_PROTOCOL_ERROR on early-rejected uploads.
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Enterprise /v1/* routes get a Stripe-style error envelope and no CORS
    # headers (S2S clients don't run in browsers — CORS is irrelevant and a
    # browser hitting these endpoints should be told plainly).
    is_enterprise = request.url.path.startswith("/v1/")

    headers = getattr(exc, "headers", None) or {}
    if not is_enterprise:
        # Echo the request Origin when it matches the static allowlist OR the
        # Firebase Hosting regex. Default to fauxlens.com so prod browsers
        # still see CORS headers on errors when the Origin header is missing.
        request_origin = request.headers.get("origin", "")
        origin_allowed = request_origin in _cors_origins or (
            bool(request_origin) and bool(re.match(_cors_origin_regex, request_origin))
        )
        allowed_origin = request_origin if origin_allowed else "https://fauxlens.com"
        headers["Access-Control-Allow-Origin"] = allowed_origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "*"
        headers["Access-Control-Allow-Headers"] = "*"

    # Drain the request body so HTTP/2 doesn't surface ERR_HTTP2_PROTOCOL_ERROR
    # to the browser.  Cap at 64 KB to prevent a DDoS vector where an attacker
    # sends a huge body on a request that will be rejected (bad token, etc.).
    # Anything larger is handled by uvicorn/h2 via RST_STREAM automatically.
    _MAX_DRAIN = 65_536  # 64 KB
    content_length = request.headers.get("content-length", "0")
    if int(content_length or 0) <= _MAX_DRAIN:
        try:
            drained = 0
            async for chunk in request.stream():
                drained += len(chunk)
                if drained >= _MAX_DRAIN:
                    break
        except RuntimeError:
            # "Stream consumed" — FastAPI already read the body. Fine to ignore.
            pass

    if is_enterprise:
        response_data = build_envelope(exc.status_code, exc.detail)
    else:
        response_data = {"detail": exc.detail}
    # Promote enterprise partner UUID from `ent:{partner_id}` user_id to its
    # own field — same rationale as in _RequestLoggingMiddleware above.
    user_id_value = user_id_var.get("")
    partner_id_value = user_id_value[4:] if user_id_value.startswith("ent:") else ""

    log_level = logger.warning if exc.status_code < 500 else logger.error
    log_level(
        "http_exception_response",
        extra={
            "action": "http_exception_response",
            "status_code": exc.status_code,
            "detail": str(exc.detail)[:500],
            "path": request.url.path,
            "method": request.method,
            "user_agent": request.headers.get("user-agent", "")[:200],
            "user_id": user_id_value,
            "partner_id": partner_id_value,
        },
    )

    # Send unexpected server errors to Sentry for developer alerting and
    # stack-trace grouping.  4xx are expected user errors — they belong in
    # Axiom dashboards, not Sentry issue trackers.
    if exc.status_code >= 500:
        sentry_sdk.capture_exception(exc)

    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=headers,
    )


# Allow localhost when either APP_ENV=dev (default for local dev) OR the
# explicit `ALLOW_LOCAL_DEV_ORIGIN` flag is set. The flag lets a founder run
# the local frontend against a backend pointed at live Lemon Squeezy / live
# Firebase without breaking CORS — useful for end-to-end checkout tests
# before deploying. Real production (Railway) never sets this.
_cors_origins = ["https://fauxlens.com"]
_allow_local = os.getenv("APP_ENV", "prod") == "dev" or os.getenv(
    "ALLOW_LOCAL_DEV_ORIGIN", ""
).lower() in ("1", "true", "yes")
if _allow_local:
    _cors_origins.extend(
        [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
    )

# Firebase Hosting canonical + preview channel URLs.
#
# Pattern is locked to the `proof-or-poof` project namespace, so no external
# .web.app or .firebaseapp.com domain (which third parties cannot register
# inside our Firebase project) can match. Matches:
#   https://proof-or-poof.web.app                                   (canonical)
#   https://proof-or-poof.firebaseapp.com                           (legacy)
#   https://www.fauxlens.com                                        (with-www)
#   https://proof-or-poof--<channel>.web.app                        (any preview)
_cors_origin_regex = (
    r"^https://(www\.fauxlens\.com"
    r"|proof-or-poof(--[a-zA-Z0-9_-]+)?\.(web\.app|firebaseapp\.com))$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-User-Balance",
        "X-Op-Ref",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-Idempotent-Replay",
        "X-Request-ID",
    ],
)
app.add_middleware(_RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Mobile Turnstile captcha page — served to Android WebView for token solving
# ---------------------------------------------------------------------------
_MOBILE_CAPTCHA_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="margin:0;padding:0;background:transparent;">
<div id="cf-turnstile"></div>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit" async defer></script>
<script>
  function initTurnstile() {
    if (!window.turnstile) { setTimeout(initTurnstile, 100); return; }
    turnstile.render('#cf-turnstile', {
      sitekey: '0x4AAAAAAC_98UvlWKoA3QBq',
      callback: function(token) {
        if (window.Android) Android.onTokenReceived(token);
      },
      'error-callback': function() {
        if (window.Android) Android.onTokenError();
      },
      'expired-callback': function() {
        if (window.Android) Android.onTokenExpired();
      },
      execution: 'render',
      appearance: 'interaction-only',
      theme: 'dark',
    });
  }
  document.addEventListener('DOMContentLoaded', initTurnstile);
</script>
</body>
</html>"""


@app.get("/mobile-captcha.html", include_in_schema=False)
async def mobile_captcha_page():
    return HTMLResponse(content=_MOBILE_CAPTCHA_HTML)


# ---------------------------------------------------------------------------
# Sentry verification route — dev only
# ---------------------------------------------------------------------------
@app.get("/sentry-debug")
async def trigger_error():
    if os.getenv("APP_ENV") != "dev":
        raise HTTPException(status_code=404)
    division_by_zero = 1 / 0


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(system.router)
app.include_router(auth.router)
app.include_router(detection.router)
app.include_router(credits.router)
app.include_router(checkout.router)
app.include_router(reports.router)
app.include_router(inpainting.router)
app.include_router(webhooks.router)
app.include_router(enterprise_analyze.router)
app.include_router(enterprise_management.router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/robots.txt", include_in_schema=False)
async def robots() -> PlainTextResponse:
    return PlainTextResponse("User-agent: *\nDisallow: /\n")
