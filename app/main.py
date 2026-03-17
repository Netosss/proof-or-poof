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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import sentry_sdk
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import configure_json_logging, device_id_var, request_id_var
from app.api import auth, checkout, credits, detection, inpainting, reports, system, webhooks
from app.integrations import firebase as firebase_module
from app.integrations import http_client as http_module
from app.integrations import redis_client as redis_module

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
        logger.critical("startup_failed", extra={
            "action": "startup_failed",
            "error": str(e),
        })
        raise

    logger.info("startup_complete", extra={
        "action": "startup_complete",
        "services": ["firebase", "redis", "http_session"],
    })

    yield

    # --- Shutdown ---
    await http_module.close()
    await redis_module.close()
    logger.info("shutdown_complete", extra={"action": "shutdown_complete"})


app = FastAPI(title="AI Provenance & Cleansing API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request lifecycle middleware — sets context vars and logs every request
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())
    device_id = request.headers.get("X-Device-ID", "")
    request_id_var.set(req_id)
    device_id_var.set(device_id)

    # Tag the Sentry scope so any error captured during this request carries
    # the device_id and client IP — makes crash reports actionable.
    sentry_sdk.set_user({
        "id": device_id or "anonymous",
        "ip_address": request.client.host if request.client else None,
    })

    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info("request_completed", extra={
        "action": "request_completed",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "ip": request.client.host if request.client else "",
        "user_agent": request.headers.get("user-agent", ""),
    })
    return response


# ---------------------------------------------------------------------------
# Global exception handler — forces CORS headers onto every error response
# so browsers can read the JSON body.  Also drains the request body to
# prevent ERR_HTTP2_PROTOCOL_ERROR on early-rejected uploads.
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    headers = getattr(exc, "headers", None) or {}
    headers["Access-Control-Allow-Origin"] = "https://fauxlens.com"
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

    response_data = {"detail": exc.detail}
    log_level = logger.warning if exc.status_code < 500 else logger.error
    log_level("http_exception_response", extra={
        "action": "http_exception_response",
        "status_code": exc.status_code,
        "detail": str(exc.detail)[:500],
        "path": request.url.path,
        "method": request.method,
        "user_agent": request.headers.get("user-agent", "")[:200],
    })

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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fauxlens.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
