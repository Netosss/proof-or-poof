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

import sentry_sdk
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import configure_json_logging, device_id_var, request_id_var
from app.api import auth, checkout, credits, detection, inpainting, reports, system, webhooks
from app.integrations import firebase as firebase_module
from app.integrations import http_client as http_module
from app.integrations import redis_client as redis_module
from app.integrations.runpod import cleanup_stale_jobs

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


async def _periodic_cleanup():
    """Background task that removes stale RunPod jobs every 30 seconds."""
    from app.config import settings
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_sec)
            cleanup_stale_jobs()
            logger.debug("cleanup_periodic", extra={"action": "cleanup_periodic"})
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("cleanup_error", extra={"action": "cleanup_error", "error": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    try:
        firebase_module.initialize()
        redis_module.initialize()
        await http_module.initialize()
    except Exception as e:
        logger.critical("startup_failed", extra={
            "action": "startup_failed",
            "error": str(e),
        })
        raise

    cleanup_task = None
    if not os.getenv("TESTING"):
        cleanup_task = asyncio.create_task(_periodic_cleanup())
    logger.info("startup_complete", extra={
        "action": "startup_complete",
        "services": ["firebase", "redis", "http_session"],
    })

    yield

    # --- Shutdown ---
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    await http_module.close()
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
    headers["Access-Control-Allow-Origin"] = "*"
    headers["Access-Control-Allow-Credentials"] = "true"
    headers["Access-Control-Allow-Methods"] = "*"
    headers["Access-Control-Allow-Headers"] = "*"

    try:
        async for _ in request.stream():
            pass
    except Exception as e:
        logger.warning("exception_handler_drain_error", extra={
            "action": "exception_handler_drain_error",
            "error": str(e),
        })

    response_data = {"detail": exc.detail}
    logger.info("http_exception_response", extra={
        "action": "http_exception_response",
        "status_code": exc.status_code,
        "detail": str(exc.detail),
        "path": request.url.path,
    })

    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=headers,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Sentry verification route
# ---------------------------------------------------------------------------
@app.get("/sentry-debug")
async def trigger_error():
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
