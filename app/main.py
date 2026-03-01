"""
Application entry point.

Responsibilities (only):
  - Create the FastAPI application instance
  - Register the lifespan context manager (integration init + background tasks)
  - Mount CORS middleware
  - Register the global HTTP-exception handler
  - Include all APIRouters
"""

import asyncio
import logging
import os

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import credits, detection, inpainting, reports, system, webhooks
from app.integrations import firebase as firebase_module
from app.integrations import http_client as http_module
from app.integrations import redis_client as redis_module
from app.integrations.runpod import cleanup_stale_jobs

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _periodic_cleanup():
    """Background task that removes stale RunPod jobs every 30 seconds."""
    from app.config import settings
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_sec)
            cleanup_stale_jobs()
            logger.debug("[CLEANUP] Periodic cleanup completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[CLEANUP] Error in periodic cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    firebase_module.initialize()
    redis_module.initialize()
    await http_module.initialize()
    cleanup_task = None
    if not os.getenv("TESTING"):
        cleanup_task = asyncio.create_task(_periodic_cleanup())
    logger.info("[STARTUP] Firebase, Redis, HTTP session initialised.")

    yield

    # --- Shutdown ---
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    await http_module.close()
    logger.info("[SHUTDOWN] Background cleanup task stopped.")


app = FastAPI(title="AI Provenance & Cleansing API", lifespan=lifespan)


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
        logger.warning(f"Error draining request stream in exception handler: {e}")

    response_data = {"detail": exc.detail}
    logger.info(f"[ERROR HANDLER] Returning {exc.status_code} to client. Body: {response_data}, Headers: {headers}")

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
# Routers
# ---------------------------------------------------------------------------
app.include_router(system.router)
app.include_router(detection.router)
app.include_router(credits.router)
app.include_router(reports.router)
app.include_router(inpainting.router)
app.include_router(webhooks.router)
