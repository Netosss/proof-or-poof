"""
Firebase integration.

`db` starts as None. Call `initialize()` inside the FastAPI lifespan
context manager before handling any requests.

Two clients are initialised:
  - firebase_admin app    — used exclusively for Firebase Auth token verification
                            (get_current_user / get_optional_user in firebase_auth.py)
  - AsyncClient (db)      — native async Firestore client; no thread pool needed.
                            google-cloud-firestore ships as a transitive dep of
                            firebase-admin so no extra package is required.
"""

import os
import json
import logging
import firebase_admin
from firebase_admin import credentials

from google.cloud.firestore_v1.async_client import AsyncClient
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# Module-level reference. Set by initialize(); all consuming modules reference
# this at call time via `from app.integrations import firebase; firebase.db`.
db: AsyncClient | None = None


def initialize() -> None:
    """Initialize Firebase Admin SDK (for Auth) and the async Firestore client."""
    global db

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    sa_info = {}

    if service_account_json:
        try:
            sa_info = json.loads(service_account_json)
        except Exception as e:
            logger.error("startup_firebase_parse_error", extra={
                "action": "startup_firebase_parse_error",
                "error": str(e),
            })

    # --- Firebase Admin app (Auth token verification only) ---
    if not firebase_admin._apps:
        try:
            if sa_info:
                cred = credentials.Certificate(sa_info)
                firebase_admin.initialize_app(cred)
            else:
                firebase_admin.initialize_app()
        except Exception as e:
            logger.error("startup_firebase_error", extra={
                "action": "startup_firebase_error",
                "error": str(e),
            })
            firebase_admin.initialize_app()

    # --- Native async Firestore client (no thread pool) ---
    try:
        if sa_info:
            google_creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            db = AsyncClient(project=sa_info["project_id"], credentials=google_creds)
        else:
            # ADC fallback (local dev without service account JSON)
            db = AsyncClient()
    except Exception as e:
        logger.error("startup_firestore_async_error", extra={
            "action": "startup_firestore_async_error",
            "error": str(e),
        })

    logger.info("startup_firebase", extra={"action": "startup_firebase"})
