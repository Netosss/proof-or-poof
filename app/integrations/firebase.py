"""
Firebase integration.

`db` starts as None. Call `initialize()` inside the FastAPI lifespan
context manager before handling any requests.
"""

import os
import json
import logging
import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

# Module-level reference. Set by initialize(); all consuming modules reference
# this at call time via `from app.integrations import firebase; firebase.db`.
db = None  # firestore.Client | None


def initialize() -> None:
    """Initialize Firebase Admin SDK and set the module-level `db` client."""
    global db

    if not firebase_admin._apps:
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if service_account_json:
            try:
                sa_info = json.loads(service_account_json)
                cred = credentials.Certificate(sa_info)
                firebase_admin.initialize_app(cred)
            except Exception as e:
                logger.error(f"Error initializing Firebase with service account: {e}")
                firebase_admin.initialize_app()
        else:
            firebase_admin.initialize_app()

    db = firestore.client()
    logger.info("[STARTUP] Firebase initialized")
