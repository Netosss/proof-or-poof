"""
Shared report management: publishing, fetching, and background TTL extension.

Firebase and Redis clients are accessed at call-time via integration modules
so they pick up instances initialized during the FastAPI lifespan.
"""

import json
import logging
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException

from app.config import settings
from app.integrations import firebase as firebase_module
from app.integrations import redis_client as redis_module

logger = logging.getLogger(__name__)


def _get_db():
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    return db


def create_share_link(short_id: str) -> dict:
    """
    Publishes a cached scan result as a permanent shared report.
    Idempotent: re-publishing the same short_id returns the existing report_id.
    """
    rc = redis_module.client
    db = _get_db()

    if rc and rc.get(f"is_shared:{short_id}"):
        return {"report_id": short_id}

    raw = rc.get(f"report:{short_id}") if rc else None
    if not raw:
        raise HTTPException(status_code=404, detail="Share link expired or invalid.")

    payload = json.loads(raw) if isinstance(raw, str) else raw
    now = datetime.now(timezone.utc)
    payload["created_at"] = now
    payload["expires_at"] = now + timedelta(days=settings.report_ttl_days)

    db.collection("shared_reports").document(short_id).set(payload)

    if rc:
        rc.setex(f"is_shared:{short_id}", settings.share_lock_ttl_sec, "1")

    return {"report_id": short_id}


def get_shared_report(report_id: str) -> tuple[dict, bool]:
    """
    Fetches a public shared report.
    Returns (data_dict, should_extend_ttl).
    """
    db = _get_db()
    doc = db.collection("shared_reports").document(report_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    data = doc.to_dict()
    expires_at = data.get("expires_at")

    should_extend = False
    if expires_at:
        now = datetime.now(timezone.utc)
        time_left = expires_at - now
        if time_left.days < settings.report_extend_threshold_days:
            should_extend = True

    data.pop("created_at", None)
    data.pop("expires_at", None)
    return data, should_extend


def extend_report_ttl(report_id: str, new_expiry: datetime) -> None:
    """
    Background task: extends a viral report's Firestore TTL.
    Uses a Redis nx lock to prevent duplicate writes under concurrent traffic.
    """
    rc = redis_module.client
    if rc and rc.set(f"extending:{report_id}", "1", nx=True, ex=settings.extend_lock_ttl_sec):
        db = firebase_module.db
        if db:
            db.collection("shared_reports").document(report_id).update({
                "expires_at": new_expiry
            })
