"""
Auth routes: Google Sign-In via Firebase.

POST /api/auth/me
  Verifies the Firebase ID token, creates the user account if it is their
  first visit (granting a signup bonus), and returns the user's current state.

Guest and authenticated wallets are kept entirely separate — no credit migration.
The optional device_id in the request body is stored against the user's Firestore
doc so we can look up which user owns a device_id when debugging complaints.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from google.cloud.firestore_v1 import ArrayUnion
from pydantic import BaseModel

from app.core.firebase_auth import get_current_user
from app.integrations import firebase as firebase_module
from app.services.credit_engine import get_or_create_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])


class AuthMeRequest(BaseModel):
    device_id: str | None = None


class AuthMeResponse(BaseModel):
    user_id: str
    email: str | None
    credits_balance: int
    is_new_user: bool


async def _store_device_id(uid: str, device_id: str) -> None:
    """Appends device_id to the user's known_device_ids list in Firestore.

    Runs as a background task so it never adds latency to the auth response.
    Uses ArrayUnion so duplicates are silently ignored by Firestore.
    """
    db = firebase_module.db
    if not db or not device_id:
        return
    try:
        await (
            db.collection("users")
            .document(uid)
            .update({"known_device_ids": ArrayUnion([device_id])})
        )
    except Exception as e:
        logger.warning(
            "auth_store_device_id_failed",
            extra={
                "action": "auth_store_device_id_failed",
                "error": str(e),
            },
        )


@router.post("/api/auth/me", response_model=AuthMeResponse)
async def auth_me(
    body: AuthMeRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Register or retrieve the authenticated user.

    - Verifies the Firebase ID token (Authorization: Bearer).
    - Creates the user account on first call with a signup bonus.
    - Persists the device_id against the user doc for support lookups.
    - Returns current balance and whether this is the user's first login.
    """
    uid = user["uid"]
    email = user.get("email")

    result = await get_or_create_user(uid, email)

    if body.device_id:
        background_tasks.add_task(_store_device_id, uid, body.device_id)

    logger.info(
        "auth_me_success",
        extra={
            "action": "auth_me_success",
            "is_new_user": result["is_new_user"],
            "balance": result["credits_balance"],
            "has_device_id": bool(body.device_id),
        },
    )

    return AuthMeResponse(
        user_id=result["uid"],
        email=result["email"],
        credits_balance=result["credits_balance"],
        is_new_user=result["is_new_user"],
    )
