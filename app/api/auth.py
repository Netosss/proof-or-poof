"""
Auth routes: Google Sign-In via Firebase.

POST /api/auth/me
  Verifies the Firebase ID token, creates the user account if it is their
  first visit (granting a signup bonus), and returns the user's current state.

Guest and authenticated wallets are kept entirely separate — no credit migration.
The optional device_id in the request body is accepted for backward compatibility
but is no longer used.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.firebase_auth import get_current_user
from app.logging_config import user_id_var
from app.services.credit_engine import get_or_create_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])


class AuthMeRequest(BaseModel):
    device_id: Optional[str] = None


class AuthMeResponse(BaseModel):
    user_id: str
    email: Optional[str]
    credits_balance: int
    is_new_user: bool


@router.post("/api/auth/me", response_model=AuthMeResponse)
async def auth_me(
    body: AuthMeRequest,
    user: dict = Depends(get_current_user),
):
    """
    Register or retrieve the authenticated user.

    - Verifies the Firebase ID token (Authorization: Bearer).
    - Creates the user account on first call with a signup bonus.
    - Returns current balance and whether this is the user's first login
      so the frontend can show an onboarding screen.
    """
    uid = user["uid"]
    email = user.get("email")
    user_id_var.set(uid)

    result = await get_or_create_user(uid, email)

    logger.info("auth_me_success", extra={
        "action": "auth_me_success",
        "is_new_user": result["is_new_user"],
        "balance": result["credits_balance"],
        "has_device_id": bool(body.device_id),
    })

    return AuthMeResponse(
        user_id=result["uid"],
        email=result["email"],
        credits_balance=result["credits_balance"],
        is_new_user=result["is_new_user"],
    )
