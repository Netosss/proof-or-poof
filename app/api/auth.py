"""
Auth routes: Google Sign-In via Firebase.

POST /api/auth/me
  Verifies the Firebase ID token, creates the user account if it is their
  first visit (applying the strict guest-credit migration rule), and returns
  the user's current state.

The frontend should call this endpoint immediately after a successful
Google Sign-In, passing the optional device_id of the current guest
session so any accumulated guest credits can be migrated.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.firebase_auth import get_current_user
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
    - Creates the user account on first call, applying the strict
      guest-credit migration rule (see credit_engine.get_or_create_user).
    - Returns current balance and whether this is the user's first login
      so the frontend can show an onboarding screen.
    """
    uid = user["uid"]
    email = user.get("email")

    result = get_or_create_user(uid, email, device_id=body.device_id)

    logger.info(
        f"[AUTH] /api/auth/me uid={uid} "
        f"is_new={result['is_new_user']} balance={result['credits_balance']}"
    )

    return AuthMeResponse(
        user_id=result["uid"],
        email=result["email"],
        credits_balance=result["credits_balance"],
        is_new_user=result["is_new_user"],
    )
