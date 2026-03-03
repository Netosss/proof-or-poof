"""
Firebase Auth dependency.

Extracts and verifies the Firebase ID token from the standard
`Authorization: Bearer <token>` header. Returns the decoded token
payload containing `uid` and `email`.

Usage:
    @router.post("/some-endpoint")
    async def handler(user: dict = Depends(get_current_user)):
        uid = user["uid"]
        email = user["email"]
"""

import asyncio
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from firebase_admin import auth as firebase_auth

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency that verifies a Firebase ID token from the
    Authorization: Bearer header.

    Returns dict with at minimum: uid, email (email may be None for
    some sign-in providers).

    Raises 401 if the header is missing or the token is invalid/expired.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        # verify_id_token is synchronous and performs I/O (fetches Google's
        # public keys on first call, then caches them). Run in a thread pool
        # to avoid blocking the async event loop.
        decoded = await asyncio.to_thread(firebase_auth.verify_id_token, token)
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email"),
        }
    except firebase_auth.ExpiredIdTokenError:
        logger.warning("[AUTH] Firebase token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except firebase_auth.InvalidIdTokenError as e:
        logger.warning(f"[AUTH] Invalid Firebase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"[AUTH] Unexpected token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[dict]:
    """
    Like get_current_user but returns None instead of raising 401 when
    no Authorization header is present. Used by detect/inpaint endpoints
    that support both guest and authenticated flows.
    """
    if not credentials:
        return None
    return await get_current_user(credentials)
