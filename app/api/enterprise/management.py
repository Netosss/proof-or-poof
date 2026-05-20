"""
Browser-callable enterprise management endpoints — Firebase-Auth-protected.

These are NOT the partner S2S API (`/v1/analyze`). They power the
`/enterprise/*` web pages: the application form, the dashboard, key
management, and the checkout flow. Every endpoint requires a valid Firebase
ID token (via the existing `get_current_user` dep) and operates on the
partner record linked to that token's UID.

Routes:
    GET  /api/enterprise/me                       — am I a partner?
    POST /api/enterprise/apply                    — create application
    POST /api/enterprise/checkout/create          — build LS checkout URL
    GET  /api/enterprise/usage                    — balance + recent ledger
    GET  /api/enterprise/keys                     — list credentials (no secrets)
    POST /api/enterprise/keys                     — issue new credential (plaintext once)
    POST /api/enterprise/keys/{credential_id}/revoke
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.core.firebase_auth import get_current_user
from app.integrations.http_client import request_session
from app.services.api_credentials import create_credential, revoke_credential
from app.services.enterprise_applications import (
    create_application,
    find_application_for_uid,
)
from app.services.enterprise_credit_engine import get_partner_balance
from app.services.enterprise_partners import (
    find_partner_for_uid,
    list_partner_credentials,
    list_partner_ledger,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enterprise", tags=["EnterpriseWeb"])

LEMONSQUEEZY_CHECKOUT_URL = "https://api.lemonsqueezy.com/v1/checkouts"


# Read both LS env vars at REQUEST time, not module import time.
# load_dotenv() runs in main.py AFTER the router modules have already been
# imported, so reading these constants at import time silently picks up the
# pre-dotenv empty string. Functions evaluated at request time always see
# the resolved environment.
def _resolve_ls_store_id() -> str:
    return os.getenv("LEMONSQUEEZY_STORE_ID", "")


def _resolve_ls_api_key() -> str:
    if settings.is_dev:
        return os.getenv("LEMONSQUEEZY_API_KEY_TEST_MODE", "")
    return os.getenv("LEMONSQUEEZY_API_KEY", "")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ApplicationRequest(BaseModel):
    company_name: str = Field(..., min_length=2, max_length=100)
    contact_email: str = Field(..., min_length=5, max_length=200)
    use_case: str = Field(..., description="newsroom / trust_safety / insurance / marketplace / research / other")
    expected_volume: str = Field(..., description="under_2k / 2k_10k / 10k_25k / over_25k")
    tier: str = Field(..., description="sandbox / starter / pro / scale")
    notes: str | None = Field(None, max_length=500)


class EnterpriseCheckoutRequest(BaseModel):
    variant_id: str


# ---------------------------------------------------------------------------
# GET /me — am I a partner?
# ---------------------------------------------------------------------------

@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    """Returns the partner record + latest application for the signed-in user.

    Response shape:
        {
          "partner":     { ... } | null,
          "application": { ... } | null,
        }

    The frontend uses this to decide whether to show the marketing page, the
    apply form, the "pending review" message, or the partner dashboard.
    """
    uid = user["uid"]
    partner = await find_partner_for_uid(uid)
    application = await find_application_for_uid(uid)
    # Strip volatile fields the browser shouldn't see (ledger lives elsewhere)
    if partner:
        partner.pop("credits_version", None)
    return {"partner": partner, "application": application}


# ---------------------------------------------------------------------------
# POST /apply — application intake
# ---------------------------------------------------------------------------

@router.post("/apply")
async def apply(
    body: ApplicationRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Submit (or re-fetch) an enterprise application for the signed-in user.

    Idempotent: if the user already has a pending/approved application, the
    existing record is returned without writing a new one.
    """
    # The email on the form is what they want us to contact, which may differ
    # from their Google sign-in email. We trust the form value but capture both
    # in logs for audit.
    application = await create_application(
        firebase_uid=user["uid"],
        contact_email=body.contact_email,
        company_name=body.company_name,
        use_case=body.use_case,
        expected_volume=body.expected_volume,
        tier=body.tier,
        notes=body.notes,
    )
    logger.info(
        "enterprise_apply_completed",
        extra={
            "action": "enterprise_apply_completed",
            "application_id": application.get("id"),
            "firebase_uid": user["uid"],
            "ip": request.client.host if request.client else "",
        },
    )
    return application


# ---------------------------------------------------------------------------
# POST /checkout/create — Lemon Squeezy URL with custom_data injected server-side
# ---------------------------------------------------------------------------

@router.post("/checkout/create")
async def create_checkout(
    body: EnterpriseCheckoutRequest,
    user: dict = Depends(get_current_user),
):
    """Build an enterprise Lemon Squeezy checkout URL.

    `custom_data` is set server-side and includes the partner_id so the webhook
    can route the granted credits to the right account. If the user doesn't yet
    have a partner record, we still inject their Firebase UID so the webhook
    can auto-provision one on payment.
    """
    api_key = _resolve_ls_api_key()
    store_id = _resolve_ls_store_id()
    missing: list[str] = []
    if not api_key:
        missing.append("LEMONSQUEEZY_API_KEY_TEST_MODE" if settings.is_dev else "LEMONSQUEEZY_API_KEY")
    if not store_id:
        missing.append("LEMONSQUEEZY_STORE_ID")
    if missing:
        logger.error(
            "enterprise_checkout_unconfigured",
            extra={
                "action": "enterprise_checkout_unconfigured",
                "app_env": settings.app_env,
                "missing_env_vars": missing,
            },
        )
        # Show the missing-env-var names ONLY in local-dev surfaces so a
        # founder debugging locally sees a useful error. In prod, return a
        # generic message — internal config names are operator-private.
        allow_detail = settings.is_dev or os.getenv("ALLOW_LOCAL_DEV_ORIGIN", "").lower() in (
            "1", "true", "yes"
        )
        message = (
            f"Payment service is not configured on the backend. "
            f"Missing environment variable(s): {', '.join(missing)}"
            if allow_detail
            else "Payment service is temporarily unavailable. Please try again later."
        )
        raise HTTPException(
            status_code=503,
            detail={
                "type": "service_unavailable_error",
                "code": "payment_service_unavailable",
                "message": message,
            },
        )

    partner = await find_partner_for_uid(user["uid"])
    custom_data = {
        "account_type": "enterprise",
        "env": settings.app_env,
        "firebase_uid": user["uid"],
    }
    if partner:
        custom_data["partner_id"] = partner["id"]

    payload = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_options": {"embed": True, "locale": "en"},
                "checkout_data": {
                    "custom": custom_data,
                    "email": user.get("email") or "",
                },
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": store_id}},
                "variant": {"data": {"type": "variants", "id": body.variant_id}},
            },
        }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
    }

    try:
        async with request_session() as sess:
            async with sess.post(LEMONSQUEEZY_CHECKOUT_URL, json=payload, headers=headers) as resp:
                if resp.status not in (200, 201):
                    err = await resp.text()
                    logger.error(
                        "enterprise_checkout_api_error",
                        extra={"action": "enterprise_checkout_api_error",
                               "status": resp.status, "body": err[:200]},
                    )
                    raise HTTPException(status_code=502, detail="checkout_create_failed")
                data = await resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("enterprise_checkout_unexpected",
                     extra={"action": "enterprise_checkout_unexpected", "error": str(e)})
        raise HTTPException(status_code=502, detail="checkout_create_failed")

    try:
        url = data["data"]["attributes"]["url"]
    except (KeyError, TypeError):
        raise HTTPException(status_code=502, detail="checkout_invalid_response")

    logger.info(
        "enterprise_checkout_created",
        extra={
            "action": "enterprise_checkout_created",
            "firebase_uid": user["uid"],
            "variant_id": body.variant_id,
            "partner_id": (partner or {}).get("id"),
        },
    )
    return {"checkout_url": url}


# ---------------------------------------------------------------------------
# Helpers — partner-required guard
# ---------------------------------------------------------------------------

async def _require_partner(user: dict) -> dict:
    partner = await find_partner_for_uid(user["uid"])
    if not partner:
        raise HTTPException(
            status_code=403,
            detail={
                "type": "permission_error",
                "code": "not_a_partner",
                "message": "This account is not associated with an enterprise partner.",
            },
        )
    if partner.get("status") != "active":
        # Log the specific status server-side; return only a generic code to the
        # client so we don't leak internal account-state enumeration to attackers.
        logger.warning(
            "enterprise_partner_not_active",
            extra={
                "action": "enterprise_partner_not_active",
                "partner_id": partner["id"],
                "firebase_uid": user["uid"],
                "status": partner.get("status"),
            },
        )
        raise HTTPException(
            status_code=403,
            detail={
                "type": "permission_error",
                "code": "account_not_active",
                "message": "Your enterprise account is not currently active. Contact support.",
            },
        )
    return partner


# ---------------------------------------------------------------------------
# GET /usage — balance + recent ledger
# ---------------------------------------------------------------------------

@router.get("/usage")
async def usage(user: dict = Depends(get_current_user)):
    partner = await _require_partner(user)
    balance = await get_partner_balance(partner["id"])
    ledger = await list_partner_ledger(partner["id"], limit=50)
    return {
        "partner_id": partner["id"],
        "company_name": partner["company_name"],
        "credit_balance": balance,
        "status": partner["status"],
        "rate_limit_per_min": partner.get("rate_limit_per_min") or
                              settings.enterprise_default_rate_limit_per_min,
        "ledger": ledger,
    }


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

@router.get("/keys")
async def list_keys(user: dict = Depends(get_current_user)):
    partner = await _require_partner(user)
    return {"keys": await list_partner_credentials(partner["id"])}


@router.post("/keys")
async def issue_key(user: dict = Depends(get_current_user)):
    """Provision a new credential pair. The plaintext secret is returned EXACTLY ONCE.

    No way to retrieve it after the response. The frontend must surface the
    "shown once — copy now" UX immediately and never persist the value.
    """
    partner = await _require_partner(user)
    cred = await create_credential(partner["id"])
    logger.info(
        "enterprise_key_issued_via_web",
        extra={
            "action": "enterprise_key_issued_via_web",
            "partner_id": partner["id"],
            "credential_id": cred["credential_id"],
            "firebase_uid": user["uid"],
        },
    )
    return cred


@router.post("/keys/{credential_id}/revoke")
async def revoke_key(credential_id: str, user: dict = Depends(get_current_user)):
    partner = await _require_partner(user)
    await revoke_credential(partner["id"], credential_id)
    return {"credential_id": credential_id, "status": "revoked"}
