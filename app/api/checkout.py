"""
Lemon Squeezy checkout endpoint.

POST /api/checkout/create
  Creates a Lemon Squeezy hosted checkout session for the authenticated
  user and returns the checkout URL.

  The user_id and env are embedded in custom_data so the webhook can tie
  the payment to the correct user account and skip test-mode events.

Required env vars:
  LEMONSQUEEZY_API_KEY   — Lemon Squeezy API key (Bearer token)
  LEMONSQUEEZY_STORE_ID  — Your Lemon Squeezy store ID
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.firebase_auth import get_current_user
from app.integrations.http_client import request_session
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Checkout"])

LEMONSQUEEZY_API_KEY = os.getenv("LEMONSQUEEZY_API_KEY", "")
LEMONSQUEEZY_STORE_ID = os.getenv("LEMONSQUEEZY_STORE_ID", "")
LEMONSQUEEZY_CHECKOUT_URL = "https://api.lemonsqueezy.com/v1/checkouts"

# Set to "test" when running in a staging/test environment
APP_ENV = os.getenv("APP_ENV", "prod")


class CheckoutRequest(BaseModel):
    variant_id: str


class CheckoutResponse(BaseModel):
    checkout_url: str


@router.post("/api/checkout/create", response_model=CheckoutResponse)
async def create_checkout(
    body: CheckoutRequest,
    user: dict = Depends(get_current_user),
):
    """
    Creates a Lemon Squeezy checkout for the authenticated user.

    Requires:
      Authorization: Bearer <firebase_id_token>
      Body: { "variant_id": "<lemon_squeezy_variant_id>" }

    Returns:
      { "checkout_url": "https://..." }
    """
    if not LEMONSQUEEZY_API_KEY:
        logger.error("[CHECKOUT] LEMONSQUEEZY_API_KEY is not set")
        raise HTTPException(status_code=503, detail="Payment service not configured")

    if not LEMONSQUEEZY_STORE_ID:
        logger.error("[CHECKOUT] LEMONSQUEEZY_STORE_ID is not set")
        raise HTTPException(status_code=503, detail="Payment service not configured")

    uid = user["uid"]

    payload = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "custom_price": None,
                "product_options": {},
                "checkout_options": {"embed": True},
                "checkout_data": {
                    "custom": {
                        "user_id": uid,
                        "env": APP_ENV,
                    }
                },
            },
            "relationships": {
                "store": {
                    "data": {"type": "stores", "id": LEMONSQUEEZY_STORE_ID}
                },
                "variant": {
                    "data": {"type": "variants", "id": body.variant_id}
                },
            },
        }
    }

    headers = {
        "Authorization": f"Bearer {LEMONSQUEEZY_API_KEY}",
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
    }

    try:
        async with request_session() as sess:
            async with sess.post(
                LEMONSQUEEZY_CHECKOUT_URL,
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status not in (200, 201):
                    error_body = await resp.text()
                    logger.error(
                        f"[CHECKOUT] Lemon Squeezy API error {resp.status} "
                        f"for uid={uid}: {error_body}"
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="Failed to create checkout session"
                    )

                data = await resp.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHECKOUT] Unexpected error for uid={uid}: {e}")
        raise HTTPException(status_code=502, detail="Failed to create checkout session")

    try:
        checkout_url = data["data"]["attributes"]["url"]
    except (KeyError, TypeError) as e:
        logger.error(f"[CHECKOUT] Unexpected Lemon Squeezy response shape: {e} — {data}")
        raise HTTPException(status_code=502, detail="Invalid response from payment service")

    logger.info(f"[CHECKOUT] Created checkout for uid={uid}, variant={body.variant_id}")
    return CheckoutResponse(checkout_url=checkout_url)


@router.get("/api/debug/variants")
async def debug_list_variants(user: dict = Depends(get_current_user)):
    """
    Temporary debug endpoint — lists all Lemon Squeezy variants for your store.
    Use this to find your variant IDs, then add them to LEMONSQUEEZY_VARIANTS
    in config.py and remove this endpoint.

    Requires: Authorization: Bearer <firebase_id_token>
    """
    if not LEMONSQUEEZY_API_KEY:
        raise HTTPException(status_code=503, detail="LEMONSQUEEZY_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {LEMONSQUEEZY_API_KEY}",
        "Accept": "application/vnd.api+json",
    }

    try:
        async with request_session() as sess:
            async with sess.get(
                f"https://api.lemonsqueezy.com/v1/variants?filter[store_id]={LEMONSQUEEZY_STORE_ID}",
                headers=headers,
            ) as resp:
                raw = await resp.json()

        variants = []
        for item in raw.get("data", []):
            attrs = item.get("attributes", {})
            variants.append({
                "variant_id": item.get("id"),
                "product_name": attrs.get("product_name"),
                "name": attrs.get("name"),
                "price_usd": attrs.get("price", 0) / 100,
                "status": attrs.get("status"),
            })

        current_mapping = settings.lemon_squeezy_variants

        return {
            "variants": variants,
            "current_config_mapping": current_mapping,
            "instruction": (
                "Copy the variant_id values above into LEMONSQUEEZY_VARIANTS "
                "in config.py, mapping each ID to its credit amount. "
                "Then delete this endpoint."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch variants: {e}")
