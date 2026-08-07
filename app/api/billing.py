"""
Billing routes: Google Play in-app purchase verification.

POST /api/billing/google/verify
  Verifies a consumable purchase token against the Google Play Developer API
  and grants the mapped credits exactly once per Google orderId. Works for
  both authenticated users (Authorization: Bearer) and guest wallets
  (X-Device-ID) — same dual-path model as /api/user/balance.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from pydantic import BaseModel

from app.config import settings
from app.core.auth import get_client_ip, validate_device_id
from app.core.firebase_auth import get_optional_user
from app.core.rate_limiter import check_rate_limit
from app.integrations import firebase as firebase_module
from app.integrations import redis_client as redis_module
from app.logging_config import user_id_var
from app.services.credit_engine import get_user_balance, grant_credits
from app.services.credits_service import get_guest_wallet, grant_guest_credits
from app.services.finance_service import log_transaction
from app.services.google_play_billing import (
    GooglePlayVerificationError,
    InvalidPurchaseError,
    get_product_purchase,
    is_configured,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Billing"])

# ProductPurchase.purchaseState: 0 = purchased, 1 = canceled, 2 = pending.
PURCHASE_STATE_PURCHASED = 0
# ProductPurchase.consumptionState: 0 = yet to be consumed, 1 = consumed.
CONSUMPTION_STATE_CONSUMED = 1

# Durable idempotency. Redis alone was the only thing standing between a
# purchase token and a second grant, and it holds the claim under a TTL — so a
# Redis flush, an eviction, or simply waiting out google_play_order_ttl_sec made
# every historical order re-grantable by replaying its token.
#
# Firestore is the system of record; Redis stays in front of it as the cheap
# race guard (SET NX is one round-trip and settles concurrent retries), but it is
# no longer the thing that makes double-granting impossible.
_ORDERS_COLLECTION = "processed_play_orders"


async def _durable_order_claim(order_id: str, owner_id: str, credits: int) -> bool:
    """
    Claim `order_id` permanently. True if this call won it, False if already
    granted. Fails CLOSED: if Firestore is unavailable we return False rather
    than grant, because an unverifiable claim is indistinguishable from a replay.
    """
    db = firebase_module.db
    if not db:
        logger.error(
            "google_play_durable_claim_unavailable",
            extra={"action": "google_play_durable_claim_unavailable", "order_id": order_id},
        )
        raise HTTPException(status_code=503, detail="Billing temporarily unavailable")

    ref = db.collection(_ORDERS_COLLECTION).document(order_id)
    snapshot = await ref.get()
    if snapshot.exists:
        return False

    await ref.set({
        "owner_id": owner_id,
        "credits": credits,
        "granted_at": SERVER_TIMESTAMP,
        # Written so a future voided-purchase reversal (M5) has the owner and the
        # amount to reverse without re-deriving them from Play.
        "reversed": False,
    })
    return True


async def _release_durable_claim(order_id: str) -> None:
    """
    Undo a claim whose grant then failed. Best-effort and never raises: it runs
    inside an exception handler, and masking the original billing error with a
    cleanup error would hide why the purchase failed.
    """
    db = firebase_module.db
    if not db:
        return
    try:
        await db.collection(_ORDERS_COLLECTION).document(order_id).delete()
    except Exception as e:
        # Now genuinely stuck: the grant failed AND the claim survived, so retries
        # will report "already granted" for an ungranted purchase. Loud on purpose
        # — this needs a human, and the order_id is the handle to fix it with.
        logger.error(
            "google_play_durable_claim_release_failed",
            extra={
                "action": "google_play_durable_claim_release_failed",
                "order_id": order_id,
                "error": str(e),
            },
        )


class GoogleVerifyRequest(BaseModel):
    product_id: str
    purchase_token: str


class GoogleVerifyResponse(BaseModel):
    status: str
    credits_granted: int
    new_balance: int
    order_id: str


async def _current_balance(auth_user: dict | None, owner_id: str) -> int:
    """Reads the current balance for either wallet type (duplicate-order path)."""
    if auth_user:
        return await get_user_balance(owner_id)
    wallet = await get_guest_wallet(owner_id)
    return wallet.get("credits", 0)


@router.post("/api/billing/google/verify", response_model=GoogleVerifyResponse)
async def verify_google_purchase(
    request: Request,
    body: GoogleVerifyRequest,
    device_id: str | None = Header(None, alias="X-Device-ID"),
    auth_user: dict | None = Depends(get_optional_user),
):
    """
    Verify a Google Play consumable purchase and grant credits.

    - Server-side verification via purchases.products.get — the client's word
      is never trusted for purchase validity or credit amounts.
    - Idempotent per Google orderId: replays return credits_granted=0 with
      the current balance (SETNX claim in Redis, claimed BEFORE granting).
    - The Android client consumes the purchase only after this call succeeds.

    Errors:
      400 UNKNOWN_PRODUCT        — SKU not in the server-side mapping
      400 INVALID_PURCHASE       — token unverifiable, canceled, or pending
      503 BILLING_NOT_CONFIGURED — GOOGLE_PLAY_SERVICE_ACCOUNT_JSON missing
    """
    # IP-based limit FIRST — the guest owner_id below is the caller-supplied
    # X-Device-ID header, which an attacker rotates to get a fresh bucket per
    # request. The per-IP bucket (same pattern as /api/credits/add) caps
    # outbound androidpublisher calls regardless of identifier rotation.
    await check_rate_limit(f"billing:ip:{get_client_ip(request)}")

    if auth_user:
        owner_id = auth_user["uid"]
    else:
        validate_device_id(device_id)
        owner_id = device_id
    user_id_var.set(owner_id)

    # Per-owner limit kept as defense in depth (authenticated uid / device).
    await check_rate_limit(f"billing:{owner_id}")

    credits = settings.google_play_products.get(body.product_id)
    if not credits:
        logger.warning(
            "google_play_unknown_product",
            extra={
                "action": "google_play_unknown_product",
                "product_id": body.product_id,
                "known_products": list(settings.google_play_products.keys()),
            },
        )
        raise HTTPException(status_code=400, detail="UNKNOWN_PRODUCT")

    if not is_configured():
        logger.error(
            "google_play_not_configured",
            extra={
                "action": "google_play_not_configured",
                "detail": "GOOGLE_PLAY_SERVICE_ACCOUNT_JSON is not set — verification impossible",
            },
        )
        raise HTTPException(status_code=503, detail="BILLING_NOT_CONFIGURED")

    try:
        purchase = await get_product_purchase(body.product_id, body.purchase_token)
    except InvalidPurchaseError:
        raise HTTPException(status_code=400, detail="INVALID_PURCHASE") from None
    except GooglePlayVerificationError:
        raise HTTPException(status_code=502, detail="Purchase verification unavailable") from None

    purchase_state = purchase.get("purchaseState")
    order_id = purchase.get("orderId")
    if purchase_state != PURCHASE_STATE_PURCHASED or not order_id:
        logger.warning(
            "google_play_purchase_invalid_state",
            extra={
                "action": "google_play_purchase_invalid_state",
                "product_id": body.product_id,
                "purchase_state": purchase_state,
                "has_order_id": bool(order_id),
            },
        )
        raise HTTPException(status_code=400, detail="INVALID_PURCHASE")

    # Google's own durable "this was already used" signal, and it outlives any
    # cache we keep. The client consumes ONLY after we return 200 (see
    # BillingManager), so a first grant always sees 0 here; seeing 1 means the
    # token was already granted and consumed, and replaying it must not pay out
    # again even if every cache we own has been wiped.
    if purchase.get("consumptionState") == CONSUMPTION_STATE_CONSUMED:
        balance = await _current_balance(auth_user, owner_id)
        logger.warning(
            "google_play_already_consumed",
            extra={
                "action": "google_play_already_consumed",
                "order_id": order_id,
                "product_id": body.product_id,
            },
        )
        return GoogleVerifyResponse(
            status="ok", credits_granted=0, new_balance=balance, order_id=order_id
        )

    rc = redis_module.client
    if not rc:
        # Never grant without a working idempotency store — a Redis outage
        # must not open a double-grant window. The client retries later.
        raise HTTPException(status_code=503, detail="Billing temporarily unavailable")

    order_key = f"billing:google:order:{order_id}"
    claimed = await rc.set(order_key, owner_id, nx=True, ex=settings.google_play_order_ttl_sec)
    if not claimed:
        balance = await _current_balance(auth_user, owner_id)
        logger.info(
            "google_play_duplicate_order",
            extra={
                "action": "google_play_duplicate_order",
                "order_id": order_id,
                "product_id": body.product_id,
            },
        )
        return GoogleVerifyResponse(
            status="ok", credits_granted=0, new_balance=balance, order_id=order_id
        )

    # Durable claim, taken AFTER the Redis claim so the cheap guard settles
    # concurrent retries first, and BEFORE the grant so a crash between the two
    # cannot pay out twice. Losing it means Firestore already has this order.
    if not await _durable_order_claim(order_id, owner_id, credits):
        balance = await _current_balance(auth_user, owner_id)
        logger.info(
            "google_play_duplicate_order_durable",
            extra={
                "action": "google_play_duplicate_order_durable",
                "order_id": order_id,
                "product_id": body.product_id,
            },
        )
        return GoogleVerifyResponse(
            status="ok", credits_granted=0, new_balance=balance, order_id=order_id
        )

    try:
        if auth_user:
            new_balance = await grant_credits(owner_id, credits, "google_play_purchase", order_id)
        else:
            new_balance = await grant_guest_credits(owner_id, credits)
    except Exception:
        # Release BOTH idempotency claims so a client retry can re-grant —
        # otherwise a Firestore hiccup would permanently eat the purchase.
        #
        # The durable one matters most here: it is checked before the grant, so
        # leaving it behind after a failed grant would make every retry return
        # "already granted, 0 credits" for a purchase the user paid for and never
        # received. That is strictly worse than the double-grant this whole
        # mechanism exists to prevent.
        try:
            await rc.delete(order_key)
        except Exception as release_err:
            logger.error(
                "google_play_order_key_release_failed",
                extra={
                    "action": "google_play_order_key_release_failed",
                    "order_id": order_id,
                    "error": str(release_err),
                },
            )
        await _release_durable_claim(order_id)
        raise

    log_transaction(
        "GOOGLE_PLAY",
        0.0,  # USD amount lives in Play Console reports; credits tracked in meta
        {
            "owner_id": owner_id,
            "order_id": order_id,
            "product_id": body.product_id,
            "credits": credits,
            "is_guest": auth_user is None,
        },
    )
    logger.info(
        "google_play_purchase_granted",
        extra={
            "action": "google_play_purchase_granted",
            "order_id": order_id,
            "product_id": body.product_id,
            "credits_granted": credits,
            "new_balance": new_balance,
            "is_guest": auth_user is None,
        },
    )

    return GoogleVerifyResponse(
        status="ok", credits_granted=credits, new_balance=new_balance, order_id=order_id
    )
