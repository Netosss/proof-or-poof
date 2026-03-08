"""
Credit management routes: balance check, top-up (POST & GET webhook), ads reward.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.cloud.firestore_v1.async_transaction import async_transactional

from app.config import settings
from app.core.auth import check_ip_device_limit, get_client_ip, validate_device_id
from app.core.firebase_auth import get_current_user
from app.core.rate_limiter import check_rate_limit
from app.integrations import firebase as firebase_module
from app.schemas.credits import RechargeRequest
from app.services.credit_engine import grant_credits
from app.services.credits_service import get_guest_wallet, perform_recharge
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Credits"])

AD_REWARD_CREDITS = 5
AD_REWARD_DAILY_LIMIT = 3


class AdRewardResponse(BaseModel):
    credits_granted: int
    new_balance: int
    rewards_today: int


@router.get("/api/user/balance")
async def get_balance(
    request: Request,
    device_id: str = Header(..., alias="X-Device-ID"),
    turnstile_token: Optional[str] = Header(None, alias="X-Turnstile-Token")
):
    """
    Returns the current credit balance for a guest device.
    Auto-creates a wallet with welcome credits if one does not exist.
    """
    validate_device_id(device_id)
    ip = get_client_ip(request)
    await check_ip_device_limit(ip, device_id, turnstile_token)
    wallet = await get_guest_wallet(device_id)
    balance = wallet.get("credits", 0)
    logger.info("balance_queried", extra={"action": "balance_queried", "balance": balance})
    return {"balance": balance}


@router.post("/api/credits/add")
async def add_credits_post(request: Request, payload: RechargeRequest):
    await check_rate_limit(f"recharge:{get_client_ip(request)}")
    result = await perform_recharge(payload.device_id, payload.amount, payload.secret_key)
    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {"device_id": payload.device_id, "credits": payload.amount}
    )
    return result


@router.get("/api/credits/webhook")
async def add_credits_get(
    request: Request,
    user_id: str = Query(..., alias="device_id"),
    amount: int = settings.default_recharge_amount,
    key: str = Query(..., alias="secret_key")
):
    await check_rate_limit(f"recharge:{get_client_ip(request)}")
    result = await perform_recharge(user_id, amount, key)
    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {"device_id": user_id, "credits": amount}
    )
    return result


@router.post("/api/ads/reward", response_model=AdRewardResponse)
async def ads_reward(
    user: dict = Depends(get_current_user),
):
    """
    Grant credits to an authenticated user for watching an ad.

    - Maximum 3 rewards per UTC day (server-side date, never client-provided).
    - Each reward grants 5 credits.
    - Idempotency enforced via Firestore ad_rewards/{uid}_{date} document.

    Requires:
      Authorization: Bearer <firebase_id_token>
    """
    uid = user["uid"]
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc_id = f"{uid}_{today}"
    reward_ref = db.collection("ad_rewards").document(doc_id)

    @async_transactional
    async def _check_and_grant(transaction, ref):
        snap = await ref.get(transaction=transaction)
        count = snap.to_dict().get("count", 0) if snap.exists else 0

        if count >= AD_REWARD_DAILY_LIMIT:
            return None, count

        transaction.set(ref, {
            "user_id": uid,
            "date": today,
            "count": count + 1,
            "last_reward_at": SERVER_TIMESTAMP,
        }, merge=True)
        return AD_REWARD_CREDITS, count + 1

    try:
        txn = db.transaction()
        credits_to_grant, new_count = await _check_and_grant(txn, reward_ref)
    except Exception as e:
        logger.error("ads_reward_failed", extra={"action": "ads_reward_failed", "error": str(e)})
        raise HTTPException(status_code=500, detail="Ad reward failed")

    if credits_to_grant is None:
        raise HTTPException(
            status_code=429,
            detail=f"Daily ad reward limit reached ({AD_REWARD_DAILY_LIMIT} per day)"
        )

    new_balance = await grant_credits(uid, credits_to_grant, "ad_reward", doc_id)
    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {"uid": uid, "credits": credits_to_grant, "rewards_today": new_count}
    )

    logger.info("ads_reward_granted", extra={
        "action": "ads_reward_granted",
        "credits_granted": credits_to_grant,
        "rewards_today": new_count,
        "daily_limit": AD_REWARD_DAILY_LIMIT,
        "new_balance": new_balance,
    })

    return AdRewardResponse(
        credits_granted=credits_to_grant,
        new_balance=new_balance,
        rewards_today=new_count,
    )
