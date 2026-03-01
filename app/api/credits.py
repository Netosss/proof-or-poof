"""
Credit management routes: balance check, top-up (POST & GET webhook).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, Query, Request

from app.core.auth import check_ip_device_limit, get_client_ip, validate_device_id
from app.schemas.credits import RechargeRequest
from app.services.credits_service import get_guest_wallet, perform_recharge
from app.services.finance_service import log_transaction
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Credits"])


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
    wallet = get_guest_wallet(device_id)
    balance = wallet.get("credits", 0)
    logger.info(f"[BALANCE] Device: {device_id} | Credits: {balance}")
    return {"balance": balance}


@router.post("/api/credits/add")
async def add_credits_post(payload: RechargeRequest):
    result = perform_recharge(payload.device_id, payload.amount, payload.secret_key)
    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {"device_id": payload.device_id, "credits": payload.amount}
    )
    return result


@router.get("/api/credits/webhook")
async def add_credits_get(
    user_id: str = Query(..., alias="device_id"),
    amount: int = settings.default_recharge_amount,
    key: str = Query(..., alias="secret_key")
):
    result = perform_recharge(user_id, amount, key)
    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {"device_id": user_id, "credits": amount}
    )
    return result
