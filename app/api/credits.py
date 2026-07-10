"""
Credit management routes: balance check, top-up (POST & GET webhook), ads reward.
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.cloud.firestore_v1.async_transaction import async_transactional
from pydantic import BaseModel

from app.config import settings
from app.core.auth import check_ip_device_limit, get_client_ip, validate_device_id
from app.core.firebase_auth import get_current_user, get_optional_user
from app.core.rate_limiter import check_rate_limit
from app.integrations import firebase as firebase_module
from app.logging_config import user_id_var
from app.schemas.credits import RechargeRequest
from app.services.admob_ssv import verified_params, verify_ssv
from app.services.credit_engine import get_user_balance, grant_credits
from app.services.credits_service import get_guest_wallet, perform_recharge
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Credits"])

AD_REWARD_CREDITS = 20
AD_REWARD_DAILY_LIMIT = 3


class AdRewardResponse(BaseModel):
    credits_granted: int
    new_balance: int
    rewards_today: int


class AdSsvResponse(BaseModel):
    status: str  # "ok" | "duplicate" | "capped"


@router.get("/api/user/balance")
async def get_balance(
    request: Request,
    device_id: str | None = Header(None, alias="X-Device-ID"),
    turnstile_token: str | None = Header(None, alias="X-Turnstile-Token"),
    auth_user: dict | None = Depends(get_optional_user),
):
    """
    Returns the current credit balance.

    - Authenticated (Authorization: Bearer token): reads from users/{uid}.
      Subject to a per-uid rate limit.
    - Guest (no Authorization header): reads from guest_wallets/{device_id}.
      Subject to IP/device rate limiting as before.
    """
    if auth_user:
        user_id_var.set(auth_user["uid"])
        await check_rate_limit(f"balance:{auth_user['uid']}")
        balance = await get_user_balance(auth_user["uid"])
    else:
        user_id_var.set(device_id or "")
        validate_device_id(device_id)
        ip = get_client_ip(request)
        await check_ip_device_limit(ip, device_id, turnstile_token)
        wallet = await get_guest_wallet(device_id)
        balance = wallet.get("credits", 0)
    logger.info("balance_queried", extra={"action": "balance_queried", "balance": balance})
    return {"balance": balance}


@router.post("/api/credits/add")
async def add_credits_post(request: Request, payload: RechargeRequest):
    user_id_var.set(payload.device_id)
    await check_rate_limit(f"recharge:{get_client_ip(request)}")
    result = await perform_recharge(payload.device_id, payload.amount, payload.secret_key)
    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {"device_id": payload.device_id, "credits": payload.amount},
    )
    return result


@router.get("/api/credits/webhook")
async def add_credits_get(
    request: Request,
    user_id: str = Query(..., alias="device_id"),
    amount: int = settings.default_recharge_amount,
    key: str = Query(..., alias="secret_key"),
):
    user_id_var.set(user_id)
    await check_rate_limit(f"recharge:{get_client_ip(request)}")
    result = await perform_recharge(user_id, amount, key)
    log_transaction(
        "AD_REWARD", settings.ad_revenue_per_reward, {"device_id": user_id, "credits": amount}
    )
    return result


async def _release_cap_slot(db, reward_ref) -> None:
    """
    Compensating decrement of the daily ad-reward counter, used when a grant
    fails AFTER the cap slot was already committed — so a transient error never
    silently burns one of the user's daily rewards.
    """

    @async_transactional
    async def _dec(transaction, ref):
        snap = await ref.get(transaction=transaction)
        count = snap.to_dict().get("count", 0) if snap.exists else 0
        if count > 0:
            transaction.update(ref, {"count": count - 1})

    await _dec(db.transaction(), reward_ref)


async def _apply_ad_reward(
    uid: str, reference_id: str | None = None
) -> tuple[int | None, int, int | None]:
    """
    Cap-checked ad-reward grant shared by BOTH the client endpoint and the AdMob
    SSV callback, so the two paths share ONE daily cap and can never together
    exceed AD_REWARD_DAILY_LIMIT grants per UTC day.

    Enforces the cap via a Firestore transaction on ad_rewards/{uid}_{date},
    then grants AD_REWARD_CREDITS through the credit engine.

    Returns (credits_granted, rewards_today, new_balance). credits_granted (and
    new_balance) are None when the daily cap is already reached.
    """
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    doc_id = f"{uid}_{today}"
    reward_ref = db.collection("ad_rewards").document(doc_id)

    @async_transactional
    async def _check_and_grant(transaction, ref):
        snap = await ref.get(transaction=transaction)
        count = snap.to_dict().get("count", 0) if snap.exists else 0
        if count >= AD_REWARD_DAILY_LIMIT:
            return None, count
        transaction.set(
            ref,
            {
                "user_id": uid,
                "date": today,
                "count": count + 1,
                "last_reward_at": SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return AD_REWARD_CREDITS, count + 1

    txn = db.transaction()
    credits_to_grant, new_count = await _check_and_grant(txn, reward_ref)
    if credits_to_grant is None:
        return None, new_count, None

    # The cap slot is committed above. If the grant fails, hand the slot back so
    # a transient error doesn't permanently consume a daily reward. Use a unique
    # per-grant reference_id (never the same doc_id 3x/day) for a clean ledger.
    try:
        new_balance = await grant_credits(
            uid, credits_to_grant, "ad_reward", reference_id or f"{doc_id}_{new_count}"
        )
    except Exception:
        try:
            await _release_cap_slot(db, reward_ref)
        except Exception as release_err:
            logger.error(
                "ad_reward_cap_release_failed",
                extra={
                    "action": "ad_reward_cap_release_failed",
                    "uid": uid,
                    "error": str(release_err),
                },
            )
        raise

    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {"uid": uid, "credits": credits_to_grant, "rewards_today": new_count},
    )
    return credits_to_grant, new_count, new_balance


@router.post("/api/ads/reward", response_model=AdRewardResponse)
async def ads_reward(
    user: dict = Depends(get_current_user),
):
    """
    Grant credits to an authenticated user for watching an ad (client-attested).

    - Maximum 3 rewards per UTC day (server-side date, never client-provided).
    - Each reward grants 20 credits.
    - Shares the daily cap doc with the AdMob SSV callback.

    NOTE: this endpoint trusts the client's word that an ad was watched. The
    secure path is /api/ads/ssv (Google-signed). Once SSV is enabled in the
    AdMob console and the client stops calling this endpoint, it can be removed.

    Requires:
      Authorization: Bearer <firebase_id_token>
    """
    uid = user["uid"]
    user_id_var.set(uid)

    try:
        credits_to_grant, new_count, new_balance = await _apply_ad_reward(uid)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ads_reward_failed", extra={"action": "ads_reward_failed", "error": str(e)})
        raise HTTPException(status_code=500, detail="Ad reward failed") from e

    if credits_to_grant is None:
        raise HTTPException(
            status_code=429,
            detail=f"Daily ad reward limit reached ({AD_REWARD_DAILY_LIMIT} per day)",
        )
    # _apply_ad_reward returns credits and balance together: a non-None grant
    # always carries a non-None balance.
    assert new_balance is not None

    logger.info(
        "ads_reward_granted",
        extra={
            "action": "ads_reward_granted",
            "credits_granted": credits_to_grant,
            "rewards_today": new_count,
            "daily_limit": AD_REWARD_DAILY_LIMIT,
            "new_balance": new_balance,
        },
    )

    return AdRewardResponse(
        credits_granted=credits_to_grant,
        new_balance=new_balance,
        rewards_today=new_count,
    )


@router.get("/api/ads/ssv", response_model=AdSsvResponse)
async def ads_ssv(request: Request):
    """
    AdMob rewarded Server-Side Verification (SSV) callback.

    Google calls this GET with a signed query string after a verified ad view.
    We verify the ECDSA signature against Google's published keys, then grant
    credits to the user named in `custom_data` (the Firebase UID stamped by the
    Android client), idempotent per `transaction_id` and capped per day.

    AdMob ignores the response body — it only checks for HTTP 200. We return
    200 on accept/duplicate/ignored, 403 only on an invalid signature.

    Set this endpoint's URL in AdMob → rewarded ad unit → Server-side verification.
    """
    signature = request.query_params.get("signature", "")
    key_id = request.query_params.get("key_id", "")
    if not await verify_ssv(request.url.query, signature, key_id):
        logger.warning(
            "admob_ssv_invalid_signature",
            extra={"action": "admob_ssv_invalid_signature"},
        )
        raise HTTPException(status_code=403, detail="Invalid signature")

    # Read acted-upon fields from the SIGNED content only. request.query_params is
    # last-wins on duplicate keys, so reading from it would let an attacker append
    # `&transaction_id=forged&custom_data=other` after the signature and smuggle
    # forged values past verification. verified_params() ignores anything after
    # `&signature=`.
    signed = verified_params(request.url.query)
    uid = signed.get("custom_data")
    transaction_id = signed.get("transaction_id")
    # AdMob's own "verify callback URL" ping is validly signed but carries NO
    # custom_data, and guest ad views have no Firebase uid to stamp either. In
    # both cases the signature is legitimate, so ACK with 200 (AdMob requires a
    # 200 to accept the URL) and simply grant nothing — only signed-in views that
    # stamp custom_data reach the grant path below.
    if not uid or not transaction_id:
        logger.info(
            "admob_ssv_no_custom_data",
            extra={"action": "admob_ssv_no_custom_data"},
        )
        return {"status": "ignored"}
    user_id_var.set(uid)

    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    # Idempotency: exactly one grant per AdMob transaction_id. Google may retry
    # the callback, so we claim the transaction transactionally before granting.
    txn_ref = db.collection("ad_ssv_rewards").document(transaction_id)

    @async_transactional
    async def _claim(transaction, ref):
        snap = await ref.get(transaction=transaction)
        if snap.exists:
            return False
        transaction.set(
            ref,
            {
                "user_id": uid,
                "transaction_id": transaction_id,
                "created_at": SERVER_TIMESTAMP,
            },
        )
        return True

    try:
        first_time = await _claim(db.transaction(), txn_ref)
    except Exception as e:
        logger.error(
            "admob_ssv_claim_failed",
            extra={"action": "admob_ssv_claim_failed", "error": str(e)},
        )
        raise HTTPException(status_code=500, detail="SSV processing failed") from e

    if not first_time:
        logger.info(
            "admob_ssv_duplicate",
            extra={"action": "admob_ssv_duplicate", "transaction_id": transaction_id},
        )
        return {"status": "duplicate"}

    # The claim doc is already committed. If the grant fails we MUST release it,
    # else Google's retry sees the claim, returns "duplicate", and the real ad
    # view is credited to no one (permanent silent loss). Mirrors billing.py.
    try:
        credits_to_grant, new_count, _ = await _apply_ad_reward(uid, f"ssv_{transaction_id}")
    except Exception as e:
        try:
            await txn_ref.delete()
        except Exception as release_err:
            logger.error(
                "admob_ssv_claim_release_failed",
                extra={
                    "action": "admob_ssv_claim_release_failed",
                    "transaction_id": transaction_id,
                    "error": str(release_err),
                },
            )
        logger.error(
            "admob_ssv_grant_failed",
            extra={
                "action": "admob_ssv_grant_failed",
                "transaction_id": transaction_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="SSV grant failed") from e

    logger.info(
        "admob_ssv_processed",
        extra={
            "action": "admob_ssv_processed",
            "credits_granted": credits_to_grant,
            "rewards_today": new_count,
            "transaction_id": transaction_id,
        },
    )
    return {"status": "ok" if credits_to_grant is not None else "capped"}
