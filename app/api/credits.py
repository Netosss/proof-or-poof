"""
Credit management routes: balance check, top-up (POST), ads reward, AdMob SSV.
"""

import logging
import re
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.cloud.firestore_v1.async_transaction import async_transactional
from pydantic import BaseModel

from app.config import settings
from app.core.auth import get_client_ip, validate_device_id
from app.core.firebase_auth import get_current_user, get_optional_user
from app.core.rate_limiter import check_rate_limit
from app.integrations import firebase as firebase_module
from app.logging_config import user_id_var
from app.schemas.credits import RechargeRequest
from app.services.admob_ssv import verified_params, verify_ssv
from app.services.credit_engine import get_user_balance, grant_credits
from app.services.credits_service import (
    get_guest_wallet,
    grant_guest_credits,
    perform_recharge,
)
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Credits"])

AD_REWARD_CREDITS = settings.ad_reward_credits
AD_REWARD_DAILY_LIMIT = settings.ad_reward_daily_limit

# custom_data as stamped by the Android client (RewardedAdManager):
#   "uid:<firebase-uid>"    signed in
#   "device:<device-id>"    guest
# The subject charset mirrors validate_device_id, because this value becomes a
# Firestore document id and it is ATTACKER-INFLUENCED: a modified client can
# stamp any string and Google will faithfully sign it. A valid signature proves
# Google sent the callback, never that the subject is honest. Without the charset
# guard a subject containing "/" becomes a nested Firestore path, yielding an
# unlimited supply of fresh cap documents and making the daily limit moot.
_CUSTOM_DATA_RE = re.compile(r"^(uid|device):([a-zA-Z0-9\-_.]{1,128})$")
# Transitional: builds predating the prefix stamp a bare uid.
_BARE_UID_RE = re.compile(r"^[a-zA-Z0-9\-_.]{1,128}$")
# Firestore rejects these as document-id segments — reaching a write with one
# would be a 500 rather than a clean ignore.
_RESERVED_IDS = {".", ".."}


def parse_custom_data(raw: str | None) -> tuple[str, str] | None:
    """
    Parses SSV `custom_data` into (kind, subject_id), or None when unusable.

    `kind` is "uid" (signed-in → users/) or "device" (guest → guest_wallets/).
    Those are separate collections with different balance fields, so the prefix
    decides which wallet is credited and must never be dropped.

    Returns None rather than raising: a validly-signed callback we cannot act on
    is ACKed with 200, because a 4xx makes AdMob back off the whole ad unit's
    callbacks.
    """
    if not raw:
        return None
    match = _CUSTOM_DATA_RE.match(raw)
    if match:
        kind, subject = match.group(1), match.group(2)
    elif _BARE_UID_RE.match(raw):
        # Legacy client — an unprefixed value is a Firebase uid.
        kind, subject = "uid", raw
    else:
        return None
    if subject in _RESERVED_IDS or subject.startswith("__"):
        return None
    return kind, subject


class AdRewardResponse(BaseModel):
    credits_granted: int
    new_balance: int
    rewards_today: int


class AdSsvResponse(BaseModel):
    # "ok" | "duplicate" | "capped" | "ignored"
    # "ignored" covers every validly-signed callback we deliberately do not act
    # on: AdMob's own callback-verify ping (no custom_data), a foreign ad_unit,
    # a stale timestamp, or an unparseable subject. All are 200 by design.
    status: str


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
        # A balance READ is free and grants nothing, so it must NOT be gated by the
        # per-IP new-device limit. That gate raised 403 STRICT_CAPTCHA_REQUIRED
        # before get_guest_wallet ran, so a new guest behind a shared IP (multiple
        # testers, or one person reinstalling — each reinstall = a fresh device id)
        # never got their wallet created and saw 0 credits instead of the 40 welcome
        # bonus. Abuse is gated where it actually costs: /detect and /inpaint require
        # a Turnstile token. Keep a per-IP rate limit here to stop hammering the read.
        await check_rate_limit(f"balance:ip:{get_client_ip(request)}")
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


# GET /api/credits/webhook was removed here, deliberately and not replaced.
#
# It took the unlimited-credit-mint secret as a URL QUERY PARAMETER, which means
# the secret was written verbatim into every access log, proxy log, and Referer
# header on the path — Cloudflare and Railway included. `amount` was unbounded
# and unvalidated on top of that. A GET cannot be fixed while keeping its shape,
# because the secret being in the URL *is* the shape.
#
# POST /api/credits/add does the same job with the secret in a JSON body and
# `amount` now bounded to [1, settings.max_recharge_amount]. Nothing called this
# route: not the web frontend (which only calls /api/user/balance), not the
# Android client, only its own tests.
#
# If an external integration turns out to have depended on it, restore from git
# history — but restore it as a POST with the secret in a header, never as a GET.


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
    subject_id: str, kind: str = "uid", reference_id: str | None = None
) -> tuple[int | None, int, int | None]:
    """
    Cap-checked ad-reward grant shared by BOTH the client endpoint and the AdMob
    SSV callback, so the two paths share ONE daily cap and can never together
    exceed the daily limit per UTC day.

    `kind` selects the wallet: "uid" credits users/{id} via the credit engine
    (ledgered); "device" credits guest_wallets/{id} via grant_guest_credits
    (no ledger, different balance field).

    **The cap document is namespaced by kind** — `ad_rewards/{kind}_{id}_{date}`.
    Keying on the bare id would let an attacker stamp `device:<a-real-firebase-
    uid>` and burn that user's daily rewards using their own ad views: a free,
    untraceable griefing vector. Note this changes the doc id from the previous
    `{uid}_{date}`, so counters reset once on deploy — at most one extra set of
    rewards per user, which is the right trade for closing the collision.

    Returns (credits_granted, rewards_today, new_balance). credits_granted (and
    new_balance) are None when the daily cap is already reached.
    """
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    limit = (
        settings.ad_reward_guest_daily_limit if kind == "device" else settings.ad_reward_daily_limit
    )
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    doc_id = f"{kind}_{subject_id}_{today}"
    reward_ref = db.collection("ad_rewards").document(doc_id)

    @async_transactional
    async def _check_and_grant(transaction, ref):
        snap = await ref.get(transaction=transaction)
        count = snap.to_dict().get("count", 0) if snap.exists else 0
        if count >= limit:
            return None, count
        transaction.set(
            ref,
            {
                "user_id": subject_id,
                "subject_kind": kind,
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
        if kind == "device":
            # Guests have no ledger and a different balance field, so they go
            # through the guest service rather than the credit engine.
            new_balance = await grant_guest_credits(subject_id, credits_to_grant)
        else:
            new_balance = await grant_credits(
                subject_id, credits_to_grant, "ad_reward", reference_id or f"{doc_id}_{new_count}"
            )
    except Exception:
        try:
            await _release_cap_slot(db, reward_ref)
        except Exception as release_err:
            logger.error(
                "ad_reward_cap_release_failed",
                extra={
                    "action": "ad_reward_cap_release_failed",
                    "uid": subject_id,
                    "subject_kind": kind,
                    "error": str(release_err),
                },
            )
        raise

    log_transaction(
        "AD_REWARD",
        settings.ad_revenue_per_reward,
        {
            "uid": subject_id,
            "subject_kind": kind,
            "credits": credits_to_grant,
            "rewards_today": new_count,
        },
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
    We verify the ECDSA signature against Google's published keys, confirm the
    callback is for OUR ad unit, then grant credits to the subject named in
    `custom_data` — idempotent per `transaction_id` and capped per UTC day.

    AdMob ignores the response body — it only checks for HTTP 200. We return
    200 on accept/duplicate/ignored, 403 only on an invalid signature. A 4xx on
    anything else would make AdMob back off the whole ad unit's callbacks.

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

    # ---- Bind the callback to OUR ad unit. This is the security boundary. ----
    # AdMob's SSV verifier keys are a SINGLE GLOBAL KEY SET shared by every
    # publisher, so a valid signature proves only that *some* AdMob server sent
    # this — never that our app did. Without this check anyone can create their
    # own AdMob account, point its SSV callback URL here, stamp custom_data with
    # a wallet of their choosing, and mint credits funded by an impression Google
    # pays THEM for. ad_unit is inside the signed content, so it cannot be forged.
    expected_unit = settings.admob_rewarded_ad_unit_id
    if not expected_unit:
        # Fail closed: an unconfigured deployment must not grant.
        logger.error(
            "admob_ssv_unit_not_configured",
            extra={"action": "admob_ssv_unit_not_configured"},
        )
        return {"status": "ignored"}
    if signed.get("ad_unit") != expected_unit:
        logger.warning(
            "admob_ssv_foreign_ad_unit",
            extra={"action": "admob_ssv_foreign_ad_unit", "ad_unit": signed.get("ad_unit")},
        )
        return {"status": "ignored"}

    # Freshness — defence in depth behind the per-transaction_id claim. AdMob
    # sends `timestamp` in milliseconds and it is inside the signed content.
    max_age = settings.admob_ssv_max_age_sec
    if max_age > 0:
        raw_ts = signed.get("timestamp")
        if raw_ts:
            try:
                age_sec = abs(datetime.now(UTC).timestamp() - int(raw_ts) / 1000)
            except (TypeError, ValueError):
                age_sec = None
            if age_sec is not None and age_sec > max_age:
                logger.warning(
                    "admob_ssv_stale_callback",
                    extra={"action": "admob_ssv_stale_callback", "age_sec": int(age_sec)},
                )
                return {"status": "ignored"}

    transaction_id = signed.get("transaction_id")
    # AdMob's own "verify callback URL" ping is validly signed but carries NO
    # custom_data. That case, and any custom_data we cannot safely act on, is
    # ACKed with 200 (AdMob requires a 200 to accept the URL) and grants nothing.
    parsed = parse_custom_data(signed.get("custom_data"))
    if parsed is None or not transaction_id:
        logger.info(
            "admob_ssv_no_custom_data",
            extra={"action": "admob_ssv_no_custom_data"},
        )
        return {"status": "ignored"}
    subject_kind, subject_id = parsed
    user_id_var.set(subject_id)

    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    # Idempotency: exactly one grant per AdMob transaction_id. Google may retry
    # the callback, so we claim the transaction transactionally before granting.
    # transaction_id is Google-generated and inside the signed content, but it is
    # still used as a document id — validate the shape rather than trusting it.
    if not _BARE_UID_RE.match(transaction_id) or transaction_id in _RESERVED_IDS:
        logger.warning(
            "admob_ssv_bad_transaction_id",
            extra={"action": "admob_ssv_bad_transaction_id"},
        )
        return {"status": "ignored"}
    txn_ref = db.collection("ad_ssv_rewards").document(transaction_id)

    @async_transactional
    async def _claim(transaction, ref):
        snap = await ref.get(transaction=transaction)
        if snap.exists:
            return False
        transaction.set(
            ref,
            {
                "user_id": subject_id,
                "subject_kind": subject_kind,
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
        credits_to_grant, new_count, _ = await _apply_ad_reward(
            subject_id, subject_kind, f"ssv_{transaction_id}"
        )
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
