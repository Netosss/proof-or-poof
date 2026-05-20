"""
Enterprise Lemon Squeezy webhook handlers.

Extracted from `app/api/webhooks.py` to keep that file under the project's
800-line ceiling and to co-locate enterprise concerns under `app/api/enterprise/*`.

The top-level webhook router still lives in `webhooks.py` (single URL configured
in Lemon Squeezy). It routes by `meta.custom_data.account_type` and delegates to
the two functions in this module for enterprise-tagged events.
"""

import logging

from fastapi import HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.cloud.firestore_v1.async_transaction import async_transactional

from app.config import settings
from app.integrations import firebase as firebase_module
from app.services.enterprise_applications import (
    find_application_for_uid,
    mark_application_approved,
)
from app.services.enterprise_credit_engine import grant_credit as ent_grant_credit
from app.services.enterprise_credit_engine import refund_credit as ent_refund_credit
from app.services.enterprise_partners import create_partner, find_partner_for_uid
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)


async def handle_order_paid(meta, attributes, order_id: str, custom_data: dict):
    """Grant enterprise credits when a partner pays for a prepaid package.

    Reuses the `purchases/{order_id}` idempotency surface and disambiguates from
    consumer purchases on the `account_type` field.
    """
    payload_env = custom_data.get("env", "")
    is_test_payload = meta.get("test_mode") is True

    if settings.is_dev:
        if not is_test_payload or payload_env != "dev":
            logger.info(
                "lemonsqueezy_enterprise_payload_skipped",
                extra={
                    "action": "lemonsqueezy_enterprise_payload_skipped",
                    "order_id": order_id,
                    "reason": "wrong_env_for_dev_server",
                },
            )
            return {"status": "ok", "skipped": "wrong_env_for_dev_server"}
    else:
        if is_test_payload or payload_env != "prod":
            logger.info(
                "lemonsqueezy_enterprise_payload_skipped",
                extra={
                    "action": "lemonsqueezy_enterprise_payload_skipped",
                    "order_id": order_id,
                    "reason": "wrong_env_for_prod_server",
                },
            )
            return {"status": "ok", "skipped": "wrong_env_for_prod_server"}

    partner_id = custom_data.get("partner_id")
    firebase_uid = custom_data.get("firebase_uid")

    if not partner_id:
        if not firebase_uid:
            logger.error(
                "lemonsqueezy_enterprise_missing_partner",
                extra={
                    "action": "lemonsqueezy_enterprise_missing_partner",
                    "order_id": order_id,
                    "reason": "no_partner_id_and_no_firebase_uid",
                },
            )
            return {"status": "error", "reason": "missing partner_id and firebase_uid"}

        existing = await find_partner_for_uid(firebase_uid)
        if existing:
            partner_id = existing["id"]
            logger.info(
                "lemonsqueezy_enterprise_partner_resolved",
                extra={
                    "action": "lemonsqueezy_enterprise_partner_resolved",
                    "order_id": order_id,
                    "firebase_uid": firebase_uid,
                    "partner_id": partner_id,
                },
            )
        else:
            # First payment → create partner from whatever identity we have.
            application = await find_application_for_uid(firebase_uid)
            if application:
                company_name = application["company_name"]
                contact_email = application["contact_email"]
            else:
                contact_email = (
                    attributes.get("user_email")
                    or custom_data.get("buyer_email")
                    or ""
                )
                company_name = (
                    attributes.get("user_name")
                    or custom_data.get("buyer_name")
                    or contact_email.split("@", 1)[0]
                    or "Enterprise customer"
                )

            new_partner = await create_partner(
                company_name=company_name,
                contact_email=contact_email,
                initial_credits=0,
                firebase_uid=firebase_uid,
            )
            partner_id = new_partner["id"]
            if application:
                await mark_application_approved(application["id"], partner_id)
            logger.info(
                "lemonsqueezy_enterprise_partner_auto_provisioned",
                extra={
                    "action": "lemonsqueezy_enterprise_partner_auto_provisioned",
                    "order_id": order_id,
                    "firebase_uid": firebase_uid,
                    "partner_id": partner_id,
                    "application_id": (application or {}).get("id"),
                    "source": "application" if application else "ls_payload",
                },
            )

    first_item = attributes.get("first_order_item") or {}
    variant_id = str(first_item.get("variant_id", ""))
    credits = settings.active_enterprise_ls_variants.get(variant_id)
    if not credits:
        logger.error(
            "lemonsqueezy_enterprise_variant_not_found",
            extra={
                "action": "lemonsqueezy_enterprise_variant_not_found",
                "order_id": order_id,
                "variant_id": variant_id,
                "known_variants": list(settings.active_enterprise_ls_variants.keys()),
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Unknown enterprise variant {variant_id} — update ENTERPRISE_LS_VARIANTS",
        )

    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")

    purchase_ref = db.collection("purchases").document(order_id)
    total_usd = attributes.get("total", 0) / 100.0

    try:
        await purchase_ref.create({
            "lemon_order_id": order_id,
            "lemon_variant_id": variant_id,
            "partner_id": partner_id,
            "account_type": "enterprise",
            "credits_granted": credits,
            "status": "pending",
            "test_mode": is_test_payload,
            "created_at": SERVER_TIMESTAMP,
        })
    except Exception as e:
        if "AlreadyExists" in type(e).__name__ or "ALREADY_EXISTS" in str(e).upper():
            existing = await purchase_ref.get()
            existing_status = existing.to_dict().get("status") if existing.exists else None
            if existing_status == "paid":
                logger.info(
                    "lemonsqueezy_enterprise_duplicate_order",
                    extra={
                        "action": "lemonsqueezy_enterprise_duplicate_order",
                        "order_id": order_id,
                        "partner_id": partner_id,
                    },
                )
                return {"status": "ok", "skipped": "duplicate"}
            logger.warning(
                "lemonsqueezy_enterprise_retry_grant",
                extra={
                    "action": "lemonsqueezy_enterprise_retry_grant",
                    "order_id": order_id,
                    "partner_id": partner_id,
                    "prior_status": existing_status,
                },
            )
        else:
            raise HTTPException(status_code=500, detail="Purchase record failed")

    try:
        new_balance = await ent_grant_credit(
            partner_id, amount=credits, reason="purchase", reference_id=order_id,
        )
    except Exception as e:
        await purchase_ref.update({"status": "grant_failed"})
        logger.error(
            "lemonsqueezy_enterprise_grant_failed",
            extra={
                "action": "lemonsqueezy_enterprise_grant_failed",
                "order_id": order_id,
                "partner_id": partner_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Credit grant failed — will retry")

    await purchase_ref.update({"status": "paid"})

    log_transaction(
        "LEMONSQUEEZY_ENTERPRISE",
        total_usd,
        {"partner_id": partner_id, "order_id": order_id, "credits": credits},
    )
    logger.info(
        "lemonsqueezy_enterprise_order_paid",
        extra={
            "action": "lemonsqueezy_enterprise_order_paid",
            "order_id": order_id,
            "partner_id": partner_id,
            "credits_granted": credits,
            "total_usd": total_usd,
            "new_balance": new_balance,
        },
    )
    return {"status": "ok"}


async def handle_order_refunded(order_id: str):
    """Deduct enterprise credits when a partner order is refunded."""
    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")

    purchase_ref = db.collection("purchases").document(order_id)

    @async_transactional
    async def _claim_refund(transaction, ref):
        snap = await ref.get(transaction=transaction)
        if not snap.exists:
            return None, None, "not_found"
        purchase = snap.to_dict()
        if purchase.get("account_type") != "enterprise":
            return None, None, "wrong_account_type"
        if purchase.get("status") == "refunded":
            return None, None, "already_refunded"
        transaction.update(ref, {"status": "refunded"})
        return purchase.get("partner_id"), purchase.get("credits_granted", 0), "ok"

    try:
        txn = db.transaction()
        partner_id, credits_granted, outcome = await _claim_refund(txn, purchase_ref)
    except Exception as e:
        logger.error(
            "lemonsqueezy_enterprise_refund_failed",
            extra={
                "action": "lemonsqueezy_enterprise_refund_failed",
                "order_id": order_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="Refund processing failed")

    if outcome == "not_found":
        return {"status": "ok", "skipped": "purchase_not_found"}
    if outcome == "wrong_account_type":
        return {"status": "ok", "skipped": "not_enterprise_purchase"}
    if outcome == "already_refunded":
        return {"status": "ok", "skipped": "already_refunded"}

    # Refund semantics: take back the credits previously granted by passing a
    # negative amount to the additive ledger writer.
    try:
        new_balance = await ent_refund_credit(
            partner_id, amount=-credits_granted, reason="refund", reference_id=order_id,
        )
        logger.info(
            "lemonsqueezy_enterprise_order_refunded",
            extra={
                "action": "lemonsqueezy_enterprise_order_refunded",
                "order_id": order_id,
                "partner_id": partner_id,
                "credits_deducted": credits_granted,
                "new_balance": new_balance,
            },
        )
    except Exception as e:
        logger.error(
            "lemonsqueezy_enterprise_refund_deduction_failed",
            extra={
                "action": "lemonsqueezy_enterprise_refund_deduction_failed",
                "order_id": order_id,
                "partner_id": partner_id,
                "credits_granted": credits_granted,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="Enterprise refund deduction failed")

    return {"status": "ok"}
