"""
Google Play Billing verification via the Play Developer API (androidpublisher v3).

Calls purchases.products.get to verify a consumable in-app purchase token
server-side. Auth uses a Google service account whose JSON key is provided
through the GOOGLE_PLAY_SERVICE_ACCOUNT_JSON env var (the JSON content itself,
same pattern as FIREBASE_SERVICE_ACCOUNT).

The env var is read at REQUEST time (not import time) so a rotated key takes
effect on the next request. Credentials are cached in-process and only rebuilt
when the env var content changes; access-token refresh is synchronous I/O in
google-auth, so it runs in the default thread pool.
"""

import asyncio
import json
import logging
import os
from urllib.parse import quote

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from app.config import settings
from app.integrations import http_client as http_module

logger = logging.getLogger(__name__)

_ANDROID_PUBLISHER_SCOPE = "https://www.googleapis.com/auth/androidpublisher"
_PURCHASE_URL = (
    "https://androidpublisher.googleapis.com/androidpublisher/v3/applications/"
    "{package_name}/purchases/products/{product_id}/tokens/{token}"
)

# Cached credentials + the raw JSON they were built from, so a rotated
# GOOGLE_PLAY_SERVICE_ACCOUNT_JSON invalidates the cache automatically.
_credentials: service_account.Credentials | None = None
_credentials_source: str | None = None


class InvalidPurchaseError(Exception):
    """Google Play rejected the product/token combination (bad or consumed token)."""


class GooglePlayVerificationError(Exception):
    """Google Play API unreachable or returned an unexpected error."""


def is_configured() -> bool:
    """True when the service-account key env var is present."""
    return bool(os.getenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON"))


def _load_credentials(raw_json: str) -> service_account.Credentials:
    global _credentials, _credentials_source
    if _credentials is None or _credentials_source != raw_json:
        info = json.loads(raw_json)
        _credentials = service_account.Credentials.from_service_account_info(
            info, scopes=[_ANDROID_PUBLISHER_SCOPE]
        )
        _credentials_source = raw_json
    return _credentials


async def _get_access_token() -> str:
    """Returns a valid OAuth2 access token for the androidpublisher scope."""
    raw_json = os.getenv("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", "")
    try:
        creds = _load_credentials(raw_json)
        if not creds.valid:
            # google-auth refresh is synchronous network I/O — run it in the
            # thread pool so the event loop is never blocked.
            await asyncio.to_thread(creds.refresh, GoogleAuthRequest())
        return creds.token
    except Exception as e:
        logger.error(
            "google_play_token_failed",
            extra={
                "action": "google_play_token_failed",
                "error": str(e),
            },
        )
        raise GooglePlayVerificationError("Service account token refresh failed") from e


async def get_product_purchase(product_id: str, purchase_token: str) -> dict:
    """
    Fetches the ProductPurchase resource for a consumable in-app purchase.

    GET /androidpublisher/v3/applications/{pkg}/purchases/products/{sku}/tokens/{token}

    Returns the parsed JSON resource on HTTP 200. The caller is responsible
    for checking purchaseState (0 = purchased) and extracting orderId.

    Raises:
        InvalidPurchaseError          — Google returned 400/404/410 (bad token,
                                        wrong SKU, or purchase no longer exists).
        GooglePlayVerificationError   — auth failure, network error, or any
                                        other unexpected Google response.
    """
    access_token = await _get_access_token()
    url = _PURCHASE_URL.format(
        package_name=quote(settings.android_package_name, safe=""),
        product_id=quote(product_id, safe=""),
        token=quote(purchase_token, safe=""),
    )

    try:
        async with http_module.request_session() as sess:
            async with sess.get(
                url, headers={"Authorization": f"Bearer {access_token}"}
            ) as response:
                if response.status == 200:
                    return await response.json()

                body_preview = (await response.text())[:500]
                if response.status in (400, 404, 410):
                    logger.warning(
                        "google_play_purchase_rejected",
                        extra={
                            "action": "google_play_purchase_rejected",
                            "status_code": response.status,
                            "product_id": product_id,
                            "body": body_preview,
                        },
                    )
                    raise InvalidPurchaseError(f"Google Play returned {response.status}")

                logger.error(
                    "google_play_api_error",
                    extra={
                        "action": "google_play_api_error",
                        "status_code": response.status,
                        "product_id": product_id,
                        "body": body_preview,
                    },
                )
                raise GooglePlayVerificationError(f"Google Play returned {response.status}")
    except (InvalidPurchaseError, GooglePlayVerificationError):
        raise
    except Exception as e:
        logger.error(
            "google_play_request_failed",
            extra={
                "action": "google_play_request_failed",
                "product_id": product_id,
                "error": str(e),
            },
        )
        raise GooglePlayVerificationError("Google Play request failed") from e
