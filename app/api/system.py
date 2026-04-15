"""
System / health routes.
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter(tags=["System"])


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nDisallow: /"


@router.get("/.well-known/assetlinks.json")
async def assetlinks():
    """Android App Links verification — required for report URL deeplinks."""
    sha256 = os.getenv("ANDROID_SHA256_FINGERPRINT")
    if not sha256:
        raise HTTPException(status_code=404, detail="Not configured")
    data = [
        {
            "relation": ["delegate_permission/common.handle_all_urls"],
            "target": {
                "namespace": "android_app",
                "package_name": "com.fauxlens.android",
                "sha256_cert_fingerprints": [sha256],
            },
        }
    ]
    return JSONResponse(content=data)
