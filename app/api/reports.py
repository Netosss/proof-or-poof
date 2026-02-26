"""
Report sharing routes: create shareable link and fetch a shared report.
"""

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, BackgroundTasks

from app.config import settings
from app.schemas.reports import ShareRequest, ShareResponse
from app.services.reports_service import create_share_link, extend_report_ttl, get_shared_report

router = APIRouter(tags=["Reports"])


@router.post("/api/v1/reports/share", response_model=ShareResponse, status_code=201)
async def create_share_link_route(request: ShareRequest):
    """
    Publishes a cached scan result as a permanent shared report.
    Idempotent: re-publishing the same short_id returns the existing report_id.
    """
    return create_share_link(request.short_id)


@router.get("/api/v1/reports/share/{report_id}")
async def get_shared_report_route(report_id: str, background_tasks: BackgroundTasks):
    """
    Fetches a public shared report. No auth required.
    Auto-extends TTL by report_extend_days if fewer than report_extend_threshold_days remain.
    """
    data, should_extend = get_shared_report(report_id)

    if should_extend:
        new_expiry = datetime.now(timezone.utc) + timedelta(days=settings.report_extend_days)
        background_tasks.add_task(extend_report_ttl, report_id, new_expiry)

    return data
