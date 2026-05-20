"""
Enterprise S2S detection endpoint: POST /v1/analyze

Pipeline (mirrors consumer /detect, replaces Turnstile with HMAC + IP allowlist):
    1. HMAC authentication (headers only — body untouched)
    2. Rate limit (per-credential)
    3. Idempotency check (header required; cached replay short-circuits work)
    4. Stream body to disk (no Turnstile, no device_id)
    5. Re-verify body SHA-256 against header digest
    6. File validation (existing 3-layer validator)
    7. Reserve credits (atomic — race-free)
    8. Run detect_ai_media — reuses the exact consumer pipeline
    9. On synthetic-failure verdicts OR exceptions → refund credits (idempotent)
   10. Cache idempotent response, return JSON envelope with rate-limit headers
"""

import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.enterprise_auth import authenticate
from app.core.enterprise_rate_limiter import check_and_track, headers_for
from app.core.file_validator import ALLOWED_VIDEO_EXTENSIONS, validate_file
from app.core.idempotency import claim_or_replay, get_idempotency_key, store
from app.detection.pipeline import detect_ai_media
from app.logging_config import user_id_var
from app.services.detection_service import _generate_short_id, download_media_to_disk
from app.services.enterprise_credit_engine import refund_credit, reserve_credit
from app.services.finance_service import log_transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Enterprise"])

_UPLOAD_CHUNK = 65_536


async def _ingest_body_to_disk(request: Request, dest_path: str, max_bytes: int) -> tuple[str, str]:
    """
    Stream the raw request body to `dest_path` while computing SHA-256.
    Returns (sha256_hex, content_type_from_header).
    Raises HTTPException(413) on size cap.
    """
    h = hashlib.sha256()
    total = 0
    with open(dest_path, "wb") as fp:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={"type": "request_too_large_error", "code": "payload_too_large",
                            "message": f"Body exceeds {max_bytes // (1024 * 1024)} MB limit."},
                )
            h.update(chunk)
            fp.write(chunk)
    return h.hexdigest(), request.headers.get("content-type", "")


def _parse_multipart_disk(temp_path: str, content_type: str) -> tuple[str, str | None, str]:
    """
    Single-part multipart parser for enterprise uploads.
    Expected: one `file` part containing the image/video bytes.

    Uses stdlib email.parser (Python 3.11+ stable; `cgi` was removed in 3.13).
    For our single-part shape this is faster, dependency-free, and fully
    compatible with all common HTTP multipart clients.

    Returns (filename, content_type_of_file_part, extracted_file_path).
    """
    from email.parser import BytesParser
    from email.policy import default as default_policy

    # Prepend a synthetic Content-Type header so email.parser treats the body
    # as a multipart/form-data message.
    with open(temp_path, "rb") as fp:
        header_blob = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        msg_bytes = header_blob + fp.read()

    msg = BytesParser(policy=default_policy).parsebytes(msg_bytes)

    if not msg.is_multipart():
        raise HTTPException(
            status_code=400,
            detail={"type": "invalid_request_error", "code": "invalid_multipart",
                    "message": "Body is not a valid multipart message."},
        )

    for part in msg.iter_parts():
        cd = part.get("Content-Disposition", "")
        # Match the form field named exactly "file" (with or without quotes).
        if 'name="file"' in cd or "name=file" in cd:
            filename = part.get_filename() or "upload.bin"
            part_content_type = part.get_content_type()
            payload = part.get_payload(decode=True)
            if payload is None:
                raise HTTPException(
                    status_code=400,
                    detail={"type": "invalid_request_error", "code": "empty_file",
                            "message": "Multipart `file` part is empty."},
                )
            extracted_path = temp_path + ".extracted"
            with open(extracted_path, "wb") as out:
                out.write(payload)
            return filename, part_content_type, extracted_path

    raise HTTPException(
        status_code=400,
        detail={"type": "invalid_request_error", "code": "missing_file",
                "message": "Multipart body must include a `file` field."},
    )


@router.post("/analyze")
async def analyze(request: Request):
    """Enterprise S2S detection endpoint. See module docstring for the pipeline."""

    # --- 1. Authenticate (headers only — body untouched here) ---
    principal = await authenticate(request)

    user_id_var.set(f"ent:{principal.partner_id}")

    # --- 2. Rate limit ---
    rate_info = await check_and_track(principal.credential_id)
    rate_headers = headers_for(rate_info)

    # --- 3. Idempotency ---
    idem_key = get_idempotency_key(request.headers)
    replay = await claim_or_replay(principal.credential_id, idem_key)
    if replay:
        replay_headers = dict(rate_headers)
        replay_headers["X-Idempotent-Replay"] = "true"
        return JSONResponse(
            status_code=replay.status_code,
            content=replay.body,
            headers=replay_headers,
        )

    fd, body_path = tempfile.mkstemp(suffix=".body")
    os.close(fd)
    extracted_path: str | None = None
    reserved = False
    new_balance: int | None = None
    # Always have a usable request_id for ledger reference + logs even when the
    # caller omits X-Request-ID. Using uuid4 avoids the temp-file-basename
    # fallback that produced meaningless audit entries and could (in theory)
    # collide with a prior refund's reference_id.
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4()}"

    try:
        # --- 4. Stream body to disk while hashing ---
        body_sha, content_type = await _ingest_body_to_disk(
            request, body_path, settings.enterprise_max_request_bytes,
        )

        # --- 5. Body hash matches header digest? Use the streaming-time digest
        #        directly — the file has already been written + hashed once, no
        #        need to re-read 200 MB from disk for a second hash.
        expected_sha = (principal.content_sha256 or "").lower()
        if not hmac.compare_digest(body_sha, expected_sha):
            raise HTTPException(
                status_code=400,
                detail={"type": "invalid_request_error", "code": "content_hash_mismatch",
                        "message": f"Body SHA-256 differs from {expected_sha[:12]}... "
                                   "in X-FauxLens-Content-SHA256 header."},
            )

        # --- 6. Determine input mode ---
        ct = content_type.lower()
        filename = "upload.bin"
        upload_content_type: str | None = None

        if "application/json" in ct:
            try:
                with open(body_path, "rb") as fp:
                    payload = json.loads(fp.read().decode("utf-8") or "{}")
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail={"type": "invalid_request_error", "code": "invalid_json",
                            "message": "Body could not be parsed as JSON."},
                )
            url = payload.get("url")
            if not url:
                raise HTTPException(
                    status_code=400,
                    detail={"type": "invalid_request_error", "code": "missing_url",
                            "message": "JSON body must include `url`."},
                )
            fd2, extracted_path = tempfile.mkstemp(suffix=".tmp")
            os.close(fd2)
            filename = await download_media_to_disk(url, extracted_path)
        elif "multipart/form-data" in ct:
            filename, upload_content_type, extracted_path = _parse_multipart_disk(body_path, content_type)
        else:
            raise HTTPException(
                status_code=415,
                detail={"type": "unsupported_media_type_error", "code": "unsupported_media_type",
                        "message": "Use application/json or multipart/form-data."},
            )

        suffix = os.path.splitext(filename)[1].lower() or ".jpg"
        proper_path = os.path.splitext(extracted_path)[0] + suffix
        if proper_path != extracted_path:
            os.rename(extracted_path, proper_path)
            extracted_path = proper_path

        filesize = os.path.getsize(extracted_path)

        # --- 7. File validation (3-layer — MIME, ext, magic bytes) ---
        await validate_file(filename, filesize, extracted_path, upload_content_type)

        # --- 8. Reserve credit BEFORE running the model ---
        cost = settings.enterprise_credit_cost
        new_balance = await reserve_credit(
            principal.partner_id, cost=cost, reason="api_scan_deduction",
            reference_id=request_id,
        )
        reserved = True

        # --- 9. Run pipeline ---
        start = time.time()
        try:
            result = await detect_ai_media(extracted_path)
        except Exception as e:
            logger.error(
                "enterprise_pipeline_crashed",
                extra={
                    "action": "enterprise_pipeline_crashed",
                    "partner_id": principal.partner_id,
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            # Refund — idempotent ledger check prevents double-refund on retry.
            await refund_credit(
                principal.partner_id, amount=cost, reason="api_refund:pipeline_crash",
                reference_id=request_id,
            )
            reserved = False
            raise HTTPException(
                status_code=500,
                detail={"type": "api_error", "code": "pipeline_error",
                        "message": "Detection pipeline failed; credit refunded."},
            )

        duration = time.time() - start

        # Failed verdicts: refund and surface a clean error to the partner.
        if result.get("summary") in ("Analysis Failed", "File too large to scan"):
            await refund_credit(
                principal.partner_id, amount=cost,
                reason=f"api_refund:{result.get('summary', '').lower().replace(' ', '_')}",
                reference_id=request_id,
            )
            reserved = False
            raise HTTPException(
                status_code=422,
                detail={"type": "invalid_request_error", "code": "analysis_failed",
                        "message": result.get("summary", "Analysis failed; credit refunded.")},
            )

        # --- 10. Build response, cache idempotent result, log ---
        is_gemini_used = result.get("is_gemini_used", False)
        is_cached = result.get("is_cached", False)
        actual_gpu_time_ms = result.get("gpu_time_ms", 0.0)
        actual_gpu_sec = actual_gpu_time_ms / 1000.0

        if is_cached:
            cost_usd = 0.0
            log_transaction("CACHE", 0.0,
                            {"file": filename, "partner_id": principal.partner_id, "request_id": request_id})
        elif is_gemini_used:
            cost_usd = settings.gemini_fixed_cost
            log_transaction("GEMINI", -cost_usd,
                            {"file": filename, "partner_id": principal.partner_id, "request_id": request_id})
        elif actual_gpu_sec > 0:
            cost_usd = actual_gpu_sec * settings.gpu_rate_per_sec
            log_transaction("GPU", -cost_usd,
                            {"file": filename, "partner_id": principal.partner_id,
                             "request_id": request_id, "duration": actual_gpu_sec})
        else:
            cost_usd = duration * settings.cpu_rate_per_sec
            log_transaction("CPU", -cost_usd,
                            {"file": filename, "partner_id": principal.partner_id,
                             "request_id": request_id, "duration": duration})

        result.pop("gpu_time_ms", None)
        result.pop("is_gemini_used", None)
        result.pop("is_cached", None)

        short_id = _generate_short_id()
        result["short_id"] = short_id
        # NOTE: We intentionally do NOT publish to the report cache here — enterprise
        # scans should not surface in the consumer share-link surface unless the
        # partner explicitly opts in.

        body_envelope = {
            "data": result,
            "request_id": request_id,
            "credits_remaining": int(new_balance),
            "idempotent_replay": False,
        }

        # Cache for replay.
        await store(principal.credential_id, idem_key, status_code=200, body=body_envelope)

        logger.info(
            "enterprise_scan_completed",
            extra={
                "action": "enterprise_scan_completed",
                "partner_id": principal.partner_id,
                "credential_id": principal.credential_id,
                "request_id": request_id,
                "outcome": result.get("summary"),
                "confidence_score": result.get("confidence_score"),
                "media_type": "video" if suffix in ALLOWED_VIDEO_EXTENSIONS else "image",
                "duration_ms": round(duration * 1000, 1),
                "cost_usd": cost_usd,
                "credits_consumed": cost,
            },
        )

        return JSONResponse(status_code=200, content=body_envelope, headers=rate_headers)

    except HTTPException as exc:
        # If we reserved credits and we're about to fail with a non-quota error
        # that didn't already refund, refund here as a safety net.
        if reserved:
            try:
                await refund_credit(
                    principal.partner_id, amount=settings.enterprise_credit_cost,
                    reason="api_refund:request_aborted", reference_id=request_id,
                )
            except Exception as ref_err:
                # Loud failure — partner is owed credits and the cleanup refund
                # failed. Operator MUST see this in Axiom/Sentry to recover manually.
                logger.error(
                    "enterprise_refund_cleanup_failed",
                    extra={
                        "action": "enterprise_refund_cleanup_failed",
                        "partner_id": principal.partner_id,
                        "request_id": request_id,
                        "amount": settings.enterprise_credit_cost,
                        "original_status": exc.status_code,
                        "error": str(ref_err),
                        "error_type": type(ref_err).__name__,
                    },
                    exc_info=True,
                )
        # Forward HTTPException with rate-limit headers attached so partner SDKs
        # see consistent telemetry on every response.
        merged = dict(getattr(exc, "headers", None) or {})
        for k, v in rate_headers.items():
            merged.setdefault(k, v)
        exc.headers = merged
        raise
    except Exception as e:
        # Truly unexpected — refund if needed, log, raise generic 500.
        if reserved:
            try:
                await refund_credit(
                    principal.partner_id, amount=settings.enterprise_credit_cost,
                    reason="api_refund:unhandled_error", reference_id=request_id,
                )
            except Exception as ref_err:
                logger.error(
                    "enterprise_refund_cleanup_failed",
                    extra={
                        "action": "enterprise_refund_cleanup_failed",
                        "partner_id": principal.partner_id,
                        "request_id": request_id,
                        "amount": settings.enterprise_credit_cost,
                        "original_error_type": type(e).__name__,
                        "error": str(ref_err),
                        "error_type": type(ref_err).__name__,
                    },
                    exc_info=True,
                )
        logger.error(
            "enterprise_analyze_unhandled_error",
            extra={
                "action": "enterprise_analyze_unhandled_error",
                "partner_id": principal.partner_id,
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={"type": "api_error", "code": "internal_error",
                    "message": "Unexpected server error."},
            headers=rate_headers,
        )
    finally:
        for p in (body_path, extracted_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
