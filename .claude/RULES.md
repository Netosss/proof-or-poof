# Backend-Python — Coding Rules

> These rules are specific to `backend-python/`. They supplement the root CLAUDE.md and `~/.claude/rules/`.

---

## Non-Negotiables (must never violate)

### 1. No `print()` in production code
Use structured logging instead:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("scan_completed", extra={"user_id": uid, "score": score})
```
`print()` bypasses the JSON logger and breaks log aggregation in Railway/Axiom.

### 2. Async I/O everywhere
All file reads, network calls, and DB operations must use `async def` + `await`.
Never call sync Firestore, sync aiohttp, or `open()` on the hot path.
Use `asyncio.to_thread()` for unavoidable blocking I/O (e.g., PIL, ffprobe).

### 3. Credits via `credit_engine` only
Never mutate `credits_balance` in Firestore directly. Always:
```python
from app.services.credit_engine import consume_credits, grant_credits

await consume_credits(user_id, cost=10, reason="detect", reference_id=filename)
await grant_credits(user_id, amount=20, reason="ad_reward", reference_id=doc_id)
```
Direct writes skip the immutable ledger and break accounting.

### 4. All URL downloads via `detection_service.download_media_to_disk()`
Never do raw `aiohttp.get(url)` or `requests.get(url)` on user-supplied URLs.
The service has SSRF protection (DNS validation, private IP blocking, scheme check).

### 5. All uploads through `file_validator.py` before processing
```python
from app.core.file_validator import validate_upload
await validate_upload(file)
```
Magic-byte validation is the real security gate — MIME/extension alone are spoofable.

### 6. Secrets from `settings` (config.py) only
```python
# CORRECT
from app.config import settings
key = settings.gemini_api_key

# WRONG
import os
key = os.environ["GEMINI_API_KEY"]
```
`settings` is validated at startup and provides type-safe access.

---

## Architecture Patterns

### 7. New endpoint checklist
Every new API endpoint must, in order:
1. Validate Turnstile token (if user-facing) via `core/auth.py`
2. Authenticate user via `get_current_user` or `get_optional_user`
3. Check rate limit via `rate_limiter`
4. Validate file/input
5. Deduct credits (before processing — prevents free rides on 500 errors)
6. Process
7. Return response with proper status code

### 8. CORS headers on all exception handlers
Custom exception handlers MUST include CORS headers. Browsers reject error responses without them:
```python
response = JSONResponse(status_code=xxx, content={"detail": msg})
response.headers["Access-Control-Allow-Origin"] = settings.allowed_origin
return response
```

### 9. Redis key namespaces (do not invent new ones)
Follow the established namespace from `CONTEXT.md`:
- Detection cache: `forensic:{hash}`
- Rate limit: `rate_limit:{identifier}`
- Guest wallet: `guest:{device_id}:balance`
- Report: `report:{short_id}`
- Op-ref token: `op_ref:{user_id}:{token}`
- IP devices: `ip_devices:{ip}`

### 10. Gemini calls via `integrations/gemini/client.py` only
Never call `genai` directly in route handlers or services. Retry logic, quality context, and noise hints live in the client module.

### 11. Thread pool expansion is already done in lifespan
Do NOT call `loop.set_default_executor()` elsewhere. It was already expanded to 30 threads in `main.py` lifespan to support parallel Gemini calls.

---

## Testing Rules

### 12. Test file placement
All tests go in `tests/`. File naming: `test_{module_name}.py`.

### 13. Use `pytest-asyncio` for async tests
```python
import pytest

@pytest.mark.asyncio
async def test_consume_credits_deducts_balance():
    ...
```

### 14. Mocks for external services
Mock Firebase, Redis, Gemini, and Modal in unit tests. Do NOT hit live services in unit tests.
Use real services only in dedicated integration test files (prefixed `test_integration_`).

### 15. Minimum coverage for new services
Any new service file must have a corresponding `tests/test_{service}.py` with at least:
- Happy path test
- Insufficient credits / permission error test
- Invalid input test

---

## Common Mistakes (learned the hard way)

- **CORS headers missing on errors** → browser shows "Network Error" instead of the real error message
- **Sync Firestore client in async context** → silently deadlocks under load
- **Using `trusted_metadata` without validation** → mobile client can spoof EXIF scores
- **Missing `reference_id` in credit transactions** → can't audit which file triggered the deduction
- **ffprobe called without `asyncio.to_thread()`** → blocks the event loop during video processing
- **PIL without pixel limit** → DecompressionBomb on adversarial images (use `settings.pil_max_image_pixels`)
