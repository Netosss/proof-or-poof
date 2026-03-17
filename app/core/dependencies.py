"""
SecurityManager — central orchestrator for request-level security.

Delegates specific concerns to specialized modules:
  - Rate limiting  → core/rate_limiter.py
  - File validation → core/file_validator.py

`security_manager` is a module-level singleton for use in route handlers.
"""

import hashlib
import logging
import time
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request

from app.core.rate_limiter import check_rate_limit
from app.core.file_validator import validate_file, sanitize_log_message

logger = logging.getLogger(__name__)


class SecurityManager:
    """Orchestrates rate limiting, file validation, and secure execution."""

    async def check_rate_limit(self, identifier: str) -> None:
        await check_rate_limit(identifier)

    async def validate_file(
        self,
        filename: str,
        filesize: int,
        file_path: str | None = None,
        content_type: str | None = None,
        *,
        mode: str = "detect",
    ) -> bool:
        return await validate_file(filename, filesize, file_path, content_type, mode=mode)

    def sanitize_log_message(self, message: str) -> str:
        return sanitize_log_message(message)

    def get_safe_hash(self, data: bytes) -> str:
        """Securely hash data for caching to prevent collisions/poisoning."""
        return hashlib.sha256(data).hexdigest()

    async def secure_execute(
        self,
        request: Request,
        filename: str,
        filesize: int,
        temp_path: str,
        func: Callable,
        uid: Optional[str] = None,
        *args,
        **kwargs
    ) -> Any:
        """Rate-limits, validates, executes, and sanitizes logs for a media processing call."""
        identifier = uid or request.client.host
        await self.check_rate_limit(identifier)
        await self.validate_file(filename, filesize, temp_path)

        try:
            start_time = time.time()
            result = await func(temp_path, *args, **kwargs)
            duration = time.time() - start_time
            logger.info("media_processed_success", extra={
                "action": "media_processed_success",
                "duration_ms": round(duration * 1000, 1),
                "media_file": filename,
                "filesize_bytes": filesize,
            })
            return result
        except Exception as e:
            logger.error("media_processed_error", extra={
                "action": "media_processed_error",
                "media_file": filename,
                "filesize_bytes": filesize,
                "error": self.sanitize_log_message(str(e)),
                "error_type": type(e).__name__,
            }, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal processing error.")


security_manager = SecurityManager()
