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

    def check_rate_limit(self, identifier: str) -> None:
        check_rate_limit(identifier)

    def validate_file(self, filename: str, filesize: int, file_path: str = None) -> bool:
        return validate_file(filename, filesize, file_path)

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
        self.check_rate_limit(identifier)
        self.validate_file(filename, filesize, temp_path)

        try:
            start_time = time.time()
            result = await func(temp_path, *args, **kwargs)
            duration = time.time() - start_time
            safe_msg = self.sanitize_log_message(f"Successfully processed {filename} in {duration:.2f}s")
            logger.info(safe_msg)
            return result
        except Exception as e:
            err_msg = self.sanitize_log_message(f"Error processing {filename}: {str(e)}")
            logger.error(err_msg)
            raise HTTPException(status_code=500, detail="Internal processing error.")


security_manager = SecurityManager()
