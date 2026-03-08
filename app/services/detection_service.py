"""
Detection request helpers: URL/base64 image downloading, short ID generation,
and memory usage logging.

Security: download_image() defends against SSRF by:
  1. Accepting only http/https schemes.
  2. Resolving the hostname via async DNS and rejecting private / reserved /
     loopback / link-local IP addresses (covers AWS/GCP metadata endpoints,
     localhost, RFC-1918 ranges, etc.).
  3. Streaming the response body in chunks rather than buffering the entire
     response before checking the size — prevents memory exhaustion if the
     remote server starts streaming a huge body.
"""

import asyncio
import base64
import ipaddress
import logging
import os
import secrets
import socket
import string
from urllib.parse import urlparse

import aiohttp
import psutil
from fastapi import HTTPException

from app.config import settings
from app.integrations import http_client as http_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSRF: private / reserved IP ranges that must never be fetched
# ---------------------------------------------------------------------------
_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),        # loopback
    ipaddress.ip_network("169.254.0.0/16"),      # link-local / AWS metadata
    ipaddress.ip_network("100.64.0.0/10"),       # Carrier-grade NAT (Railway internal)
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),             # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),            # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),           # IPv6 link-local
)


def _is_blocked_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or any(ip in net for net in _BLOCKED_NETWORKS)
        )
    except ValueError:
        return True  # fail closed on unparseable addresses


async def _assert_url_safe(url: str) -> None:
    """
    Raise HTTPException(400) if the URL is not safe to fetch:
      - scheme must be http or https
      - resolved hostname must not be a private / internal IP

    DNS resolution is done in a thread-pool executor (asyncio-safe).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http/https URLs are supported")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: missing hostname")

    try:
        loop = asyncio.get_running_loop()
        # getaddrinfo returns all address families / addresses for the host.
        addr_infos = await loop.getaddrinfo(hostname, None)
    except (OSError, socket.gaierror):
        raise HTTPException(status_code=400, detail="Could not resolve URL hostname")

    for addr_info in addr_infos:
        ip_str = addr_info[4][0]
        if _is_blocked_ip(ip_str):
            logger.warning("ssrf_attempt_blocked", extra={
                "action": "ssrf_attempt_blocked",
                "hostname": hostname,
                "resolved_ip": ip_str,
            })
            raise HTTPException(status_code=400, detail="URL points to a disallowed address")


def _suffix_from_content_type(content_type: str, url: str) -> str:
    """Best-effort file extension from Content-Type header or URL path."""
    ct = content_type.lower()
    if "png" in ct:        return ".png"
    if "jpeg" in ct or "jpg" in ct:  return ".jpg"
    if "webp" in ct:       return ".webp"
    if "gif" in ct:        return ".gif"
    if "heic" in ct:       return ".heic"
    if "heif" in ct:       return ".heif"
    if "tiff" in ct:       return ".tiff"
    if "bmp" in ct:        return ".bmp"
    if "mp4" in ct:        return ".mp4"
    if "quicktime" in ct or "mov" in ct: return ".mov"
    if "webm" in ct:       return ".webm"

    # Fall back to URL path extension when Content-Type is generic.
    if not ct or "application" in ct or "octet-stream" in ct:
        lower_url = url.lower().split("?")[0]  # strip query string
        for ext in (".png", ".webp", ".gif", ".heic", ".heif",
                    ".tiff", ".tif", ".bmp", ".mp4", ".mov", ".webm"):
            if lower_url.endswith(ext):
                return ext

    return ".jpg"


def _generate_short_id(length: int = settings.short_id_length) -> str:
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def log_memory(stage: str) -> None:
    """Log current process and system memory usage. Only runs when DEBUG logging is active."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()
    logger.debug("memory_usage", extra={
        "action": "memory_usage",
        "stage": stage,
        "process_rss_mb": round(mem_info.rss / 1024 / 1024, 2),
        "system_available_mb": round(sys_mem.available / 1024 / 1024, 2),
        "system_total_mb": round(sys_mem.total / 1024 / 1024, 2),
    })


async def download_image(url: str, max_size: int = settings.max_image_download_bytes) -> tuple[bytes, str]:
    """
    Downloads a media file from a URL or decodes a base64 data URI.

    Security guarantees:
      - Data URIs: size is capped before base64 decoding to prevent memory spikes.
      - HTTP URLs: SSRF-protected (see _assert_url_safe); body is streamed in
        chunks so the size limit fires before the full payload is buffered.
    """
    if url.startswith("data:"):
        try:
            header, data_str = url.split(",", 1)
            if ";base64" not in header:
                raise HTTPException(status_code=400, detail="Only base64 data URIs are supported")

            # Pre-check: base64 expands by ~4/3 — cap raw string length too.
            max_b64_len = (max_size * 4 // 3) + 64
            if len(data_str) > max_b64_len:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large (max {max_size // (1024 * 1024)} MB)",
                )

            content = base64.b64decode(data_str)
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large (max {max_size // (1024 * 1024)} MB)",
                )

            mime_type = header.split(":")[1].split(";")[0]
            suffix = _suffix_from_content_type(mime_type, "")
            return content, f"pasted_image{suffix}"
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("url_decode_failed", extra={"action": "url_decode_failed", "error": str(exc)})
            raise HTTPException(status_code=400, detail="Invalid data URI")

    # --- HTTP / HTTPS URL ---

    # SSRF guard: validate scheme and ensure hostname resolves to a public IP.
    await _assert_url_safe(url)

    async with http_module.request_session() as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch image from URL: Status {response.status}",
                    )

                # Stream in chunks — do NOT call response.read() which buffers
                # the entire body before we can check the size.
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(65_536):
                    total += len(chunk)
                    if total > max_size:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image too large (max {max_size // (1024 * 1024)} MB)",
                        )
                    chunks.append(chunk)
                content = b"".join(chunks)

                content_type = response.headers.get("Content-Type", "")
                suffix = _suffix_from_content_type(content_type, url)
                return content, f"downloaded_media{suffix}"

        except HTTPException:
            raise
        except aiohttp.ClientError as exc:
            raise HTTPException(status_code=400, detail=f"Error fetching image: {exc}")
