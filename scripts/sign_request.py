"""
Helper: sign and POST a request to /v1/analyze.

Computes the HMAC, builds the multipart body locally, and either prints the
equivalent curl command (--dry-run) or executes it via httpx so the partner
can see exactly what to replicate.

Usage:
    python scripts/sign_request.py \\
        --api-key fxl_test_xxx --secret fxs_test_xxx \\
        --file path/to/image.jpg \\
        [--host http://localhost:8000] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import time
import uuid
from pathlib import Path


BOUNDARY = "----FauxLensSmokeTest"


def build_multipart_body(file_path: Path, field_name: str = "file") -> tuple[bytes, str]:
    """Build a minimal multipart/form-data body. Returns (bytes, content_type)."""
    file_bytes = file_path.read_bytes()
    # Guess MIME from extension — same allowlist as file_validator.
    ext = file_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
        ".heic": "image/heic", ".heif": "image/heif",
        ".avif": "image/avif", ".bmp": "image/bmp", ".tiff": "image/tiff",
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    }.get(ext, "application/octet-stream")

    parts = []
    parts.append(f"--{BOUNDARY}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{file_path.name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(file_bytes)
    parts.append(f"\r\n--{BOUNDARY}--\r\n".encode())
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={BOUNDARY}"


def sign(api_key: str, secret: str, method: str, path: str,
         body: bytes) -> dict[str, str]:
    """Compute all four FauxLens auth headers."""
    timestamp = str(int(time.time()))
    content_sha = hashlib.sha256(body).hexdigest()
    payload = f"{timestamp}\n{method.upper()}\n{path}\n{content_sha}".encode("utf-8")
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return {
        "X-FauxLens-Key": api_key,
        "X-FauxLens-Timestamp": timestamp,
        "X-FauxLens-Content-SHA256": content_sha,
        "X-FauxLens-Signature": signature,
        "Idempotency-Key": str(uuid.uuid4()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--secret", required=True)
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--path", default="/v1/analyze")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the curl command instead of executing")
    args = ap.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 1

    body, content_type = build_multipart_body(args.file)
    headers = sign(args.api_key, args.secret, "POST", args.path, body)
    headers["Content-Type"] = content_type

    print("=== Request preview ===")
    print(f"  URL: {args.host}{args.path}")
    print(f"  body size: {len(body):,} bytes")
    print(f"  content_sha256: {headers['X-FauxLens-Content-SHA256']}")
    print(f"  timestamp: {headers['X-FauxLens-Timestamp']}")
    print(f"  signature: {headers['X-FauxLens-Signature'][:20]}...")
    print(f"  idempotency_key: {headers['Idempotency-Key']}")
    print()

    if args.dry_run:
        print("=== Equivalent curl ===")
        header_args = " \\\n    ".join(f"-H '{k}: {v}'" for k, v in headers.items())
        print(f"curl -X POST '{args.host}{args.path}' \\\n    {header_args} \\\n    "
              f"--data-binary @{args.file}")
        return 0

    import httpx

    print("=== Sending ===")
    t0 = time.time()
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{args.host}{args.path}", content=body, headers=headers)
    dt = round((time.time() - t0) * 1000, 1)

    print(f"  HTTP {r.status_code}  ({dt} ms)")
    print(f"  X-RateLimit-Limit:     {r.headers.get('X-RateLimit-Limit')}")
    print(f"  X-RateLimit-Remaining: {r.headers.get('X-RateLimit-Remaining')}")
    print(f"  X-RateLimit-Reset:     {r.headers.get('X-RateLimit-Reset')}")
    if r.headers.get("X-Idempotent-Replay"):
        print(f"  X-Idempotent-Replay:   {r.headers.get('X-Idempotent-Replay')}")
    print()

    print("=== Response body ===")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text[:2000])

    return 0 if r.status_code < 400 else 2


if __name__ == "__main__":
    sys.exit(main())
