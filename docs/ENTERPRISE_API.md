# Faux Lens Enterprise API

Server-to-server (S2S) AI-detection API for enterprise partners. HMAC-signed,
prepaid-credit, no browser CAPTCHA required.

**Base URL:** `https://web-production-6a994.up.railway.app` (production)
**Stable path:** `POST /v1/analyze`

---

## 1. Authentication

Every request must include four headers:

| Header | Value |
|---|---|
| `X-FauxLens-Key` | Public API key (`fxl_live_...` or `fxl_test_...`) |
| `X-FauxLens-Timestamp` | UNIX epoch seconds when the request was signed |
| `X-FauxLens-Content-SHA256` | Lowercase hex SHA-256 of the raw request body |
| `X-FauxLens-Signature` | Hex HMAC-SHA256 of the canonical signing payload |

### 1.1 Canonical signing payload

```
{timestamp}\n{method}\n{path_with_query}\n{content_sha256}
```

- `\n` is the literal LF character (0x0A).
- `method` is uppercase (e.g. `POST`).
- `path_with_query` is the URL path AND the query string when present —
  e.g. `/v1/analyze` for a plain POST, `/v1/analyze?mode=fast` when a query
  string is sent. **The query string MUST be included in the signed payload
  whenever it appears on the request.** Omitting it causes `invalid_signature`.
  The leading `?` is part of the signed string; do not URL-encode the
  separator.
- `content_sha256` is the lowercase hex SHA-256 of the raw request body
  (the exact bytes sent on the wire, including multipart framing).

### 1.2 Signature algorithm

```
signature = hex( HMAC_SHA256( secret_key, signing_payload ) )
```

### 1.3 Worked example (Python)

```python
import hashlib
import hmac
import os
import time
import uuid

import requests

API_KEY    = "fxl_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
SECRET_KEY = "fxs_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
URL        = "https://web-production-6a994.up.railway.app/v1/analyze"

with open("photo.jpg", "rb") as fp:
    body = fp.read()

content_sha256 = hashlib.sha256(body).hexdigest()
timestamp = str(int(time.time()))
# Include the query string when present, e.g. "/v1/analyze?mode=fast".
path_with_query = "/v1/analyze"
method = "POST"

payload = f"{timestamp}\n{method}\n{path_with_query}\n{content_sha256}".encode("utf-8")
signature = hmac.new(SECRET_KEY.encode(), payload, hashlib.sha256).hexdigest()

resp = requests.post(
    URL,
    headers={
        "X-FauxLens-Key": API_KEY,
        "X-FauxLens-Timestamp": timestamp,
        "X-FauxLens-Content-SHA256": content_sha256,
        "X-FauxLens-Signature": signature,
        "Idempotency-Key": str(uuid.uuid4()),
    },
    files={"file": ("photo.jpg", body, "image/jpeg")},
    timeout=60,
)
print(resp.status_code, resp.json())
```

### 1.4 Worked example (Node.js)

```js
import crypto from "node:crypto";
import fs from "node:fs";
import { randomUUID } from "node:crypto";

const API_KEY    = process.env.FAUXLENS_API_KEY;
const SECRET_KEY = process.env.FAUXLENS_SECRET_KEY;
const URL        = "https://web-production-6a994.up.railway.app/v1/analyze";

const body = fs.readFileSync("photo.jpg");
const contentSha = crypto.createHash("sha256").update(body).digest("hex");
const ts = Math.floor(Date.now() / 1000).toString();
// Include the query string when present, e.g. "/v1/analyze?mode=fast".
const pathWithQuery = "/v1/analyze";
const payload = `${ts}\nPOST\n${pathWithQuery}\n${contentSha}`;
const signature = crypto.createHmac("sha256", SECRET_KEY).update(payload).digest("hex");

const form = new FormData();
form.append("file", new Blob([body], { type: "image/jpeg" }), "photo.jpg");

const res = await fetch(URL, {
  method: "POST",
  headers: {
    "X-FauxLens-Key": API_KEY,
    "X-FauxLens-Timestamp": ts,
    "X-FauxLens-Content-SHA256": contentSha,
    "X-FauxLens-Signature": signature,
    "Idempotency-Key": randomUUID(),
  },
  body: form,
});
console.log(res.status, await res.json());
```

### 1.5 Replay protection

- Requests with a timestamp drift greater than **300 seconds** are rejected
  with `401 timestamp_expired`. The error message includes the server's
  current UNIX time so you can self-diagnose clock skew without operator
  support. Synchronize your hosts via NTP.
- Each signature is single-use within the timestamp window. Re-sending the
  exact same request yields `401 replay_detected`. The nonce is scoped to
  your credential — a leaked signature cannot be used by another principal
  to pre-claim the nonce slot and deny service to you.

### 1.6 IP allowlist (optional)

When provisioning a credential the operator may attach a list of CIDRs. Any
request whose source IP falls outside the list returns `403 ip_not_allowed`.

The server resolves the partner IP from the rightmost entry of the
`X-Forwarded-For` header (set by the FauxLens edge proxy). If you call from
behind your own corporate proxy or NAT, allowlist the egress IP of that
proxy, not your internal hosts.

---

## 2. Idempotency

Every `POST /v1/analyze` request **MUST** include an `Idempotency-Key` header
with a unique value per logical operation (UUIDv4 recommended).

- Successful responses are cached for **24 hours** keyed on
  `(credential_id, idempotency_key)`.
- Replaying the same key returns the original response with
  `X-Idempotent-Replay: true`. Your account is not charged again.
- Concurrent requests with the same key wait for the original to complete
  (lock TTL: 5 minutes), eliminating double-charging on network retries.

---

## 3. Endpoint: `POST /v1/analyze`

### 3.1 Request

**Content types:** `multipart/form-data` (file upload) or
`application/json` (URL reference).

#### Multipart

```
POST /v1/analyze
Content-Type: multipart/form-data; boundary=...

(form field) file: <binary image or video>
```

#### JSON

```
POST /v1/analyze
Content-Type: application/json

{ "url": "https://example.com/some-image.jpg" }
```

URLs MUST be HTTPS. Private/loopback IPs are blocked (SSRF protection).

### 3.2 Limits

- Image: **≤ 20 MB**
- Video: **≤ 200 MB**
- File extensions: see the consumer API allowlist (JPEG, PNG, WebP, HEIC,
  AVIF, MP4, MOV, WebM, MKV, and many more)

### 3.3 Successful response (200)

```json
{
  "data": {
    "summary": "Likely Authentic",
    "confidence_score": 0.95,
    "is_short_circuited": true,
    "evidence_chain": [
      { "layer": "Metadata Check", "status": "passed",
        "label": "Device Metadata", "detail": "Valid camera metadata found." }
    ],
    "short_id": "Ab3cD4eF"
  },
  "request_id": "req_<uuid>",
  "credits_remaining": 4999,
  "idempotent_replay": false
}
```

### 3.4 Response headers

| Header | Description |
|---|---|
| `X-RateLimit-Limit` | Request budget per minute |
| `X-RateLimit-Remaining` | Requests remaining in the current window |
| `X-RateLimit-Reset` | Unix epoch (seconds) when the window resets |
| `Retry-After` | Sent only on `429`; seconds to wait |
| `X-Idempotent-Replay` | `true` when the response was served from cache |
| `X-Request-ID` | Server-assigned trace id, also echoed in error envelopes |

---

## 4. Error envelope

All non-2xx responses use this shape:

```json
{
  "error": {
    "type": "authentication_error",
    "code": "invalid_signature",
    "message": "HMAC signature did not match canonical payload.",
    "request_id": "req_<uuid>"
  }
}
```

### Error codes

| HTTP | type | code | When |
|---|---|---|---|
| 400 | invalid_request_error | missing_idempotency_key | `Idempotency-Key` header missing |
| 400 | invalid_request_error | content_hash_mismatch | Body SHA-256 differs from header value |
| 400 | invalid_request_error | invalid_json | Bad JSON body |
| 400 | invalid_request_error | missing_url | JSON body missing `url` |
| 400 | invalid_request_error | invalid_multipart | Multipart boundary missing |
| 400 | invalid_request_error | missing_file | Multipart body has no `file` part |
| 401 | authentication_error | missing_auth_headers | One of the four auth headers absent |
| 401 | authentication_error | invalid_timestamp | Timestamp header not an integer |
| 401 | authentication_error | timestamp_expired | Drift > 300s |
| 401 | authentication_error | replay_detected | Signature reused within window |
| 401 | authentication_error | invalid_api_key | Key unknown, revoked, or expired |
| 401 | authentication_error | invalid_signature | HMAC mismatch |
| 402 | payment_required_error | insufficient_credits | Balance < cost |
| 403 | permission_error | ip_not_allowed | Source IP outside CIDR allowlist |
| 403 | permission_error | partner_suspended / partner_frozen | Account paused by operator |
| 404 | not_found_error | partner_not_found | Internal lookup failure |
| 413 | request_too_large_error | payload_too_large | Body > 200 MB |
| 415 | unsupported_media_type_error | unsupported_media_type | MIME / extension / codec rejection |
| 422 | invalid_request_error | analysis_failed | Pipeline could not analyze (credit refunded) |
| 429 | rate_limit_error | rate_limited | Per-minute budget exhausted |
| 500 | api_error | pipeline_error / internal_error | Unexpected failure (credit refunded) |
| 503 | api_error | service_unavailable | Database or downstream service down |

---

## 5. Credits and refunds

- Each successful scan deducts **1 enterprise credit** (configurable per
  deployment via `ENTERPRISE_CREDIT_COST`).
- Credits are deducted **before** the pipeline runs (race-free under
  concurrent load).
- Pipeline failures (`Analysis Failed`, `File too large`, unhandled
  exceptions) trigger an **automatic, idempotent refund**. Your balance is
  not charged for failed scans.
- Top up via Lemon Squeezy. Each purchase appends a `purchase` entry to your
  partner credit ledger.

### 5.1 Operational ledger

Every credit movement (purchase, scan deduction, refund, manual adjustment)
is recorded as an append-only document under:

```
enterprise_partners/{partner_id}/credit_ledger/{auto_id}
```

The operator can export this on request for billing reconciliation.

---

## 6. Rate limits

- Default: **60 requests per minute per credential**.
- Per-partner overrides available on request.
- 429 responses include `Retry-After`. Use exponential backoff with jitter.

---

## 7. Versioning

The API is versioned in the URL (`/v1/`). Breaking changes ship under a new
prefix (`/v2/`). Additive changes (new fields, new error codes) may land in
`/v1/` without notice — write tolerant parsers.
