# API (Mobile Integration)

## Endpoint

`POST /detect`

- **Content-Type**: `multipart/form-data`
- **Body**: one required file + optional capture-time fields + optional JSON sidecar.

## Request (multipart/form-data)

### Required
- **`file`**: the media file (image or video)

### Optional (recommended)
- **`trusted_metadata`**: JSON string with device/capture metadata (sidecar)
  - Example keys you may send (all optional): `Make`, `Model`, `Software`, `DateTime`, `width`, `height`, `fileSize`, `LensModel`, …

### Optional (capture-time provenance fields)
These are merged into `trusted_metadata` server-side.

- **`captured_in_app`**: boolean (default `false`)
- **`capture_session_id`**: string
- **`capture_timestamp_ms`**: integer (milliseconds since epoch)
- **`capture_path`**: string (your internal path/identifier, not a device filesystem path)
- **`capture_signature`**: string (optional HMAC; required if server sets `CAPTURE_HMAC_SECRET`)

#### Signature (optional hardening)
If the server environment variable `CAPTURE_HMAC_SECRET` is set, the API will **only trust**
`captured_in_app=true` when:

- `capture_session_id` is present
- `capture_timestamp_ms` is present
- `capture_signature` matches:

`HMAC_SHA256(secret, "{session_id}|{timestamp_ms}|{path}")` (hex digest)

## Response (JSON)

### Base fields
- **`summary`**: human-readable summary
- **`confidence_score`**: `0.0..1.0` (capped at `0.99` unless cryptographically verified)
- **`gpu_bypassed`**: boolean
- **`gpu_time_ms`**: float milliseconds (0 when bypassed)

### Layers
- **`layers.layer1_metadata`**
  - `status`: `"verified_ai" | "verified_original" | "likely_original" | "not_found" | ...`
  - `provider`: string or null
  - `description`: string
  - *(optional)* `human_score`, `ai_score`, `signals`
- **`layers.layer2_forensics`**
  - `status`: `"skipped" | "detected" | "not_detected" | "suspicious" | "error"`
  - `probability`: `0.0..1.0`
  - `signals`: list of strings explaining key evidence

### Metadata summary (added for mobile debugging)
- **`metadata`** (optional object)
  - `human_score`: float
  - `ai_score`: float
  - `signals`: list of key signals (e.g., `"Standard/Professional ICC color profile"`, `"Lens model: ..."`, …)
  - `extracted`: object with a few extracted fields (Make/Model/Software/etc)
  - `bypass_reason`: string when GPU was bypassed (e.g. `captured_in_app`, `metadata_verified_original`, `metadata_likely_ai`)

## Example curl

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@/path/to/image.jpg" \
  -F 'trusted_metadata={"Make":"Apple","Model":"iPhone 15 Pro","width":3024,"height":4032,"fileSize":1234567}' \
  -F "captured_in_app=true" \
  -F "capture_session_id=abc123" \
  -F "capture_timestamp_ms=1730000000000" \
  -F "capture_path=session/abc123/frame/0001"
```


