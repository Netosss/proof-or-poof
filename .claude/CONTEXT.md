# Faux Lens — Python Backend Context

> Read this at the start of every session. It covers everything needed to work on the backend without re-exploring the codebase.

---

## What Is This Repo?

FastAPI server powering all AI detection, object removal, authentication, credits, and payments for Faux Lens. Deployed on Railway.app via Docker. Uses Google Gemini AI for forensic image/video analysis.

**Production URL:** `https://web-production-6a994.up.railway.app`  
**Staging URL:** `https://web-staging-dfc2.up.railway.app`  
**Health Check:** `GET /health` → `{ "status": "healthy" }`

---

## Project Structure

```
backend-python/
├── app/
│   ├── main.py                      # FastAPI app, CORS, lifespan, middleware, exception handlers
│   ├── config.py                    # Pydantic Settings (250+ fields, all env-overridable)
│   ├── logging_config.py            # JSON logging + Sentry setup
│   │
│   ├── api/                         # Route handlers
│   │   ├── auth.py                  # POST /api/auth/me
│   │   ├── detection.py             # POST /detect (main AI endpoint)
│   │   ├── credits.py               # GET /api/user/balance, POST /api/ads/reward, POST /api/credits/add
│   │   ├── checkout.py              # POST /api/checkout/create (Lemon Squeezy)
│   │   ├── inpainting.py            # POST /inpaint/image (object removal via Modal GPU)
│   │   ├── reports.py               # POST /api/v1/reports/share, GET /api/v1/reports/share/{id}
│   │   ├── system.py                # GET /health, /robots.txt, /.well-known/assetlinks.json
│   │   └── webhooks.py              # POST /webhook/runpod, POST /webhooks/lemonsqueezy
│   │
│   ├── core/                        # Authentication + security
│   │   ├── auth.py                  # Turnstile verification, IP device limiting
│   │   ├── firebase_auth.py         # Firebase ID token verification (get_current_user, get_optional_user)
│   │   ├── file_validator.py        # 3-layer file validation (MIME → extension → magic bytes)
│   │   ├── rate_limiter.py          # Redis sliding-window rate limiter
│   │   └── dependencies.py          # SecurityManager orchestrator (combines auth + captcha + rate limits)
│   │
│   ├── detection/                   # AI detection logic
│   │   ├── pipeline.py              # detect_ai_media() — main entry, routes to image/video
│   │   ├── image_detector.py        # Full image detection logic (metadata → Gemini)
│   │   ├── video_detector.py        # Video: ffprobe metadata + tri-frame Gemini batch
│   │   ├── metadata_scorer.py       # EXIF/metadata forensic scoring (human vs AI signals)
│   │   ├── hashing.py               # Smart file hashing (full for small / chunked for large)
│   │   ├── cache.py                 # Redis cache + in-memory LRU fallback
│   │   └── constants.py             # Known AI generator signatures (100+), provenance whitelist
│   │
│   ├── services/                    # Business logic
│   │   ├── credit_engine.py         # Firestore atomic credit transactions (consume/grant)
│   │   ├── credits_service.py       # Guest wallet management (Redis)
│   │   ├── detection_service.py     # URL download, SSRF protection, short ID generation
│   │   ├── reports_service.py       # Share link creation + TTL management
│   │   └── finance_service.py       # Transaction logging (Axiom)
│   │
│   └── integrations/                # External service clients
│       ├── firebase.py              # Firebase Admin + async Firestore client init
│       ├── redis_client.py          # Redis connection pool
│       ├── http_client.py           # aiohttp session management
│       ├── c2pa.py                  # C2PA manifest extraction wrapper
│       ├── runpod.py                # RunPod GPU job dispatch
│       ├── gpu_worker.py            # Modal GPU inpainting dispatch
│       └── gemini/
│           ├── client.py            # Gemini API client (analyze_image_pro_turbo, batch)
│           ├── prompts.py           # Forensic detection system instructions
│           └── quality.py           # Pre-calculated image quality context
│
├── requirements.txt                 # 28 Python dependencies
├── pyproject.toml                   # Ruff, mypy, pytest config
├── Dockerfile                       # Multi-stage, jemalloc, model baking
├── railway.toml                     # Health check + restart policy
└── .claude/
    └── CONTEXT.md                   # This file
```

---

## All API Endpoints

### Auth

```
POST /api/auth/me
  Body: { device_id: string }
  Headers: Authorization: Bearer <Firebase ID token>
  Purpose: Register user on first call, return/migrate credits
  Response: { user_id, email, credits_balance, is_new_user }
```

### Detection (main AI endpoint)

```
POST /detect
  Body: multipart/form-data OR application/json
    file: binary (image or video)
    url: string (alternative to file)
    trusted_metadata: JSON string (optional, sidecar EXIF from mobile)
  Headers:
    X-Device-ID: device fingerprint
    X-Turnstile-Token: Cloudflare CAPTCHA token (required)
    Authorization: Bearer <token> (optional — guest mode supported)
  Cost: 10 credits (or deducted from guest wallet)
  Response: {
    summary: "AI-Generated" | "Likely AI-Generated" | "Likely Authentic" | "No AI Detected",
    confidence_score: 0.0-1.0,
    evidence_chain: [{ layer, status, label, detail }],
    new_balance: int,
    short_id: string | null,     // 8-char alphanumeric ID for sharing
    is_cached: bool,
    is_short_circuited: bool,
    media_type: "image" | "video"
  }
```

### Credits

```
GET /api/user/balance
  Headers: Authorization: Bearer OR X-Device-ID (guest)
  Response: { balance: int }

POST /api/ads/reward
  Headers: Authorization: Bearer OR X-Device-ID
  Purpose: Claim ad-watch reward (max 3/UTC day)
  Response: { credits_granted: 20, new_balance: int, rewards_today: int }

POST /api/credits/add
  Query: device_id, amount, secret_key
  Purpose: Webhook-triggered credit recharge (internal use)
  Response: { device_id, new_balance, success }
```

### Checkout (Lemon Squeezy)

```
POST /api/checkout/create
  Body: { variant_id: string, user_id: string }
  Response: { checkout_url: string }

GET /api/debug/variants
  Purpose: List all Lemon Squeezy product variants (temp debug endpoint)
```

### Inpainting (Object Removal)

```
POST /inpaint/image
  Body: multipart/form-data
    image: binary (original image)
    mask: binary (painted mask — white = remove area)
  Headers:
    X-Device-ID: device fingerprint
    X-Turnstile-Token: CAPTCHA token (required)
    Authorization: Bearer (optional)
    X-Op-Ref: free retry token (optional, 10 min TTL for free refinement)
  Cost: 20 credits (waived if valid X-Op-Ref)
  Response body: PNG bytes (inpainted result)
  Response headers:
    X-User-Balance: int (new balance)
    X-Op-Ref: string (new free retry token for next refinement)
```

### Reports (Sharing)

```
POST /api/v1/reports/share
  Body: { short_id: string }
  Headers: Authorization: Bearer
  Purpose: Publish a scan result as a public share link (14-day TTL)
  Response: { report_id: string, shared_url: string }
  Note: Idempotent — re-publishing same short_id returns existing link

GET /api/v1/reports/share/{report_id}
  No auth required (public)
  Purpose: Fetch a shared scan result
  Auto-extends TTL if < 3 days remain
  Response: detection result object
```

### Webhooks

```
POST /webhook/runpod
  Query: secret=<RUNPOD_WEBHOOK_SECRET>
  Purpose: Receive RunPod GPU job completion callbacks
  Publishes to Redis channel: runpod:result:{job_id}

POST /webhooks/lemonsqueezy
  Headers: X-Signature: HMAC-SHA256
  Events: order_created, order_paid, order_refunded
  Grants/deducts credits via atomic Firestore transactions
  Idempotent: duplicate events safely skipped
```

### System

```
GET /health            → { "status": "healthy" }
GET /robots.txt        → "User-agent: *\nDisallow: /"
GET /.well-known/assetlinks.json  → Android App Links verification JSON
GET /mobile-captcha.html         → Cloudflare Turnstile HTML page (for Android WebView)
```

---

## Detection Pipeline (app/detection/pipeline.py)

The core AI detection flow in `detect_ai_media()`:

```
1. C2PA Manifest Check (instant — cryptographic provenance)
   ├─ Parse C2PA assertions embedded in file (c2pa-python library)
   ├─ Match against 100+ known AI generators (DALL-E, Midjourney, Sora, etc.)
   ├─ Exclude real camera makes (Sony, Canon, Nikon, Leica, Apple)
   └─ → Instant return (confidence 1.0) if AI generator found in manifest

2a. Video Route (if video file)
    ├─ ffprobe metadata extraction (async subprocess)
    ├─ Score metadata: human_score (device markers, GPS, FPS patterns) + ai_score (FFmpeg, round FPS, AI encoders)
    ├─ Early exits:
    │  ├─ human_score ≥ 0.60 && ai_score < 0.30 → "No AI Detected" (0.99)
    │  └─ ai_score ≥ 0.70 && human_score < 0.20 → "AI-Generated" (0.99)
    ├─ Extract 3 frames at 20%, 50%, 80% of duration (quality-filtered)
    └─ Batch Gemini analysis → consensus result

2b. Image Route (if image file)
    ├─ Extract EXIF metadata (or use trusted_metadata from mobile)
    ├─ Score metadata:
    │  ├─ human_score: camera device provenance, physical consistency (exposure, ISO, flash), GPS, timestamps
    │  ├─ ai_score: AI software keywords, missing camera info, AI-typical dimensions (512/768/1024px)
    │  └─ tiered_score: known AI generator strings in metadata
    ├─ Early exits (skip Gemini to save cost):
    │  ├─ human_score ≥ 0.60 → "Likely Authentic" (0.99)
    │  ├─ ai_score ≥ 0.5 → "Likely AI-Generated" (0.95)
    │  └─ tiered_score ≥ 0.99 → instant return
    ├─ Redis cache check (forensic:{hash} — 24h TTL)
    ├─ Gemini forensic analysis (if no cache hit)
    └─ Cache result + apply score boost for AI-likely results
```

---

## Gemini AI Integration (integrations/gemini/client.py)

**Model:** `gemini-2.0-flash-exp` (or equivalent — configured in `config.py`)  
**API Key:** `GEMINI_API_KEY` env var

**Single image analysis: `analyze_image_pro_turbo(image_source)`**
1. Open image with PIL
2. Resize if > 4MP (2048×2048 cap)
3. Compute noise_cv (spatial uniformity signal) — if in range [0.20, 0.25], inject forensic hint
4. Call Gemini with forensic system prompt + image
5. Returns: `{ confidence: float, signal_category: str, quality_score: int }`

**Video batch analysis: `analyze_batch_images_pro_turbo(frames)`**
- Sends all 3 frames in a single Gemini request
- Returns consensus: `{ confidence, explanation, quality_context }`

**Retry config:**
- Max retries: 2
- Retry on: 408, 429, 500, 502, 503, 504
- Backoff: 1s → 2s (exponential, max 5s)
- HTTP timeout: 15 seconds

---

## Authentication & Security

### Firebase Auth (core/firebase_auth.py)

```python
# Protected endpoint (requires sign-in)
async def get_current_user(credentials: HTTPAuthorizationCredentials) -> dict:
    # Verifies Firebase ID token, returns { uid, email }
    # Raises 401 if missing/invalid/expired

# Optional auth (guest + user both work)
async def get_optional_user(credentials) -> dict | None:
    # Returns None instead of 401 if no token
    # Used by /detect and /inpaint
```

### Cloudflare Turnstile CAPTCHA (core/auth.py)

Required on every `/detect` and `/inpaint` request. Verified via:
```
POST https://challenges.cloudflare.com/turnstile/v0/siteverify
  secret: TURNSTILE_SECRET_KEY
  response: <X-Turnstile-Token header>
```

### File Validation (core/file_validator.py) — 3 layers

1. **MIME type** — Block SVG, WMF, EPS, Flash. Allow all other image/* and video/*
2. **Extension** — Allowlist: `.jpg/.png/.gif/.webp/.heic/.avif/.mp4/.mov/.webm/.mkv` etc.
3. **Magic bytes** — PIL `verify()` + `load()` for images; ffprobe for videos. Catches polyglots, decompression bombs

### SSRF Protection (services/detection_service.py)

For URL-based scans:
- Reject `http://` (HTTPS only)
- Resolve hostname via DNS
- Block private IP ranges (10.x, 192.168.x, 127.x, etc.)
- Handle max 1 redirect (re-validate after redirect)
- Stream to disk in 64 KB chunks (never hold payload in RAM)

### Rate Limiting (core/rate_limiter.py)

Redis sliding-window per user or device ID:
- Key format: `rate_limit:{identifier}`
- Window: configurable (default 24h)
- `/detect`: enforced via credit cost (insufficient credits = blocked)
- `/api/user/balance`: 10 req/60s
- `/api/ads/reward`: 3 per UTC day (hard idempotency in Firestore)

### IP/Device Limiting (core/auth.py)

- Redis key: `ip_devices:{ip}` — set of device IDs seen from this IP
- Max 3 new device IDs per IP per 24h
- Mobile-fallback IDs: limit 1
- Excess requests require solving Turnstile

---

## Database (Firebase Firestore — async)

**Firestore Collections:**

```
users/{uid}
  ├─ email: string
  ├─ credits_balance: int
  ├─ credits_version: int         # For optimistic locking
  ├─ created_at: timestamp
  └─ credit_ledger/{auto_id}      # Immutable transaction log
       ├─ delta: int               # Positive (grant) or negative (consume)
       ├─ reason: string           # detect, inpaint, purchase, signup_bonus, ad_reward, refund
       ├─ reference_id: string     # filename, order_id, etc.
       ├─ balance_after: int
       └─ created_at: SERVER_TIMESTAMP

guest_wallets/{device_id}
  ├─ credits: int
  ├─ created_at: timestamp
  └─ last_activity: timestamp     # TTL: 180 days of inactivity

shared_reports/{report_id}
  ├─ short_id: string             # 8-char ID linking back to detection result
  ├─ scan_result: object          # Full detection response
  └─ expires_at: timestamp        # 14 days, auto-extended on access

purchases/{order_id}
  ├─ lemon_order_id: string
  ├─ lemon_variant_id: string
  ├─ user_id: string
  ├─ credits_granted: int
  ├─ status: "pending"|"paid"|"grant_failed"|"refunded"
  ├─ test_mode: boolean
  └─ created_at: timestamp

ad_rewards/{uid}_{YYYY-MM-DD}
  ├─ user_id: string
  ├─ date: string
  ├─ count: int                   # 0-3 per day
  └─ last_reward_at: timestamp
```

---

## Redis (integrations/redis_client.py)

**Connection:** `REDIS_URL` env var (Railway managed Redis)

**Key namespace:**

```
forensic:{hash}              → Detection result cache (24h TTL)
report:{short_id}            → Detection result for sharing (24h TTL)
ip_devices:{ip}              → Set of device IDs from this IP (24h TTL)
runpod:result:{job_id}       → RunPod job output (30s TTL, pub/sub)
guest:{device_id}:balance    → Guest wallet balance (persistent)
extending:{report_id}        → TTL-extend lock (60s — prevents duplicate extends)
op_ref:{user_id}:{token}     → Free inpaint retry token (10 min TTL)
rate_limit:{identifier}      → Zset of request timestamps (24h window)
```

---

## Credit Engine (services/credit_engine.py)

**Atomic Firestore transactions:**

```python
# Deduct credits (raises 402 if insufficient)
consume_credits(user_id, cost=10, reason="detect", reference_id="filename.jpg")

# Add credits (can be negative for refunds)
grant_credits(user_id, amount=40, reason="signup_bonus", reference_id="welcome")

# Get or create user (auto-grants signup bonus on first call)
get_or_create_user(uid, email) → { credits_balance, is_new_user }
```

Every debit/credit appends an immutable entry to `credit_ledger/{auto_id}`.

---

## Payments (Lemon Squeezy)

**Webhook:** `POST /webhooks/lemonsqueezy`  
Auth: `X-Signature` header (HMAC-SHA256 of body, key = `LEMONSQUEEZY_WEBHOOK_SECRET`)

**Events handled:**
- `order_paid` → grant credits to user, write to `purchases/` collection
- `order_refunded` → deduct credits (negative grant), update status

**Variant IDs** are mapped to credit amounts in `config.py`:
```python
lemon_squeezy_variants: dict       # Live mode variant ID → credits
lemon_squeezy_test_variants: dict  # Test mode variant ID → credits
```

**App env toggle:** `APP_ENV=prod` uses live keys; anything else uses test keys.

---

## GPU Workers

### Modal (Object Removal)
- `gpu_worker.py` dispatches to a Modal remote function
- Model: LaMa (`big-lama.pt`) — pre-baked into Docker image at build time
- Input: image bytes + mask bytes
- Output: PNG bytes
- Cost logged to Axiom: `INPAINT` transaction type

### RunPod (Detection — GPU path)
- `runpod.py` dispatches GPU inference jobs
- Webhook callback: `POST /webhook/runpod?secret=<RUNPOD_WEBHOOK_SECRET>`
- Result delivered via Redis pub/sub (`runpod:result:{job_id}`)
- Used as a fallback/enhanced path alongside Gemini

---

## Configuration (app/config.py)

Pydantic `BaseSettings` — all values overridable via env vars. Key fields:

```python
# Credits
welcome_credits: int = 40
detect_credit_cost: int = 10
inpaint_credit_cost: int = 20
default_recharge_amount: int = 20   # Per ad reward

# TTLs (days)
wallet_ttl_days: int = 180
report_ttl_days: int = 14
report_extend_days: int = 14
report_extend_threshold_days: int = 3

# File limits
max_image_upload_mb: int = 20
max_video_upload_mb: int = 200
pil_max_image_pixels: int = 20_000_000

# Gemini
gemini_http_timeout_ms: int = 15_000
gemini_max_retries: int = 2
gemini_temperature: float = 1.0
gemini_max_pixels: int = 4_194_304    # 2048×2048

# Detection thresholds
ai_confidence_threshold: float = 0.5
video_human_early_exit_score: float = 0.60
video_ai_early_exit_score: float = 0.70
```

---

## Environment Variables

```bash
# Firebase (required)
FIREBASE_SERVICE_ACCOUNT='{"type":"service_account",...}'

# AI (required)
GEMINI_API_KEY=

# CAPTCHA (required)
TURNSTILE_SECRET_KEY=

# Redis (required)
REDIS_URL=redis://...

# Lemon Squeezy (required for payments)
LEMONSQUEEZY_STORE_ID=
LEMONSQUEEZY_API_KEY=
LEMONSQUEEZY_API_KEY_TEST_MODE=
LEMONSQUEEZY_WEBHOOK_SECRET=

# RunPod (required for GPU detection)
RUNPOD_WEBHOOK_SECRET=

# App mode
APP_ENV=prod    # or dev

# Android App Links
ANDROID_SHA256_FINGERPRINT=

# Monitoring (optional)
SENTRY_DSN=
AXIOM_API_KEY=

# Overridable config fields (examples)
DETECT_CREDIT_COST=10
INPAINT_CREDIT_COST=20
WELCOME_CREDITS=40
```

---

## Middleware & App Setup (app/main.py)

### CORS
```python
allow_origins = ["https://fauxlens.com"]
allow_credentials = True
allow_methods = ["*"]
allow_headers = ["*"]
expose_headers = ["X-User-Balance", "X-Op-Ref"]
```

### Request Logging Middleware
Every request logs: method, path, status_code, duration_ms, IP, user_agent, device_id, request_id (UUID). Sets Sentry scope with device_id + IP.

### Exception Handler
- Drains up to 64 KB of request body (prevents HTTP/2 errors)
- Forces CORS headers on error responses (browsers need CORS headers on 4xx/5xx)
- Sends 5xx errors to Sentry

### Lifespan (startup/shutdown)
```python
# Startup
threadpool → 30 threads (for parallel Gemini calls)
firebase_module.initialize()
redis_module.initialize()
http_module.initialize()

# Shutdown
http_module.close()
redis_module.close()
```

---

## Deployment (Railway.app + Docker)

**Dockerfile highlights:**
- Base: `python:3.11-slim`
- System deps: `libgl1`, `libglib2.0-0`, `ffmpeg`, `libjemalloc2`
- PyTorch: CPU-only (saves bandwidth)
- LaMa model `big-lama.pt` pre-downloaded during build (no cold start)
- Jemalloc preloaded: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- Workers: 1 (prevents double-loading AI models)
- Event loop: uvloop

**railway.toml:**
```toml
[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

**Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT --loop uvloop --workers 1`

---

## Code Quality & Linting

**Ruff** (in `pyproject.toml`):
```toml
line-length = 100
target-version = "py311"
select = ["E", "W", "F", "I", "B", "UP"]
```

**Mypy:**
```toml
python_version = "3.11"
strict = false
ignore_missing_imports = true
```

**Pytest:**
```toml
asyncio_mode = "auto"
testpaths = ["tests"]
```

Run via:
```bash
cd backend-python
ruff check app/
mypy app/
pytest
```

---

## Key Architectural Decisions

1. **Async-first**: All I/O is `async/await`. Firestore uses native async client. Redis uses async client. aiohttp for HTTP.
2. **Metadata-first detection**: EXIF/metadata scoring runs first and can short-circuit before calling Gemini, saving cost.
3. **Dual billing paths**: Authenticated users use Firestore credit ledger. Guests use Redis wallet. Credits migrate on sign-in.
4. **Free retry token (X-Op-Ref)**: After inpainting, the response includes a token giving one free refinement within 10 minutes, improving UX without double-charging.
5. **CORS on errors**: Custom exception handler ensures CORS headers appear even on 4xx/5xx responses.
6. **Thread pool expansion**: Default Python thread pool (~12) is too small for parallel Gemini calls — expanded to 30 at startup.

---

## Gotchas & Common Traps

### CORS headers missing from custom exception handlers
If you add a new `@app.exception_handler(SomeError)`, it MUST call the CORS header helper. Without it, the browser shows "Network Error" (a CORS error) instead of the actual error message. The existing `custom_http_exception_handler` in `main.py` handles this for `HTTPException` — copy that pattern.

### Redis unavailable → silent fallback, no exception
`detection/cache.py` falls back to an in-memory LRU cache if Redis is down. Detection still works but results are not shared across workers and are lost on restart. No error is thrown — watch logs for `redis_unavailable` events.

### Async Firestore client must be initialized before first request
The `AsyncClient` is initialized in the lifespan `startup`. If you access `firestore_client` before startup completes (e.g., in a module-level call), you get `AttributeError: 'NoneType'`. Always access it via the initialized module.

### `trusted_metadata` from mobile can spoof EXIF scores
The `trusted_metadata` field lets the Android app send sidecar EXIF data. This is intentional for images where EXIF is stripped during transit — but it means the metadata scoring can be influenced by the client. The field is validated for structure but not authenticity. It's a trust-the-client design choice.

### Gemini rate limits: ~1000 req/min on Flash tier
Under heavy load, Gemini returns 429. The client retries up to 2 times with exponential backoff. If all retries fail, the detection falls back to metadata-only scoring (not an error). Watch `gemini_rate_limit` log events.

### Video frames that all fail quality filter → fallback to middle frame
`video_detector.py` extracts 3 frames and quality-filters them (brightness + sharpness). If all 3 fail the filter, it grabs the middle frame unconditionally. There is never a case where zero frames are sent to Gemini.

### `c2pa-python` returns `None` for non-C2PA files
`get_c2pa_manifest()` returns `None` for any file without a C2PA manifest. The pipeline treats this as `status: "not_found"` and continues to metadata scoring. Always null-check the return value.

### PIL decompression bomb on adversarial images
PIL raises `DecompressionBombWarning` (or error) above `pil_max_image_pixels` (default 20M). This is intentional — it prevents adversarial images that claim to be 50,000×50,000 pixels. Do not raise the limit.

### Thread pool exhaustion under parallel load
The thread pool was expanded to 30 threads in lifespan startup. If you add new CPU-heavy sync operations that use `asyncio.to_thread()`, be aware the pool is shared. 30 threads supports ~10 concurrent scans at 3 threads each.

### Lemon Squeezy webhooks are idempotent by design
The `purchases/{order_id}` Firestore doc is written on first `order_paid` event. Duplicate webhooks (Lemon Squeezy retries) are safely ignored because the doc already exists.

---

## Local Development

```bash
cd backend-python

# Install dependencies
pip install -r requirements.txt

# Run dev server (hot-reload)
uvicorn app.main:app --reload --port 8000

# Run tests
python -m pytest tests/ -x -q --tb=short

# Lint + format
ruff format app/
ruff check --fix app/
mypy app/ --ignore-missing-imports
```

**Required env vars for local dev** (create `backend-python/.env`):
```bash
FIREBASE_SERVICE_ACCOUNT='{"type":"service_account",...}'
GEMINI_API_KEY=your_key
TURNSTILE_SECRET_KEY=1x0000000000000000000000000000000AA  # Cloudflare test key
REDIS_URL=redis://localhost:6379
APP_ENV=dev
```

---

## Agent Routing (from root CLAUDE.md)

| Task | Agent |
|------|-------|
| Code review | `python-reviewer` |
| Build failures | `build-error-resolver` |
| Security issues | `security-reviewer` |
| Architecture decisions | `architect` (Opus) |
| Database / Firestore | `database-reviewer` |
