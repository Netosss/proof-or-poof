"""
Central application configuration.

Every tunable value lives here as a typed, documented field.
Any field can be overridden at runtime via an environment variable of the
same name (case-insensitive), e.g.:

    DETECT_CREDIT_COST=3 uvicorn app.main:app    # one-off promotion
    export WALLET_TTL_DAYS=90                     # staging override

A `.env` file at the project root is loaded automatically.
"""

import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # DETECT_CREDIT_COST == detect_credit_cost
        extra="ignore",  # silently drop unknown env vars
    )

    # ------------------------------------------------------------------ #
    # Firestore TTLs (days)                                               #
    # ------------------------------------------------------------------ #
    wallet_ttl_days: int = Field(180, description="Guest wallet inactivity expiry (6 months)")
    report_ttl_days: int = Field(14, description="Shared report default lifetime")
    report_extend_days: int = Field(
        14, description="Days added when a viral report is auto-extended"
    )
    report_extend_threshold_days: int = Field(
        3, description="Trigger extension when fewer than N days remain"
    )

    # ------------------------------------------------------------------ #
    # Redis TTLs (seconds)                                                #
    # ------------------------------------------------------------------ #
    report_cache_ttl_sec: int = Field(
        86_400, description="24 h — shareable report payload (report:{short_id})"
    )
    share_lock_ttl_sec: int = Field(
        86_400, description="24 h — idempotency lock after Firestore write"
    )
    extend_lock_ttl_sec: int = Field(
        60, description="60 s — background-task dedup lock (extending:{report_id})"
    )
    deepfake_cache_ttl_sec: int = Field(
        86_400, description="24 h — forensic analysis cache (forensic:{hash})"
    )
    rate_limit_window_sec: int = Field(
        86_400, description="24 h — IP/device rate-limit sliding window"
    )
    deepfake_dedupe_ttl_sec: int = Field(600, description="10 min — per-device duplicate-scan lock")

    # ------------------------------------------------------------------ #
    # Credits & Billing                                                   #
    # ------------------------------------------------------------------ #
    welcome_credits: int = Field(40, description="Credits granted to every brand-new wallet")
    detect_credit_cost: int = Field(10, description="Credits charged per /detect call")
    inpaint_credit_cost: int = Field(20, description="Credits charged per /inpaint call")
    default_recharge_amount: int = Field(20, description="Default credits per ad-reward recharge")

    # ------------------------------------------------------------------ #
    # Lemon Squeezy variant → credit mapping                             #
    # The backend is the single source of truth for credit amounts.      #
    # Never trust the webhook payload for credit amounts — look up here. #
    #                                                                    #
    # APP_ENV=prod (default) → uses lemon_squeezy_variants +            #
    #                           LEMONSQUEEZY_API_KEY (live)              #
    # APP_ENV=dev            → uses lemon_squeezy_test_variants +        #
    #                           LEMONSQUEEZY_API_KEY_TEST_MODE           #
    # ------------------------------------------------------------------ #
    app_env: str = Field(
        "prod",
        description="Runtime environment: 'prod' or 'dev'. Controls which LS keys are used.",
    )

    lemon_squeezy_variants: dict = Field(
        default_factory=dict,
        description="Live variant ID (str) → credits (int). Used when app_env=prod.",
    )

    lemon_squeezy_test_variants: dict = Field(
        default_factory=dict,
        description="Test variant ID (str) → credits (int). Used when app_env=dev.",
    )

    @field_validator("lemon_squeezy_variants", "lemon_squeezy_test_variants", mode="before")
    @classmethod
    def parse_variants(cls, v):
        """
        Handles all formats Railway/shell might produce:
          - Already a dict              → use as-is
          - '{"1360784": 100, ...}'     → parse JSON directly
          - '"{\"1360784\": 100, ...}"' → strip outer quotes, unescape, parse
        """
        if not v:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
            s = s.replace('\\"', '"')
            return json.loads(s)
        return v

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    @property
    def active_ls_variants(self) -> dict:
        """Returns the correct variant map for the current environment."""
        return self.lemon_squeezy_test_variants if self.is_dev else self.lemon_squeezy_variants

    # ------------------------------------------------------------------ #
    # Enterprise API (S2S) — separate credit ledger from consumer users  #
    # ------------------------------------------------------------------ #
    enterprise_credit_cost: int = Field(
        1, description="Enterprise credits charged per /v1/analyze call (1:1 by default)"
    )
    enterprise_default_rate_limit_per_min: int = Field(
        60, description="Per-credential rate limit ceiling (req/min); overridable per partner"
    )
    enterprise_timestamp_drift_sec: int = Field(
        300, description="Max allowed clock drift on X-FauxLens-Timestamp (seconds)"
    )
    enterprise_replay_window_sec: int = Field(
        300, description="Redis nonce TTL for signature replay protection (seconds)"
    )
    enterprise_idempotency_ttl_sec: int = Field(
        86_400, description="24 h — Idempotency-Key cached-response TTL"
    )
    enterprise_max_request_bytes: int = Field(
        209_715_200,
        description="200 MB — hard ceiling for enterprise multipart bodies; mirrors video upload cap",
    )
    enterprise_ls_variants: dict = Field(
        default_factory=dict,
        description="Live LS variant ID (str) → enterprise credits (int). Used when app_env=prod.",
    )
    enterprise_ls_test_variants: dict = Field(
        default_factory=dict,
        description="Test LS variant ID (str) → enterprise credits (int). Used when app_env=dev.",
    )

    # Variant ID → per-credential rate limit (req/min) for the tier that
    # variant represents. The LS webhook reads this on order_paid and raises
    # partner.rate_limit_per_min so the published Pricing-page numbers
    # (60 / 120 / 300 req/min) take effect without manual intervention.
    # Partners are only ever RAISED — buying a smaller top-up never lowers
    # a higher-tier customer's ceiling.
    enterprise_variant_rate_limits: dict = Field(
        default_factory=dict,
        description=(
            "Live LS variant ID (str) → rate limit per minute (int). "
            "Auto-applied to partner.rate_limit_per_min on purchase. "
            "Used when app_env=prod."
        ),
    )
    enterprise_variant_test_rate_limits: dict = Field(
        default_factory=dict,
        description=(
            "Test LS variant ID (str) → rate limit per minute (int). "
            "Auto-applied to partner.rate_limit_per_min on purchase. "
            "Used when app_env=dev."
        ),
    )

    @field_validator(
        "enterprise_ls_variants",
        "enterprise_ls_test_variants",
        "enterprise_variant_rate_limits",
        "enterprise_variant_test_rate_limits",
        mode="before",
    )
    @classmethod
    def parse_enterprise_variants(cls, v):
        """Same parsing logic as consumer variants — handles dict / JSON / quoted-JSON."""
        if not v:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
            s = s.replace('\\"', '"')
            return json.loads(s)
        return v

    @property
    def active_enterprise_ls_variants(self) -> dict:
        """Returns the correct enterprise variant map for the current environment."""
        return self.enterprise_ls_test_variants if self.is_dev else self.enterprise_ls_variants

    @property
    def active_enterprise_variant_rate_limits(self) -> dict:
        """Returns the variant→rate-limit map for the current environment."""
        return (
            self.enterprise_variant_test_rate_limits
            if self.is_dev
            else self.enterprise_variant_rate_limits
        )

    # ------------------------------------------------------------------ #
    # Pricing (USD per unit)                                              #
    # ------------------------------------------------------------------ #
    gpu_rate_per_sec: float = Field(0.0019, description="RunPod A5000/L4 rate for detection")
    inpaint_rate_per_sec: float = Field(0.000222, description="Modal L4 rate for inpainting")
    cpu_rate_per_sec: float = Field(0.0001, description="Estimated Railway CPU rate")
    gemini_fixed_cost: float = Field(
        0.0033,
        description="Cost per Gemini 3 Flash Preview request (verified: ~6,380 input tokens × $0.50/1M + ~27 output tokens × $3.00/1M)",
    )
    ad_revenue_per_reward: float = Field(0.015, description="Avg eCPM for a verified ad view")

    # ------------------------------------------------------------------ #
    # File Size Limits                                                    #
    # ------------------------------------------------------------------ #
    max_image_download_mb: int = Field(50, description="Max MB for URL / data-URI downloads")
    max_image_upload_mb: int = Field(20, description="Max MB for multipart image uploads")
    max_video_upload_mb: int = Field(200, description="Max MB for video uploads")
    pil_max_image_pixels: int = Field(
        20_000_000, description="PIL decompression-bomb guard (pixels)"
    )

    # ------------------------------------------------------------------ #
    # Rate Limiting                                                       #
    # ------------------------------------------------------------------ #
    max_new_devices_per_ip: int = Field(3, description="New wallets allowed per IP per 24 h window")
    rate_limit_request_window_sec: int = Field(
        60, description="Sliding window for per-user request rate (seconds)"
    )
    rate_limit_max_requests: int = Field(
        10, description="Max requests allowed within the rate-limit window"
    )
    rate_limit_memory_limit: int = Field(
        1000, description="Max keys before in-memory rate-limit map is pruned"
    )

    # ------------------------------------------------------------------ #
    # Short ID                                                            #
    # ------------------------------------------------------------------ #
    short_id_length: int = Field(
        8, description="Characters in a share-link short ID (62^8 ≈ 218 T combos)"
    )

    # ------------------------------------------------------------------ #
    # Background Tasks                                                    #
    # ------------------------------------------------------------------ #
    cleanup_interval_sec: int = Field(
        30, description="How often the periodic job-cleanup task runs (seconds)"
    )

    # ------------------------------------------------------------------ #
    # Detector: Local Cache (Redis-absent fallback)                       #
    # ------------------------------------------------------------------ #
    local_cache_max_size: int = Field(100, description="Max entries in the in-memory LRU cache")
    local_cache_ttl_sec: int = Field(3_600, description="1 h — local cache entry lifetime")

    # ------------------------------------------------------------------ #
    # Detector: Hashing                                                   #
    # ------------------------------------------------------------------ #
    hash_chunk_threshold_mb: int = Field(
        10, description="Below this file size (MB), hash the full file"
    )
    hash_chunk_size_kb: int = Field(512, description="Chunk size (KB) for large-file smart hash")
    image_hash_header_mb: int = Field(2, description="Bytes read (MB) for path-based image hash")

    # ------------------------------------------------------------------ #
    # Detector: Video Processing                                          #
    # ------------------------------------------------------------------ #
    video_header_read_mb: int = Field(1, description="Video container header read size (MB)")
    ffprobe_timeout_sec: int = Field(10, description="Timeout for ffprobe subprocess (seconds)")
    video_jpeg_quality: int = Field(
        95, description="JPEG quality when encoding video frames for analysis"
    )
    frame_min_brightness: float = Field(
        20.0, description="Reject frame if mean brightness is below this"
    )
    frame_min_sharpness: float = Field(
        50.0, description="Reject frame if Laplacian variance is below this"
    )

    # ------------------------------------------------------------------ #
    # AI Decision Threshold                                               #
    # ------------------------------------------------------------------ #
    ai_confidence_threshold: float = Field(
        0.55,
        description=(
            "Score above this → classified as AI-generated. "
            "Aligned with the prompt-side dead zone [0.40, 0.70) — anything ≥ 0.55 "
            "must come from the suspicion band (0.70+) under normal operation."
        ),
    )

    # ------------------------------------------------------------------ #
    # Gemini Client                                                       #
    # ------------------------------------------------------------------ #
    gemini_model: str = Field(
        "gemini-3-flash-preview",
        description=(
            "Gemini model ID for forensic inference. EMPIRICAL: 3-flash-preview "
            "outperforms gemini-3.5-flash by ~20 points on our forensic gold set — "
            "3.5-flash systematically misses polished AI portraits (Flux/Midjourney "
            "studio-style) regardless of prompt. Do NOT 'upgrade' to 3.5 without "
            "re-running the gold set to confirm the regression is gone."
        ),
    )
    gemini_thinking_level: str = Field(
        "LOW",
        description=(
            "Reasoning budget for the forensic inference call. With the lean prompt, "
            "LOW + temp=0.0 produces deterministic verdicts and lets the model trust "
            "its first-pass perception instead of rationalising AI signals away via "
            "extended reasoning."
        ),
    )
    gemini_http_timeout_ms: int = Field(20_000, description="HTTP client total timeout (ms)")
    gemini_max_retries: int = Field(2, description="Max retry attempts on transient errors")
    gemini_retry_initial_delay: float = Field(1.0, description="First retry delay (seconds)")
    gemini_retry_max_delay: float = Field(5.0, description="Max retry back-off delay (seconds)")
    gemini_retry_exp_base: float = Field(2.0, description="Exponential back-off multiplier")
    gemini_max_pixels: int = Field(4_194_304, description="2048×2048 — resize cap before upload")
    gemini_jpeg_quality: int = Field(
        95, description="JPEG quality for single-image forensic upload"
    )
    gemini_batch_jpeg_quality: int = Field(
        95,
        description=(
            "JPEG quality for batch / video-frame upload. Bumped from 85 → 95 "
            "so we don't introduce fresh compression artifacts the prompt then "
            "mistakes for AI smoothness."
        ),
    )
    gemini_temperature: float = Field(1.0, description="Sampling temperature for Gemini model")
    gemini_ai_vote_threshold: float = Field(
        0.55,
        description=(
            "Per-frame confidence threshold for an AI vote. Aligned with "
            "ai_confidence_threshold and the prompt-side dead zone."
        ),
    )

    # ------------------------------------------------------------------ #
    # Gemini Quality Scoring                                              #
    # ------------------------------------------------------------------ #
    quality_score_init: int = Field(100, description="Starting quality score")
    quality_dqt_severe_threshold: int = Field(
        30, description="DQT avg > this → severe JPEG compression"
    )
    quality_dqt_severe_penalty: int = Field(
        40, description="Score deduction for severe compression"
    )
    quality_dqt_moderate_threshold: int = Field(
        20, description="DQT avg > this → moderate compression"
    )
    quality_dqt_moderate_penalty: int = Field(
        20, description="Score deduction for moderate compression"
    )
    quality_pixels_tiny: int = Field(250_000, description="< this → 'tiny' resolution")
    quality_pixels_tiny_penalty: int = Field(50, description="Score deduction for tiny resolution")
    quality_pixels_small: int = Field(800_000, description="< this → 'small' resolution")
    quality_pixels_small_penalty: int = Field(
        40, description="Score deduction for small resolution"
    )
    quality_blur_zero_threshold: int = Field(
        10, description="Laplacian var < this → near-zero detail"
    )
    quality_blur_zero_penalty: int = Field(60, description="Score deduction for near-zero detail")
    quality_blur_extreme_threshold: int = Field(
        25, description="Laplacian var < this → extreme blur"
    )
    quality_blur_extreme_penalty: int = Field(25, description="Score deduction for extreme blur")
    quality_blur_soft_threshold: int = Field(50, description="Laplacian var < this → soft/smooth")
    quality_blur_soft_penalty: int = Field(10, description="Score deduction for soft focus")
    quality_sharp_high_threshold: int = Field(
        2000, description="Laplacian var > this → exceptional sharpness"
    )
    quality_sharp_high_bonus: int = Field(20, description="Score bonus for exceptional sharpness")
    quality_sharp_med_threshold: int = Field(
        1500, description="Laplacian var > this → high sharpness"
    )
    quality_sharp_med_bonus: int = Field(15, description="Score bonus for high sharpness")
    quality_sharp_uncomp_threshold: int = Field(
        600, description="Sharp + uncompressed bonus threshold"
    )
    quality_sharp_uncomp_bonus: int = Field(20, description="Score bonus for sharp + uncompressed")
    quality_sharp_ok_threshold: int = Field(300, description="Moderately sharp + uncompressed")
    quality_sharp_ok_bonus: int = Field(10, description="Score bonus for moderately sharp")
    quality_low_threshold: int = Field(
        50, description="Score < this → 'low quality' Gemini context"
    )
    quality_medium_threshold: int = Field(
        75, description="Score < this → 'medium quality' Gemini context"
    )

    # ------------------------------------------------------------------ #
    # Forensic Pre-processing                                             #
    # ------------------------------------------------------------------ #
    forensic_noise_radius: int = Field(
        2,
        description=(
            "Gaussian blur radius for noise residual map generation. "
            "radius=2 captures pixel-level noise; raise to 3-4 to also capture "
            "broader diffusion-model noise patterns."
        ),
    )
    forensic_noise_cv_floor: float = Field(
        0.20,
        description=(
            "Lower bound of noise_cv range that triggers Gemini hint injection. "
            "Calibrated with corrected float32 signed-residual math against 419 real "
            "Open Images V7 photos. [0.20, 0.25) catches frame_debug_129193.jpg AI "
            "(ncv=0.2110) with 0 gold-set real FPs. ~11% of production real images fall "
            "in this band but Gemini treats the hint as a soft signal and still classifies "
            "them correctly (all 8 gold real images pass in every test run)."
        ),
    )
    forensic_noise_cv_ceil: float = Field(
        0.25,
        description=(
            "Upper bound of noise_cv range that triggers Gemini hint injection. "
            "Disabling the hint (floor > ceil) drops accuracy from 100% to 90.5% on "
            "the 21-image gold set — frame_debug_129193.jpg becomes a reliable miss. "
            "Kept at 0.25; raise floor if production false-positive rate climbs."
        ),
    )

    # ------------------------------------------------------------------ #
    # Inpainting (remover.py)                                             #
    # ------------------------------------------------------------------ #
    max_inpaint_dimension: int = Field(
        2048, description="Max image dimension before downscale for GPU"
    )
    inpaint_bucket_step: int = Field(
        64, description="Round dimensions to nearest N px (prevents JIT cache bloat)"
    )
    torch_num_threads: int = Field(
        4, description="Math threads on CPU (sweet spot for 1280 px images)"
    )
    torch_num_interop_threads: int = Field(
        1, description="Graph threads on CPU (prevents 48-thread overhead)"
    )

    # ------------------------------------------------------------------ #
    # Derived byte-level properties (computed from MB fields)             #
    # ------------------------------------------------------------------ #
    @property
    def max_image_download_bytes(self) -> int:
        return self.max_image_download_mb * 1024 * 1024

    @property
    def max_image_upload_bytes(self) -> int:
        return self.max_image_upload_mb * 1024 * 1024

    @property
    def max_video_upload_bytes(self) -> int:
        return self.max_video_upload_mb * 1024 * 1024

    @property
    def hash_chunk_threshold_bytes(self) -> int:
        return self.hash_chunk_threshold_mb * 1024 * 1024

    @property
    def hash_chunk_size_bytes(self) -> int:
        return self.hash_chunk_size_kb * 1024

    @property
    def image_hash_header_bytes(self) -> int:
        return self.image_hash_header_mb * 1024 * 1024

    @property
    def video_header_read_bytes(self) -> int:
        return self.video_header_read_mb * 1024 * 1024

    # Image-hash thumbnail sizes (algorithmic, not env-overridable)
    @property
    def image_hash_thumb_fast(self) -> tuple[int, int]:
        return (32, 32)

    @property
    def image_hash_thumb_full(self) -> tuple[int, int]:
        return (64, 64)


# Single shared instance — import this everywhere.
settings = Settings()
