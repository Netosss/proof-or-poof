"""
Central application configuration.

Every tunable value lives here as a typed, documented field.
Any field can be overridden at runtime via an environment variable of the
same name (case-insensitive), e.g.:

    DETECT_CREDIT_COST=3 uvicorn app.main:app    # one-off promotion
    export WALLET_TTL_DAYS=90                     # staging override

A `.env` file at the project root is loaded automatically.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # DETECT_CREDIT_COST == detect_credit_cost
        extra="ignore",         # silently drop unknown env vars
    )

    # ------------------------------------------------------------------ #
    # Firestore TTLs (days)                                               #
    # ------------------------------------------------------------------ #
    wallet_ttl_days: int = Field(
        180, description="Guest wallet inactivity expiry (6 months)"
    )
    report_ttl_days: int = Field(
        14, description="Shared report default lifetime"
    )
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
    deepfake_dedupe_ttl_sec: int = Field(
        600, description="10 min — per-device duplicate-scan lock"
    )

    # ------------------------------------------------------------------ #
    # Credits & Billing                                                   #
    # ------------------------------------------------------------------ #
    welcome_credits: int = Field(
        10, description="Credits granted to every brand-new wallet"
    )
    detect_credit_cost: int = Field(
        5, description="Credits charged per /detect call"
    )
    inpaint_credit_cost: int = Field(
        2, description="Credits charged per /inpaint call"
    )
    default_recharge_amount: int = Field(
        5, description="Default credits per ad-reward recharge"
    )

    # ------------------------------------------------------------------ #
    # Pricing (USD per unit)                                              #
    # ------------------------------------------------------------------ #
    gpu_rate_per_sec: float = Field(
        0.0019, description="RunPod A5000/L4 rate for detection"
    )
    inpaint_rate_per_sec: float = Field(
        0.00031, description="RunPod RTX 4090 rate for inpainting"
    )
    cpu_rate_per_sec: float = Field(
        0.0001, description="Estimated Railway CPU rate"
    )
    gemini_fixed_cost: float = Field(
        0.0024, description="Cost per Gemini 3.0 Pro analysis request"
    )
    ad_revenue_per_reward: float = Field(
        0.015, description="Avg eCPM for a verified ad view"
    )

    # ------------------------------------------------------------------ #
    # File Size Limits                                                    #
    # ------------------------------------------------------------------ #
    max_image_download_mb: int = Field(
        50, description="Max MB for URL / data-URI downloads"
    )
    max_image_upload_mb: int = Field(
        20, description="Max MB for multipart image uploads"
    )
    max_video_upload_mb: int = Field(
        200, description="Max MB for video uploads"
    )
    pil_max_image_pixels: int = Field(
        20_000_000, description="PIL decompression-bomb guard (pixels)"
    )

    # ------------------------------------------------------------------ #
    # Rate Limiting                                                       #
    # ------------------------------------------------------------------ #
    max_new_devices_per_ip: int = Field(
        3, description="New wallets allowed per IP per 24 h window"
    )
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
    local_cache_max_size: int = Field(
        100, description="Max entries in the in-memory LRU cache"
    )
    local_cache_ttl_sec: int = Field(
        3_600, description="1 h — local cache entry lifetime"
    )

    # ------------------------------------------------------------------ #
    # Detector: Hashing                                                   #
    # ------------------------------------------------------------------ #
    hash_chunk_threshold_mb: int = Field(
        10, description="Below this file size (MB), hash the full file"
    )
    hash_chunk_size_kb: int = Field(
        512, description="Chunk size (KB) for large-file smart hash"
    )
    image_hash_header_mb: int = Field(
        2, description="Bytes read (MB) for path-based image hash"
    )

    # ------------------------------------------------------------------ #
    # Detector: Video Processing                                          #
    # ------------------------------------------------------------------ #
    video_header_read_mb: int = Field(
        1, description="Video container header read size (MB)"
    )
    ffprobe_timeout_sec: int = Field(
        10, description="Timeout for ffprobe subprocess (seconds)"
    )
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
        0.5, description="Score above this → classified as AI-generated"
    )

    # ------------------------------------------------------------------ #
    # Gemini Client                                                       #
    # ------------------------------------------------------------------ #
    gemini_http_timeout_ms: int = Field(
        15_000, description="HTTP client total timeout (ms)"
    )
    gemini_max_retries: int = Field(
        2, description="Max retry attempts on transient errors"
    )
    gemini_retry_initial_delay: float = Field(
        1.0, description="First retry delay (seconds)"
    )
    gemini_retry_max_delay: float = Field(
        5.0, description="Max retry back-off delay (seconds)"
    )
    gemini_retry_exp_base: float = Field(
        2.0, description="Exponential back-off multiplier"
    )
    gemini_max_pixels: int = Field(
        4_194_304, description="2048×2048 — resize cap before upload"
    )
    gemini_jpeg_quality: int = Field(
        95, description="JPEG quality for single-image forensic upload"
    )
    gemini_batch_jpeg_quality: int = Field(
        85, description="JPEG quality for batch / video-frame upload"
    )
    gemini_temperature: float = Field(
        1.0, description="Sampling temperature for Gemini model"
    )
    gemini_ai_vote_threshold: float = Field(
        0.5, description="Per-frame confidence threshold for an AI vote"
    )

    # ------------------------------------------------------------------ #
    # Gemini Quality Scoring                                              #
    # ------------------------------------------------------------------ #
    quality_score_init: int = Field(100, description="Starting quality score")
    quality_dqt_severe_threshold: int = Field(30, description="DQT avg > this → severe JPEG compression")
    quality_dqt_severe_penalty: int = Field(40, description="Score deduction for severe compression")
    quality_dqt_moderate_threshold: int = Field(20, description="DQT avg > this → moderate compression")
    quality_dqt_moderate_penalty: int = Field(20, description="Score deduction for moderate compression")
    quality_pixels_tiny: int = Field(250_000, description="< this → 'tiny' resolution")
    quality_pixels_tiny_penalty: int = Field(50, description="Score deduction for tiny resolution")
    quality_pixels_small: int = Field(800_000, description="< this → 'small' resolution")
    quality_pixels_small_penalty: int = Field(40, description="Score deduction for small resolution")
    quality_blur_zero_threshold: int = Field(10, description="Laplacian var < this → near-zero detail")
    quality_blur_zero_penalty: int = Field(60, description="Score deduction for near-zero detail")
    quality_blur_extreme_threshold: int = Field(25, description="Laplacian var < this → extreme blur")
    quality_blur_extreme_penalty: int = Field(25, description="Score deduction for extreme blur")
    quality_blur_soft_threshold: int = Field(50, description="Laplacian var < this → soft/smooth")
    quality_blur_soft_penalty: int = Field(10, description="Score deduction for soft focus")
    quality_sharp_high_threshold: int = Field(2000, description="Laplacian var > this → exceptional sharpness")
    quality_sharp_high_bonus: int = Field(20, description="Score bonus for exceptional sharpness")
    quality_sharp_med_threshold: int = Field(1500, description="Laplacian var > this → high sharpness")
    quality_sharp_med_bonus: int = Field(15, description="Score bonus for high sharpness")
    quality_sharp_uncomp_threshold: int = Field(600, description="Sharp + uncompressed bonus threshold")
    quality_sharp_uncomp_bonus: int = Field(20, description="Score bonus for sharp + uncompressed")
    quality_sharp_ok_threshold: int = Field(300, description="Moderately sharp + uncompressed")
    quality_sharp_ok_bonus: int = Field(10, description="Score bonus for moderately sharp")
    quality_low_threshold: int = Field(50, description="Score < this → 'low quality' Gemini context")
    quality_medium_threshold: int = Field(80, description="Score < this → 'medium quality' Gemini context")

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
