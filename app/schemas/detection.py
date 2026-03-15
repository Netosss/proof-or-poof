from pydantic import BaseModel, Field
from typing import Optional, List, Literal


# ---------------------------------------------------------------------------
# Closed-list of forensic signal categories.
# Gemini picks exactly one; we own the user-facing copy via SIGNAL_CATEGORY_LABELS.
# Adding a new category here requires a matching entry in image_detector.py.
# ---------------------------------------------------------------------------
SIGNAL_CATEGORIES = Literal[
    # Rule 1 — Watermark / Content Credentials
    "watermark_or_content_credentials_detected",
    # Rule 2 — Photorealism artifacts
    "skin_or_surface_texture_is_waxy_or_plastic",
    "objects_merge_or_dissolve_at_boundaries",
    "geometry_or_perspective_is_physically_impossible",
    # Rule 3 — Text
    "text_or_signage_contains_gibberish_characters",
    # Rule 4 — Lighting / Physics
    "shadow_directions_conflict_with_light_source",
    "foreground_lit_despite_backlit_light_source",
    # Rule 7 — Environmental / Outdoor physics
    "environmental_physics_inconsistency",
    # Rule 5 — Portrait / Fabric / Hair
    "clothing_or_fabric_lacks_3d_depth",
    "hair_or_fur_bleeds_into_background",
    "glass_or_small_objects_lack_3d_structure",
    # Noise forensic hint
    "uniform_diffusion_noise_pattern_detected",
    # Rule 8 — Facial micro-details (HIGH quality only)
    "facial_detail_inconsistency_detected",
    # Rule 9 — Repetition / cloning
    "repeated_or_cloned_elements_detected",
    # Catch-all for multi-signal or ambiguous cases
    "multiple_subtle_ai_artifacts_present",
    # Clean image
    "no_visual_anomalies_detected",
]


class EvidenceItem(BaseModel):
    layer: str      # e.g., "Metadata Check", "Technical Analysis", "Deep Forensics"
    status: str     # "passed" (Green), "warning" (Yellow), "flagged" (Red), "info" (Neutral)
    label: str      # Short title e.g., "Digital Signature"
    detail: str     # User-facing explanation e.g., "Valid camera metadata found."


class DetectionResponse(BaseModel):
    summary: str
    confidence_score: float
    new_balance: Optional[int] = None
    is_short_circuited: bool = False
    evidence_chain: List[EvidenceItem]
    short_id: Optional[str] = None


class DetectionResult(BaseModel):
    """Gemini structured output schema — one result per image frame."""
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    signal_category: SIGNAL_CATEGORIES = Field(
        description=(
            "The single primary forensic signal that determined the verdict. "
            "Must be one of the exact enum keys listed in the system instructions."
        )
    )
