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
    # Rule 10 — Architectural / scene composition
    "scene_composition_is_synthetically_uniform",
    # Rule 11 — Color grading
    "unnatural_color_grading_or_saturation_detected",
    # Rule 12 — Lens optics
    "lens_optics_absence_detected",
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
    # 32-char hex UUID issued by the /detect route. Clients poll
    # /detect/progress/{task_id} during the wait to drive the stage UI.
    task_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Macro forensic categories returned by the combined Gemini schema. Mapped
# back to the legacy 19-category strings (SIGNAL_CATEGORIES above) by
# V2_TO_LEGACY_CATEGORY for the user-facing response shape.
# ---------------------------------------------------------------------------
SIGNAL_CATEGORIES_V2 = Literal[
    "peripheral_or_background_structural_collapse",
    "objects_merge_or_dissolve_at_boundaries",
    "geometry_or_perspective_is_physically_impossible",
    "in_scene_text_is_melted_or_gibberish",
    "multiple_subtle_ai_artifacts_present",
    "no_visual_anomalies_detected",
]


# Mapping from combined-schema macro buckets → legacy 19-category strings
# used by image_detector._label_for and the downstream API response shape.
# Keeps the Gemini-side category space small (5 macros) without breaking
# downstream consumers that expect the legacy taxonomy.
V2_TO_LEGACY_CATEGORY: dict[str, str] = {
    "peripheral_or_background_structural_collapse": "facial_detail_inconsistency_detected",
    "objects_merge_or_dissolve_at_boundaries": "objects_merge_or_dissolve_at_boundaries",
    "geometry_or_perspective_is_physically_impossible": "geometry_or_perspective_is_physically_impossible",
    "in_scene_text_is_melted_or_gibberish": "text_or_signage_contains_gibberish_characters",
    "multiple_subtle_ai_artifacts_present": "multiple_subtle_ai_artifacts_present",
    "no_visual_anomalies_detected": "no_visual_anomalies_detected",
}


class CombinedDetectionResult(BaseModel):
    """
    Forensic verdict from the combined Gemini call.

    Single response covering all three forensic perspectives (anatomy,
    physics, composition). `region_anchor` enforces structural anti-
    anchoring: any AI verdict (confidence >= 0.5) must point to a specific
    named image region — combats the model's tendency to confabulate
    authenticity claims without evidence.

    `content_plausibility` is decoded FIRST: a semantic read of the signage
    that catches fabricated/AI concept ads whose tell is the offer itself
    (an impossible-to-deliver or non-existent product), not a pixel artifact.
    """
    content_plausibility: str = Field(
        description=(
            "FIRST: read ALL visible text/signage/prices VERBATIM, then judge ONLY whether the "
            "offer's OWN STATED TERMS cohere as a real purchasable thing. This tests the WORDS of "
            "the offer — NOT the pixels, and NOT who the seller is. WHO sells WHAT is irrelevant: "
            "any seller may stock anything, run any stunt, giveaway, novelty, pop-up, or "
            "regional/satirical product. 'Surprising for this seller' is ALWAYS CONTENT_PLAUSIBLE.\n"
            "Apply THREE clauses and EMIT each (with a reason) BEFORE the verdict:\n"
            "  clause1 DELIVERABILITY — could a real seller physically hand the buyer this good "
            "through the depicted channel? An intangible/digital-only service (compute, API "
            "tokens, cloud storage, bandwidth, a subscription tier) presented as a boxed/shelf/"
            "checkout retail item CANNOT be handed over => clause1=FALSE. A tangible good (even a "
            "vehicle, aircraft, or gold bar) CAN => clause1=TRUE.\n"
            "  clause2 REFERENT — does every named product, version, model, and spec actually "
            "exist and use a coherent unit? A non-existent version/edition, an impossible spec, or "
            "a spec stated in a unit that cannot measure that thing => clause2=FALSE. Real "
            "products in coherent units => clause2=TRUE.\n"
            "  clause3 RETAILABILITY — could an ordinary private consumer COMPLETE this as a real "
            "retail purchase? The principle is a completable consumer transaction, NOT 'unusual "
            "for this store'. A FUNCTIONAL weapon-of-war or military munition — guided missiles, "
            "rockets, launchers, air-defense / interceptor systems, ordnance, warheads, military "
            "armored vehicles, combat aircraft — or another good no retailer can lawfully sell to "
            "a civilian (restricted hazmat, human organs, protected wildlife), offered FOR SALE to "
            "a private individual is a transaction no retailer can complete => clause3=FALSE. "
            "clause3=FALSE ONLY on a genuine CONSUMER RETAIL OFFER (a price / for-sale-to-you term "
            "on the item itself). FENCE — these are clause3=TRUE (NOT a violation): firearms and "
            "ammunition (lawful consumer goods in many markets); toy / replica / airsoft / "
            "paintball / scale-model weapons; deactivated, demilitarized, surplus, or collectible "
            "militaria; movie/theatrical props; consumer fireworks. A military item shown on "
            "DISPLAY with NO consumer price — museum, monument, gate-guardian, airshow, "
            "defense-expo / B2B booth, or news/editorial/war photo — is NOT a consumer offer => "
            "clause3=TRUE (and the verdict falls to NO_CLAIM absent other commercial text). Judge "
            "clause3 on the OFFER, never on the seller.\n"
            "Output EXACTLY: \"clause1=<TRUE/FALSE: reason>; clause2=<TRUE/FALSE: reason>; "
            "clause3=<TRUE/FALSE: reason> => <verdict>\". Verdict is 'ABSURD_CONTENT' if ANY clause "
            "is FALSE, 'CONTENT_PLAUSIBLE' if ALL THREE are TRUE, or 'NO_CLAIM' if there is no "
            "commercial offer text.\n"
            "OCR FENCE: if any cited text is unreadable, low-resolution, in a script you cannot "
            "fully parse, or you are INFERRING rather than reading it verbatim => NO_CLAIM. Most "
            "candid photos, portraits, and personal snapshots are NO_CLAIM."
        )
    )
    findings: str = Field(
        description="Brief description of the strongest forensic signal observed. Max 40 words."
    )
    region_anchor: str = Field(
        description=(
            "The SPECIFIC, NAMED image region where the strongest anomaly is "
            "observed. E.g. 'left temple', 'upper-left corner', 'subject's right "
            "armpit', 'background near the AC unit'. Use 'none' ONLY when the "
            "image appears authentic (confidence < 0.5) — any AI verdict MUST "
            "be anchored to a named region."
        ),
    )
    confidence: float = Field(
        description=(
            "0.0 (clearly real) to 1.0 (clearly AI). "
            "CONTENT OVERRIDE: if content_plausibility ends in ABSURD_CONTENT (ANY of clause1 / "
            "clause2 / clause3 is FALSE — judged on the offer's stated terms read verbatim, NOT on "
            "who the seller is), confidence MUST be >= 0.70 and you MUST set region_anchor to the "
            "sign/claim location (never 'none'). If all three clauses are TRUE, content does not "
            "affect confidence."
        )
    )
    signal_category: SIGNAL_CATEGORIES_V2 = Field(
        description="Exactly one macro forensic bucket from the closed list."
    )
