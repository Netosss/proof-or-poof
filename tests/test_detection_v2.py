"""
Tests for the combined-engine forensic detection.

Covers:
- Combined prompt structure (all three perspectives + StudioException + Hollywood exception).
- CombinedDetectionResult schema enforcement (closed-list signal_category + required fields).
- V2→legacy category mapping coverage.
- Image-detector dispatch directly calls the combined engine (no engine flag, no canary,
  no fallback — the old engines were removed in this branch).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.detection import CombinedDetectionResult, V2_TO_LEGACY_CATEGORY
from app.integrations.gemini.prompts_combined import get_combined_prompt


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

class TestCombinedPromptStructure:
    def test_contains_required_xml_sections(self):
        prompt = get_combined_prompt("HIGH quality test context")
        for tag in (
            "<Persona>", "</Persona>",
            "<AnchoredEvidenceRule>", "</AnchoredEvidenceRule>",
            "<StudioException>", "</StudioException>",
            "<LandmarkSignageException>", "</LandmarkSignageException>",
            "<StrictLiability>", "</StrictLiability>",
            "<FocusedRule>", "</FocusedRule>",
            "<OutputFormat>", "</OutputFormat>",
            "<DynamicContext>", "</DynamicContext>",
        ):
            assert tag in prompt, f"missing XML tag: {tag}"

    def test_dynamic_context_is_injected(self):
        prompt = get_combined_prompt("HIGH quality 1080p photograph")
        assert "HIGH quality 1080p photograph" in prompt

    def test_all_three_perspectives_present(self):
        prompt = get_combined_prompt("ctx")
        assert "PERSPECTIVE 1 — ANATOMY" in prompt
        assert "PERSPECTIVE 2 — PHYSICS" in prompt
        assert "PERSPECTIVE 3 — COMPOSITION" in prompt

    def test_exception_blocks_present(self):
        """StudioException + LandmarkSignageException stop FPs on real images."""
        prompt = get_combined_prompt("ctx")
        assert "studio" in prompt.lower()
        assert "Hollywood" in prompt

    def test_all_macro_signal_categories_listed(self):
        prompt = get_combined_prompt("ctx")
        for cat in (
            "peripheral_or_background_structural_collapse",
            "objects_merge_or_dissolve_at_boundaries",
            "geometry_or_perspective_is_physically_impossible",
            "in_scene_text_is_melted_or_gibberish",
            "multiple_subtle_ai_artifacts_present",
            "no_visual_anomalies_detected",
        ):
            assert cat in prompt, f"missing macro signal_category: {cat}"


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------

class TestCombinedDetectionResult:
    def _valid_payload(self) -> dict:
        return {
            "findings": "Clean studio portrait with articulated hands.",
            "region_anchor": "none",
            "confidence": 0.1,
            "signal_category": "no_visual_anomalies_detected",
        }

    def test_valid_payload_parses(self):
        parsed = CombinedDetectionResult(**self._valid_payload())
        assert parsed.confidence == 0.1
        assert parsed.signal_category == "no_visual_anomalies_detected"

    def test_invalid_signal_category_rejected(self):
        payload = self._valid_payload()
        payload["signal_category"] = "totally_made_up_category"
        with pytest.raises(ValidationError):
            CombinedDetectionResult(**payload)

    def test_missing_region_anchor_rejected(self):
        payload = self._valid_payload()
        del payload["region_anchor"]
        with pytest.raises(ValidationError):
            CombinedDetectionResult(**payload)

    def test_missing_findings_rejected(self):
        payload = self._valid_payload()
        del payload["findings"]
        with pytest.raises(ValidationError):
            CombinedDetectionResult(**payload)


# ---------------------------------------------------------------------------
# V2 → legacy category mapping
# ---------------------------------------------------------------------------

class TestV2LegacyMapping:
    def test_all_v2_categories_mapped(self):
        v2_keys = {
            "peripheral_or_background_structural_collapse",
            "objects_merge_or_dissolve_at_boundaries",
            "geometry_or_perspective_is_physically_impossible",
            "in_scene_text_is_melted_or_gibberish",
            "multiple_subtle_ai_artifacts_present",
            "no_visual_anomalies_detected",
        }
        assert v2_keys == set(V2_TO_LEGACY_CATEGORY.keys())

    def test_mapped_values_are_legacy_category_keys(self):
        from app.schemas.detection import SIGNAL_CATEGORIES
        legacy_literals = set(SIGNAL_CATEGORIES.__args__)
        for v2_key, legacy_key in V2_TO_LEGACY_CATEGORY.items():
            assert legacy_key in legacy_literals, (
                f"mapping for {v2_key} -> {legacy_key} is not a valid legacy category"
            )


# ---------------------------------------------------------------------------
# Wiring — image_detector calls combined directly
# ---------------------------------------------------------------------------

class TestCombinedEngineWiring:
    def test_combined_analyzer_importable(self):
        from app.integrations.gemini.client_combined import analyze_image_combined_async
        assert callable(analyze_image_combined_async)

    def test_image_detector_imports_combined(self):
        """No more dispatcher / engine flag — image_detector imports combined directly."""
        import app.detection.image_detector as image_detector
        assert hasattr(image_detector, "analyze_image_combined_async")

    def test_video_frames_analyzer_importable(self):
        """Video pipeline uses the same combined prompt + schema as image."""
        from app.integrations.gemini.client_combined import analyze_video_frames_async
        assert callable(analyze_video_frames_async)


class TestVideoHighConvictionOverride:
    """A single high-confidence anchored frame must flip the verdict to AI
    regardless of the per-frame majority vote — watermark detection in 1-of-3
    frames is essentially proof and should not be averaged out."""

    @pytest.mark.asyncio
    async def test_single_high_anchored_frame_overrides_majority(self, monkeypatch):
        from app.integrations.gemini import client_combined as cc
        from app.config import settings

        monkeypatch.setattr(settings, "video_high_conviction_threshold", 0.90)

        # Build fake parsed results: 1 strong AI hit (anchored watermark)
        # + 2 clean REAL — under the plain vote rule this would lose 1 vs 2.
        class FakeRow:
            def __init__(self, conf, sig, anchor):
                self.confidence = conf
                self.signal_category = sig
                self.region_anchor = anchor

        fake_response = type("R", (), {
            "parsed": [
                FakeRow(0.98, "in_scene_text_is_melted_or_gibberish",
                        "bottom-right corner: 'Generated by AI' caption"),
                FakeRow(0.05, "no_visual_anomalies_detected", "none"),
                FakeRow(0.05, "no_visual_anomalies_detected", "none"),
            ],
            "usage_metadata": None,
        })()

        # Stub the SDK call so we don't hit Gemini
        async def fake_generate_content(*args, **kwargs):
            return fake_response

        monkeypatch.setattr(cc.client.aio.models, "generate_content", fake_generate_content)
        # Stub image prep so we don't need real bytes
        monkeypatch.setattr(cc, "_prepare_pil_for_gemini", lambda src: (src, []))
        monkeypatch.setattr(cc, "_encode_pil_as_jpeg", lambda img, q: b"\xff\xd8\xff\xd9")
        # Stub quality_context — sync, just returns a stub tuple
        monkeypatch.setattr(
            "app.integrations.gemini.quality.get_quality_context",
            lambda src: ("**CONTEXT: HIGH.**", 90),
        )

        result = await cc.analyze_video_frames_async([b"f1", b"f2", b"f3"])
        # Override fired — AI verdict from the high-conviction frame
        assert result["confidence"] >= 0.9
        assert result["signal_category"] == "text_or_signage_contains_gibberish_characters"

    @pytest.mark.asyncio
    async def test_high_but_unanchored_frame_does_not_override(self, monkeypatch):
        """An unanchored 0.95 (model confabulating) must NOT flip the vote."""
        from app.integrations.gemini import client_combined as cc
        from app.config import settings

        monkeypatch.setattr(settings, "video_high_conviction_threshold", 0.90)
        monkeypatch.setattr(settings, "gemini_ai_vote_threshold", 0.55)

        class FakeRow:
            def __init__(self, conf, sig, anchor):
                self.confidence = conf
                self.signal_category = sig
                self.region_anchor = anchor

        fake_response = type("R", (), {
            "parsed": [
                # 0.95 but anchor 'none' — model confabulated, must be ignored by override
                FakeRow(0.95, "objects_merge_or_dissolve_at_boundaries", "none"),
                FakeRow(0.05, "no_visual_anomalies_detected", "none"),
                FakeRow(0.05, "no_visual_anomalies_detected", "none"),
            ],
            "usage_metadata": None,
        })()

        async def fake_generate_content(*args, **kwargs):
            return fake_response

        monkeypatch.setattr(cc.client.aio.models, "generate_content", fake_generate_content)
        monkeypatch.setattr(cc, "_prepare_pil_for_gemini", lambda src: (src, []))
        monkeypatch.setattr(cc, "_encode_pil_as_jpeg", lambda img, q: b"\xff\xd8\xff\xd9")
        monkeypatch.setattr(
            "app.integrations.gemini.quality.get_quality_context",
            lambda src: ("**CONTEXT: HIGH.**", 90),
        )

        result = await cc.analyze_video_frames_async([b"f1", b"f2", b"f3"])
        # Override skipped; vote rule fires: 1 AI vote (0.95 > 0.55) vs 2 REAL → REAL wins
        # (per the existing aggregator: not_ai_votes wins on ties / minority AI)
        assert result["confidence"] < 0.5
