"""Tests for the V2 forensic detection engine.

Covers:
- V2 prompt XML structure & key sections (Persona, AntiAnchoring, StudioException,
  ForensicRules 1–6, OutputFormat with 5 macro categories, examples).
- DetectionResultV2 schema enforcement (closed-list signal_category).
- V2→legacy category mapping coverage.
- Analyzer dispatch wiring (settings.detection_engine routes to v1 vs v2).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.detection import DetectionResultV2
from app.integrations.gemini.prompts_v2 import get_system_instruction_v2
from app.integrations.gemini.client_v2 import V2_TO_LEGACY_CATEGORY


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

class TestV2PromptStructure:
    def test_contains_required_xml_sections(self):
        prompt = get_system_instruction_v2("HIGH quality test context")
        for tag in (
            "<Persona>", "</Persona>",
            "<AntiAnchoring>", "</AntiAnchoring>",
            "<StudioException>", "</StudioException>",
            "<QualityGuard>", "</QualityGuard>",
            "<ForensicRules>", "</ForensicRules>",
            "<OutputFormat>", "</OutputFormat>",
            "<Examples>", "</Examples>",
            "<DynamicContext>", "</DynamicContext>",
        ):
            assert tag in prompt, f"missing XML tag: {tag}"

    def test_dynamic_context_is_injected(self):
        prompt = get_system_instruction_v2("HIGH quality 1080p photograph")
        assert "HIGH quality 1080p photograph" in prompt

    def test_all_six_forensic_rules_present(self):
        prompt = get_system_instruction_v2("ctx")
        # numbered rule headings, all under ForensicRules
        for marker in (
            "1. EXTREMITIES",
            "2. OBJECT BOUNDARIES",
            "3. PHYSICS",
            "4. IN-SCENE TEXT",
            "5. BACKGROUND CLUTTER",
            "6. DIFFUSION FINGERPRINTS",
            "7. SELF-VERIFICATION",
        ):
            assert marker in prompt, f"missing rule marker: {marker}"

    def test_all_macro_signal_categories_listed(self):
        prompt = get_system_instruction_v2("ctx")
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

class TestDetectionResultV2:
    def _valid_payload(self) -> dict:
        return {
            "step_1_edge_and_background_scan": "Hands fully articulated; distant background subjects coherent.",
            "step_2_physics_and_boundary_scan": "Single overhead light source; consistent shadows; no boundary fusion.",
            "visual_scan": "Clean photograph with consistent lighting and articulated extremities.",
            "confidence": 0.1,
            "signal_category": "no_visual_anomalies_detected",
        }

    def test_valid_payload_parses(self):
        parsed = DetectionResultV2(**self._valid_payload())
        assert parsed.confidence == 0.1
        assert parsed.signal_category == "no_visual_anomalies_detected"

    def test_invalid_signal_category_rejected(self):
        payload = self._valid_payload()
        payload["signal_category"] = "totally_made_up_category"
        with pytest.raises(ValidationError):
            DetectionResultV2(**payload)

    def test_missing_cot_step_rejected(self):
        payload = self._valid_payload()
        del payload["step_1_edge_and_background_scan"]
        with pytest.raises(ValidationError):
            DetectionResultV2(**payload)


# ---------------------------------------------------------------------------
# Category mapping
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

    def test_mapped_values_are_legacy_keys(self):
        from app.schemas.detection import SIGNAL_CATEGORIES
        legacy_literals = set(SIGNAL_CATEGORIES.__args__)
        for v2_key, legacy_key in V2_TO_LEGACY_CATEGORY.items():
            assert legacy_key in legacy_literals, (
                f"mapping for {v2_key} -> {legacy_key} is not a valid legacy category"
            )


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------

class TestEngineDispatch:
    def test_v1_default_selects_v1_analyzer(self, monkeypatch):
        from app.detection import image_detector
        from app.integrations.gemini import client as gemini_v1

        monkeypatch.setattr(image_detector.settings, "detection_engine", "v1")
        assert image_detector._select_analyzer() is gemini_v1.analyze_image_pro_turbo

    def test_v2_flag_selects_v2_analyzer(self, monkeypatch):
        from app.detection import image_detector
        from app.integrations.gemini import client_v2

        monkeypatch.setattr(image_detector.settings, "detection_engine", "v2")
        assert image_detector._select_analyzer() is client_v2.analyze_image_pro_turbo_v2
