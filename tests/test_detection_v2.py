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
            "7. STRICT LIABILITY",
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
            "scan_hands_and_boundaries": "Hands fully articulated with clear knuckles; no fusion with objects.",
            "scan_background_and_physics": "Single overhead light source; consistent shadows; no boundary fusion.",
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
        del payload["scan_hands_and_boundaries"]
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

class TestBoostScoreCap:
    def test_ai_boost_capped_at_099(self):
        from app.detection.image_detector import boost_score
        assert boost_score(0.99, is_ai_likely=True) <= 0.99
        assert boost_score(1.0, is_ai_likely=True) == 0.99

    def test_authentic_path_also_capped(self):
        from app.detection.image_detector import boost_score
        assert boost_score(1.0, is_ai_likely=False) == 0.99

    def test_low_ai_score_boosted_but_under_cap(self):
        from app.detection.image_detector import boost_score
        out = boost_score(0.55, is_ai_likely=True)
        assert 0.55 < out < 0.99


class TestGeminiResponseBuilder:
    def test_emits_two_row_evidence_chain(self):
        from app.detection.image_detector import _build_gemini_evidence_response
        resp = _build_gemini_evidence_response(
            summary="Likely AI-Generated", confidence=0.83,
            is_ai_likely=True, visual_detail="merged earring",
            context_quality="HIGH", is_gemini_used=True, is_cached=False,
        )
        assert resp["summary"] == "Likely AI-Generated"
        assert resp["confidence_score"] == 0.83
        assert resp["is_cached"] is False
        assert len(resp["evidence_chain"]) == 2
        assert resp["evidence_chain"][1]["status"] == "flagged"
        assert resp["evidence_chain"][1]["context_quality"] == "HIGH"

    def test_authentic_path_emits_passed_status(self):
        from app.detection.image_detector import _build_gemini_evidence_response
        resp = _build_gemini_evidence_response(
            summary="Likely Authentic", confidence=0.92,
            is_ai_likely=False, visual_detail="clean scan",
            context_quality="MEDIUM", is_gemini_used=True, is_cached=True,
        )
        assert resp["evidence_chain"][1]["status"] == "passed"


class TestCacheKeyNamespacedByEngine:
    def test_prefix_includes_engine(self, monkeypatch):
        from app.detection import cache as cache_module
        monkeypatch.setattr(cache_module.settings, "detection_engine", "v1")
        assert cache_module._cache_prefix() == "forensic_v2:v1:"
        monkeypatch.setattr(cache_module.settings, "detection_engine", "v2")
        assert cache_module._cache_prefix() == "forensic_v2:v2:"


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
