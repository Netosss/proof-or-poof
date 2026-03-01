"""
Unit tests for app/detection/image_detector.py — detect_ai_media_image_logic().

Uses a real tiny JPEG so PIL.Image.open() succeeds.
All external calls (EXIF, Gemini, cache, quality) are mocked.
"""

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from tests.conftest import make_tiny_jpeg

# Rich EXIF from a real iPhone — forensic_score >= 0.60
RICH_EXIF = {
    "Make": "Apple",
    "Model": "iPhone 14 Pro",
    "ISOSpeedRatings": 400,
    "FNumber": 1.78,
    "ExposureTime": 0.008,
    "DateTimeOriginal": "2024:06:15 14:32:00",
    "ColorSpace": 65535,
    "Orientation": 1,
}

AI_EXIF = {"Software": "Stable Diffusion"}

_GEMINI_AI_RESP = {"confidence": 0.92, "explanation": "Synthetic details.", "quality_context": "high"}
_GEMINI_HUMAN_RESP = {"confidence": 0.05, "explanation": "Natural noise.", "quality_context": "high"}
_GEMINI_ERROR = {"confidence": -1.0, "explanation": "Error"}


def _apply_base_patches(stack: ExitStack, exif=None, cached=None, gemini_resp=None):
    """Enter common patches into an ExitStack. Returns mock_gemini if gemini_resp given."""
    if exif is None:
        exif = {}

    stack.enter_context(patch("app.detection.image_detector.get_exif_data", return_value=exif))
    stack.enter_context(patch("app.detection.image_detector.get_image_hash", return_value="img_hash_abc"))
    stack.enter_context(patch("app.detection.image_detector.get_cached_result", return_value=cached))
    stack.enter_context(patch("app.detection.image_detector.set_cached_result"))
    stack.enter_context(patch("app.detection.image_detector.get_quality_context", return_value=("high", 95)))

    if gemini_resp is not None:
        return stack.enter_context(
            patch("app.detection.image_detector.analyze_image_pro_turbo", return_value=gemini_resp)
        )
    return stack.enter_context(patch("app.detection.image_detector.analyze_image_pro_turbo"))


# ---------------------------------------------------------------------------
# Early-exit: high human score
# ---------------------------------------------------------------------------


async def test_high_human_score_early_exit_no_gemini(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(make_tiny_jpeg())

    from app.detection.image_detector import detect_ai_media_image_logic

    with ExitStack() as stack:
        mock_gemini = _apply_base_patches(stack, exif=RICH_EXIF)
        result = await detect_ai_media_image_logic(str(p), {})

    assert result["summary"] == "Likely Authentic"
    assert result["is_short_circuited"] is True
    mock_gemini.assert_not_called()


# ---------------------------------------------------------------------------
# Early-exit: high AI score from metadata
# ---------------------------------------------------------------------------


async def test_ai_metadata_early_exit(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(make_tiny_jpeg())

    from app.detection.image_detector import detect_ai_media_image_logic

    with ExitStack() as stack:
        mock_gemini = _apply_base_patches(stack, exif=AI_EXIF)
        result = await detect_ai_media_image_logic(str(p), {})

    assert "AI" in result["summary"]
    assert result["is_short_circuited"] is True
    mock_gemini.assert_not_called()


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------


async def test_cache_hit_skips_gemini(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(make_tiny_jpeg())

    cached_data = {
        "ai_score": 0.85,
        "explanation": "Previously analysed.",
        "is_gemini_used": True,
        "gpu_time_ms": 0,
        "quality_context": "high",
    }

    from app.detection.image_detector import detect_ai_media_image_logic

    with ExitStack() as stack:
        mock_gemini = _apply_base_patches(stack, exif={}, cached=cached_data)
        result = await detect_ai_media_image_logic(str(p), {})

    assert result.get("is_cached") is True
    mock_gemini.assert_not_called()


# ---------------------------------------------------------------------------
# Ambiguous → Gemini called
# ---------------------------------------------------------------------------


async def test_ambiguous_calls_gemini_ai_result(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(make_tiny_jpeg())

    from app.detection.image_detector import detect_ai_media_image_logic

    with ExitStack() as stack:
        _apply_base_patches(stack, exif={}, cached=None, gemini_resp=_GEMINI_AI_RESP)
        result = await detect_ai_media_image_logic(str(p), {})

    assert result["summary"] == "Likely AI-Generated"
    assert result["is_gemini_used"] is True
    assert result["is_short_circuited"] is False


async def test_ambiguous_calls_gemini_human_result(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(make_tiny_jpeg())

    from app.detection.image_detector import detect_ai_media_image_logic

    with ExitStack() as stack:
        _apply_base_patches(stack, exif={}, cached=None, gemini_resp=_GEMINI_HUMAN_RESP)
        result = await detect_ai_media_image_logic(str(p), {})

    assert result["summary"] == "Likely Authentic"
    assert result["is_gemini_used"] is True


# ---------------------------------------------------------------------------
# Gemini error → Analysis Failed
# ---------------------------------------------------------------------------


async def test_gemini_error_returns_analysis_failed(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(make_tiny_jpeg())

    from app.detection.image_detector import detect_ai_media_image_logic

    with ExitStack() as stack:
        _apply_base_patches(stack, exif={}, cached=None, gemini_resp=_GEMINI_ERROR)
        result = await detect_ai_media_image_logic(str(p), {})

    assert result["summary"] == "Analysis Failed"
