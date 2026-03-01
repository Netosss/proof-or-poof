"""
Unit tests for app/detection/pipeline.py — detect_ai_media().

All external I/O is mocked: C2PA, hashing, cache, video metadata, Gemini, image logic.
A tiny real file is used so path/extension checks pass.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AI_MANIFEST = {
    "claim_generator_info": [{"name": "DALL-E"}],
    "assertions": [
        {
            "label": "c2pa.actions.v2",
            "data": {
                "actions": [
                    {"digitalSourceType": "trainedAlgorithmicMedia"}
                ]
            },
        }
    ],
}

HUMAN_MANIFEST = {
    "claim_generator_info": [{"name": "Adobe Camera Raw"}],
    "assertions": [],  # no AI actions
}

_GEMINI_AI = {"confidence": 0.9, "explanation": "Synthetic textures detected.", "quality_context": "high"}
_GEMINI_HUMAN = {"confidence": 0.1, "explanation": "Natural noise patterns.", "quality_context": "high"}

_IMAGE_RESULT = {
    "summary": "Likely Authentic",
    "confidence_score": 0.99,
    "is_short_circuited": True,
    "evidence_chain": [],
}


# ---------------------------------------------------------------------------
# C2PA layer
# ---------------------------------------------------------------------------


async def test_c2pa_ai_manifest_short_circuits(tmp_path):
    fake_file = tmp_path / "photo.jpg"
    fake_file.write_bytes(b"fake_jpeg")

    with patch("app.detection.pipeline.get_c2pa_manifest", return_value=AI_MANIFEST):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    assert result["summary"] == "AI-Generated"
    assert result["confidence_score"] == 1.0
    assert result["is_short_circuited"] is True


async def test_c2pa_human_manifest_short_circuits(tmp_path):
    fake_file = tmp_path / "photo.jpg"
    fake_file.write_bytes(b"fake_jpeg")

    with patch("app.detection.pipeline.get_c2pa_manifest", return_value=HUMAN_MANIFEST):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    assert result["summary"] == "No AI Detected"
    assert result["confidence_score"] == 1.0
    assert result["is_short_circuited"] is True


# ---------------------------------------------------------------------------
# Video path
# ---------------------------------------------------------------------------


async def test_video_cache_hit_returns_immediately(tmp_path):
    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"fake_video")
    cached = {"summary": "Likely AI-Generated", "confidence_score": 0.92, "is_cached": True}

    with (
        patch("app.detection.pipeline.get_c2pa_manifest", return_value=None),
        patch("app.detection.pipeline.get_smart_file_hash", return_value="hash123"),
        patch("app.detection.pipeline.get_cached_result", return_value=cached),
    ):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    assert result["summary"] == "Likely AI-Generated"
    assert result.get("is_cached") is True


async def test_video_early_exit_human(tmp_path):
    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"fake_video")

    with (
        patch("app.detection.pipeline.get_c2pa_manifest", return_value=None),
        patch("app.detection.pipeline.get_smart_file_hash", return_value="hash_video"),
        patch("app.detection.pipeline.get_cached_result", return_value=None),
        patch("app.detection.pipeline.get_video_metadata", new_callable=AsyncMock, return_value={}),
        patch(
            "app.detection.pipeline.get_video_metadata_score",
            return_value=(0.95, 0.05, ["Camera model: GoPro"], "human"),
        ),
        patch("app.detection.pipeline.set_cached_result"),
    ):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    assert result["summary"] == "No AI Detected"
    assert result["is_short_circuited"] is True


async def test_video_early_exit_ai(tmp_path):
    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"fake_video")

    with (
        patch("app.detection.pipeline.get_c2pa_manifest", return_value=None),
        patch("app.detection.pipeline.get_smart_file_hash", return_value="hash_aivid"),
        patch("app.detection.pipeline.get_cached_result", return_value=None),
        patch("app.detection.pipeline.get_video_metadata", new_callable=AsyncMock, return_value={}),
        patch(
            "app.detection.pipeline.get_video_metadata_score",
            return_value=(0.0, 0.99, ["Sora v1 detected"], "ai"),
        ),
        patch("app.detection.pipeline.set_cached_result"),
    ):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    assert result["summary"] == "AI-Generated"
    assert result["is_short_circuited"] is True


async def test_video_gemini_high_confidence(tmp_path):
    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"fake_video")
    fake_frames = [MagicMock()]

    with (
        patch("app.detection.pipeline.get_c2pa_manifest", return_value=None),
        patch("app.detection.pipeline.get_smart_file_hash", return_value="hash_gem"),
        patch("app.detection.pipeline.get_cached_result", return_value=None),
        patch("app.detection.pipeline.get_video_metadata", new_callable=AsyncMock, return_value={}),
        patch(
            "app.detection.pipeline.get_video_metadata_score",
            return_value=(0.1, 0.1, [], None),
        ),
        patch(
            "app.detection.pipeline.extract_video_frames",
            return_value=(fake_frames, 0),
        ),
        patch(
            "app.detection.pipeline.analyze_batch_images_pro_turbo",
            return_value=_GEMINI_AI,
        ),
        patch("app.detection.pipeline.set_cached_result"),
    ):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    assert result["summary"] == "Likely AI-Generated"
    assert result["is_gemini_used"] is True


async def test_video_frame_extraction_fails(tmp_path):
    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"fake_video")

    with (
        patch("app.detection.pipeline.get_c2pa_manifest", return_value=None),
        patch("app.detection.pipeline.get_smart_file_hash", return_value="hash_fail"),
        patch("app.detection.pipeline.get_cached_result", return_value=None),
        patch("app.detection.pipeline.get_video_metadata", new_callable=AsyncMock, return_value={}),
        patch(
            "app.detection.pipeline.get_video_metadata_score",
            return_value=(0.1, 0.1, [], None),
        ),
        patch("app.detection.pipeline.extract_video_frames", return_value=([], 0)),
    ):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    assert result["summary"] == "Analysis Failed"


# ---------------------------------------------------------------------------
# Image path — delegates to image_detector
# ---------------------------------------------------------------------------


async def test_image_path_delegates_to_image_logic(tmp_path):
    fake_file = tmp_path / "photo.jpg"
    fake_file.write_bytes(b"fake_jpeg")

    with (
        patch("app.detection.pipeline.get_c2pa_manifest", return_value=None),
        patch(
            "app.detection.pipeline.detect_ai_media_image_logic",
            new_callable=AsyncMock,
            return_value=_IMAGE_RESULT,
        ) as mock_image_logic,
    ):
        from app.detection.pipeline import detect_ai_media

        result = await detect_ai_media(str(fake_file))

    mock_image_logic.assert_called_once()
    assert result["summary"] == "Likely Authentic"
