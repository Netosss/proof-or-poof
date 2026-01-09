
import pytest
import os
import random
from unittest.mock import MagicMock, patch
from app.detectors import detect_ai_media

# Path to test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_test_files(category, subcategory):
    """Helper to get all files in a specific test directory."""
    dir_path = os.path.join(TEST_DATA_DIR, category, subcategory)
    if not os.path.exists(dir_path):
        return []
    return [
        os.path.join(dir_path, f) 
        for f in os.listdir(dir_path) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.mov'))
    ]

# --- MOCKING ---

async def mock_run_deep_forensics(file_path, width, height):
    """Mock GPU inference for images."""
    if "/ai/" in file_path:
        return {"ai_score": random.uniform(0.85, 0.99), "gpu_time_ms": 100}
    elif "/original/" in file_path:
        return {"ai_score": random.uniform(0.01, 0.15), "gpu_time_ms": 100}
    # Fallback for screenshot/screen_record paths if they don't have "ai" or "original" in the path string explicitly 
    # (but they will based on directory structure)
    return {"ai_score": 0.5, "gpu_time_ms": 100}

# --- TESTS ---

@pytest.mark.asyncio
async def test_detect_pure_ai_images():
    """Test AI images are detected as AI."""
    files = get_test_files("ai", "images")
    if not files:
        pytest.skip("No AI images found in tests/data/ai/images")

    with patch("app.detectors.core.run_deep_forensics", side_effect=mock_run_deep_forensics):
        for fpath in files:
            result = await detect_ai_media(fpath)
            summary = result["summary"]
            assert "AI" in summary or "Suspicious" in summary, f"Failed on {os.path.basename(fpath)}"

@pytest.mark.asyncio
async def test_detect_pure_original_videos():
    """Test Original videos are detected as Original."""
    files = get_test_files("original", "videos")
    if not files:
        pytest.skip("No Original videos found in tests/data/original/videos")

    mock_batch_result = {
        "results": [{"ai_score": random.uniform(0.01, 0.10)} for _ in range(3)],
        "gpu_time_ms": 150
    }

    with patch("app.detectors.core.run_batch_forensics", return_value=mock_batch_result):
        for fpath in files:
            result = await detect_ai_media(fpath)
            summary = result["summary"]
            assert "Original" in summary, f"Failed on {os.path.basename(fpath)}"

@pytest.mark.asyncio
async def test_detect_pure_ai_videos():
    """Test AI videos are detected as AI."""
    files = get_test_files("ai", "videos")
    if not files:
        pytest.skip("No AI videos found in tests/data/ai/videos")

    mock_batch_result = {
        "results": [{"ai_score": random.uniform(0.90, 0.99)} for _ in range(3)],
        "gpu_time_ms": 150
    }

    with patch("app.detectors.core.run_batch_forensics", return_value=mock_batch_result):
        for fpath in files:
            result = await detect_ai_media(fpath)
            summary = result["summary"]
            assert "AI" in summary, f"Failed on {os.path.basename(fpath)}"

@pytest.mark.asyncio
async def test_detect_screenshots_original():
    """Test Original Screenshots (should rely on GPU to confirm Original/Human)."""
    files = get_test_files("screenshots", "original")
    if not files:
        pytest.skip("No Original screenshots found")

    # Mock Low AI score (Original)
    with patch("app.detectors.core.run_deep_forensics", side_effect=mock_run_deep_forensics):
        for fpath in files:
            result = await detect_ai_media(fpath)
            summary = result["summary"]
            # Should be "Likely Original" even if metadata is missing/suspicious
            assert "Original" in summary or "Human" in summary, f"Failed: {summary}" 

@pytest.mark.asyncio
async def test_detect_screenshots_ai():
    """Test AI Screenshots."""
    files = get_test_files("screenshots", "ai")
    if not files:
        pytest.skip("No AI screenshots found")

    with patch("app.detectors.core.run_deep_forensics", side_effect=mock_run_deep_forensics):
        for fpath in files:
            result = await detect_ai_media(fpath)
            summary = result["summary"]
            assert "AI" in summary, f"Failed: {summary}"

@pytest.mark.asyncio
async def test_detect_screen_records_original():
    """Test Original Screen Records."""
    files = get_test_files("screen_records", "original")
    if not files:
        pytest.skip("No Original screen records found")
        
    mock_batch_result = {
        "results": [{"ai_score": random.uniform(0.01, 0.10)} for _ in range(3)],
        "gpu_time_ms": 150
    }

    with patch("app.detectors.core.run_batch_forensics", return_value=mock_batch_result):
        for fpath in files:
            result = await detect_ai_media(fpath)
            summary = result["summary"]
            assert "Original" in summary, f"Failed: {summary}"

@pytest.mark.asyncio
async def test_detect_screen_records_ai():
    """Test AI Screen Records."""
    files = get_test_files("screen_records", "ai")
    if not files:
        pytest.skip("No AI screen records found")
        
    mock_batch_result = {
        "results": [{"ai_score": random.uniform(0.90, 0.99)} for _ in range(3)],
        "gpu_time_ms": 150
    }

    with patch("app.detectors.core.run_batch_forensics", return_value=mock_batch_result):
        for fpath in files:
            result = await detect_ai_media(fpath)
            summary = result["summary"]
            assert "AI" in summary, f"Failed: {summary}"

