"""
Unit tests for _compute_noise_cv() in app/integrations/gemini/client.py.

All tests use synthetic PIL images — no file I/O, no Gemini API calls.
"""

import io
import random
import pytest
from PIL import Image

# conftest.py stubs GEMINI_API_KEY before any import, so client.py is safe to import.
from app.integrations.gemini.client import _compute_noise_cv


def _solid_rgb(size=(100, 100), color=(128, 128, 128)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _noisy_rgb(size=(100, 100), seed=42) -> Image.Image:
    """Random pixel values — high spatial noise variance."""
    rng = random.Random(seed)
    pixels = [
        (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        for _ in range(size[0] * size[1])
    ]
    img = Image.new("RGB", size)
    img.putdata(pixels)
    return img


def _gradient_rgb(size=(100, 100)) -> Image.Image:
    """Left-to-right gradient — smooth, low-noise real-photo-like image."""
    w, h = size
    pixels = [
        (int(x / w * 255), int(x / w * 200), 100)
        for y in range(h)
        for x in range(w)
    ]
    img = Image.new("RGB", size)
    img.putdata(pixels)
    return img


# ---------------------------------------------------------------------------
# Small-image guard (< 8px in either dimension → return 0.0 immediately)
# ---------------------------------------------------------------------------


def test_tiny_image_1x1_returns_zero():
    img = _solid_rgb(size=(1, 1))
    assert _compute_noise_cv(img) == 0.0


def test_tiny_image_7x7_returns_zero():
    img = _solid_rgb(size=(7, 7))
    assert _compute_noise_cv(img) == 0.0


def test_image_exactly_8x8_does_not_return_zero():
    """8×8 is the minimum size that passes the guard — must run the full path."""
    img = _solid_rgb(size=(8, 8))
    result = _compute_noise_cv(img)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Return-type smoke test
# ---------------------------------------------------------------------------


def test_returns_float_for_normal_image():
    img = _solid_rgb(size=(100, 100))
    result = _compute_noise_cv(img)
    assert isinstance(result, float)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# Behavioural correctness
# ---------------------------------------------------------------------------


def test_uniform_gray_image_returns_near_zero():
    """Perfectly flat image → residual is zero everywhere → CV ≈ 0."""
    img = _solid_rgb(size=(100, 100), color=(128, 128, 128))
    result = _compute_noise_cv(img)
    assert result < 0.01


def test_noisy_image_returns_positive_cv():
    """Random pixel noise → non-trivial spatial variance → CV > 0."""
    img = _noisy_rgb(size=(100, 100))
    result = _compute_noise_cv(img)
    assert result > 0.0


def test_non_rgb_mode_handled():
    """RGBA and L-mode inputs must not raise — the function converts internally."""
    rgba = Image.new("RGBA", (100, 100), color=(100, 150, 200, 255))
    result = _compute_noise_cv(rgba)
    assert isinstance(result, float)

    gray = Image.new("L", (100, 100), color=128)
    result2 = _compute_noise_cv(gray)
    assert isinstance(result2, float)
