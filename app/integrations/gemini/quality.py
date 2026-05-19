"""
Image quality analysis for Gemini context generation.

Analyzes image quality using a weighted scoring system (DQT, Resolution, Blur)
and produces a human-readable context string that is injected into the Gemini prompt.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
from typing import Union

from app.config import settings


def get_quality_context(image_source: Union[str, Image.Image, bytes]) -> tuple[str, int]:
    """
    Analyzes image quality using a weighted scoring system (DQT, Resolution, Blur).
    Returns a tuple of (context string for Gemini, quality score).
    """
    score = settings.quality_score_init
    issues = []
    avg_q_val = 0

    try:
        # 1. Standardize Input (Handle Path vs PIL vs Bytes)
        if isinstance(image_source, str):
            try:
                img_pil = Image.open(image_source)
            except Exception:
                return "**CONTEXT: QUALITY UNKNOWN (Could not open).**", 0
            was_path = True
            filename = os.path.basename(image_source)
        elif isinstance(image_source, bytes):
            try:
                img_pil = Image.open(io.BytesIO(image_source))
            except Exception:
                return "**CONTEXT: QUALITY UNKNOWN (Could not decode bytes).**", 0
            was_path = True  # Treat as temp object that needs closing
            filename = "Video Frame Bytes"
        else:
            img_pil = image_source
            was_path = False
            filename = "Image Object"

        # --- LAYER 1: DQT (JPEG Artifacts) ---
        try:
            if hasattr(img_pil, 'quantization') and img_pil.quantization:
                table = img_pil.quantization.get(0)  # Luminance table
                if table:
                    avg_q_val = sum(table) / len(table)
                    if avg_q_val > settings.quality_dqt_severe_threshold:
                        score -= settings.quality_dqt_severe_penalty
                        issues.append(f"Severe Compression (~Q{int(100 - avg_q_val*2)})")
                    elif avg_q_val > settings.quality_dqt_moderate_threshold:
                        score -= settings.quality_dqt_moderate_penalty
                        issues.append(f"Compression Artifacts (~Q{int(100 - avg_q_val*2)})")
        except Exception:
            pass

        # --- LAYER 2: PIXEL ANALYSIS (Resolution & Blur) ---
        try:
            img_arr = np.array(img_pil.convert('RGB'))
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

            h, w = gray.shape
            pixels = h * w

            # Resolution Penalties
            if pixels < settings.quality_pixels_tiny:
                score -= settings.quality_pixels_tiny_penalty
                issues.append(f"Tiny Resolution ({w}x{h})")
            elif pixels < settings.quality_pixels_small:
                score -= settings.quality_pixels_small_penalty
                issues.append(f"Low Resolution ({w}x{h})")

            # Blur Penalties (Laplacian Variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < settings.quality_blur_zero_threshold:
                score -= settings.quality_blur_zero_penalty
                issues.append(f"Near Zero Detail (Score: {blur_score:.0f})")
            elif blur_score < settings.quality_blur_extreme_threshold:
                score -= settings.quality_blur_extreme_penalty
                issues.append(f"Extreme Blur (Score: {blur_score:.0f})")
            elif blur_score < settings.quality_blur_soft_threshold:
                score -= settings.quality_blur_soft_penalty
                issues.append(f"Soft Focus/Smooth (Score: {blur_score:.0f})")
            elif blur_score > settings.quality_sharp_high_threshold:
                score += settings.quality_sharp_high_bonus
                issues.append(f"Exceptional Sharpness (Score: {blur_score:.0f})")
            elif blur_score > settings.quality_sharp_med_threshold:
                score += settings.quality_sharp_med_bonus
                issues.append(f"High Sharpness (Score: {blur_score:.0f})")
            elif blur_score > settings.quality_sharp_uncomp_threshold and avg_q_val < settings.quality_dqt_moderate_threshold:
                score += settings.quality_sharp_uncomp_bonus
                issues.append(f"High Sharpness (Score: {blur_score:.0f})")
            elif blur_score > settings.quality_sharp_ok_threshold and avg_q_val < settings.quality_dqt_moderate_threshold:
                score += settings.quality_sharp_ok_bonus

        except Exception:
            pass

        if was_path:
            img_pil.close()

        # --- FINAL VERDICT ---
        score = max(0, score)

        if score < settings.quality_low_threshold:
            return (
                f"**CRITICAL CONTEXT: LOW QUALITY IMAGE ({', '.join(issues)}).** "
                f"INSTRUCTION: This image is heavily compressed/low-res. "
                f"Warning: This quality naturally causes **WAXY SKIN**, **SMOOTH TEXTURES**, distorted faces, and blur. "
                f"**IGNORE** any 'waxy' or 'smooth' appearance—it is due to low quality, not AI. "
                f"Only flag **STRUCTURAL IMPOSSIBILITIES** that compression CANNOT explain "
                f"(e.g., gibberish text, extra limbs). If unsure, assume it is compression.",
                score
            )
        elif score < settings.quality_medium_threshold:
            return (
                f"**CONTEXT: MEDIUM QUALITY ({', '.join(issues)}).** "
                f"INSTRUCTION: Image has reduced quality. DO NOT flag soft edges or indistinct textures as AI. "
                f"However, compression does NOT explain structural impossibilities "
                f"(like extra fingers, gibberish text, or impossible physics). "
                f"If you see these CLEAR logic errors, FLAG THEM.",
                score
            )

        return "**CONTEXT: HIGH QUALITY IMAGE.** Image is sharp and clean. Any visual anomaly is likely a sign of manipulation.", score

    except Exception:
        return "**CONTEXT: QUALITY UNKNOWN.** Proceed with standard analysis.", 0
