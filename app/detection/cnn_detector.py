"""
CNN/ViT AI-image classifier — supplementary vote for the ensemble engine.

Uses Organika/sdxl-detector (ViT-based binary classifier trained on SDXL output
plus negatives). Empirically still useful as a second opinion on Midjourney v6
and Flux-class outputs despite the training-data distribution gap noted in the
architect's analysis — its failure modes are orthogonal to Gemini's, so any
non-trivial confidence adds genuinely new signal.

Model is lazy-loaded on first inference call to keep import-time cheap. The
processor + model are cached in module-level globals so subsequent calls reuse
the warmed state. Inference is CPU-only; on Railway's container this runs in
~300–500ms per image, which is well under the parallel-budget of an ensemble
call.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


# Module-level singleton — initialised on first call, then reused.
_MODEL = None
_PROCESSOR = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED = False

_MODEL_ID = "Organika/sdxl-detector"


def _load() -> bool:
    """Load processor + model once, thread-safe. Returns False if load failed."""
    global _MODEL, _PROCESSOR, _LOAD_FAILED

    if _MODEL is not None:
        return True
    if _LOAD_FAILED:
        return False

    with _LOAD_LOCK:
        if _MODEL is not None:
            return True
        if _LOAD_FAILED:
            return False

        try:
            # Imports are inside the lock so a missing dependency doesn't
            # crash the entire process at module-import time.
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch

            logger.info("cnn_detector_loading", extra={
                "action": "cnn_detector_loading",
                "model": _MODEL_ID,
            })

            _PROCESSOR = AutoImageProcessor.from_pretrained(_MODEL_ID)
            _MODEL = AutoModelForImageClassification.from_pretrained(_MODEL_ID)
            _MODEL.eval()
            # CPU-only; matches the project's Railway deployment profile.
            _MODEL.to("cpu")

            logger.info("cnn_detector_ready", extra={
                "action": "cnn_detector_ready",
                "model": _MODEL_ID,
                "id2label": _MODEL.config.id2label,
            })
            return True
        except Exception as exc:
            _LOAD_FAILED = True
            logger.warning("cnn_detector_load_failed", extra={
                "action": "cnn_detector_load_failed",
                "model": _MODEL_ID,
                "error": str(exc),
                "error_type": type(exc).__name__,
            })
            return False


def predict_ai_probability(image: Image.Image | str | bytes) -> Optional[float]:
    """
    Run the CNN classifier. Returns the probability the image is AI [0.0, 1.0]
    or None if the model is unavailable / inference failed. Callers must treat
    None as "this voter abstains" — never as "real".
    """
    if not _load():
        return None

    try:
        import torch

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        else:
            img = image.convert("RGB") if image.mode != "RGB" else image

        inputs = _PROCESSOR(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = _MODEL(**inputs)
            probs = outputs.logits.softmax(dim=-1)[0]

        # id2label varies by model — for Organika/sdxl-detector:
        # {0: "Human", 1: "AI"}. Be defensive: pick the label whose lowercased
        # name contains "ai", "fake", "generated", or "synthetic".
        id2label = _MODEL.config.id2label
        ai_idx: Optional[int] = None
        for idx, label in id2label.items():
            if any(tok in label.lower() for tok in ("ai", "fake", "generated", "synthetic")):
                ai_idx = int(idx)
                break

        if ai_idx is None:
            # Shouldn't happen for this model, but fall back to index 1 by
            # convention (most binary AI/Human detectors put AI at 1).
            ai_idx = 1

        ai_prob = float(probs[ai_idx].item())
        return ai_prob
    except Exception as exc:
        logger.warning("cnn_detector_inference_failed", extra={
            "action": "cnn_detector_inference_failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        return None
