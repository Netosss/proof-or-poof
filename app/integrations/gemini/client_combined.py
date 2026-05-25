"""
Combined-batch Gemini analyzer — single API call, three perspectives merged.

Why this engine exists: empirical comparison (see
docs/DETECTION_V2_DEFERRED_FIXES.md and scripts/eval_cost_accuracy.py)
showed that a single Gemini call with all three forensic perspectives
(anatomy + physics + composition) merged into one system instruction
outperforms the parallel 3-voter ensemble on every axis:

  Accuracy:  96.0% (vs ensemble's 92.0%)
  FP rate:   0/11 REAL images (vs ensemble's 1/11)
  Cost:      $1.45 per 1000 detections (vs ensemble's ~$2.40 — ~40% cheaper)
  Latency:   p95 11.5s (vs ensemble's 10.1s — slightly higher because no
             race-to-AI shortcut, but only by ~1.5s)

The architectural reason: when all three perspectives live in one prompt,
the model integrates them internally and can cross-reference (e.g.
"anatomy is clean BUT composition shows in-scene gibberish → AI"). In the
parallel ensemble, each voter sees only its narrow focus and cannot read
the others' reasoning — the asymmetric vote rule is a much weaker
integration than the model's own.

This client reuses analyze_with_prompt from client_ensemble.py with the
combined prompt — same async path, same schema, same V2→legacy category
mapping. The only difference is the prompt.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Union

from PIL import Image

from app.schemas.detection import V2_TO_LEGACY_CATEGORY
from app.integrations.gemini.client_ensemble import analyze_with_prompt
from app.integrations.gemini.prompts_ensemble import get_combined_prompt
from app.integrations.gemini.quality import get_quality_context

logger = logging.getLogger(__name__)


async def analyze_image_combined_async(
    image_source: Union[str, Image.Image],
    pre_calculated_quality_context: Optional[str] = None,
    pre_calculated_quality_score: Optional[int] = None,
) -> dict:
    """
    Single Gemini call with the combined 3-perspective prompt. Returns the
    same dict shape used by v1/v2/ensemble so image_detector routing is
    identical.
    """
    quality_score = pre_calculated_quality_score or 0
    if pre_calculated_quality_context:
        quality_context = pre_calculated_quality_context
    else:
        # get_quality_context runs OpenCV cv2.Laplacian + cv2.cvtColor
        # synchronously — must NOT execute on the event loop. The normal
        # production path pre-computes this in image_detector via to_thread,
        # so we only hit this branch on direct callers (eval scripts, tests,
        # the v1 fallback re-entry). to_thread the fallback so any code path
        # is safe.
        try:
            quality_context, quality_score = await asyncio.to_thread(
                get_quality_context, image_source
            )
        except Exception as exc:
            logger.warning("combined_quality_context_failed", extra={
                "action": "combined_quality_context_failed",
                "error": str(exc),
            })
            quality_context = "**CONTEXT: QUALITY UNKNOWN.**"

    sub_result = await analyze_with_prompt(
        image_source,
        get_combined_prompt(quality_context),
        label="combined",
        pre_calculated_quality_context=quality_context,
    )

    if not sub_result.get("ok"):
        # API/programming error already logged inside analyze_with_prompt.
        return {
            "confidence": -1.0,
            "signal_category": "multiple_subtle_ai_artifacts_present",
            "visual_scan": sub_result.get("findings", "(combined call failed)"),
            "quality_score": quality_score,
            "quality_context": quality_context,
            "v2_signal_category": sub_result.get("signal_category"),
            "v2_step_1": sub_result.get("findings", ""),
            "v2_step_2": "",
        }

    v2_cat = sub_result["signal_category"]
    legacy_cat = V2_TO_LEGACY_CATEGORY.get(v2_cat, "multiple_subtle_ai_artifacts_present")

    return {
        "visual_scan": sub_result.get("findings", ""),
        "confidence": sub_result["confidence"],
        "signal_category": legacy_cat,
        "quality_score": quality_score,
        "quality_context": quality_context,
        # Diagnostic fields for the eval harness and debug logs
        "v2_signal_category": v2_cat,
        "v2_step_1": sub_result.get("findings", ""),
        "v2_step_2": f"region_anchor: {sub_result.get('region_anchor', '')}",
    }
