"""
Ensemble detection engine — 3 specialised Gemini prompts + 1 CNN classifier,
all in parallel, combined via asymmetric voting.

Architecture (per the Opus architect consult):
- 3 Gemini calls with anatomy-only, physics-only, composition-only prompts
- 1 ONNX/transformers CNN classifier (Organika/sdxl-detector) running on CPU
- All four voters run concurrently via asyncio.gather → wall-clock latency ≈
  max(individual call durations), not the sum
- Asymmetric vote: ANY single voter with confidence > settings.ensemble_ai_threshold
  flips the verdict to AI, returning the max confidence across AI voters.
  Matches the empirical asymmetry of the current system (it over-says REAL,
  so a single confident AI signal should dominate).

Returns the same dict shape as v1/v2 so image_detector.py routing is identical.
"""

from __future__ import annotations

import asyncio
import logging
import time
from statistics import mean
from typing import Union

from PIL import Image

from app.config import settings
from app.detection import cnn_detector
from app.integrations.gemini.client_ensemble import analyze_with_prompt
from app.integrations.gemini.prompts_ensemble import (
    get_anatomy_prompt,
    get_physics_prompt,
    get_composition_prompt,
)
from app.integrations.gemini.client_v2 import V2_TO_LEGACY_CATEGORY
from app.integrations.gemini.quality import get_quality_context

logger = logging.getLogger(__name__)


def _label_for_findings(findings: str) -> str:
    """Trim a sub-call findings string into a 1-line label for the verdict scan."""
    txt = findings.strip().replace("\n", " ")
    return (txt[:200] + "…") if len(txt) > 200 else txt


async def _run_cnn(image_source: Union[str, Image.Image]) -> dict:
    """Run the CNN detector in a worker thread. Returns the same shape as a Gemini sub-call."""
    t0 = time.perf_counter()
    try:
        prob = await asyncio.to_thread(cnn_detector.predict_ai_probability, image_source)
        duration_ms = round((time.perf_counter() - t0) * 1000)
        if prob is None:
            return {
                "label": "cnn", "confidence": -1.0,
                "signal_category": "no_visual_anomalies_detected",
                "findings": "(CNN unavailable or inference failed)",
                "duration_ms": duration_ms, "ok": False,
            }
        return {
            "label": "cnn", "confidence": float(prob),
            "signal_category": (
                "multiple_subtle_ai_artifacts_present" if prob > 0.5
                else "no_visual_anomalies_detected"
            ),
            "findings": f"CNN AI-probability {prob:.2f}",
            "duration_ms": duration_ms, "ok": True,
        }
    except Exception as exc:
        return {
            "label": "cnn", "confidence": -1.0,
            "signal_category": "no_visual_anomalies_detected",
            "findings": f"(error: {type(exc).__name__})",
            "duration_ms": round((time.perf_counter() - t0) * 1000),
            "ok": False,
        }


async def _run_gemini_subcall(
    image_source: Union[str, Image.Image],
    system_prompt: str,
    label: str,
    quality_context: str,
) -> dict:
    """Wraps the sync Gemini sub-call in to_thread so it can join the gather()."""
    return await asyncio.to_thread(
        analyze_with_prompt,
        image_source, system_prompt, label, quality_context,
    )


def _asymmetric_vote(voters: list[dict]) -> dict:
    """
    Combine voter results into a single verdict.

    Rule: any voter (Gemini or CNN) with confidence > settings.ensemble_ai_threshold
    flips the verdict to AI. The winning confidence is the MAX across AI voters
    (the most confident accuser wins). If no voter clears the threshold, return
    REAL with the MEAN confidence across responding voters.

    Voters with ok=False / confidence=-1.0 are excluded from both vote and mean.
    """
    valid = [v for v in voters if v.get("ok") and v.get("confidence", -1.0) >= 0]
    if not valid:
        return {
            "confidence": -1.0,
            "signal_category": "multiple_subtle_ai_artifacts_present",
            "visual_scan": "(all voters failed)",
            "voters": voters,
        }

    threshold = settings.ensemble_ai_threshold
    ai_voters = [v for v in valid if v["confidence"] > threshold]

    if ai_voters:
        # Most confident accuser wins; carry their signal_category and findings.
        winner = max(ai_voters, key=lambda v: v["confidence"])
        return {
            "confidence": winner["confidence"],
            "signal_category": winner["signal_category"],
            "visual_scan": (
                f"[{winner['label']}] {_label_for_findings(winner['findings'])}"
            ),
            "voters": voters,
        }

    # No AI vote crosses threshold — return REAL with mean confidence and the
    # findings of the most-suspicious-but-still-clean voter.
    avg = mean(v["confidence"] for v in valid)
    least_clean = max(valid, key=lambda v: v["confidence"])
    return {
        "confidence": round(avg, 2),
        "signal_category": "no_visual_anomalies_detected",
        "visual_scan": (
            f"[ensemble REAL] {_label_for_findings(least_clean['findings'])}"
        ),
        "voters": voters,
    }


async def analyze_image_ensemble_async(
    image_source: Union[str, Image.Image],
    pre_calculated_quality_context: str | None = None,
) -> dict:
    """
    Run all ensemble voters in parallel. Returns the same dict shape used by v1/v2.
    """
    # Compute quality_context once (cheap) so all three Gemini sub-calls
    # share it without duplicating the work.
    quality_score = 0
    if pre_calculated_quality_context:
        quality_context = pre_calculated_quality_context
    else:
        try:
            quality_context, quality_score = get_quality_context(image_source)
        except Exception as exc:
            logger.warning("ensemble_quality_context_failed", extra={
                "action": "ensemble_quality_context_failed",
                "error": str(exc),
            })
            quality_context = "**CONTEXT: QUALITY UNKNOWN.**"

    t0 = time.perf_counter()

    voter_tasks = [
        _run_gemini_subcall(image_source, get_anatomy_prompt(quality_context),     "anatomy",     quality_context),
        _run_gemini_subcall(image_source, get_physics_prompt(quality_context),     "physics",     quality_context),
        _run_gemini_subcall(image_source, get_composition_prompt(quality_context), "composition", quality_context),
        _run_cnn(image_source),
    ]
    voters = await asyncio.gather(*voter_tasks, return_exceptions=False)
    total_ms = round((time.perf_counter() - t0) * 1000)

    verdict = _asymmetric_vote(voters)
    legacy_category = V2_TO_LEGACY_CATEGORY.get(
        verdict["signal_category"], "multiple_subtle_ai_artifacts_present"
    )

    logger.info("ensemble_verdict", extra={
        "action": "ensemble_verdict",
        "wall_clock_ms": total_ms,
        "confidence": verdict["confidence"],
        "signal_category": verdict["signal_category"],
        "voters": [
            {"label": v["label"], "confidence": v["confidence"],
             "ok": v["ok"], "duration_ms": v["duration_ms"]}
            for v in voters
        ],
    })

    return {
        "visual_scan": verdict["visual_scan"],
        "confidence": verdict["confidence"],
        "signal_category": legacy_category,
        "quality_score": quality_score,
        "quality_context": quality_context,
        # Ensemble-specific diagnostic fields — visible to the eval harness.
        "v2_signal_category": verdict["signal_category"],
        "v2_step_1": " | ".join(
            f"{v['label']}={v['confidence']:.2f}" for v in voters if v["ok"]
        ),
        "v2_step_2": " | ".join(
            f"{v['label']}: {_label_for_findings(v['findings'])}" for v in voters if v["ok"]
        ),
        "ensemble_voters": voters,
    }


def analyze_image_ensemble(
    image_source: Union[str, Image.Image],
    pre_calculated_quality_context: str | None = None,
) -> dict:
    """
    Sync wrapper so the existing image_detector dispatch (which uses
    asyncio.to_thread on a sync analyzer) keeps working without modification.
    Internally runs the parallel async ensemble via asyncio.run.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Caller is already in an event loop (FastAPI request handler going
            # through asyncio.to_thread). We're now on a worker thread — make
            # a fresh loop to run the gather().
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(
                    analyze_image_ensemble_async(
                        image_source, pre_calculated_quality_context
                    )
                )
            finally:
                new_loop.close()
    except RuntimeError:
        pass

    return asyncio.run(
        analyze_image_ensemble_async(image_source, pre_calculated_quality_context)
    )
