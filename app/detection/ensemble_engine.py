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

from PIL import Image

from app.config import settings
from app.integrations.gemini.client_ensemble import analyze_with_prompt
from app.integrations.gemini.prompts_ensemble import (
    get_anatomy_prompt,
    get_physics_prompt,
    get_composition_prompt,
)
from app.schemas.detection import V2_TO_LEGACY_CATEGORY
from app.integrations.gemini.quality import get_quality_context

logger = logging.getLogger(__name__)


def _label_for_findings(findings: str) -> str:
    """Trim a sub-call findings string into a 1-line label for the verdict scan."""
    txt = findings.strip().replace("\n", " ")
    return (txt[:200] + "…") if len(txt) > 200 else txt


async def _run_gemini_subcall(
    image_source: str | Image.Image,
    system_prompt: str,
    label: str,
    quality_context: str,
) -> dict:
    """
    Wraps the sync Gemini sub-call in to_thread so it can join the gather().
    The returned voter dict carries the RAW v2-macro signal_category from the
    model. Downstream consumers (the eval harness, debug logs) can read
    voter["signal_category"] without needing to know about the legacy mapping,
    which is only applied to the final aggregated verdict in _asymmetric_vote.
    """
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

    # No AI vote crosses threshold — return REAL with the mean confidence.
    # Preserve the most-suspicious voter's signal_category and findings so the
    # diagnostic trail (and any downstream threshold-tuning) keeps the strongest
    # sub-threshold signal instead of silently hard-coding "clean". The verdict
    # is still REAL (caller compares confidence to the AI threshold) — only the
    # accompanying signal_category and visual_scan carry the diagnostic detail.
    avg = mean(v["confidence"] for v in valid)
    least_clean = max(valid, key=lambda v: v["confidence"])
    return {
        "confidence": round(avg, 2),
        "signal_category": least_clean["signal_category"],
        "visual_scan": (
            f"[ensemble REAL via {least_clean['label']}] "
            f"{_label_for_findings(least_clean['findings'])}"
        ),
        "voters": voters,
    }


async def analyze_image_ensemble_async(
    image_source: str | Image.Image,
    pre_calculated_quality_context: str | None = None,
    pre_calculated_quality_score: int | None = None,
) -> dict:
    """
    Run all ensemble voters in parallel with RACE-TO-AI early-exit.

    As soon as any single voter reports confidence > settings.ensemble_ai_threshold,
    cancel the still-pending voters and return AI immediately. For images where
    no voter ever crosses the threshold (true REAL images), wait for all voters
    up to settings.ensemble_voter_timeout_s and combine via asymmetric vote.

    This trades a small amount of accuracy on borderline-suspicious cases
    (where a second voter might have pushed the score higher) for a large
    latency win on the AI cases that matter most.
    """
    quality_score = pre_calculated_quality_score or 0
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
    timeout_s = settings.ensemble_voter_timeout_s
    threshold = settings.ensemble_ai_threshold

    async def _bounded(coro, label: str) -> dict:
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError:
            return {
                "label": label, "confidence": -1.0,
                "signal_category": "no_visual_anomalies_detected",
                "findings": f"(timeout after {timeout_s}s)",
                "duration_ms": int(timeout_s * 1000), "ok": False,
            }

    # N-parallel anatomy calls use Gemini's non-determinism constructively:
    # on hard cases where a single call has ~33% catch rate, racing 3 in
    # parallel raises the effective catch rate to ~70%. Race-to-AI still
    # applies — the first to cross threshold cancels the rest.
    anatomy_n = max(1, settings.ensemble_anatomy_parallel_calls)
    voter_tasks: list[asyncio.Task[dict]] = []
    for i in range(anatomy_n):
        label = f"anatomy_{i + 1}" if anatomy_n > 1 else "anatomy"
        voter_tasks.append(asyncio.create_task(_bounded(
            _run_gemini_subcall(image_source, get_anatomy_prompt(quality_context), label, quality_context),
            label,
        )))
    voter_tasks.append(asyncio.create_task(_bounded(
        _run_gemini_subcall(image_source, get_physics_prompt(quality_context), "physics", quality_context),
        "physics",
    )))
    voter_tasks.append(asyncio.create_task(_bounded(
        _run_gemini_subcall(image_source, get_composition_prompt(quality_context), "composition", quality_context),
        "composition",
    )))

    completed: list[dict] = []
    early_exit = False
    for finished in asyncio.as_completed(voter_tasks):
        result = await finished
        completed.append(result)
        if result.get("ok") and result.get("confidence", -1.0) > threshold:
            early_exit = True
            # Cancel the still-pending voters; collect whatever they emit
            # by the time the cancellation propagates (best-effort, doesn't
            # block the verdict).
            for t in voter_tasks:
                if not t.done():
                    t.cancel()
            break

    # Drain any voters that finished between the last as_completed yield and
    # the cancel — so we have the most complete diagnostic record possible.
    for t in voter_tasks:
        if t.done() and not t.cancelled():
            try:
                r = t.result()
                if r not in completed:
                    completed.append(r)
            except Exception:
                pass

    # Fill in placeholders for any voter that never reported (cancelled), so
    # the diagnostic record always lists every expected voter label.
    reported_labels = {v["label"] for v in completed}
    expected_labels: list[str] = []
    if anatomy_n > 1:
        expected_labels.extend(f"anatomy_{i + 1}" for i in range(anatomy_n))
    else:
        expected_labels.append("anatomy")
    expected_labels.extend(("physics", "composition"))
    for label in expected_labels:
        if label not in reported_labels:
            completed.append({
                "label": label, "confidence": -1.0,
                "signal_category": "no_visual_anomalies_detected",
                "findings": "(cancelled by race-to-AI early exit)",
                "duration_ms": 0, "ok": False,
            })

    voters = completed
    total_ms = round((time.perf_counter() - t0) * 1000)
    verdict = _asymmetric_vote(voters)
    verdict["early_exit"] = early_exit
    legacy_category = V2_TO_LEGACY_CATEGORY.get(
        verdict["signal_category"], "multiple_subtle_ai_artifacts_present"
    )

    logger.info("ensemble_verdict", extra={
        "action": "ensemble_verdict",
        "wall_clock_ms": total_ms,
        "confidence": verdict["confidence"],
        "signal_category": verdict["signal_category"],
        "early_exit": early_exit,
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
    image_source: str | Image.Image,
    pre_calculated_quality_context: str | None = None,
    pre_calculated_quality_score: int | None = None,
) -> dict:
    """
    Sync wrapper for eval scripts / CLI use.

    The FastAPI path bypasses this entirely and awaits
    `analyze_image_ensemble_async` directly so cancellations can propagate
    without waiting for in-flight Gemini threads. This wrapper exists for
    scripts/eval_v2_rolling.py and ad-hoc CLI use.

    Detection logic:
      - If there is NO running event loop in the current thread, fall through
        to `asyncio.run(...)` — the normal sync-CLI case.
      - If there IS a running loop (we were invoked from a worker thread that
        an outer `asyncio.to_thread(...)` spawned), create a fresh inner loop,
        run the ensemble, and SHUTDOWN ITS DEFAULT EXECUTOR with timeout=0
        before closing. Without timeout=0, `loop.close()` blocks until every
        Gemini HTTP call in that inner pool returns — defeating the entire
        race-to-AI early exit (abandoned calls can take 10+ seconds).

    Uses `asyncio.get_running_loop()` rather than the deprecated
    `asyncio.get_event_loop()` so this is correct on Python 3.12+.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run() directly.
        return asyncio.run(
            analyze_image_ensemble_async(
                image_source,
                pre_calculated_quality_context,
                pre_calculated_quality_score,
            )
        )

    # Running loop found — we're on an inner thread spawned by asyncio.to_thread.
    # Build a fresh inner loop for the parallel ensemble.
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(
            analyze_image_ensemble_async(
                image_source,
                pre_calculated_quality_context,
                pre_calculated_quality_score,
            )
        )
    finally:
        try:
            new_loop.run_until_complete(
                new_loop.shutdown_default_executor(timeout=0.0)
            )
        except Exception as exc:
            logger.warning("ensemble_executor_shutdown_failed", extra={
                "action": "ensemble_executor_shutdown_failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
            })
        new_loop.close()
