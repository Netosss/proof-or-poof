"""
Cost / accuracy / latency comparison for three detection architectures:

  1. anatomy-solo       — single voter, single API call (best single-voter)
  2. combined-batch     — all 3 perspectives merged into ONE API call
  3. ensemble (current) — 3 parallel API calls, race-to-AI vote

Records per-call: confidence, verdict, latency, prompt_tokens, output_tokens,
calculated USD cost. Reports per-variant: accuracy, FP rate, FN rate,
avg / p95 latency, total cost, cost per 1000 detections.

Pricing (Gemini 3 Flash Preview):
  input  $0.50 / 1M tokens
  output $3.00 / 1M tokens
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("DETECTION_ENGINE", "ensemble")

from app.config import settings  # noqa: E402
from app.integrations.gemini.client import client  # noqa: E402
from app.integrations.gemini.prompts_ensemble import (  # noqa: E402
    get_anatomy_prompt,
    get_combined_prompt,
)
from app.detection.ensemble_engine import analyze_image_ensemble_async  # noqa: E402

GOLD = _REPO_ROOT / "test_prompt_gold.json"

PRICE_INPUT_PER_TOKEN = 0.50 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 3.00 / 1_000_000


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def _cost(in_t: int, out_t: int) -> float:
    return in_t * PRICE_INPUT_PER_TOKEN + out_t * PRICE_OUTPUT_PER_TOKEN


async def _call_with_metrics(image_path: str, system_prompt: str, label: str) -> dict:
    """Single Gemini call; reads usage_metadata from response for token counts."""
    from google.genai import types
    from app.integrations.gemini.client import _prepare_pil_for_gemini, _encode_pil_as_jpeg
    from app.schemas.detection import EnsembleSubResult

    img_working, to_close = _prepare_pil_for_gemini(image_path)
    try:
        image_bytes = _encode_pil_as_jpeg(img_working, settings.gemini_jpeg_quality)
    finally:
        for o in to_close:
            try:
                o.close()
            except Exception:
                pass

    config_kwargs: dict = dict(
        system_instruction=system_prompt,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        temperature=settings.gemini_temperature,
        response_mime_type="application/json",
        response_schema=EnsembleSubResult,
    )
    if "gemini-3" in settings.gemini_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=settings.gemini_thinking_level
        )
    config = types.GenerateContentConfig(**config_kwargs)

    t0 = time.perf_counter()
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "Analyse this image strictly within the focus area defined by the system instructions. Return ONLY the JSON.",
        ],
        config=config,
    )
    ms = round((time.perf_counter() - t0) * 1000)
    parsed = response.parsed
    in_t = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0
    out_t = getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0
    return {
        "label": label,
        "confidence": parsed.confidence if parsed else -1.0,
        "signal_category": parsed.signal_category if parsed else "no_visual_anomalies_detected",
        "ms": ms,
        "in_tokens": in_t or 0,
        "out_tokens": out_t or 0,
        "cost_usd": _cost(in_t or 0, out_t or 0),
    }


async def _run_solo_variant(cases, prompt_fn, variant_name: str) -> dict:
    from app.integrations.gemini.quality import get_quality_context
    print(f"\n=== {variant_name.upper()} (1 API call/image) ===")
    rows = []
    for path, expected_ai in cases:
        try:
            qc, _ = get_quality_context(path)
            r = await _call_with_metrics(path, prompt_fn(qc), variant_name)
            verdict_ai = r["confidence"] > 0.5
            ok = verdict_ai == expected_ai
            r["expected_ai"] = expected_ai
            r["ok"] = ok
            r["file"] = os.path.basename(path)
            rows.append(r)
            mark = "PASS" if ok else "FAIL"
            print(f"  {os.path.basename(path):<35} exp={'AI ' if expected_ai else 'REAL'} "
                  f"got={'AI ' if verdict_ai else 'REAL'} conf={r['confidence']:.2f} "
                  f"{r['ms']:>5}ms in={r['in_tokens']:>5} out={r['out_tokens']:>3} ${r['cost_usd']:.4f} {mark}")
        except Exception as e:
            print(f"  {os.path.basename(path):<35} ERROR: {type(e).__name__}: {e}")
        await asyncio.sleep(0.3)
    return _summarise(rows, variant_name)


async def run_ensemble(cases) -> dict:
    print("\n=== ENSEMBLE (3 parallel voters, race-to-AI) ===")
    rows = []
    for path, expected_ai in cases:
        t0 = time.perf_counter()
        try:
            r = await analyze_image_ensemble_async(path)
        except Exception as e:
            print(f"  {os.path.basename(path):<35} ERROR: {type(e).__name__}: {e}")
            continue
        ms = round((time.perf_counter() - t0) * 1000)
        c = r.get("confidence", -1.0)
        verdict_ai = c > 0.5
        ok = verdict_ai == expected_ai
        n_voters = sum(1 for v in r.get("ensemble_voters", []) if v.get("ok"))
        row = {
            "file": os.path.basename(path), "expected_ai": expected_ai, "ok": ok,
            "confidence": c, "ms": ms, "n_voters": n_voters,
        }
        rows.append(row)
        mark = "PASS" if ok else "FAIL"
        print(f"  {os.path.basename(path):<35} exp={'AI ' if expected_ai else 'REAL'} "
              f"got={'AI ' if verdict_ai else 'REAL'} conf={c:.2f} {ms:>5}ms voters={n_voters} {mark}")
        await asyncio.sleep(0.3)
    return _summarise_ensemble(rows)


def _summarise(rows: list[dict], variant: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"variant": variant, "n": 0}
    passed = sum(1 for r in rows if r["ok"])
    fps = [r for r in rows if not r["ok"] and not r["expected_ai"]]
    fns = [r for r in rows if not r["ok"] and r["expected_ai"]]
    ms_vals = [r["ms"] for r in rows]
    costs = [r["cost_usd"] for r in rows]
    in_tokens = [r["in_tokens"] for r in rows]
    out_tokens = [r["out_tokens"] for r in rows]
    return {
        "variant": variant,
        "n": n,
        "accuracy_pct": round(passed / n * 100, 1),
        "fp_count": len(fps),
        "fn_count": len(fns),
        "fps": [r["file"] for r in fps],
        "fns": [r["file"] for r in fns],
        "latency_avg_ms": round(sum(ms_vals) / n),
        "latency_p95_ms": round(_p95(ms_vals)),
        "latency_max_ms": max(ms_vals),
        "avg_in_tokens": round(sum(in_tokens) / n),
        "avg_out_tokens": round(sum(out_tokens) / n),
        "cost_per_call_usd": round(sum(costs) / n, 5),
        "cost_total_usd": round(sum(costs), 4),
        "cost_per_1000_detections_usd": round(sum(costs) / n * 1000, 2),
    }


def _summarise_ensemble(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"variant": "ensemble", "n": 0}
    passed = sum(1 for r in rows if r["ok"])
    fps = [r for r in rows if not r["ok"] and not r["expected_ai"]]
    fns = [r for r in rows if not r["ok"] and r["expected_ai"]]
    ms_vals = [r["ms"] for r in rows]
    voter_counts = [r["n_voters"] for r in rows]
    return {
        "variant": "ensemble",
        "n": n,
        "accuracy_pct": round(passed / n * 100, 1),
        "fp_count": len(fps),
        "fn_count": len(fns),
        "fps": [r["file"] for r in fps],
        "fns": [r["file"] for r in fns],
        "latency_avg_ms": round(sum(ms_vals) / n),
        "latency_p95_ms": round(_p95(ms_vals)),
        "latency_max_ms": max(ms_vals),
        "avg_voters_responded": round(sum(voter_counts) / n, 2),
        "note_on_cost": (
            "Ensemble cost ≈ avg_voters_responded × anatomy-solo cost-per-call. "
            "Race-to-AI cancellation may reduce actual billed cost when AI cases "
            "early-exit, but cancelled calls' tokens are still consumed up to "
            "the cancellation point — treat 3× single-voter as a worst-case upper bound."
        ),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["anatomy", "combined", "ensemble", "all"], default="all")
    args = parser.parse_args()

    if not GOLD.exists():
        sys.exit(f"Gold file not found: {GOLD}")
    raw = json.loads(GOLD.read_text())
    cases = [(p, bool(m.get("is_ai"))) for p, m in raw.items() if os.path.exists(p)]
    print(f"Cases: {len(cases)} ({sum(1 for _, ai in cases if ai)} AI, "
          f"{sum(1 for _, ai in cases if not ai)} REAL)")
    print(f"Model: {settings.gemini_model}  temp={settings.gemini_temperature}  "
          f"thinking={settings.gemini_thinking_level}")

    summaries: list[dict] = []
    if args.mode in ("anatomy", "all"):
        summaries.append(await _run_solo_variant(cases, get_anatomy_prompt, "anatomy-solo"))
    if args.mode in ("combined", "all"):
        summaries.append(await _run_solo_variant(cases, get_combined_prompt, "combined-batch"))
    if args.mode in ("ensemble", "all"):
        summaries.append(await run_ensemble(cases))

    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    for s in summaries:
        print(f"\n{s['variant']}:")
        for k, v in s.items():
            if k == "variant":
                continue
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
