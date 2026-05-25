"""
Cost / accuracy / latency measurement for the combined detection engine.

Runs the combined Gemini call on every image in test_prompt_gold.json and
reports per-call token counts + USD cost + latency, then a summary table.
Useful for validating accuracy holds after prompt or model changes.

Pricing (Gemini 3 Flash Preview):
  input  $0.50 / 1M tokens
  output $3.00 / 1M tokens
"""

from __future__ import annotations

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

from app.config import settings  # noqa: E402
from app.integrations.gemini.client import client  # noqa: E402
from app.integrations.gemini.prompts_combined import get_combined_prompt  # noqa: E402

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


async def _call(image_path: str) -> dict:
    """One combined Gemini call, capturing token + latency metrics."""
    from google.genai import types
    from app.integrations.gemini.client import _prepare_pil_for_gemini, _encode_pil_as_jpeg
    from app.integrations.gemini.quality import get_quality_context
    from app.schemas.detection import CombinedDetectionResult

    quality_context, _ = await asyncio.to_thread(get_quality_context, image_path)

    def _prep() -> bytes:
        img_working, to_close = _prepare_pil_for_gemini(image_path)
        try:
            return _encode_pil_as_jpeg(img_working, settings.gemini_jpeg_quality)
        finally:
            for o in to_close:
                try:
                    o.close()
                except Exception:
                    pass

    image_bytes = await asyncio.to_thread(_prep)

    config_kwargs: dict = dict(
        system_instruction=get_combined_prompt(quality_context),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        temperature=settings.gemini_temperature,
        response_mime_type="application/json",
        response_schema=CombinedDetectionResult,
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
    in_t = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
    out_t = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
    return {
        "confidence": parsed.confidence if parsed else -1.0,
        "signal_category": parsed.signal_category if parsed else "no_visual_anomalies_detected",
        "ms": ms,
        "in_tokens": in_t,
        "out_tokens": out_t,
        "cost_usd": _cost(in_t, out_t),
    }


async def main() -> int:
    if not GOLD.exists():
        sys.exit(f"Gold file not found: {GOLD}")
    raw = json.loads(GOLD.read_text())
    cases = [(p, bool(m.get("is_ai"))) for p, m in raw.items() if os.path.exists(p)]
    print(f"Cases: {len(cases)} ({sum(1 for _, ai in cases if ai)} AI, "
          f"{sum(1 for _, ai in cases if not ai)} REAL)")
    print(f"Model: {settings.gemini_model}  temp={settings.gemini_temperature}  "
          f"thinking={settings.gemini_thinking_level}\n")

    rows = []
    for path, expected_ai in cases:
        try:
            r = await _call(path)
        except Exception as exc:
            print(f"  {os.path.basename(path):<35} ERROR: {type(exc).__name__}: {exc}")
            continue
        pred_ai = r["confidence"] > 0.5
        ok = pred_ai == expected_ai
        r.update({"expected_ai": expected_ai, "ok": ok, "file": os.path.basename(path)})
        rows.append(r)
        print(f"  {os.path.basename(path):<35} "
              f"exp={'AI' if expected_ai else 'REAL':<4} "
              f"got={'AI' if pred_ai else 'REAL':<4} "
              f"conf={r['confidence']:.2f} {r['ms']:>5}ms "
              f"in={r['in_tokens']:>5} out={r['out_tokens']:>3} "
              f"${r['cost_usd']:.4f} {'PASS' if ok else 'FAIL'}")
        await asyncio.sleep(0.3)

    n = len(rows)
    if not n:
        print("no rows")
        return 1
    passed = sum(1 for r in rows if r["ok"])
    fps = [r for r in rows if not r["ok"] and not r["expected_ai"]]
    fns = [r for r in rows if not r["ok"] and r["expected_ai"]]
    ms_vals = [r["ms"] for r in rows]
    costs = [r["cost_usd"] for r in rows]

    print(f"\nAccuracy:           {passed}/{n} ({passed / n * 100:.1f}%)")
    print(f"False positives:    {len(fps)} ({[r['file'] for r in fps]})")
    print(f"False negatives:    {len(fns)} ({[r['file'] for r in fns]})")
    print(f"Latency avg/p95:    {round(sum(ms_vals) / n)}ms / {round(_p95(ms_vals))}ms")
    print(f"Cost per call:      ${round(sum(costs) / n, 5)}")
    print(f"Cost per 1k:        ${round(sum(costs) / n * 1000, 2)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
