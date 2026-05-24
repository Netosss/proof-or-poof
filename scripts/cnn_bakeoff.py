"""
CNN-only bake-off — compare 3 candidate AI-image classifiers on the gold set.

Loads each model in sequence (memory-conscious), runs inference on every
gold image plus the 3 stuck test images, and prints a per-model accuracy
table with false-positive / false-negative breakdown.

NO Gemini calls — this is purely about understanding the CNN voter we're
considering for the ensemble.

Usage:
  python scripts/cnn_bakeoff.py            # all 3 models
  python scripts/cnn_bakeoff.py --models organika   # single model

The 3 candidates were chosen for orthogonal coverage:
  - Organika/sdxl-detector      Swin Transformer, trained on SDXL outputs (~2023)
  - cmckinle/sdxl-flux-detector ViT, trained on SDXL + Flux (newer, broader)
  - prithivMLmods/Deep-Fake-Detector-v2-Model  Different lineage, broader dataset

Each model's AI-class index is auto-detected from id2label (we look for any
label containing "ai", "fake", "generated", "synthetic"). Threshold = 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GOLD_FILE = _REPO_ROOT / "test_prompt_gold.json"

# 3 stuck test images we've been struggling with — append to the gold run so
# we see how each CNN handles them specifically.
EXTRA_CASES = [
    ("/Users/netanel.ossi/Downloads/167954.jpg",       True,  "gym (stuck)"),
    ("/Users/netanel.ossi/Downloads/170605 (1).jpg",   True,  "bibi (stuck)"),
    ("/Users/netanel.ossi/Downloads/171528.jpg",       True,  "bbq (stuck)"),
]

CANDIDATES = {
    "organika":     "Organika/sdxl-detector",
    "umm_maybe":    "umm-maybe/AI-image-detector",
    "prithivmlods": "prithivMLmods/Deep-Fake-Detector-v2-Model",
}


def _load_cases() -> list[tuple[str, bool, str]]:
    """Returns list of (path, expected_ai, label) covering gold + extras."""
    if not GOLD_FILE.exists():
        sys.exit(f"Gold file not found: {GOLD_FILE}")
    raw = json.loads(GOLD_FILE.read_text())
    cases = [(path, bool(meta.get("is_ai")), Path(path).name)
             for path, meta in raw.items()]
    cases.extend(EXTRA_CASES)
    return cases


def _ai_idx_from_labels(id2label: dict) -> int:
    """Pick the AI-class index by searching for AI-ish keywords in labels."""
    for idx, label in id2label.items():
        if any(tok in str(label).lower()
               for tok in ("ai", "fake", "generated", "synthetic")):
            return int(idx)
    # Fall back to 1 by convention (most binary AI/Human models put AI at 1).
    return 1


def evaluate_model(model_id: str, cases: list[tuple[str, bool, str]]) -> dict:
    """Load + run a single classifier across all cases; return stats."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch

    print(f"\n{'=' * 70}\nMODEL: {model_id}\n{'=' * 70}")

    t0 = time.perf_counter()
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.eval()
        model.to("cpu")
    except Exception as exc:
        print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
        return {"model": model_id, "error": str(exc)}
    load_ms = round((time.perf_counter() - t0) * 1000)

    id2label = model.config.id2label
    ai_idx = _ai_idx_from_labels(id2label)
    print(f"  Loaded in {load_ms}ms  |  id2label={id2label}  |  AI-class idx={ai_idx}")

    results: list[dict] = []
    for path, expected_ai, label in cases:
        p = Path(path)
        if not p.exists():
            print(f"  [SKIP] {label}: file not found")
            continue

        try:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            t1 = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits.softmax(dim=-1)[0]
            inf_ms = round((time.perf_counter() - t1) * 1000)
            ai_prob = float(probs[ai_idx].item())
            predicted_ai = ai_prob > 0.5
            outcome = "PASS" if predicted_ai == expected_ai else "FAIL"
            tag = ("FP" if predicted_ai and not expected_ai
                   else "FN" if not predicted_ai and expected_ai
                   else "")
            results.append({
                "label": label, "expected_ai": expected_ai,
                "predicted_ai": predicted_ai, "ai_prob": ai_prob,
                "inf_ms": inf_ms, "outcome": outcome, "tag": tag,
            })
        except Exception as exc:
            print(f"  [ERROR] {label}: {type(exc).__name__}: {exc}")
            results.append({"label": label, "error": str(exc)})

    # Per-case printout
    for r in results:
        if "error" in r:
            continue
        exp = "AI  " if r["expected_ai"] else "REAL"
        pred = "AI  " if r["predicted_ai"] else "REAL"
        print(f"  {r['label']:<28} exp={exp} pred={pred} "
              f"p_ai={r['ai_prob']:.2f} {r['inf_ms']:>4}ms {r['outcome']:<4} {r['tag']}")

    # Stats
    valid = [r for r in results if "outcome" in r]
    passed = sum(1 for r in valid if r["outcome"] == "PASS")
    fps = [r for r in valid if r["tag"] == "FP"]
    fns = [r for r in valid if r["tag"] == "FN"]
    total = len(valid)
    avg_ms = round(sum(r["inf_ms"] for r in valid) / total) if total else 0

    print(f"\n  Accuracy: {passed}/{total} ({passed / total * 100:.1f}%)" if total else "  no results")
    print(f"  False positives: {len(fps)} (real images flagged AI)")
    if fps:
        for r in fps:
            print(f"     - {r['label']} p_ai={r['ai_prob']:.2f}")
    print(f"  False negatives: {len(fns)} (AI images flagged real)")
    if fns:
        for r in fns:
            print(f"     - {r['label']} p_ai={r['ai_prob']:.2f}")
    print(f"  Avg inference: {avg_ms}ms")

    return {
        "model": model_id, "load_ms": load_ms, "total": total,
        "passed": passed, "fps": len(fps), "fns": len(fns),
        "avg_inf_ms": avg_ms, "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(CANDIDATES.keys()),
                        help=f"Subset of: {list(CANDIDATES.keys())}")
    args = parser.parse_args()

    cases = _load_cases()
    print(f"Cases: {len(cases)} ({sum(1 for _, ai, _ in cases if ai)} AI, "
          f"{sum(1 for _, ai, _ in cases if not ai)} REAL)")

    summaries = []
    for key in args.models:
        if key not in CANDIDATES:
            print(f"skipping unknown model key: {key}")
            continue
        summary = evaluate_model(CANDIDATES[key], cases)
        summaries.append(summary)

    # Final comparison table
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'model':<50} {'acc':>6} {'FP':>4} {'FN':>4} {'avg_ms':>7}")
    for s in summaries:
        if "error" in s:
            print(f"{s['model']:<50} LOAD FAILED")
            continue
        acc = s["passed"] / s["total"] * 100 if s["total"] else 0
        print(f"{s['model']:<50} {acc:>5.1f}% {s['fps']:>4} {s['fns']:>4} {s['avg_inf_ms']:>5}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
