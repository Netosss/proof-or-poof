"""
Gold set accuracy check via the production combined client (temperature 0.2).
Usage: python scripts/eval_gold_set.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from app.integrations.gemini.client_combined import analyze_image_combined_async

# Portable gold set: labels live in the repo (tests/gold_set/labels.json) keyed by
# repo-relative paths ("images/<name>"); the images sit alongside under
# tests/gold_set/images/ (not committed — see tests/gold_set/README.md to restore).
# Override the location with GOLD_SET_DIR when the images live elsewhere.
GOLD_DIR = Path(
    os.environ.get("GOLD_SET_DIR", Path(__file__).resolve().parents[1] / "tests" / "gold_set")
)
LABELS = GOLD_DIR / "labels.json"


def _load_cases():
    data = json.loads(LABELS.read_text())
    cases = []
    for rel, meta in data.items():
        img = GOLD_DIR / rel
        if img.exists():
            cases.append((str(img), bool(meta.get("is_ai"))))
    return cases


async def main():
    cases = _load_cases()
    ai_count = sum(1 for _, ai in cases if ai)
    real_count = sum(1 for _, ai in cases if not ai)
    print(f"Gold set: {len(cases)} cases  ({ai_count} AI / {real_count} Real)\n")

    rows = []
    for path, expected_ai in cases:
        fname = os.path.basename(path)
        t0 = time.perf_counter()
        try:
            r = await analyze_image_combined_async(path)
        except Exception as exc:
            print(f"  {fname:<38} ERROR: {exc}")
            continue
        ms = round((time.perf_counter() - t0) * 1000)
        conf = r.get("confidence", -1.0)
        pred_ai = conf > 0.5
        ok = pred_ai == expected_ai
        rows.append({
            "file": fname,
            "expected_ai": expected_ai,
            "pred_ai": pred_ai,
            "conf": conf,
            "ok": ok,
        })
        label_e = "AI  " if expected_ai else "REAL"
        label_p = "AI  " if pred_ai else "REAL"
        status = "PASS" if ok else "FAIL !!!"
        print(f"  {fname:<38} exp={label_e}  got={label_p}  conf={conf:.2f}  {ms:>5}ms  {status}")
        await asyncio.sleep(0.15)

    n = len(rows)
    if not n:
        print(f"No rows — gold images not found under {GOLD_DIR}/images/")
        print("Restore them per tests/gold_set/README.md (or set GOLD_SET_DIR).")
        return

    passed = sum(1 for r in rows if r["ok"])
    fps = [r["file"] for r in rows if not r["ok"] and not r["expected_ai"]]
    fns = [r["file"] for r in rows if not r["ok"] and r["expected_ai"]]
    print(f"\n{'='*65}")
    print(f"Accuracy:        {passed}/{n} ({passed / n * 100:.1f}%)")
    print(f"False positives: {len(fps)}  → {fps}")
    print(f"False negatives: {len(fns)}  → {fns}")
    print(f"Baseline:        22/25 (88.0%)  — 1 FP ['linkdin profile.jpeg'], 2 FN ['130188.jpg', 'sofa.jpeg']")


asyncio.run(main())
