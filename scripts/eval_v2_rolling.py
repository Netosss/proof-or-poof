"""
Rolling-4 evaluation harness for the V2 forensic detection engine.

Protocol (per /plan):
  1. Read test_prompt_gold.json (full 25-case gold set).
  2. Run cases in batches of 4 (cheap, fast feedback).
  3. If every case in the batch passes → advance to the next 4 and continue.
  4. If ANY case fails → stop, dump failing cases + reasoning, exit non-zero.
  5. The full 25-case run is gated to `--full` (used only at pre-merge time).

Usage:
  # default rolling-4 run, stops on first failing batch
  python scripts/eval_v2_rolling.py

  # only re-run a specific subset of files (after a prompt tweak)
  python scripts/eval_v2_rolling.py --only file1.jpg file2.jpg

  # full 25-case run (pre-merge gate)
  python scripts/eval_v2_rolling.py --full

  # force v1 for side-by-side baseline
  python scripts/eval_v2_rolling.py --engine v1

Output:
  Prints per-case PASS/FAIL inline. On any failure, writes a JSONL report to
  eval_runs/v2_<timestamp>.jsonl containing the case path, expected/predicted
  labels, confidence, v2 CoT step_1 + step_2 transcripts, and visual_scan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Adjust import path so we can run this from repo root: `python scripts/...`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import settings  # noqa: E402
from app.integrations.gemini.client import analyze_image_pro_turbo  # noqa: E402
from app.integrations.gemini.client_v2 import analyze_image_pro_turbo_v2  # noqa: E402


GOLD_FILE = _REPO_ROOT / "test_prompt_gold.json"
EVAL_OUT_DIR = _REPO_ROOT / "eval_runs"
BATCH_SIZE = 4
PASS_THRESHOLD = 0.5  # confidence > this = predicted AI


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(math.ceil(0.95 * len(s)) - 1, 0)
    return s[idx]


def _load_gold() -> list[tuple[str, bool]]:
    if not GOLD_FILE.exists():
        sys.exit(f"Gold file not found: {GOLD_FILE}")
    raw = json.loads(GOLD_FILE.read_text())
    return [(path, bool(meta.get("is_ai", False))) for path, meta in raw.items()]


def _select_analyzer(engine: str):
    if engine == "v2":
        return analyze_image_pro_turbo_v2
    return analyze_image_pro_turbo


def _run_case(analyzer, path: str, expected_ai: bool) -> dict:
    filename = os.path.basename(path)
    if not os.path.exists(path):
        return {"file": filename, "path": path, "skipped": True, "reason": "file_not_found"}

    t0 = time.perf_counter()
    try:
        result = analyzer(path)
    except Exception as exc:  # pragma: no cover — surfaced inline
        return {
            "file": filename, "path": path, "error": str(exc),
            "error_type": type(exc).__name__,
        }
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    confidence = float(result.get("confidence", -1.0))
    if confidence == -1.0:
        return {
            "file": filename, "path": path, "error": "api_failure",
            "elapsed_ms": elapsed_ms,
        }

    predicted_ai = confidence > PASS_THRESHOLD
    passed = predicted_ai == expected_ai

    return {
        "file": filename,
        "path": path,
        "expected_ai": expected_ai,
        "predicted_ai": predicted_ai,
        "confidence": confidence,
        "elapsed_ms": elapsed_ms,
        "passed": passed,
        "signal_category": result.get("signal_category"),
        "v2_signal_category": result.get("v2_signal_category"),
        "v2_step_1": result.get("v2_step_1"),
        "v2_step_2": result.get("v2_step_2"),
        "visual_scan": result.get("visual_scan"),
    }


def _print_case(idx: int, total: int, rec: dict) -> None:
    fname = rec["file"]
    if rec.get("skipped"):
        print(f"[{idx:>2}/{total}] {fname:<35} | SKIP ({rec.get('reason')})")
        return
    if "error" in rec:
        print(f"[{idx:>2}/{total}] {fname:<35} | ERROR: {rec['error']}")
        return
    exp = "AI  " if rec["expected_ai"] else "REAL"
    pred = "AI  " if rec["predicted_ai"] else "REAL"
    status = "PASS" if rec["passed"] else "FAIL"
    print(
        f"[{idx:>2}/{total}] {fname:<35} | exp:{exp} pred:{pred} "
        f"conf:{rec['confidence']:.2f} {rec['elapsed_ms']:>5}ms | {status}"
    )


def _write_report(records: list[dict], engine: str, mode: str) -> Path:
    EVAL_OUT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = EVAL_OUT_DIR / f"{engine}_{mode}_{ts}.jsonl"
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return out


def _summarise(records: list[dict]) -> None:
    evaluated = [r for r in records if "passed" in r]
    if not evaluated:
        print("\n(no cases evaluated)")
        return
    passed = sum(1 for r in evaluated if r["passed"])
    latencies = [r["elapsed_ms"] for r in evaluated]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Accuracy:    {passed}/{len(evaluated)} ({passed / len(evaluated) * 100:.1f}%)")
    print(f"  Latency avg: {round(sum(latencies) / len(latencies))} ms")
    print(f"  Latency p95: {round(_p95(latencies))} ms")
    print(f"  Latency max: {round(max(latencies))} ms")
    failures = [r for r in evaluated if not r["passed"]]
    if failures:
        print("\nFAILURES:")
        for r in failures:
            print(f"  - {r['file']} (exp {'AI' if r['expected_ai'] else 'REAL'}, "
                  f"got conf={r['confidence']:.2f}, sig={r.get('v2_signal_category') or r.get('signal_category')})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["v1", "v2"], default="v2")
    parser.add_argument("--full", action="store_true",
                        help="Run the entire gold set (pre-merge gate only).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these gold-set files (basenames or paths).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.engine == "v2":
        os.environ["DETECTION_ENGINE"] = "v2"
    elif args.engine == "v1":
        os.environ["DETECTION_ENGINE"] = "v1"

    cases = _load_gold()

    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if os.path.basename(c[0]) in wanted or c[0] in wanted]
        if not cases:
            sys.exit("No gold cases matched --only filter.")

    analyzer = _select_analyzer(args.engine)
    print(f"Engine: {args.engine}  |  Cases: {len(cases)}  |  Batch size: {args.batch_size}  "
          f"|  Mode: {'FULL' if args.full else 'ROLLING'}\n")

    records: list[dict] = []
    total = len(cases)
    first_failure_idx: int | None = None

    for batch_start in range(0, total, args.batch_size):
        batch = cases[batch_start: batch_start + args.batch_size]
        batch_failed = False
        for offset, (path, expected_ai) in enumerate(batch):
            idx = batch_start + offset + 1
            rec = _run_case(analyzer, path, expected_ai)
            records.append(rec)
            _print_case(idx, total, rec)
            if "passed" in rec and not rec["passed"]:
                batch_failed = True
                if first_failure_idx is None:
                    first_failure_idx = idx
            # gentle pacing — Gemini RPM headroom
            time.sleep(0.5)

        if batch_failed and not args.full:
            print(f"\nBatch {batch_start // args.batch_size + 1} failed — stopping rolling eval.")
            break

    mode = "full" if args.full else "rolling"
    out_path = _write_report(records, args.engine, mode)
    _summarise(records)
    print(f"\nReport: {out_path.relative_to(_REPO_ROOT)}")

    has_failures = any("passed" in r and not r["passed"] for r in records)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
