"""
5-image / deferred-fix verification harness.

Runs a fixed 5-image set (3 AI + 2 REAL) through the ensemble engine N times
each and prints a verdict table. Used to compare detection accuracy before
and after each deferred fix from docs/DETECTION_V2_DEFERRED_FIXES.md.

Usage:
    python scripts/eval_5_phase.py            # 2 runs per image
    python scripts/eval_5_phase.py --runs 3   # bump for tighter variance reads
    python scripts/eval_5_phase.py --label "D4 after"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("DETECTION_ENGINE", "ensemble")

from app.detection.ensemble_engine import analyze_image_ensemble_async  # noqa: E402

TEST_SET = [
    ("first_image", "/Users/netanel.ossi/Downloads/first_image.png", True),
    ("130188",      "/Users/netanel.ossi/Downloads/130188.jpg",       True),
    ("sofa",        "/Users/netanel.ossi/Downloads/sofa.jpeg",        True),
    ("fiverr",      "/Users/netanel.ossi/Downloads/fiverr.jpeg",      False),
    ("linkdin",     "/Users/netanel.ossi/Downloads/linkdin profile.jpeg", False),
]


async def run(runs: int, label: str) -> None:
    print(f"\n=== {label}  ({runs} runs/image) ===")
    pass_count = 0
    total = 0
    voter_wins: dict[str, int] = {}
    for case_label, path, expected_ai in TEST_SET:
        for i in range(runs):
            t0 = time.perf_counter()
            r = await analyze_image_ensemble_async(path)
            ms = round((time.perf_counter() - t0) * 1000)
            conf = r.get("confidence", -1.0)
            verdict = "AI  " if conf > 0.5 else "REAL"
            expected = "AI  " if expected_ai else "REAL"
            ok = (verdict.strip() == "AI") == expected_ai
            outcome = "PASS" if ok else "FAIL"
            if ok:
                pass_count += 1
            total += 1
            # Identify which voter "won" — most-confident OK voter
            winners = [v for v in r.get("ensemble_voters", []) if v.get("ok")]
            top = max(winners, key=lambda v: v["confidence"]) if winners else None
            wlabel = top["label"] if top else "—"
            voter_wins[wlabel] = voter_wins.get(wlabel, 0) + 1
            print(f"  {case_label:<12} run{i+1} | exp={expected} got={verdict} "
                  f"conf={conf:.2f} {ms:>5}ms top={wlabel:<14} {outcome}")
    print(f"  Accuracy: {pass_count}/{total} ({pass_count / total * 100:.1f}%)")
    print(f"  Voter-wins: " + ", ".join(f"{k}={v}" for k, v in sorted(voter_wins.items())))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--label", type=str, default="baseline")
    args = parser.parse_args()
    asyncio.run(run(args.runs, args.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
