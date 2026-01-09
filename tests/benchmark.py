import argparse
import asyncio
import os
import logging
import warnings
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

# Ensure repo root is on sys.path so `import app` works when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class SampleResult:
    path: str
    label: str  # "ai" or "original"
    predicted: str
    correct: bool
    bypassed: bool
    meta_human: float
    meta_ai: float
    meta_exit: str  # "original" | "ai" | "gpu"
    meta_correct: bool
    summary: str
    l2_status: str
    signals: List[str]

@dataclass
class MetaSample:
    path: str
    label: str
    human_score: float
    ai_score: float
    # Optional debugging payload (kept for reporting; not required for sweep decisions)
    human_signals: List[str]


def infer_label_from_path(path: str) -> Optional[str]:
    """Infer ground truth label from folder names."""
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    # common conventions in this repo
    ai_markers = {"ai", "aiartdata"}
    original_markers = {"original", "real", "realart"}

    if any(p in ai_markers for p in parts):
        return "ai"
    if any(p in original_markers for p in parts):
        return "original"
    return None


def is_bypassed(result: dict) -> bool:
    if result.get("gpu_bypassed") is True:
        return True
    l2 = (result.get("layers") or {}).get("layer2_forensics") or {}
    return l2.get("status") == "skipped"


def predict_label(result: dict) -> str:
    """Convert detector output into binary label."""
    summary = (result.get("summary") or "").lower()
    l2 = (result.get("layers") or {}).get("layer2_forensics") or {}
    status = (l2.get("status") or "").lower()
    prob = l2.get("probability", 0.0)

    # Prefer explicit status
    if status in {"detected", "suspicious"}:
        return "ai"
    if status in {"not_detected"}:
        return "original"

    # Fallback to summary keywords
    if "verified ai" in summary or "likely ai" in summary or "possible ai" in summary:
        return "ai"
    if "suspicious" in summary:
        return "ai"
    if "verified original" in summary or "likely original" in summary or "human" in summary:
        return "original"

    # Last resort threshold
    try:
        return "ai" if float(prob) > 0.5 else "original"
    except Exception:
        return "original"


class NoCache:
    def get(self, _key):
        return None

    def put(self, _key, _value):
        return None


def patch_detector_for_benchmark():
    """
    Patch detector module globals:
    - disable cache
    - mock GPU calls to return correct verdict based on path label
    """
    from app.detectors import core as detector_core

    # Ensure detector runs in tuning semantics:
    # - no cache
    # - do not override (mocked) GPU outputs using metadata conflict resolution
    os.environ["AI_DETECTOR_BENCHMARK"] = "1"

    detector_core.forensic_cache = NoCache()

    # Track whether the detector attempted to call "GPU" per file.
    # Keyed by absolute file path.
    gpu_called: Dict[str, int] = defaultdict(int)
    detector_core.__benchmark_gpu_called__ = gpu_called  # type: ignore[attr-defined]
    detector_core.__benchmark_current_path__ = ""  # type: ignore[attr-defined]

    async def mock_run_deep_forensics(file_path_or_img):
        # Detect label from file path if possible
        file_path = file_path_or_img if isinstance(file_path_or_img, str) else ""
        if file_path:
            gpu_called[os.path.abspath(file_path)] += 1
        label = infer_label_from_path(file_path) or "original"
        return {
            "ai_score": 0.95 if label == "ai" else 0.05,
            # Non-zero so detect_ai_media doesn't mark this as bypassed.
            # Benchmark still computes bypass using metadata policy (meta_pred).
            "gpu_time_ms": 1.0,
        }

    async def mock_run_batch_forensics(frames):
        # For videos, infer label from the currently-running file path (set by run_one()).
        current_path = ""
        if hasattr(detector_core, "__benchmark_current_path__"):
            current_path = str(getattr(detector_core, "__benchmark_current_path__", ""))
        if current_path:
            gpu_called[os.path.abspath(current_path)] += 1
        label = infer_label_from_path(current_path) or "original"
        return {
            "results": [{"ai_score": 0.95 if label == "ai" else 0.05} for _ in (frames or [])],
            "gpu_time_ms": 1.0,
        }

    detector_core.run_deep_forensics = mock_run_deep_forensics
    detector_core.run_batch_forensics = mock_run_batch_forensics


def iter_media_files(root: str, include_videos: bool) -> List[str]:
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS or (include_videos and ext in VID_EXTS):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def iter_datasets(base_root: str, dataset_names: List[str]) -> List[Tuple[str, List[str]]]:
    out = []
    for ds in dataset_names:
        ds_root = os.path.abspath(os.path.join(base_root, ds))
        if os.path.exists(ds_root):
            out.append((ds, ds_root))
    return out

def discover_labeled_datasets(base_root: str) -> List[Tuple[str, str]]:
    """
    Discover dataset roots that contain at least one of:
      - ai/ + original/
      - AI/ + Real/
      - AiArtData/ + RealArt/
    We treat each discovered root as a separate dataset.
    """
    pairs = [
        ({"ai", "original"}, "ai/original"),
        ({"aiartdata", "realart"}, "AiArtData/RealArt"),
        ({"ai", "real"}, "AI/Real"),
    ]
    found: List[Tuple[str, str]] = []
    base_root = os.path.abspath(base_root)
    for dirpath, dirnames, _filenames in os.walk(base_root):
        dn = {d.lower() for d in dirnames}
        for required, _tag in pairs:
            if required.issubset(dn):
                # Use relative name for readability
                rel = os.path.relpath(dirpath, base_root)
                name = rel.replace(os.sep, "/")
                # Skip the root itself; it mixes multiple datasets.
                if name == ".":
                    break
                found.append((name, dirpath))
                break
    # De-dupe & stable sort
    uniq = {}
    for name, path in found:
        uniq[path] = name
    return sorted([(name, path) for path, name in uniq.items()], key=lambda x: x[0])

def metadata_decision(
    label: str,
    human_score: float,
    ai_score: float,
    human_signals: List[str],
    *,
    human_exit_high: float,
    human_exit_low: float,
    human_low_ai_max: float,
    human_low_no_ai_min: float,
    ai_exit_meta: float,
) -> Tuple[bool, str, Optional[str], bool]:
    """
    Returns (bypassed, meta_exit, meta_pred, meta_correct_under_gpu_perfect).
    If meta_pred is None => sent to GPU => correct (GPU perfect assumption).
    """
    # 1) Verified Original (high-bypass policy: score-only, mirrors production)
    if human_score >= human_exit_high:
        meta_exit = "original"
        meta_pred = "original"
    # 2) Likely Original
    elif (human_score >= human_exit_low and ai_score < human_low_ai_max) or (ai_score == 0 and human_score >= human_low_no_ai_min):
        meta_exit = "original"
        meta_pred = "original"
    # 3) Likely AI
    elif ai_score >= ai_exit_meta:
        meta_exit = "ai"
        meta_pred = "ai"
    else:
        meta_exit = "gpu"
        meta_pred = None

    bypassed = meta_pred is not None
    meta_correct = True if meta_pred is None else (meta_pred == label)
    return bypassed, meta_exit, meta_pred, meta_correct

def evaluate_thresholds(
    samples: List[MetaSample],
    *,
    human_exit_high: float,
    human_exit_low: float,
    human_low_ai_max: float,
    human_low_no_ai_min: float,
    ai_exit_meta: float,
) -> Dict[str, float]:
    total = len(samples)
    bypass = 0
    correct = 0
    fp = 0
    fn = 0
    gpu = 0
    for s in samples:
        bypassed, meta_exit, meta_pred, meta_correct = metadata_decision(
            s.label,
            s.human_score,
            s.ai_score,
            s.human_signals,
            human_exit_high=human_exit_high,
            human_exit_low=human_exit_low,
            human_low_ai_max=human_low_ai_max,
            human_low_no_ai_min=human_low_no_ai_min,
            ai_exit_meta=ai_exit_meta,
        )
        if bypassed:
            bypass += 1
            if meta_exit == "ai" and s.label == "original":
                fp += 1
            if meta_exit == "original" and s.label == "ai":
                fn += 1
        else:
            gpu += 1
        if meta_correct:
            correct += 1

    return {
        "total": total,
        "bypass": bypass,
        "bypass_pct": (bypass / total * 100.0) if total else 0.0,
        "accuracy": correct,
        "accuracy_pct": (correct / total * 100.0) if total else 0.0,
        "fp": fp,
        "fn": fn,
        "gpu": gpu,
    }

def load_meta_samples(paths: List[str]) -> List[MetaSample]:
    from app.detectors.metadata import get_forensic_metadata_score, get_ai_suspicion_score
    from app.detectors.utils import get_exif_data
    from PIL import Image

    samples: List[MetaSample] = []
    for path in paths:
        label = infer_label_from_path(path)
        if label is None:
            continue
        exif = {}
        width = height = 0
        file_size = 0
        try:
            exif = get_exif_data(path)
            with Image.open(path) as img:
                width, height = img.size
            file_size = os.path.getsize(path)
        except Exception:
            exif = {}
        human_score, human_signals = get_forensic_metadata_score(exif)
        ai_score, _ = get_ai_suspicion_score(exif, width, height, file_size, filename=os.path.basename(path))
        samples.append(
            MetaSample(
                path=path,
                label=label,
                human_score=float(human_score),
                ai_score=float(ai_score),
                human_signals=[str(s) for s in (human_signals or [])],
            )
        )
    return samples


def load_meta_samples_with_videos(paths: List[str]) -> List[MetaSample]:
    """
    Like load_meta_samples, but also includes videos by using video-metadata scoring
    as the metadata feature vector. This helps sweep across video-only datasets too.
    """
    from app.detectors.metadata import get_forensic_metadata_score, get_ai_suspicion_score
    from app.detectors.utils import get_exif_data
    from app.detectors.video import get_video_metadata, get_video_metadata_score
    from PIL import Image
    import asyncio

    async def get_video_scores(p: str) -> Tuple[float, float]:
        meta = await get_video_metadata(p)
        h, a, _signals, _exit = get_video_metadata_score(meta, filename=os.path.basename(p), file_path=p)
        return float(h), float(a)

    samples: List[MetaSample] = []
    loop = asyncio.get_event_loop()
    for path in paths:
        label = infer_label_from_path(path)
        if label is None:
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in VID_EXTS:
            try:
                h, a = loop.run_until_complete(get_video_scores(path))
                # Video samples don't have "human_signals" in this sweep (image-only for now).
                samples.append(MetaSample(path=path, label=label, human_score=h, ai_score=a, human_signals=[]))
            except Exception:
                continue
            continue

        exif = {}
        width = height = 0
        file_size = 0
        try:
            exif = get_exif_data(path)
            with Image.open(path) as img:
                width, height = img.size
            file_size = os.path.getsize(path)
        except Exception:
            exif = {}
        human_score, human_signals = get_forensic_metadata_score(exif)
        ai_score, _ = get_ai_suspicion_score(exif, width, height, file_size, filename=os.path.basename(path))
        samples.append(
            MetaSample(
                path=path,
                label=label,
                human_score=float(human_score),
                ai_score=float(ai_score),
                human_signals=[str(s) for s in (human_signals or [])],
            )
        )

    return samples

def sweep_thresholds(
    datasets: Dict[str, List[MetaSample]],
    *,
    min_accuracy_pct: float,
) -> Dict[str, float]:
    """
    Brute-force grid search. Objective: maximize GLOBAL bypass% across all samples
    while satisfying GLOBAL accuracy >= min_accuracy_pct.
    (Accuracy assumes GPU is perfect; only metadata early-exits can be wrong.)
    """
    # Search space (kept compact to run fast, but wide enough to find true maxima)
    # Goal: maximize GLOBAL bypass% while keeping GLOBAL accuracy >= min_accuracy_pct.
    #
    # Notes:
    # - Lowering HUMAN_* thresholds increases "Likely Original" bypass but can create AI false negatives.
    # - Lowering AI_EXIT_META increases "Likely AI" bypass but can create false positives on stripped-metadata reals.
    human_exit_high_grid = [0.15, 0.20, 0.25, 0.30, 0.35]
    human_exit_low_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    human_low_ai_max_grid = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.30, 0.35]
    human_low_no_ai_min_grid = [0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]
    ai_exit_meta_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    best = None
    best_score = (-1.0, -1.0, 0.0)  # (global_bypass, global_accuracy, -errors)

    for heh in human_exit_high_grid:
        for hel in human_exit_low_grid:
            if hel > heh:
                continue
            for hlam in human_low_ai_max_grid:
                for hlnam in human_low_no_ai_min_grid:
                    if hlnam < hel:
                        continue
                    for aem in ai_exit_meta_grid:
                        per_ds = {}
                        global_total = 0
                        global_bypass = 0
                        global_correct = 0
                        global_fp = 0
                        global_fn = 0
                        for name, samples in datasets.items():
                            m = evaluate_thresholds(
                                samples,
                                human_exit_high=heh,
                                human_exit_low=hel,
                                human_low_ai_max=hlam,
                                human_low_no_ai_min=hlnam,
                                ai_exit_meta=aem,
                            )
                            per_ds[name] = m
                            global_total += int(m["total"])
                            global_bypass += int(m["bypass"])
                            global_correct += int(m["accuracy"])
                            global_fp += int(m["fp"])
                            global_fn += int(m["fn"])

                        if global_total == 0:
                            continue

                        global_bypass_pct = (global_bypass / global_total) * 100.0
                        global_acc_pct = (global_correct / global_total) * 100.0
                        if global_acc_pct < min_accuracy_pct:
                            continue

                        total_errors = global_fp + global_fn
                        score = (global_bypass_pct, global_acc_pct, -float(total_errors))
                        if score > best_score:
                            best_score = score
                            best = {
                                "human_exit_high": heh,
                                "human_exit_low": hel,
                                "human_low_ai_max": hlam,
                                "human_low_no_ai_min": hlnam,
                                "ai_exit_meta": aem,
                                "global_bypass_pct": global_bypass_pct,
                                "global_accuracy_pct": global_acc_pct,
                                "global_errors": total_errors,
                            }
    return best or {}


def summarize_common_signals(results: List[SampleResult], kind: str) -> List[Tuple[str, int]]:
    c = Counter()
    for r in results:
        if kind == "fp" and (r.label == "original" and r.predicted == "ai"):
            c.update(r.signals)
        if kind == "fn" and (r.label == "ai" and r.predicted == "original"):
            c.update(r.signals)
    return c.most_common(20)

def summarize_common_meta_signals(results: List[SampleResult], kind: str) -> List[Tuple[str, int]]:
    """
    For FP/FN in this benchmark, the only errors come from metadata early-exits.
    So the most useful explanation is the metadata signals (not the generic layer2
    consensus list).
    """
    c = Counter()
    for r in results:
        if kind == "fp" and (r.label == "original" and r.meta_exit == "ai" and r.bypassed):
            c.update(r.signals)
        if kind == "fn" and (r.label == "ai" and r.meta_exit == "original" and r.bypassed):
            c.update(r.signals)
    return c.most_common(20)

async def run_one(path: str) -> Optional[SampleResult]:
    from app.detectors.core import detect_ai_media
    from app.detectors import core as detector_core
    from app.detectors.metadata import get_forensic_metadata_score, get_ai_suspicion_score
    from PIL import Image
    from app.scoring_config import ScoringConfig
    from app.detectors.video import get_video_metadata, get_video_metadata_score

    label = infer_label_from_path(path)
    if label is None:
        return None

    abs_path = os.path.abspath(path)
    # Allow the mocked batch GPU path to infer correct label for videos.
    if hasattr(detector_core, "__benchmark_current_path__"):
        detector_core.__benchmark_current_path__ = abs_path  # type: ignore[attr-defined]
    # Reset per-sample GPU call counter
    if hasattr(detector_core, "__benchmark_gpu_called__"):
        detector_core.__benchmark_gpu_called__[abs_path] = 0  # type: ignore[index]

    # ---- Metadata-only decision (what we'd do if we maximized bypass) ----
    exif = {}
    width = height = 0
    file_size = 0
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in VID_EXTS:
            # For videos, use ffprobe-based metadata scoring (no PIL Image.open).
            meta = await get_video_metadata(path)
            h, a, _signals, _exit = get_video_metadata_score(meta, filename=os.path.basename(path), file_path=path)
            human_score = float(h)
            ai_score = float(a)
            human_signals = _signals
        else:
            from app.detectors.utils import get_exif_data
            exif = get_exif_data(path)
            with Image.open(path) as img:
                width, height = img.size
            file_size = os.path.getsize(path)
            human_score, human_signals = get_forensic_metadata_score(exif)
            ai_score, _ai_signals = get_ai_suspicion_score(
                exif, width, height, file_size, filename=os.path.basename(path)
            )
    except Exception:
        exif = {}
        human_score, human_signals = 0.0, []
        ai_score = 0.0

    # Mirror current early-exit policy (read from ScoringConfig)
    bypassed, meta_exit, meta_pred, meta_correct = metadata_decision(
        label,
        float(human_score),
        float(ai_score),
        [str(s) for s in (human_signals or [])],
        human_exit_high=float(ScoringConfig.THRESHOLDS["HUMAN_EXIT_HIGH"]),
        human_exit_low=float(ScoringConfig.THRESHOLDS["HUMAN_EXIT_LOW"]),
        human_low_ai_max=float(ScoringConfig.THRESHOLDS.get("HUMAN_LOW_AI_MAX", 0.10)),
        human_low_no_ai_min=float(ScoringConfig.THRESHOLDS.get("HUMAN_LOW_NO_AI_MIN", 0.25)),
        ai_exit_meta=float(ScoringConfig.THRESHOLDS["AI_EXIT_META"]),
    )

    result = await detect_ai_media(path, trusted_metadata=None, original_filename=os.path.basename(path))
    predicted = predict_label(result)
    gpu_called = 0
    if hasattr(detector_core, "__benchmark_gpu_called__"):
        gpu_called = int(detector_core.__benchmark_gpu_called__.get(abs_path, 0))  # type: ignore[attr-defined]
    # Bypass (for cost) should match meta_exit: if meta_exit != gpu, we bypassed GPU.
    bypassed = meta_pred is not None

    # For FP/FN, we care about *metadata* signals that triggered the early exit.
    # When we go to GPU, layer2 contains generic consensus signals; in that case
    # we keep it, but FP/FN lists only include bypassed samples anyway.
    l2 = (result.get("layers") or {}).get("layer2_forensics") or {}
    signals = l2.get("signals") or []
    if not isinstance(signals, list):
        signals = [str(signals)]

    return SampleResult(
        path=path,
        label=label,
        predicted=predicted,
        # Overall accuracy with perfect GPU: only meta early exits can be wrong.
        correct=meta_correct,
        bypassed=bypassed,
        meta_human=float(human_score),
        meta_ai=float(ai_score),
        meta_exit=meta_exit,
        meta_correct=meta_correct,
        summary=result.get("summary", ""),
        l2_status=l2.get("status", ""),
        signals=[str(s) for s in signals],
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.path.expanduser(os.getenv("AI_DETECTOR_DATASETS_ROOT", "tests/data")),
        help="Dataset root folder (or set AI_DETECTOR_DATASETS_ROOT)",
    )
    ap.add_argument("--dataset", default="", help="Optional subfolder under root (e.g., hf_200_benchmark)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = no limit)")
    ap.add_argument("--include-videos", action="store_true", help="Include videos (slower)")
    ap.add_argument("--sweep", action="store_true", help="Run threshold sweep (metadata-only, GPU mocked)")
    ap.add_argument("--min-accuracy", type=float, default=95.0, help="Minimum accuracy percentage constraint for sweep")
    args = ap.parse_args()

    root = os.path.join(args.root, args.dataset) if args.dataset else args.root
    root = os.path.abspath(root)

    # Reduce log noise from detector internals during benchmarks
    logging.getLogger().setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)

    patch_detector_for_benchmark()

    if args.sweep:
        # Sweep across ALL discovered labeled datasets under tests/data.
        base_root = os.path.abspath(args.root)
        ds_roots = discover_labeled_datasets(base_root)
        datasets: Dict[str, List[MetaSample]] = {}
        for name, ds_root in ds_roots:
            # Images only for sweep (video-only datasets will be skipped).
            files = iter_media_files(ds_root, include_videos=False)
            if args.limit and args.limit > 0:
                files = files[: args.limit]
            datasets[name] = load_meta_samples(files)

        # Drop any dataset that still has 0 samples (should be rare).
        datasets = {k: v for k, v in datasets.items() if len(v) > 0}

        best = sweep_thresholds(datasets, min_accuracy_pct=args.min_accuracy)
        if not best:
            print("No configuration found meeting constraints.")
            return

        print("=== Best thresholds (grid search) ===")
        print(best)
        print("")
        for name, samples in datasets.items():
            m = evaluate_thresholds(
                samples,
                human_exit_high=best["human_exit_high"],
                human_exit_low=best["human_exit_low"],
                human_low_ai_max=best["human_low_ai_max"],
                human_low_no_ai_min=best["human_low_no_ai_min"],
                ai_exit_meta=best["ai_exit_meta"],
            )
            print(f"[{name}] bypass={m['bypass_pct']:.2f}% acc={m['accuracy_pct']:.2f}% fp={int(m['fp'])} fn={int(m['fn'])} gpu={int(m['gpu'])}/{int(m['total'])}")
        return

    files = iter_media_files(root, include_videos=args.include_videos)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    results: List[SampleResult] = []
    for p in files:
        r = await run_one(p)
        if r:
            results.append(r)

    total = len(results)
    if total == 0:
        print("No labeled samples found under:", root)
        return

    correct = sum(1 for r in results if r.correct)
    bypass = sum(1 for r in results if r.bypassed)
    # Meta-only errors are the only possible errors under "GPU perfect" assumption.
    fp = [r for r in results if r.bypassed and r.label == "original" and r.meta_exit == "ai"]
    fn = [r for r in results if r.bypassed and r.label == "ai" and r.meta_exit == "original"]
    gpu_needed = [r for r in results if not r.bypassed]

    print("=== Benchmark Report ===")
    print("Root:", root)
    print(f"Samples: {total}")
    print(f"Bypass%: {bypass/total*100:.2f}% ({bypass}/{total})")
    print(f"Accuracy%: {correct/total*100:.2f}% ({correct}/{total})")
    print("")
    print(f"False Positives (original→ai): {len(fp)} (metadata early-exit mistakes)")
    print(f"False Negatives (ai→original): {len(fn)} (metadata early-exit mistakes)")
    print(f"Sent to GPU (ambiguous): {len(gpu_needed)}")
    print("")

    print("Top FP signals:")
    for sig, n in summarize_common_meta_signals(results, "fp")[:10]:
        print(f"  {n:>4}  {sig}")

    print("")
    print("Top FN signals:")
    for sig, n in summarize_common_meta_signals(results, "fn")[:10]:
        print(f"  {n:>4}  {sig}")

    # Show a few concrete examples for debugging/tuning
    def show_examples(title: str, items: List[SampleResult]):
        print("")
        print(title)
        for r in items[:10]:
            print(f"- {r.path}")
            print(
                f"  label={r.label} meta_exit={r.meta_exit} meta_human={r.meta_human:.2f} meta_ai={r.meta_ai:.2f} "
                f"bypassed={r.bypassed} summary={r.summary}"
            )
            if r.signals:
                print(f"  meta_signals: {', '.join(r.signals[:8])}{' ...' if len(r.signals) > 8 else ''}")

    show_examples("FP examples (first 10):", fp)
    show_examples("FN examples (first 10):", fn)


if __name__ == "__main__":
    asyncio.run(main())


