#!/usr/bin/env python3
"""
Systematic Grid Search Threshold Tuning
Optimizes thresholds to maximize GPU bypass while maintaining 95%+ accuracy.

Objective Function:
- GPU Bypass: 60% weight (cost savings is priority!)
- Real Safety: 30% weight (must maintain safety)
- AI Detection: 10% weight (acceptable to miss some AI if it saves money)
"""

import os
import sys
import json
import requests
import subprocess
import time
from typing import Dict, List, Tuple

API_URL = "http://localhost:8000/detect"
SCORING_CONFIG_PATH = "app/scoring_config.py"

# Datasets for tuning (use smaller subsets for speed)
TUNE_DATASETS = [
    ("Kaggle_Sample", "tests/data/kaggle_benchmark/AiArtData/AiArtData", "tests/data/kaggle_benchmark/RealArt/RealArt", 50, 50),
    ("HF_200", "tests/data/hf_200_benchmark/AI", "tests/data/hf_200_benchmark/Real", 100, 100),
]

def get_sample_files(directory, prefix, count):
    """Get sample files from directory."""
    if not os.path.exists(directory):
        return []
    files = [os.path.join(directory, f) for f in os.listdir(directory) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if prefix:
        files = [f for f in files if os.path.basename(f).lower().startswith(prefix)]
    return files[:count]

def update_threshold(threshold_name: str, value: float):
    """Update a threshold in scoring_config.py."""
    with open(SCORING_CONFIG_PATH, 'r') as f:
        content = f.read()
    
    # Find and replace the threshold value
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if f'"{threshold_name}":' in line:
            # Extract current value and replace
            parts = line.split(':')
            if len(parts) >= 2:
                indent = line[:len(line) - len(line.lstrip())]
                comment = ''
                if '#' in parts[1]:
                    comment = ' ' + parts[1].split('#', 1)[1]
                lines[i] = f'{indent}"{threshold_name}": {value},{comment}'
                break
    
    with open(SCORING_CONFIG_PATH, 'w') as f:
        f.write('\n'.join(lines))

def restart_server():
    """Restart the FastAPI server."""
    subprocess.run("kill $(pgrep -f uvicorn) || true", shell=True, capture_output=True)
    time.sleep(1)
    subprocess.Popen(
        ["/Users/netanel.ossi/Desktop/ai detector/.venv/bin/uvicorn", "app.main:app", "--port", "8000"],
        cwd="/Users/netanel.ossi/Desktop/ai detector",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3)

def run_benchmark_fast(ai_files: List[str], real_files: List[str]) -> Dict:
    """Run benchmark and return metrics."""
    ai_passed = 0
    real_passed = 0
    gpu_bypasses = 0
    
    for f in ai_files:
        try:
            filename = f.replace('tests/data/', '')
            with open(f, 'rb') as img:
                r = requests.post(API_URL, files={'file': (filename, img)}, timeout=5)
                data = r.json()
                if "AI" in data.get('summary', ''): ai_passed += 1
                if data.get('gpu_bypassed', False): gpu_bypasses += 1
        except:
            pass
    
    for f in real_files:
        try:
            filename = f.replace('tests/data/', '')
            with open(f, 'rb') as img:
                r = requests.post(API_URL, files={'file': (filename, img)}, timeout=5)
                data = r.json()
                summary = data.get('summary', '')
                if "Original" in summary or "Human" in summary: real_passed += 1
                if data.get('gpu_bypassed', False): gpu_bypasses += 1
        except:
            pass
    
    total = len(ai_files) + len(real_files)
    return {
        "ai_acc": ai_passed / len(ai_files) if ai_files else 0,
        "real_acc": real_passed / len(real_files) if real_files else 0,
        "bypass_rate": gpu_bypasses / total if total > 0 else 0,
        "total": total
    }

def objective_function(ai_acc: float, real_acc: float, bypass_rate: float) -> float:
    """
    Objective function to maximize.
    Prioritizes GPU bypass (cost savings) while maintaining safety.
    """
    # Hard constraint: Real safety must be >= 95%
    if real_acc < 0.95:
        return 0.0
    
    # Weighted score
    score = (
        (bypass_rate * 0.60) +  # 60% weight on GPU bypass (COST SAVINGS!)
        (real_acc * 0.30) +      # 30% weight on Real safety
        (ai_acc * 0.10)          # 10% weight on AI detection
    )
    return score

def tune_threshold(threshold_name: str, value_range: List[float]) -> Tuple[float, Dict]:
    """
    Tune a single threshold across a range of values.
    Returns best value and its metrics.
    """
    print(f"\n{'='*80}")
    print(f"TUNING: {threshold_name}")
    print(f"{'='*80}")
    print(f"Testing {len(value_range)} values: {value_range}")
    
    best_value = None
    best_score = -1
    best_metrics = None
    results = []
    
    for value in value_range:
        print(f"\nTesting {threshold_name} = {value}...")
        update_threshold(threshold_name, value)
        restart_server()
        
        # Run on all tune datasets
        total_ai_acc = 0
        total_real_acc = 0
        total_bypass = 0
        total_samples = 0
        
        for name, ai_dir, real_dir, ai_count, real_count in TUNE_DATASETS:
            ai_files = get_sample_files(ai_dir, None, ai_count)
            real_files = get_sample_files(real_dir, None, real_count)
            
            metrics = run_benchmark_fast(ai_files, real_files)
            total_ai_acc += metrics['ai_acc'] * len(ai_files)
            total_real_acc += metrics['real_acc'] * len(real_files)
            total_bypass += metrics['bypass_rate'] * metrics['total']
            total_samples += metrics['total']
        
        # Calculate averages
        avg_ai_acc = total_ai_acc / (sum(c[3] for c in TUNE_DATASETS))
        avg_real_acc = total_real_acc / (sum(c[4] for c in TUNE_DATASETS))
        avg_bypass = total_bypass / total_samples
        
        score = objective_function(avg_ai_acc, avg_real_acc, avg_bypass)
        
        result = {
            "value": value,
            "ai_acc": avg_ai_acc,
            "real_acc": avg_real_acc,
            "bypass_rate": avg_bypass,
            "score": score
        }
        results.append(result)
        
        print(f"  AI: {avg_ai_acc*100:.1f}% | Real: {avg_real_acc*100:.1f}% | Bypass: {avg_bypass*100:.1f}% | Score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_value = value
            best_metrics = result
    
    print(f"\n{'='*80}")
    print(f"BEST VALUE for {threshold_name}: {best_value}")
    print(f"  AI: {best_metrics['ai_acc']*100:.1f}% | Real: {best_metrics['real_acc']*100:.1f}% | Bypass: {best_metrics['bypass_rate']*100:.1f}%")
    print(f"  Score: {best_score:.3f}")
    print(f"{'='*80}")
    
    # Save results
    with open(f"tuning_results_{threshold_name}.json", 'w') as f:
        json.dump({"best": best_metrics, "all_results": results}, f, indent=2)
    
    return best_value, best_metrics

if __name__ == "__main__":
    print("SYSTEMATIC GRID SEARCH THRESHOLD TUNING")
    print("Objective: Maximize GPU bypass while maintaining 95%+ accuracy")
    print("="*80)
    
    # Define thresholds to tune and their ranges
    THRESHOLDS_TO_TUNE = [
        # Start with early exit thresholds (biggest impact on bypass rate)
        ("AI_EXIT_META", [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]),
        ("HUMAN_EXIT_HIGH", [0.50, 0.55, 0.60, 0.65, 0.70]),
        ("HUMAN_EXIT_LOW", [0.30, 0.35, 0.40, 0.45, 0.50]),
        # Then conflict resolution
        ("CONFLICT_AI_SCORE", [0.15, 0.20, 0.25, 0.30, 0.35]),
        ("CONFLICT_MODEL_LOW", [0.75, 0.80, 0.85, 0.90, 0.95]),
    ]
    
    optimal_values = {}
    
    for threshold_name, value_range in THRESHOLDS_TO_TUNE:
        best_value, metrics = tune_threshold(threshold_name, value_range)
        optimal_values[threshold_name] = best_value
        
        # Update to best value before tuning next threshold
        update_threshold(threshold_name, best_value)
    
    # Final summary
    print(f"\n{'='*80}")
    print("OPTIMAL THRESHOLD VALUES")
    print(f"{'='*80}")
    for name, value in optimal_values.items():
        print(f"{name}: {value}")
    
    print(f"\n{'='*80}")
    print("Tuning complete! Restart server with optimal values.")
    print(f"{'='*80}")
