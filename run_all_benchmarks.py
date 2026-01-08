#!/usr/bin/env python3
"""
Comprehensive Benchmark Script - All Datasets
Tests all 9 available datasets with mocked GPU for threshold fine-tuning.
"""

import os
import requests
import json

API_URL = "http://localhost:8000/detect"
DATA_ROOT = os.path.expanduser(os.getenv("AI_DETECTOR_DATASETS_ROOT", "tests/data"))

# Dataset configurations: (name, ai_dir, real_dir, ai_prefix, real_prefix)
DATASETS = [
    ("Kaggle", os.path.join(DATA_ROOT, "kaggle_benchmark/AiArtData/AiArtData"), os.path.join(DATA_ROOT, "kaggle_benchmark/RealArt/RealArt"), None, None),
    ("HF_200", os.path.join(DATA_ROOT, "hf_200_benchmark/AI"), os.path.join(DATA_ROOT, "hf_200_benchmark/Real"), "ai_hf_", "real_hf_"),
    ("GenImage", os.path.join(DATA_ROOT, "genimage_benchmark"), os.path.join(DATA_ROOT, "genimage_benchmark"), "gen_any_", "nature_"),
    ("Benchmark_50", os.path.join(DATA_ROOT, "benchmark_50/ai"), os.path.join(DATA_ROOT, "benchmark_50/original"), None, None),
    ("Benchmark_HF_50", os.path.join(DATA_ROOT, "benchmark_hf_50/ai"), os.path.join(DATA_ROOT, "benchmark_hf_50/original"), None, None),
    ("User_Data", os.path.join(DATA_ROOT, "ai/images"), os.path.join(DATA_ROOT, "original/images"), None, None),
]

def get_files(directory, prefix=None):
    if not os.path.exists(directory):
        return []
    files = [os.path.join(directory, f) for f in os.listdir(directory) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if prefix:
        files = [f for f in files if os.path.basename(f).lower().startswith(prefix)]
    return files

def run_benchmark(name, ai_files, real_files):
    print(f"\n{'='*80}")
    print(f"{name} BENCHMARK")
    print(f"{'='*80}")
    
    ai_passed = 0
    real_passed = 0
    gpu_bypasses = 0
    
    # Test AI files
    for f in ai_files:
        try:
            # Pass the original path as filename for ground truth detection
            filename = f.replace(DATA_ROOT.rstrip("/") + "/", "")  # Keep relative path for mock detection
            with open(f, 'rb') as img:
                r = requests.post(API_URL, files={'file': (filename, img)}, timeout=10)
                data = r.json()
                summary = data.get('summary', 'Error')
                is_ai = "AI" in summary
                gpu_bypassed = data.get('gpu_bypassed', False)
                if gpu_bypassed: gpu_bypasses += 1
                if is_ai: ai_passed += 1
        except Exception as e:
            print(f"ERROR on {os.path.basename(f)}: {e}")
    
    # Test Real files
    for f in real_files:
        try:
            # Pass the original path as filename for ground truth detection
            filename = f.replace(DATA_ROOT.rstrip("/") + "/", "")  # Keep relative path for mock detection
            with open(f, 'rb') as img:
                r = requests.post(API_URL, files={'file': (filename, img)}, timeout=10)
                data = r.json()
                summary = data.get('summary', 'Error')
                is_real = "Original" in summary or "Human" in summary
                gpu_bypassed = data.get('gpu_bypassed', False)
                if gpu_bypassed: gpu_bypasses += 1
                if is_real: real_passed += 1
        except Exception as e:
            print(f"ERROR on {os.path.basename(f)}: {e}")
    
    total = len(ai_files) + len(real_files)
    ai_acc = (ai_passed / len(ai_files) * 100) if ai_files else 0
    real_acc = (real_passed / len(real_files) * 100) if real_files else 0
    bypass_rate = (gpu_bypasses / total * 100) if total > 0 else 0
    
    print(f"AI Detection:  {ai_passed}/{len(ai_files)} ({ai_acc:.1f}%)")
    print(f"Real Safety:   {real_passed}/{len(real_files)} ({real_acc:.1f}%)")
    print(f"GPU Bypassed:  {gpu_bypasses}/{total} ({bypass_rate:.1f}%)")
    print(f"Overall Acc:   {(ai_passed + real_passed)}/{total} ({(ai_passed + real_passed)/total*100:.1f}%)")
    
    return {
        "name": name,
        "ai_detection": ai_acc,
        "real_safety": real_acc,
        "gpu_bypass": bypass_rate,
        "overall": (ai_passed + real_passed) / total * 100 if total > 0 else 0
    }

if __name__ == "__main__":
    print("COMPREHENSIVE BENCHMARK - ALL DATASETS")
    print("GPU Mock: ENABLED | Cache: DISABLED")
    
    results = []
    
    for name, ai_dir, real_dir, ai_prefix, real_prefix in DATASETS:
        ai_files = get_files(ai_dir, ai_prefix)
        real_files = get_files(real_dir, real_prefix)
        
        if not ai_files and not real_files:
            print(f"\n⚠️  Skipping {name}: No files found")
            continue
        
        result = run_benchmark(name, ai_files, real_files)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - ALL DATASETS")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} | {'AI Det':<8} | {'Real Safe':<10} | {'GPU Bypass':<12} | {'Overall':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} | {r['ai_detection']:>6.1f}% | {r['real_safety']:>8.1f}% | {r['gpu_bypass']:>10.1f}% | {r['overall']:>6.1f}%")
    
    # Calculate averages
    if results:
        avg_ai = sum(r['ai_detection'] for r in results) / len(results)
        avg_real = sum(r['real_safety'] for r in results) / len(results)
        avg_bypass = sum(r['gpu_bypass'] for r in results) / len(results)
        avg_overall = sum(r['overall'] for r in results) / len(results)
        
        print("-" * 80)
        print(f"{'AVERAGE':<20} | {avg_ai:>6.1f}% | {avg_real:>8.1f}% | {avg_bypass:>10.1f}% | {avg_overall:>6.1f}%")
