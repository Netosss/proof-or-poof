
import os
import requests
import json
import random

random.seed(42)

API_URL = "http://localhost:8000/detect"
DATA_ROOT = "tests/data/benchmark_hf_50"

def run_benchmark():
    metrics = {"ai": {"pass": 0, "total": 0}, "original": {"pass": 0, "total": 0}}
    
    for label in ["ai", "original"]:
        folder = os.path.join(DATA_ROOT, label)
        if not os.path.exists(folder): continue
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        print(f"\nScanning {label.upper()} files ({len(files)})...")
        print(f"{'File':<40} | {'Raw %':<6} | {'Conf':<6} | {'Status':<10} | {'Summary'}")
        print("-" * 130)
        
        for filename in files:
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    response = requests.post(API_URL, files={'file': f}, timeout=15)
                    data = response.json()
                    conf = data.get('confidence_score', 0)
                    summary = data.get('summary', '')
                    raw_prob = data.get('layers', {}).get('layer2_forensics', {}).get('probability', 0)
                    is_bypassed = data.get('gpu_bypassed', False)
                    
                    # Final determination
                    is_ai = any(k in summary for k in ["Likely AI", "Possible AI", "Verified AI", "Suspicious"])
                    
                    success = is_ai if label == "ai" else not is_ai
                    status = "PASS" if success else "FAIL"
                    
                    if success: metrics[label]["pass"] += 1
                    metrics[label]["total"] += 1
                    if is_bypassed: metrics[label].setdefault("bypassed", 0); metrics[label]["bypassed"] += 1
                    
                    bypass_mark = "[META]" if is_bypassed else "[GPU ]"
                    print(f"{filename[:40]:<40} | {raw_prob:<6.2f} | {conf:<6.2f} | {status:<10} | {bypass_mark} {summary}")
            except Exception as e:
                print(f"{filename[:40]:<40} | ERROR  | {str(e)}")

    print("\n" + "="*80)
    print("HF BENCHMARK RESULTS (METADATA-ONLY)")
    print("="*80)
    ai_acc = (metrics["ai"]["pass"]/metrics["ai"]["total"]*100) if metrics["ai"]["total"] > 0 else 0
    org_acc = (metrics["original"]["pass"]/metrics["original"]["total"]*100) if metrics["original"]["total"] > 0 else 0
    print(f"AI Detection:     {metrics['ai']['pass']}/{metrics['ai']['total']} ({ai_acc:.1f}%)")
    print(f"Original Safety:  {metrics['original']['pass']}/{metrics['original']['total']} ({org_acc:.1f}%)")
    print("-" * 80)
    ai_bypass = metrics["ai"].get("bypassed", 0)
    org_bypass = metrics["original"].get("bypassed", 0)
    total_bypass = ai_bypass + org_bypass
    total_files = metrics["ai"]["total"] + metrics["original"]["total"]
    bypass_rate = (total_bypass / total_files * 100) if total_files > 0 else 0
    print(f"GPU Bypassed:     {total_bypass}/{total_files} ({bypass_rate:.1f}%) [Money Saved!]")
    print("-" * 80)
    print(f"Overall Accuracy: {(ai_acc + org_acc)/2:.1f}%")

if __name__ == "__main__":
    run_benchmark()
