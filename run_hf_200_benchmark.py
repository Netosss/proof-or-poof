
import os
import requests
import json
import time

API_URL = "http://localhost:8000/detect"

def run_benchmark():
    ai_dir = "tests/data/hf_200_benchmark/AI"
    real_dir = "tests/data/hf_200_benchmark/Real"
    
    results = []
    false_positives_meta = []
    
    print("Scanning AI files (100)...")
    for fn in os.listdir(ai_dir):
        if not fn.endswith(('.jpg', '.jpeg', '.png')): continue
        path = os.path.join(ai_dir, fn)
        with open(path, 'rb') as f:
            r = requests.post(API_URL, files={'file': f})
            data = r.json()
            is_ai = data['summary'].startswith("Likely AI") or data['summary'].startswith("Possible AI")
            results.append({'file': fn, 'type': 'AI', 'correct': is_ai, 'summary': data['summary'], 'bypass': data.get('gpu_bypassed', False)})

    print("Scanning REAL files (100)...")
    for fn in os.listdir(real_dir):
        if not fn.endswith(('.jpg', '.jpeg', '.png')): continue
        path = os.path.join(real_dir, fn)
        with open(path, 'rb') as f:
            r = requests.post(API_URL, files={'file': f})
            data = r.json()
            # In a mocked environment, a real image is a False Positive if it's flagged as AI by metadata
            is_real = not (data['summary'].startswith("Likely AI") or data['summary'].startswith("Possible AI"))
            
            # Specifically check for Metadata-driven false positives
            is_meta_fp = data['summary'] == "Likely AI (Metadata Evidence)"
            if is_meta_fp:
                # Capture full response for analysis
                false_positives_meta.append({'file': fn, 'response': data})
            
            # General accuracy check
            results.append({'file': fn, 'type': 'Real', 'correct': is_real, 'summary': data['summary'], 'bypass': data.get('gpu_bypassed', False)})

    # Summary
    ai_correct = sum(1 for r in results if r['type'] == 'AI' and r['correct'])
    real_correct = sum(1 for r in results if r['type'] == 'Real' and r['correct'])
    bypass_count = sum(1 for r in results if r['bypass'])
    
    print("\n" + "="*80)
    print("HF 200 BENCHMARK RESULTS")
    print("="*80)
    print(f"AI Detection:     {ai_correct}/100 ({ai_correct/100*100:.1f}%)")
    print(f"Real Safety:      {real_correct}/100 ({real_correct/100*100:.1f}%)")
    print("-" * 80)
    print(f"GPU Bypassed:     {bypass_count}/200 ({bypass_count/200*100:.1f}%)")
    print("-" * 80)
    print(f"Overall Accuracy: {(ai_correct+real_correct)/200*100:.1f}%")
    
    # Save False Positives Metadata
    with open("hf_200_false_positives_meta.json", "w") as f:
        json.dump(false_positives_meta, f, indent=2)
    print(f"\nLogged {len(false_positives_meta)} metadata false positives to hf_200_false_positives_meta.json")

if __name__ == "__main__":
    run_benchmark()
