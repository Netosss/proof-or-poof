
import os
import random
import requests
import json
import time

API_URL = "http://localhost:8000/detect"

KAG_AI_DIR = "tests/data/kaggle_benchmark/AiArtData/AiArtData"
KAG_REAL_DIR = "tests/data/kaggle_benchmark/RealArt/RealArt"
HF_AI_DIR = "tests/data/hf_200_benchmark/AI"
HF_REAL_DIR = "tests/data/hf_200_benchmark/Real"

def get_samples(directory, count=15):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return random.sample(files, min(count, len(files)))

def run_bench(name, ai_files, real_files):
    print(f"\n=== {name} BENCHMARK (LIVE GPU) ===")
    
    ai_passed = 0
    real_passed = 0
    gpu_bypasses = 0
    
    print("\nProcessing AI Files:")
    for f in ai_files:
        try:
            with open(f, 'rb') as img:
                r = requests.post(API_URL, files={'file': img})
                data = r.json()
                summary = data.get('summary', 'Error')
                is_ai = "AI" in summary
                gpu_bypassed = data.get('gpu_bypassed', False)
                if gpu_bypassed: gpu_bypasses += 1
                
                status = "PASS" if is_ai else "FAIL"
                if is_ai: ai_passed += 1
                
                print(f"{os.path.basename(f):<40} | {status} | {summary} {'(Bypassed)' if gpu_bypassed else ''}")
        except Exception as e:
            print(f"{os.path.basename(f):<40} | ERROR | {e}")

    print("\nProcessing Real Files:")
    for f in real_files:
        try:
            with open(f, 'rb') as img:
                r = requests.post(API_URL, files={'file': img})
                data = r.json()
                summary = data.get('summary', 'Error')
                is_real = "Original" in summary or "Human" in summary
                gpu_bypassed = data.get('gpu_bypassed', False)
                if gpu_bypassed: gpu_bypasses += 1
                
                status = "PASS" if is_real else "FAIL"
                if is_real: real_passed += 1
                
                print(f"{os.path.basename(f):<40} | {status} | {summary} {'(Bypassed)' if gpu_bypassed else ''}")
        except Exception as e:
            print(f"{os.path.basename(f):<40} | ERROR | {e}")

    total = len(ai_files) + len(real_files)
    print(f"\nRESULTS for {name}:")
    print(f"AI Detection: {ai_passed}/{len(ai_files)} ({ai_passed/len(ai_files)*100:.1f}%)")
    print(f"Real Safety:  {real_passed}/{len(real_files)} ({real_passed/len(real_files)*100:.1f}%)")
    print(f"GPU Bypasses: {gpu_bypasses}/{total} ({gpu_bypasses/total*100:.1f}%)")

if __name__ == "__main__":
    # Sample 15 AI and 15 Real from each to make a total of 30 "per benchmark" (or 60 total?)
    # The user said "sample of 30 from each benchmark". I'll do 30 AI and 30 Real (60 total per bench) to be thorough.
    
    kag_ai = get_samples(KAG_AI_DIR, 30)
    kag_real = get_samples(KAG_REAL_DIR, 30)
    
    hf_ai = get_samples(HF_AI_DIR, 30)
    hf_real = get_samples(HF_REAL_DIR, 30)
    
    run_bench("KAGGE", kag_ai, kag_real)
    run_bench("HF 200", hf_ai, hf_real)
