import kagglehub
import shutil
import os
import random
import requests
import json
import time

random.seed(42) # FIX SEED FOR SCIENTIFIC TUNING

API_URL = "http://localhost:8000/detect"
BENCHMARK_DIR = os.path.abspath("tests/data/benchmark_50")

def setup_data():
    # If benchmark dir exists, remove it to ensure clean slate (since previous run mixed data)
    if os.path.exists(BENCHMARK_DIR):
        shutil.rmtree(BENCHMARK_DIR)

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
    print(f"Dataset downloaded to: {path}")

    # Inspect structure to find Real vs AI folders
    ai_files = []
    real_files = []
    
    for root, dirs, files in os.walk(path):
        for f in files:
            lower_name = f.lower()
            if not lower_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            
            full_path = os.path.join(root, f)
            # Use specific folder names, distinct from dataset name
            if "AiArtData" in root:
                ai_files.append(full_path)
            elif "RealArt" in root:
                real_files.append(full_path)
    
    print(f"Found {len(ai_files)} AI images and {len(real_files)} Real images.")
    
    # Sample 50
    if len(ai_files) > 50: ai_files = random.sample(ai_files, 50)
    if len(real_files) > 50: real_files = random.sample(real_files, 50)
    
    # Prepare target dirs
    if os.path.exists(BENCHMARK_DIR): shutil.rmtree(BENCHMARK_DIR)
    os.makedirs(os.path.join(BENCHMARK_DIR, "ai"), exist_ok=True)
    os.makedirs(os.path.join(BENCHMARK_DIR, "original"), exist_ok=True)
    
    for f in ai_files:
        shutil.copy(f, os.path.join(BENCHMARK_DIR, "ai", os.path.basename(f)))
    for f in real_files:
        shutil.copy(f, os.path.join(BENCHMARK_DIR, "original", os.path.basename(f)))
        
    print(f"Prepared 50 AI and 50 Original images in {BENCHMARK_DIR}")
    return os.path.join(BENCHMARK_DIR, "ai"), os.path.join(BENCHMARK_DIR, "original")

def scan_folder(folder_path, expected_type):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\nScanning {len(files)} {expected_type} files...")
    
    # Stats for 0.50
    pass_50 = 0
    # Stats for 0.45
    pass_45 = 0
    
    print(f"{'File':<40} | {'Prob':<6} | {'Th=0.50':<10} | {'Th=0.45':<10} | {'Summary'}")
    print("-" * 115)
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "rb") as f:
                resp = requests.post(API_URL, files={"file": (filename, f)})
                if resp.status_code == 200:
                    data = resp.json()
                    ai_prob = data.get("layers", {}).get("layer2_forensics", {}).get("probability", 0.0)
                    summary = data.get("summary", "")
                    
                    # Check 0.50
                    status_50 = "FAIL"
                    if expected_type == "AI":
                        if ai_prob > 0.50: status_50 = "PASS"
                    else: 
                        if ai_prob < 0.50: status_50 = "PASS"
                        
                    # Check 0.45
                    status_45 = "FAIL"
                    if expected_type == "AI":
                        if ai_prob > 0.45: status_45 = "PASS"
                    else: 
                        if ai_prob < 0.45: status_45 = "PASS"

                    reset = "\033[0m"
                    c50 = "\033[92m" if status_50 == "PASS" else "\033[91m"
                    c45 = "\033[92m" if status_45 == "PASS" else "\033[91m"
                    
                    print(f"{filename[-40:]:<40} | {ai_prob:<6.2f} | {c50}{status_50:<10}{reset} | {c45}{status_45:<10}{reset} | {summary}")
                    
                    if status_50 == "PASS": pass_50 += 1
                    if status_45 == "PASS": pass_45 += 1
                else:
                    print(f"{filename}: Error {resp.status_code}")
        except Exception as e:
            print(f"{filename}: Exception {e}")
            
    return pass_50, pass_45, len(files)

def main():
    setup_data()
    # Wait for server
    time.sleep(2)
    
    ai_path = os.path.join(BENCHMARK_DIR, "ai")
    orig_path = os.path.join(BENCHMARK_DIR, "original")
    
    ai_pass50, ai_pass45, ai_total = scan_folder(ai_path, "AI")
    orig_pass50, orig_pass45, orig_total = scan_folder(orig_path, "Original")
    
    if orig_total == 0: orig_total = 1 # Prevent ZERO DIV
    if ai_total == 0: ai_total = 1

    print("\n" + "="*80)
    print(f"{'METRIC':<20} | {'THRESH 0.50 (Default)':<25} | {'THRESH 0.45 (Aggressive)':<25}")
    print("="*80)
    print(f"{'AI Detection':<20} | {ai_pass50}/{ai_total} ({ai_pass50/ai_total*100:.1f}%)           | {ai_pass45}/{ai_total} ({ai_pass45/ai_total*100:.1f}%)")
    print(f"{'Original Safety':<20} | {orig_pass50}/{orig_total} ({orig_pass50/orig_total*100:.1f}%)           | {orig_pass45}/{orig_total} ({orig_pass45/orig_total*100:.1f}%)")
    print("-" * 80)
    acc50 = (ai_pass50 + orig_pass50) / (ai_total + orig_total) * 100
    acc45 = (ai_pass45 + orig_pass45) / (ai_total + orig_total) * 100
    print(f"{'Overall Schema':<20} | {acc50:.1f}%                        | {acc45:.1f}%")

if __name__ == "__main__":
    main()
