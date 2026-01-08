
import os
import requests
import json

API_URL = "http://localhost:8000/detect"

def scan_folder(folder_path, expected_label):
    print(f"\nScanning {expected_label} folder: {folder_path}")
    print("-" * 80)
    print(f"{'File':<40} | {'Prob':<6} | {'Status':<15} | {'Summary'}")
    print("-" * 80)
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    files.sort()
    
    passed = 0
    total = 0
    bypassed = 0
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'rb') as f:
                response = requests.post(API_URL, files={'file': f}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    prob = data.get('confidence_score', 0)
                    summary = data.get('summary', 'Unknown')
                    is_bypassed = data.get('gpu_bypassed', False)
                    if is_bypassed: bypassed += 1
                    
                    # For AI images, we want prob > 0.5
                    # For Original images, we want prob < 0.5
                    is_ai = any(k in summary for k in ["Likely AI", "Possible AI", "Verified AI", "Suspicious"])
                    # Special case: "Likely Original" with high score means it's Original.
                    is_original = "Likely Original" in summary or "Verified Original" in summary
                    
                    success = is_ai if expected_label == "AI" else is_original
                    
                    status = "PASS" if success else "FAIL"
                    if success: passed += 1
                    total += 1
                    
                    bypass_mark = "[META]" if is_bypassed else "[GPU ]"
                    print(f"{filename:<40} | {prob:<6.2f} | {status:<15} | {bypass_mark} {summary}")
                else:
                    print(f"{filename:<40} | ERROR  | {response.status_code}")
        except Exception as e:
            print(f"{filename:<40} | EXCEP  | {str(e)}")
            
    if total > 0:
        print("-" * 80)
        print(f"RESULT: {passed}/{total} ({passed/total*100:.1f}%) PASSED")
        print(f"GPU Bypassed: {bypassed}/{total} ({bypassed/total*100:.1f}%) [Money Saved!]")

if __name__ == "__main__":
    scan_folder("tests/data/ai/images", "AI")
    scan_folder("tests/data/original/images", "Original")
