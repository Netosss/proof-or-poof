
import os
import requests
import json
import time

API_URL = "http://localhost:8000/detect"
TEST_DATA_DIR = "tests/data"

def get_all_test_files():
    files = []
    for root, dirnames, filenames in os.walk(TEST_DATA_DIR):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov', '.avi', '.mkv', '.webm')):
                files.append(os.path.join(root, filename))
    return files

def scan_file(filepath):
    try:
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f)}
            # Optional: pass trusted metadata if we want to test that path, but standard test is without.
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    test_files = get_all_test_files()
    if not test_files:
        print("No files found in tests/data.")
        return

    print(f"Found {len(test_files)} files in {TEST_DATA_DIR}. Running integration tests...")
    print("-" * 120)
    print(f"{'Path':<60} | {'Score':<6} | {'Status':<15} | {'Summary'}")
    print("-" * 120)
    
    # Sort for consistent output
    test_files.sort()

    passed = 0
    failed = 0
    
    for filepath in test_files:
        filename = os.path.basename(filepath)
        # Determine expected category from path
        expected = "UNKNOWN"
        if "/ai/" in filepath: expected = "AI"
        elif "/original/" in filepath: expected = "Original"
        
        # Determine strictness
        is_screenshot = ("screenshots" in filepath or "screen_records" in filepath)
        
        result = scan_file(filepath)
        
        if "error" in result:
            print(f"{filepath[-60:]:<60} | {'ERR':<6} | {'ERROR':<15} | {result['error']}")
            failed += 1
            continue

        score = result.get("confidence_score", 0.0)
        summary = result.get("summary", "")
        
        # Test assertion logic
        status = "UNKNOWN"
        if expected == "AI":
            # AI Should have high score OR summary indicating AI/Suspicious
            if score > 0.5 or "AI" in summary: status = "PASS"
            else: status = "FAIL"
        elif expected == "Original":
            # Original should have low score OR summary indicating Human/Original
            # Exception: Suspicious (0.8-0.9) is often acceptable for stripped metadata
            if "Original" in summary or "Human" in summary: status = "PASS"
            elif score > 0.8 and "Suspicious" in summary and is_screenshot: status = "WARN (Susp)" # Tolerable for screenshots without metadata
            elif score > 0.6: status = "FAIL"
            else: status = "PASS"
            
        color = ""
        if status == "PASS": color = "\033[92m" # Green
        elif status == "FAIL": color = "\033[91m" # Red
        elif "WARN" in status: color = "\033[93m" # Yellow
        reset = "\033[0m"
        
        print(f"{filepath[-60:]:<60} | {score:<6.2f} | {color}{status:<15}{reset} | {summary}")
        
        if status == "PASS": passed += 1
        else: failed += 1
        
    print("-" * 120)
    print(f"Total: {len(test_files)} | Passed: {passed} | Other: {failed}")

if __name__ == "__main__":
    main()
