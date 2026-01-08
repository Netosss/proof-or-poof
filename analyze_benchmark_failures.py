
import requests
import json
import os
import sys

API_URL = "http://localhost:8000/detect"

def scan_and_print(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\n{'='*80}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*80}")
    
    # Extract label from path (original/ai)
    label = "ai" if "ai" in filepath.lower() else "original"
    mock_filename = f"AI_{os.path.basename(filepath)}" if label == "ai" else f"REAL_{os.path.basename(filepath)}"

    try:
        with open(filepath, "rb") as f:
            files = {"file": (mock_filename, f)}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")

def main():
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            scan_and_print(f)
    else:
        # Default failures from last run
        failures = [
            "tests/data/benchmark_hf_50/original/10-tips-for-stunning-portrait-photography-7.jpg",
            "tests/data/benchmark_hf_50/original/.amazonaws.com2Fpublic2Fimages2Fcb32a00a-bf52-48fe-9ba6-4e21cf4c1c57_800x800.png"
        ]
        for f in failures:
            scan_and_print(f)

if __name__ == "__main__":
    main()
