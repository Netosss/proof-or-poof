
import requests
import json
import os

API_URL = "http://localhost:8000/detect"
FAILURES = [
    "tests/data/ai/images/130206.jpg",
    "tests/data/original/images/129502.jpg"
]

def scan_and_print(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\n{'='*80}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*80}")
    
    try:
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f)}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")

def main():
    for f in FAILURES:
        scan_and_print(f)

if __name__ == "__main__":
    main()
