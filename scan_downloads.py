
import os
import glob
import requests
import json
from pathlib import Path

# Config
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
API_URL = "http://localhost:8000/detect"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
LIMIT = 20  # Increased limit to find videos

def get_recent_images(directory, limit):
    # Get all files
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, "*" + ext)))
        files.extend(glob.glob(os.path.join(directory, "*" + ext.upper())))
    
    # Sort by modification time (descending)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:limit]

def scan_file(filepath):
    print(f"Scanning: {os.path.basename(filepath)}...")
    try:
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f)}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    images = get_recent_images(DOWNLOADS_DIR, LIMIT)
    
    if not images:
        print("No images found in Downloads.")
        return

    print(f"Found {len(images)} recent images. Processing...")
    print("-" * 60)
    print(f"{'Filename':<30} | {'Score':<8} | {'Summary'}")
    print("-" * 60)
    
    results = []
    for img in images:
        result = scan_file(img)
        fname = os.path.basename(img)[:28]
        
        if "error" in result:
            print(f"{fname:<30} | {'ERR':<8} | {result['error']}")
        else:
            score = result.get("confidence_score", 0.0)
            summary = result.get("summary", "N/A")
            print(f"{fname:<30} | {score:<8.2f} | {summary}")
            
            # Print details if high confidence
            layers = result.get("layers", {})
            l2 = layers.get("layer2_forensics", {})
            signals = l2.get("signals", [])
            if signals:
                print(f"  > Signals: {signals[:2]}")
                
        results.append(result)

if __name__ == "__main__":
    main()
