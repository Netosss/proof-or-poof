import requests
import os
import time

API_URL = "http://localhost:8000/detect"
IMAGES_DIR = "/Users/netanel.ossi/Desktop/ai-detector-datasets/ai/images"

def test_folder():
    print(f"Testing images from: {IMAGES_DIR}")
    if not os.path.exists(IMAGES_DIR):
        print("Directory not found!")
        return

    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images.")

    for filename in files[:5]: # Test first 5 for speed
        filepath = os.path.join(IMAGES_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        with open(filepath, "rb") as f:
            start = time.time()
            try:
                response = requests.post(API_URL, files={"file": f})
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Status: {response.status_code}")
                    print(f"Time: {elapsed:.2f}s")
                    print(f"Summary: {data.get('summary')}")
                    print(f"Confidence: {data.get('confidence_score')}")
                    print(f"GPU Bypassed: {data.get('gpu_bypassed')}")
                else:
                    print(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Request failed: {e}")

if __name__ == "__main__":
    # Wait for server to start
    time.sleep(2)
    test_folder()
