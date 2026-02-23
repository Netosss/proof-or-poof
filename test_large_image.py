import os
import json
import time
import gemini_client
from PIL import Image

def test_large_image():
    # Large 16K image
    img_path = "/Users/netanel.ossi/Downloads/andreas-brun-BUu1gsIZqdE-unsplash.jpg"
    print(f"Analyzing Large Image: {os.path.basename(img_path)}...")
    
    try:
        # Check original size
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(img_path) as img:
            print(f"Original Size: {img.size} ({img.size[0]*img.size[1]:,} pixels)")

        start_time = time.time()
        result = gemini_client.analyze_image_pro_turbo(img_path)
        end_time = time.time()
        
        print("\n--- Result ---")
        print(json.dumps(result, indent=2))
        print(f"\nLatency: {end_time - start_time:.2f}s")
        
        if "usage" in result:
            print(f"Tokens Used: {result['usage']['total_tokens']}")
            
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_large_image()
