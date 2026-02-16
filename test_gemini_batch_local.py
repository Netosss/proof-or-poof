import os
import json
from dotenv import load_dotenv
from gemini_client import analyze_image_pro_turbo

# Load environment variables
load_dotenv()

# Define the downloaded image to test
image_paths = [
    "downloaded_test_image.jpg"
]

print("Starting analysis for downloaded LinkedIn image...\n")

for path in image_paths:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
        
    print(f"Analyzing: {path}")
    try:
        result = analyze_image_pro_turbo(path)
        print(f"Confidence: {result.get('confidence')}")
        print(f"Explanation: {result.get('explanation')}")
        print("-" * 40)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        print("-" * 40)
