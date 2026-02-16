import os
import json
from dotenv import load_dotenv
from gemini_client import analyze_image_pro_turbo

# Load environment variables
load_dotenv()

# Define ALL images to test (Both Sets)
image_paths = [
    # Set 1
    "/Users/netanel.ossi/Downloads/129502.jpg",
    "/Users/netanel.ossi/Downloads/129705.jpg",
    "/Users/netanel.ossi/Downloads/129127.jpg",
    "/Users/netanel.ossi/Downloads/130207.jpg",
    "/Users/netanel.ossi/Downloads/130206.jpg",
    "/Users/netanel.ossi/Downloads/130208.jpg",
    "/Users/netanel.ossi/Downloads/ai.jpeg",
    # Set 2
    "/Users/netanel.ossi/Downloads/132215.jpg",
    "/Users/netanel.ossi/Downloads/132206.png",
    "/Users/netanel.ossi/Downloads/132321.jpg",
    "/Users/netanel.ossi/Downloads/132186.jpg"
]

print("Starting FULL batch analysis with current prompt...\n")

for path in image_paths:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
        
    print(f"Analyzing: {os.path.basename(path)}")
    try:
        result = analyze_image_pro_turbo(path)
        print(f"Confidence: {result.get('confidence')}")
        print(f"Explanation: {result.get('explanation')}")
        print("-" * 40)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        print("-" * 40)
