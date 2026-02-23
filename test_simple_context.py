import os
import gemini_client
import json

def simple_context(image_source):
    return "**CONTEXT: QUALITY UNKNOWN.** Proceed with standard analysis.", 0

# Monkeypatch the function
gemini_client.get_quality_context = simple_context

def test_single_image():
    img_path = "/Users/netanel.ossi/Downloads/130206.jpg"
    print(f"Analyzing {os.path.basename(img_path)} with SIMPLE context...")
    
    try:
        result = gemini_client.analyze_image_pro_turbo(img_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_image()
