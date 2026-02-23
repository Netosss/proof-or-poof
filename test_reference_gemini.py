import os
import gemini_client
import json
import debug_reference_method

# Monkeypatch with the reference method
gemini_client.get_quality_context = debug_reference_method.get_quality_context_original

def test_reference_method():
    img_path = "/Users/netanel.ossi/Downloads/130206.jpg"
    print(f"Analyzing {os.path.basename(img_path)} with REFERENCE quality method...")
    
    try:
        result = gemini_client.analyze_image_pro_turbo(img_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_reference_method()
