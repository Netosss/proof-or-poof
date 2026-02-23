import os
import json
import io
import time
from PIL import Image
from google import genai
from google.genai import types

# Initialize Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60000)
)

def test_media_resolution_support(image_path):
    print(f"Testing media_resolution support with {os.path.basename(image_path)}...")
    
    try:
        img = Image.open(image_path)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        image_bytes = img_byte_arr.getvalue()
        
        # Try configuration WITH media_resolution
        config_with_res = types.GenerateContentConfig(
            media_resolution="MEDIA_RESOLUTION_HIGH",
            temperature=1.0, 
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {                    
                    "confidence": {"type": "NUMBER"},
                    "explanation": {"type": "STRING"}
                },
                "required": ["confidence", "explanation"]
            }
        )
        
        print("\nSending request WITH media_resolution='MEDIA_RESOLUTION_HIGH'...")
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Analyze this image."
            ],
            config=config_with_res
        )
        
        print("SUCCESS! The model accepted the parameter.")
        print(response.text)
        
    except Exception as e:
        print(f"\nFAILED! The model rejected the parameter.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_media_resolution_support("/Users/netanel.ossi/Downloads/130206.jpg")
