import os
import json
import io
from PIL import Image
from google import genai
from google.genai import types

# Initialize Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60000)
)

def test_token_cost(image_path):
    print(f"Analyzing Token Cost for: {os.path.basename(image_path)}")
    
    # Allow large images
    Image.MAX_IMAGE_PIXELS = None
    
    img = Image.open(image_path)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=95)
    image_bytes = img_byte_arr.getvalue()
    
    # 1. WITH media_resolution="MEDIA_RESOLUTION_HIGH"
    print("\n--- WITH MEDIA_RESOLUTION_HIGH ---")
    config_high = types.GenerateContentConfig(
        media_resolution="MEDIA_RESOLUTION_HIGH",
        temperature=1.0, 
        response_mime_type="application/json"
    )
    
    response_high = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), "Analyze this image."],
        config=config_high
    )
    
    if hasattr(response_high, "usage_metadata"):
        print(f"Prompt Tokens: {response_high.usage_metadata.prompt_token_count}")
        print(f"Total Tokens:  {response_high.usage_metadata.total_token_count}")
    else:
        print("Usage metadata not available.")

    # 2. WITHOUT media_resolution (Default)
    print("\n--- WITHOUT MEDIA_RESOLUTION (DEFAULT) ---")
    config_default = types.GenerateContentConfig(
        temperature=1.0, 
        response_mime_type="application/json"
    )
    
    response_default = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), "Analyze this image."],
        config=config_default
    )
    
    if hasattr(response_default, "usage_metadata"):
        print(f"Prompt Tokens: {response_default.usage_metadata.prompt_token_count}")
        print(f"Total Tokens:  {response_default.usage_metadata.total_token_count}")
    else:
        print("Usage metadata not available.")

if __name__ == "__main__":
    test_token_cost("/Users/netanel.ossi/Downloads/image (1)_imgupscaler.ai_General_16K.jpg")
