import os
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

def _resize_if_needed(img: Image.Image) -> Image.Image:
    """Same resize logic to ensure apples-to-apples comparison"""
    MAX_PIXELS = 12_000_000 
    Image.MAX_IMAGE_PIXELS = None
    w, h = img.size
    pixels = w * h
    if pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / pixels) ** 0.5
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

def test_no_media_res(image_path):
    print(f"Testing WITHOUT media_resolution for: {os.path.basename(image_path)}")
    
    # Enable loading large images
    Image.MAX_IMAGE_PIXELS = None
    
    # 1. Prepare Image (Resize to avoid 400 error)
    img_original = Image.open(image_path)
    img = _resize_if_needed(img_original)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
    image_bytes = img_byte_arr.getvalue()
    
    # 2. Config WITHOUT media_resolution
    config_default = types.GenerateContentConfig(
        temperature=1.0, 
        response_mime_type="application/json"
    )
    
    # 3. Call API
    print("Sending request...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), "Analyze this image."],
        config=config_default
    )
    
    if hasattr(response, "usage_metadata"):
        print(f"\n--- RESULTS (Default) ---")
        print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Total Tokens:  {response.usage_metadata.total_token_count}")
    else:
        print("No usage metadata.")

if __name__ == "__main__":
    test_no_media_res("/Users/netanel.ossi/Downloads/image (1)_imgupscaler.ai_General_16K.jpg")
