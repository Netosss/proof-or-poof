import os
import json
import io
import time
from PIL import Image
from google import genai
from google.genai import types
from typing import Union

# Initialize Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=20000)
)

def analyze_image_pro_turbo(image_source: Union[str, Image.Image]) -> dict:
    """
    GEMINI 3.0 FLASH - TURBO MODE
    Accepts either a file path (str) or a PIL Image object.
    """
    try:
        # 1. Image Prep
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize logic
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=92, optimize=True)
        image_bytes = img_byte_arr.getvalue()

        # Close if we opened it from a path
        if isinstance(image_source, str):
            img.close()

        # 2. CONFIGURATION
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            temperature=0.0, 
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

        # 3. PROMPT
        prompt = """
        Analyze the image for SYNTHETIC GENERATION ARTIFACTS.
        STABILITY RULES:
        1. "High Quality" or "Smoothness" is NOT evidence of AI. Do not flag it.
        2. You must find a LOGICAL or STRUCTURAL error to assign a high score.
        3. If no artifacts are found, the score MUST be < 0.1.
        4. synthid markers are present in the image. -> give a high score.
        5. If the image was edited with AI, give a high score.

        SCORING GUIDE:
        - 0.01 - 0.10: Clean image. No structural melting, no physics errors.
        - 0.90 - 1.00: Visible glitch (melting hands, gibberish text, asymmetrical pupils).

        OUTPUT RULES:
        - If AI (>0.5): explain in a professional understandable sentence (max 7 words) why it's AI.
        - If NOT AI: just write "No visual anomalies detected".
        """

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=config
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"confidence": -1.0}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Analyzing: {path}...")
        start = time.perf_counter()
        result = analyze_image_pro_turbo(path)
        end = time.perf_counter()
        print(f"Result: {json.dumps(result, indent=2)}")
        print(f"Latency: {end - start:.4f}s")
