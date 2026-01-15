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
    http_options=types.HttpOptions(timeout=10000)
)

# --- GEO-ROUTING CONFIGURATION ---
# Low ad revenue -> use the ultra-cheap model (Gemini 1.5 Flash)
TIER_3_LIST = [
    'PK', 'BD', 'LK', 'NP', 'UZ', 'NG', 'ET', 'KE', 
    'UG', 'TZ', 'SN', 'CG', 'GT', 'JM', 'HT', 'BZ'
]

def calculate_flash_cost(usage_metadata):
    """Calculates exact cost for Gemini 1.5 Flash (Jan 2026 Rates)."""
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    input_cost = (input_tokens / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.30
    total_bill = input_cost + output_cost
    return {
        "model": "Gemini 1.5 Flash",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_bill": f"${total_bill:.6f}",
        "details": f"Input: ${input_cost:.6f} | Output: ${output_cost:.6f}"
    }

def calculate_gemini3_cost(usage_metadata):
    """Calculates exact cost for Gemini 3 Pro (Jan 2026 Rates)."""
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    is_long_context = input_tokens > 200_000
    if is_long_context:
        price_input_per_1m = 4.00
        price_output_per_1m = 18.00
    else:
        price_input_per_1m = 2.00
        price_output_per_1m = 12.00
    input_cost = (input_tokens / 1_000_000) * price_input_per_1m
    output_cost = (output_tokens / 1_000_000) * price_output_per_1m
    total_bill = input_cost + output_cost
    return {
        "model": "Gemini 3 Pro (Long)" if is_long_context else "Gemini 3 Pro (Standard)",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_bill": f"${total_bill:.6f}",
        "details": f"Input: ${input_cost:.6f} | Output: ${output_cost:.6f}"
    }

def analyze_image_smart_route(image_source: Union[str, Image.Image], country_code: str = "US") -> dict:
    """
    SMART ROUTER: Decides between Flash (Cheap) and Pro (Quality) based on Geo-tier.
    """
    if country_code.upper() in TIER_3_LIST:
        return analyze_image_flash(image_source)
    else:
        return analyze_image_pro_turbo(image_source)

def analyze_image_flash(image_source: Union[str, Image.Image]) -> dict:
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source
        if img.mode != "RGB": img = img.convert("RGB")
        if max(img.size) > 512: img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        image_bytes = img_byte_arr.getvalue()
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), "Rate 0.0-1.0 AI."],
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", 
                                            response_schema={"type": "OBJECT", "properties": {"confidence": {"type": "NUMBER"}}, "required": ["confidence"]})
        )
        result = json.loads(response.text)
        result["billing"] = calculate_flash_cost(response.usage_metadata)
        result["is_flash"] = True
        return result
    except Exception as e:
        return {"confidence": -1.0, "error": str(e)}

def analyze_image_pro_turbo(image_source: Union[str, Image.Image]) -> dict:
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source
        if img.mode != "RGB": img = img.convert("RGB")
        if max(img.size) > 1024: img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=92)
        image_bytes = img_byte_arr.getvalue()

        prompt = """
        Analyze strictly for AI artifacts.
        STABILITY RULES:
        1. High quality is NOT AI.
        2. Must find structural errors for high score.
        3. synthid markers = high score.
        4. edited with AI = high score.
        """

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                                            temperature=0.0, response_mime_type="application/json", 
                                            response_schema={"type": "OBJECT", "properties": {"confidence": {"type": "NUMBER"}}, "required": ["confidence"]})
        )
        result = json.loads(response.text)
        result["billing"] = calculate_gemini3_cost(response.usage_metadata)
        result["is_flash"] = False
        return result
    except Exception as e:
        return {"confidence": -1.0, "error": str(e)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        cc = sys.argv[2] if len(sys.argv) > 2 else "US"
        print(f"Routing check for {path} (Country: {cc})...")
        print(json.dumps(analyze_image_smart_route(path, cc), indent=2))
