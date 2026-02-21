import os
import json
import io
import time
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from typing import Union

# Initialize Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=20000)
)

def get_quality_context(image_source: Union[str, Image.Image]) -> str:
    """
    Analyzes image quality using a weighted scoring system (DQT, Resolution, Blur).
    Returns a context string for Gemini.
    """
    score = 100
    issues = []
    
    try:
        # 1. Standardize Input (Handle Path vs PIL)
        if isinstance(image_source, str):
            try:
                img_pil = Image.open(image_source)
            except Exception:
                return "**CONTEXT: QUALITY UNKNOWN (Could not open).**"
            was_path = True
            filename = os.path.basename(image_source)
        else:
            img_pil = image_source
            was_path = False
            filename = "Image Object"

        # --- LAYER 1: DQT (JPEG Artifacts) ---
        # Only works if original metadata is preserved
        try:
            if hasattr(img_pil, 'quantization') and img_pil.quantization:
                table = img_pil.quantization.get(0) # Luminance table
                if table:
                    avg_q_val = sum(table) / len(table)
                    if avg_q_val > 30: # Very heavy compression (Q < 40)
                        score -= 40
                        issues.append(f"Severe Compression (~Q{int(100 - avg_q_val*2)})")
                    elif avg_q_val > 20: # Moderate compression (Q < 60)
                        score -= 20
                        issues.append(f"Compression Artifacts (~Q{int(100 - avg_q_val*2)})")
        except Exception:
            pass

        # --- LAYER 2: PIXEL ANALYSIS (Resolution & Blur) ---
        # Convert PIL to CV2 without re-reading from disk
        try:
            img_arr = np.array(img_pil.convert('RGB')) 
            # Convert RGB (PIL) to BGR (OpenCV)
            img_cv = img_arr[:, :, ::-1].copy() 
            
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            h, w = gray.shape
            pixels = h * w
            
            # Resolution Penalties
            if pixels < 250_000: # Very small (< 500x500)
                score -= 50
                issues.append(f"Tiny Resolution ({w}x{h})")
            elif pixels < 800_000: # Small-ish (< 1000x800)
                score -= 40
                issues.append(f"Low Resolution ({w}x{h})")

            # Blur Penalties (Laplacian Variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < 10: # Almost no edges / Solid color / severe blur
                score -= 60
                issues.append(f"Near Zero Detail (Score: {blur_score:.0f})")
            elif blur_score < 25: # Unusable / Very Blurry
                score -= 25
                issues.append(f"Extreme Blur (Score: {blur_score:.0f})")
            elif blur_score < 50: # Soft / Slightly Out of Focus (or Smooth AI)
                score -= 10
                issues.append(f"Soft Focus/Smooth (Score: {blur_score:.0f})")
            elif blur_score > 600 and avg_q_val < 20: # Very Sharp AND NOT Compressed
                score += 20
                issues.append(f"High Sharpness (Score: {blur_score:.0f})")
            elif blur_score > 300 and avg_q_val < 20: # Sharp AND NOT Compressed
                score += 10
                issues.append(f"Good Sharpness (Score: {blur_score:.0f})")
                
        except Exception:
            pass # CV2 conversion failed, skip pixel checks

        # Close if we opened it locally
        if was_path:
            img_pil.close()

        # --- FINAL VERDICT ---
        score = max(0, score) # Clamp at 0
        
        if score < 50:
            return f"**CRITICAL CONTEXT: LOW QUALITY IMAGE ({', '.join(issues)}).** INSTRUCTION: This image is heavily compressed/low-res. Warning: This quality naturally causes **WAXY SKIN**, **SMOOTH TEXTURES**, distorted faces, and blur. **IGNORE** any 'waxy' or 'smooth' appearance—it is due to low quality, not AI. Only flag **STRUCTURAL IMPOSSIBILITIES** that compression CANNOT explain (e.g., gibberish text, extra limbs). If unsure, assume it is compression."
        elif score < 80:
             return f"**CONTEXT: MEDIUM QUALITY ({', '.join(issues)}).** INSTRUCTION: Image has reduced quality. Do NOT flag soft edges or indistinct textures as AI. However, compression does NOT explain structural impossibilities (like extra fingers, gibberish text, or impossible physics). If you see these CLEAR logic errors, FLAG THEM."
        
        return "**CONTEXT: HIGH QUALITY IMAGE.** Image is sharp and clean. Any visual anomaly is likely a sign of manipulation."

    except Exception as e:
        return "**CONTEXT: QUALITY UNKNOWN.** Proceed with standard analysis."

def analyze_image_pro_turbo(image_source: Union[str, Image.Image]) -> dict:
    """
    GEMINI 3.0 FLASH - TURBO MODE
    Accepts either a file path (str) or a PIL Image object.
    """
    try:
        # 1. Get Quality Context (Run on the source image for best accuracy)
        quality_context = get_quality_context(image_source)

        # 2. Image Prep
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

        prompt = f"""
You are an expert AI Detection System. Analyze the image for synthetic generation.

### CORE INSTRUCTIONS (Trust Your Intuition, But Follow These Rules):
First try to use synthID or found any AI generated details in the image.(watermark, color strips, etc.)
if you found any, give a high score and stop the analysis.
if you didn't find any, then continue with the following instructions.
{quality_context}
1.  **CONTEXT MATTERS:**
    * **Art/Memes:** Be lenient. Do NOT flag stylized anatomy or "bad photoshop" edges as AI.
    * **Photos:** Be strict. Look for "plastic/waxy" skin, merging objects, and dream-like physics, and any physical errors (in any object in the image). be context aware, try to find any incorrect logic errors and contextually impossible details..

2.  **THE "TEXT" TRAP:**
    * If you see text (any language), **READ IT**. If the letters form **gibberish/non-words** (e.g., "הצסיהת") or the sentence is incorrect logically, it is AI. 
    * **Ignore Dates:** The year is irrelevant. Do not use "future dates" as a signal.

3.  **THE "WATERMARK" CHECK:**
    * Scan corners for **SynthID**, **DALL-E color strips**, or **"CR"** icons. If found -> AUTOMATIC AI.
### OUTPUT DECISION (Strict Format):
* **If AI (>0.5):**
    * **Reason:** [Max 7 words, e.g., "Menu text contains gibberish words", "Background hand merges into chips"].
    * **Confidence Score:** [0.6 - 1.0]
* **If NOT AI:**
    * **Reason:** "No visual anomalies detected."
    * **Confidence Score:** [0.0 - 0.4]
"""
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=config
        )

        result = json.loads(response.text)

        # Add usage metadata for cost tracking
        if hasattr(response, "usage_metadata"):
            result["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }

        return result

    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"confidence": -1.0}

def analyze_batch_images_pro_turbo(image_sources: list[Union[str, Image.Image]]) -> dict:
    """
    Analyzes a batch of images for synthetic generation artifacts using Gemini 3.0 Flash.
    Returns a single aggregated result based on the median decision.
    """
    try:
        # 1. Image Prep (Loop through all inputs)
        image_parts = []
        for src in image_sources:
            if isinstance(src, str):
                img = Image.open(src)
            else:
                img = src

            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize logic (Applied to each)
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=92, optimize=True)
            
            # Append to our list of parts
            image_parts.append(
                types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg")
            )

            if isinstance(src, str):
                img.close()

        # 2. CONFIGURATION
        try:
             # Use the second image (index 1) for quality context if available, else first
             idx = 1 if len(image_sources) > 1 else 0
             quality_context = get_quality_context(image_sources[idx])
        except:
             quality_context = "**CONTEXT: QUALITY UNKNOWN.**"

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            temperature=0.0, 
            response_mime_type="application/json",
            response_schema={
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {                    
                        "confidence": {"type": "NUMBER"},
                        "explanation": {"type": "STRING"}
                    },
                    "required": ["confidence", "explanation"]
                }
            }
        )

        # 3. PROMPT
        prompt_text = f"""
        Analyze EACH of the attached images for SYNTHETIC GENERATION ARTIFACTS.
        Return a JSON list where each item corresponds to the images in order.
        

        ### CORE INSTRUCTIONS (Trust Your Intuition, But Follow These Rules):
        First try to use synthID or found any AI generated details in the image.(watermark, color strips, etc.)
        if you found any, give a high score and stop the analysis.
        if you didn't find any, then continue with the following instructions.
        {quality_context}

        1.  **CONTEXT MATTERS:**
            * **Art/Memes:** Be lenient. Do NOT flag stylized anatomy or "bad photoshop" edges as AI.
            * **Photos:** Be strict. Look for "plastic/waxy" skin, merging objects, and dream-like physics, and any physical errors (in any object in the image). be context aware, try to find any incorrect logic errors and contextually impossible details..

        2.  **THE "TEXT" TRAP:**
            * If you see text (any language), **READ IT**. If the letters form **gibberish/non-words** (e.g., "הצסיהת") or the sentence is incorrect logically, it is AI. 
            * **Ignore Dates:** The year is irrelevant. Do not use "future dates" as a signal.

        3.  **THE "WATERMARK" CHECK:**
            * Scan corners for **SynthID**, **DALL-E color strips**, or **"CR"** icons. If found -> AUTOMATIC AI.

        SCORING GUIDE:
        - 0.01 - 0.10: Clean image. No structural melting, no physics errors.
        - 0.90 - 1.00: Visible glitch (melting hands, gibberish text, asymmetrical pupils).

        OUTPUT RULES:
        - If AI (>0.5): explain in a professional understandable sentence (max 7 words) why it's AI.
        - If NOT AI: just write "No visual anomalies detected".
        """

        # Combine images + prompt into one content list
        request_contents = image_parts + [prompt_text]

        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=request_contents,
            config=config
        )

        raw_results = json.loads(response.text)
        
        # 4. Aggregation Logic (Median/Majority Vote)
        if not raw_results:
             return {"confidence": 0.5, "explanation": "Suspicious: No clear analysis returned."}

        ai_votes = [r for r in raw_results if r.get("confidence", 0) > 0.5]
        not_ai_votes = [r for r in raw_results if r.get("confidence", 0) <= 0.5]

        final_result = {}

        # Determine Majority
        if len(ai_votes) > len(not_ai_votes):
            # AI wins
            # Confidence: Average of the AI votes
            avg_conf = sum(r["confidence"] for r in ai_votes) / len(ai_votes)
            
            # Explanation: From the HIGHEST confidence AI vote
            best_explanation_item = max(ai_votes, key=lambda x: x["confidence"])
            final_result = {
                "confidence": round(avg_conf, 2),
                "explanation": best_explanation_item["explanation"]
            }
        else:
            # Not AI wins (or tie)
            # Confidence: Average of the Not-AI votes
            if not_ai_votes:
                avg_conf = sum(r["confidence"] for r in not_ai_votes) / len(not_ai_votes)
                # Explanation: From the LOWEST confidence Not-AI vote (most "clean")
                best_explanation_item = min(not_ai_votes, key=lambda x: x["confidence"])
                explanation = best_explanation_item["explanation"]
            else:
                # Edge case: No votes? (Shouldn't happen if list not empty)
                avg_conf = 0.0
                explanation = "No visual anomalies detected"

            final_result = {
                "confidence": round(avg_conf, 2),
                "explanation": explanation
            }
            
        # Add usage metadata if available
        if hasattr(response, "usage_metadata"):
            final_result["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }

        return final_result

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
