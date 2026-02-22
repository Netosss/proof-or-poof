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
from pydantic import BaseModel, Field

# Enable loading truncated images globally
Image.MAX_IMAGE_PIXELS = None

# Initialize Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=15000,
        retry_options=types.HttpRetryOptions(
            attempts=2, # Try up to 2 times
            initial_delay=1.0, # Wait 1 second before first retry
            max_delay=5.0, # Never wait more than 5 seconds between retries
            exp_base=2.0, # Exponential backoff (1s, 2s, 4s...)
            http_status_codes=[408, 429, 500, 502, 503, 504] # Only retry on these specific errors
        )
    )
)

class DetectionResult(BaseModel):
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    explanation: str = Field(description="Max 10 words explaining the artifact")

def get_quality_context(image_source: Union[str, Image.Image, bytes]) -> tuple[str, int]:
    """
    Analyzes image quality using a weighted scoring system (DQT, Resolution, Blur).
    Returns a tuple of (context string for Gemini, quality score).
    """
    score = 100
    issues = []
    avg_q_val = 0 # Default to 0 (High Quality/No Compression Artifacts)
    
    try:
        # 1. Standardize Input (Handle Path vs PIL vs Bytes)
        if isinstance(image_source, str):
            try:
                img_pil = Image.open(image_source)
            except Exception:
                return "**CONTEXT: QUALITY UNKNOWN (Could not open).**", 0
            was_path = True
            filename = os.path.basename(image_source)
        elif isinstance(image_source, bytes):
            try:
                # Decode bytes to PIL for consistency
                # We use cv2 for decoding as it might be faster for the subsequent cv2 ops, 
                # but PIL is needed for DQT check. Let's just use BytesIO -> PIL.
                img_pil = Image.open(io.BytesIO(image_source))
            except Exception:
                return "**CONTEXT: QUALITY UNKNOWN (Could not decode bytes).**", 0
            was_path = True # Treat as temp object that needs closing
            filename = "Video Frame Bytes"
        else:
            img_pil = image_source
            was_path = False
            filename = "Image Object"

        # --- LAYER 1: DQT (JPEG Artifacts) ---
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
        try:
            img_arr = np.array(img_pil.convert('RGB')) 
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            
            h, w = gray.shape
            pixels = h * w
            
            # Resolution Penalties
            if pixels < 250_000: # Very small (< 500x500)
                score -= 50
                issues.append(f"Tiny Resolution ({w}x{h})")
            elif pixels < 800_000: # Small-ish (< 1000x800)
                score -= 40
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
            elif blur_score > 2000: # Exceptionally Sharp (Override compression)
                score += 20
                issues.append(f"Exceptional Sharpness (Score: {blur_score:.0f})")
            elif blur_score > 1500: # Very Sharp (Override compression)
                score += 15
                issues.append(f"High Sharpness (Score: {blur_score:.0f})")
            elif blur_score > 600 and avg_q_val < 20: # Very Sharp AND NOT Compressed
                score += 20
                issues.append(f"High Sharpness (Score: {blur_score:.0f})")
            elif blur_score > 300 and avg_q_val < 20: # Sharp AND NOT Compressed
                score += 10
                
        except Exception:
            pass 

        if was_path:
            img_pil.close()

        # --- FINAL VERDICT ---
        score = max(0, score) 
        
        if score < 50:
            return f"**CRITICAL CONTEXT: LOW QUALITY IMAGE ({', '.join(issues)}).** INSTRUCTION: This image is heavily compressed/low-res. Warning: This quality naturally causes **WAXY SKIN**, **SMOOTH TEXTURES**, distorted faces, and blur. **IGNORE** any 'waxy' or 'smooth' appearance—it is due to low quality, not AI. Only flag **STRUCTURAL IMPOSSIBILITIES** that compression CANNOT explain (e.g., gibberish text, extra limbs). If unsure, assume it is compression.", score
        elif score < 80:
             return f"**CONTEXT: MEDIUM QUALITY ({', '.join(issues)}).** INSTRUCTION: Image has reduced quality. Do NOT flag soft edges or indistinct textures as AI. However, compression does NOT explain structural impossibilities (like extra fingers, gibberish text, or impossible physics). If you see these CLEAR logic errors, FLAG THEM.", score
        
        return "**CONTEXT: HIGH QUALITY IMAGE.** Image is sharp and clean. Any visual anomaly is likely a sign of manipulation.", score

    except Exception as e:
        return "**CONTEXT: QUALITY UNKNOWN.** Proceed with standard analysis.", 0

def get_system_instruction(quality_context: str) -> str:
    """Returns the strict PTCF prompt schema, formatted for Gemini 3 Flash."""
    return f"""[PERSONA]
You are an expert forensic AI image detection system analyzing visual data for generative anomalies.

[TASK]
Analyze the provided image and determine if it was generated or manipulated by artificial intelligence.

[DYNAMIC CONTEXT]
{quality_context}

[FORENSIC RULES]
1. THE "WATERMARK" CHECK (EARLY EXIT):
   * Actively scan corners and borders for SynthID patterns, DALL-E color strips, or "CR" (Content Credentials) icons. 
   * IF FOUND: Stop all further analysis immediately. Return a confidence score of 1.0 and state ONLY the watermark found. Do not evaluate any other rules.

2. CONTEXT MATTERS:
   * Art/Memes/Cartoons: Be lenient. Do NOT flag stylized anatomy, brush strokes, or "bad photoshop" as AI.
   * Photorealism: Be strict. Hunt for "plastic/waxy" skin, merging foreground/background objects, non-Euclidean geometry, and physical impossibilities (e.g., shadows pointing toward the light source).
   
3. THE "TEXT" TRAP:
   * If you see text in any language, read it carefully. If the letters form gibberish/non-words (e.g., English "Welcme tp th" or Hebrew "הצסיהת") or the structural logic of the sign fails, it is AI.
   * Ignore the actual date or year. Do not use "future dates" as a manipulation signal.

[OUTPUT FORMAT & EXAMPLES]
You must respond strictly in JSON.
* If AI (>0.5): The explanation must be a single, clinical sentence (max 10 words) isolating the specific artifact.
* If NOT AI (<=0.5): The explanation must exactly read: "No visual anomalies detected."

### FEW-SHOT EXAMPLES:

Example 1 (Watermark Early Exit):
{{
  "confidence": 1.0,
  "explanation": "DALL-E color strip detected in lower right corner."
}}

Example 2 (Clean High-Res Photo):
{{
  "confidence": 0.05,
  "explanation": "No visual anomalies detected."
}}

Example 3 (AI Generated Portrait):
{{
  "confidence": 0.95,
  "explanation": "Subject's left earring merges seamlessly into the jawline."
}}

Example 4 (AI Generated Street Sign):
{{
  "confidence": 0.88,
  "explanation": "Background stop sign contains illegible, gibberish characters."
}}
"""

def _resize_if_needed(img: Image.Image) -> Image.Image:
    """
    Resizes image if it exceeds 4MP (~2048x2048) to limit token usage and avoid payload errors.
    Keeps aspect ratio.
    """
    MAX_PIXELS = 4_194_304 # 2048 x 2048
    
    w, h = img.size
    pixels = w * h
    
    if pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return img

def analyze_image_pro_turbo(image_source: Union[str, Image.Image], pre_calculated_quality_context: str = None) -> dict:
    """
    GEMINI 3.0 FLASH - OPTIMIZED FOR FORENSIC DETECTION
    """
    img_to_close = [] # Keep a list of intermediate images to close
    
    try:
        if isinstance(image_source, str):
            img_original = Image.open(image_source)
            img_to_close.append(img_original)
        else:
            img_original = image_source

        # 1. Analyze Quality on ORIGINAL resolution (if not pre-calculated)
        quality_score = 0
        if pre_calculated_quality_context:
            quality_context = pre_calculated_quality_context
        else:
            quality_context, quality_score = get_quality_context(img_original)

        # 2. Resize for Upload
        img_working = _resize_if_needed(img_original)
        if img_working is not img_original:
            img_to_close.append(img_working)

        # 3. Ensure RGB
        if img_working.mode != "RGB":
            img_rgb = img_working.convert("RGB")
            img_to_close.append(img_rgb)
            img_working = img_rgb
        
        # 4. Save to bytes
        img_byte_arr = io.BytesIO()
        # Use Q=95 for maximum fidelity (forensic analysis)
        img_working.save(img_byte_arr, format='JPEG', quality=95)
        image_bytes = img_byte_arr.getvalue()

        # 5. Clean up ALL intermediate objects immediately
        for img_obj in img_to_close:
            img_obj.close()
            
        # --- CONFIGURATION ---
        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            temperature=1.0, 
            response_mime_type="application/json",
            response_schema=DetectionResult,
        )

        # --- THE PAYLOAD (With x2 Prompt Repetition Hack) ---
        execution_query = "Carefully analyze this image for generative AI manipulation, strictly following the system instructions."
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                execution_query,
                execution_query # Duplicated for 360-degree mathematical context
            ],
            config=config
        )

        parsed_result = response.parsed
        
        result = {
            "confidence": parsed_result.confidence,
            "explanation": parsed_result.explanation,
            "quality_score": quality_score
        }

        if hasattr(response, "usage_metadata"):
            result["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
        
        # result["quality_score"] = quality_score # Already added above
        result["quality_context"] = quality_context
        return result

    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"confidence": -1.0}

def analyze_batch_images_pro_turbo(image_sources: list[Union[str, Image.Image, bytes]]) -> dict:
    """
    Analyzes a batch of images for synthetic generation artifacts using Gemini 3.0 Flash.
    Returns a single aggregated result based on the median decision.
    """
    try:
        image_parts = []
        quality_context = None

        idx_to_scan = 1 if len(image_sources) > 1 else 0

        for i, src in enumerate(image_sources):
            # Check for quality context on the selected frame
            if i == idx_to_scan:
                # This works for bytes too now!
                try:
                    quality_context, _ = get_quality_context(src)
                except Exception as e:
                    print(f"Failed to get quality context for video frame: {e}")

            # Optimization: Direct bytes (Video Frame)
            if isinstance(src, bytes):
                # We assume it's already a valid JPEG from cv2.imencode
                image_parts.append(
                    types.Part.from_bytes(data=src, mime_type="image/jpeg")
                )
                continue

            img_to_close = [] # Track objects for this specific image in the loop
            
            if isinstance(src, str):
                img_original = Image.open(src)
                img_to_close.append(img_original)
            else:
                img_original = src

            if i == idx_to_scan and not quality_context:
                 # Fallback if src was PIL/Path and loop order logic requires it here
                 # (Though we moved it up, so this is just safety)
                quality_context, _ = get_quality_context(img_original)

            img_working = _resize_if_needed(img_original)
            if img_working is not img_original:
                img_to_close.append(img_working)

            if img_working.mode != "RGB":
                img_rgb = img_working.convert("RGB")
                img_to_close.append(img_rgb)
                img_working = img_rgb
            
            img_byte_arr = io.BytesIO()
            img_working.save(img_byte_arr, format='JPEG', quality=85)
            
            image_parts.append(
                types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg")
            )

            # Clean up immediately after byte extraction
            for img_obj in img_to_close:
                img_obj.close()
                
        if not quality_context:
             # Fallback if no quality context could be determined (e.g. decoding failed)
             quality_context = "**CONTEXT: QUALITY UNKNOWN.**"

        # --- CONFIGURATION ---
        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(quality_context),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            temperature=1.0, 
            response_mime_type="application/json",
            response_schema=list[DetectionResult],
        )

        # --- THE PAYLOAD (With x2 Prompt Repetition Hack) ---
        execution_query = "Analyze EACH of the attached images for SYNTHETIC GENERATION ARTIFACTS, strictly following the system instructions."
        request_contents = image_parts + [execution_query, execution_query]

        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=request_contents,
            config=config
        )

        raw_results = response.parsed
        
        if not raw_results:
             return {"confidence": 0.5, "explanation": "Suspicious: No clear analysis returned."}

        ai_votes = [r for r in raw_results if r.confidence > 0.5]
        not_ai_votes = [r for r in raw_results if r.confidence <= 0.5]

        final_result = {}

        if len(ai_votes) > len(not_ai_votes):
            avg_conf = sum(r.confidence for r in ai_votes) / len(ai_votes)
            best_explanation_item = max(ai_votes, key=lambda x: x.confidence)
            final_result = {
                "confidence": round(avg_conf, 2),
                "explanation": best_explanation_item.explanation,
                "quality_context": quality_context
            }
        else:
            if not_ai_votes:
                avg_conf = sum(r.confidence for r in not_ai_votes) / len(not_ai_votes)
                best_explanation_item = min(not_ai_votes, key=lambda x: x.confidence)
                explanation = best_explanation_item.explanation
            else:
                avg_conf = 0.0
                explanation = "No visual anomalies detected"

            final_result = {
                "confidence": round(avg_conf, 2),
                "explanation": explanation,
                "quality_context": quality_context
            }
            
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