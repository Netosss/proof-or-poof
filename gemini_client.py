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

        prompt = """
        ### TASK: FORENSIC AI IMAGE ANALYSIS
        You are an expert AI Image Auditor. Your goal is to detect generative artifacts while strictly avoiding false positives on human-edited memes, collages, or digital art.

        ### PHASE 1: TECHNICAL & META SCAN (Priority 1)
        1. **SynthID / Watermarks:** Scan for digital watermarks (e.g., Google SynthID, DALL-E color squares in bottom right, "Made with AI" metadata).
        -> IF FOUND: **Score = 1.0**. Stop analysis. Output: "Positive Match: [Name of Marker] detected."

        ### PHASE 2: STYLE & LOGIC BRANCHING (Crucial)
        Determine the image category to select the correct detection rules.

        **CATEGORY A: PHOTOREALISTIC (Real Life Context)**
        *Trigger:* Image appears to be a camera photo of real people/places.
        * **SCAN FOR (Physics Logic):**
            * **Anatomy:** Melting fingers (>5 or <4), merging limbs, teeth that look like a solid block.
            * **Texture:** Skin that looks like "smooth plastic" or wax (common in Midjourney/Flux).
            * **Physics:** Objects clipping through others, shadows contradicting light sources.
            * **Background:** "Bokeh" blur that creates nonsensical blobs instead of blurred objects.

        **CATEGORY B: NON-PHOTOREALISTIC (Art, Anime, Vector, 3D Render)**
        *Trigger:* Image is a painting, drawing, cartoon, or clearly edited meme.
        * **SCAN FOR (Pattern Logic):**
            * **Pattern Meltdown:** Intricate details (lace, crowds, scales) that start structured but turn into random scribbles/noise.
            * **Incoherent Blending:** Objects merging into the background (e.g., a sword handle melting into a cape).
            * **Gibberish Text:** Background text that looks like alien hieroglyphs or unreadable symbols.
        * **DO NOT FLAG:** "Weird" anatomy (big eyes), floating objects, or "bad" photoshop edges. These are artistic choices, not AI errors.

        ### PHASE 3: FALSE POSITIVE EXCLUSIONS (The "Safety Net")
        1.  **The "Meme" Rule:** If you see **floating text**, **white caption bars**, or **hard-cut edges** (like a face pasted on a body), these are HUMAN EDITS. Ignore the edges. Only check the *content* of the face/body itself.
        2.  **The "Date" Rule:** IGNORE the *chronological value* of any date (e.g., "Feb 2030" is fine). ONLY flag if the text characters *themselves* are malformed/melting.

        ### SCORING & OUTPUT
        * **Score > 0.9 (AI Detected):** Clear evidence found in the correct Category logic.
        * **Score < 0.1 (Clean):** No generative anomalies found.

        **FINAL OUTPUT FORMAT (Strict):**
        - **If AI (>0.5):** [Explain in max 7 words, e.g., "SynthID detected" or "Fingers merging into soda can"]
        - **If NOT AI:** "No visual anomalies detected"
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
        prompt_text = """
        Analyze EACH of the attached images for SYNTHETIC GENERATION ARTIFACTS.
        Return a JSON list where each item corresponds to the images in order.
        
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
