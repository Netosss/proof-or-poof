import os
import cv2
import numpy as np
from PIL import Image

def get_quality_context_original(image_source):
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
                return "**CONTEXT: QUALITY UNKNOWN (Could not open).**", 0
            was_path = True
            filename = os.path.basename(image_source)
        else:
            img_pil = image_source
            was_path = False
            filename = "Image Object"

        # --- LAYER 1: DQT (JPEG Artifacts) ---
        # Only works if original metadata is preserved
        avg_q_val = 0
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

        if was_path:
            img_pil.close()

        # --- FINAL VERDICT ---
        score = max(0, score) # Clamp at 0
        
        context_str = "**CONTEXT: HIGH QUALITY IMAGE.** Image is sharp and clean. Any visual anomaly is likely a sign of manipulation."
        if score < 50:
            context_str = f"**CRITICAL CONTEXT: LOW QUALITY IMAGE ({', '.join(issues)}).** INSTRUCTION: This image is heavily compressed/low-res. Warning: This quality naturally causes **WAXY SKIN**, **SMOOTH TEXTURES**, distorted faces, and blur. **IGNORE** any 'waxy' or 'smooth' appearance—it is due to low quality, not AI. Only flag **STRUCTURAL IMPOSSIBILITIES** that compression CANNOT explain (e.g., gibberish text, extra limbs). If unsure, assume it is compression."
        elif score < 80:
             context_str = f"**CONTEXT: MEDIUM QUALITY ({', '.join(issues)}).** INSTRUCTION: Image has reduced quality. Do NOT flag soft edges or indistinct textures as AI. However, compression does NOT explain structural impossibilities (like extra fingers, gibberish text, or impossible physics). If you see these CLEAR logic errors, FLAG THEM."
        
        return context_str, score

    except Exception as e:
        return "**CONTEXT: QUALITY UNKNOWN.** Proceed with standard analysis.", 0

def detailed_debug(filename):
    img_path = os.path.join("/Users/netanel.ossi/Downloads", filename)
    print(f"Analyzing {filename} with ORIGINAL (reference) method...")
    
    if not os.path.exists(img_path):
        print("File not found.")
        return

    context, score = get_quality_context_original(img_path)
    print(f"\nFinal Score: {score}")
    print(f"Context String:\n{context}")

if __name__ == "__main__":
    detailed_debug("130206.jpg")
