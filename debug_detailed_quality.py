import os
import io
import cv2
import numpy as np
from PIL import Image

def detailed_quality_debug(filename):
    downloads_path = "/Users/netanel.ossi/Downloads"
    img_path = os.path.join(downloads_path, filename)
    
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return

    print(f"--- DETAILED QUALITY ANALYSIS FOR: {filename} ---\n")
    
    try:
        img_pil = Image.open(img_path)
        
        # --- 1. DQT ANALYSIS ---
        dqt_score = 0
        avg_q_val = 0
        
        if hasattr(img_pil, 'quantization') and img_pil.quantization:
            table = img_pil.quantization.get(0) # Luminance table
            if table:
                avg_q_val = sum(table) / len(table)
                print(f"[DQT] Average Quantization Value: {avg_q_val:.2f}")
                
                if avg_q_val > 30: 
                    dqt_score = -40
                    print(f"  -> Severe Compression detected (-40 pts)")
                elif avg_q_val > 20: 
                    dqt_score = -20
                    print(f"  -> Moderate Compression detected (-20 pts)")
                else:
                    print(f"  -> Low Compression (Pass)")
        else:
            print("[DQT] No quantization table found (likely PNG/WEBP or stripped)")

        # --- 2. RESOLUTION ANALYSIS ---
        res_score = 0
        w, h = img_pil.size
        pixels = w * h
        print(f"\n[RESOLUTION] Dimensions: {w}x{h} ({pixels:,} pixels)")
        
        if pixels < 1_000_000: # Below 720p equivalent
            res_score = -50
            print(f"  -> Low Resolution (< 1MP) (-50 pts)")
        elif pixels < 2_500_000: # Roughly 1080p
            res_score = -20
            print(f"  -> Medium Resolution (< 2.5MP) (-20 pts)")
        else:
            print(f"  -> High Resolution (Pass)")

        # --- 3. BLUR ANALYSIS ---
        blur_raw_score = 0
        blur_deduction = 0
        sharpness_bonus = 0
        
        img_arr = np.array(img_pil.convert('RGB')) 
        img_cv = img_arr[:, :, ::-1].copy() 
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        blur_raw_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"\n[BLUR] Laplacian Variance Score: {blur_raw_score:.2f}")

        if blur_raw_score < 10: 
            blur_deduction = -60
            print(f"  -> Near Zero Detail (< 10) (-60 pts)")
        elif blur_raw_score < 100: 
            blur_deduction = -25
            print(f"  -> Extreme Blur (< 100) (-25 pts)")
        elif blur_raw_score < 50: 
            blur_deduction = -10
            print(f"  -> Soft Focus (< 50) (-10 pts)")
        elif blur_raw_score > 600 and avg_q_val < 20: 
            sharpness_bonus = 20
            print(f"  -> High Sharpness (> 600 & clean) (+20 pts)")
        elif blur_raw_score > 300 and avg_q_val < 20: 
            sharpness_bonus = 10
            print(f"  -> Good Sharpness (> 300 & clean) (+10 pts)")

        # --- FINAL CALCULATION ---
        base_score = 100
        total_score = base_score + dqt_score + res_score + blur_deduction + sharpness_bonus
        final_score = max(0, total_score)

        print("\n" + "="*40)
        print(f"BASE SCORE:       {base_score}")
        print(f"DQT DEDUCTION:    {dqt_score}")
        print(f"RES DEDUCTION:    {res_score}")
        print(f"BLUR DEDUCTION:   {blur_deduction}")
        print(f"SHARPNESS BONUS:  +{sharpness_bonus}")
        print("-" * 40)
        print(f"FINAL SCORE:      {final_score}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    detailed_quality_debug("130206.jpg")
