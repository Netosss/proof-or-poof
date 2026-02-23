import os
import io
import cv2
import numpy as np
from PIL import Image

def analyze_208_quality(filename):
    downloads_path = "/Users/netanel.ossi/Downloads"
    img_path = os.path.join(downloads_path, filename)
    
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return

    print(f"--- QUALITY ANALYSIS FOR: {filename} ---\n")
    
    try:
        img_pil = Image.open(img_path)
        
        # --- 1. DQT ANALYSIS ---
        dqt_deduction = 0
        avg_q_val = 0
        if hasattr(img_pil, 'quantization') and img_pil.quantization:
            table = img_pil.quantization.get(0) # Luminance table
            if table:
                avg_q_val = sum(table) / len(table)
                print(f"[DQT] Average Quantization Value: {avg_q_val:.2f}")
                
                # Logic from gemini_client.py
                if avg_q_val > 30: 
                    dqt_deduction = -40
                    print(f"  -> Severe Compression (> 30) [-40 pts]")
                elif avg_q_val > 20: 
                    dqt_deduction = -20
                    print(f"  -> Moderate Compression (> 20) [-20 pts]")
        else:
            print("[DQT] No quantization table found.")

        # --- 2. RESOLUTION ANALYSIS ---
        res_deduction = 0
        w, h = img_pil.size
        pixels = w * h
        print(f"\n[RESOLUTION] Dimensions: {w}x{h} ({pixels:,} pixels)")
        
        if pixels < 250_000:
            res_deduction = -50
            print(f"  -> Tiny Resolution (< 250k) [-50 pts]")
        elif pixels < 800_000:
            res_deduction = -40
            print(f"  -> Low Resolution (< 800k) [-40 pts]")
        else:
            print(f"  -> Good Resolution (> 800k) [0 pts]")

        # --- 3. BLUR ANALYSIS ---
        blur_deduction = 0
        img_arr = np.array(img_pil.convert('RGB')) 
        img_cv = img_arr[:, :, ::-1].copy() 
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"\n[BLUR] Laplacian Variance Score: {blur_score:.2f}")

        if blur_score < 10:
            blur_deduction = -60
            print(f"  -> Near Zero Detail (< 10) [-60 pts]")
        elif blur_score < 25:
            blur_deduction = -25
            print(f"  -> Extreme Blur (< 25) [-25 pts]")
        elif blur_score < 50:
            blur_deduction = -10
            print(f"  -> Soft Focus (< 50) [-10 pts]")
        else:
            print(f"  -> Acceptable Sharpness (> 50) [0 pts]")

        # --- FINAL SCORE ---
        base_score = 100
        final_score = max(0, base_score + dqt_deduction + res_deduction + blur_deduction)

        print("\n" + "="*40)
        print(f"BASE SCORE:       {base_score}")
        print(f"DQT DEDUCTION:    {dqt_deduction}")
        print(f"RES DEDUCTION:    {res_deduction}")
        print(f"BLUR DEDUCTION:   {blur_deduction}")
        print("-" * 40)
        print(f"FINAL SCORE:      {final_score}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    analyze_208_quality("130208.jpg")
