#!/usr/bin/env python3
"""
Extract deep metadata from Kaggle RealArt false positives.
Goal: Identify missing "Original" signals we can use to improve accuracy.
"""

import os
import sys
from PIL import Image
from app.detectors.utils import get_exif_data

# Known false positives from live GPU benchmark
FALSE_POSITIVES = [
    "forest-landscape_71767-127.jpg",
    "homeless-man-color-poverty.jpg",
    "Portrait075a-819x1024.jpg",
    "4-23-4-22-8-4-13m.jpg",
    "large-beautiful-print-of-village-scenery-waterproof-texture-original-imafzgbusjtaysub.jpeg",
    "what-color-paintings-sell-best.jpg",
    "ef2646b821cca54a5ad1cdfcf95d2a1a.jpg",
    "PaintingBBR147_1.jpg",
    "360_F_380747975_sS1hCVB0qPqFCWBMZ3qJ5xTqH6rtaDBI.jpg",
]

BASE_DIR = "tests/data/kaggle_benchmark/RealArt/RealArt"

def analyze_metadata(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ File not found: {filename}")
        return
    
    print(f"\n{'='*80}")
    print(f"FILE: {filename}")
    print(f"{'='*80}")
    
    img = Image.open(path)
    exif = get_exif_data(img)
    
    print(f"Format: {img.format}")
    print(f"Mode: {img.mode}")
    print(f"Size: {img.size}")
    print(f"File Size: {os.path.getsize(path)} bytes")
    
    if exif:
        print(f"\n[EXIF DATA] ({len(exif)} fields)")
        for key, value in sorted(exif.items()):
            # Truncate long values
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"  {key}: {val_str}")
    else:
        print("\n[EXIF DATA] None found")
    
    # Check PIL info
    if img.info:
        print(f"\n[PIL INFO] ({len(img.info)} fields)")
        for key, value in img.info.items():
            if key not in ['exif', 'icc_profile']:  # Skip binary data
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  {key}: {val_str}")

if __name__ == "__main__":
    print("KAGGLE REALART FALSE POSITIVE METADATA ANALYSIS")
    print("=" * 80)
    
    for fp in FALSE_POSITIVES:
        analyze_metadata(fp)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete. Total files analyzed: {len(FALSE_POSITIVES)}")
