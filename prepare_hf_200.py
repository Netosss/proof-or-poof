
import os
import random
from datasets import load_dataset
from PIL import Image
import io

def prepare_hf_200():
    target_dir = "tests/data/hf_200_benchmark"
    os.makedirs(os.path.join(target_dir, "AI"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "Real"), exist_ok=True)

    # Load existing filenames to exclude
    existing_files = set()
    if os.path.exists("existing_files.txt"):
        with open("existing_files.txt", "r") as f:
            for line in f:
                fn = line.strip()
                if fn:
                    existing_files.add(fn.lower())

    print(f"Loaded {len(existing_files)} existing filenames to exclude.")

    # Load dataset
    print("Loading dataset Hemg/AI-Generated-vs-Real-Images-Datasets...")
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", streaming=True)

    ai_count = 0
    real_count = 0
    target_count = 100

    print("Sampling images...")
    
    # We use streaming to avoid loading the whole 153k dataset if possible,
    # though it might be slow to find non-duplicates if we are unlucky.
    # But since we only need 100 each, it should be fast.
    
    for i, item in enumerate(ds):
        label = item['label']
        image = item['image']
        
        # In this dataset, images might not have a specific 'filename' field in features.
        # Let's check item keys.
        # Features: ['image', 'label'] usually.
        
        # If no filename, we use the index or something. 
        # But user said "validate by file name". 
        # Maybe the image object has some meta?
        
        # Let's assume for now we use index as filename if none found, 
        # but check if we can find any original filename.
        
        orig_fn = f"hf_{i}.jpg" # Default
        
        # If it's a PIL image, filename might be in info or something if it was loaded from disk, 
        # but here it's from parquet.
        
        if label == 0 and ai_count < target_count:
            fn = f"ai_{orig_fn}"
            if fn.lower() not in existing_files:
                image.convert("RGB").save(os.path.join(target_dir, "AI", fn))
                ai_count += 1
                if ai_count % 10 == 0: print(f"Sampled {ai_count} AI images")
        
        elif label == 1 and real_count < target_count:
            fn = f"real_{orig_fn}"
            if fn.lower() not in existing_files:
                image.convert("RGB").save(os.path.join(target_dir, "Real", fn))
                real_count += 1
                if real_count % 10 == 0: print(f"Sampled {real_count} Real images")

        if ai_count >= target_count and real_count >= target_count:
            break

    print(f"Preparation complete. AI: {ai_count}, Real: {real_count}")

if __name__ == "__main__":
    prepare_hf_200()
