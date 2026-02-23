import os
import glob
import time
import json
import gemini_client

def run_test():
    downloads_path = "/Users/netanel.ossi/Downloads"
    # Specific files from the screenshot
    target_filenames = [
        "132215.jpg",
        "132321.jpg", 
        "132186.jpg",
        "1000216823.jpg",
        "129502.jpg",
        "129705.jpg",
        "129127.jpg",
        "130207.jpg",
        "130206.jpg",
        "130208.jpg",
        "ai.jpeg",
        "AI ART ITA.jpg",
        "130188.jpg",
        "129177 (2).jpg",
        "frame_debug_129896.jpg"
    ]
    
    image_files = []
    for fname in target_filenames:
        full_path = os.path.join(downloads_path, fname)
        if os.path.exists(full_path):
            image_files.append(full_path)
        else:
            print(f"Warning: File not found: {fname}")
    
    # Sort for consistent order
    image_files.sort()
    
    if not image_files:
        print(f"No jpg/jpeg images found in {downloads_path}")
        return

    print(f"Found {len(image_files)} images. Starting analysis...\n")
    print(f"{'Filename':<50} | {'Conf':<6} | {'Qual':<4} | {'Explanation'}")
    print("-" * 120)

    total_start = time.time()
    
    results = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Analyzing {filename}...", end='\r', flush=True)
        try:
            start_time = time.time()
            result = gemini_client.analyze_image_pro_turbo(img_path)
            end_time = time.time()
            latency = end_time - start_time
            
            confidence = result.get('confidence', -1)
            explanation = result.get('explanation', 'Error')
            quality_score = result.get('quality_score', 0)
            
            # Truncate explanation for display
            display_expl = (explanation[:55] + '..') if len(explanation) > 55 else explanation
            
            print(f"{filename:<50} | {confidence:<6.2f} | {quality_score:<4} | {display_expl}")
            
            results.append({
                "filename": filename,
                "path": img_path,
                "confidence": confidence,
                "quality_score": quality_score,
                "explanation": explanation,
                "latency": latency,
                "usage": result.get("usage", {})
            })
            
        except Exception as e:
            print(f"{filename:<50} | ERROR  | {str(e)}")

    total_end = time.time()
    print("-" * 120)
    print(f"\nTotal time: {total_end - total_start:.2f}s")
    
    # Save full results to a file for review
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to test_results.json")

if __name__ == "__main__":
    run_test()
