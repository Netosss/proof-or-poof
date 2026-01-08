
import os
from PIL import Image
from app.detectors.utils import get_exif_data
from app.detectors.metadata import get_ai_suspicion_score
from app.scoring_config import ScoringConfig

def diagnose_fp():
    # Use one of the Kaggle files that failed before
    # Note: These are in tests/data/kaggle_benchmark/Real/ or AI/
    # Let's try to find them.
    
    # portrait-faces-and-photography-french-wo.jpg
    # beautiful-scenery-rock-formations-sea-qu
    
    target_files = [
        "tests/data/kaggle_benchmark/RealArt/RealArt/portrait-faces-and-photography-french-woman-clara.jpg",
        "tests/data/kaggle_benchmark/RealArt/RealArt/beautiful-scenery-rock-formations-sea-queens-bath-kauai-hawaii-sunset-186645179.jpg"
    ]
    
    print(f"{'File':<40} | {'Score':<6} | {'Signals'}")
    print("-" * 100)
    
    for path in target_files:
        if not os.path.exists(path):
            # Try without prefix
            alt_path = path.replace("real_", "").replace("ai_", "")
            if os.path.exists(alt_path): path = alt_path
            else: continue
            
        img = Image.open(path)
        exif = get_exif_data(img)
        file_size = os.path.getsize(path)
        filename = os.path.basename(path)
        
        score, signals = get_ai_suspicion_score(exif, img.width, img.height, file_size, filename)
        
        print(f"{filename:<40} | {score:<6.2f} | {', '.join(signals)}")

if __name__ == "__main__":
    diagnose_fp()
