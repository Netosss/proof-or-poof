
import os
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Hemg/AI-Generated-vs-Real-Images-Datasets"
DEST_ROOT = "tests/data/benchmark_hf_50"

def download_subset(folder, label, count):
    print(f"Downloading {count} {label} images...")
    dest_dir = os.path.join(DEST_ROOT, label.lower())
    os.makedirs(dest_dir, exist_ok=True)
    
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    # Filter for the specific folder and image extensions
    target_files = [f for f in all_files if f.startswith(folder) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    # Take the first 'count' files
    subset = target_files[:count]
    
    for i, file_path in enumerate(subset):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{count}] Downloading {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        # Move file to the correct location because hf_hub_download preserves structure
        final_dest = os.path.join(dest_dir, filename)
        source_path = os.path.join(dest_dir, file_path)
        if os.path.exists(source_path):
            os.rename(source_path, final_dest)

if __name__ == "__main__":
    download_subset("AiArtData/AiArtData/", "AI", 25)
    download_subset("RealArt/RealArt/", "Original", 25)
    # Cleanup empty nested folders
    import shutil
    for folder in ["ai", "original"]:
        nested_ai = os.path.join(DEST_ROOT, folder, "AiArtData")
        nested_real = os.path.join(DEST_ROOT, folder, "RealArt")
        if os.path.exists(nested_ai): shutil.rmtree(nested_ai)
        if os.path.exists(nested_real): shutil.rmtree(nested_real)
