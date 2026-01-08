
import kagglehub
import shutil
import os

def main():
    print("Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
    print(f"Downloaded to cache: {path}")

    target_dir = os.path.abspath("tests/data/kaggle_benchmark")
    
    if os.path.exists(target_dir):
        print(f"Removing existing benchmark directory: {target_dir}")
        shutil.rmtree(target_dir)

    print(f"Copying to {target_dir}...")
    shutil.copytree(path, target_dir)
    
    # List contents to verify structure
    print("\nDataset structure:")
    for root, dirs, files in os.walk(target_dir):
        level = root.replace(target_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")

    print("\nDone.")

if __name__ == "__main__":
    main()
