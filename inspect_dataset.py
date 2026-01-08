
import kagglehub
import os

def main():
    path = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
    print(f"Dataset path: {path}")
    
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")
        if len(files) > 0:
            print(f"{indent}  Sample: {files[0]}")

if __name__ == "__main__":
    main()
