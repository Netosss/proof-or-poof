import os
import json
from PIL import Image
from PIL.ExifTags import TAGS

def get_exif_data(file_path: str) -> dict:
    """
    Extract metadata from the image (EXIF for JPEG/TIFF, 'info' for PNG/WebP).
    """
    try:
        with Image.open(file_path) as img:
            metadata = {}
            print(f"File: {os.path.basename(file_path)}")
            print(f"Format: {img.format}")
            
            # 1. Standard EXIF (JPEG, TIFF, some WebP)
            exif = img._getexif()
            if exif:
                print("Found EXIF data")
                for tag, value in exif.items():
                    decoded = TAGS.get(tag, tag)
                    metadata[decoded] = value
            
            # 2. PNG/WebP Metadata (Text chunks, etc.)
            if hasattr(img, 'info') and img.info:
                print(f"Found info dictionary keys: {list(img.info.keys())}")
                for key, value in img.info.items():
                    if isinstance(key, str) and isinstance(value, (str, int, float)):
                        if key not in metadata:
                            metadata[key] = value
            
            return metadata
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            res = get_exif_data(path)
            # print(f"Metadata summary: {list(res.keys())}")
            print(f"Metadata content: {json.dumps({k: str(v)[:50] for k, v in res.items()}, indent=2)}")
            print("-" * 30)

