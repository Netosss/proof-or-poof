import sys
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def dump_metadata(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\n{'='*80}")
    print(f"DEEP METADATA DUMP: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    try:
        with Image.open(file_path) as img:
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
            print(f"Info Keys: {list(img.info.keys())}")
            
            # 1. Basic Info
            if 'icc_profile' in img.info:
                print(f"[ICC] Profile Present (Size: {len(img.info['icc_profile'])} bytes)")
            
            # 2. EXIF
            exif = img._getexif()
            if exif:
                print("\n--- EXIF DATA ---")
                for tag, value in exif.items():
                    decoded = TAGS.get(tag, tag)
                    if decoded == "GPSInfo":
                        print("  [GPS Info Present]")
                        for t in value:
                            sub_tag = GPSTAGS.get(t, t)
                            print(f"    {sub_tag}: {value[t]}")
                    else:
                        print(f"  {decoded}: {value}")
            else:
                print("\n[EXIF] No EXIF data found.")

            # 3. XMP (String dump if exists)
            if 'xmp' in img.info:
                print("\n--- XMP DATA (Raw) ---")
                try:
                    # Often bytes, we try to decode
                    xmp_str = img.info['xmp'].decode('utf-8', errors='ignore')
                    # Print first 500 chars to avoid bloat
                    print(xmp_str[:1000] + "...")
                except:
                    print("  [XMP] Could not decode bytes.")
            
            # 4. IPTC (via info)
            if 'photoshop' in img.info:
                print("\n--- IPTC / Photoshop Data Present ---")

    except Exception as e:
        print(f"Error reading metadata: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deep_metadata_dump.py <path_to_image>")
    else:
        dump_metadata(sys.argv[1])
