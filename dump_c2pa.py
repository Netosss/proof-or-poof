import json
import c2pa
import sys

def dump_manifest(file_path):
    try:
        with c2pa.Reader(file_path) as reader:
            manifest_store = json.loads(reader.json())
            print(json.dumps(manifest_store, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    dump_manifest(sys.argv[1])

