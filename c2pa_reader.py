import json
from typing import Optional, Dict, Any

# Simple import: the Dockerfile handles the installation from the vendored folder
try:
    import c2pa
except ImportError as e:
    print(f"C2PA Import Error: {e}")
    raise

def get_c2pa_manifest(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Reads and validates C2PA manifest data from a media file.
    """
    try:
        with c2pa.Reader(file_path) as reader:
            manifest_store = json.loads(reader.json())
            active_label = manifest_store.get("active_manifest")
            
            if active_label:
                return manifest_store["manifests"][active_label]
    except Exception:
        # If no manifest is found or error occurs, return None
        pass
    return None
