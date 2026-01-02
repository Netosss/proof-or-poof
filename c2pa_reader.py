import sys
import os
import json
from typing import Optional, Dict, Any

# Vendor the c2pa-python library by adding its source to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party", "c2pa-python", "src"))

try:
    import c2pa
except ImportError:
    # Fallback for environments where it might be installed normally
    import c2pa

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



