"""C2PA manifest reader (wraps the c2pa-python SDK)."""

import json
import c2pa
from typing import Optional, Dict, Any


def get_c2pa_manifest(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Reads and validates C2PA manifest data from a media file.
    Returns the active manifest dict, or None if not present / on any error.
    """
    try:
        with c2pa.Reader(file_path) as reader:
            manifest_store = json.loads(reader.json())
            active_label = manifest_store.get("active_manifest")

            if active_label:
                return manifest_store["manifests"][active_label]
    except Exception:
        pass
    return None
