import logging
from app.c2pa_reader import get_c2pa_manifest

logger = logging.getLogger(__name__)

async def detect_ai_media(file_path: str) -> dict:
    """
    Detect if an image/video is AI-generated based on C2PA metadata.
    This is a deterministic provenance check.
    """
    manifest = get_c2pa_manifest(file_path)
    
    if manifest:
        # Try to get the specific product name from claim_generator_info
        gen_info = manifest.get("claim_generator_info", [])
        if gen_info and isinstance(gen_info, list) and len(gen_info) > 0:
            # Often the first item contains the most specific info
            generator = gen_info[0].get("name", "Unknown AI")
        else:
            # Fallback to the top-level claim_generator string
            generator = manifest.get("claim_generator", "Unknown AI")

        logger.info(f"C2PA Manifest found! Provider: {generator}")

        return {
            "is_ai": True,
            "provider": generator,
            "method": "c2pa",
            "confidence": 1.0
        }
    
    logger.info("No C2PA manifest found in file.")
    # If missing, we cannot claim AI deterministically
    return {
        "is_ai": None,
        "provider": None,
        "method": "none",
        "confidence": 0.0
    }

