import logging
import cv2
import numpy as np
import asyncio
import hashlib
from collections import OrderedDict
from PIL import Image
from PIL.ExifTags import TAGS
from app.c2pa_reader import get_c2pa_manifest
from app.runpod_client import run_deep_forensics

logger = logging.getLogger(__name__)

# LRU Cache implementation for forensic results
class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

forensic_cache = LRUCache(capacity=1000)

def get_image_hash(file_path: str) -> str:
    """Generate a quick MD5 hash of the image to use for caching."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # Read only the first 1MB for speed
        hasher.update(f.read(1024 * 1024))
    return hasher.hexdigest()

def get_exif_data(file_path: str) -> dict:
    """Extract EXIF metadata from the image."""
    try:
        img = Image.open(file_path)
        exif = img._getexif() or {}
        exif_data = {}
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            exif_data[decoded] = value
        return exif_data
    except Exception as e:
        logger.error(f"Error extracting EXIF: {e}")
        return {}

def _run_fft_sync(image_path: str) -> float:
    # Deprecated: FFT now handled by RunPod worker for consensus
    return 0.0

async def get_fft_score(image_path: str) -> float:
    # Deprecated: FFT now handled by RunPod worker for consensus
    return 0.0

async def detect_ai_media(file_path: str) -> dict:
    """
    Final Optimized Consensus Engine.
    
    Layer 1: C2PA (Source of Truth) - 100% Weight, early exit.
    Layer 2: Pre-filter (Exif/Res/Software) - High confidence human check.
    Layer 3: Forensic Fallback (FFT + SigLIP2 GPU) - Weighted deep scan.
    """
    
    # Initialize default metadata object
    l1_data = {
        "status": "not_found",
        "provider": None,
        "description": "No cryptographic signature found. Digital 'passport' may have been stripped."
    }

    # --- 1️⃣ LAYER 1: C2PA (Source of Truth) ---
    manifest = get_c2pa_manifest(file_path)
    if manifest:
        gen_info = manifest.get("claim_generator_info", [])
        generator = gen_info[0].get("name", "Unknown AI") if gen_info else manifest.get("claim_generator", "Unknown AI")

        is_generative_ai = False
        assertions = manifest.get("assertions", [])
        for assertion in assertions:
            if assertion.get("label") == "c2pa.actions.v2":
                actions = assertion.get("data", {}).get("actions", [])
                for action in actions:
                    source_type = action.get("digitalSourceType", "")
                    if "trainedAlgorithmicMedia" in source_type:
                        is_generative_ai = True
                    desc = action.get("description", "").lower()
                    if any(term in desc for term in ["generative fill", "ai-modified", "edited with ai", "ai generated"]):
                        is_generative_ai = True
            if is_generative_ai: break

        l1_data = {
            "status": "verified_ai" if is_generative_ai else "verified_human",
            "provider": generator,
            "description": f"Verified AI signature found ({generator})." if is_generative_ai else "Verified human-captured content."
        }

        # Trust cryptographic proof and EXIT.
        return {
            "summary": "Verified AI" if is_generative_ai else "Verified Human",
            "confidence_score": 1.0,
            "layers": {
                "layer1_metadata": l1_data,
                "layer2_forensics": {
                    "status": "skipped", 
                    "probability": 1.0 if is_generative_ai else 0.0, 
                    "signals": ["Source verified via cryptographic signature."]
                }
            }
        }

    # --- 2️⃣ LAYER 2: Pre-filter (Exif / Resolution / Software) ---
    exif = get_exif_data(file_path)
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()
    
    try:
        with Image.open(file_path) as img:
            width, height = img.size
    except:
        width, height = 0, 0

    # Calculate Pre-filter Human Score
    pre_filter_human_score = 0.0
    
    # Real camera manufacturer (+0.3)
    trusted_makes = ["canon", "nikon", "sony", "fujifilm", "panasonic", "olympus", "leica", "apple", "samsung", "google"]
    if any(m in make for m in trusted_makes):
        pre_filter_human_score += 0.3
        
    # Known professional software (+0.3)
    trusted_software = ["lightroom", "photoshop", "capture one", "gimp", "apple", "android"]
    if any(s in software for s in trusted_software):
        pre_filter_human_score += 0.3
        
    # High-resolution image (+0.2)
    if width > 2000 and height > 2000:
        pre_filter_human_score += 0.2

    # Decision: If score >= 0.7, confidently human.
    if pre_filter_human_score >= 0.7:
        logger.info(f"Pre-filter Confidence ({pre_filter_human_score}) passed. Skipping GPU.")
        return {
            "summary": "Verified Human (Exif Confidence)",
            "confidence_score": 1.0,
            "layers": {
                "layer0_prefilter": {
                    "status": "trusted_human",
                    "details": f"Make: {make}, Software: {software}, Res: {width}x{height}, Score: {pre_filter_human_score}"
                },
                "layer1_metadata": {"status": "skipped", "provider": make, "description": "EXIF indicates professional photography workflow."},
                "layer2_forensics": {"status": "skipped", "probability": 0.0, "signals": ["Skipped: High Exif confidence."]}
            }
        }

    # --- 3️⃣ LAYER 3: Forensic Fallback (SigLIP2 + Worker-side Consensus) ---
    # We no longer apply a local bias here because the RunPod worker 
    # now performs its own consensus (SigLIP2 + FFT + High-Res Bias).
    
    # Deep Forensic (SigLIP2 on RunPod)
    img_hash = get_image_hash(file_path)
    cached_score = forensic_cache.get(img_hash)
    if cached_score is not None:
        forensic_probability = cached_score
    else:
        logger.info(f"Ambiguous Pre-filter ({pre_filter_human_score}). Running GPU Scan...")
        # Get the final consensus score from the worker
        forensic_probability = await run_deep_forensics(file_path, width, height)
        forensic_cache.put(img_hash, forensic_probability)
    
    l2_data = {
        "status": "detected" if forensic_probability > 0.85 else "suspicious" if forensic_probability > 0.5 else "not_detected",
        "probability": round(forensic_probability, 4),
        "signals": []
    }
    
    if forensic_probability > 0.85: l2_data["signals"].append("Deep Learning identifies generative AI textures")
    if forensic_probability > 0.5: l2_data["signals"].append("Pixel forensics suggest artificial origin")
    l2_data["signals"].append("Multi-layered consensus applied (SigLIP2 + FFT)")

    # --- Verdict Logic ---
    if forensic_probability > 0.92: 
        summary = "Likely AI (High Confidence)"
    elif forensic_probability > 0.75: 
        summary = "Possible AI (Forensic Match)"
    elif forensic_probability > 0.5: 
        summary = "Suspicious (Inconsistent Pixels)"
    elif forensic_probability > 0.2: 
        summary = "Likely Human (Minor Noise)"
    else: 
        summary = "Likely Human"
    
    confidence_score = forensic_probability if forensic_probability > 0.5 else (1.0 - forensic_probability)
        
    return {
        "summary": summary,
        "confidence_score": round(confidence_score, 2),
        "layers": {"layer1_metadata": l1_data, "layer2_forensics": l2_data}
    }
