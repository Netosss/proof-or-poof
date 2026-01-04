import logging
import cv2
import numpy as np
import asyncio
import hashlib
import os
import io
import tempfile
from collections import OrderedDict
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Optional, List, Union
from app.c2pa_reader import get_c2pa_manifest
from app.runpod_client import run_deep_forensics
from app.security import security_manager

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

def get_image_hash(source: Union[str, Image.Image]) -> str:
    """Generate a secure SHA-256 hash of the image source."""
    if isinstance(source, str):
        with open(source, 'rb') as f:
            # Hash first 2MB for speed, but use secure method
            return security_manager.get_safe_hash(f.read(2048 * 1024))
    else:
        # For PIL Images, hash a small thumbnail
        thumb = source.copy()
        thumb.thumbnail((128, 128))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG")
        return security_manager.get_safe_hash(buf.getvalue())

def get_exif_data(file_path: str) -> dict:
    """Extract EXIF metadata from the image. Explicitly closed via 'with'."""
    try:
        with Image.open(file_path) as img:
            exif = img._getexif() or {}
            exif_data = {}
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
            return exif_data
    except Exception:
        return {}

def extract_video_frames(video_path: str, num_frames: int = 2) -> list:
    """Extract N frames from a video file."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
            
        # Sampling points
        sample_points = [int(total_frames * 0.1), int(total_frames * 0.9)]
        
        for pos in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
    return frames

async def detect_ai_media(file_path: str) -> dict:
    """Final Optimized Consensus Engine."""
    l1_data = {
        "status": "not_found",
        "provider": None,
        "description": "No cryptographic signature found."
    }

    # --- 1️⃣ LAYER 1: C2PA ---
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

    is_video = file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
    
    if is_video:
        safe_path = security_manager.sanitize_log_message(file_path)
        logger.info(f"Detecting AI in video (Parallel Frames): {safe_path}")
        frames = extract_video_frames(file_path)
        if not frames:
            return {"error": "Could not extract frames from video."}
            
        # Process frames in parallel for 2x speedup
        tasks = [detect_ai_media_image_logic(None, frame=f) for f in frames]
        frame_results = await asyncio.gather(*tasks)
        
        # Smart Early Exit: If any frame is confidently human, the video is likely real
        for res in frame_results:
            if res.get("summary") == "Verified Human (Exif Confidence)":
                logger.info(f"Video Early Exit: Frame scan found trusted human metadata.")
                return res
        
        avg_prob = sum(r['layers']['layer2_forensics']['probability'] for r in frame_results) / len(frame_results)
        
        if avg_prob > 0.85: summary = "Likely AI Video"
        elif avg_prob > 0.5: summary = "Suspicious Video"
        else: summary = "Likely Human Video"
        
        return {
            "summary": summary,
            "confidence_score": round(avg_prob if avg_prob > 0.5 else 1.0 - avg_prob, 2),
            "layers": {
                "layer1_metadata": l1_data,
                "layer2_forensics": {
                    "status": "detected" if avg_prob > 0.5 else "not_detected",
                    "probability": round(avg_prob, 4),
                    "signals": [f"Analyzed {len(frame_results)} frames via Parallel Temporal Sampling"]
                }
            }
        }
    else:
        return await detect_ai_media_image_logic(file_path, l1_data)

async def detect_ai_media_image_logic(file_path: Optional[str], l1_data: dict = None, frame: Image.Image = None) -> dict:
    """Refactored image logic. No more long-lived file handles."""
    if l1_data is None:
        l1_data = {"status": "not_found", "provider": None, "description": "N/A"}

    # --- 2️⃣ LAYER 2: Pre-filter ---
    if frame:
        img_for_res = frame
        exif = {} 
        source_for_hash = frame
        source_path = None
        width, height = img_for_res.size
    else:
        exif = get_exif_data(file_path)
        try:
            # Use 'with' to get size and close handle immediately
            with Image.open(file_path) as img:
                width, height = img.size
            source_for_hash = file_path
            source_path = file_path
        except:
            return {"error": "Invalid image file"}
    
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()

    pre_filter_human_score = 0.0
    ai_suspicion_score = 0.0
    
    # Trusted Signals
    trusted_makes = ["canon", "nikon", "sony", "fujifilm", "panasonic", "olympus", "leica", "apple", "samsung", "google"]
    has_trusted_make = any(m in make for m in trusted_makes)
    if has_trusted_make:
        pre_filter_human_score += 0.3
        
    trusted_software = ["lightroom", "photoshop", "capture one", "gimp", "apple", "android", "darktable"]
    if any(s in software for s in trusted_software):
        pre_filter_human_score += 0.3
        
    if width > 2500 and height > 2000:
        pre_filter_human_score += 0.2
        if has_trusted_make:
            pre_filter_human_score += 0.2 

    # AI Suspicion
    is_pow2_w = (width > 0) and (width & (width - 1) == 0)
    is_pow2_h = (height > 0) and (height & (height - 1) == 0)
    if (is_pow2_w and is_pow2_h) and width <= 1024:
        ai_suspicion_score += 0.25 

    if has_trusted_make and (width * height) < 500000:
        pre_filter_human_score -= 0.1 
        ai_suspicion_score += 0.1

    net_human_confidence = pre_filter_human_score - ai_suspicion_score

    if net_human_confidence >= 0.7:
        logger.info(f"Pre-filter Confidence ({net_human_confidence}) passed. Skipping GPU.")
        return {
            "summary": "Verified Human (Exif Confidence)",
            "confidence_score": 1.0,
            "layers": {
                "layer0_prefilter": {
                    "status": "trusted_human",
                    "details": f"Make: {make}, Software: {software}, Res: {width}x{height}, NetScore: {round(net_human_confidence, 2)}"
                },
                "layer1_metadata": {"status": "skipped", "provider": make, "description": "EXIF indicates professional photography workflow."},
                "layer2_forensics": {"status": "skipped", "probability": 0.0, "signals": ["Skipped: High Exif confidence."]}
            }
        }

    # --- 3️⃣ LAYER 3: Forensic Fallback ---
    # GPU COST GUARD: Skip huge files
    if not frame and os.path.exists(file_path):
        f_size = os.path.getsize(file_path)
        if f_size > 50 * 1024 * 1024:
            return {"summary": "Ambiguous (File too large)", "confidence_score": 0.5, "layers": {"layer2_forensics": {"status": "skipped"}}}

    img_hash = get_image_hash(source_for_hash)
    cached_score = forensic_cache.get(img_hash)
    if cached_score is not None:
        forensic_probability = cached_score
    else:
        logger.info(f"Ambiguous Pre-filter. Running In-Memory GPU Scan...")
        # NO DISK: Pass PIL Image (source_for_hash) directly to RunPod client
        forensic_probability = await run_deep_forensics(source_for_hash, width, height)
        forensic_cache.put(img_hash, forensic_probability)
    
    l2_data = {
        "status": "detected" if forensic_probability > 0.85 else "suspicious" if forensic_probability > 0.5 else "not_detected",
        "probability": round(forensic_probability, 4),
        "signals": ["Multi-layered consensus applied (Deep Learning + FFT)"]
    }
    
    if forensic_probability > 0.92: summary = "Likely AI (High Confidence)"
    elif forensic_probability > 0.75: summary = "Possible AI (Forensic Match)"
    elif forensic_probability > 0.5: summary = "Suspicious (Inconsistent Pixels)"
    elif forensic_probability > 0.2: summary = "Likely Human (Minor Noise)"
    else: summary = "Likely Human"
    
    confidence_score = forensic_probability if forensic_probability > 0.5 else (1.0 - forensic_probability)
        
    return {
        "summary": summary,
        "confidence_score": round(confidence_score, 2),
        "layers": {"layer1_metadata": l1_data, "layer2_forensics": l2_data}
    }
