import asyncio
import cv2
import numpy as np
import os
import shutil
from app.detector import detect_ai_media
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.detector")
logger.setLevel(logging.INFO)

def create_synthetic_video(path, duration=2, fps=30, is_ai_like=False):
    """Creates a synthetic video."""
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        if is_ai_like:
            # Static noise (simulating high frequency details)
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        else:
            # Moving gradient
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [(i % 255), ((i * 2) % 255), ((i * 3) % 255)]
            cv2.putText(frame, f"Frame {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
        out.write(frame)
    out.release()
    return path

async def verify_video_detection():
    # Use real files from Downloads
    downloads_dir = "/Users/netanel.ossi/Downloads"
    test_files = [
        "video4083730102.mp4"
    ]

    print(f"Verifying real videos from: {downloads_dir}\n")

    print(f"{'File Name':<30} | {'Result':<20} | {'Score':<5} | {'Gemini?'} | {'Quality Context'}")
    print("-" * 120)

    for filename in test_files:
        path = os.path.join(downloads_dir, filename)
        if not os.path.exists(path):
            print(f"{filename:<30} | SKIPPED (Not found)")
            continue

        try:
            result = await detect_ai_media(path)
            
            summary = result.get('summary', 'Unknown')
            score = result.get('confidence_score', 0.0)
            gemini_used = "Yes" if result.get('is_gemini_used') else "No"
            
            # Extract quality context from evidence chain
            quality_context = "N/A"
            for ev in result.get('evidence_chain', []):
                if ev.get('layer') == 'Layer 3: Visual Context':
                    quality_context = ev.get('context_quality', 'N/A')
            
            # Truncate for display
            if len(quality_context) > 50:
                quality_context = quality_context[:47] + "..."

            print(f"{filename:<30} | {summary:<20} | {score:<5} | {gemini_used} | {quality_context}")
            
        except Exception as e:
            print(f"{filename:<30} | FAILED: {str(e)}")

if __name__ == "__main__":
    asyncio.run(verify_video_detection())