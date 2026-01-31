import os
import io
import torch
import logging
import pillow_heif
from PIL import Image, ImageOps
from simple_lama_inpainting import SimpleLama

# Configure logging
logger = logging.getLogger(__name__)

# 1. Register HEIC opener
pillow_heif.register_heif_opener()

class SafeSimpleLama(SimpleLama):
    """
    A wrapper around SimpleLama to ensure safe loading on CPU-only environments.
    The original library fails to load CUDA-saved models on CPU machines because
    it misses map_location='cpu'.
    """
    def __init__(self):
        # Do not call super().__init__() because it crashes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Manual model discovery logic (aligned with Dockerfile paths)
        # 1. Try env var path
        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            model_path = os.path.join(torch_home, "hub/checkpoints/big-lama.pt")
        else:
            # 2. Fallback to default cache
            model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/big-lama.pt")
            
        if not os.path.exists(model_path):
            # 3. If file missing, let the library download it (but this might crash on load again)
            # We assume it exists because we baked it or curled it.
            logger.warning(f"Model not found at {model_path}, attempting default load (may crash on CPU)...")
            super().__init__()
            return

        logger.info(f"Loading model from {model_path} with map_location={self.device}")
        
        # THE FIX: map_location=self.device
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

class FauxLensRemover:
    def __init__(self):
        """
        Initialize the LaMA model for object removal.
        Loads the model into memory on startup (Warm Start).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 FauxLensRemover initializing on device: {self.device}")
        
        try:
            # Use our safe wrapper
            self.model = SafeSimpleLama()
            # No need to call .to() again as the wrapper did it, but it doesn't hurt
            if hasattr(self.model, "model"):
                 self.model.model.to(self.device)
            logger.info("✅ LaMA model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load LaMA model: {e}")
            raise e

    def process_image(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        """
        Process the image to remove the area defined by the mask.
        
        Args:
            image_bytes: Raw bytes of the original image (JPG/PNG/HEIC).
            mask_bytes: Raw bytes of the mask image (White = Remove).
            
        Returns:
            bytes: The processed image in PNG format.
        """
        # --- STAGE 1: LOAD & FIX FORMATS ---
        try:
            # Handle iPhone (HEIC), WebP, PNG, etc. automatically
            image = Image.open(io.BytesIO(image_bytes))
            mask = Image.open(io.BytesIO(mask_bytes))
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            raise ValueError("Invalid image format. Please upload JPG, PNG, or HEIC.")

        # Force RGB (Fixes Transparent PNGs and CMYK print images)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Mask must be Grayscale (L)
        if mask.mode != "L": 
            mask = mask.convert("L")

        # --- STAGE 2: SMART RESIZE (The "500k" Answer) ---
        # Strategy: If image is HUGE (>2048px), shrink it to prevent crash/timeout.
        # If it's small, keep it original quality.
        MAX_SIZE = 2048 
        
        if max(image.size) > MAX_SIZE:
            # Calculate new size maintaining aspect ratio
            scale = MAX_SIZE / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            
            logger.info(f"Resizing huge image from {image.size} to {new_size}")
            
            # High-quality downscale (LANCZOS)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            # Mask must match exactly!
            mask = mask.resize(new_size, Image.Resampling.NEAREST)

        # --- STAGE 3: SAFETY CHECK ---
        # Edge Case: User uploads a mask that doesn't match the image size
        if image.size != mask.size:
             logger.warning(f"Mask size {mask.size} != Image size {image.size}. Resizing mask.")
             mask = mask.resize(image.size, Image.Resampling.NEAREST)

        # --- STAGE 4: INFERENCE ---
        # The library handles the "whole image" context automatically.
        result = self.model(image, mask)

        # --- STAGE 5: OUTPUT ---
        output_buffer = io.BytesIO()
        # Save as PNG (Lossless) so we don't add JPEG artifacts to the fixed area
        result.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
