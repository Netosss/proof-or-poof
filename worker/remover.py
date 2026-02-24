import os
import io
import gc
import torch
import logging
import pillow_heif
from PIL import Image
from simple_lama_inpainting import SimpleLama

logger = logging.getLogger(__name__)
pillow_heif.register_heif_opener()

class OptimizedLama(SimpleLama):
    """Bypasses standard loading to force CUDA and use local baked weights."""
    def __init__(self):
        self.device = torch.device("cuda")
        model_path = os.path.join(os.environ.get("TORCH_HOME", "/app/models"), "hub/checkpoints/big-lama.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing baked model at {model_path}")
            
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device) # Explicitly ensure all sub-modules are on GPU

class FauxLensRemover:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")
            
        logger.info("Initializing FauxLensRemover on RTX 4090...")
        self.model = OptimizedLama()
        self._warmup()

    def _warmup(self):
        """Pre-compiles the JIT graph to eliminate cold-start latency."""
        logger.info("Executing JIT warmup...")
        dummy_img = Image.new('RGB', (512, 512), (0, 0, 0))
        dummy_mask = Image.new('L', (512, 512), 0)
        
        # Kept the developer's Float32 fix to prevent NaN/Black screens
        with torch.inference_mode():
            _ = self.model(dummy_img, dummy_mask)
        logger.info("Warmup complete. Worker ready.")

    # API FIX: Renamed back to process_image to match handler.py
    def process_image(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # THE MEMORY FIX: 1536px cap for Float32
        # Float32 uses double the VRAM. 2048px will exceed 24GB and cause GPU freezing/swapping.
        MAX_SIZE = 1536 
        BUCKET = 64
        w, h = img.size
        scale = min(1.0, MAX_SIZE / max(w, h))
        
        if scale < 1.0 or w % BUCKET != 0 or h % BUCKET != 0:
            new_w = max(BUCKET, (int(w * scale) // BUCKET) * BUCKET)
            new_h = max(BUCKET, (int(h * scale) // BUCKET) * BUCKET)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

        if img.size != mask.size:
             mask = mask.resize(img.size, Image.Resampling.NEAREST)

        # RTX 4090 Inference in Float32 (Accurate, no black boxes)
        with torch.inference_mode():
            result = self.model(img, mask)

        # Removed the manual Tensor conversion block. 
        # The SimpleLama wrapper natively returns a PIL Image.

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        
        # Standard GC, avoiding empty_cache() to keep CUDA allocator fast
        del img, mask, result
        gc.collect() 
        
        return out_io.getvalue()