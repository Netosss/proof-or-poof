import os
import io
import gc
import torch
import logging
import numpy as np
import pillow_heif
from PIL import Image, ImageOps
from simple_lama_inpainting import SimpleLama

# ------------------------------------------------------------------------
# [CRITICAL FIX] Disable JIT profiling and executor.
# This prevents the C++ backend from throwing a Segmentation Fault (SIGSEGV)
# when a user uploads an image with different dimensions than the warmup image.
# ------------------------------------------------------------------------
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

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

    # Renaming process_image to process to match handler.py's expected method name
    def process(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        MAX_SIZE = 1536
        BUCKET = 64
        w, h = img.size
        original_size = (w, h)
        scale = min(1.0, MAX_SIZE / max(w, h))

        if scale < 1.0:
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            mask = mask.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)
        else:
            scaled_w, scaled_h = w, h

        pad_w = (BUCKET - scaled_w % BUCKET) % BUCKET
        pad_h = (BUCKET - scaled_h % BUCKET) % BUCKET
        if pad_w or pad_h:
            img = Image.fromarray(
                np.pad(np.array(img), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            )
            mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0)

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.Resampling.NEAREST)

        logger.info(f"Running inference on size {img.size}")
        with torch.inference_mode():
            result = self.model(img, mask)
        logger.info("Inference complete")

        if pad_w or pad_h:
            result = result.crop((0, 0, scaled_w, scaled_h))
        if (scaled_w, scaled_h) != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")

        del img, mask, result
        gc.collect()

        return out_io.getvalue()