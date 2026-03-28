"""
Modal GPU worker for LaMa inpainting.

Replaces the RunPod-based worker (Dockerfile + handler.py + remover.py).
The container image is defined programmatically; the model is baked in at
build time so cold starts only need to restore from a GPU memory snapshot.

Deploy:  modal deploy worker/modal_app.py --env main
Dev:     modal serve worker/modal_app.py --env dev
"""

import modal

app = modal.App("proof-or-poof-inpainting")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.10"
    )
    .pip_install(
        "torch==2.2.1",
        "torchvision",
        "Pillow>=10.0.0",
        "pillow-heif>=0.15.0",
        "simple-lama-inpainting==0.1.1",
    )
    .apt_install("wget", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands(
        "mkdir -p /app/models/hub/checkpoints",
        "wget -q -O /app/models/hub/checkpoints/big-lama.pt "
        "'https://github.com/Netosss/proof-or-poof/releases/download/inpaint_model/big-lama.pt'",
    )
    .env({"TORCH_HOME": "/app/models"})
)


@app.cls(
    image=image,
    gpu="L4",
    timeout=30,
    scaledown_window=30,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Inpainter:
    """GPU-accelerated LaMa inpainting with memory-snapshot fast restores."""

    @modal.enter(snap=True)
    def load(self):
        import ctypes
        import time as _time

        cu = ctypes.CDLL("libcuda.so.1")
        for attempt in range(10):
            if cu.cuInit(0) == 0:
                break
            if attempt < 9:
                print(f"cuInit attempt {attempt + 1}/10 failed, retrying...")
                _time.sleep(0.5)
        else:
            print("CUDA init failed after 10 attempts; stopping container")
            modal.experimental.stop_fetching_inputs()
            return

        import torch
        from PIL import Image as PILImage
        from simple_lama_inpainting import SimpleLama

        # Prevents SIGSEGV when image dimensions differ from warmup
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.model = SimpleLama()

        dummy_img = PILImage.new("RGB", (512, 512), (0, 0, 0))
        dummy_mask = PILImage.new("L", (512, 512), 0)
        with torch.inference_mode():
            _ = self.model(dummy_img, dummy_mask)
        print("SimpleLama loaded and JIT warmup complete.")

    @modal.method()
    def process(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        import io
        import gc
        import numpy as np
        import torch
        from PIL import Image as PILImage, ImageOps

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = PILImage.open(io.BytesIO(mask_bytes)).convert("L")

        MAX_SIZE = 1536
        BUCKET = 64
        w, h = img.size
        original_size = (w, h)
        scale = min(1.0, MAX_SIZE / max(w, h))

        if scale < 1.0:
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            img = img.resize((scaled_w, scaled_h), PILImage.Resampling.LANCZOS)
            mask = mask.resize((scaled_w, scaled_h), PILImage.Resampling.NEAREST)
        else:
            scaled_w, scaled_h = w, h

        pad_w = (BUCKET - scaled_w % BUCKET) % BUCKET
        pad_h = (BUCKET - scaled_h % BUCKET) % BUCKET
        if pad_w or pad_h:
            img = PILImage.fromarray(
                np.pad(np.array(img), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            )
            mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0)

        if img.size != mask.size:
            mask = mask.resize(img.size, PILImage.Resampling.NEAREST)

        print(f"Running inference on size {img.size}")
        with torch.inference_mode():
            result = self.model(img, mask)
        print("Inference complete")

        if pad_w or pad_h:
            result = result.crop((0, 0, scaled_w, scaled_h))
        if (scaled_w, scaled_h) != original_size:
            result = result.resize(original_size, PILImage.Resampling.LANCZOS)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")

        del img, mask, result
        gc.collect()

        return out_io.getvalue()
