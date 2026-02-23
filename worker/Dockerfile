# Use the stable RunPod PyTorch devel image (best for CUDA compatibility)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered (shows up immediately in RunPod UI)
ENV PYTHONUNBUFFERED=1
# RunPod FlashBoot mounts cached models here
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub

WORKDIR /app

# Copy requirements
COPY worker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- MODEL LOADING ---
# Model is pre-loaded via RunPod FlashBoot (see endpoint settings)
# If FlashBoot is disabled, uncomment below to bake model into image:
# RUN python3 -c "import torch; from transformers import AutoImageProcessor, AutoModelForImageClassification; \
# model_id = 'haywoodsloan/ai-image-detector-dev-deploy'; \
# AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True); \
# AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True);"

# Copy the handler
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]