# Use the stable RunPod PyTorch devel image (best for CUDA compatibility)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered (shows up immediately in RunPod UI)
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache

WORKDIR /app

# Copy requirements
COPY worker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD MODEL ---
# This ensures the model is baked into the image layers for fast cold starts
RUN python3 - << 'EOF'
from transformers import AutoImageProcessor, AutoModelForImageClassification
model_id = 'haywoodsloan/ai-image-detector-dev-deploy'
AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=True)
EOF

# Copy the handler
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
