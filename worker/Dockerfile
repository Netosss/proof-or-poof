# Use the stable RunPod PyTorch devel image (best for CUDA compatibility)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered (shows up immediately in RunPod UI)
ENV PYTHONUNBUFFERED=1

# --- CACHE SETTINGS ---
# We bake models INTO the image at this location.
# This avoids downloading at runtime and avoids network volume latency.
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

WORKDIR /app

# Copy requirements (Assumes build context is inside 'worker' folder)
COPY worker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- BAKE MODELS ---
# Copy builder script
COPY worker/builder.py .
# Run builder to download models into HF_HOME
RUN python builder.py && \
    rm builder.py

# Copy the handler
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
