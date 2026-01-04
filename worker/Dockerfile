# Use the CUDA 11.8 base image as seen in your logs
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Copy requirements (Fixed paths to be relative to the context)
COPY requirements.txt .

# Upgrade pip and then upgrade torch/torchvision specifically for cu118
# Removed torchaudio to save space and build time (not needed for image detection)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the handler
COPY handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
