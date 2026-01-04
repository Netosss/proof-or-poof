# Use the CUDA 11.8 base image as seen in your logs
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Copy requirements from the worker directory
# Assuming build context is the project root
COPY worker/requirements.txt .

# Upgrade pip and then upgrade torch/torchvision specifically for cu118
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the handler from the worker directory
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
