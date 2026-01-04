# Use a stable devel image (verified to exist on RunPod Docker Hub)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Copy requirements from the worker directory
COPY worker/requirements.txt .

# We ONLY install the extra libraries. 
# We DO NOT reinstall torch/torchvision because they are already in the base image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the handler
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
