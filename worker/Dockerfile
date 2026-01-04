# Use a stable devel image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Copy requirements
COPY worker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD MODEL ---
# This ensures the model is cached in the container image
RUN python3 -c "from transformers import pipeline; pipeline('image-classification', model='haywoodsloan/ai-image-detector-dev-deploy')"

# Copy the handler
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
