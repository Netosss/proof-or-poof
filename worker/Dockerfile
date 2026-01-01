# Stage 1: Model Downloader
FROM python:3.10-slim as builder

WORKDIR /models
RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='runwayml/stable-diffusion-v1-5', local_dir='/models/stable-diffusion-v1-5')"

# Stage 2: Final Runtime
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models from builder
COPY --from=builder /models /models

# Copy handler
COPY handler.py .

CMD [ "python", "-u", "handler.py" ]

