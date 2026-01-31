FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (OpenCV + FFprobe for video metadata)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# --- BAKE MODEL INTO IMAGE ---
# This is the exact default path used by torch/hub
ENV TORCH_HOME=/root/.cache/torch
# Create the directory
RUN mkdir -p $TORCH_HOME/hub/checkpoints/
# Copy explicitly from local models/ directory to the container cache
COPY models/big-lama.pt $TORCH_HOME/hub/checkpoints/big-lama.pt

# Copy the application code
COPY . .

# Ensure the app package is recognized
RUN touch app/__init__.py

# ✅ Railway uses the PORT environment variable
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"
