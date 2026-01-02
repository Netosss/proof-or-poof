FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for C2PA and OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Set PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

# Bind to the dynamic $PORT provided by Railway
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
