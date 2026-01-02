FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

WORKDIR /app

# Install system dependencies for C2PA, OpenCV, and Build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Bind to the dynamic $PORT provided by Railway
# We run the script directly, which now handles the PORT environment variable
CMD ["python", "app/main.py"]
