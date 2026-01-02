FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code (including vendored c2pa-python)
COPY . .

# Build/Install the vendored c2pa-python
RUN pip install --no-cache-dir ./third_party/c2pa-python

# Ensure the app folder is treated as a package
RUN touch app/__init__.py

# Run your app using the main.py entry point
CMD ["python", "app/main.py"]
