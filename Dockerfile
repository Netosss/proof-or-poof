# Stage 1: Build Stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build-time dependencies (Essential for Rust compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy the entire repo including third_party/
COPY . .

# Build and install the vendored c2pa-python library
# This will use the Rust toolchain installed above
RUN pip install --no-cache-dir --user ./third_party/c2pa-python

# Stage 2: Runtime Stage (Smaller & Cleaner)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app
ENV PATH /root/.local/bin:$PATH

WORKDIR /app

# Install only the runtime system libraries
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libgl1 libglib2.0-0 libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the installed packages from the builder
COPY --from=builder /root/.local /root/.local

# Copy the application code
COPY . .

# Ensure the app folder is treated as a package
RUN touch app/__init__.py

# Run your app
CMD ["python", "app/main.py"]
