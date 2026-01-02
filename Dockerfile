# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies (needed for compiling Rust extensions in c2pa-python)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into the user local directory
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app
# Ensure binaries from the local user install are in PATH
ENV PATH /root/.local/bin:$PATH

WORKDIR /app

# Install runtime-only system dependencies (Combined to reduce layers)
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libgl1 libglib2.0-0 libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Install runtime-only system dependencies + git
RUN apt-get update && apt-get install -y \
    git \
    libsm6 libxext6 libgl1 libglib2.0-0 libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

RUN git submodule update --init --recursive

# Copy the application code
COPY . .

# Ensure the app folder is treated as a package
RUN touch app/__init__.py

# Expose port for clarity and local dev
EXPOSE 8000

# Healthcheck instruction for Railway/Docker
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run your app using the main.py entry point
CMD ["python", "app/main.py"]
