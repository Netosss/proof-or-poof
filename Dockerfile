FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

WORKDIR /app

# Install system dependencies for OpenCV and C2PA
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from PyPI
# This uses pre-built wheels, avoiding the Rust compilation issues
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure the app folder is treated as a package
RUN touch app/__init__.py

# Run your app
CMD ["python", "app/main.py"]
