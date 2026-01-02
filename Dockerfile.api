FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for C2PA and OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

EXPOSE 8000

# Use uvicorn for production, binding to the dynamic $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]


