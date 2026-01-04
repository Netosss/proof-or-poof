# Use a specialized PyTorch base image to avoid downloading 5GB of CUDA drivers every build
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Since Build Context is '.' (root), we point to the worker folder
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler from the worker folder
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
