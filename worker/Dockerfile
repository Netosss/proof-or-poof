# Use a more modern PyTorch base image (2.2.1+) to avoid the _pytree registration error
FROM runpod/pytorch:2.2.1-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Since Build Context is '.' (root), we point to the worker folder
COPY worker/requirements.txt .

# Upgrade pip and force install dependencies to clear any old artifacts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the handler from the worker folder
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
