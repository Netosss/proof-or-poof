FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/models

WORKDIR /app

COPY worker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Bake the model directly into the image for instant cold starts
RUN mkdir -p /app/models/hub/checkpoints && \
    curl -L -o /app/models/hub/checkpoints/big-lama.pt \
    "https://github.com/geekyutao/Inpaint-Anything/releases/download/v1.0/big-lama.pt"

COPY worker/handler.py .
COPY worker/remover.py .

CMD [ "python", "-u", "handler.py" ]