FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    python3 -m pip install -r /app/requirements.txt

COPY main.py /app/main.py

ENV FASTSAM_MODEL_PATH=/app/FastSAM-x.pt
ENV SAM_MODEL_PATH=/app/sam_b.pt
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
