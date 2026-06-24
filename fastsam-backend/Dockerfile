FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

RUN python - <<'PY'
from ultralytics.utils.downloads import attempt_download_asset

for name in ("FastSAM-x.pt", "FastSAM-s.pt", "mobile_sam.pt", "sam_b.pt"):
    path = attempt_download_asset(name, release="latest")
    print(f"downloaded {name} -> {path}")
PY

COPY main.py /app/main.py

ENV FASTSAM_MODEL_PATH=/app/FastSAM-x.pt
ENV SAM_MODEL_PATH=/app/sam_b.pt
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
