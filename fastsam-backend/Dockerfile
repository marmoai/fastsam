FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV MODEL_CACHE_DIR=/tmp/models
ENV MODEL_BASE_URL=https://www.marmoai.cn/models/fastsam

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Keep model weights out of the image. The runtime downloads them into
# MODEL_CACHE_DIR when first needed; only the deployable backend modules are
# copied into the image.
COPY main.py api.py candidate_selection.py diagnostics.py mask_ops.py matte_ops.py sam_runtime.py segmentation_core.py segmentation_policy.py segmentation_primitives.py /app/

ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
