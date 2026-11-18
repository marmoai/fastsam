FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV MODEL_CACHE_DIR=/tmp/models
ENV MODEL_BASE_URL=https://www.marmoai.cn/models/fastsam
ENV SAM_HQ_L_MODEL_PATH=/tmp/models/sam_hq_vit_l.pth
ENV SAM_HQ_L_MODEL_URL=https://www.marmoai.cn/models/fastsam/sam_hq_vit_l.pth

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

COPY requirements-v5.txt /app/requirements-v5.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements-v5.txt

# V5 runtime dependency closure only. Model weights remain outside the image.
COPY main_v5.py api_v5.py segmentation_core_v5.py segmentation_policy_v5.py sam_runtime_v5.py sam_diagnostics.py /app/
COPY alpha_guard_v5.py boundary_audit_v5.py boundary_control_v5.py boundary_refine_v5.py candidate_pool_v5.py component_recovery_v5.py fragment_ownership_v5.py hq_candidate_review_v5.py independent_neighbour_v5.py joint_ownership_v5.py local_completeness_v5.py matting_v5.py ownership_v5.py query_candidates_v5.py soft_backdrop_alpha.py transparency_contract_v5.py /app/
COPY vendor_hq_sam/LICENSE /app/vendor_hq_sam/LICENSE
COPY vendor_hq_sam/segment_anything /app/vendor_hq_sam/segment_anything

ENV PORT=8000
ENV SAMV5_PORT=8000

EXPOSE 8000

CMD ["python", "main_v5.py"]
