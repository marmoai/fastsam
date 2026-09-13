"""SAM model lifecycle and inference transport.

This module owns model download/cache, B/L predictor switching, CUDA lifecycle
diagnostics, and shared inference sizing. It deliberately contains no HTTP or
semantic candidate-selection policy.
"""

import gc
import os
import threading
from urllib.request import urlopen

# Keep direct `uvicorn api:app` launches equivalent to `python main.py`.
# These must be set before importing torch/Ultralytics.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import FastSAM, SAM

# Runtime configuration belongs to the model lifecycle module. Keeping these
# values here avoids the old main.py globals leaking into request/policy code
# and makes importing the API sufficient for both server and CLI execution.
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://www.marmoai.cn/models/fastsam").rstrip("/")
MODEL_DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "300"))
OSS_PUBLIC_BASE_URL = os.getenv("OSS_PUBLIC_BASE_URL", "https://www.marmoai.cn").rstrip("/")
FASTSAM_MODEL_PATH = os.getenv("FASTSAM_MODEL_PATH", "FastSAM-x.pt")
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", "sam_b.pt")
SAM_B_MODEL_PATH = os.getenv("SAM_B_MODEL_PATH", SAM_MODEL_PATH)
SAM_L_MODEL_PATH = os.getenv("SAM_L_MODEL_PATH", "sam_l.pt")
FASTSAM_MODEL_URL = os.getenv("FASTSAM_MODEL_URL", f"{MODEL_BASE_URL}/FastSAM-x.pt")
SAM_MODEL_URL = os.getenv("SAM_MODEL_URL", f"{MODEL_BASE_URL}/sam_b.pt")
SAM_B_MODEL_URL = os.getenv("SAM_B_MODEL_URL", SAM_MODEL_URL)
SAM_L_MODEL_URL = os.getenv("SAM_L_MODEL_URL", f"{MODEL_BASE_URL}/sam_l.pt")

fastsam_model = None
sam_models = {}
sam_runtime_lock = threading.RLock()


def resolve_model_url(model_url):
    if not model_url:
        return model_url
    if not model_url.startswith("oss://"):
        return model_url

    bucket_and_key = model_url[len("oss://"):]
    _, _, object_key = bucket_and_key.partition("/")
    if not object_key:
        raise ValueError(f"Invalid OSS model URL: {model_url}")
    return f"{OSS_PUBLIC_BASE_URL}/{object_key.lstrip('/')}"


def ensure_model_file(configured_path, fallback_url, filename, label):
    if configured_path and os.path.isfile(configured_path):
        return configured_path

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    cached_path = os.path.join(MODEL_CACHE_DIR, filename)
    if os.path.isfile(cached_path):
        print(f"Using cached {label} model: {cached_path}")
        return cached_path

    model_url = resolve_model_url(fallback_url)
    if not model_url:
        raise FileNotFoundError(
            f"{label} model is missing. Set {label.upper()}_MODEL_PATH or {label.upper()}_MODEL_URL."
        )

    temp_path = f"{cached_path}.part"
    print(f"Downloading {label} model from {model_url} to {cached_path}")
    try:
        with urlopen(model_url, timeout=MODEL_DOWNLOAD_TIMEOUT) as response, open(temp_path, "wb") as file_obj:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
        os.replace(temp_path, cached_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    size_mb = os.path.getsize(cached_path) / (1024 * 1024)
    print(f"Cached {label} model at {cached_path} ({size_mb:.2f} MB)")
    return cached_path


def get_fastsam_model():
    global fastsam_model
    if fastsam_model is not None:
        return fastsam_model

    model_path = ensure_model_file(FASTSAM_MODEL_PATH, FASTSAM_MODEL_URL, "FastSAM-x.pt", "fastsam")
    print(f"Loading FastSAM model: {model_path}")
    fastsam_model = FastSAM(model_path)
    print("FastSAM model loaded.")
    return fastsam_model


def get_sam_model(model_variant="b"):
    variant = "l" if str(model_variant).lower() == "l" else "b"
    log_cuda_memory(f"request_{variant}_model", variant)

    # Keep only one high-precision SAM model resident. B and L cannot safely
    # coexist on the deployment GPU, especially at 1536px hard-edge inference.
    for loaded_variant in list(sam_models):
        if loaded_variant != variant:
            release_sam_model(loaded_variant, reason=f"switch_to_{variant}")

    if variant in sam_models:
        return sam_models[variant]

    if variant == "l":
        configured_path = SAM_L_MODEL_PATH
        model_url = SAM_L_MODEL_URL
        filename = "sam_l.pt"
    else:
        configured_path = SAM_B_MODEL_PATH
        model_url = SAM_B_MODEL_URL
        filename = "sam_b.pt"
    model_path = ensure_model_file(configured_path, model_url, filename, f"sam_{variant}")
    print(f"Loading high precision SAM-{variant.upper()} model: {model_path}")
    log_cuda_memory(f"before_{variant}_load", variant)
    sam_models[variant] = SAM(model_path)
    print(f"High precision SAM-{variant.upper()} model loaded.")
    log_cuda_memory(f"after_{variant}_load", variant)
    return sam_models[variant]


def release_sam_model(model_variant, reason="manual_release"):
    """Release one SAM variant and return its CUDA allocator memory to the pool."""
    variant = "l" if str(model_variant).lower() == "l" else "b"
    sam = sam_models.pop(variant, None)
    if sam is None:
        return False

    log_cuda_memory(f"before_release:{reason}", variant)
    predictor = getattr(sam, "predictor", None)
    # Move both Ultralytics' model wrapper and predictor model off the GPU
    # before dropping references. This is more reliable than empty_cache()
    # alone when a large SAM-L forward pass has just completed or failed.
    for model_object in (
        getattr(sam, "model", None),
        getattr(predictor, "model", None),
    ):
        try:
            if model_object is not None and hasattr(model_object, "to"):
                model_object.to("cpu")
        except Exception as error:
            print(f"SAM-{variant.upper()} CPU release warning: {error}")
    # The loop variable and the predictor attribute can otherwise keep the
    # last CUDA model object alive until after empty_cache().
    try:
        sam.predictor = None
    except Exception:
        pass
    model_object = None
    predictor = None
    sam = None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        print(
            f"Released high precision SAM-{variant.upper()} model: "
            f"reason={reason} cudaAllocated={allocated_mb:.0f}MB "
            f"cudaReserved={reserved_mb:.0f}MB"
        )
        log_cuda_memory(f"after_release:{reason}", variant)
    else:
        print(f"Released high precision SAM-{variant.upper()} model: reason={reason}")
    return True


def get_sam_predictor(model_variant="b"):
    variant = "l" if str(model_variant).lower() == "l" else "b"
    sam = get_sam_model(variant)
    predictor = getattr(sam, "predictor", None)
    if predictor is not None:
        return predictor

    # Ultralytics SAM supports multimask_output in the predictor inference path,
    # but some package versions reject that kwarg at the higher Model.predict layer.
    predictor = sam._smart_load("predictor")(
        overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": 1024,
            "batch": 1,
            "save": False,
            "verbose": False
        },
        _callbacks=sam.callbacks
    )
    predictor.setup_model(model=sam.model, verbose=False)
    sam.predictor = predictor
    return predictor


def run_sam_bbox_inference(
    img,
    target_bbox,
    multimask_output=True,
    imgsz=1024,
    points=None,
    labels=None,
    masks=None,
    model_variant="b"
):
    with sam_runtime_lock:
        predictor = get_sam_predictor(model_variant)
        previous_imgsz = getattr(predictor.args, "imgsz", 1024)
        predictor.args.imgsz = imgsz
        if hasattr(predictor.model, "set_imgsz"):
            predictor.model.set_imgsz((imgsz, imgsz))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        log_cuda_memory("inference_before", model_variant)
        try:
            return predictor(
                source=img,
                bboxes=[target_bbox],
                points=points,
                labels=labels,
                masks=masks,
                multimask_output=multimask_output
            )
        finally:
            log_cuda_memory("inference_after", model_variant)
            if hasattr(predictor.model, "set_imgsz"):
                predictor.model.set_imgsz((previous_imgsz, previous_imgsz))
            predictor.args.imgsz = previous_imgsz


def run_sam_mask_refine_inference(
    img,
    points=None,
    labels=None,
    masks=None,
    imgsz=1024,
    multimask_output=True,
    model_variant="b"
):
    with sam_runtime_lock:
        predictor = get_sam_predictor(model_variant)
        previous_imgsz = getattr(predictor.args, "imgsz", 1024)
        previous_direct_mask_mode = getattr(predictor.model, "use_mask_input_as_output_without_sam", False)
        predictor.args.imgsz = imgsz
        if hasattr(predictor.model, "set_imgsz"):
            predictor.model.set_imgsz((imgsz, imgsz))
        predictor.model.use_mask_input_as_output_without_sam = False
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        log_cuda_memory("mask_refine_before", model_variant)
        try:
            return predictor(
                source=img,
                points=points,
                labels=labels,
                masks=masks,
                multimask_output=multimask_output
            )
        finally:
            log_cuda_memory("mask_refine_after", model_variant)
            predictor.model.use_mask_input_as_output_without_sam = previous_direct_mask_mode
            if hasattr(predictor.model, "set_imgsz"):
                predictor.model.set_imgsz((previous_imgsz, previous_imgsz))
            predictor.args.imgsz = previous_imgsz


def run_sam_auto_inference(img, imgsz=1024, model_variant="b"):
    with sam_runtime_lock:
        predictor = get_sam_predictor(model_variant)
        previous_imgsz = getattr(predictor.args, "imgsz", 1024)
        predictor.args.imgsz = imgsz
        if hasattr(predictor.model, "set_imgsz"):
            predictor.model.set_imgsz((imgsz, imgsz))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        log_cuda_memory("auto_before", model_variant)
        try:
            return predictor(source=img)
        finally:
            log_cuda_memory("auto_after", model_variant)
            if hasattr(predictor.model, "set_imgsz"):
                predictor.model.set_imgsz((previous_imgsz, previous_imgsz))
            predictor.args.imgsz = previous_imgsz



BBOX_EXPAND_RATIO = 0.18
SOFT_EDGE_SAM_IMGSZ = 1536
HARD_EDGE_SAM_IMGSZ = 1536
HARD_EDGE_LOCAL_SCALE = 2.0
SAM_MEMORY_LOGGING = str(os.getenv("SAM_MEMORY_LOGGING", "1")).lower() not in {"0", "false", "no"}
SAM_L_LARGE_IMAGE_MAX_SIDE = int(os.getenv("SAM_L_LARGE_IMAGE_MAX_SIDE", "2400"))
SAM_L_LARGE_IMAGE_IMGSZ = int(os.getenv("SAM_L_LARGE_IMAGE_IMGSZ", "1280"))
SAM_L_OOM_RETRY_IMGSZ = int(os.getenv("SAM_L_OOM_RETRY_IMGSZ", "1024"))
LOCAL_UPSCALE_ENABLED = str(os.getenv("SAM_LOCAL_UPSCALE_ENABLED", "1")).lower() not in {"0", "false", "no"}
LOCAL_UPSCALE_SAM_IMGSZ = int(os.getenv("SAM_LOCAL_UPSCALE_SAM_IMGSZ", "1280"))
LOCAL_UPSCALE_MAX_BBOX_SIDE = int(os.getenv("SAM_LOCAL_UPSCALE_MAX_BBOX_SIDE", "960"))
LOCAL_UPSCALE_MAX_SOURCE_SIDE = int(os.getenv("SAM_LOCAL_UPSCALE_MAX_SOURCE_SIDE", "1800"))
# Keep local crops isolated to the lighting route. Furniture uses the
# B/L arbitration path; injecting a local candidate there can make routing
# accept B while candidate selection still keeps the original B mask.
LOCAL_UPSCALE_STRATEGIES = {"lighting"}
HARD_EDGE_STRATEGIES = {
    "furniture",
    "lighting",
    "wall_art",
    "decor_arrangement",
    "decor_atomic"
}
LOCAL_REFINE_EXPAND_RATIO = 0.22
LOCAL_REFINE_MAX_SIDE = 1280
MASK_COMPONENT_MIN_PIXELS = 36
MASK_KEEP_COMPONENT_MAX_GAP = 36
MASK_HOLE_MIN_AREA = 24
GRABCUT_ITER_COUNT = 2
HARD_EDGE_GRABCUT_ITER_COUNT = 5
MIN_BBOX_OVERLAP_RATIO = 0.08
MIN_MASK_AREA_RATIO_IN_BBOX = 0.01
MAX_MASK_AREA_RATIO_IN_BBOX = 1.15
MIN_MASK_INSIDE_TARGET_RATIO = 0.25
MAX_TARGET_FILL_RATIO = 0.82
MAX_MERGED_TARGET_FILL_RATIO = 0.68
MAX_BACKGROUND_RISK_FILL_RATIO = 0.78
MAX_ATTACHMENT_DISTANCE_PX = 22
MAX_EXCLUDE_MASK_RATIO = 0.45
MIN_RUNTIME_ACCEPT_SCORE = 0.45
MIN_RUNTIME_ACCEPT_FILL_RATIO = 0.06
MAX_RUNTIME_ACCEPT_FILL_RATIO = 0.76
MIN_RUNTIME_ACCEPT_PRIMARY_SCORE = -0.05
MIN_RECTANGULARITY = 0.48
THIN_VERTICAL_ASPECT_RATIO = 0.38
THIN_VERTICAL_MIN_HEIGHT_RATIO = 0.28
BLOCKLIKE_MIN_ASPECT_RATIO = 0.45
BLOCKLIKE_MAX_ASPECT_RATIO = 2.4
BLOCKLIKE_MIN_WIDTH_RATIO = 0.12
BLOCKLIKE_MIN_HEIGHT_RATIO = 0.12
BOTTOM_BAND_TOLERANCE = 0.12
TABLE_SUPPORT_MAX_ASPECT_RATIO = 0.95
TABLE_SUPPORT_MAX_WIDTH_RATIO = 0.42
TABLE_SUPPORT_MIN_HEIGHT_RATIO = 0.22


def cuda_memory_snapshot():
    """Return a small, comparable CUDA memory snapshot for lifecycle logs."""
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "available": True,
            "allocatedMB": round(torch.cuda.memory_allocated() / (1024 * 1024)),
            "reservedMB": round(torch.cuda.memory_reserved() / (1024 * 1024)),
            "freeMB": round(free_bytes / (1024 * 1024)),
            "totalMB": round(total_bytes / (1024 * 1024)),
            "peakAllocatedMB": round(torch.cuda.max_memory_allocated() / (1024 * 1024))
        }
    except Exception as error:
        return {"available": True, "error": str(error)}


def log_cuda_memory(stage, model_variant=None):
    if not SAM_MEMORY_LOGGING:
        return
    snapshot = cuda_memory_snapshot()
    variant = f" model={str(model_variant).upper()}" if model_variant else ""
    print(f"SAM CUDA memory stage={stage}{variant}: {snapshot}")


def is_cuda_oom(error):
    message = str(error).lower()
    return "cuda out of memory" in message or "out of memory" in message


def choose_sam_imgsz(img, strategy_type=None, model_variant="b", policy=None):
    """Choose internal inference size without changing the source/output resolution."""
    if policy is not None:
        base_imgsz = int(policy.get("samImgSize", 1024))
    elif strategy_type == "soft_edge":
        base_imgsz = SOFT_EDGE_SAM_IMGSZ
    elif strategy_type in HARD_EDGE_STRATEGIES:
        base_imgsz = HARD_EDGE_SAM_IMGSZ
    else:
        base_imgsz = 1024

    if str(model_variant).lower() != "l":
        return base_imgsz
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return base_imgsz
    if max(img.shape[:2]) >= SAM_L_LARGE_IMAGE_MAX_SIDE:
        return min(base_imgsz, SAM_L_LARGE_IMAGE_IMGSZ)
    return base_imgsz
