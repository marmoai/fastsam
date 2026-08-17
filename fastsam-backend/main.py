from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import FastSAM, SAM
import cv2
import numpy as np
import base64
import time
import io
import os
from PIL import Image
from urllib.request import urlopen

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

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
    sam_models[variant] = SAM(model_path)
    print(f"High precision SAM-{variant.upper()} model loaded.")
    return sam_models[variant]


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
    predictor = get_sam_predictor(model_variant)
    previous_imgsz = getattr(predictor.args, "imgsz", 1024)
    predictor.args.imgsz = imgsz
    if hasattr(predictor.model, "set_imgsz"):
        predictor.model.set_imgsz((imgsz, imgsz))
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
        if hasattr(predictor.model, "set_imgsz"):
            predictor.model.set_imgsz((previous_imgsz, previous_imgsz))
        predictor.args.imgsz = previous_imgsz


@app.get("/healthz")
async def healthz():
    return {"ok": True}


def run_sam_mask_refine_inference(
    img,
    points=None,
    labels=None,
    masks=None,
    imgsz=1024,
    multimask_output=True,
    model_variant="b"
):
    predictor = get_sam_predictor(model_variant)
    previous_imgsz = getattr(predictor.args, "imgsz", 1024)
    previous_direct_mask_mode = getattr(predictor.model, "use_mask_input_as_output_without_sam", False)
    predictor.args.imgsz = imgsz
    if hasattr(predictor.model, "set_imgsz"):
        predictor.model.set_imgsz((imgsz, imgsz))
    predictor.model.use_mask_input_as_output_without_sam = False
    try:
        return predictor(
            source=img,
            points=points,
            labels=labels,
            masks=masks,
            multimask_output=multimask_output
        )
    finally:
        predictor.model.use_mask_input_as_output_without_sam = previous_direct_mask_mode
        if hasattr(predictor.model, "set_imgsz"):
            predictor.model.set_imgsz((previous_imgsz, previous_imgsz))
        predictor.args.imgsz = previous_imgsz


def run_sam_auto_inference(
    img,
    imgsz=1024,
    model_variant="b"
):
    predictor = get_sam_predictor(model_variant)
    previous_imgsz = getattr(predictor.args, "imgsz", 1024)
    predictor.args.imgsz = imgsz
    if hasattr(predictor.model, "set_imgsz"):
        predictor.model.set_imgsz((imgsz, imgsz))
    try:
        return predictor(
            source=img
        )
    finally:
        if hasattr(predictor.model, "set_imgsz"):
            predictor.model.set_imgsz((previous_imgsz, previous_imgsz))
        predictor.args.imgsz = previous_imgsz

BBOX_EXPAND_RATIO = 0.18
SOFT_EDGE_SAM_IMGSZ = 1536
HARD_EDGE_SAM_IMGSZ = 1536
HARD_EDGE_LOCAL_SCALE = 2.0
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

def get_layer_strategy(layer_meta):
    extraction_profile = str(layer_meta.get("extractionProfile", "")).lower()
    semantic_type = str(layer_meta.get("semanticType", "")).lower()
    design_role = str(layer_meta.get("designRole", "")).lower()
    profile_text = " ".join([
        str(layer_meta.get("name", "")),
        str(layer_meta.get("semanticType", "")),
        str(layer_meta.get("designRole", "")),
        str(layer_meta.get("category", "")),
        str(layer_meta.get("runtimeType", ""))
    ]).lower()

    # layout_embedded_product is a broad upstream profile used for poster
    # subjects, but it can also be attached to a physical surface by the
    # semantic layer pass. Resolve explicit hard-surface signals first so a
    # table/console cannot be routed through the food-specific pipeline.
    explicit_food_layer = (
        semantic_type in {"product_food", "product_drink"} or
        design_role == "product_image"
    )
    embedded_table_signal = any(token in profile_text for token in [
        "table", "desk", "console", "sideboard", "cabinet", "counter",
        "茶几", "桌面", "台面", "边几", "矮几", "桌", "柜"
    ])
    if (
        extraction_profile == "layout_embedded_product" and
        embedded_table_signal and
        not explicit_food_layer
    ):
        return {
            "type": "table",
            "max_fill": 0.66,
            "max_merged_fill": 0.72,
            "max_attachment_distance": 7,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    if extraction_profile in ["vector_layout_element", "text_layer", "background_plate"]:
        return {
            "type": "flat_shape",
            "max_fill": 0.92,
            "max_merged_fill": 0.82,
            "max_attachment_distance": 6,
            "allow_attachments": False,
            "prefer_rectangular": extraction_profile != "text_layer",
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    if extraction_profile == "layout_embedded_product":
        return {
            "type": "food_product",
            "max_fill": 0.94,
            "max_merged_fill": 0.95,
            "max_attachment_distance": 16,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 5,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "multi_part_hard_product":
        return {
            "type": "hard_product",
            "max_fill": 0.88,
            "max_merged_fill": 0.86,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 6,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "compound_object":
        return {
            "type": "decor_arrangement",
            "max_fill": 0.92,
            "max_merged_fill": 0.90,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 8,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "multi_part_hard_object":
        if any(token in profile_text for token in [
            "lamp", "chandelier", "pendant", "lighting", "light fixture",
            "吊灯", "灯具", "灯饰", "照明", "吸顶灯", "壁灯"
        ]):
            return {
                "type": "lighting",
                "max_fill": 0.62,
                "max_merged_fill": 0.54,
                "max_attachment_distance": 6,
                "allow_attachments": True,
                "prefer_rectangular": False,
                "max_masks": 3,
                "require_overlap_for_attachments": True
            }
        if any(token in profile_text for token in [
            "table", "desk", "console", "sideboard", "cabinet", "台", "桌", "柜", "玄关"
        ]):
            return {
                "type": "table",
                "max_fill": 0.66,
                "max_merged_fill": 0.72,
                "max_attachment_distance": 7,
                "allow_attachments": True,
                "prefer_rectangular": False,
                "max_masks": 3,
                "require_overlap_for_attachments": True
            }
        return {
            "type": "furniture",
            "max_fill": 0.70,
            "max_merged_fill": 0.58,
            "max_attachment_distance": 7,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    # An explicit hard-object profile takes precedence over descriptive words
    # such as "glass" or "transparent" in the layer name. Tables and cabinets
    # must not be routed through the soft-edge prompt/matte chain.
    if any(token in profile_text for token in [
        "feather", "plume", "hair", "fur", "smoke", "cloud", "fog",
        "sheer", "curtain", "tulle", "transparent fabric", "glass",
        "羽毛", "毛发", "绒毛", "烟", "云", "雾", "窗纱", "纱", "薄纱",
        "玻璃", "透明"
    ]):
        return {
            "type": "soft_edge",
            "max_fill": 0.96,
            "max_merged_fill": 0.96,
            "max_attachment_distance": 24,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 8,
            "require_overlap_for_attachments": False
        }

    text = " ".join([
        str(layer_meta.get("name", "")),
        str(layer_meta.get("semanticType", "")),
        str(layer_meta.get("category", "")),
        str(layer_meta.get("runtimeType", ""))
    ]).lower()

    if any(token in text for token in [
        "price_badge", "price", "badge", "价格", "价签", "徽章", "$"
    ]):
        return {
            "type": "flat_shape",
            "max_fill": 0.86,
            "max_merged_fill": 0.72,
            "max_attachment_distance": 6,
            "allow_attachments": False,
            "prefer_rectangular": False,
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in [
        "shape_panel", "ad_background", "cta_button", "logo_mark", "element_text",
        "text_node", "panel", "label", "card", "shape", "button", "logo",
        "面板", "底板", "色块", "背景框", "文字背景", "标签底板", "文字", "文本", "标志"
    ]):
        return {
            "type": "flat_shape",
            "max_fill": 0.92,
            "max_merged_fill": 0.82,
            "max_attachment_distance": 6,
            "allow_attachments": False,
            "prefer_rectangular": True,
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in ["wall art", "painting", "artwork", "picture", "poster", "挂画", "画", "装饰画"]):
        return {
            "type": "wall_art",
            # A painting can legitimately occupy almost its entire semantic
            # bbox. High fill is not background evidence for a bounded plane.
            "max_fill": 0.98,
            "max_merged_fill": 0.98,
            "max_attachment_distance": 8,
            "allow_attachments": False,
            "prefer_rectangular": True,
            "max_masks": 1,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in [
        "lamp", "chandelier", "pendant", "lighting", "light fixture",
        "吊灯", "灯具", "灯饰", "照明", "吸顶灯", "壁灯"
    ]):
        return {
            "type": "lighting",
            "max_fill": 0.62,
            "max_merged_fill": 0.54,
            "max_attachment_distance": 6,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in [
        "feather", "plume", "hair", "fur", "smoke", "cloud", "fog",
        "sheer", "curtain", "tulle", "transparent fabric", "glass",
        "羽毛", "毛发", "绒毛", "烟", "云", "雾", "窗纱", "纱", "薄纱",
        "玻璃", "透明"
    ]):
        return {
            "type": "soft_edge",
            "max_fill": 0.96,
            "max_merged_fill": 0.96,
            "max_attachment_distance": 24,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 8,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "food", "dish", "meal", "plate", "rice", "fried rice", "pork", "roasted",
        "burger", "pizza", "noodle", "salad", "cola", "tea", "coffee", "choco",
        "drink", "beverage", "cup", "product_food", "product_drink",
        "食物", "食品", "菜品", "餐盘", "炒饭", "米饭", "猪肉", "烤肉", "饮料",
        "可乐", "茶", "咖啡", "热巧", "杯"
    ]):
        return {
            "type": "food_product",
            "max_fill": 0.94,
            "max_merged_fill": 0.95,
            "max_attachment_distance": 16,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 5,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "product_image", "product_packaging", "product", "earphone", "earphones",
        "earbud", "earbuds", "headphone", "headphones", "case", "device",
        "electronics", "gadget", "packaging", "商品图", "产品图", "商品",
        "产品", "耳机", "蓝牙耳机", "充电盒", "电子产品", "包装"
    ]):
        return {
            "type": "hard_product",
            "max_fill": 0.88,
            "max_merged_fill": 0.86,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 6,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "vases_flowers", "vase flowers", "vase and flowers", "flowers in vase",
        "flower arrangement", "bouquet", "plant arrangement", "potted plant",
        "pot plant", "arrangement", "花瓶花艺", "花艺", "插花", "花束", "盆栽", "植物组合"
    ]):
        return {
            "type": "decor_arrangement",
            "max_fill": 0.82,
            "max_merged_fill": 0.78,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 7,
            "require_overlap_for_attachments": False
        }

    if any(token in text for token in [
        "vase", "bowl", "sculpture", "ornament", "column", "stacked", "cylinder",
        "decor_vase", "decor_bowl", "decor_sculpture",
        "花瓶", "碗", "摆件", "雕塑", "柱状", "立柱", "叠柱", "装饰柱", "装饰碗"
    ]):
        return {
            "type": "decor_atomic",
            "max_fill": 0.74,
            "max_merged_fill": 0.64,
            "max_attachment_distance": 8,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 3,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in ["table", "desk", "coffee table", "茶几", "桌"]):
        return {
            "type": "table",
            "max_fill": 0.72,
            "max_merged_fill": 0.72,
            "max_attachment_distance": 8,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 4,
            "require_overlap_for_attachments": True
        }

    if any(token in text for token in ["chair", "sofa", "seat", "stool", "沙发", "椅", "凳"]):
        return {
            "type": "furniture",
            "max_fill": 0.78,
            "max_merged_fill": 0.68,
            "max_attachment_distance": 10,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 4,
            "require_overlap_for_attachments": True
        }

    return {
        "type": "default",
        "max_fill": MAX_TARGET_FILL_RATIO,
        "max_merged_fill": MAX_MERGED_TARGET_FILL_RATIO,
        "max_attachment_distance": MAX_ATTACHMENT_DISTANCE_PX,
        "allow_attachments": True,
        "prefer_rectangular": False,
        "max_masks": 4,
        "require_overlap_for_attachments": False
    }

def is_drink_product_layer(layer_meta):
    text = " ".join([
        str(layer_meta.get("name", "")),
        str(layer_meta.get("semanticType", "")),
        str(layer_meta.get("category", "")),
        str(layer_meta.get("runtimeType", ""))
    ]).lower()
    return any(token in text for token in [
        "product_drink", "drink", "beverage", "cola", "tea", "coffee", "choco",
        "可乐", "茶", "咖啡", "热巧", "饮料"
    ])

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def expand_bbox(x1, y1, x2, y2, img_w, img_h, ratio=BBOX_EXPAND_RATIO):
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * ratio)
    pad_y = int(box_h * ratio)
    return [
        clamp(x1 - pad_x, 0, img_w - 1),
        clamp(y1 - pad_y, 0, img_h - 1),
        clamp(x2 + pad_x, 1, img_w),
        clamp(y2 + pad_y, 1, img_h),
    ]

def dilate_and_feather_mask(mask):
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    mask_u8 = cv2.GaussianBlur(mask_u8, (3, 3), 0)
    return mask_u8


def compute_cleanup_kernel(target_bbox):
    x1, y1, x2, y2 = target_bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    base = int(round(min(box_w, box_h) * 0.018))
    size = clamp(base, 1, 5)
    return max(1, size)


def fill_small_holes(mask_binary, min_hole_area=MASK_HOLE_MIN_AREA):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    inverted = 1 - mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if num_labels <= 1:
        return mask_binary

    h, w = mask_binary.shape
    cleaned = mask_u8.copy()
    for label in range(1, num_labels):
        x, y, cw, ch, area = stats[label]
        touches_border = x == 0 or y == 0 or (x + cw) >= w or (y + ch) >= h
        if touches_border:
            continue
        if area <= min_hole_area:
            cleaned[labels == label] = 1

    return cleaned.astype(bool)


def cleanup_furniture_mask(mask, target_bbox):
    """Repair small furniture holes while retaining legs and nearby supports.

    Furniture masks commonly contain a small enclosed hole or disconnected
    thin support.  Generic morphology cannot distinguish those from a real
    opening or a false fragment, so use conservative geometry around the
    largest body component instead.
    """
    binary = np.asarray(mask > 0.5, dtype=bool)
    if not np.any(binary):
        return mask, {
            "holesFilled": 0,
            "holePixels": 0,
            "componentsRemoved": 0,
            "componentPixelsRemoved": 0,
            "supportsPreserved": 0,
        }

    height, width = binary.shape
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = max(0, min(width - 1, tx1))
    ty1 = max(0, min(height - 1, ty1))
    tx2 = max(tx1 + 1, min(width, tx2))
    ty2 = max(ty1 + 1, min(height, ty2))
    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))

    # Fill only enclosed holes that are safely inside the semantic bbox.  The
    # area and shape limits leave large intentional openings (for example a
    # chair's under-frame) untouched.
    inverted = (~binary).astype(np.uint8)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        inverted,
        connectivity=8
    )
    primary_area = max(1, int(np.count_nonzero(binary)))
    # At high resolution a large false hole can still be a background pocket
    # between adjacent cushions.  Keep this pass limited to genuinely small
    # raster gaps; larger omissions need SAM evidence, not blanket filling.
    max_hole_area = max(
        32,
        min(
            384,
            int(target_area * 0.0025),
            int(primary_area * 0.008)
        )
    )
    filled_holes = 0
    filled_hole_pixels = 0
    repaired = binary.copy()
    for label in range(1, hole_count):
        hx, hy, hw, hh, area = [int(value) for value in hole_stats[label]]
        if area > max_hole_area:
            continue
        hole = hole_labels == label
        hole_ys, hole_xs = np.where(hole)
        if hole_xs.size == 0:
            continue
        if (
            int(hole_xs.min()) <= tx1 or int(hole_ys.min()) <= ty1 or
            int(hole_xs.max()) >= tx2 - 1 or int(hole_ys.max()) >= ty2 - 1
        ):
            continue
        if hw >= max(8, int((tx2 - tx1) * 0.36)) and hh >= max(8, int((ty2 - ty1) * 0.36)):
            continue
        repaired[hole] = True
        filled_holes += 1
        filled_hole_pixels += int(area)

    # Identify the main body after hole repair.  Components close to it are
    # retained even when small; this is what protects a detached-looking leg
    # or narrow arm caused by rasterization.
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        repaired.astype(np.uint8),
        connectivity=8
    )
    if component_count <= 2:
        return repaired.astype(np.float32), {
            "holesFilled": filled_holes,
            "holePixels": filled_hole_pixels,
            "componentsRemoved": 0,
            "componentPixelsRemoved": 0,
            "supportsPreserved": 0,
        }

    component_rows = []
    for label in range(1, component_count):
        x, y, cw, ch, area = [int(value) for value in component_stats[label]]
        if area <= 0:
            continue
        component_rows.append((area, label, [x, y, x + cw, y + ch]))
    if not component_rows:
        return repaired.astype(np.float32), {
            "holesFilled": filled_holes,
            "holePixels": filled_hole_pixels,
            "componentsRemoved": 0,
            "componentPixelsRemoved": 0,
            "supportsPreserved": 0,
        }

    component_rows.sort(reverse=True)
    primary_label = component_rows[0][1]
    primary_component = component_labels == primary_label
    min_side = max(1, min(tx2 - tx1, ty2 - ty1))
    proximity_gap = max(8, min(48, int(round(min_side * 0.035))))
    proximity_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (proximity_gap * 2 + 1, proximity_gap * 2 + 1)
    )
    near_primary = cv2.dilate(
        primary_component.astype(np.uint8),
        proximity_kernel,
        iterations=1
    ) > 0
    small_component_limit = max(
        64,
        min(6000, int(target_area * 0.006))
    )
    cleaned = repaired.copy()
    removed_components = 0
    removed_pixels = 0
    preserved_supports = 0

    for area, label, component_bbox in component_rows[1:]:
        component = component_labels == label
        overlaps_primary_neighborhood = bool(np.any(component & near_primary))
        shape = compute_shape_features(component_bbox, target_bbox, area)
        inside_target = int(np.count_nonzero(component[ty1:ty2, tx1:tx2])) / max(1, area)
        touches_bottom = component_bbox[3] >= ty2 - max(4, int(round((ty2 - ty1) * 0.04)))
        support_like = (
            shape["isThinVertical"] or
            shape["isTableSupport"] or
            (shape["relativeHeight"] >= 0.45 and shape["relativeWidth"] <= 0.42)
        )

        # Only remove genuinely small, isolated noise.  A support-like piece
        # remains protected when it is near the body or reaches the bbox base.
        preserve_support = support_like and (
            overlaps_primary_neighborhood or
            (inside_target >= 0.12 and touches_bottom)
        )
        if preserve_support:
            preserved_supports += 1
            continue
        if area <= small_component_limit and not overlaps_primary_neighborhood:
            cleaned[component] = False
            removed_components += 1
            removed_pixels += int(area)

    return cleaned.astype(np.float32), {
        "holesFilled": filled_holes,
        "holePixels": filled_hole_pixels,
        "componentsRemoved": removed_components,
        "componentPixelsRemoved": removed_pixels,
        "supportsPreserved": preserved_supports,
    }


def recover_protected_furniture_gaps(img, mask, target_bbox):
    """Restore only enclosed furniture gaps that match nearby subject pixels.

    The color test is deliberately conservative.  It prevents a large white
    floor/background pocket from being filled merely because morphology closed
    a narrow seam in the SAM mask.
    """
    binary = np.asarray(mask > 0.5, dtype=bool)
    if not np.any(binary):
        return mask.astype(np.float32), np.zeros_like(binary, dtype=bool), {
            "gapsFilled": 0,
            "gapPixels": 0,
            "colorRejected": 0,
        }

    height, width = binary.shape
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = max(0, min(width - 1, tx1))
    ty1 = max(0, min(height - 1, ty1))
    tx2 = max(tx1 + 1, min(width, tx2))
    ty2 = max(ty1 + 1, min(height, ty2))
    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    subject_area = max(1, int(np.count_nonzero(binary)))
    min_side = max(1, min(tx2 - tx1, ty2 - ty1))

    base_seam_radius = max(2, min(8, int(round(min_side * 0.009))))
    max_gap_area = max(
        96,
        min(24000, int(target_area * 0.12), int(subject_area * 0.20))
    )
    # A texture/lighting break can leave a wider connection to the outside
    # than the first raster pass can close. Try a few bounded scales. The
    # largest scale is still small relative to the bbox and is followed by
    # the same ring/color checks below.
    seam_radii = sorted(set([
        base_seam_radius,
        min(14, max(8, base_seam_radius)),
        min(22, max(14, base_seam_radius))
    ]))
    proposed = np.zeros_like(binary, dtype=bool)
    selected_seam_radius = base_seam_radius
    for seam_radius in seam_radii:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (seam_radius * 2 + 1, seam_radius * 2 + 1)
        )
        closed = cv2.morphologyEx(
            binary.astype(np.uint8),
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1
        ) > 0
        inverse_count, inverse_labels, inverse_stats, _ = cv2.connectedComponentsWithStats(
            (~closed).astype(np.uint8),
            connectivity=8
        )
        margin = seam_radius + 1
        scale_proposed = np.zeros_like(binary, dtype=bool)
        for label in range(1, inverse_count):
            x, y, comp_w, comp_h, area = [int(value) for value in inverse_stats[label]]
            if area < 24 or area > max_gap_area:
                continue
            inverse_component = inverse_labels == label
            ys, xs = np.where(inverse_component)
            if xs.size == 0:
                continue
            if (
                int(xs.min()) <= tx1 + margin or int(ys.min()) <= ty1 + margin or
                int(xs.max()) >= tx2 - margin - 1 or int(ys.max()) >= ty2 - margin - 1
            ):
                continue
            scale_proposed |= inverse_component
        if np.any(scale_proposed):
            proposed |= scale_proposed
            selected_seam_radius = seam_radius
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        proposed.astype(np.uint8),
        connectivity=8
    )
    if component_count <= 1:
        return mask.astype(np.float32), proposed, {
            "gapsFilled": 0,
            "gapPixels": 0,
            "colorRejected": 0,
        }

    ring_radius = max(2, min(8, selected_seam_radius + 1))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ring_radius * 2 + 1, ring_radius * 2 + 1)
    )
    repaired = binary.copy()
    repair_pixels = np.zeros_like(binary, dtype=bool)
    gap_count = 0
    gap_pixels = 0
    color_rejected = 0
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_border = np.zeros(binary.shape, dtype=bool)
    target_border[ty1:ty2, tx1:tx2] = True
    target_border &= ~(
        cv2.erode(
            target_border.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1
        ) > 0
    )

    for label in range(1, component_count):
        x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
        if area < 24 or area > max_gap_area:
            continue
        component = labels == label
        ys, xs = np.where(component)
        if xs.size == 0:
            continue
        if (
            int(xs.min()) <= tx1 + margin or int(ys.min()) <= ty1 + margin or
            int(xs.max()) >= tx2 - margin - 1 or int(ys.max()) >= ty2 - margin - 1
        ):
            continue
        aspect = max(comp_w, comp_h) / max(1, min(comp_w, comp_h))
        if aspect > 9.0 and area < int(max_gap_area * 0.25):
            continue
        surrounding = cv2.dilate(component.astype(np.uint8), ring_kernel, iterations=1) > 0
        surrounding &= ~component
        support_ratio = int(np.count_nonzero(surrounding & binary)) / max(
            1,
            int(np.count_nonzero(surrounding))
        )
        if support_ratio < 0.60:
            continue

        subject_ring = surrounding & binary
        subject_pixels = image_lab[subject_ring]
        candidate_pixels = image_lab[component]
        if subject_pixels.size == 0 or candidate_pixels.size == 0:
            color_rejected += 1
            continue

        subject_center = np.median(subject_pixels, axis=0)
        candidate_distance = np.linalg.norm(
            candidate_pixels - subject_center.reshape(1, 3),
            axis=1
        )
        # A textured fabric can vary substantially, so use both a robust
        # median and a majority test rather than exact color equality.
        subject_match_ratio = float(np.mean(candidate_distance <= 58.0))
        if float(np.median(candidate_distance)) > 56.0 or subject_match_ratio < 0.62:
            color_rejected += 1
            continue

        # Compare against the semantic-box border as a weak scene prior. This
        # is not used to create foreground, only to reject obvious floor/wall
        # colors that happen to be enclosed by the coarse mask.
        border_pixels = image_lab[target_border]
        if border_pixels.size:
            background_center = np.median(border_pixels, axis=0)
            background_distance = float(np.linalg.norm(
                np.median(candidate_pixels, axis=0) - background_center
            ))
            subject_distance = float(np.linalg.norm(
                np.median(candidate_pixels, axis=0) - subject_center
            ))
            if background_distance + 8.0 < subject_distance:
                color_rejected += 1
                continue

        repaired[component] = True
        repair_pixels[component] = True
        gap_count += 1
        gap_pixels += int(area)

    return repaired.astype(np.float32), repair_pixels, {
        "gapsFilled": gap_count,
        "gapPixels": gap_pixels,
        "colorRejected": color_rejected,
    }


def build_furniture_internal_refine_prompts(img, mask, target_bbox):
    """Find plausible missing upholstery regions for one constrained SAM retry.

    A furniture mask may split at a fabric seam or shadow that is still
    connected to the scene through a narrow opening.  Those regions are not
    holes, so a post-processing fill cannot safely repair them.  This helper
    only proposes pixels that a bounded closing would surround and whose color
    is continuous with the immediately adjacent accepted furniture mask.
    """
    binary = np.asarray(mask > 0.5, dtype=bool)
    height, width = binary.shape[:2]
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = clamp(tx1, 0, width - 1)
    ty1 = clamp(ty1, 0, height - 1)
    tx2 = clamp(tx2, tx1 + 1, width)
    ty2 = clamp(ty2, ty1 + 1, height)
    subject_area = int(np.count_nonzero(binary))
    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    if subject_area < 128:
        return [], [], np.zeros_like(binary, dtype=bool), {
            "candidates": 0,
            "candidatePixels": 0,
            "colorRejected": 0,
        }

    crop_mask = binary[ty1:ty2, tx1:tx2]
    crop_img = img[ty1:ty2, tx1:tx2]
    if crop_img.size == 0 or not np.any(crop_mask):
        return [], [], np.zeros_like(binary, dtype=bool), {
            "candidates": 0,
            "candidatePixels": 0,
            "colorRejected": 0,
        }

    min_side = max(1, min(crop_mask.shape[:2]))
    # These radii only generate candidate points. They never directly alter
    # the output mask, and all SAM growth is checked below before acceptance.
    seam_radii = sorted(set([
        max(4, min(12, int(round(min_side * 0.012)))),
        max(8, min(20, int(round(min_side * 0.025)))),
        max(14, min(30, int(round(min_side * 0.040)))),
    ]))
    max_gap_area = max(
        160,
        min(32000, int(target_area * 0.14), int(subject_area * 0.22))
    )
    image_lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_mask = np.zeros_like(crop_mask, dtype=bool)
    color_rejected = 0

    for seam_radius in seam_radii:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (seam_radius * 2 + 1, seam_radius * 2 + 1)
        )
        closed = cv2.morphologyEx(
            crop_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        ) > 0
        bridged = closed & (~crop_mask)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            bridged.astype(np.uint8),
            connectivity=8
        )
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(5, seam_radius // 2) * 2 + 1, max(5, seam_radius // 2) * 2 + 1)
        )
        margin = seam_radius + 2
        for label in range(1, count):
            x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
            if area < 48 or area > max_gap_area:
                continue
            if x <= margin or y <= margin or x + comp_w >= crop_mask.shape[1] - margin or y + comp_h >= crop_mask.shape[0] - margin:
                continue
            component = labels == label
            # Avoid proposing broad concave exterior regions. A real missing
            # upholstery region is surrounded mostly by the accepted subject.
            ring = cv2.dilate(component.astype(np.uint8), ring_kernel, iterations=1) > 0
            ring &= ~component
            subject_ring = ring & crop_mask
            support_ratio = int(np.count_nonzero(subject_ring)) / max(1, int(np.count_nonzero(ring)))
            if support_ratio < 0.56:
                continue
            subject_pixels = image_lab[subject_ring]
            candidate_pixels = image_lab[component]
            if subject_pixels.size == 0 or candidate_pixels.size == 0:
                continue
            subject_center = np.median(subject_pixels, axis=0)
            distances = np.linalg.norm(candidate_pixels - subject_center.reshape(1, 3), axis=1)
            if float(np.median(distances)) > 62.0 or float(np.mean(distances <= 66.0)) < 0.58:
                color_rejected += 1
                continue
            new_part = component & (~candidate_mask)
            if not np.any(new_part):
                continue
            candidate_mask |= component

    if not np.any(candidate_mask):
        return [], [], np.zeros_like(binary, dtype=bool), {
            "candidates": 0,
            "candidatePixels": 0,
            "colorRejected": color_rejected,
        }

    # Put positive points at the deep interior of every proposed omission,
    # then retain several deep points in the existing object as anchors.
    positive_points = []
    seen = set()

    def append_point(point):
        px, py = int(point[0]), int(point[1])
        key = (px, py)
        if key not in seen:
            seen.add(key)
            positive_points.append([px, py])

    component_count, component_labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8),
        connectivity=8
    )
    for label in range(1, component_count):
        component = component_labels == label
        if int(np.count_nonzero(component)) < 48:
            continue
        distances = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        py, px = np.unravel_index(int(np.argmax(distances)), distances.shape)
        append_point((tx1 + px, ty1 + py))
        if len(positive_points) >= 5:
            break

    core_distance = cv2.distanceTransform(crop_mask.astype(np.uint8), cv2.DIST_L2, 5)
    anchor_spacing = max(18, int(round(min_side * 0.15)))
    for _ in range(5):
        py, px = np.unravel_index(int(np.argmax(core_distance)), core_distance.shape)
        if core_distance[py, px] < 3:
            break
        append_point((tx1 + px, ty1 + py))
        cv2.circle(core_distance, (int(px), int(py)), anchor_spacing, 0, thickness=-1)

    negative_points = build_boundary_negative_points(binary, [tx1, ty1, tx2, ty2], max_points=4)
    full_candidates = np.zeros_like(binary, dtype=bool)
    full_candidates[ty1:ty2, tx1:tx2] = candidate_mask
    return positive_points, negative_points, full_candidates, {
        "candidates": int(component_count - 1),
        "candidatePixels": int(np.count_nonzero(candidate_mask)),
        "colorRejected": color_rejected,
        "positiveCount": len(positive_points),
        "negativeCount": len(negative_points),
    }


def refine_furniture_mask_with_internal_points(img, mask, target_bbox, layer_name):
    """Retry SAM once when a textured furniture silhouette has supported gaps."""
    positive_points, negative_points, expected_growth, prompt_debug = build_furniture_internal_refine_prompts(
        img,
        mask,
        target_bbox
    )
    if not positive_points or not np.any(expected_growth):
        return mask, False, {
            **prompt_debug,
            "status": "skipped:no_supported_internal_gaps",
        }

    prompt_points = positive_points + negative_points
    prompt_labels = ([1] * len(positive_points)) + ([0] * len(negative_points))
    try:
        results = run_sam_bbox_inference(
            img,
            target_bbox,
            multimask_output=True,
            imgsz=HARD_EDGE_SAM_IMGSZ,
            points=[prompt_points],
            labels=[prompt_labels]
        )
        candidates = normalize_result_masks(
            results,
            img.shape[1],
            img.shape[0],
            interpolation=cv2.INTER_LINEAR
        )
    except Exception as error:
        print(f"Furniture internal SAM refine failed for {layer_name}: {error}")
        return mask, False, {
            **prompt_debug,
            "status": "failed:sam_error",
        }

    original = np.asarray(mask > 0.5, dtype=bool)
    original_area = max(1, int(np.count_nonzero(original)))
    min_side = max(1, min(target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1]))
    core_radius = max(2, min(10, int(round(min_side * 0.012))))
    core = cv2.erode(
        original.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (core_radius * 2 + 1, core_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    support_radius = max(10, min(32, int(round(min_side * 0.045))))
    supported_growth = cv2.dilate(
        original.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (support_radius * 2 + 1, support_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    supported_growth |= cv2.dilate(
        expected_growth.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1
    ) > 0

    best = None
    best_score = None
    expected_area = max(1, int(np.count_nonzero(expected_growth)))
    for candidate in candidates:
        candidate_binary = constrain_mask_to_bbox(candidate, target_bbox) > 0.5
        candidate_area = int(np.count_nonzero(candidate_binary))
        if candidate_area <= 0:
            continue
        preserved_core = int(np.count_nonzero(candidate_binary & core)) / max(1, int(np.count_nonzero(core)))
        preserved_subject = int(np.count_nonzero(candidate_binary & original)) / original_area
        new_pixels = candidate_binary & (~original)
        new_area = int(np.count_nonzero(new_pixels))
        expected_covered = int(np.count_nonzero(new_pixels & expected_growth)) / expected_area
        unsupported = int(np.count_nonzero(new_pixels & (~supported_growth)))
        growth_ratio = candidate_area / original_area
        if (
            preserved_core < 0.992 or
            preserved_subject < 0.965 or
            expected_covered < 0.20 or
            growth_ratio > 1.28 or
            new_area > max(int(original_area * 0.25), int(expected_area * 2.4)) or
            unsupported > max(64, int(new_area * 0.12))
        ):
            continue
        score = (expected_covered * 4.0) + (preserved_subject * 2.0) - (unsupported / max(1, new_area))
        if best_score is None or score > best_score:
            best = candidate_binary.astype(np.float32)
            best_score = score

    if best is None:
        return mask, False, {
            **prompt_debug,
            "status": "rejected:no_candidate_preserved_subject",
        }

    return best, True, {
        **prompt_debug,
        "status": "accepted",
        "areaBefore": original_area,
        "areaAfter": int(np.count_nonzero(best > 0.5)),
    }


def recover_entity_topology_gaps(mask, target_bbox):
    """Recover bounded mask gaps without using a class- or color-specific rule.

    SAM can leave an interior background pocket connected to the outside by a
    narrow raster crack.  Closing the crack reveals the pocket, but the
    closing itself is not used as the output silhouette. Only a pocket that
    remains well inside the semantic bbox and is substantially surrounded by
    the accepted mask is restored. This keeps real exterior openings and
    appendage spaces transparent.
    """
    binary = np.asarray(mask > 0.5, dtype=bool)
    height, width = binary.shape[:2]
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = clamp(tx1, 0, width - 1)
    ty1 = clamp(ty1, 0, height - 1)
    tx2 = clamp(tx2, tx1 + 1, width)
    ty2 = clamp(ty2, ty1 + 1, height)
    crop = binary[ty1:ty2, tx1:tx2]
    target_area = max(1, int(np.count_nonzero(crop)))
    if target_area < 256:
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:small_subject",
        }

    min_side = max(1, min(crop.shape[:2]))
    # The largest radius is bounded relative to the bbox. It seals narrow
    # segmentation cracks but cannot turn a broad exterior region into a
    # foreground object.
    radii = sorted(set([
        max(4, min(16, int(round(min_side * 0.012)))),
        max(8, min(30, int(round(min_side * 0.024)))),
        max(14, min(52, int(round(min_side * 0.038)))),
    ]))
    max_gap_area = max(
        192,
        min(60000, int(crop.size * 0.18), int(target_area * 0.38))
    )
    proposed = np.zeros_like(crop, dtype=bool)
    selected_radius = 0

    for radius in radii:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1)
        )
        closed = cv2.morphologyEx(
            crop.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        ) > 0
        # Close only a narrow raster crack first. The remaining background
        # component is the useful evidence: if it stays inside the crop after
        # the crack is sealed, it is a candidate missing pocket rather than
        # the exterior scene.
        added = closed & (~crop)
        remaining_background = (~closed).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            remaining_background,
            connectivity=8
        )
        margin = radius + 2
        ring_radius = max(3, min(10, radius // 2))
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (ring_radius * 2 + 1, ring_radius * 2 + 1)
        )
        for label in range(1, count):
            x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
            if area < 32 or area > max_gap_area:
                continue
            # Components touching this inner crop boundary are exterior
            # background, not a recoverable internal omission.
            if (
                x <= margin or y <= margin or
                x + comp_w >= crop.shape[1] - margin or
                y + comp_h >= crop.shape[0] - margin
            ):
                continue
            component = labels == label
            ring = cv2.dilate(
                component.astype(np.uint8),
                ring_kernel,
                iterations=1
            ) > 0
            ring &= ~component
            support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
            if support < 0.68:
                continue

            # Include the narrow bridge that was sealed around this pocket,
            # but only where it is immediately adjacent to the pocket.
            bridge = added & cv2.dilate(
                component.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (radius * 2 + 1, radius * 2 + 1)
                ),
                iterations=1
            ).astype(bool)
            # A fully enclosed intentional opening has no sealed crack. Do
            # not fill it merely because it is surrounded by the object.
            if not np.any(bridge):
                continue
            candidate = component | bridge
            candidate_area = int(np.count_nonzero(candidate))
            if candidate_area > max_gap_area:
                continue
            proposed |= candidate
            selected_radius = max(selected_radius, radius)

    if not np.any(proposed):
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:no_enclosed_supported_gap",
        }

    # Do not allow several nearby candidates to become a broad replacement
    # for the original silhouette. Each connected proposed region is checked
    # against the original mask and the bbox interior again.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        proposed.astype(np.uint8),
        connectivity=8
    )
    accepted = np.zeros_like(crop, dtype=bool)
    accepted_components = 0
    accepted_pixels = 0
    for label in range(1, count):
        x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
        component = labels == label
        if area < 32 or area > max_gap_area:
            continue
        ring_radius = max(3, min(10, selected_radius // 2 or 3))
        ring = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (ring_radius * 2 + 1, ring_radius * 2 + 1)
            ),
            iterations=1
        ) > 0
        ring &= ~component
        support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
        if support < 0.68:
            continue
        accepted |= component
        accepted_components += 1
        accepted_pixels += int(area)

    if not np.any(accepted):
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:unsupported_after_merge",
        }

    repaired = binary.copy()
    repaired[ty1:ty2, tx1:tx2] |= accepted
    return repaired.astype(np.float32), {
        "components": accepted_components,
        "pixels": accepted_pixels,
        "status": "accepted",
    }


def recover_micro_entity_gaps(mask, target_bbox):
    """Seal only tiny exterior-connected raster gaps in an entity mask.

    The broader topology pass handles missing regions behind a larger crack.
    This pass is deliberately limited to small components and a small closing
    radius, which targets isolated texture/highlight holes without filling a
    real open space between structural parts.
    """
    binary = np.asarray(mask > 0.5, dtype=bool)
    height, width = binary.shape[:2]
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = clamp(tx1, 0, width - 1)
    ty1 = clamp(ty1, 0, height - 1)
    tx2 = clamp(tx2, tx1 + 1, width)
    ty2 = clamp(ty2, ty1 + 1, height)
    crop = binary[ty1:ty2, tx1:tx2]
    subject_area = max(1, int(np.count_nonzero(crop)))
    if subject_area < 256:
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:small_subject",
        }

    min_side = max(1, min(crop.shape[:2]))
    max_gap_area = max(96, min(3200, int(subject_area * 0.012)))
    max_gap_side = max(10, min(96, int(round(min_side * 0.12))))
    proposed = np.zeros_like(crop, dtype=bool)
    selected_radius = 0

    for radius in (2, 3, 4, 5, 6):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1)
        )
        closed = cv2.morphologyEx(
            crop.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        ) > 0
        added = closed & (~crop)
        if not np.any(added):
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~closed).astype(np.uint8),
            connectivity=8
        )
        for label in range(1, count):
            x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
            if area < 8 or area > max_gap_area:
                continue
            if max(comp_w, comp_h) > max_gap_side:
                continue
            margin = radius + 1
            if (
                x <= margin or y <= margin or
                x + comp_w >= crop.shape[1] - margin or
                y + comp_h >= crop.shape[0] - margin
            ):
                continue
            component = labels == label
            ring_radius = max(2, min(6, radius))
            ring = cv2.dilate(
                component.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (ring_radius * 2 + 1, ring_radius * 2 + 1)
                ),
                iterations=1
            ) > 0
            ring &= ~component
            support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
            if support < 0.72:
                continue
            bridge = added & cv2.dilate(
                component.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (radius * 2 + 1, radius * 2 + 1)
                ),
                iterations=1
            ).astype(bool)
            if not np.any(bridge):
                continue
            proposed |= component | bridge
            selected_radius = max(selected_radius, radius)

    if not np.any(proposed):
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:no_micro_gap",
        }

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        proposed.astype(np.uint8),
        connectivity=8
    )
    accepted = np.zeros_like(crop, dtype=bool)
    accepted_components = 0
    accepted_pixels = 0
    for label in range(1, count):
        x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
        if area < 8 or area > max_gap_area or max(comp_w, comp_h) > max_gap_side:
            continue
        component = labels == label
        ring_radius = max(2, min(6, selected_radius or 2))
        ring = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (ring_radius * 2 + 1, ring_radius * 2 + 1)
            ),
            iterations=1
        ) > 0
        ring &= ~component
        support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
        if support < 0.72:
            continue
        accepted |= component
        accepted_components += 1
        accepted_pixels += int(area)

    if not np.any(accepted):
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:micro_support_failed",
        }

    repaired = binary.copy()
    repaired[ty1:ty2, tx1:tx2] |= accepted
    return repaired.astype(np.float32), {
        "components": accepted_components,
        "pixels": accepted_pixels,
        "status": "accepted",
    }


def recover_residual_entity_gaps(mask, target_bbox):
    """Final bounded recovery for small, irregular holes left by rasterization.

    This is intentionally geometry-only.  It can recover a small enclosed
    hole or a pocket reached through a thinner crack than the normal micro-gap
    pass, but it caps both each component and the total added area so broad
    structural openings remain transparent.
    """
    binary = np.asarray(mask > 0.5, dtype=bool)
    height, width = binary.shape[:2]
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = clamp(tx1, 0, width - 1)
    ty1 = clamp(ty1, 0, height - 1)
    tx2 = clamp(tx2, tx1 + 1, width)
    ty2 = clamp(ty2, ty1 + 1, height)
    crop = binary[ty1:ty2, tx1:tx2]
    subject_area = max(1, int(np.count_nonzero(crop)))
    if subject_area < 256:
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:small_subject",
        }

    min_side = max(1, min(crop.shape[:2]))
    max_component_area = max(128, min(12000, int(subject_area * 0.025)))
    max_component_side = max(20, min(150, int(round(min_side * 0.18))))
    max_total_added = max(256, min(18000, int(subject_area * 0.035)))
    proposed = np.zeros_like(crop, dtype=bool)
    selected_radius = 0

    # First collect genuinely enclosed components from the original mask.
    # They do not need a morphology bridge and are the common source of tiny
    # transparent islands after SAM rasterization.
    original_background = (~crop).astype(np.uint8)
    base_count, base_labels, base_stats, _ = cv2.connectedComponentsWithStats(
        original_background,
        connectivity=8
    )
    for label in range(1, base_count):
        x, y, comp_w, comp_h, area = [int(value) for value in base_stats[label]]
        if area < 8 or area > max_component_area or max(comp_w, comp_h) > max_component_side:
            continue
        if x <= 0 or y <= 0 or x + comp_w >= crop.shape[1] or y + comp_h >= crop.shape[0]:
            continue
        component = base_labels == label
        ring_radius = max(4, min(10, int(round(min_side * 0.018))))
        ring = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (ring_radius * 2 + 1, ring_radius * 2 + 1)
            ),
            iterations=1
        ) > 0
        ring &= ~component
        support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
        if support >= 0.62:
            proposed |= component
            selected_radius = max(selected_radius, ring_radius)

    for radius in (6, 8, 10, 12):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1)
        )
        closed = cv2.morphologyEx(
            crop.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        ) > 0
        added = closed & (~crop)
        if not np.any(added):
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~closed).astype(np.uint8),
            connectivity=8
        )
        for label in range(1, count):
            x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
            if area < 8 or area > max_component_area:
                continue
            if max(comp_w, comp_h) > max_component_side:
                continue
            margin = radius + 1
            if (
                x <= margin or y <= margin or
                x + comp_w >= crop.shape[1] - margin or
                y + comp_h >= crop.shape[0] - margin
            ):
                continue
            component = labels == label
            ring_radius = max(4, min(10, radius // 2))
            ring = cv2.dilate(
                component.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (ring_radius * 2 + 1, ring_radius * 2 + 1)
                ),
                iterations=1
            ) > 0
            ring &= ~component
            support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
            if support < 0.62:
                continue
            bridge = added & cv2.dilate(
                component.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (radius * 2 + 1, radius * 2 + 1)
                ),
                iterations=1
            ).astype(bool)
            # A bridge is optional here because fully enclosed holes were
            # collected in the first pass above.
            candidate = component | bridge
            candidate &= ~crop
            if int(np.count_nonzero(candidate)) > max_component_area:
                continue
            proposed |= candidate
            selected_radius = max(selected_radius, radius)

    if not np.any(proposed):
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:no_residual_gap",
        }

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        proposed.astype(np.uint8),
        connectivity=8
    )
    accepted = np.zeros_like(crop, dtype=bool)
    accepted_components = 0
    accepted_pixels = 0
    ring_radius = max(4, min(10, selected_radius // 2 or 4))
    for label in range(1, count):
        x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
        if area < 8 or area > max_component_area:
            continue
        if max(comp_w, comp_h) > max_component_side:
            continue
        component = labels == label
        ring = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (ring_radius * 2 + 1, ring_radius * 2 + 1)
            ),
            iterations=1
        ) > 0
        ring &= ~component
        support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
        if support < 0.62:
            continue
        if accepted_pixels + int(area) > max_total_added:
            continue
        accepted |= component
        accepted_components += 1
        accepted_pixels += int(area)

    if not np.any(accepted):
        return mask.astype(np.float32), {
            "components": 0,
            "pixels": 0,
            "status": "skipped:residual_support_failed",
        }

    repaired = binary.copy()
    repaired[ty1:ty2, tx1:tx2] |= accepted
    return repaired.astype(np.float32), {
        "components": accepted_components,
        "pixels": accepted_pixels,
        "status": "accepted",
    }


def select_clean_components(mask_binary, target_bbox, strategy_type=None):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_binary

    tx1, ty1, tx2, ty2 = target_bbox
    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    candidates = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < MASK_COMPONENT_MIN_PIXELS:
            continue

        component_bbox = [x, y, x + w, y + h]
        overlap = intersection_area(component_bbox, target_bbox)
        overlap_ratio = overlap / target_area
        component_mask = labels == label
        inside_target = int(np.count_nonzero(component_mask[ty1:ty2, tx1:tx2]))
        inside_ratio = inside_target / max(1, area)
        distance = bbox_distance(component_bbox, target_bbox)
        score = (overlap_ratio * 2.2) + (inside_ratio * 1.4) - (distance / max(32.0, (target_area ** 0.5) * 0.35))

        candidates.append({
            "label": label,
            "bbox": component_bbox,
            "area": area,
            "overlap_ratio": overlap_ratio,
            "inside_ratio": inside_ratio,
            "distance": distance,
            "score": score
        })

    if not candidates:
        return mask_binary

    candidates.sort(key=lambda item: item["score"], reverse=True)
    primary = candidates[0]
    keep_labels = {primary["label"]}

    for candidate in candidates[1:]:
        close_limit = MASK_KEEP_COMPONENT_MAX_GAP
        if strategy_type in {"furniture", "table", "lighting"}:
            close_limit = max(MASK_KEEP_COMPONENT_MAX_GAP, 48)
        close_to_primary = bbox_distance(candidate["bbox"], primary["bbox"]) <= close_limit
        overlaps_target = candidate["overlap_ratio"] >= 0.015 or candidate["inside_ratio"] >= 0.20
        support_like = False
        if strategy_type in {"furniture", "table", "lighting"}:
            comp_bbox = candidate["bbox"]
            comp_mask_area = candidate["area"]
            comp_shape = compute_shape_features(comp_bbox, target_bbox, comp_mask_area)
            support_like = (
                comp_shape["isThinVertical"] or
                comp_shape["isTableSupport"] or
                comp_shape["isHorizontalSurface"]
            ) and candidate["inside_ratio"] >= 0.12

        if ((close_to_primary and overlaps_target and candidate["area"] >= primary["area"] * 0.02) or support_like):
            keep_labels.add(candidate["label"])

    return np.isin(labels, list(keep_labels))


def cleanup_mask(mask, target_bbox, strategy_type=None):
    mask_binary = mask > 0.5
    if not np.any(mask_binary):
        return mask.astype(np.float32)

    # Do not prune fine/soft structures such as feather strands or sheer fabric.
    if strategy_type == "soft_edge":
        return mask.astype(np.float32)

    kernel_size = compute_cleanup_kernel(target_bbox)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = select_clean_components(mask_binary, target_bbox, strategy_type=strategy_type).astype(np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    if kernel_size >= 2 and strategy_type not in {"furniture", "table", "lighting"}:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = fill_small_holes(cleaned > 0).astype(np.uint8)
    return cleaned.astype(np.float32)


def constrain_mask_to_bbox(mask, target_bbox):
    """Keep segmentation pixels inside the semantic bbox supplied by the caller."""
    if mask is None:
        return mask

    constrained = np.zeros_like(mask, dtype=np.float32)
    x1, y1, x2, y2 = target_bbox
    constrained[y1:y2, x1:x2] = np.asarray(mask[y1:y2, x1:x2], dtype=np.float32)
    return constrained


def constrain_candidate_masks_to_bbox(candidate_masks, target_bbox):
    if candidate_masks is None or len(candidate_masks) == 0:
        return candidate_masks
    return np.stack([
        constrain_mask_to_bbox(mask, target_bbox)
        for mask in candidate_masks
    ], axis=0)


def derive_safe_entity_bbox(mask, target_bbox, strategy_type=None):
    """Extend an under-tight semantic bbox only on evidenced object sides.

    Semantic boxes are usually an output constraint, but they can occasionally
    stop a few pixels before the visible object edge. Only extend hard entities
    when the mask itself continues outside the box, remains connected to the
    inside silhouette, and occupies a substantial part of the corresponding
    boundary. This prevents a broad prompt expansion from becoming an output
    expansion.
    """
    if strategy_type not in HARD_EDGE_STRATEGIES.union({"table", "furniture"}):
        return target_bbox, None
    if mask is None or not np.any(mask > 0.5):
        return target_bbox, None

    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    binary = mask > 0.5

    # Direction-specific extension is capped relative to the box. The cap is
    # deliberately small because this is a bbox correction, not segmentation.
    max_dx = max(2, min(32, int(round((x2 - x1) * 0.06))))
    max_dy = max(2, min(32, int(round((y2 - y1) * 0.06))))
    result = [x1, y1, x2, y2]
    evidence = {}

    def connected_to_inside(outside_mask):
        if not np.any(outside_mask):
            return np.zeros_like(outside_mask)
        inside = np.zeros_like(binary, dtype=np.uint8)
        inside[y1:y2, x1:x2] = binary[y1:y2, x1:x2].astype(np.uint8)
        seed = outside_mask & (
            cv2.dilate(inside, np.ones((3, 3), np.uint8), iterations=1) > 0
        )
        # The outside component must be pixel-connected to the in-bbox
        # silhouette, not merely located in the same narrow border band.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            outside_mask.astype(np.uint8), connectivity=8
        )
        accepted = np.zeros_like(outside_mask, dtype=bool)
        for label in range(1, count):
            component = labels == label
            if np.any(component & seed):
                accepted |= component
        return accepted

    # Bottom/top use horizontal occupancy; left/right use vertical occupancy.
    directions = [
        ("bottom", y2, min(height, y2 + max_dy), (slice(y1, y2), slice(x1, x2)), max(1, x2 - x1)),
        ("top", max(0, y1 - max_dy), y1, (slice(y1, y2), slice(x1, x2)), max(1, x2 - x1)),
        ("right", x2, min(width, x2 + max_dx), (slice(y1, y2), slice(x1, x2)), max(1, y2 - y1)),
        ("left", max(0, x1 - max_dx), x1, (slice(y1, y2), slice(x1, x2)), max(1, y2 - y1)),
    ]
    for name, start, end, _, boundary_length in directions:
        if end <= start:
            continue
        if name in {"bottom", "top"}:
            outside = np.zeros_like(binary, dtype=bool)
            outside[start:end, x1:x2] = binary[start:end, x1:x2]
            boundary = binary[y2 - 1, x1:x2] if name == "bottom" else binary[y1, x1:x2]
            extension = end - y2 if name == "bottom" else y1 - start
        else:
            outside = np.zeros_like(binary, dtype=bool)
            outside[y1:y2, start:end] = binary[y1:y2, start:end]
            boundary = binary[y1:y2, x2 - 1] if name == "right" else binary[y1:y2, x1]
            extension = end - x2 if name == "right" else x1 - start

        connected = connected_to_inside(outside)
        occupied = int(np.count_nonzero(connected))
        boundary_occupied = int(np.count_nonzero(boundary))
        # A side must have both a connected continuation and broad contact with
        # the original silhouette. This rejects isolated background fragments.
        contact_ratio = boundary_occupied / max(1, boundary_length)
        band_ratio = occupied / max(1, boundary_length * extension)
        if contact_ratio >= 0.16 and band_ratio >= 0.12:
            evidence[name] = {
                "extension": int(extension),
                "contact": round(contact_ratio, 3),
                "band": round(band_ratio, 3),
            }
            if name == "bottom":
                result[3] = min(height, y2 + extension)
            elif name == "top":
                result[1] = max(0, y1 - extension)
            elif name == "right":
                result[2] = min(width, x2 + extension)
            else:
                result[0] = max(0, x1 - extension)

    if result == [x1, y1, x2, y2]:
        return target_bbox, None
    return result, evidence


def remove_food_detached_artifacts(img, mask, target_bbox):
    mask_binary = mask > 0.5
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary.astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        return mask.astype(np.float32), 0

    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area <= 0:
            continue
        components.append({
            "label": label,
            "bbox": [x, y, x + w, y + h],
            "area": int(area),
            "w": int(w),
            "h": int(h)
        })

    if len(components) <= 1:
        return mask.astype(np.float32), 0

    components.sort(key=lambda item: item["area"], reverse=True)
    primary = components[0]
    keep_labels = {primary["label"]}
    removed = 0
    primary_area = max(1, primary["area"])
    primary_bbox = primary["bbox"]

    for component in components[1:]:
        label = component["label"]
        component_mask = labels == label
        x1, y1, x2, y2 = component["bbox"]
        crop = img[y1:y2, x1:x2]
        crop_mask = component_mask[y1:y2, x1:x2]
        if crop.size == 0 or not np.any(crop_mask):
            continue

        pixels = crop[crop_mask]
        if pixels.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[crop_mask]
        gray_std = float(np.std(gray))
        color_std = float(np.mean(np.std(pixels.astype(np.float32), axis=0)))
        area_ratio = component["area"] / primary_area
        aspect = component["w"] / max(1, component["h"])
        distance = bbox_distance(component["bbox"], primary_bbox)
        overlap_ratio = intersection_area(component["bbox"], target_bbox) / max(1, bbox_area(target_bbox))
        component_bbox_area = max(1, component["w"] * component["h"])
        rectangularity = component["area"] / component_bbox_area
        primary_overlap_width = max(
            0,
            min(component["bbox"][2], primary_bbox[2]) - max(component["bbox"][0], primary_bbox[0])
        )
        primary_overlap_ratio = primary_overlap_width / max(
            1,
            min(component["w"], primary_bbox[2] - primary_bbox[0])
        )
        tx1, ty1, tx2, ty2 = target_bbox
        target_height = max(1, ty2 - ty1)
        component_center_y = ((component["bbox"][1] + component["bbox"][3]) / 2 - ty1) / target_height
        primary_center_y = ((primary_bbox[1] + primary_bbox[3]) / 2 - ty1) / target_height
        vertical_gap = max(
            0,
            primary_bbox[1] - component["bbox"][3],
            component["bbox"][1] - primary_bbox[3]
        )
        support_gap_limit = max(24.0, target_height * 0.16, (primary_area ** 0.5) * 0.42)
        is_lower_support = (
            component_center_y >= 0.36 and
            component_center_y >= primary_center_y - 0.08 and
            component["bbox"][3] >= ty1 + target_height * 0.52
        )

        # A plate or tray can be detached by a one-pixel gap after SAM removes
        # an attached graphic. It is broad, near the lower half of the food
        # bbox, and non-rectangular because only its visible rim is masked.
        # Flat panels remain removable: they are typically rectangular and do
        # not horizontally support the primary food component.
        is_food_support = (
            area_ratio >= 0.012 and
            0.85 <= aspect <= 6.5 and
            is_lower_support and
            primary_overlap_ratio >= 0.48 and
            rectangularity <= 0.90 and
            vertical_gap <= support_gap_limit and
            distance <= max(64.0, (primary_area ** 0.5) * 0.52)
        )

        is_flat_graphic = (
            area_ratio <= 0.48 and
            gray_std <= 32.0 and
            color_std <= 30.0 and
            0.45 <= aspect <= 2.35 and
            overlap_ratio <= 0.32 and
            distance <= max(96.0, (primary_area ** 0.5) * 0.75)
        )

        if is_flat_graphic and not is_food_support:
            print(
                f"Food detached cleanup removed component: "
                f"area={component['area']} ratio={area_ratio:.3f} aspect={aspect:.2f} "
                f"overlap={primary_overlap_ratio:.2f} rect={rectangularity:.2f} "
                f"centerY={component_center_y:.2f} gap={vertical_gap:.1f}"
            )
            removed += 1
            continue

        if is_food_support:
            print(
                f"Food detached cleanup kept support component: "
                f"area={component['area']} overlap={primary_overlap_ratio:.2f} "
                f"rect={rectangularity:.2f} centerY={component_center_y:.2f} "
                f"gap={vertical_gap:.1f}"
            )

        keep_labels.add(label)

    cleaned = np.isin(labels, list(keep_labels)).astype(np.float32)
    return cleaned, removed


def normalize_points(points):
    if not points:
        return None
    return [[int(x), int(y)] for x, y in points]


def build_positive_points_from_mask(mask_binary, bbox):
    ys, xs = np.where(mask_binary)
    if len(xs) == 0 or len(ys) == 0:
        x1, y1, x2, y2 = bbox
        return [[(x1 + x2) // 2, (y1 + y2) // 2]]

    x1, y1, x2, y2 = bbox
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    points = {(cx, cy)}
    points.add((int((x1 + x2) / 2), cy))
    points.add((cx, int((y1 + y2) / 2)))
    return normalize_points(list(points))


def build_food_positive_points_from_mask(mask_binary, bbox):
    base_points = build_positive_points_from_mask(mask_binary, bbox) or []
    ys, xs = np.where(mask_binary)
    if len(xs) == 0 or len(ys) == 0:
        return base_points

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    probe_points = [
        (int(x1 + width * 0.18), int(y1 + height * 0.18)),
        (int(x1 + width * 0.82), int(y1 + height * 0.18)),
        (int(x1 + width * 0.18), int(y1 + height * 0.82)),
        (int(x1 + width * 0.82), int(y1 + height * 0.82)),
    ]

    for px, py in probe_points:
        if 0 <= py < mask_binary.shape[0] and 0 <= px < mask_binary.shape[1] and mask_binary[py, px]:
            base_points.append([int(px), int(py)])

    # Keep points unique while preserving order.
    deduped = []
    seen = set()
    for point in base_points:
        key = (int(point[0]), int(point[1]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([key[0], key[1]])
    return deduped


def build_negative_points_from_mask(mask_binary, max_points=4):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return []

    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 16:
            continue
        components.append({
            "label": label,
            "area": int(area),
            "centroid": centroids[label],
            "bbox": [x, y, x + w, y + h]
        })

    if not components:
        return []

    components.sort(key=lambda item: item["area"], reverse=True)
    selected = components[:max_points]
    points = []
    for item in selected:
        cx, cy = item["centroid"]
        points.append([int(round(cx)), int(round(cy))])
    return points


def build_boundary_negative_points(mask_binary, target_bbox, max_points=4):
    h, w = mask_binary.shape[:2]
    x1, y1, x2, y2 = target_bbox
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 1, w)
    y2 = clamp(y2, 1, h)
    if x2 <= x1 or y2 <= y1:
        return []

    region = mask_binary[y1:y2, x1:x2]
    if region.size == 0:
        return []

    points = []
    rows = [
        (0, region.shape[0] // 2),
        (region.shape[0] - 1, region.shape[1] // 2),
        (region.shape[0] // 2, 0),
        (region.shape[0] // 2, region.shape[1] - 1),
    ]
    for ry, rx in rows:
        if len(points) >= max_points:
            break
        gy = int(y1 + ry)
        gx = int(x1 + rx)
        if 0 <= gy < h and 0 <= gx < w and not mask_binary[gy, gx]:
            points.append([gx, gy])
    return points


def shrink_bbox(box, ratio=0.18):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(round(width * ratio))
    pad_y = int(round(height * ratio))
    return [
        x1 + pad_x,
        y1 + pad_y,
        x2 - pad_x,
        y2 - pad_y
    ]


def bbox_from_center(cx, cy, half_w, half_h, img_w, img_h):
    return [
        clamp(int(round(cx - half_w)), 0, img_w - 1),
        clamp(int(round(cy - half_h)), 0, img_h - 1),
        clamp(int(round(cx + half_w)), 1, img_w),
        clamp(int(round(cy + half_h)), 1, img_h)
    ]


def sample_points_in_bbox(box, fractions):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    points = []
    for fx, fy in fractions:
        px = clamp(int(round(x1 + width * fx)), x1, max(x1, x2 - 1))
        py = clamp(int(round(y1 + height * fy)), y1, max(y1, y2 - 1))
        points.append([px, py])
    return points


def point_in_bbox(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px < x2 and y1 <= py < y2


def point_in_any_bbox(point, boxes):
    return any(point_in_bbox(point, box) for box in boxes)


def collect_context_bboxes(layer_meta, context_layers, target_bbox, img_w, img_h, predicate, overlap_ratio=0.0):
    boxes = []
    if not isinstance(context_layers, list):
        return boxes

    for other in context_layers:
        if same_layer(layer_meta, other) or not predicate(other):
            continue
        bbox = other.get("bbox") if isinstance(other, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        other_bbox = normalize_context_bbox_to_pixel(bbox, img_w, img_h)
        overlap = intersection_area(other_bbox, target_bbox)
        if overlap <= 0:
            continue
        if overlap_ratio > 0:
            other_area = max(1, bbox_area(other_bbox))
            if (overlap / other_area) < overlap_ratio:
                continue
        boxes.append({
            "layer": other,
            "bbox": other_bbox,
            "overlap": overlap
        })
    return boxes


def score_food_label_prompt_entry(entry, target_bbox):
    layer = entry.get("layer") or {}
    bbox = entry.get("bbox")
    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", ""))
    ]).lower()
    score = 0.0
    if any(token in text for token in ["price_badge", "price", "badge", "circle", "round", "coin", "sticker", "$", "价格", "价签", "徽章"]):
        score += 4.0
    if any(token in text for token in ["text", "label", "caption", "tag", "文字", "文本", "标签"]):
        score += 1.0
    score += float(entry.get("overlap", 0)) / max(1.0, float(bbox_area(target_bbox)))
    if bbox is not None:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        tx1, ty1, tx2, ty2 = target_bbox
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2
        dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
        scale = max(32.0, (bbox_area(target_bbox) ** 0.5))
        score += max(0.0, 1.5 - (dist / scale))
    return score


def build_negative_points_from_context_entries(entries, max_entries=4):
    points = []
    if not entries:
        return points

    for entry in entries[:max_entries]:
        bbox = entry.get("bbox")
        if not bbox:
            continue
        points.extend(sample_points_in_bbox(
            bbox,
            [
                (0.5, 0.5),
                (0.42, 0.5)
            ]
        ))
    return points


def is_price_like_text(text):
    return any(token in text for token in [
        "price_badge", "price", "badge", "circle", "round", "coin", "sticker", "$",
        "价格", "价签", "徽章"
    ])


def build_food_prompt_positive_points(target_bbox, avoid_boxes):
    fractions = [
        (0.5, 0.5),
        (0.36, 0.38),
        (0.64, 0.38),
        (0.28, 0.48),
        (0.72, 0.48),
        (0.5, 0.34),
        (0.5, 0.66),
        (0.34, 0.66),
        (0.66, 0.66),
        (0.24, 0.60),
        (0.76, 0.60)
    ]
    points = []
    for point in sample_points_in_bbox(target_bbox, fractions):
        if not point_in_any_bbox(point, avoid_boxes):
            points.append(point)
    if len(points) < 6:
        points = sample_points_in_bbox(target_bbox, fractions[:9])
    return points


def filter_points_outside_bboxes(points, boxes):
    return [point for point in points if not point_in_any_bbox(point, boxes)]


def build_prompt_seed_mask(target_bbox, img_w, img_h, negative_mask=None):
    seed_mask = np.zeros((img_h, img_w), dtype=bool)
    inner_bbox = shrink_bbox(target_bbox, ratio=0.20)
    x1, y1, x2, y2 = inner_bbox
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = target_bbox
    seed_mask[y1:y2, x1:x2] = True
    if negative_mask is not None:
        seed_mask &= ~negative_mask
    if np.count_nonzero(seed_mask) >= 16:
        return seed_mask

    fallback = np.zeros((img_h, img_w), dtype=bool)
    core_bbox = shrink_bbox(target_bbox, ratio=0.32)
    fx1, fy1, fx2, fy2 = core_bbox
    if fx2 <= fx1 or fy2 <= fy1:
        fx1, fy1, fx2, fy2 = target_bbox
    fallback[fy1:fy2, fx1:fx2] = True
    return fallback


def build_sam_prompt_inputs(layer_meta, context_layers, target_bbox, img_w, img_h):
    strategy = get_layer_strategy(layer_meta or {})
    strategy_type = strategy.get("type")

    exclude_entries = build_exclude_bboxes(layer_meta or {}, context_layers or [], target_bbox, img_w, img_h)
    strong_excludes = [entry for entry in exclude_entries if is_strong_exclude(entry)]
    strong_exclude_mask = build_exclude_mask(strong_excludes, img_w, img_h) if strong_excludes else np.zeros((img_h, img_w), dtype=bool)

    label_cleanup_mask = np.zeros((img_h, img_w), dtype=bool)
    flat_cleanup_mask = np.zeros((img_h, img_w), dtype=bool)
    negative_mask = strong_exclude_mask.copy()
    label_context_entries = []

    if strategy_type == "food_product":
        # Food/menu layouts are fragile when we push negative prompts into SAM.
        # Keep extraction generous and handle cleanup in a later dedicated pass.
        label_context_entries = []
        label_cleanup_mask = np.zeros((img_h, img_w), dtype=bool)
        flat_cleanup_mask = np.zeros((img_h, img_w), dtype=bool)

    seed_mask = build_prompt_seed_mask(target_bbox, img_w, img_h, negative_mask=negative_mask)
    if strategy_type == "food_product":
        positive_points = build_food_prompt_positive_points(target_bbox, [])
        positive_points.extend(build_food_positive_points_from_mask(seed_mask, target_bbox) or [])
    else:
        positive_points = build_positive_points_from_mask(seed_mask, target_bbox) or []

    if not positive_points:
        positive_points = sample_points_in_bbox(target_bbox, [(0.5, 0.5)])

    negative_points = []
    if strategy_type == "food_product":
        negative_points = []
    elif np.any(label_cleanup_mask):
        negative_points.extend(build_negative_points_from_mask(label_cleanup_mask, max_points=4))
    if strong_excludes:
        negative_points.extend(build_negative_points_from_mask(strong_exclude_mask, max_points=4))
    if strategy_type != "food_product":
        negative_points.extend(build_boundary_negative_points(seed_mask, target_bbox, max_points=4))

    prompt_points = []
    prompt_labels = []
    seen = set()
    for point in positive_points:
        key = (int(point[0]), int(point[1]), 1)
        if key in seen:
            continue
        seen.add(key)
        prompt_points.append([key[0], key[1]])
        prompt_labels.append(1)

    for point in negative_points:
        key = (int(point[0]), int(point[1]), 0)
        if key in seen:
            continue
        seen.add(key)
        prompt_points.append([key[0], key[1]])
        prompt_labels.append(0)

    if prompt_points:
        prompt_points = [prompt_points]
        prompt_labels = [prompt_labels]
    else:
        prompt_points = None
        prompt_labels = None

    return {
        "points": prompt_points,
        "labels": prompt_labels,
        "labelCleanupMask": label_cleanup_mask,
        "flatCleanupMask": flat_cleanup_mask,
        "strongExcludeMask": strong_exclude_mask,
        "strategyType": strategy_type
    }


def build_soft_edge_prompt_inputs(img, target_bbox):
    """Build SAM points from color separation without assuming object geometry."""
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = target_bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return {"points": None, "labels": None, "positive": [], "negative": []}

    crop_h, crop_w = crop.shape[:2]
    border_x = max(2, int(round(crop_w * 0.06)))
    border_y = max(2, int(round(crop_h * 0.06)))
    border_pixels = np.concatenate([
        crop[:border_y].reshape(-1, 3),
        crop[-border_y:].reshape(-1, 3),
        crop[:, :border_x].reshape(-1, 3),
        crop[:, -border_x:].reshape(-1, 3)
    ], axis=0).astype(np.float32)
    bg_lab = np.median(cv2.cvtColor(border_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_distance = np.linalg.norm(crop_lab - bg_lab.reshape(1, 1, 3), axis=2)

    border_lab = cv2.cvtColor(border_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    border_distance = np.linalg.norm(border_lab - bg_lab.reshape(1, 3), axis=1)
    bg_center = float(np.median(border_distance))
    bg_mad = float(np.median(np.abs(border_distance - bg_center)))
    threshold = max(
        float(np.percentile(color_distance, 88)),
        bg_center + max(8.0, bg_mad * 4.0)
    )

    seed = color_distance >= threshold
    inner_x1 = max(0, int(round(crop_w * 0.03)))
    inner_y1 = max(0, int(round(crop_h * 0.03)))
    inner_x2 = min(crop_w, crop_w - inner_x1)
    inner_y2 = min(crop_h, crop_h - inner_y1)
    inner = np.zeros((crop_h, crop_w), dtype=bool)
    inner[inner_y1:inner_y2, inner_x1:inner_x2] = True
    seed &= inner

    positive = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(8, int(crop_w * crop_h * 0.00008)):
            continue
        components.append((area, centroids[label]))
    components.sort(reverse=True, key=lambda item: item[0])
    for _, centroid in components[:6]:
        positive.append([int(round(x1 + centroid[0])), int(round(y1 + centroid[1]))])

    if not positive:
        flat_scores = color_distance.copy()
        flat_scores[~inner] = -1
        for flat_index in np.argsort(flat_scores.reshape(-1))[::-1][:6]:
            py, px = np.unravel_index(flat_index, flat_scores.shape)
            if flat_scores[py, px] < 0:
                break
            positive.append([int(x1 + px), int(y1 + py)])

    negative = []
    low_threshold = max(threshold * 0.35, bg_center + bg_mad * 1.5)
    low_score = color_distance <= low_threshold
    border_candidates = np.zeros((crop_h, crop_w), dtype=bool)
    border_candidates[:border_y] = True
    border_candidates[-border_y:] = True
    border_candidates[:, :border_x] = True
    border_candidates[:, -border_x:] = True
    low_score &= border_candidates
    negative_pixels = np.argwhere(low_score)
    if len(negative_pixels):
        step = max(1, len(negative_pixels) // 6)
        for py, px in negative_pixels[::step][:6]:
            negative.append([int(x1 + px), int(y1 + py)])

    points = positive + negative
    labels_out = [1] * len(positive) + [0] * len(negative)
    return {
        "points": [points] if points else None,
        "labels": [labels_out] if points else None,
        "positive": positive,
        "negative": negative
    }


def filter_soft_edge_masks_by_points(candidate_masks, prompt_inputs):
    """Reject SAM multimasks that ignore fine-object foreground/background points."""
    if candidate_masks is None or len(candidate_masks) == 0 or not prompt_inputs:
        return candidate_masks

    positive = prompt_inputs.get("positive") or []
    negative = prompt_inputs.get("negative") or []
    if not positive and not negative:
        return candidate_masks

    scored = []
    for index, mask in enumerate(candidate_masks):
        mask_binary = mask > 0.5
        positive_hits = sum(
            1 for x, y in positive
            if 0 <= y < mask_binary.shape[0] and 0 <= x < mask_binary.shape[1] and mask_binary[y, x]
        )
        negative_hits = sum(
            1 for x, y in negative
            if 0 <= y < mask_binary.shape[0] and 0 <= x < mask_binary.shape[1] and mask_binary[y, x]
        )
        positive_ratio = positive_hits / max(1, len(positive))
        negative_ratio = negative_hits / max(1, len(negative))
        scored.append((positive_ratio - (negative_ratio * 1.5), positive_ratio, negative_ratio, index))

    valid = [row for row in scored if row[1] >= 0.50 and row[2] <= 0.50]
    if not valid:
        valid = sorted(scored, reverse=True)[:1]
    selected_indexes = {row[3] for row in valid}
    print(
        "Soft-edge SAM point validation: "
        f"kept={sorted(selected_indexes)} "
        f"scores={[round(row[0], 3) for row in sorted(scored, reverse=True)]}"
    )
    return np.stack([
        mask for index, mask in enumerate(candidate_masks)
        if index in selected_indexes
    ], axis=0)


def build_soft_edge_alpha(img, mask, target_bbox, context_bbox=None):
    """Recover translucent strands from the SAM silhouette using border color."""
    binary = mask > 0.5
    x1, y1, x2, y2 = context_bbox or expand_bbox(*target_bbox, img.shape[1], img.shape[0])
    crop = img[y1:y2, x1:x2]
    crop_mask = binary[y1:y2, x1:x2]
    if crop.size == 0 or not np.any(crop_mask):
        return dilate_and_feather_mask(mask)

    crop_h, crop_w = crop.shape[:2]
    border_x = max(2, int(round(crop_w * 0.06)))
    border_y = max(2, int(round(crop_h * 0.06)))
    border_region = np.zeros((crop_h, crop_w), dtype=bool)
    border_region[:border_y] = True
    border_region[-border_y:] = True
    border_region[:, :border_x] = True
    border_region[:, -border_x:] = True
    border_pixels = crop[border_region]
    if border_pixels.size == 0:
        return dilate_and_feather_mask(mask)

    border_lab = cv2.cvtColor(
        border_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = np.median(border_lab, axis=0)
    distance = np.linalg.norm(crop_lab - bg_lab.reshape(1, 1, 3), axis=2)
    border_distance = distance[border_region]
    bg_median = float(np.median(border_distance))
    bg_mad = float(np.median(np.abs(border_distance - bg_median)))
    low = max(8.0, bg_median + max(6.0, bg_mad * 4.0), float(np.percentile(border_distance, 98)))

    core_kernel_size = max(3, int(round(min(crop_w, crop_h) * 0.012)))
    if core_kernel_size % 2 == 0:
        core_kernel_size += 1
    core = cv2.erode(crop_mask.astype(np.uint8), np.ones((core_kernel_size, core_kernel_size), np.uint8), iterations=1) > 0
    core_distances = distance[core]
    if core_distances.size == 0:
        core_distances = distance[crop_mask]
    if core_distances.size == 0:
        return dilate_and_feather_mask(mask)
    high = max(low + 20.0, float(np.percentile(core_distances, 72)))
    high = min(high, max(low + 24.0, float(np.percentile(distance, 99.5))))

    normalized = np.clip((distance - low) / max(1.0, high - low), 0.0, 1.0)
    color_alpha = (normalized * normalized * (3.0 - 2.0 * normalized) * 255.0).astype(np.uint8)

    # Allow a narrow halo around SAM's coarse silhouette so fine tips can return,
    # while the color test still rejects the surrounding solid background.
    support_size = max(5, min(31, int(round(min(crop_w, crop_h) * 0.035))))
    if support_size % 2 == 0:
        support_size += 1
    support = cv2.dilate(crop_mask.astype(np.uint8), np.ones((support_size, support_size), np.uint8), iterations=1) > 0
    alpha_crop = np.where(support, color_alpha, 0).astype(np.uint8)
    alpha_crop[crop_mask & (color_alpha > 0)] = np.maximum(
        alpha_crop[crop_mask & (color_alpha > 0)],
        color_alpha[crop_mask & (color_alpha > 0)]
    )
    alpha_crop = cv2.GaussianBlur(alpha_crop, (3, 3), 0)

    alpha = np.zeros(binary.shape, dtype=np.uint8)
    alpha[y1:y2, x1:x2] = alpha_crop
    return alpha


def despill_soft_edge_image(img, alpha, target_bbox, context_bbox=None):
    """Remove sampled background color from translucent soft-edge pixels."""
    x1, y1, x2, y2 = context_bbox or expand_bbox(*target_bbox, img.shape[1], img.shape[0])
    crop = img[y1:y2, x1:x2].astype(np.float32)
    alpha_crop = alpha[y1:y2, x1:x2].astype(np.float32) / 255.0
    if crop.size == 0:
        return img

    crop_h, crop_w = crop.shape[:2]
    border_x = max(2, int(round(crop_w * 0.06)))
    border_y = max(2, int(round(crop_h * 0.06)))
    border = np.concatenate([
        crop[:border_y].reshape(-1, 3),
        crop[-border_y:].reshape(-1, 3),
        crop[:, :border_x].reshape(-1, 3),
        crop[:, -border_x:].reshape(-1, 3)
    ], axis=0)
    bg = np.median(border, axis=0)

    soft = (alpha_crop > 0.08) & (alpha_crop < 0.92)
    if np.any(soft):
        safe_alpha = np.maximum(alpha_crop[soft, None], 0.28)
        recovered = (crop[soft] - (bg[None, :] * (1.0 - safe_alpha))) / safe_alpha
        recovered = np.clip(recovered, 0.0, 255.0)
        correction = np.clip((0.92 - alpha_crop[soft]) / 0.64, 0.0, 1.0)[:, None]
        crop[soft] = crop[soft] * (1.0 - correction) + recovered * correction

    output = img.copy()
    output[y1:y2, x1:x2] = np.clip(crop, 0, 255).astype(np.uint8)
    return output


def build_hard_edge_alpha(mask, target_bbox):
    """Anti-alias the accepted silhouette without expanding or blurring it."""
    x1, y1, x2, y2 = target_bbox
    crop_probability = np.clip(
        np.asarray(mask[y1:y2, x1:x2], dtype=np.float32),
        0.0,
        1.0
    )
    crop_binary = (crop_probability > 0.5).astype(np.uint8)
    if crop_binary.size == 0 or not np.any(crop_binary):
        return np.asarray(mask * 255.0, dtype=np.uint8)

    # Always supersample the accepted silhouette. Returning SAM's resized
    # probability raster directly makes low-resolution mask stair-steps
    # visible on small hard objects. Supersampling only computes fractional
    # coverage of the accepted contour; it does not expand the silhouette.
    scale = 8
    crop_h, crop_w = crop_binary.shape[:2]
    hi = np.zeros((crop_h * scale, crop_w * scale), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        crop_binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.asarray(mask * 255.0, dtype=np.uint8)

    scaled_contours = [
        np.rint(contour.astype(np.float32) * scale).astype(np.int32)
        for contour in contours
    ]
    for index, contour in enumerate(scaled_contours):
        # Preserve holes in the accepted silhouette instead of filling an
        # opening just because the contour is rasterized at a larger size.
        color = 0 if hierarchy[0][index][3] >= 0 else 255
        cv2.drawContours(hi, [contour], 0, color, thickness=cv2.FILLED)
    refined_crop = cv2.resize(
        hi,
        (crop_w, crop_h),
        interpolation=cv2.INTER_AREA
    )

    alpha = np.zeros(mask.shape, dtype=np.uint8)
    alpha[y1:y2, x1:x2] = refined_crop
    return alpha


def generate_table_safe_matte(mask, target_bbox):
    """Rasterize the accepted table SAM silhouette without reclassifying it.

    Tables often contain large, low-contrast surfaces.  GrabCut can turn the
    floor or wall inside the semantic bbox into probable foreground, or erase
    a bright stone surface.  For this profile SAM's accepted silhouette is
    the authoritative foreground; only contour coverage antialiasing is
    allowed here, with no expansion and no color-model pass.
    """
    constrained = constrain_mask_to_bbox(mask, target_bbox)
    if not np.any(constrained > 0.0):
        return dilate_and_feather_mask(constrained)
    alpha = build_hard_edge_alpha(constrained, target_bbox)
    # Keep fractional contour coverage from crossing a concavity or a small
    # hole in the accepted silhouette.
    return np.where(constrained > 0.5, alpha, 0).astype(np.uint8)


def despill_hard_edge_image(img, alpha, target_bbox):
    """Replace edge spill with the nearest opaque subject color."""
    x1, y1, x2, y2 = target_bbox
    crop = img[y1:y2, x1:x2].copy()
    crop_alpha = alpha[y1:y2, x1:x2]
    if crop.size == 0:
        return img

    opaque = crop_alpha >= 245
    edge = (crop_alpha > 0) & (crop_alpha < 245)
    if not np.any(opaque) or not np.any(edge):
        return img

    distance_input = (~opaque).astype(np.uint8)
    distances, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL
    )
    max_label = int(labels.max())
    if max_label <= 0:
        return img

    nearest_colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
    nearest_colors[labels[opaque]] = crop[opaque]
    min_side = max(1, min(x2 - x1, y2 - y1))
    edge_radius = max(2, min(4, int(round(min_side * 0.012))))
    near_edge = edge & (distances <= edge_radius + 1.0) & (labels > 0)
    if np.any(near_edge):
        crop[near_edge] = nearest_colors[labels[near_edge]]

    output = img.copy()
    output[y1:y2, x1:x2] = crop
    return output


def filter_hard_edge_recovery_by_color(crop, recovered, coarse, local_target, strategy_type=None):
    """Remove newly recovered pixels that look more like the bbox surround than the object."""
    new_pixels = recovered & (~coarse) & local_target
    if not np.any(new_pixels):
        return recovered, 0

    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_border = local_target & ~(
        cv2.erode(local_target.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    )
    background_pixels = crop_lab[(~local_target) | target_border]
    foreground_pixels = crop_lab[coarse]
    if len(background_pixels) == 0 or len(foreground_pixels) == 0:
        return recovered, 0

    background_center = np.median(background_pixels, axis=0)
    # Use the central foreground colors as the reference, so a few SAM pixels
    # on a dark cable do not make the whole light fixture look like background.
    foreground_center = np.median(foreground_pixels, axis=0)
    threshold_by_strategy = {
        "lighting": 30.0,
        "table": 38.0,
        "furniture": 38.0,
        "decor_arrangement": 44.0,
        "decor_atomic": 38.0
    }
    max_foreground_distance = threshold_by_strategy.get(strategy_type, 38.0)

    ys, xs = np.where(new_pixels)
    keep = np.ones(len(xs), dtype=bool)
    radius = max(4, min(10, int(round(min(crop.shape[:2]) * 0.025))))
    for index, (py, px) in enumerate(zip(ys, xs)):
        y1 = max(0, int(py) - radius)
        y2 = min(crop.shape[0], int(py) + radius + 1)
        x1 = max(0, int(px) - radius)
        x2 = min(crop.shape[1], int(px) + radius + 1)
        local_subject = coarse[y1:y2, x1:x2]
        if not np.any(local_subject):
            keep[index] = False
            continue

        pixel = crop_lab[int(py), int(px)]
        nearby_subject = crop_lab[y1:y2, x1:x2][local_subject]
        foreground_distance = float(np.min(np.linalg.norm(nearby_subject - pixel, axis=1)))
        background_distance = float(np.linalg.norm(pixel - background_center))
        # Reject pixels that are both unlike nearby subject colors and close to
        # the surrounding background. The second condition catches the dark
        # halo/islands that GrabCut can classify as probable foreground.
        if (
            foreground_distance > max_foreground_distance and
            background_distance + 8.0 < foreground_distance
        ):
            keep[index] = False

    filtered = recovered.copy()
    filtered[ys[~keep], xs[~keep]] = False
    return filtered, int(np.count_nonzero(~keep))


def generate_hard_edge_matte(img, mask, target_bbox, strategy_type=None):
    """Recover low-contrast hard-object edges without trusting the coarse mask as a boundary."""
    height, width = img.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    # Keep a small background ring for GrabCut's color model, while using the
    # semantic bbox as the hard limit for any recovered pixels.
    work_bbox = expand_bbox(x1, y1, x2, y2, width, height, ratio=0.12)
    wx1, wy1, wx2, wy2 = work_bbox
    crop = img[wy1:wy2, wx1:wx2]
    if crop.size == 0:
        return dilate_and_feather_mask(mask)

    crop_h, crop_w = crop.shape[:2]
    local_target = np.zeros((crop_h, crop_w), dtype=bool)
    local_target[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = True
    coarse = (mask[wy1:wy2, wx1:wx2] > 0.5) & local_target
    if not np.any(coarse):
        return dilate_and_feather_mask(mask)

    gc_mask = np.full((crop_h, crop_w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[~local_target] = cv2.GC_BGD
    gc_mask[local_target] = cv2.GC_PR_FGD

    # SAM pixels are reliable foreground seeds. Erosion protects thin edges
    # from being used as background by GrabCut's color model.
    coarse_u8 = coarse.astype(np.uint8)
    core_kernel_size = max(3, min(9, int(round(min(crop_h, crop_w) * 0.025)) | 1))
    core_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (core_kernel_size, core_kernel_size)
    )
    sure_fg = cv2.erode(coarse_u8, core_kernel, iterations=1) > 0
    if not np.any(sure_fg):
        sure_fg = coarse.copy()
    gc_mask[coarse] = cv2.GC_PR_FGD
    gc_mask[sure_fg] = cv2.GC_FGD

    # Pixels along the semantic bbox border are the strongest background
    # evidence available when the object blends into the scene. Keep already
    # selected SAM pixels protected even if they touch that border.
    border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    target_border = local_target & ~(
        cv2.erode(local_target.astype(np.uint8), border_kernel, iterations=1) > 0
    )
    gc_mask[target_border & ~coarse] = cv2.GC_PR_BGD
    gc_mask[~local_target] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            crop,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            HARD_EDGE_GRABCUT_ITER_COUNT,
            cv2.GC_INIT_WITH_MASK
        )
    except Exception as error:
        print(f"Hard-edge bbox matting failed: {error}")
        return dilate_and_feather_mask(mask)

    recovered_crop = np.logical_or(
        gc_mask == cv2.GC_FGD,
        gc_mask == cv2.GC_PR_FGD
    ) & local_target
    recovered_crop |= coarse

    # GrabCut can classify a low-contrast background island inside the bbox as
    # probable foreground. Recovery is only valid in a narrow band adjacent to
    # the SAM silhouette; it must not jump across an internal empty area.
    support_ratio_by_strategy = {
        "lighting": 0.018,
        "table": 0.030,
        "furniture": 0.030,
        # A framed plane has no fine appendages to recover. Keep the SAM
        # silhouette intact and allow only a minimal contour correction.
        "wall_art": 0.012,
        "decor_arrangement": 0.035,
        "decor_atomic": 0.030
    }
    support_ratio = support_ratio_by_strategy.get(strategy_type, 0.030)
    support_radius = max(4, min(18, int(round(min(y2 - y1, x2 - x1) * support_ratio))))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (support_radius * 2 + 1, support_radius * 2 + 1)
    )
    recovery_support = cv2.dilate(
        coarse.astype(np.uint8),
        support_kernel,
        iterations=1
    ) > 0
    recovered_crop = (recovered_crop & recovery_support) | coarse
    recovered_crop, color_removed = filter_hard_edge_recovery_by_color(
        crop,
        recovered_crop,
        coarse,
        local_target,
        strategy_type=strategy_type
    )

    coarse_area = max(1, int(np.count_nonzero(coarse)))
    recovered_area = int(np.count_nonzero(recovered_crop))
    target_area = max(1, int(np.count_nonzero(local_target)))
    preserve_ratio = int(np.count_nonzero(recovered_crop & coarse)) / coarse_area
    growth_ratio = recovered_area / coarse_area
    max_fill_by_strategy = {
        "lighting": 0.62,
        "table": 0.72,
        "furniture": 0.76,
        "wall_art": 0.98,
        "decor_arrangement": 0.90,
        "decor_atomic": 0.72
    }
    max_fill = max_fill_by_strategy.get(strategy_type, 0.76)
    recovered_fill = recovered_area / target_area
    max_growth_by_strategy = {
        # Lighting often has a low-contrast halo around the shade. Do not let
        # bbox matting replace SAM's silhouette with that surrounding plateau.
        "lighting": 1.045,
        "table": 1.65,
        "furniture": 1.65,
        "wall_art": 1.15,
        "decor_arrangement": 1.90,
        "decor_atomic": 1.65
    }
    max_growth = max_growth_by_strategy.get(strategy_type, 1.65)
    if (
        preserve_ratio < 0.985 or
        growth_ratio > max_growth or
        recovered_fill > max_fill
    ):
        print(
            f"Hard-edge bbox matting rejected for area safety: "
            f"preserve={preserve_ratio:.3f} growth={growth_ratio:.3f} "
            f"fill={recovered_fill:.3f} limit={max_fill:.3f}"
        )
        return build_hard_edge_alpha(mask, target_bbox)

    recovered_mask = np.zeros(mask.shape, dtype=np.float32)
    recovered_mask[wy1:wy2, wx1:wx2] = recovered_crop.astype(np.float32)
    # Rasterize the accepted contour at 4x instead of blurring a binary mask.
    # Blur creates a low-alpha ring containing the original background color.
    alpha = build_hard_edge_alpha(recovered_mask, target_bbox)
    alpha = constrain_mask_to_bbox(alpha.astype(np.float32), target_bbox).astype(np.uint8)
    print(
        f"Hard-edge bbox matting accepted for area={coarse_area}->{recovered_area} "
        f"growth={growth_ratio:.3f} fill={recovered_fill:.3f} "
        f"colorRemoved={color_removed}"
    )
    return alpha


def build_refine_prompts(coarse_mask, local_bbox, cleanup_mask=None, strategy_type=None):
    coarse_binary = coarse_mask > 0.5
    positive_points = build_positive_points_from_mask(coarse_binary, local_bbox) or []
    core_kernel = np.ones((3, 3), np.uint8)
    core_mask = cv2.erode(coarse_binary.astype(np.uint8), core_kernel, iterations=1) > 0
    core_points = build_positive_points_from_mask(core_mask, local_bbox) if np.any(core_mask) else []
    positive_points.extend(core_points or [])
    negative_points = []

    if cleanup_mask is not None:
        cleanup_binary = cleanup_mask > 0
        cleanup_negative_limit = 2 if strategy_type == "food_product" else 4
        negative_points.extend(build_negative_points_from_mask(cleanup_binary, max_points=cleanup_negative_limit))
    if strategy_type != "food_product":
        negative_points.extend(build_boundary_negative_points(coarse_binary, local_bbox, max_points=4))

    prompt_points = []
    prompt_labels = []
    seen = set()
    for point in positive_points:
        key = (int(point[0]), int(point[1]))
        if key in seen:
            continue
        seen.add(key)
        prompt_points.append([key[0], key[1]])
        prompt_labels.append(1)
    for point in negative_points:
        key = (int(point[0]), int(point[1]))
        if key in seen:
            continue
        seen.add(key)
        prompt_points.append([key[0], key[1]])
        prompt_labels.append(0)

    if prompt_points:
        batched_points = [prompt_points]
        batched_labels = [prompt_labels]
    else:
        batched_points = None
        batched_labels = None

    return batched_points, batched_labels, None


def build_hard_edge_completion_points(img, mask, target_bbox, max_points=4):
    """Find conservative positive probes just outside an incomplete silhouette."""
    binary = mask > 0.5
    x1, y1, x2, y2 = target_bbox
    crop = img[y1:y2, x1:x2]
    crop_mask = binary[y1:y2, x1:x2]
    if crop.size == 0 or not np.any(crop_mask):
        return []

    min_side = max(1, min(crop.shape[:2]))
    ring_radius = max(3, min(9, int(round(min_side * 0.018))))
    ring_size = ring_radius * 2 + 1
    dilated = cv2.dilate(
        crop_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_size, ring_size)),
        iterations=1
    ) > 0
    ring = dilated & (~crop_mask)
    if not np.any(ring):
        return []

    # A probe should be close to the accepted object and look like its local
    # material. This avoids blindly sampling all empty bbox pixels.
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    local_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    nearby_subject = cv2.dilate(
        crop_mask.astype(np.uint8),
        local_kernel,
        iterations=1
    ) > 0
    candidate_pixels = np.argwhere(ring & nearby_subject)
    if len(candidate_pixels) == 0:
        return []

    rows = []
    for py, px in candidate_pixels[::max(1, len(candidate_pixels) // 500)]:
        y_lo = max(0, int(py) - 3)
        y_hi = min(crop.shape[0], int(py) + 4)
        x_lo = max(0, int(px) - 3)
        x_hi = min(crop.shape[1], int(px) + 4)
        local_subject = crop_mask[y_lo:y_hi, x_lo:x_hi]
        if not np.any(local_subject):
            continue
        subject_colors = crop_lab[y_lo:y_hi, x_lo:x_hi][local_subject]
        color_distance = float(np.linalg.norm(
            crop_lab[int(py), int(px)] - np.median(subject_colors, axis=0)
        ))
        # Prefer probes with more nearby subject support and avoid isolated
        # single-pixel noise along the silhouette.
        support = int(np.count_nonzero(local_subject))
        rows.append((color_distance - min(12, support) * 0.8, int(px), int(py)))

    rows.sort(key=lambda item: item[0])
    points = []
    min_gap = max(5, int(round(min_side * 0.06)))
    for _, px, py in rows:
        gx, gy = x1 + px, y1 + py
        if any((gx - ox) ** 2 + (gy - oy) ** 2 < min_gap ** 2 for ox, oy in points):
            continue
        points.append([gx, gy])
        if len(points) >= max_points:
            break
    return points


def recover_hard_edge_mask_with_points(img, mask, target_bbox, layer_name):
    """Ask SAM to recover hard-object regions omitted by the bbox candidate."""
    positive = build_hard_edge_completion_points(img, mask, target_bbox)
    if not positive:
        return mask, False

    crop, crop_bounds = crop_region_from_bbox(img, target_bbox, 0.12)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_h, crop_w = crop.shape[:2]
    if crop.size == 0:
        return mask, False

    local_bbox = [
        target_bbox[0] - crop_x1,
        target_bbox[1] - crop_y1,
        target_bbox[2] - crop_x1,
        target_bbox[3] - crop_y1
    ]
    local_points = [[x - crop_x1, y - crop_y1] for x, y in positive]
    try:
        results = run_sam_bbox_inference(
            crop,
            local_bbox,
            multimask_output=True,
            imgsz=HARD_EDGE_SAM_IMGSZ,
            points=[local_points],
            labels=[[1] * len(local_points)]
        )
        candidates = normalize_result_masks(
            results,
            crop_w,
            crop_h,
            interpolation=cv2.INTER_LINEAR
        )
    except Exception as error:
        print(f"Hard-edge completion SAM failed for {layer_name}: {error}")
        return mask, False

    coarse_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2] > 0.5
    coarse_area = max(1, int(np.count_nonzero(coarse_crop)))
    completion_radius = max(4, min(12, int(round(min(crop_h, crop_w) * 0.035))))
    completion_support = cv2.dilate(
        coarse_crop.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (completion_radius * 2 + 1, completion_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    best = None
    best_score = None
    for candidate in candidates:
        candidate_binary = candidate > 0.5
        candidate_area = int(np.count_nonzero(candidate_binary))
        overlap = int(np.count_nonzero(candidate_binary & coarse_crop))
        new_pixels = candidate_binary & (~coarse_crop)
        new_area = int(np.count_nonzero(new_pixels))
        outside_support_area = int(np.count_nonzero(new_pixels & (~completion_support)))
        preserve_ratio = overlap / coarse_area
        area_ratio = candidate_area / coarse_area
        inside_ratio = int(np.count_nonzero(candidate_binary[local_bbox[1]:local_bbox[3], local_bbox[0]:local_bbox[2]])) / max(1, candidate_area)
        new_ratio = new_area / coarse_area
        if (
            preserve_ratio < 0.82 or
            area_ratio < 1.01 or
            area_ratio > 1.12 or
            new_ratio > 0.12 or
            outside_support_area > 0 or
            inside_ratio < 0.90
        ):
            continue
        if not all(
            0 <= y < candidate_binary.shape[0] and
            0 <= x < candidate_binary.shape[1] and
            candidate_binary[y, x]
            for x, y in local_points
        ):
            continue
        score = preserve_ratio * 0.48 + min(1.12, area_ratio) * 0.22 + inside_ratio * 0.20 + min(0.12, new_ratio) * 0.10
        if best_score is None or score > best_score:
            best_score = score
            best = candidate_binary

    if best is None:
        print(
            f"Hard-edge completion for {layer_name}: rejected unsafe expansion "
            f"from {len(positive)} probes"
        )
        return mask, False

    recovered = mask.copy().astype(np.float32)
    recovered[crop_y1:crop_y2, crop_x1:crop_x2] = np.maximum(
        recovered[crop_y1:crop_y2, crop_x1:crop_x2],
        best.astype(np.float32)
    )
    recovered = constrain_mask_to_bbox(recovered, target_bbox)
    print(
        f"Hard-edge completion for {layer_name}: probes={len(positive)} "
        f"area={int(np.count_nonzero(mask > 0.5))}->{int(np.count_nonzero(recovered > 0.5))} "
        "accepted=edge_limited"
    )
    return recovered, True


def choose_local_refine_imgsz(crop_w, crop_h):
    return 1024


def crop_region_from_bbox(img, bbox, expand_ratio):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(x1, y1, x2, y2, w, h, ratio=expand_ratio)
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, [crop_x1, crop_y1, crop_x2, crop_y2]


def select_best_refine_mask(candidate_masks, coarse_mask_crop, local_bbox, strategy_type=None):
    if candidate_masks is None or len(candidate_masks) == 0:
        return None

    bbox_area_value = max(1, bbox_area(local_bbox))
    best_mask = None
    best_score = None
    coarse_binary = coarse_mask_crop > 0.5

    for mask in candidate_masks:
        mask_binary = mask > 0.5
        if not np.any(mask_binary):
            continue
        current_bbox = mask_bbox(mask_binary)
        if not current_bbox:
            continue
        overlap = int(np.count_nonzero(mask_binary & coarse_binary))
        union = int(np.count_nonzero(mask_binary | coarse_binary))
        iou = overlap / max(1, union)
        box_overlap = intersection_area(current_bbox, local_bbox) / bbox_area_value
        fill_ratio = int(np.count_nonzero(mask_binary[local_bbox[1]:local_bbox[3], local_bbox[0]:local_bbox[2]])) / bbox_area_value
        spill_penalty = max(0.0, (np.count_nonzero(mask_binary) - overlap) / max(1, np.count_nonzero(mask_binary)))
        if strategy_type == "food_product":
            score = (iou * 0.44) + (box_overlap * 0.24) + (fill_ratio * 0.26) - (spill_penalty * 0.10)
        else:
            score = (iou * 0.58) + (box_overlap * 0.26) + (fill_ratio * 0.18) - (spill_penalty * 0.22)
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask_binary.astype(np.float32)

    return best_mask


def refine_mask_with_local_sam(img, coarse_mask, target_bbox, cleanup_mask=None, strategy_type=None):
    coarse_binary = coarse_mask > 0.5
    coarse_bbox = mask_bbox(coarse_binary)
    if not coarse_bbox:
        return coarse_mask, False

    crop, crop_bounds = crop_region_from_bbox(img, coarse_bbox, LOCAL_REFINE_EXPAND_RATIO)
    crop_x1, crop_y1, _, _ = crop_bounds
    crop_h, crop_w = crop.shape[:2]
    if crop_h <= 2 or crop_w <= 2:
        return coarse_mask, False

    local_bbox = [
        coarse_bbox[0] - crop_x1,
        coarse_bbox[1] - crop_y1,
        coarse_bbox[2] - crop_x1,
        coarse_bbox[3] - crop_y1
    ]
    coarse_crop = coarse_binary[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
    cleanup_crop = None
    if cleanup_mask is not None:
        cleanup_crop = cleanup_mask[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
    points, labels, mask_prompt = build_refine_prompts(coarse_crop, local_bbox, cleanup_crop, strategy_type=strategy_type)
    refine_imgsz = (
        HARD_EDGE_SAM_IMGSZ
        if strategy_type in HARD_EDGE_STRATEGIES
        else choose_local_refine_imgsz(crop_w, crop_h)
    )

    try:
        refine_points = points
        refine_labels = labels
        if strategy_type == "food_product":
            # Keep food refinement mask-guided only to avoid prompt shape conflicts.
            refine_points = None
            refine_labels = None
        results = run_sam_mask_refine_inference(
            crop,
            imgsz=refine_imgsz,
            points=refine_points,
            labels=refine_labels,
            # Ultralytics 8.x preprocesses mask prompts to image size before
            # passing them to SAM. Point prompts avoid the incompatible
            # crop-size mask path while retaining the coarse mask as guidance.
            masks=None,
            multimask_output=True
        )
        candidate_masks = normalize_result_masks(results, crop_w, crop_h)
    except Exception as error:
        print(f"Local SAM refine failed: {error}")
        return coarse_mask, False

    refined_crop_mask = select_best_refine_mask(candidate_masks, coarse_crop.astype(np.float32), local_bbox, strategy_type=strategy_type)
    if refined_crop_mask is None:
        return coarse_mask, False

    coarse_area = max(1, int(np.count_nonzero(coarse_crop)))
    refined_area = int(np.count_nonzero(refined_crop_mask > 0.5))
    preserved_overlap = int(np.count_nonzero((refined_crop_mask > 0.5) & (coarse_crop > 0.5)))
    if strategy_type == "food_product":
        min_area_ratio = 0.78
        min_overlap_ratio = 0.72
    else:
        min_area_ratio = 0.62
        min_overlap_ratio = 0.60
    if refined_area < coarse_area * min_area_ratio or preserved_overlap < coarse_area * min_overlap_ratio:
        return coarse_mask, False

    refined_full = coarse_binary.astype(np.float32)
    refined_full[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w] = refined_crop_mask
    return refined_full.astype(np.float32), True


def build_food_support_bbox(target_bbox, img_w, img_h):
    x1, y1, x2, y2 = target_bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return [
        clamp(x1 - int(round(width * 0.08)), 0, img_w - 1),
        clamp(y1 + int(round(height * 0.40)), 0, img_h - 1),
        clamp(x2 + int(round(width * 0.08)), 1, img_w),
        clamp(y2 + int(round(height * 0.10)), 1, img_h)
    ]


def build_food_support_prompt_points(support_bbox, label_cleanup_mask=None):
    positive_points = sample_points_in_bbox(
        support_bbox,
        [
            (0.24, 0.34),
            (0.50, 0.42),
            (0.76, 0.34),
            (0.20, 0.72),
            (0.50, 0.80),
            (0.80, 0.72)
        ]
    )
    negative_points = []
    if label_cleanup_mask is not None and np.any(label_cleanup_mask):
        negative_points.extend(build_negative_points_from_mask(label_cleanup_mask > 0, max_points=2))

    prompt_points = []
    prompt_labels = []
    for point in positive_points:
        prompt_points.append([int(point[0]), int(point[1])])
        prompt_labels.append(1)
    for point in negative_points:
        prompt_points.append([int(point[0]), int(point[1])])
        prompt_labels.append(0)
    return [prompt_points], [prompt_labels]


def recover_food_support_mask(img, base_mask, target_bbox, label_cleanup_mask=None):
    base_binary = base_mask > 0.5
    if not np.any(base_binary):
        return base_mask, 0

    base_bbox = mask_bbox(base_binary)
    if not base_bbox:
        return base_mask, 0

    img_h, img_w = img.shape[:2]
    support_bbox = build_food_support_bbox(target_bbox, img_w, img_h)
    prompt_points, prompt_labels = build_food_support_prompt_points(support_bbox, label_cleanup_mask=label_cleanup_mask)

    try:
        results = run_sam_bbox_inference(
            img,
            support_bbox,
            multimask_output=True,
            points=prompt_points,
            labels=prompt_labels
        )
        candidate_masks = normalize_result_masks(results, img_w, img_h)
    except Exception as error:
        print(f"Food support recovery failed: {error}")
        return base_mask, 0

    if candidate_masks is None or len(candidate_masks) == 0:
        return base_mask, 0

    base_area = max(1, int(np.count_nonzero(base_binary)))
    target_area = max(1, bbox_area(target_bbox))
    dilated_base = cv2.dilate(base_binary.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1) > 0
    tx1, ty1, tx2, ty2 = target_bbox
    support_candidates = []
    target_height = max(1, ty2 - ty1)

    for mask in candidate_masks:
        mask_binary = mask > 0.5
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary.astype(np.uint8), connectivity=8)
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < 24:
                continue
            component = labels == label
            comp_bbox = [x, y, x + w, y + h]
            comp_shape = compute_shape_features(comp_bbox, target_bbox, int(area))
            if comp_bbox[1] < ty1 + int(round(target_height * 0.28)):
                continue
            if comp_shape["bottomBand"] < 0.64 or comp_shape["centerY"] < 0.50:
                continue

            comp_target_area = int(np.count_nonzero(component[ty1:ty2, tx1:tx2]))
            target_fill = comp_target_area / target_area
            inside_ratio = comp_target_area / max(1, int(area))
            if target_fill < 0.015 or inside_ratio < 0.42:
                continue

            label_ratio = 0.0
            if label_cleanup_mask is not None and np.any(label_cleanup_mask):
                label_ratio = int(np.count_nonzero(component & (label_cleanup_mask > 0))) / max(1, int(area))
            if label_ratio > 0.10:
                continue

            overlap_base = int(np.count_nonzero(component & dilated_base)) / max(1, int(area))
            horiz_overlap = horizontal_overlap_ratio(comp_bbox, base_bbox)
            distance = bbox_distance(comp_bbox, base_bbox)
            if overlap_base <= 0.03 and horiz_overlap < 0.22 and distance > 34:
                continue

            if (
                comp_shape["relativeWidth"] < 0.16 and
                not comp_shape["isHorizontalSurface"] and
                not comp_shape["isBlockLike"]
            ):
                continue

            score = (
                (overlap_base * 0.34) +
                (horiz_overlap * 0.18) +
                (target_fill * 0.18) +
                (comp_shape["relativeWidth"] * 0.16) +
                (0.10 if comp_shape["isHorizontalSurface"] else 0.0) +
                (0.08 if comp_shape["isBlockLike"] else 0.0) -
                (label_ratio * 0.40)
            )
            support_candidates.append({
                "mask": component,
                "score": score,
                "area": int(area)
            })

    if not support_candidates:
        return base_mask, 0

    support_candidates.sort(key=lambda item: item["score"], reverse=True)
    merged = base_binary.copy()
    added = 0
    max_added_area = int(base_area * 0.32)
    total_added_area = 0

    for candidate in support_candidates[:5]:
        component = candidate["mask"]
        new_pixels = component & (~merged)
        new_area = int(np.count_nonzero(new_pixels))
        if new_area <= 0:
            continue
        if total_added_area + new_area > max_added_area:
            continue
        merged |= component
        added += 1
        total_added_area += new_area

    if added == 0:
        return base_mask, 0

    return merged.astype(np.float32), added


def generate_alpha_matte(
    img,
    mask,
    target_bbox,
    cleanup_mask=None,
    strategy_type=None,
    label_cleanup_mask=None,
    flat_cleanup_mask=None
):
    # Table SAM masks are authoritative.  Do not let GrabCut reinterpret a
    # bright stone surface or a floor patch inside the semantic bbox.
    if strategy_type in {"table", "furniture"}:
        return generate_table_safe_matte(mask, target_bbox)

    if strategy_type in HARD_EDGE_STRATEGIES:
        return generate_hard_edge_matte(img, mask, target_bbox, strategy_type=strategy_type)

    binary_mask = (mask > 0.5).astype(np.uint8)
    if not np.any(binary_mask):
        return dilate_and_feather_mask(mask)

    # Keep SAM's soft-edge alpha for feathers, hair, smoke, clouds, and sheer
    # materials; GrabCut tends to convert their translucent strands to holes.
    if strategy_type == "soft_edge":
        return dilate_and_feather_mask(mask)

    object_bbox = mask_bbox(binary_mask)
    if not object_bbox:
        return dilate_and_feather_mask(mask)

    crop, crop_bounds = crop_region_from_bbox(img, object_bbox, 0.18)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_mask = binary_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0 or crop_mask.size == 0:
        return dilate_and_feather_mask(mask)

    kernel_size = max(1, compute_cleanup_kernel(target_bbox))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sure_fg = cv2.erode(crop_mask, kernel, iterations=1)
    sure_bg = 1 - cv2.dilate(crop_mask, kernel, iterations=2)

    gc_mask = np.full(crop_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[sure_bg > 0] = cv2.GC_BGD
    gc_mask[crop_mask > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    protected_core = sure_fg > 0
    label_crop = None
    flat_crop = None
    if label_cleanup_mask is not None:
        label_crop = label_cleanup_mask[crop_y1:crop_y2, crop_x1:crop_x2] > 0
    if flat_cleanup_mask is not None:
        flat_crop = flat_cleanup_mask[crop_y1:crop_y2, crop_x1:crop_x2] > 0
    if cleanup_mask is not None and label_crop is None and flat_crop is None:
        flat_crop = cleanup_mask[crop_y1:crop_y2, crop_x1:crop_x2] > 0

    if (label_crop is not None and np.any(label_crop)) or (flat_crop is not None and np.any(flat_crop)):
        core_kernel = np.ones((max(2, kernel_size + 1), max(2, kernel_size + 1)), np.uint8)
        protected_core = cv2.erode(crop_mask, core_kernel, iterations=2) > 0

        if label_crop is not None and np.any(label_crop):
            label_force_bg = label_crop & (crop_mask == 0)
            label_soft_bg = label_crop & (crop_mask > 0) & (~protected_core)
            gc_mask[label_force_bg] = cv2.GC_BGD
            gc_mask[label_soft_bg] = cv2.GC_PR_BGD

        if flat_crop is not None and np.any(flat_crop):
            flat_force_bg = flat_crop & (crop_mask == 0)
            flat_soft_bg = flat_crop & (crop_mask > 0) & (~protected_core)
            gc_mask[flat_force_bg] = cv2.GC_BGD
            gc_mask[flat_soft_bg] = cv2.GC_PR_BGD

        gc_mask[protected_core] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model, GRABCUT_ITER_COUNT, cv2.GC_INIT_WITH_MASK)
        fg = np.logical_or(gc_mask == cv2.GC_FGD, gc_mask == cv2.GC_PR_FGD).astype(np.uint8) * 255
    except Exception as error:
        print(f"GrabCut matting failed: {error}")
        return dilate_and_feather_mask(mask)

    fg = cv2.GaussianBlur(fg, (5, 5), 0)
    fg = np.maximum(fg, (sure_fg * 255).astype(np.uint8))

    if strategy_type == "food_product":
        object_ys, object_xs = np.where(crop_mask > 0)
        if len(object_xs) > 0 and len(object_ys) > 0:
            obj_y1 = int(object_ys.min())
            obj_y2 = int(object_ys.max()) + 1
            obj_h = max(1, obj_y2 - obj_y1)
            support_start_y = int(round(obj_y1 + obj_h * 0.52))
            row_grid = np.arange(crop_mask.shape[0])[:, None]
            support_band = (crop_mask > 0) & (row_grid >= support_start_y)
            if label_crop is not None and np.any(label_crop):
                support_band &= ~label_crop
            fg[support_band] = np.maximum(fg[support_band], 255)

    alpha = np.zeros(mask.shape, dtype=np.uint8)
    alpha[crop_y1:crop_y2, crop_x1:crop_x2] = fg
    return alpha

def mask_bbox(mask):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_center_inside(box, target):
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = target
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2

def bbox_distance(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx * dx + dy * dy) ** 0.5

def compute_shape_features(box, target_bbox, mask_area):
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = target_bbox
    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)
    target_width = max(1, tx2 - tx1)
    target_height = max(1, ty2 - ty1)
    aspect_ratio = bbox_width / bbox_height
    relative_width = bbox_width / target_width
    relative_height = bbox_height / target_height
    center_x = (((x1 + x2) / 2) - tx1) / target_width
    center_y = (((y1 + y2) / 2) - ty1) / target_height
    bottom_band = (y2 - ty1) / target_height
    rectangularity = mask_area / max(1, bbox_width * bbox_height)
    is_thin_vertical = (
        aspect_ratio <= THIN_VERTICAL_ASPECT_RATIO and
        relative_height >= THIN_VERTICAL_MIN_HEIGHT_RATIO
    )
    is_horizontal_surface = (
        aspect_ratio >= 1.8 and
        relative_width >= 0.25 and
        relative_height <= 0.45
    )
    is_table_support = (
        aspect_ratio <= TABLE_SUPPORT_MAX_ASPECT_RATIO and
        relative_width <= TABLE_SUPPORT_MAX_WIDTH_RATIO and
        relative_height >= TABLE_SUPPORT_MIN_HEIGHT_RATIO and
        bottom_band >= 0.35
    )
    is_block_like = (
        BLOCKLIKE_MIN_ASPECT_RATIO <= aspect_ratio <= BLOCKLIKE_MAX_ASPECT_RATIO and
        relative_width >= BLOCKLIKE_MIN_WIDTH_RATIO and
        relative_height >= BLOCKLIKE_MIN_HEIGHT_RATIO
    )
    is_rectangular_plane = (
        rectangularity >= MIN_RECTANGULARITY and
        0.35 <= aspect_ratio <= 3.8 and
        not is_thin_vertical
    )

    return {
        "aspectRatio": aspect_ratio,
        "bboxWidth": bbox_width,
        "bboxHeight": bbox_height,
        "relativeWidth": relative_width,
        "relativeHeight": relative_height,
        "isThinVertical": is_thin_vertical,
        "isHorizontalSurface": is_horizontal_surface,
        "isTableSupport": is_table_support,
        "isBlockLike": is_block_like,
        "isRectangularPlane": is_rectangular_plane,
        "rectangularity": rectangularity,
        "bottomBand": bottom_band,
        "centerX": center_x,
        "centerY": center_y
    }

def round_shape_features(features):
    return {
        "aspectRatio": round(float(features["aspectRatio"]), 3),
        "bboxWidth": int(features["bboxWidth"]),
        "bboxHeight": int(features["bboxHeight"]),
        "relativeWidth": round(float(features["relativeWidth"]), 3),
        "relativeHeight": round(float(features["relativeHeight"]), 3),
        "isThinVertical": bool(features["isThinVertical"]),
        "isHorizontalSurface": bool(features["isHorizontalSurface"]),
        "isTableSupport": bool(features["isTableSupport"]),
        "isBlockLike": bool(features["isBlockLike"]),
        "isRectangularPlane": bool(features["isRectangularPlane"]),
        "rectangularity": round(float(features["rectangularity"]), 3),
        "bottomBand": round(float(features["bottomBand"]), 3),
        "centerX": round(float(features["centerX"]), 3),
        "centerY": round(float(features["centerY"]), 3)
    }

def shape_strategy_gate(shape_features, strategy):
    strategy_type = strategy.get("type")
    if strategy_type == "wall_art":
        return shape_features["isRectangularPlane"], "shape_not_rectangular_plane"
    if strategy_type == "furniture":
        if shape_features["isThinVertical"]:
            return False, "thin_vertical_furniture"
        return True, ""
    if strategy_type == "table":
        if shape_features["isThinVertical"] or shape_features["isHorizontalSurface"]:
            return True, ""
        return True, ""
    return True, ""

def has_close_bottom_band(a, b, tolerance=BOTTOM_BAND_TOLERANCE):
    return abs(a["bottomBand"] - b["bottomBand"]) <= tolerance

def horizontal_overlap_ratio(a, b):
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    return overlap / max(1, min(ax2 - ax1, bx2 - bx1))

def is_decor_base_shape(shape_features):
    return (
        shape_features["centerY"] >= 0.48 and
        shape_features["bottomBand"] >= 0.65 and
        shape_features["relativeHeight"] >= 0.18 and
        (
            shape_features["isBlockLike"] or
            shape_features["aspectRatio"] <= 1.35
        )
    )

def is_food_support_shape(shape_features):
    return bool(
        shape_features.get("isBlockLike") or
        shape_features.get("isHorizontalSurface") or
        shape_features.get("isRectangularPlane")
    )

def build_quality_gate(score, primary_score, target_fill_ratio, selected, strategy_type, high_coverage_entity=False):
    min_score = MIN_RUNTIME_ACCEPT_SCORE
    min_primary_score = MIN_RUNTIME_ACCEPT_PRIMARY_SCORE
    min_fill_ratio = MIN_RUNTIME_ACCEPT_FILL_RATIO
    max_fill_ratio = MAX_RUNTIME_ACCEPT_FILL_RATIO

    if strategy_type == "food_product":
        min_score = 0.28
        min_primary_score = 0.22
        min_fill_ratio = 0.05
        max_fill_ratio = 0.82
    elif strategy_type == "wall_art":
        # A framed painting is a bounded opaque plane. Its valid silhouette may
        # cover nearly all of its semantic bbox, unlike room/background masks.
        max_fill_ratio = 0.98
    if high_coverage_entity:
        # The candidate has already passed the structural high-coverage test;
        # do not let the generic runtime fill gate contradict that decision.
        max_fill_ratio = max(max_fill_ratio, 0.98)
        # The final runtime score is still based on the generic 0.42 fill
        # target. Structural validation is the stronger signal for this path.
        min_score = min(min_score, 0.30)

    issues = []
    if not selected:
        issues.append("no_selected_mask")
    if score < min_score:
        issues.append("low_quality_score")
    if primary_score < min_primary_score:
        issues.append("low_primary_score")
    if target_fill_ratio < min_fill_ratio:
        issues.append("low_target_fill")
    if target_fill_ratio > max_fill_ratio:
        issues.append("high_target_fill")

    should_generate_runtime_layer = len(issues) == 0
    return {
        "status": "ok" if should_generate_runtime_layer else "low_quality",
        "runtimeAction": "accept" if should_generate_runtime_layer else "hold",
        "shouldGenerateRuntimeLayer": should_generate_runtime_layer,
        "needsHigherPrecision": not should_generate_runtime_layer,
        "issues": issues,
        "recommendedEngine": "fastsam_multi_mask" if should_generate_runtime_layer else (
            "matting_or_hq_sam" if strategy_type in ["furniture", "table"] else "hq_sam"
        )
    }

def same_layer(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return (
        (a.get("id") and a.get("id") == b.get("id")) or
        (a.get("name") and a.get("name") == b.get("name"))
    )

def get_sibling_exclusion_family(layer):
    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", ""))
    ]).lower()

    if any(token in text for token in ["chair", "sofa", "seat", "stool", "沙发", "椅", "凳"]):
        return "furniture_peer"
    if any(token in text for token in ["wall art", "painting", "artwork", "picture", "poster", "挂画", "画", "装饰画"]):
        return "wall_art_peer"
    if any(token in text for token in ["lamp", "chandelier", "pendant", "lighting", "吊灯", "灯"]):
        return "lighting_peer"
    if any(token in text for token in [
        "product_food", "product_drink", "food", "dish", "meal", "plate", "rice",
        "pork", "cola", "tea", "coffee", "drink", "beverage", "食物", "食品",
        "菜品", "餐盘", "炒饭", "猪肉", "饮料", "可乐", "茶", "咖啡"
    ]):
        return "product_peer"
    if any(token in text for token in ["table", "desk", "coffee table", "茶几", "桌"]):
        return "surface_or_support"
    if any(token in text for token in [
        "vase", "flower", "bouquet", "plant", "potted", "bowl", "sculpture",
        "ornament", "column", "stacked", "decor", "花瓶", "花", "植物", "盆栽",
        "碗", "摆件", "雕塑", "柱状", "装饰"
    ]):
        return "nested_decor"
    return "other"

def should_strong_exclude_sibling(layer_meta, other):
    family = get_sibling_exclusion_family(layer_meta)
    other_family = get_sibling_exclusion_family(other)
    strong_peer_families = {"furniture_peer", "wall_art_peer", "lighting_peer"}
    return family == other_family and family in strong_peer_families

def normalize_context_bbox_to_pixel(bbox, img_w, img_h):
    ymin_n, xmin_n, ymax_n, xmax_n = bbox
    return [
        int((xmin_n / 1000.0) * img_w),
        int((ymin_n / 1000.0) * img_h),
        int((xmax_n / 1000.0) * img_w),
        int((ymax_n / 1000.0) * img_h)
    ]

def expand_target_bbox_for_cleanup(target_bbox, img_w, img_h, ratio=0.14, min_pixels=8):
    x1, y1, x2, y2 = target_bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(min_pixels, int(round(box_w * ratio)))
    pad_y = max(min_pixels, int(round(box_h * ratio)))
    return [
        clamp(x1 - pad_x, 0, img_w - 1),
        clamp(y1 - pad_y, 0, img_h - 1),
        clamp(x2 + pad_x, 1, img_w),
        clamp(y2 + pad_y, 1, img_h)
    ]

def build_exclude_bboxes(layer_meta, context_layers, target_bbox, img_w, img_h):
    excludes = []
    if not isinstance(context_layers, list):
        return excludes

    for other in context_layers:
        bbox = other.get("bbox") if isinstance(other, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if same_layer(layer_meta, other):
            continue

        other_bbox = normalize_context_bbox_to_pixel(bbox, img_w, img_h)
        if intersection_area(other_bbox, target_bbox) <= 0:
            continue
        same_parent = (
            layer_meta.get("parentLayerId") and
            other.get("parentLayerId") and
            layer_meta.get("parentLayerId") == other.get("parentLayerId")
        )
        excludes.append({
            "bbox": other_bbox,
            "strong": bool(same_parent and should_strong_exclude_sibling(layer_meta, other))
        })

    return excludes

def get_exclude_bbox(entry):
    return entry.get("bbox") if isinstance(entry, dict) else entry

def is_strong_exclude(entry):
    return bool(entry.get("strong")) if isinstance(entry, dict) else False

def count_mask_in_bboxes(mask_binary, bboxes):
    total = 0
    for entry in bboxes:
        x1, y1, x2, y2 = get_exclude_bbox(entry)
        total += int(np.count_nonzero(mask_binary[y1:y2, x1:x2]))
    return total

def build_exclude_mask(bboxes, img_w, img_h):
    exclude_mask = np.zeros((img_h, img_w), dtype=bool)
    for entry in bboxes:
        x1, y1, x2, y2 = get_exclude_bbox(entry)
        exclude_mask[y1:y2, x1:x2] = True
    return exclude_mask


def is_non_subject_layout_layer(layer):
    if not isinstance(layer, dict):
        return False
    extraction_profile = str(layer.get("extractionProfile", "")).lower()
    if extraction_profile in {"text_layer", "vector_layout_element"}:
        return True
    return is_flat_ad_cleanup_layer(layer) or is_food_label_like_layer(layer)


def is_food_foreign_sibling_layer(layer):
    """Return whether a nearby layer can be foreign to a food product.

    This is intentionally structural: it does not depend on labels such as
    "price", "badge", or "panel". Other extracted products are excluded here
    because they are independent semantic subjects, not poster decoration.
    """
    if not isinstance(layer, dict):
        return False

    extraction_profile = str(layer.get("extractionProfile", "")).lower()
    if extraction_profile in {"text_layer", "vector_layout_element", "background_plate"}:
        return True

    strategy_type = get_layer_strategy(layer).get("type")
    if strategy_type in {"flat_shape", "wall_art"}:
        return True
    return strategy_type not in {
        "food_product", "hard_product", "furniture", "table", "lighting",
        "decor_arrangement", "decor_atomic", "soft_edge"
    }


def attached_layout_priority(entry, target_bbox):
    layer = entry.get("layer") or {}
    bbox = entry.get("bbox")
    if not bbox:
        return entry.get("score", 0.0)

    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", "")),
        str(layer.get("extractionProfile", ""))
    ]).lower()

    tx1, ty1, tx2, ty2 = target_bbox
    target_area = max(1, bbox_area(target_bbox))
    area_ratio = bbox_area(bbox) / target_area
    center_y = entry.get("centerY", 0.5)

    score = float(entry.get("score", 0.0))
    if is_price_like_text(text):
        score += 3.2
    if any(token in text for token in ["text", "文字", "文本", "caption", "label", "tag", "$"]):
        score += 1.2
    if any(token in text for token in ["card", "panel", "背景", "底板", "卡片背景", "panel", "shape_panel"]):
        score += 0.8
    if any(token in text for token in ["background", "ad_background", "波纹背景", "橙色波纹背景"]):
        score -= 2.4
    if center_y <= 0.36 and area_ratio >= 0.18:
        score -= 1.8
    if area_ratio >= 0.55:
        score -= 2.6
    elif area_ratio >= 0.35:
        score -= 1.2
    return score


def collect_attached_layout_entries(layer_meta, context_layers, target_bbox, img_w, img_h):
    entries = []
    if not isinstance(context_layers, list):
        return entries

    expanded_target_bbox = expand_target_bbox_for_cleanup(target_bbox, img_w, img_h, ratio=0.18, min_pixels=10)
    tx1, ty1, tx2, ty2 = target_bbox
    target_area = max(1, bbox_area(target_bbox))

    for other in context_layers:
        if same_layer(layer_meta, other) or not is_non_subject_layout_layer(other):
            continue
        bbox = other.get("bbox") if isinstance(other, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        other_bbox = normalize_context_bbox_to_pixel(bbox, img_w, img_h)
        overlap = intersection_area(other_bbox, expanded_target_bbox)
        if overlap <= 0:
            continue

        ox1, oy1, ox2, oy2 = other_bbox
        other_area = max(1, bbox_area(other_bbox))
        overlap_ratio = overlap / other_area
        center_y = ((oy1 + oy2) / 2 - ty1) / max(1, ty2 - ty1)
        distance = bbox_distance(other_bbox, target_bbox)
        score = (overlap / target_area) + (overlap_ratio * 0.8) - (distance / max(40.0, target_area ** 0.5))

        entries.append({
            "layer": other,
            "bbox": other_bbox,
            "score": score,
            "centerY": center_y
        })

    entries.sort(key=lambda item: attached_layout_priority(item, target_bbox), reverse=True)
    return entries


def collect_food_layout_conflict_entries(layer_meta, context_layers, target_bbox, img_w, img_h):
    """Collect nearby food-ad graphics, including panels just outside the food bbox."""
    entries = []
    if not isinstance(context_layers, list):
        return entries

    tx1, ty1, tx2, ty2 = target_bbox
    target_area = max(1, bbox_area(target_bbox))
    target_diagonal = max(1.0, target_area ** 0.5)
    # A panel can be outside the semantic food bbox while its visible edge is
    # only a few pixels away. The ordinary 18% attached-layout scope misses it.
    scope = expand_target_bbox_for_cleanup(target_bbox, img_w, img_h, ratio=0.68, min_pixels=20)

    for other in context_layers:
        if same_layer(layer_meta, other) or not is_food_foreign_sibling_layer(other):
            continue
        bbox = other.get("bbox") if isinstance(other, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        other_bbox = normalize_context_bbox_to_pixel(bbox, img_w, img_h)
        if intersection_area(other_bbox, scope) <= 0:
            continue

        other_area = max(1, bbox_area(other_bbox))
        overlap = intersection_area(other_bbox, target_bbox)
        overlap_ratio = overlap / other_area
        distance = bbox_distance(other_bbox, target_bbox)
        if distance > max(28.0, target_diagonal * 0.72):
            continue

        ox1, oy1, ox2, oy2 = other_bbox
        center_y = ((oy1 + oy2) / 2 - ty1) / max(1, ty2 - ty1)
        score = (overlap / target_area) + (overlap_ratio * 0.8) - (distance / max(40.0, target_diagonal))
        entries.append({
            "layer": other,
            "bbox": other_bbox,
            "score": score,
            "centerY": center_y,
            "foodConflictScope": True
        })

    # Metadata only narrows the nearby sibling set. The real SAM mask and its
    # overlap with the subject decide whether this layer is actually foreign.
    entries.sort(key=lambda item: item["score"], reverse=True)
    return entries


def order_food_foreign_sibling_entries(entries):
    """Process local containers before their child text/icon layers.

    A real mask for a container can later be reused as ownership evidence for
    its child layers, avoiding repeated SAM calls for the same graphic.
    """
    def contained_children(entry):
        outer_bbox = entry.get("bbox")
        if not outer_bbox:
            return 0
        count = 0
        for other in entries:
            if other is entry or not other.get("bbox"):
                continue
            inner_area = max(1, bbox_area(other["bbox"]))
            if intersection_area(outer_bbox, other["bbox"]) / inner_area >= 0.82:
                count += 1
        return count

    return sorted(
        entries,
        key=lambda entry: (
            -contained_children(entry),
            bbox_area(entry.get("bbox") or [0, 0, 0, 0]),
            -float(entry.get("score", 0.0))
        )
    )


def entry_bbox_is_covered_by_foreign_masks(entry_bbox, verified_foreign_masks):
    """Return true when a child layer already belongs to a verified sibling."""
    if not entry_bbox:
        return False
    x1, y1, x2, y2 = entry_bbox
    region_area = max(1, bbox_area(entry_bbox))
    for mask in verified_foreign_masks:
        coverage = int(np.count_nonzero(mask[y1:y2, x1:x2])) / region_area
        if coverage >= 0.72:
            return True
    return False


def sibling_touches_or_overlaps_target(entry_bbox, target_bbox):
    """Keep only siblings that touch the original target box or its edge."""
    if not entry_bbox or not target_bbox:
        return False
    if intersection_area(entry_bbox, target_bbox) > 0:
        return True

    target_width = max(1, target_bbox[2] - target_bbox[0])
    target_height = max(1, target_bbox[3] - target_bbox[1])
    edge_tolerance = max(10.0, min(target_width, target_height) * 0.10)
    return bbox_distance(entry_bbox, target_bbox) <= edge_tolerance


def segment_attached_layout_mask(img, entry, context_layers):
    bbox = entry.get("bbox")
    layer = entry.get("layer") or {}
    if not bbox or len(bbox) != 4:
        return None, None

    img_h, img_w = img.shape[:2]
    prompt_bbox = expand_bbox(bbox[0], bbox[1], bbox[2], bbox[3], img_w, img_h, ratio=0.08)
    try:
        results = run_sam_bbox_inference(
            img,
            prompt_bbox,
            multimask_output=True
        )
    except Exception as error:
        print(f"Attached layout segmentation failed for {layer.get('name') or 'unknown'}: {error}")
        return None, None

    candidate_masks = normalize_result_masks(results, img_w, img_h)
    if candidate_masks is None or len(candidate_masks) == 0:
        return None, None

    layout_mask, _, quality = select_and_merge_masks(
        candidate_masks,
        prompt_bbox,
        img_w,
        img_h,
        layer,
        context_layers
    )
    return layout_mask, quality


def build_layout_bbox_mask(mask_shape, bbox, is_price_like=False, is_text_like=False):
    h, w = mask_shape
    x1, y1, x2, y2 = bbox
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 1, w)
    y2 = clamp(y2, 1, h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=bool)

    mask = np.zeros((h, w), dtype=np.uint8)
    if is_price_like:
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        rx = max(2, int(round((x2 - x1) * 0.62)))
        ry = max(2, int(round((y2 - y1) * 0.62)))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    elif is_text_like:
        pad_x = max(1, int(round((x2 - x1) * 0.10)))
        pad_y = max(1, int(round((y2 - y1) * 0.16)))
        rx1 = clamp(x1 - pad_x, 0, w - 1)
        ry1 = clamp(y1 - pad_y, 0, h - 1)
        rx2 = clamp(x2 + pad_x, 1, w)
        ry2 = clamp(y2 + pad_y, 1, h)
        mask[ry1:ry2, rx1:rx2] = 255
    else:
        mask[y1:y2, x1:x2] = 255
    return mask > 0


def merge_bboxes(a, b):
    if not a:
        return b
    if not b:
        return a
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3])
    ]


def build_attached_layout_entry_mask(mask_shape, entry):
    layer = entry.get("layer") or {}
    bbox = entry.get("bbox")
    if not bbox:
        return np.zeros(mask_shape, dtype=bool)

    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", "")),
        str(layer.get("extractionProfile", ""))
    ]).lower()
    is_price_like = is_price_like_text(text)
    is_text_like = any(token in text for token in ["text", "文字", "文本", "caption", "label", "tag", "$"])
    return build_layout_bbox_mask(
        mask_shape,
        bbox,
        is_price_like=is_price_like,
        is_text_like=(is_text_like and not is_price_like)
    )


def is_broad_background_entry(entry):
    layer = entry.get("layer") or {}
    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", "")),
        str(layer.get("extractionProfile", ""))
    ]).lower()
    return any(token in text for token in [
        "background_master", "ad_background", "波纹背景", "橙色波纹背景", "橙色渐变背景", "渐变背景"
    ])


def get_food_conflict_kind(entry):
    """Limit food refinement to price graphics and explicit local panels."""
    layer = entry.get("layer") or {}
    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", "")),
        str(layer.get("designRole", "")),
        str(layer.get("extractionProfile", ""))
    ]).lower()
    if is_price_like_text(text):
        return "price"
    if any(token in text for token in [
        "shape_panel", "panel", "card background", "background_plate",
        "板块背景", "背景板", "底板", "卡片背景", "面板"
    ]):
        return "panel"
    return None


def is_food_price_badge_entry(entry):
    """A badge has its own visible shape; price text alone must never be subtracted."""
    layer = entry.get("layer") or {}
    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", "")),
        str(layer.get("designRole", "")),
        str(layer.get("extractionProfile", ""))
    ]).lower()
    return any(token in text for token in [
        "price_badge", "price badge", "badge_background", "price background",
        "标签背景", "价格标签背景", "价签背景", "价格圆圈", "圆形价格"
    ])


def collect_food_conflict_entries_for_mask(food_mask, layer_meta, context_layers, target_bbox, img_w, img_h, max_entries=3):
    entries = collect_food_layout_conflict_entries(
        layer_meta or {},
        context_layers or [],
        target_bbox,
        img_w,
        img_h
    )
    if not entries:
        return []

    food_binary = food_mask > 0.5
    conflicts = []
    for entry in entries:
        if is_broad_background_entry(entry):
            continue

        entry_mask = build_attached_layout_entry_mask(food_binary.shape, entry)
        if not np.any(entry_mask):
            continue

        overlap_area = int(np.count_nonzero(food_binary & entry_mask))

        entry_area = max(1, int(np.count_nonzero(entry_mask)))
        overlap_ratio = overlap_area / entry_area
        entry_distance = bbox_distance(entry.get("bbox"), target_bbox)
        if overlap_ratio < 0.005 and entry_distance > max(24.0, bbox_area(target_bbox) ** 0.5 * 0.55):
            continue

        entry_area_ratio = entry_area / max(1, bbox_area(target_bbox))
        if entry_area_ratio >= 3.5 and not is_price_like_text(" ".join(str(value) for value in (entry.get("layer") or {}).values())):
            continue

        conflicts.append({
            **entry,
            "entryMask": entry_mask,
            "overlapArea": overlap_area,
            "overlapRatio": overlap_ratio
        })

    conflicts.sort(
        key=lambda item: (
            item.get("overlapArea", 0),
            item.get("overlapRatio", 0.0),
            attached_layout_priority(item, target_bbox)
        ),
        reverse=True
    )
    return conflicts[:max(8, max_entries)]


def build_food_conflict_positive_points(mask_binary, avoid_mask=None):
    local_h, local_w = mask_binary.shape[:2]
    bbox = [0, 0, local_w, local_h]
    positive_points = build_food_positive_points_from_mask(mask_binary, bbox) or []
    if avoid_mask is not None and np.any(avoid_mask):
        filtered = []
        for point in positive_points:
            px, py = int(point[0]), int(point[1])
            if 0 <= py < local_h and 0 <= px < local_w and avoid_mask[py, px]:
                continue
            filtered.append([px, py])
        positive_points = filtered

    if not positive_points:
        positive_points = build_positive_points_from_mask(mask_binary, bbox) or sample_points_in_bbox(bbox, [(0.5, 0.5)])
    return positive_points


def score_food_conflict_refine_candidate(candidate_mask, coarse_mask, conflict_mask):
    candidate_binary = candidate_mask > 0.5
    coarse_binary = coarse_mask > 0.5
    conflict_binary = conflict_mask > 0.5
    if not np.any(candidate_binary):
        return None

    coarse_area = max(1, int(np.count_nonzero(coarse_binary)))
    candidate_area = int(np.count_nonzero(candidate_binary))
    preserved_overlap = int(np.count_nonzero(candidate_binary & coarse_binary))
    coarse_conflict = int(np.count_nonzero(coarse_binary & conflict_binary))
    candidate_conflict = int(np.count_nonzero(candidate_binary & conflict_binary))

    preserved_ratio = preserved_overlap / coarse_area
    area_ratio = candidate_area / coarse_area
    conflict_keep_ratio = candidate_conflict / max(1, coarse_conflict) if coarse_conflict > 0 else 0.0
    conflict_removed_ratio = 1.0 - conflict_keep_ratio if coarse_conflict > 0 else 0.0
    spill_ratio = max(0.0, (candidate_area - preserved_overlap) / max(1, candidate_area))

    score = (
        (preserved_ratio * 0.58) +
        (conflict_removed_ratio * 0.34) +
        (min(1.0, area_ratio) * 0.10) -
        (spill_ratio * 0.16)
    )

    return {
        "score": score,
        "preservedRatio": preserved_ratio,
        "areaRatio": area_ratio,
        "conflictRemovedRatio": conflict_removed_ratio,
        "candidateConflict": candidate_conflict,
        "coarseConflict": coarse_conflict
    }


def build_food_layout_core_mask(mask_binary, entry_bbox):
    """Keep only the reliable interior of a separately segmented layout layer."""
    mask_u8 = (mask_binary > 0.5).astype(np.uint8)
    if not np.any(mask_u8):
        return np.zeros_like(mask_u8, dtype=bool)

    x1, y1, x2, y2 = entry_bbox
    shortest_side = max(1, min(x2 - x1, y2 - y1))
    radius = max(1, min(5, int(round(shortest_side * 0.055))))
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    core = cv2.erode(mask_u8, kernel, iterations=1) > 0

    # A very small text or badge can vanish during erosion. It is safer to skip
    # that ambiguous layout layer than to fall back to its rectangular bbox.
    minimum_area = max(12, int(np.count_nonzero(mask_u8) * 0.08))
    return core if int(np.count_nonzero(core)) >= minimum_area else np.zeros_like(core, dtype=bool)


def select_food_layout_mask_candidate(candidate_masks, entry_bbox, img_w, img_h):
    """Choose the SAM mask that is most contained by an independent layout layer."""
    x1, y1, x2, y2 = entry_bbox
    region = np.zeros((img_h, img_w), dtype=bool)
    region[y1:y2, x1:x2] = True
    core_bbox = shrink_bbox(entry_bbox, ratio=0.22)
    cx1, cy1, cx2, cy2 = core_bbox
    core_region = np.zeros((img_h, img_w), dtype=bool)
    core_region[cy1:cy2, cx1:cx2] = True

    best_mask = None
    best_metrics = None
    region_area = max(1, int(np.count_nonzero(region)))
    core_area = max(1, int(np.count_nonzero(core_region)))
    for candidate in candidate_masks:
        candidate_binary = candidate > 0.5
        candidate_area = int(np.count_nonzero(candidate_binary))
        if candidate_area == 0:
            continue

        inside_area = int(np.count_nonzero(candidate_binary & region))
        core_coverage = int(np.count_nonzero(candidate_binary & core_region)) / core_area
        containment = inside_area / candidate_area
        region_coverage = inside_area / region_area
        score = (core_coverage * 0.52) + (containment * 0.38) + (region_coverage * 0.10)
        metrics = {
            "score": score,
            "coreCoverage": core_coverage,
            "containment": containment,
            "regionCoverage": region_coverage
        }
        if core_coverage < 0.52 or containment < 0.54:
            continue
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            # The bbox only bounds the independently segmented layout evidence.
            # It is never subtracted from the food mask directly.
            best_mask = (candidate_binary & region).astype(np.float32)
            best_metrics = metrics

    return best_mask, best_metrics


def segment_food_layout_conflict_mask(img, entry):
    """Extract a real mask for one foreign sibling before using it as evidence."""
    img_h, img_w = img.shape[:2]
    entry_bbox = entry.get("bbox")
    layer = entry.get("layer") or {}
    if not entry_bbox:
        return None, None

    prompt_bbox = expand_bbox(*entry_bbox, img_w, img_h, ratio=0.06)
    x1, y1, x2, y2 = entry_bbox
    center = [[int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]]
    try:
        results = run_sam_bbox_inference(
            img,
            prompt_bbox,
            multimask_output=True,
            imgsz=1024,
            points=[center],
            labels=[[1]]
        )
        candidate_masks = normalize_result_masks(results, img_w, img_h)
    except Exception as error:
        print(f"Food semantic layout mask failed for {layer.get('name') or 'unknown'}: {error}")
        return None, None

    layout_mask, metrics = select_food_layout_mask_candidate(
        candidate_masks,
        entry_bbox,
        img_w,
        img_h
    )
    if layout_mask is None:
        return None, None

    core_mask = build_food_layout_core_mask(layout_mask, entry_bbox)
    if not np.any(core_mask):
        return None, None
    # Keep the independently segmented visible contour for the narrowly scoped
    # price-badge fallback. The eroded core remains the only mask used for SAM
    # negative prompts.
    return core_mask, {**metrics, "layoutMask": layout_mask > 0.5}


def sample_food_interior_points(mask_binary, max_points, min_distance=14):
    """Sample spatially separated, high-confidence points from the interior of a mask."""
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return []

    distances = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    points = []
    for _ in range(max_points):
        _, max_value, _, max_location = cv2.minMaxLoc(distances)
        if max_value <= 0:
            break
        px, py = int(max_location[0]), int(max_location[1])
        points.append([px, py])
        cv2.circle(distances, (px, py), max(2, min_distance), 0, thickness=-1)
    return points


def build_food_support_region(mask_binary, target_bbox, conflict_mask=None):
    """Find lower, horizontally supported subject pixels such as a plate rim."""
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return np.zeros_like(mask_u8, dtype=bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_u8, dtype=bool)

    components = []
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area <= 0:
            continue
        components.append({
            "label": label,
            "bbox": [int(x), int(y), int(x + width), int(y + height)],
            "area": int(area),
            "width": int(width),
            "height": int(height)
        })
    if not components:
        return np.zeros_like(mask_u8, dtype=bool)

    components.sort(key=lambda item: item["area"], reverse=True)
    primary = components[0]
    px1, py1, px2, py2 = primary["bbox"]
    tx1, ty1, tx2, ty2 = target_bbox
    target_width = max(1, tx2 - tx1)
    target_height = max(1, ty2 - ty1)
    primary_support = np.zeros_like(mask_u8, dtype=bool)

    # The bottom 28% of the main food component is where a visible plate rim
    # or tray edge can survive as a thin, disconnected region.
    primary_lower_y = max(py1, int(round(py1 + (py2 - py1) * 0.68)))
    primary_support = (labels == primary["label"]) & (np.indices(mask_u8.shape)[0] >= primary_lower_y)
    support = primary_support.copy()

    primary_width = max(1, px2 - px1)
    for component in components[1:]:
        x1, y1, x2, y2 = component["bbox"]
        center_y = ((y1 + y2) / 2 - ty1) / target_height
        horizontal_overlap = max(0, min(x2, px2) - max(x1, px1))
        overlap_ratio = horizontal_overlap / max(1, min(component["width"], primary_width))
        component_mask = labels == component["label"]
        component_lower_y = ((y2 - ty1) / target_height)
        component_aspect = component["width"] / max(1, component["height"])
        if (
            component["width"] >= target_width * 0.14 and
            component_aspect >= 1.25 and
            center_y >= 0.48 and
            component_lower_y >= 0.62 and
            overlap_ratio >= 0.42
        ):
            support |= component_mask

    # Keep a thin rim even when it is only a few pixels thick. Conflict cores
    # still have absolute priority and cannot become positive subject anchors.
    if conflict_mask is not None:
        support &= ~(conflict_mask > 0)
    return support


def build_food_support_points(support_mask, target_bbox, max_points=5):
    """Choose positive points along a lower support contour, not only thick food interiors."""
    ys, xs = np.where(support_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return []

    x1, y1, x2, y2 = target_bbox
    points = []
    # Use horizontal bins so a plate arc is represented across its width.
    for fraction in np.linspace(0.18, 0.82, max_points):
        target_x = x1 + (x2 - x1) * float(fraction)
        distance = np.abs(xs - target_x)
        candidate_indices = np.argsort(distance)[:max(8, len(xs) // 18)]
        if len(candidate_indices) == 0:
            continue
        # Prefer the lower visible contour in each bin, then move one pixel
        # inward when possible so the prompt remains on the subject.
        candidate_indices = candidate_indices[np.argsort(ys[candidate_indices])[::-1]]
        px = int(xs[candidate_indices[0]])
        py = int(ys[candidate_indices[0]])
        points.append([px, py])

    deduped = []
    seen = set()
    for point in points:
        key = (point[0], point[1])
        if key not in seen:
            seen.add(key)
            deduped.append(point)
    return deduped


def score_food_semantic_refine_candidate(
    candidate_mask,
    coarse_mask,
    safe_subject_mask,
    conflict_core_mask,
    positive_points,
    negative_points,
    support_mask=None,
    support_points=None
):
    candidate_binary = candidate_mask > 0.5
    if not np.any(candidate_binary):
        return None

    coarse_binary = coarse_mask > 0.5
    candidate_area = int(np.count_nonzero(candidate_binary))
    coarse_area = max(1, int(np.count_nonzero(coarse_binary)))
    safe_area = max(1, int(np.count_nonzero(safe_subject_mask)))
    conflict_area = max(1, int(np.count_nonzero(conflict_core_mask)))
    support_area = max(1, int(np.count_nonzero(support_mask))) if support_mask is not None and np.any(support_mask) else 1

    positive_coverage = sum(
        1 for px, py in positive_points
        if 0 <= py < candidate_binary.shape[0] and 0 <= px < candidate_binary.shape[1] and candidate_binary[py, px]
    ) / max(1, len(positive_points))
    negative_hit_ratio = sum(
        1 for px, py in negative_points
        if 0 <= py < candidate_binary.shape[0] and 0 <= px < candidate_binary.shape[1] and candidate_binary[py, px]
    ) / max(1, len(negative_points)) if negative_points else 0.0
    safe_preserved_ratio = int(np.count_nonzero(candidate_binary & safe_subject_mask)) / safe_area
    conflict_keep_ratio = int(np.count_nonzero(candidate_binary & conflict_core_mask)) / conflict_area
    support_preserved_ratio = (
        int(np.count_nonzero(candidate_binary & support_mask)) / support_area
        if support_mask is not None and np.any(support_mask) else 1.0
    )
    support_point_coverage = (
        sum(
            1 for px, py in (support_points or [])
            if 0 <= py < candidate_binary.shape[0] and 0 <= px < candidate_binary.shape[1] and candidate_binary[py, px]
        ) / max(1, len(support_points))
        if support_points else 1.0
    )
    area_ratio = candidate_area / coarse_area
    area_penalty = min(0.22, abs(1.0 - area_ratio) * 0.22)
    score = (
        (positive_coverage * 0.34) +
        (safe_preserved_ratio * 0.25) +
        ((1.0 - conflict_keep_ratio) * 0.25) +
        (support_preserved_ratio * 0.10) +
        (support_point_coverage * 0.06) -
        (negative_hit_ratio * 0.18) -
        area_penalty
    )
    return {
        "score": score,
        "positiveCoverage": positive_coverage,
        "negativeHitRatio": negative_hit_ratio,
        "safePreservedRatio": safe_preserved_ratio,
        "conflictKeepRatio": conflict_keep_ratio,
        "supportPreservedRatio": support_preserved_ratio,
        "supportPointCoverage": support_point_coverage,
        "areaRatio": area_ratio
    }


def refine_food_mask_stage_with_sam(img, current_mask, target_bbox, conflict_cores, stage_name="foreign_sibling"):
    """Run one monotonic SAM cleanup stage and retain the input on rejection."""
    img_h, img_w = img.shape[:2]
    current_binary = current_mask > 0.5
    conflict_mask = np.logical_or.reduce(conflict_cores)
    guarded_conflicts = cv2.dilate(
        conflict_mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1
    ) > 0
    safe_subject = current_binary & (~guarded_conflicts)
    if int(np.count_nonzero(safe_subject)) < 32:
        return current_mask, False, "insufficient_safe_subject"

    # Build plate/tray anchors before excluding conflicts. This keeps the lower
    # support represented when a nearby panel overlaps its visual neighborhood.
    support_mask = build_food_support_region(current_binary, target_bbox)
    support_points = [
        point for point in build_food_support_points(support_mask, target_bbox, max_points=5)
        if not guarded_conflicts[int(point[1]), int(point[0])]
    ]
    positive_points = sample_food_interior_points(safe_subject, max_points=7)
    for point in support_points:
        if point not in positive_points:
            positive_points.append(point)
    negative_points = []
    for core_mask in conflict_cores:
        negative_points.extend(sample_food_interior_points(core_mask, max_points=2, min_distance=10))
    if not positive_points or not negative_points:
        return current_mask, False, "insufficient_prompts"

    prompt_points = positive_points + negative_points
    prompt_labels = ([1] * len(positive_points)) + ([0] * len(negative_points))
    print(
        f"Food semantic {stage_name} stage: conflicts={len(conflict_cores)} "
        f"positives={len(positive_points)} negatives={len(negative_points)}"
    )
    try:
        results = run_sam_bbox_inference(
            img,
            target_bbox,
            multimask_output=True,
            imgsz=1024,
            points=[prompt_points],
            labels=[prompt_labels]
        )
        candidate_masks = normalize_result_masks(results, img_w, img_h)
    except Exception as error:
        return current_mask, False, f"sam_failed={error}"

    baseline = score_food_semantic_refine_candidate(
        current_binary,
        current_binary,
        safe_subject,
        conflict_mask,
        positive_points,
        negative_points,
        support_mask=support_mask,
        support_points=support_points
    )
    best_candidate = None
    best_metrics = None
    for candidate_mask in candidate_masks:
        # Refinement may only remove pixels from the accepted subject. This
        # prevents a later panel pass from restoring labels or poster graphics.
        candidate_binary = (candidate_mask > 0.5) & current_binary
        metrics = score_food_semantic_refine_candidate(
            candidate_binary,
            current_binary,
            safe_subject,
            conflict_mask,
            positive_points,
            negative_points,
            support_mask=support_mask,
            support_points=support_points
        )
        if not metrics:
            continue

        preserved = metrics["safePreservedRatio"] >= 0.90
        support_preserved = (
            not support_points or (
                metrics["supportPointCoverage"] >= 0.80 and
                metrics["supportPreservedRatio"] >= 0.80
            )
        )
        min_area_ratio = 0.78
        min_conflict_gain = 0.14
        if not preserved or not support_preserved or metrics["positiveCoverage"] < 0.86:
            continue
        if metrics["negativeHitRatio"] > 0.34 or metrics["areaRatio"] < min_area_ratio:
            continue
        conflict_gain = baseline["conflictKeepRatio"] - metrics["conflictKeepRatio"]
        metrics["conflictGain"] = conflict_gain
        if conflict_gain < min_conflict_gain:
            continue
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_candidate = candidate_binary.astype(np.float32)
            best_metrics = metrics

    if best_candidate is None:
        return current_mask, False, "no_candidate_preserved_subject"
    return best_candidate, True, (
        f"accepted conflictGain={best_metrics['conflictGain']:.3f} "
        f"safe={best_metrics['safePreservedRatio']:.3f} "
        f"support={best_metrics['supportPreservedRatio']:.3f}"
    )


def remove_verified_small_foreign_sibling(food_mask, foreign_entry, target_bbox):
    """Safely remove one small, independently segmented foreign sibling.

    This fallback is ownership-based. It uses only a sibling's actual SAM mask,
    never its bbox or name, and is unavailable for large regions that may be a
    plate, tray, or poster background.
    """
    current = food_mask > 0.5
    original_area = max(1, int(np.count_nonzero(current)))
    target_area = max(1, bbox_area(target_bbox))
    layout_mask = (foreign_entry.get("metrics") or {}).get("layoutMask")
    if layout_mask is None:
        return food_mask, False, 0

    foreign_mask = np.asarray(layout_mask, dtype=bool)
    foreign_area = int(np.count_nonzero(foreign_mask))
    overlap = int(np.count_nonzero(current & foreign_mask))
    # The cap is deliberately conservative. Large foreign layers must be
    # separated by the prompted SAM stage rather than a deterministic removal.
    if foreign_area < 16 or foreign_area > int(target_area * 0.14):
        return food_mask, False, 0
    if overlap < max(12, int(foreign_area * 0.12)):
        return food_mask, False, 0

    removal = current & foreign_mask
    removed = int(np.count_nonzero(removal))
    retained_ratio = (original_area - removed) / original_area
    if removed < 12 or removed > int(original_area * 0.12) or retained_ratio < 0.88:
        return food_mask, False, 0
    return (current & (~foreign_mask)).astype(np.float32), True, removed


def refine_food_mask_with_conflict_sam(
    img,
    coarse_mask,
    target_bbox,
    layer_meta,
    context_layers,
    original_target_bbox=None
):
    """Refine food against each nearby foreign sibling with ownership evidence."""
    img_h, img_w = img.shape[:2]
    coarse_binary = coarse_mask > 0.5
    debug_rows = []
    if not np.any(coarse_binary):
        return coarse_mask, False, debug_rows

    entries = collect_food_layout_conflict_entries(
        layer_meta or {}, context_layers or [], target_bbox, img_w, img_h
    )
    foreign_siblings = []
    target_area = max(1, bbox_area(target_bbox))
    subject_area = max(1, int(np.count_nonzero(coarse_binary)))
    adjacency_bbox = original_target_bbox or target_bbox
    verified_foreign_masks = []
    for entry in order_food_foreign_sibling_entries(entries[:12]):
        name = str((entry.get("layer") or {}).get("name") or "unknown")
        entry_bbox = entry.get("bbox")
        if not sibling_touches_or_overlaps_target(entry_bbox, adjacency_bbox):
            debug_rows.append({
                "name": name,
                "status": "skipped",
                "reason": "outside_bbox_before_sam"
            })
            continue
        entry_area = bbox_area(entry_bbox) if entry_bbox else 0
        # Large foreign regions have repeatedly failed subject-preservation
        # checks. Skip their own SAM pass before inference; a smaller overlapping
        # sibling can still provide safe, exact exclusion evidence.
        if not entry_bbox or entry_area >= min(target_area * 1.20, subject_area * 1.55):
            debug_rows.append({"name": name, "status": "skipped", "reason": "too_large_before_sam"})
            continue
        if entry_bbox_is_covered_by_foreign_masks(entry_bbox, verified_foreign_masks):
            debug_rows.append({"name": name, "status": "skipped", "reason": "covered_by_verified_sibling"})
            continue
        core_mask, metrics = segment_food_layout_conflict_mask(img, entry)
        if core_mask is None:
            debug_rows.append({"name": name, "status": "skipped", "reason": "no_reliable_layout_mask"})
            continue
        overlap = int(np.count_nonzero(core_mask & coarse_binary))
        core_area = max(1, int(np.count_nonzero(core_mask)))
        if overlap < max(8, int(core_area * 0.04)):
            debug_rows.append({
                "name": name,
                "status": "skipped",
                "reason": f"outside_food_mask overlap={overlap} core={core_area}"
            })
            continue
        verified_foreign_masks.append(metrics["layoutMask"])
        foreign_siblings.append({
            "entry": entry,
            "core": core_mask,
            "overlap": overlap,
            "coreArea": core_area,
            "metrics": metrics
        })
        debug_rows.append({
            "name": name,
            "status": "foreign_mask",
            "reason": f"core={core_area} containment={metrics['containment']:.2f} overlap={overlap}"
        })

    current = coarse_binary.astype(np.float32)
    changed = False
    # Larger overlap is handled first, but every sibling gets an isolated SAM
    # decision. A failed sibling cannot roll back an earlier accepted result.
    foreign_siblings.sort(key=lambda item: item["overlap"], reverse=True)
    for item in foreign_siblings[:3]:
        name = str(((item.get("entry") or {}).get("layer") or {}).get("name") or "unknown")
        current_binary = current > 0.5
        current_overlap = int(np.count_nonzero(item["core"] & current_binary))
        if current_overlap < max(12, int(item["coreArea"] * 0.035)):
            debug_rows.append({"name": name, "status": "skipped", "reason": "already_outside_subject"})
            continue
        candidate, accepted, reason = refine_food_mask_stage_with_sam(
            img, current, target_bbox, [item["core"]], "foreign_sibling"
        )
        if accepted:
            current = candidate
            changed = True
            debug_rows.append({"name": name, "status": "accepted", "reason": reason})
            continue

        fallback_mask, fallback_changed, removed = remove_verified_small_foreign_sibling(
            current,
            item,
            target_bbox
        )
        if fallback_changed:
            current = fallback_mask
            changed = True
            debug_rows.append({"name": name, "status": "fallback_accepted", "reason": f"real_sibling_mask removed={removed}"})
        else:
            debug_rows.append({"name": name, "status": "rejected", "reason": reason})

    return current, changed, debug_rows


def subtract_attached_layout_from_food_mask(food_mask, attached_layout_mask, target_bbox, layer_meta=None, entry_bbox=None):
    food_binary = food_mask > 0.5
    layout_binary = attached_layout_mask > 0.5
    if not np.any(food_binary) or not np.any(layout_binary):
        return food_mask, False, 0

    text = " ".join([
        str((layer_meta or {}).get("name", "")),
        str((layer_meta or {}).get("semanticType", "")),
        str((layer_meta or {}).get("category", "")),
        str((layer_meta or {}).get("runtimeType", "")),
        str((layer_meta or {}).get("compositeRole", "")),
        str((layer_meta or {}).get("extractionProfile", ""))
    ]).lower()
    is_price_like = is_price_like_text(text)
    is_text_like = any(token in text for token in ["text", "文字", "文本", "caption", "label", "tag", "$"])

    if entry_bbox is not None:
        bbox_mask = build_layout_bbox_mask(layout_binary.shape, entry_bbox, is_price_like=is_price_like, is_text_like=is_text_like)
        if np.any(bbox_mask):
            layout_binary = layout_binary | bbox_mask

    if is_price_like:
        layout_binary = cv2.dilate(layout_binary.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    elif is_text_like:
        layout_binary = cv2.dilate(layout_binary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0

    overlap = food_binary & layout_binary
    overlap_area = int(np.count_nonzero(overlap))
    if overlap_area == 0:
        return food_mask, False, 0

    kernel_size = max(2, compute_cleanup_kernel(target_bbox) + 1)
    core_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    protected_iterations = 1 if is_price_like else 2
    protected_core = cv2.erode(food_binary.astype(np.uint8), core_kernel, iterations=protected_iterations) > 0

    removal = overlap & (~protected_core)
    removal_area = int(np.count_nonzero(removal))
    if removal_area == 0:
        return food_mask, False, 0

    original_area = max(1, int(np.count_nonzero(food_binary)))
    retained = food_binary & (~removal)
    retained_area = int(np.count_nonzero(retained))
    retained_ratio = retained_area / original_area
    overlap_ratio = overlap_area / original_area

    min_retained_ratio = 0.68
    min_overlap_ratio = 0.006
    if is_price_like:
        min_retained_ratio = 0.62
        min_overlap_ratio = 0.003
    elif is_text_like:
        min_retained_ratio = 0.65
        min_overlap_ratio = 0.004

    if retained_ratio < min_retained_ratio or overlap_ratio < min_overlap_ratio:
        return food_mask, False, 0

    return retained.astype(np.float32), True, removal_area

def is_flat_ad_cleanup_layer(layer):
    if not isinstance(layer, dict):
        return False
    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", ""))
    ]).lower()
    return any(token in text for token in [
        "shape_panel", "price_badge", "cta_button", "logo_mark", "element_text",
        "text_node", "文字", "文本", "价格", "价签", "徽章", "面板", "底板",
        "标签底板", "文字背景", "panel", "badge", "button", "logo"
    ])

def is_food_label_like_layer(layer):
    if not isinstance(layer, dict):
        return False
    text = " ".join([
        str(layer.get("name", "")),
        str(layer.get("semanticType", "")),
        str(layer.get("category", "")),
        str(layer.get("runtimeType", "")),
        str(layer.get("compositeRole", ""))
    ]).lower()
    return any(token in text for token in [
        "price", "badge", "label", "tag", "text", "caption", "promo", "circle",
        "round", "pill", "chip", "coin", "sticker",
        "价格", "价签", "标签", "文字", "文本", "说明", "徽章", "$"
    ])

def build_flat_ad_cleanup_mask(layer_meta, context_layers, target_bbox, img_w, img_h):
    cleanup_mask = np.zeros((img_h, img_w), dtype=bool)
    cleanup_count = 0
    if not isinstance(context_layers, list):
        return cleanup_mask, cleanup_count

    expanded_target_bbox = expand_target_bbox_for_cleanup(target_bbox, img_w, img_h, ratio=0.18, min_pixels=10)

    for other in context_layers:
        if same_layer(layer_meta, other) or not is_flat_ad_cleanup_layer(other):
            continue
        bbox = other.get("bbox") if isinstance(other, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        other_bbox = normalize_context_bbox_to_pixel(bbox, img_w, img_h)
        if intersection_area(other_bbox, expanded_target_bbox) <= 0:
            continue
        x1, y1, x2, y2 = other_bbox
        cleanup_mask[y1:y2, x1:x2] = True
        cleanup_count += 1

    return cleanup_mask, cleanup_count

def build_food_label_cleanup_mask(layer_meta, context_layers, target_bbox, img_w, img_h):
    cleanup_mask = np.zeros((img_h, img_w), dtype=bool)
    cleanup_count = 0
    if not isinstance(context_layers, list):
        return cleanup_mask, cleanup_count

    expanded_target_bbox = expand_target_bbox_for_cleanup(target_bbox, img_w, img_h, ratio=0.22, min_pixels=12)

    for other in context_layers:
        if same_layer(layer_meta, other) or not is_food_label_like_layer(other):
            continue
        bbox = other.get("bbox") if isinstance(other, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        other_bbox = normalize_context_bbox_to_pixel(bbox, img_w, img_h)
        if intersection_area(other_bbox, expanded_target_bbox) <= 0:
            continue
        x1, y1, x2, y2 = other_bbox
        cleanup_mask[y1:y2, x1:x2] = True
        cleanup_count += 1

    return cleanup_mask, cleanup_count

def has_flat_ad_cleanup_context(layer_meta, context_layers, target_bbox, img_w, img_h):
    _, cleanup_count = build_flat_ad_cleanup_mask(layer_meta, context_layers, target_bbox, img_w, img_h)
    return cleanup_count > 0


def is_high_coverage_entity(metrics, strategy, layer_meta, context_layers, img_w, img_h, exclude_bboxes):
    """Recognize a tight entity mask that legitimately fills most of its bbox.

    A high fill ratio is ambiguous: it can mean either a well-fitted object or
    a background plane. Prefer geometry and layer ownership over object names.
    """
    fill = metrics["target_fill_ratio"]
    inside = metrics["mask_inside_target_ratio"]
    area_ratio = metrics["mask_area_ratio"]
    overlap = metrics["bbox_overlap_ratio"]
    touch_count = metrics["bbox_touch_count"]
    shape = metrics["shape_features"]
    target_bbox = metrics["target_bbox"]

    target_width = max(1, target_bbox[2] - target_bbox[0])
    target_height = max(1, target_bbox[3] - target_bbox[1])
    # A bbox covering virtually the whole image is not a reliable object
    # ownership signal. Do not turn a full-scene mask into an entity.
    if target_width / max(1, img_w) >= 0.90 and target_height / max(1, img_h) >= 0.90:
        return False, "target_bbox_too_broad"
    if area_ratio > MAX_MASK_AREA_RATIO_IN_BBOX:
        return False, "coverage_or_inside_gate"
    if overlap < 0.70 and not metrics["center_inside"]:
        return False, "bbox_overlap_gate"
    if metrics.get("strong_exclude_mask_ratio", 0.0) >= 0.22:
        return False, "strong_sibling_overlap"

    has_entity_shape = bool(
        shape.get("isBlockLike") or
        shape.get("isThinVertical") or
        shape.get("isHorizontalSurface") or
        shape.get("isTableSupport") or
        shape.get("isRectangularPlane")
    )
    if not has_entity_shape:
        return False, "entity_shape_gate"

    # A mask that touches most bbox sides but leaves a large outside fraction
    # is the typical background failure. A tight inside mask is still allowed
    # to touch the bbox because the bbox may describe the visible object edge.
    if touch_count >= 3 and inside < 0.95:
        tight_entity = (
            fill >= 0.82 and
            inside >= 0.84 and
            area_ratio <= 1.05 and
            overlap >= 0.82 and
            bool(
                shape.get("isBlockLike") or
                shape.get("isHorizontalSurface") or
                shape.get("isRectangularPlane")
            )
        )
        if not tight_entity:
            return False, "boundary_spill_gate"

    if inside < 0.90:
        tight_entity = (
            fill >= 0.82 and
            inside >= 0.84 and
            area_ratio <= 1.05 and
            overlap >= 0.82 and
            bool(
                shape.get("isBlockLike") or
                shape.get("isHorizontalSurface") or
                shape.get("isRectangularPlane")
            )
        )
        if not tight_entity:
            return False, "inside_ratio_gate"

    # Strong, same-parent layout siblings remain authoritative exclusions. This
    # catches a panel/background that occupies the same region without naming a
    # particular object type.
    target_bbox = metrics["target_bbox"]
    candidate_bbox = metrics["bbox"]
    target_area = max(1, bbox_area(target_bbox))
    candidate_area = max(1, bbox_area(candidate_bbox))
    for other in context_layers or []:
        if same_layer(layer_meta or {}, other):
            continue
        other_bbox_norm = other.get("bbox") if isinstance(other, dict) else None
        if not isinstance(other_bbox_norm, list) or len(other_bbox_norm) != 4:
            continue
        other_bbox = normalize_context_bbox_to_pixel(other_bbox_norm, img_w, img_h)
        sibling_target_overlap = intersection_area(other_bbox, target_bbox) / target_area
        sibling_candidate_overlap = intersection_area(other_bbox, candidate_bbox) / candidate_area
        if sibling_target_overlap <= 0 or sibling_candidate_overlap < 0.82:
            continue
        same_parent = bool(
            layer_meta.get("parentLayerId") and
            other.get("parentLayerId") and
            layer_meta.get("parentLayerId") == other.get("parentLayerId")
        )
        other_profile = str(other.get("extractionProfile", "")).lower()
        other_strategy = get_layer_strategy(other).get("type")
        broad_sibling = (
            other_profile in {"background_plate", "vector_layout_element"} or
            other_strategy in {"flat_shape", "wall_art"}
        )
        sibling_area_ratio = bbox_area(other_bbox) / target_area
        if same_parent and broad_sibling and sibling_area_ratio >= 1.05:
            return False, "broad_layout_sibling_overlap"

    return True, "high_coverage_entity"


def count_mask_in_strong_excludes(mask_binary, exclude_bboxes):
    strong_bboxes = [entry for entry in exclude_bboxes if is_strong_exclude(entry)]
    return count_mask_in_bboxes(mask_binary, strong_bboxes) if strong_bboxes else 0


def detect_multi_entity_coverage(mask_binary, target_bbox):
    """Recognize several substantial, enclosed entities in one semantic bbox."""
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    crop = np.asarray(mask_binary[ty1:ty2, tx1:tx2], dtype=np.uint8)
    target_area = max(1, int(crop.shape[0] * crop.shape[1]))
    if crop.size == 0:
        return False, 0

    total_fill = int(np.count_nonzero(crop)) / target_area
    if total_fill < 0.55 or total_fill > 0.96:
        return False, 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    min_component_area = max(64, int(target_area * 0.035))
    substantial = []
    for label in range(1, count):
        _, _, _, _, area = [int(value) for value in stats[label]]
        if area >= min_component_area:
            substantial.append(area)

    if len(substantial) < 2:
        return False, len(substantial)
    substantial.sort(reverse=True)
    if substantial[0] / target_area > 0.82:
        return False, len(substantial)
    return True, len(substantial)


def score_candidate(metrics, strategy):
    # Prefer masks that are mostly inside the target and occupy a plausible part
    # of the bbox. Penalize large bbox-filling masks because they are often room
    # surfaces/backgrounds rather than object parts.
    fill = metrics["target_fill_ratio"]
    inside = metrics["mask_inside_target_ratio"]
    area = metrics["mask_area_ratio"]
    overlap = metrics["bbox_overlap_ratio"]
    center_bonus = 0.08 if metrics["center_inside"] else 0
    if strategy.get("type") == "food_product":
        plausible_fill = 1.0 - min(1.0, abs(fill - 0.64) / 0.64)
        background_penalty = max(0, fill - 0.86) * 1.0 + max(0, area - 0.96) * 0.35
    elif strategy.get("type") == "decor_arrangement":
        plausible_fill = 1.0 - min(1.0, abs(fill - 0.58) / 0.58)
        background_penalty = max(0, fill - 0.86) * 1.2 + max(0, area - 0.92) * 0.45
    elif strategy.get("type") == "soft_edge":
        plausible_fill = 1.0 - min(1.0, abs(fill - 0.42) / 0.42)
        background_penalty = max(0, fill - 0.88) * 1.3 + max(0, area - 0.92) * 0.55
    elif strategy.get("type") == "wall_art":
        # Framed art is a bounded, opaque plane. It often fills the supplied
        # semantic bbox, so a high fill must not be treated as scene spill.
        plausible_fill = 1.0 - min(1.0, abs(fill - 0.78) / 0.78)
        background_penalty = max(0, area - 1.15) * 0.45
    else:
        plausible_fill = 1.0 - min(1.0, abs(fill - 0.38) / 0.38)
        background_penalty = max(0, fill - 0.58) * 1.8 + max(0, area - 0.75) * 0.9
    background_penalty += metrics.get("exclude_mask_ratio", 0) * 0.2
    rectangular_bonus = 0
    if strategy.get("prefer_rectangular"):
        bbox = metrics["bbox"]
        bbox_area_value = max(1, bbox_area(bbox))
        rectangularity = metrics["mask_area"] / bbox_area_value
        rectangular_bonus = min(0.18, rectangularity * 0.18)
        background_penalty += max(0, metrics["bbox_touch_count"] - 1) * 0.14
    shape_bonus = 0
    shape_features = metrics.get("shape_features", {})
    if strategy.get("type") == "furniture" and shape_features.get("isBlockLike"):
        shape_bonus += 0.1
    if strategy.get("type") == "table" and (
        shape_features.get("isHorizontalSurface") or
        shape_features.get("isThinVertical") or
        shape_features.get("isTableSupport")
    ):
        shape_bonus += 0.08
    if strategy.get("type") == "wall_art" and shape_features.get("isRectangularPlane"):
        shape_bonus += 0.12
    if strategy.get("type") == "decor_arrangement":
        if shape_features.get("isBlockLike"):
            shape_bonus += 0.08
        if shape_features.get("isThinVertical"):
            shape_bonus += 0.04
        if shape_features.get("bottomBand", 0) >= 0.45:
            shape_bonus += 0.04
        if is_decor_base_shape(shape_features):
            shape_bonus += 0.16
    if strategy.get("type") == "decor_atomic" and shape_features.get("isBlockLike"):
        shape_bonus += 0.08
    if strategy.get("type") == "food_product":
        if shape_features.get("isBlockLike"):
            shape_bonus += 0.08
        if shape_features.get("isHorizontalSurface") or shape_features.get("isRectangularPlane"):
            shape_bonus += 0.14
        if metrics.get("center_inside"):
            shape_bonus += 0.06
        if 0.14 <= fill <= 0.78:
            shape_bonus += 0.08
    if metrics.get("high_coverage_entity"):
        # A tight semantic bbox can legitimately be almost full. Once the
        # candidate is structurally validated, reduce the generic background
        # penalty instead of forcing every object type to raise max_fill.
        background_penalty *= 0.22
        shape_bonus += 0.10
    if metrics.get("multi_entity_coverage"):
        # One semantic bbox may intentionally describe a group of separate
        # hard objects. Reward the verified group instead of treating its
        # higher coverage as a room-surface mask.
        plausible_group_fill = 1.0 - min(1.0, abs(fill - 0.72) / 0.72)
        background_penalty *= 0.18
        shape_bonus += 0.22
        return (
            (inside * 0.42) +
            (plausible_group_fill * 0.28) +
            (overlap * 0.22) +
            center_bonus + rectangular_bonus + shape_bonus - background_penalty
        )
    return (inside * 0.42) + (plausible_fill * 0.28) + (overlap * 0.22) + center_bonus + rectangular_bonus + shape_bonus - background_penalty


def select_wall_art_fallback_candidate(fallback_candidates, target_bbox):
    """Keep a usable wall-art mask when strict contextual gates reject all masks.

    Wall art commonly fills most of its bbox, so the normal background gate can
    reject every SAM result when nearby layout exclusions cover the same plane.
    This fallback still requires a meaningful target coverage and bounded mask;
    it never accepts a full-image or tiny fragment mask.
    """
    if not fallback_candidates:
        return None

    tx1, ty1, tx2, ty2 = target_bbox
    target_area = max(1, bbox_area(target_bbox))
    eligible = []
    for item in fallback_candidates:
        mask = item["mask"]
        metrics = item["metrics"]
        fill = metrics["target_fill_ratio"]
        inside = metrics["mask_inside_target_ratio"]
        area_ratio = metrics["mask_area_ratio"]
        if fill < 0.10 or inside < 0.45:
            continue
        if area_ratio > 1.35:
            continue
        if metrics["bbox_overlap_ratio"] < 0.55:
            continue

        # Prefer a well-covered rectangular plane, but keep the score bounded so
        # a contextual exclusion ratio cannot make every valid plane negative.
        fallback_score = (
            (inside * 0.42) +
            (min(1.0, fill / 0.68) * 0.30) +
            (min(1.0, metrics["bbox_overlap_ratio"]) * 0.18) +
            (0.10 if metrics["center_inside"] else 0.0) -
            (max(0.0, area_ratio - 1.0) * 0.12)
        )
        eligible.append({
            **item,
            "fallbackScore": fallback_score
        })

    if not eligible:
        return None
    return max(eligible, key=lambda item: item["fallbackScore"])


def build_food_context_prior_masks(layer_meta, context_layers, target_bbox, img_w, img_h):
    entries = collect_attached_layout_entries(layer_meta or {}, context_layers or [], target_bbox, img_w, img_h)
    price_mask = np.zeros((img_h, img_w), dtype=bool)
    text_mask = np.zeros((img_h, img_w), dtype=bool)
    panel_mask = np.zeros((img_h, img_w), dtype=bool)

    for entry in entries[:12]:
        layer = entry.get("layer") or {}
        bbox = entry.get("bbox")
        if not bbox:
            continue
        text = " ".join([
            str(layer.get("name", "")),
            str(layer.get("semanticType", "")),
            str(layer.get("category", "")),
            str(layer.get("runtimeType", "")),
            str(layer.get("compositeRole", "")),
            str(layer.get("extractionProfile", ""))
        ]).lower()
        is_price = is_price_like_text(text)
        is_text = any(token in text for token in ["text", "文字", "文本", "caption", "label", "tag", "$"])
        is_panel = any(token in text for token in ["card", "panel", "背景", "底板", "卡片背景", "shape_panel"])
        is_broad_background = any(token in text for token in ["波纹背景", "background_master", "ad_background", "橙色波纹背景"])

        bbox_mask = build_layout_bbox_mask(
            (img_h, img_w),
            bbox,
            is_price_like=is_price,
            is_text_like=(is_text and not is_price)
        )
        if is_price:
            price_mask |= bbox_mask
            text_mask |= bbox_mask
        elif is_text:
            text_mask |= bbox_mask
        if is_panel and not is_broad_background:
            panel_mask |= bbox_mask

    return {
        "entries": entries,
        "priceMask": price_mask,
        "textMask": text_mask,
        "panelMask": panel_mask
    }


def evaluate_food_mask_variant(mask_binary, target_bbox, context_priors):
    if not np.any(mask_binary):
        return None

    cleaned = cleanup_mask(mask_binary.astype(np.float32), target_bbox) > 0.5
    if not np.any(cleaned):
        return None

    current_bbox = mask_bbox(cleaned)
    if not current_bbox:
        return None

    tx1, ty1, tx2, ty2 = target_bbox
    target_area = max(1, bbox_area(target_bbox))
    mask_area = int(np.count_nonzero(cleaned))
    target_mask_area = int(np.count_nonzero(cleaned[ty1:ty2, tx1:tx2]))
    target_fill_ratio = target_mask_area / target_area
    mask_inside_target_ratio = target_mask_area / max(1, mask_area)
    bbox_overlap_ratio = intersection_area(current_bbox, target_bbox) / target_area
    bbox_touch_count = int(current_bbox[0] <= tx1 + 2) + int(current_bbox[1] <= ty1 + 2) + int(current_bbox[2] >= tx2 - 2) + int(current_bbox[3] >= ty2 - 2)
    shape_features = compute_shape_features(current_bbox, target_bbox, mask_area)

    text_only_mask = context_priors["textMask"] & (~context_priors["priceMask"])
    price_overlap = int(np.count_nonzero(cleaned & context_priors["priceMask"])) / max(1, mask_area)
    text_overlap = int(np.count_nonzero(cleaned & text_only_mask)) / max(1, mask_area)
    panel_overlap = int(np.count_nonzero(cleaned & context_priors["panelMask"])) / max(1, mask_area)

    fill_plausible = 1.0 - min(1.0, abs(target_fill_ratio - 0.46) / 0.46)
    structural_bonus = 0.0
    if is_food_support_shape(shape_features):
        structural_bonus += 0.10
    if shape_features["bottomBand"] >= 0.78:
        structural_bonus += 0.06
    if shape_features["relativeWidth"] >= 0.34:
        structural_bonus += 0.04
    if box_center_inside(current_bbox, target_bbox):
        structural_bonus += 0.06
    if 0.20 <= target_fill_ratio <= 0.62:
        structural_bonus += 0.06

    exclusion_penalty = min(0.72, (
        (price_overlap * 0.58) +
        (text_overlap * 0.26) +
        (panel_overlap * 0.14)
    ))
    top_clip_penalty = 0.08 if (current_bbox[1] <= ty1 + 4 and shape_features["centerY"] <= 0.40) else 0.0
    edge_penalty = max(0, bbox_touch_count - 2) * 0.035
    overspan_penalty = 0.0
    if bbox_touch_count >= 3:
        overspan_penalty += 0.10
    if bbox_touch_count >= 3 and mask_inside_target_ratio < 0.92:
        overspan_penalty += 0.10
    if shape_features["bottomBand"] >= 1.02:
        overspan_penalty += 0.08

    raw_score = (
        0.16 +
        (mask_inside_target_ratio * 0.28) +
        (fill_plausible * 0.20) +
        (bbox_overlap_ratio * 0.14) +
        structural_bonus -
        exclusion_penalty -
        top_clip_penalty -
        edge_penalty -
        overspan_penalty
    )
    score = max(0.0, min(1.0, raw_score))

    return {
        "mask": cleaned.astype(np.float32),
        "bbox": current_bbox,
        "maskArea": mask_area,
        "targetFillRatio": target_fill_ratio,
        "maskInsideTargetRatio": mask_inside_target_ratio,
        "bboxOverlapRatio": bbox_overlap_ratio,
        "bboxTouchCount": bbox_touch_count,
        "shapeFeatures": shape_features,
        "priceOverlap": price_overlap,
        "textOverlap": text_overlap,
        "panelOverlap": panel_overlap,
        "score": score,
        "rawScore": raw_score
    }


def select_food_masks_from_candidates(candidate_masks, target_bbox, img_w, img_h, layer_meta=None, context_layers=None):
    context_priors = build_food_context_prior_masks(layer_meta or {}, context_layers or [], target_bbox, img_w, img_h)
    variants = []

    for index, mask in enumerate(candidate_masks):
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        raw_binary = mask > 0.5
        # Bboxes describe layout ownership, not exact visible pixels. Subtracting
        # price/text bboxes here cuts white food and plate regions when they
        # overlap a badge. Semantic cleanup runs later with real SAM masks.
        candidate_variants = [("raw", raw_binary)]
        for variant_name, variant_mask in candidate_variants:
            evaluated = evaluate_food_mask_variant(variant_mask, target_bbox, context_priors)
            if not evaluated:
                continue
            variants.append({
                "index": index,
                "variant": variant_name,
                **evaluated
            })

    if not variants:
        return None, 0, None

    variants.sort(key=lambda item: item["score"], reverse=True)
    primary = variants[0]
    selected = [primary]
    merged = primary["mask"].copy()

    for candidate in variants[1:]:
        if len(selected) >= 2:
            break

        candidate_distance = bbox_distance(candidate["bbox"], primary["bbox"])
        candidate_overlap_ratio = intersection_area(candidate["bbox"], primary["bbox"]) / max(
            1,
            min(bbox_area(candidate["bbox"]), bbox_area(primary["bbox"]))
        )
        candidate_horizontal_overlap = horizontal_overlap_ratio(candidate["bbox"], primary["bbox"])
        merge_distance_limit = max(24.0, (bbox_area(target_bbox) ** 0.5) * 0.14)
        if (
            candidate_distance > merge_distance_limit and
            candidate_overlap_ratio < 0.04 and
            candidate_horizontal_overlap < 0.18
        ):
            continue
        if (
            candidate["targetFillRatio"] < 0.06 and
            candidate["shapeFeatures"]["centerY"] < 0.32 and
            not is_food_support_shape(candidate["shapeFeatures"])
        ):
            continue

        new_pixels = (candidate["mask"] > 0.5) & ~(merged > 0.5)
        new_area = int(np.count_nonzero(new_pixels))
        if new_area < max(24, int(primary["maskArea"] * 0.04)):
            continue

        trial = np.maximum(merged, candidate["mask"])
        evaluated_trial = evaluate_food_mask_variant(trial > 0.5, target_bbox, context_priors)
        if not evaluated_trial:
            continue

        price_penalty_growth = evaluated_trial["priceOverlap"] - primary["priceOverlap"]
        text_penalty_growth = evaluated_trial["textOverlap"] - primary["textOverlap"]
        if price_penalty_growth > 0.015 or text_penalty_growth > 0.02:
            continue
        if evaluated_trial["score"] + 0.01 < primary["score"]:
            continue

        merged = trial.astype(np.float32)
        selected.append(candidate)
        primary = {
            **evaluated_trial,
            "index": primary["index"],
            "variant": f"{primary['variant']}+{candidate['variant']}"
        }

    merged_binary = merged > 0.5
    merged_eval = evaluate_food_mask_variant(merged_binary, target_bbox, context_priors)
    if not merged_eval:
        return None, 0, None

    score = merged_eval["score"]
    quality_gate = build_quality_gate(score, score, merged_eval["targetFillRatio"], selected, "food_product")
    debug_candidates = []
    for item in variants[:12]:
        debug_candidates.append({
            "index": item["index"],
            "score": round(float(item["score"]), 3),
            "fill": round(float(item["targetFillRatio"]), 3),
            "inside": round(float(item["maskInsideTargetRatio"]), 3),
            "area": round(float(item["maskArea"] / max(1, bbox_area(target_bbox))), 3),
            "bboxOverlap": round(float(item["bboxOverlapRatio"]), 3),
            "exclude": round(float(item["priceOverlap"] + item["textOverlap"]), 3),
            "strongExclude": round(float(item["panelOverlap"]), 3),
            "touch": int(item["bboxTouchCount"]),
            "center": bool(box_center_inside(item["bbox"], target_bbox)),
            "shapeFeatures": round_shape_features(item["shapeFeatures"]),
            "decorBase": False,
            "candidate": True,
            "selected": item in selected,
            "rejectReason": item["variant"]
        })

    quality = {
        "status": quality_gate["status"],
        "score": round(float(score), 3),
        "maskCount": len(selected),
        "targetFillRatio": round(float(merged_eval["targetFillRatio"]), 3),
        "primaryScore": round(float(merged_eval["score"]), 3),
        "rawPrimaryScore": round(float(merged_eval.get("rawScore", merged_eval["score"])), 3),
        "candidateCount": len(variants),
        "excludeBoxCount": len(context_priors["entries"]),
        "excludeReliable": True,
        "hasFlatAdCleanupContext": bool(context_priors["entries"]),
        "flatAdCleanupCount": 0,
        "flatAdCleanedFillRatio": None,
        "strategy": "food_product",
        "strategyProfile": str((layer_meta or {}).get("extractionProfile", "")).lower() or "food_product",
        "selectedIndexes": [item["index"] for item in selected],
        "debugCandidates": debug_candidates,
        "reason": "food_sam_candidate_semantic_selection",
        "runtimeAction": quality_gate["runtimeAction"],
        "shouldGenerateRuntimeLayer": quality_gate["shouldGenerateRuntimeLayer"],
        "needsHigherPrecision": quality_gate["needsHigherPrecision"],
        "issues": quality_gate["issues"],
        "recommendedEngine": quality_gate["recommendedEngine"],
        "foodSelectionMode": "sam_candidates_semantic_mask_selection"
    }
    return merged.astype(np.float32), len(selected), quality

def select_and_merge_masks(candidate_masks, target_bbox, img_w, img_h, layer_meta=None, context_layers=None):
    strategy = get_layer_strategy(layer_meta or {})
    if strategy["type"] == "food_product":
        return select_food_masks_from_candidates(
            candidate_masks,
            target_bbox,
            img_w,
            img_h,
            layer_meta=layer_meta,
            context_layers=context_layers
        )

    exclude_bboxes = build_exclude_bboxes(layer_meta or {}, context_layers or [], target_bbox, img_w, img_h)
    exclude_mask_union = build_exclude_mask(exclude_bboxes, img_w, img_h) if exclude_bboxes else None
    has_flat_cleanup_context = has_flat_ad_cleanup_context(
        layer_meta or {},
        context_layers or [],
        target_bbox,
        img_w,
        img_h
    )
    is_drink_layer = is_drink_product_layer(layer_meta or {})
    target_area = max(1, bbox_area(target_bbox))
    candidates = []
    fallback_candidates = []
    debug_candidates = []
    debug_rows = []
    tx1, ty1, tx2, ty2 = target_bbox

    for index, mask in enumerate(candidate_masks):
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        mask_binary = mask > 0.5
        current_bbox = mask_bbox(mask)
        if not current_bbox:
            continue

        mask_area = int(np.count_nonzero(mask_binary))
        if mask_area <= 0:
            continue

        target_mask_area = int(np.count_nonzero(mask_binary[ty1:ty2, tx1:tx2]))
        exclude_mask_area = int(np.count_nonzero(mask_binary & exclude_mask_union)) if exclude_mask_union is not None else 0
        exclude_mask_ratio = exclude_mask_area / mask_area
        strong_exclude_mask_area = count_mask_in_strong_excludes(mask_binary, exclude_bboxes) if exclude_bboxes else 0
        strong_exclude_mask_ratio = strong_exclude_mask_area / mask_area
        target_fill_ratio = target_mask_area / target_area
        mask_inside_target_ratio = target_mask_area / mask_area
        bbox_overlap_ratio = intersection_area(current_bbox, target_bbox) / target_area
        mask_area_ratio = mask_area / target_area
        center_inside = box_center_inside(current_bbox, target_bbox)
        bbox_touch_count = int(current_bbox[0] <= tx1 + 2) + int(current_bbox[1] <= ty1 + 2) + int(current_bbox[2] >= tx2 - 2) + int(current_bbox[3] >= ty2 - 2)
        shape_features = compute_shape_features(current_bbox, target_bbox, mask_area)
        decor_base = strategy["type"] == "decor_arrangement" and is_decor_base_shape(shape_features)
        shape_allowed, shape_reject_reason = shape_strategy_gate(shape_features, strategy)
        multi_entity_coverage, multi_entity_count = detect_multi_entity_coverage(
            mask_binary,
            target_bbox
        ) if strategy["type"] in HARD_EDGE_STRATEGIES or strategy["type"] in {"table", "furniture"} else (False, 0)
        if not multi_entity_coverage and strategy["type"] in HARD_EDGE_STRATEGIES.union({"table", "furniture"}):
            # Some SAM masks join adjacent objects through a thin contact or
            # raster bridge, so connected-component counting alone misses the
            # group. High containment plus bounded coverage is still a useful
            # generic signal for a multi-object hard-entity bbox.
            multi_entity_coverage = bool(
                0.60 <= target_fill_ratio <= 0.88 and
                mask_inside_target_ratio >= 0.975 and
                0.68 <= mask_area_ratio <= 0.94 and
                bbox_overlap_ratio >= 0.55 and
                shape_features["isBlockLike"]
            )
            if multi_entity_coverage:
                multi_entity_count = 2
        lighting_background_like = (
            strategy["type"] == "lighting" and
            shape_features["isBlockLike"] and
            target_fill_ratio >= 0.38 and
            mask_area_ratio >= 0.45 and
            (mask_inside_target_ratio < 0.95 or bbox_touch_count >= 3)
        )
        table_background_like = (
            strategy["type"] == "table" and
            bbox_touch_count >= 3 and
            shape_features["bottomBand"] > 1.03 and
            target_fill_ratio >= 0.32 and
            mask_area_ratio >= 0.34
        )
        furniture_background_like = (
            strategy["type"] == "furniture" and
            shape_features["isBlockLike"] and
            target_fill_ratio >= 0.42 and
            mask_area_ratio >= 0.50 and
            bbox_touch_count >= 3 and
            mask_inside_target_ratio < 0.96
        )

        metrics = {
            "index": index,
            "target_fill_ratio": target_fill_ratio,
            "mask_inside_target_ratio": mask_inside_target_ratio,
            "bbox_overlap_ratio": bbox_overlap_ratio,
            "mask_area_ratio": mask_area_ratio,
            "center_inside": center_inside,
            "bbox_touch_count": bbox_touch_count,
            "bbox": current_bbox,
            "target_bbox": target_bbox,
            "mask_area": mask_area,
            "exclude_mask_ratio": exclude_mask_ratio,
            "strong_exclude_mask_ratio": strong_exclude_mask_ratio,
            "shape_features": shape_features,
            "multi_entity_coverage": multi_entity_coverage,
            "multi_entity_count": multi_entity_count
        }
        high_coverage_entity, high_coverage_reason = is_high_coverage_entity(
            metrics,
            strategy,
            layer_meta or {},
            context_layers or [],
            img_w,
            img_h,
            exclude_bboxes
        )
        metrics["high_coverage_entity"] = high_coverage_entity
        metrics["high_coverage_reason"] = high_coverage_reason

        is_probable_foreground = (
            target_fill_ratio > 0
            and (
                target_fill_ratio <= strategy["max_fill"] or
                high_coverage_entity or
                multi_entity_coverage
            )
            and mask_inside_target_ratio >= MIN_MASK_INSIDE_TARGET_RATIO
            and MIN_MASK_AREA_RATIO_IN_BBOX <= mask_area_ratio <= MAX_MASK_AREA_RATIO_IN_BBOX
            and (bbox_overlap_ratio >= MIN_BBOX_OVERLAP_RATIO or center_inside)
            and shape_allowed
            and not lighting_background_like
            and not table_background_like
            and (not furniture_background_like or multi_entity_coverage)
            and strong_exclude_mask_ratio < 0.22
        )
        score = score_candidate(metrics, strategy)
        debug_candidate = {
            "index": index,
            "score": round(float(score), 3),
            "fill": round(float(target_fill_ratio), 3),
            "inside": round(float(mask_inside_target_ratio), 3),
            "area": round(float(mask_area_ratio), 3),
            "bboxOverlap": round(float(bbox_overlap_ratio), 3),
            "exclude": round(float(exclude_mask_ratio), 3),
            "strongExclude": round(float(strong_exclude_mask_ratio), 3),
            "touch": int(bbox_touch_count),
            "center": bool(center_inside),
            "shapeFeatures": round_shape_features(shape_features),
            "decorBase": bool(decor_base),
            "highCoverageEntity": bool(high_coverage_entity),
            "multiEntityCoverage": bool(multi_entity_coverage),
            "candidate": bool(is_probable_foreground),
            "selected": False,
            "rejectReason": "" if is_probable_foreground else (
                "lighting_background_like" if lighting_background_like else (
                    "table_background_like" if table_background_like else (
                        "furniture_background_like" if furniture_background_like else (
                            "sibling_overlap" if strong_exclude_mask_ratio >= 0.22 else (
                                high_coverage_reason if high_coverage_entity else (shape_reject_reason or "gate")
                            )
                        )
                    )
                )
            )
        }
        debug_candidates.append(debug_candidate)
        fallback_candidates.append({
            "mask": mask,
            "metrics": metrics,
            "score": score,
            "debug": debug_candidate
        })

        debug_rows.append(
            f"fill={target_fill_ratio:.3f}, inside={mask_inside_target_ratio:.3f}, "
            f"area={mask_area_ratio:.3f}, bboxOverlap={bbox_overlap_ratio:.3f}, "
            f"exclude={exclude_mask_ratio:.3f}, strongExclude={strong_exclude_mask_ratio:.3f}, "
            f"aspect={shape_features['aspectRatio']:.3f}, thinV={shape_features['isThinVertical']}, "
            f"support={shape_features['isTableSupport']}, block={shape_features['isBlockLike']}, "
            f"bottom={shape_features['bottomBand']:.3f}, "
            f"touch={bbox_touch_count}, center={center_inside}, score={score:.3f}, "
            f"candidate={is_probable_foreground}"
        )

        if is_probable_foreground:
            candidates.append({
                "mask": mask,
                "metrics": metrics,
                "score": score,
                "debug": debug_candidate
            })

    forced_wall_art_fallback = False
    if not candidates and strategy["type"] == "wall_art":
        fallback = select_wall_art_fallback_candidate(fallback_candidates, target_bbox)
        if fallback is not None:
            fallback["debug"] = next(
                (row for row in debug_candidates if row["index"] == fallback["metrics"]["index"]),
                None
            )
            if fallback["debug"] is not None:
                fallback["debug"]["candidate"] = True
                fallback["debug"]["selected"] = True
                fallback["debug"]["rejectReason"] = "wall_art_quality_fallback"
            candidates = [fallback]
            forced_wall_art_fallback = True
            print(
                "Wall-art quality fallback selected candidate "
                f"#{fallback['metrics']['index']} score={fallback['fallbackScore']:.3f}"
            )
        else:
            print("No candidate masks selected. Candidates:", " | ".join(debug_rows[:12]))
            return None, 0, None
    elif not candidates:
        print("No candidate masks selected. Candidates:", " | ".join(debug_rows[:12]))
        return None, 0, None

    exclude_values = [item["metrics"].get("exclude_mask_ratio", 0) for item in candidates]
    exclude_reliable = bool(exclude_bboxes) and not (
        len(exclude_values) > 0 and
        sum(1 for value in exclude_values if value >= 0.95) / len(exclude_values) >= 0.8
    )
    if not exclude_reliable:
        for item in candidates:
            item["metrics"]["exclude_mask_ratio"] = 0
            item["score"] = score_candidate(item["metrics"], strategy)
            item["debug"]["score"] = round(float(item["score"]), 3)
            item["debug"]["exclude"] = 0
        candidates.sort(key=lambda item: item["score"], reverse=True)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    primary = candidates[0]
    selected = [primary]
    primary["debug"]["selected"] = True
    primary["debug"]["rejectReason"] = "primary"

    merged = np.zeros((img_h, img_w), dtype=np.float32)
    merged = np.maximum(merged, primary["mask"])
    primary_bbox = primary["metrics"]["bbox"]
    primary_shape = primary["metrics"]["shape_features"]

    for candidate in candidates[1:]:
        if not strategy["allow_attachments"]:
            candidate["debug"]["rejectReason"] = "attachments_disabled"
            continue
        if len(selected) >= strategy["max_masks"]:
            candidate["debug"]["rejectReason"] = "max_masks"
            break
        metrics = candidate["metrics"]
        shape_features = metrics["shape_features"]
        distance = bbox_distance(primary_bbox, metrics["bbox"])
        overlaps_primary = intersection_area(primary_bbox, metrics["bbox"]) > 0
        horizontal_overlap = horizontal_overlap_ratio(primary_bbox, metrics["bbox"])
        is_attachment = overlaps_primary if strategy["require_overlap_for_attachments"] else (overlaps_primary or distance <= strategy["max_attachment_distance"])
        primary_binary = primary["mask"] > 0.5
        candidate_binary = candidate["mask"] > 0.5
        candidate_area = max(1, int(np.count_nonzero(candidate_binary)))
        mask_overlap_ratio = int(np.count_nonzero(primary_binary & candidate_binary)) / candidate_area
        primary_dilated = cv2.dilate(primary_binary.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        mask_touches_primary = bool(np.any(primary_dilated & candidate_binary))
        is_small_part = metrics["target_fill_ratio"] <= 0.28 and metrics["mask_area_ratio"] <= 0.35
        is_table_structure = (
            shape_features["isThinVertical"] or
            shape_features["isHorizontalSurface"] or
            shape_features["isTableSupport"]
        )
        is_table_related = (
            is_attachment or
            distance <= strategy["max_attachment_distance"] * 2.4 or
            horizontal_overlap >= 0.18
        )
        is_table_leg_part = (
            strategy["type"] == "table" and
            (shape_features["isThinVertical"] or shape_features["isTableSupport"]) and
            metrics["target_fill_ratio"] <= 0.22 and
            metrics["mask_area_ratio"] <= 0.30 and
            (is_attachment or distance <= strategy["max_attachment_distance"] * 1.4 or horizontal_overlap >= 0.08)
        )
        is_table_surface_part = (
            strategy["type"] == "table" and
            shape_features["isHorizontalSurface"] and
            metrics["target_fill_ratio"] <= 0.34 and
            metrics["mask_area_ratio"] <= 0.42 and
            mask_overlap_ratio >= 0.10 and
            (is_attachment or distance <= strategy["max_attachment_distance"] or horizontal_overlap >= 0.18)
        )
        is_table_complete_peer = (
            strategy["type"] == "table" and
            metrics["mask_inside_target_ratio"] >= 0.96 and
            metrics["target_fill_ratio"] >= 0.34 and
            metrics["target_fill_ratio"] <= 0.68 and
            metrics["mask_area_ratio"] <= 0.78 and
            metrics["bbox_overlap_ratio"] >= 0.72 and
            metrics["bbox_touch_count"] <= 2 and
            (is_attachment or horizontal_overlap >= 0.30)
        )
        is_furniture_block_peer = (
            strategy["type"] == "furniture" and
            shape_features["isBlockLike"] and
            primary_shape["isBlockLike"] and
            has_close_bottom_band(shape_features, primary_shape) and
            mask_overlap_ratio >= 0.18 and
            metrics["target_fill_ratio"] <= 0.30 and
            metrics["mask_area_ratio"] <= 0.44 and
            (is_attachment or distance <= strategy["max_attachment_distance"] * 1.4)
        )
        is_multi_entity_peer = (
            metrics.get("multi_entity_coverage", False) and
            strategy["type"] in HARD_EDGE_STRATEGIES.union({"table", "furniture"}) and
            metrics["mask_inside_target_ratio"] >= 0.90 and
            metrics["target_fill_ratio"] >= 0.55 and
            metrics["target_fill_ratio"] <= 0.96 and
            metrics["bbox_overlap_ratio"] >= 0.80
        )
        is_lighting_part = (
            strategy["type"] == "lighting" and
            (shape_features["isThinVertical"] or shape_features["isHorizontalSurface"]) and
            (mask_overlap_ratio >= 0.06 or (
                mask_touches_primary and
                distance <= strategy["max_attachment_distance"] and
                horizontal_overlap >= 0.22
            )) and
            metrics["target_fill_ratio"] <= 0.32 and
            metrics["mask_area_ratio"] <= 0.48
        )
        is_hard_edge_complete_part = (
            strategy["type"] in HARD_EDGE_STRATEGIES and
            metrics["mask_inside_target_ratio"] >= 0.90 and
            metrics["target_fill_ratio"] <= 0.42 and
            metrics["mask_area_ratio"] <= 0.60 and
            (
                is_attachment or
                mask_touches_primary or
                mask_overlap_ratio >= 0.04 or
                horizontal_overlap >= 0.18
            )
        )
        is_decor_compound_part = (
            strategy["type"] == "decor_arrangement" and
            metrics["target_fill_ratio"] <= 0.62 and
            metrics["mask_area_ratio"] <= 0.70 and
            metrics["mask_inside_target_ratio"] >= 0.62 and
            (
                is_attachment or
                distance <= strategy["max_attachment_distance"] * 2.2 or
                horizontal_overlap >= 0.12 or
                shape_features["isBlockLike"] or
                shape_features["isThinVertical"] or
                is_decor_base_shape(shape_features)
            )
        )
        is_soft_edge_part = (
            strategy["type"] == "soft_edge" and
            metrics["target_fill_ratio"] <= 0.62 and
            metrics["mask_area_ratio"] <= 0.78 and
            metrics["mask_inside_target_ratio"] >= 0.45 and
            (
                is_attachment or
                distance <= strategy["max_attachment_distance"] * 1.4 or
                horizontal_overlap >= 0.08
            )
        )
        is_decor_atomic_fragment = (
            strategy["type"] == "decor_atomic" and
            is_attachment and
            metrics["target_fill_ratio"] <= 0.24 and
            metrics["mask_area_ratio"] <= 0.28
        )
        is_hard_product_part = (
            strategy["type"] == "hard_product" and
            metrics["mask_inside_target_ratio"] >= 0.64 and
            metrics["target_fill_ratio"] <= 0.38 and
            metrics["mask_area_ratio"] <= 0.45 and
            metrics["bbox_touch_count"] <= 2 and
            (
                is_attachment or
                distance <= strategy["max_attachment_distance"] * 1.8 or
                horizontal_overlap >= 0.10 or
                shape_features["isBlockLike"]
            )
        )
        is_food_layout_contaminated = (
            strategy["type"] == "food_product" and
            not is_food_support_shape(shape_features) and
            (
                (
                    metrics["mask_inside_target_ratio"] < 0.62 and
                    metrics["bbox_touch_count"] >= 2 and
                    metrics["mask_area_ratio"] >= 0.72
                ) or
                (
                    metrics["mask_inside_target_ratio"] < 0.52 and
                    metrics["bbox_touch_count"] >= 2 and
                    metrics["target_fill_ratio"] >= 0.24
                ) or
                (
                    metrics["mask_inside_target_ratio"] < 0.58 and
                    metrics["target_fill_ratio"] >= 0.36 and
                    metrics["mask_area_ratio"] >= 0.45
                ) or
                (
                    current_bbox[1] <= ty1 + 4 and
                    shape_features["centerY"] <= 0.40 and
                    metrics["target_fill_ratio"] >= 0.30 and
                    metrics["mask_area_ratio"] >= 0.40
                )
            )
        )
        is_food_complete_candidate = (
            strategy["type"] == "food_product" and
            not is_food_layout_contaminated and
            metrics["target_fill_ratio"] >= 0.22 and
            metrics["target_fill_ratio"] <= 0.62 and
            metrics["mask_area_ratio"] <= 1.15 and
            metrics["mask_inside_target_ratio"] >= (0.64 if is_drink_layer else 0.72) and
            (metrics["center_inside"] or metrics["bbox_overlap_ratio"] >= 0.18) and
            shape_features["bottomBand"] <= 1.08 and
            (
                metrics["bbox_touch_count"] <= (3 if is_drink_layer else 2) or
                is_food_support_shape(shape_features) or
                (
                    has_flat_cleanup_context and
                    not is_drink_layer and
                    metrics["bbox_touch_count"] <= 2 and
                    metrics["mask_inside_target_ratio"] >= 0.82
                )
            )
        )
        is_food_small_fragment = (
            strategy["type"] == "food_product" and
            not is_food_layout_contaminated and
            metrics["target_fill_ratio"] <= 0.22 and
            metrics["mask_area_ratio"] <= 0.34 and
            metrics["mask_inside_target_ratio"] >= 0.68 and
            metrics["bbox_touch_count"] <= (3 if is_drink_layer else 2) and
            shape_features["bottomBand"] <= 1.02 and
            (
                is_attachment or
                distance <= strategy["max_attachment_distance"] * 1.8 or
                horizontal_overlap >= 0.12 or
                shape_features["isBlockLike"]
            ) and (
                is_food_support_shape(shape_features) or
                metrics["target_fill_ratio"] >= 0.14
            )
        )
        is_food_late_slot = strategy["type"] == "food_product" and len(selected) >= 4
        is_food_product_part = (
            is_food_complete_candidate or
            (is_food_small_fragment and not is_food_late_slot)
        )
        background_fill_limit = 0.78 if strategy["type"] == "food_product" else 0.55
        background_area_limit = 0.94 if strategy["type"] == "food_product" else 0.65
        background_touch_limit = 5 if strategy["type"] == "food_product" else 3
        if strategy["type"] == "table":
            # The wider table candidate can include the wall/floor inside the
            # semantic bbox. Keep the original conservative merge threshold;
            # the narrower primary plus table-part candidates preserve the legs.
            background_fill_limit = 0.55
            background_area_limit = 0.65
            background_touch_limit = 3
        elif strategy["type"] == "decor_arrangement":
            background_fill_limit = 0.86
            background_area_limit = 0.92
            background_touch_limit = 4
        elif strategy["type"] == "soft_edge":
            background_fill_limit = 0.88
            background_area_limit = 0.92
            background_touch_limit = 4
        is_background_like = (
            metrics["target_fill_ratio"] >= background_fill_limit or
            metrics["mask_area_ratio"] >= background_area_limit or
            metrics["bbox_touch_count"] >= background_touch_limit or
            is_food_layout_contaminated
        )
        if multi_entity_coverage:
            is_background_like = False
        if is_food_complete_candidate:
            is_background_like = False
        if strategy["type"] == "food_product" and is_food_support_shape(shape_features):
            is_background_like = False
        if is_decor_compound_part or is_soft_edge_part or is_hard_edge_complete_part:
            is_background_like = False
        shape_merge_allowed = (
            (strategy["type"] not in {"food_product", "furniture", "table", "lighting"} and is_attachment and is_small_part) or
            is_table_leg_part or
            is_table_surface_part or
            is_table_complete_peer or
            is_furniture_block_peer or
            is_multi_entity_peer or
            is_lighting_part or
            is_hard_edge_complete_part or
            is_decor_compound_part or
            is_soft_edge_part or
            is_decor_atomic_fragment or
            is_hard_product_part or
            is_food_product_part
        )

        if not shape_merge_allowed or is_background_like:
            if is_food_layout_contaminated:
                candidate["debug"]["rejectReason"] = "product_layout_contaminated"
            elif is_background_like:
                candidate["debug"]["rejectReason"] = "background_like"
            elif not is_attachment and not (is_table_leg_part or is_table_surface_part or is_furniture_block_peer):
                candidate["debug"]["rejectReason"] = "not_attachment"
            else:
                candidate["debug"]["rejectReason"] = "not_shape_peer"
            continue

        trial = np.maximum(merged, candidate["mask"])
        trial_binary = trial > 0.5
        trial_target_area = int(np.count_nonzero(trial_binary[ty1:ty2, tx1:tx2]))
        trial_fill_ratio = trial_target_area / target_area
        if trial_fill_ratio > strategy["max_merged_fill"] and not is_multi_entity_peer:
            candidate["debug"]["rejectReason"] = "merged_fill_limit"
            continue

        merged = trial
        selected.append(candidate)
        candidate["debug"]["selected"] = True
        candidate["debug"]["rejectReason"] = "table_leg_part" if is_table_leg_part else (
                "table_surface_part" if is_table_surface_part else (
                "furniture_block_peer" if is_furniture_block_peer else (
                "hard_edge_complete_part" if is_hard_edge_complete_part else (
                "decor_compound_part" if is_decor_compound_part else (
                    "decor_atomic_fragment" if is_decor_atomic_fragment else (
                        "hard_product_part" if is_hard_product_part else (
                            "soft_edge_part" if is_soft_edge_part else (
                            "food_product_part" if is_food_product_part else "attachment")
                        )
                    )
                )
                )
            )
            )
        )

    for candidate in debug_candidates:
        if not candidate["selected"] and not candidate["rejectReason"]:
            candidate["rejectReason"] = "not_selected"

    merged_binary = merged > 0.5
    merged_target_area = int(np.count_nonzero(merged_binary[ty1:ty2, tx1:tx2]))
    target_fill_ratio = merged_target_area / target_area
    background_risk_limit = 0.95 if strategy["type"] == "food_product" else MAX_BACKGROUND_RISK_FILL_RATIO
    has_multi_entity_selected = any(
        item["metrics"].get("multi_entity_coverage") for item in selected
    )
    if target_fill_ratio > background_risk_limit and len(selected) > 1 and not has_multi_entity_selected:
        print(f"Merged mask background risk fill={target_fill_ratio:.3f}; fallback to primary mask")
        selected = [primary]
        merged = primary["mask"]
        merged_binary = merged > 0.5
        merged_target_area = int(np.count_nonzero(merged_binary[ty1:ty2, tx1:tx2]))
        target_fill_ratio = merged_target_area / target_area

    flat_ad_cleanup_count = 0
    flat_ad_cleaned_fill_ratio = None
    if strategy["type"] == "food_product" and target_fill_ratio > 0.12:
        label_cleanup_mask, label_cleanup_count = build_food_label_cleanup_mask(
            layer_meta or {},
            context_layers or [],
            target_bbox,
            img_w,
            img_h
        )
        cleanup_mask, cleanup_count = build_flat_ad_cleanup_mask(
            layer_meta or {},
            context_layers or [],
            target_bbox,
            img_w,
            img_h
        )
        print(
            f"Food cleanup context: labels={label_cleanup_count} flat={cleanup_count} accepted=False fill={target_fill_ratio:.3f}"
        )

    if exclude_reliable and target_fill_ratio > 0.18:
        exclude_mask = np.ones((img_h, img_w), dtype=np.float32)
        for entry in exclude_bboxes:
            x1, y1, x2, y2 = get_exclude_bbox(entry)
            exclude_mask[y1:y2, x1:x2] = 0
        cleaned = merged * exclude_mask
        cleaned_binary = cleaned > 0.5
        cleaned_target_area = int(np.count_nonzero(cleaned_binary[ty1:ty2, tx1:tx2]))
        cleaned_fill_ratio = cleaned_target_area / target_area
        # Do not let semantic exclusion erase the target. It is a cleanup pass,
        # not a hard cut, because semantic bboxes are often coarse/overlapping.
        if cleaned_fill_ratio >= max(0.08, target_fill_ratio * 0.45):
            merged = cleaned
            target_fill_ratio = cleaned_fill_ratio

    score = max(0.0, min(1.0, 1.0 - abs(target_fill_ratio - 0.42)))
    high_coverage_selected = bool(primary["metrics"].get("high_coverage_entity"))
    quality_gate = build_quality_gate(
        score,
        primary["score"],
        target_fill_ratio,
        selected,
        strategy["type"],
        high_coverage_entity=high_coverage_selected
    )
    strategy_profile = str((layer_meta or {}).get("extractionProfile", "")).lower() or strategy["type"]
    quality_reason = ",".join(quality_gate["issues"]) if quality_gate["issues"] else (
        f"semantic_{strategy['type']}_primary_mask_with_attachments"
    )
    quality = {
        "status": "low_quality" if forced_wall_art_fallback else (quality_gate["status"] if selected else "failed"),
        "score": round(float(score), 3),
        "maskCount": len(selected),
        "targetFillRatio": round(float(target_fill_ratio), 3),
        "primaryScore": round(float(primary["score"]), 3),
        "candidateCount": len(candidates),
        "excludeBoxCount": len(exclude_bboxes),
        "excludeReliable": exclude_reliable,
        "hasFlatAdCleanupContext": has_flat_cleanup_context,
        "flatAdCleanupCount": flat_ad_cleanup_count,
        "flatAdCleanedFillRatio": round(float(flat_ad_cleaned_fill_ratio), 3) if flat_ad_cleaned_fill_ratio is not None else None,
        "strategy": strategy["type"],
        "strategyProfile": strategy_profile,
        "selectedIndexes": [item["metrics"]["index"] for item in selected],
        "highCoverageEntity": high_coverage_selected,
        "debugCandidates": sorted(debug_candidates, key=lambda item: item["score"], reverse=True)[:12],
        "reason": "wall_art_quality_fallback" if forced_wall_art_fallback else quality_reason,
        "runtimeAction": "accept" if forced_wall_art_fallback else quality_gate["runtimeAction"],
        "shouldGenerateRuntimeLayer": True if forced_wall_art_fallback else (quality_gate["shouldGenerateRuntimeLayer"] if selected else False),
        "needsHigherPrecision": False if forced_wall_art_fallback else (quality_gate["needsHigherPrecision"] or not selected),
        "issues": ["wall_art_quality_fallback"] if forced_wall_art_fallback else (quality_gate["issues"] if selected else ["no_selected_mask"]),
        "recommendedEngine": quality_gate["recommendedEngine"]
    }

    return merged, len(selected), quality

def base64_to_cv2(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    # Clean up whitespace and newlines
    base64_str = base64_str.replace(" ", "").replace("\n", "").replace("\r", "")
    
    # Add padding if necessary
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += "=" * (4 - missing_padding)
        
    img_bytes = base64.b64decode(base64_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def cv2_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

def normalize_requested_engine(engine_name):
    value = str(engine_name or "").strip().lower()
    if value in {"sam", "hq_sam", "high_precision_sam", "ultralytics_sam", "sam_b"}:
        return "sam"
    return "sam"

def normalize_result_masks(results, img_w, img_h, interpolation=cv2.INTER_NEAREST, debug_label=None):
    if len(results) == 0 or results[0].masks is None:
        return np.empty((0, img_h, img_w), dtype=np.float32)

    normalized_masks = []
    raw_masks = results[0].masks.data.cpu().numpy()
    raw_shapes = set()
    for mask in raw_masks:
        raw_shapes.add(tuple(mask.shape))
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=interpolation)
        normalized_masks.append(mask.astype(np.float32))

    if debug_label:
        interpolation_name = "linear" if interpolation == cv2.INTER_LINEAR else "nearest"
        print(
            f"SAM mask raster for {debug_label}: raw={sorted(raw_shapes)} "
            f"target=({img_h}, {img_w}) resample={interpolation_name}"
        )

    if not normalized_masks:
        return np.empty((0, img_h, img_w), dtype=np.float32)
    return np.stack(normalized_masks, axis=0)


def detect_multi_entity_candidate_disagreement(candidate_masks, target_bbox, strategy_type):
    """Detect when B offers a single-part mask and a larger multi-part mask.

    A bbox can intentionally describe a group of separate hard objects. In
    that case the larger mask is not automatically background: a small primary
    plus a high-coverage peer is a useful signal that B needs a second opinion.
    """
    if strategy_type not in HARD_EDGE_STRATEGIES and strategy_type not in {"table", "furniture"}:
        return False, ""
    if candidate_masks is None or len(candidate_masks) < 2:
        return False, ""

    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target_area = max(1, (x2 - x1) * (y2 - y1))
    metrics = []
    for candidate in candidate_masks:
        binary = np.asarray(candidate > 0.5, dtype=bool)
        area = int(np.count_nonzero(binary))
        if area <= 0:
            continue
        inside_pixels = int(np.count_nonzero(binary[y1:y2, x1:x2]))
        bbox = mask_bbox(candidate)
        if not bbox:
            continue
        metrics.append({
            "fill": inside_pixels / target_area,
            "inside": inside_pixels / max(1, area),
            "area": area / target_area,
            "overlap": intersection_area(bbox, target_bbox) / target_area,
        })
    if len(metrics) < 2:
        return False, ""

    for smaller in metrics:
        if not (0.16 <= smaller["fill"] <= 0.62 and smaller["inside"] >= 0.94):
            continue
        for larger in metrics:
            if larger is smaller:
                continue
            if (
                larger["fill"] >= smaller["fill"] + 0.22 and
                larger["fill"] >= 0.68 and
                larger["inside"] >= 0.90 and
                larger["area"] <= 1.08 and
                larger["overlap"] >= 0.80
            ):
                return True, (
                    f"multi_entity_disagreement={smaller['fill']:.3f}->"
                    f"{larger['fill']:.3f}"
                )
    return False, ""


def should_escalate_sam_to_l(candidate_masks, target_bbox, strategy_type):
    """Route only difficult B-model masks to the larger SAM-L model."""
    if strategy_type in {"food_product", "soft_edge"}:
        return False, "strategy_prefers_b"
    if candidate_masks is None or len(candidate_masks) == 0:
        return True, "no_b_candidates"

    multi_entity, multi_entity_reason = detect_multi_entity_candidate_disagreement(
        candidate_masks,
        target_bbox,
        strategy_type
    )
    if multi_entity:
        return True, multi_entity_reason

    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target_area = max(1, (x2 - x1) * (y2 - y1))
    # Evaluate the same likely-primary mask that will be used by B/L
    # arbitration. Looking only for the largest fill can select a background
    # mask and incorrectly suppress the L upgrade.
    reference = choose_b_reference_mask(
        candidate_masks,
        target_bbox,
        strategy_type=strategy_type
    )
    if reference is None:
        return True, "no_valid_b_reference"

    binary = reference > 0.5
    area = int(np.count_nonzero(binary))
    inside_pixels = int(np.count_nonzero(binary[y1:y2, x1:x2]))
    best_inside = inside_pixels / max(1, area)
    best_fill = inside_pixels / target_area
    inverted = (~binary[y1:y2, x1:x2]).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    best_holes = 0
    for label in range(1, count):
        hx, hy, hw, hh, hole_area = [int(value) for value in stats[label]]
        if hx > 0 and hy > 0 and hx + hw < inverted.shape[1] and hy + hh < inverted.shape[0]:
            best_holes += hole_area

    hard_entity = strategy_type in HARD_EDGE_STRATEGIES or strategy_type in {"table", "furniture"}
    fill_threshold = 0.56 if hard_entity else 0.48
    # At high resolution, a fixed hole count is too lenient for a large
    # textured object. Conversely, a pure percentage is too aggressive for a
    # small object. The hard-entity threshold catches real internal breaks
    # while leaving ordinary SAM raster noise on the B path.
    hole_threshold = max(
        1400 if hard_entity else 1200,
        int(target_area * (0.006 if hard_entity else 0.012))
    )
    if best_fill < fill_threshold:
        return True, f"low_b_fill={best_fill:.3f}"
    if best_holes > hole_threshold:
        return True, f"b_internal_gaps={best_holes}>{hole_threshold}"
    if best_inside < 0.94:
        return True, f"low_b_inside={best_inside:.3f}"
    return False, f"b_accepted_fill={best_fill:.3f}"


def choose_b_reference_mask(b_masks, target_bbox, strategy_type=None):
    """Choose the likely B primary from raw multimask output for arbitration."""
    if b_masks is None or len(b_masks) == 0:
        return None
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target_area = max(1, (x2 - x1) * (y2 - y1))
    best = None
    best_score = None
    for candidate in b_masks:
        binary = np.asarray(candidate > 0.5, dtype=bool)
        area = int(np.count_nonzero(binary))
        if area <= 0:
            continue
        inside_area = int(np.count_nonzero(binary[y1:y2, x1:x2]))
        fill = inside_area / target_area
        inside_ratio = inside_area / area
        if strategy_type in HARD_EDGE_STRATEGIES or strategy_type in {"table", "furniture"}:
            # Prefer the plausible primary object when B also returns a
            # bbox-filling envelope. The latter may be a group or background;
            # L arbitration can decide whether its extra components are real.
            score = (
                inside_ratio * 2.2 +
                (1.0 - min(1.0, abs(fill - 0.46))) -
                max(0.0, fill - 0.68) * 1.2
            )
        else:
            score = (inside_ratio * 1.8) + min(fill, 0.82) - max(0.0, fill - 0.82) * 2.5
        if best_score is None or score > best_score:
            best = np.asarray(candidate, dtype=np.float32)
            best_score = score
    return best


def arbitrate_sam_b_l_masks(b_masks, l_masks, target_bbox, strategy_type=None):
    """Accept L only where it improves B without introducing detached growth."""
    if l_masks is None or len(l_masks) == 0:
        return b_masks, "b_no_l_candidates"
    if b_masks is None or len(b_masks) == 0:
        return l_masks, "l_fallback_no_b_candidates"

    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target_area = max(1, (x2 - x1) * (y2 - y1))
    reference = choose_b_reference_mask(b_masks, target_bbox, strategy_type=strategy_type)
    if reference is None:
        return l_masks, "l_fallback_no_b_reference"
    base = reference > 0.5
    base_area = max(1, int(np.count_nonzero(base[y1:y2, x1:x2])))
    best = base
    best_score = 0.0
    best_added = 0
    for candidate in l_masks:
        l_binary = np.asarray(candidate > 0.5, dtype=bool)
        l_binary[:y1] = False
        l_binary[y2:] = False
        l_binary[:, :x1] = False
        l_binary[:, x2:] = False
        if not np.any(l_binary):
            continue
        overlap = l_binary & base
        added = l_binary & (~base)
        added_area = int(np.count_nonzero(added))
        if added_area <= 0:
            continue
        preserve = int(np.count_nonzero(overlap[y1:y2, x1:x2])) / base_area
        l_fill = int(np.count_nonzero(l_binary[y1:y2, x1:x2])) / target_area
        # Added L pixels must attach to B or lie in a narrow neighborhood of it.
        attach = cv2.dilate(
            base.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            iterations=1
        ) > 0
        detached = int(np.count_nonzero(added & (~attach)))
        detached_ratio = detached / max(1, added_area)
        growth = (base_area + added_area) / base_area
        added_components = []
        added_count, added_labels, added_stats, _ = cv2.connectedComponentsWithStats(
            added.astype(np.uint8),
            connectivity=8
        )
        for component_label in range(1, added_count):
            ax, ay, aw, ah, component_area = [int(value) for value in added_stats[component_label]]
            if component_area <= 0:
                continue
            component = added_labels == component_label
            component_inside = int(np.count_nonzero(component[y1:y2, x1:x2])) / component_area
            added_components.append((component_area, component_inside, aw, ah))
        robust_added = [
            item for item in added_components
            if item[0] >= max(64, int(target_area * 0.018)) and item[1] >= 0.90
        ]
        tiny_added = sum(item[0] for item in added_components if item not in robust_added)
        l_count, l_labels, l_stats, _ = cv2.connectedComponentsWithStats(
            l_binary[y1:y2, x1:x2].astype(np.uint8),
            connectivity=8
        )
        l_substantial_components = 0
        for l_label in range(1, l_count):
            _, _, _, _, l_area = [int(value) for value in l_stats[l_label]]
            if l_area >= max(64, int(target_area * 0.018)):
                l_substantial_components += 1
        separate_entity_peer = (
            strategy_type in HARD_EDGE_STRATEGIES or strategy_type in {"table", "furniture"}
        ) and len(robust_added) >= 1 and l_substantial_components >= 2
        multi_entity_growth = (
            strategy_type in HARD_EDGE_STRATEGIES or strategy_type in {"table", "furniture"}
        ) and bool(robust_added) and len(added_components) <= 3 and tiny_added <= max(32, int(added_area * 0.12))
        allowed_growth = 2.25 if multi_entity_growth else 1.32
        allowed_detached = 1.0 if separate_entity_peer else (0.55 if multi_entity_growth else 0.10)
        if preserve < 0.985 or detached_ratio > allowed_detached or growth > allowed_growth or l_fill > 0.92:
            continue
        score = (added_area / base_area) - (0.0 if separate_entity_peer else (detached_ratio * 2.0))
        if score > best_score:
            best = base | added
            best_score = score
            best_added = added_area

    if best_added <= 0:
        return b_masks, "b_kept_l_rejected"
    merged = np.zeros_like(b_masks[0], dtype=np.float32)
    merged[best] = 1.0
    return np.stack([merged], axis=0), f"hybrid_added={best_added}"


def run_upscaled_hard_edge_bbox_inference(
    img,
    prompt_bbox,
    target_bbox,
    img_w,
    img_h,
    layer_name
):
    """Run hard-object SAM on an enlarged local crop for finer mask sampling."""
    px1, py1, px2, py2 = [int(value) for value in prompt_bbox]
    px1 = clamp(px1, 0, img_w - 1)
    py1 = clamp(py1, 0, img_h - 1)
    px2 = clamp(px2, px1 + 1, img_w)
    py2 = clamp(py2, py1 + 1, img_h)
    crop = img[py1:py2, px1:px2]
    crop_h, crop_w = crop.shape[:2]
    if crop.size == 0 or crop_w < 2 or crop_h < 2:
        return None

    scale = HARD_EDGE_LOCAL_SCALE
    max_upscaled_side = 1800
    scale = min(scale, max_upscaled_side / max(crop_w, crop_h))
    scale = max(1.0, scale)
    up_w = max(crop_w, int(round(crop_w * scale)))
    up_h = max(crop_h, int(round(crop_h * scale)))
    if scale <= 1.01:
        return None

    upscaled = cv2.resize(crop, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    local_bbox = [
        clamp(int(round((tx1 - px1) * scale)), 0, up_w - 1),
        clamp(int(round((ty1 - py1) * scale)), 0, up_h - 1),
        clamp(int(round((tx2 - px1) * scale)), 1, up_w),
        clamp(int(round((ty2 - py1) * scale)), 1, up_h)
    ]

    try:
        results = run_sam_bbox_inference(
            upscaled,
            local_bbox,
            multimask_output=True,
            imgsz=HARD_EDGE_SAM_IMGSZ
        )
        upscaled_masks = normalize_result_masks(
            results,
            up_w,
            up_h,
            interpolation=cv2.INTER_LINEAR,
            debug_label=f"{layer_name} strategy=local_upscaled"
        )
    except Exception as error:
        print(f"Local upscaled SAM failed for {layer_name}: {error}")
        return None

    if upscaled_masks is None or len(upscaled_masks) == 0:
        return None

    local_masks = []
    for mask in upscaled_masks:
        local_mask = cv2.resize(
            mask,
            (crop_w, crop_h),
            interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        full_mask = np.zeros((img_h, img_w), dtype=np.float32)
        full_mask[py1:py2, px1:px2] = local_mask
        local_masks.append(full_mask)

    print(
        f"Local upscaled SAM for {layer_name}: "
        f"crop=({crop_w},{crop_h}) target={local_bbox} "
        f"scale={scale:.2f} output=({up_w},{up_h})"
    )
    return np.stack(local_masks, axis=0)


def embed_local_masks_into_full_image(local_masks, crop_bounds, img_w, img_h):
    if local_masks is None or len(local_masks) == 0:
        return np.empty((0, img_h, img_w), dtype=np.float32)

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_h = max(0, crop_y2 - crop_y1)
    crop_w = max(0, crop_x2 - crop_x1)
    if crop_w <= 0 or crop_h <= 0:
        return np.empty((0, img_h, img_w), dtype=np.float32)

    embedded = []
    for mask in local_masks:
        if mask.shape != (crop_h, crop_w):
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        full_mask = np.zeros((img_h, img_w), dtype=np.float32)
        full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask.astype(np.float32)
        embedded.append(full_mask)

    if not embedded:
        return np.empty((0, img_h, img_w), dtype=np.float32)
    return np.stack(embedded, axis=0)


def mask_iou(mask_a, mask_b):
    a = mask_a > 0.5
    b = mask_b > 0.5
    intersection = int(np.count_nonzero(a & b))
    if intersection <= 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return intersection / max(1, union)


def append_unique_masks(mask_list, new_masks, min_pixels=36, dedupe_iou=0.94):
    if new_masks is None or len(new_masks) == 0:
        return

    for mask in new_masks:
        mask_binary = mask > 0.5
        if int(np.count_nonzero(mask_binary)) < min_pixels:
            continue

        duplicate = False
        for existing in mask_list:
            if mask_iou(existing, mask) >= dedupe_iou:
                duplicate = True
                break
        if duplicate:
            continue

        mask_list.append(mask.astype(np.float32))


def prefilter_food_candidate_masks(candidate_masks, target_bbox, max_candidates=14):
    if candidate_masks is None or len(candidate_masks) == 0:
        return candidate_masks

    tx1, ty1, tx2, ty2 = target_bbox
    target_area = max(1, bbox_area(target_bbox))
    ranked = []

    for index, mask in enumerate(candidate_masks):
        mask_binary = mask > 0.5
        if not np.any(mask_binary):
            continue

        current_bbox = mask_bbox(mask_binary)
        if not current_bbox:
            continue

        mask_area = int(np.count_nonzero(mask_binary))
        if mask_area < 36:
            continue

        target_mask_area = int(np.count_nonzero(mask_binary[ty1:ty2, tx1:tx2]))
        target_fill_ratio = target_mask_area / target_area
        if target_fill_ratio < 0.01:
            continue

        inside_ratio = target_mask_area / max(1, mask_area)
        bbox_overlap_ratio = intersection_area(current_bbox, target_bbox) / target_area
        bbox_touch_count = int(current_bbox[0] <= tx1 + 2) + int(current_bbox[1] <= ty1 + 2) + int(current_bbox[2] >= tx2 - 2) + int(current_bbox[3] >= ty2 - 2)
        shape_features = compute_shape_features(current_bbox, target_bbox, mask_area)
        fill_plausible = 1.0 - min(1.0, abs(target_fill_ratio - 0.40) / 0.40)
        outside_ratio = max(0.0, 1.0 - inside_ratio)

        pre_score = (
            0.22 +
            (inside_ratio * 0.30) +
            (bbox_overlap_ratio * 0.18) +
            (fill_plausible * 0.14) +
            (0.06 if is_food_support_shape(shape_features) else 0.0) +
            (0.04 if shape_features["bottomBand"] >= 0.74 else 0.0) -
            (outside_ratio * 0.18) -
            (max(0, bbox_touch_count - 2) * 0.10) -
            (0.16 if (bbox_touch_count >= 3 and inside_ratio < 0.92) else 0.0) -
            (0.10 if shape_features["bottomBand"] >= 1.02 else 0.0) -
            (0.08 if (shape_features["centerY"] <= 0.28 and target_fill_ratio < 0.08) else 0.0)
        )

        ranked.append({
            "index": index,
            "score": pre_score,
            "mask": mask.astype(np.float32)
        })

    if not ranked:
        return candidate_masks

    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = [item["mask"] for item in ranked[:max_candidates]]
    return np.stack(selected, axis=0) if selected else candidate_masks


def build_food_sam_candidate_masks(img, target_bbox, img_w, img_h):
    candidate_masks = []

    try:
        full_bbox_results = run_sam_bbox_inference(
            img,
            target_bbox,
            multimask_output=True,
            imgsz=1024
        )
        full_bbox_masks = normalize_result_masks(full_bbox_results, img_w, img_h)
        append_unique_masks(candidate_masks, full_bbox_masks)
        print(f"Food full-image bbox candidates: {len(full_bbox_masks)}")
    except Exception as error:
        print(f"Food full-image bbox inference failed: {error}")

    crop, crop_bounds = crop_region_from_bbox(img, target_bbox, 0.04)
    if crop.size == 0:
        if not candidate_masks:
            return np.empty((0, img_h, img_w), dtype=np.float32)
        return np.stack(candidate_masks, axis=0)

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_h, crop_w = crop.shape[:2]
    local_bbox = [
        max(0, target_bbox[0] - crop_x1),
        max(0, target_bbox[1] - crop_y1),
        min(crop_w, target_bbox[2] - crop_x1),
        min(crop_h, target_bbox[3] - crop_y1)
    ]
    local_imgsz = choose_local_refine_imgsz(crop_w, crop_h)
    local_auto_imgsz = min(local_imgsz, 640)
    local_bbox_imgsz = min(local_imgsz, 768)

    try:
        local_auto_results = run_sam_auto_inference(
            crop,
            imgsz=local_auto_imgsz
        )
        local_auto_masks = normalize_result_masks(local_auto_results, crop_w, crop_h)
        append_unique_masks(
            candidate_masks,
            embed_local_masks_into_full_image(local_auto_masks, crop_bounds, img_w, img_h)
        )
        print(f"Food crop auto candidates: {len(local_auto_masks)}")
    except Exception as error:
        print(f"Food crop auto inference failed: {error}")

    try:
        local_bbox_results = run_sam_bbox_inference(
            crop,
            local_bbox,
            multimask_output=True,
            imgsz=local_bbox_imgsz
        )
        local_bbox_masks = normalize_result_masks(local_bbox_results, crop_w, crop_h)
        append_unique_masks(
            candidate_masks,
            embed_local_masks_into_full_image(local_bbox_masks, crop_bounds, img_w, img_h)
        )
        print(f"Food crop bbox candidates: {len(local_bbox_masks)}")
    except Exception as error:
        print(f"Food crop bbox inference failed: {error}")

    if not candidate_masks:
        return np.empty((0, img_h, img_w), dtype=np.float32)
    filtered = prefilter_food_candidate_masks(
        np.stack(candidate_masks, axis=0),
        target_bbox,
        max_candidates=14
    )
    print(f"Food candidate prefilter kept {len(filtered)} / {len(candidate_masks)}")
    return filtered

def build_cutout_entry(img, mask, crop_bbox, img_w, img_h, layer_id, extract_engine, quality, alpha_mask=None):
    layer_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    layer_img[:, :, 3] = alpha_mask if alpha_mask is not None else dilate_and_feather_mask(mask)

    bx1, by1, bx2, by2 = crop_bbox
    cropped_img = layer_img[by1:by2, bx1:bx2]
    if cropped_img.size == 0:
        return None

    layer_b64 = cv2_to_base64(cropped_img)
    norm_ymin = int((by1 / img_h) * 1000)
    norm_xmin = int((bx1 / img_w) * 1000)
    norm_ymax = int((by2 / img_h) * 1000)
    norm_xmax = int((bx2 / img_w) * 1000)

    return {
        "layerId": layer_id,
        "bbox": [norm_ymin, norm_xmin, norm_ymax, norm_xmax],
        "image": layer_b64,
        "width": int(bx2 - bx1),
        "height": int(by2 - by1),
        "extractEngine": extract_engine,
        "quality": quality
    }

def process_prompted_cutouts(
    img,
    pixel_bboxes,
    layer_ids,
    layer_metas,
    context_layers,
    mask_provider,
    engine_name,
    refine_masks=False,
    original_target_bboxes=None
):
    h, w = img.shape[:2]
    cutouts = []

    for i, target_bbox in enumerate(pixel_bboxes):
        if i >= len(layer_ids):
            break

        layer_meta = layer_metas[i] if isinstance(layer_metas, list) and i < len(layer_metas) else {}
        candidate_masks = mask_provider(target_bbox, layer_meta, i)
        if candidate_masks is None or len(candidate_masks) == 0:
            continue
        mask, selected_count, quality = select_and_merge_masks(
            candidate_masks,
            target_bbox,
            w,
            h,
            layer_meta,
            context_layers
        )
        strategy_name = quality.get("strategy") if quality else "unknown"
        strategy_profile = quality.get("strategyProfile") if quality else "unknown"
        print(
            f"Layer {layer_ids[i]} engine={engine_name} "
            f"semanticStrategy={strategy_name} profile={strategy_profile} "
            f"merged {selected_count} candidate masks"
        )
        if quality and quality.get("debugCandidates"):
            debug_summary = " | ".join([
                f"#{row['index']} s={row['score']} fill={row['fill']} excl={row['exclude']} "
                f"inside={row['inside']} area={row['area']} ov={row['bboxOverlap']} touch={row['touch']} "
                f"base={row.get('decorBase', False)} bot={row['shapeFeatures']['bottomBand']} "
                f"cy={row['shapeFeatures']['centerY']} block={row['shapeFeatures']['isBlockLike']} "
                f"thin={row['shapeFeatures']['isThinVertical']} sel={row['selected']} why={row['rejectReason']}"
                for row in quality["debugCandidates"][:8]
            ])
            print(f"Layer {layer_ids[i]} candidates: {debug_summary}")
        if mask is None:
            continue

        strategy_type = quality.get("strategy") if quality else None
        effective_bbox, bbox_evidence = derive_safe_entity_bbox(
            mask,
            target_bbox,
            strategy_type=strategy_type
        )
        if bbox_evidence:
            print(
                f"Entity bbox evidence extension for {layer_ids[i]}: "
                f"original={target_bbox} effective={effective_bbox} evidence={bbox_evidence}"
            )
            target_bbox = effective_bbox

        cleanup_candidates = None
        if quality and quality.get("postProcess"):
            cleanup_candidates = None

        if strategy_type in HARD_EDGE_STRATEGIES:
            target_area = max(1, bbox_area(target_bbox))
            mask_fill = int(np.count_nonzero(mask > 0.5)) / target_area
            if mask_fill < 0.84:
                mask, topology_repair = recover_entity_topology_gaps(
                    mask,
                    target_bbox
                )
                mask = constrain_mask_to_bbox(mask, target_bbox)
                print(
                    f"Entity topology recovery for {layer_ids[i]}: "
                    f"status={topology_repair['status']} "
                    f"components={topology_repair['components']} "
                    f"pixels={topology_repair['pixels']} "
                    f"fill={mask_fill:.3f}"
                )
        if strategy_type in {"table", "furniture"}:
            # Tables and textured furniture use the accepted SAM silhouette as
            # the source of truth. Do not run another completion pass before
            # matting: it can reinterpret floor, seams, or upholstery texture.
            mask = constrain_mask_to_bbox(mask, target_bbox)
            if strategy_type == "furniture":
                mask, furniture_cleanup = cleanup_furniture_mask(mask, target_bbox)
                mask = constrain_mask_to_bbox(mask, target_bbox)
                print(
                    f"Furniture protected cleanup for {layer_ids[i]}: "
                    f"holes={furniture_cleanup['holesFilled']} "
                    f"holePixels={furniture_cleanup['holePixels']} "
                    f"removed={furniture_cleanup['componentsRemoved']} "
                    f"removedPixels={furniture_cleanup['componentPixelsRemoved']} "
                    f"supportsPreserved={furniture_cleanup['supportsPreserved']}"
                )
                mask, protected_repair_pixels, protected_repair = recover_protected_furniture_gaps(
                    img,
                    mask,
                    target_bbox
                )
                mask = constrain_mask_to_bbox(mask, target_bbox)
                if protected_repair["gapsFilled"] or protected_repair["colorRejected"]:
                    print(
                        f"Furniture protected recovery for {layer_ids[i]}: "
                        f"gaps={protected_repair['gapsFilled']} "
                        f"pixels={protected_repair['gapPixels']} "
                        f"colorRejected={protected_repair['colorRejected']}"
                    )
                mask, micro_repair = recover_micro_entity_gaps(mask, target_bbox)
                mask = constrain_mask_to_bbox(mask, target_bbox)
                if micro_repair["components"] or micro_repair["status"] != "skipped:no_micro_gap":
                    print(
                        f"Entity micro-gap recovery for {layer_ids[i]}: "
                        f"status={micro_repair['status']} "
                        f"components={micro_repair['components']} "
                        f"pixels={micro_repair['pixels']}"
                    )
                mask, residual_repair = recover_residual_entity_gaps(mask, target_bbox)
                mask = constrain_mask_to_bbox(mask, target_bbox)
                if residual_repair["components"] or residual_repair["status"] != "skipped:no_residual_gap":
                    print(
                        f"Entity residual-gap recovery for {layer_ids[i]}: "
                        f"status={residual_repair['status']} "
                        f"components={residual_repair['components']} "
                        f"pixels={residual_repair['pixels']}"
                    )
        elif strategy_type in HARD_EDGE_STRATEGIES:
            # The SAM candidate has already passed the semantic shape gates.
            # Morphological cleanup can remove thin hard-object edges and
            # create the missing-corner artifact, so preserve this mask.
            mask = constrain_mask_to_bbox(mask, target_bbox)
            mask, completion_changed = recover_hard_edge_mask_with_points(
                img,
                mask,
                target_bbox,
                layer_ids[i]
            )
            if completion_changed:
                mask = constrain_mask_to_bbox(mask, target_bbox)
        else:
            cleaned_mask = cleanup_mask(mask, target_bbox)
            if np.any(cleaned_mask > 0.5):
                mask = cleaned_mask
            else:
                mask = mask

        if quality and quality.get("strategy") == "food_product":
            original_target_bbox = (
                original_target_bboxes[i]
                if isinstance(original_target_bboxes, list) and i < len(original_target_bboxes)
                else target_bbox
            )
            conflict_refined_mask, conflict_changed, conflict_debug = refine_food_mask_with_conflict_sam(
                img,
                mask,
                target_bbox,
                layer_meta or {},
                context_layers or [],
                original_target_bbox=original_target_bbox
            )
            if conflict_changed and np.any(conflict_refined_mask > 0.5):
                mask = conflict_refined_mask
                print(f"Food conflict-aware SAM refine accepted for {layer_ids[i]}")
            if conflict_debug:
                debug_summary = " | ".join([
                    f"{row.get('name', 'unknown')}:{row.get('status')}:{row.get('reason', '') or row.get('removed', '')}"
                    for row in conflict_debug
                ])
                print(f"Food conflict refine details for {layer_ids[i]}: {debug_summary}")

            detached_cleaned_mask, detached_removed = remove_food_detached_artifacts(img, mask, target_bbox)
            if detached_removed > 0 and np.any(detached_cleaned_mask > 0.5):
                print(f"Food detached artifact cleanup removed {detached_removed} component(s) for {layer_ids[i]}")
                mask = detached_cleaned_mask

            if quality.get("foodSelectionMode") != "sam_candidates_semantic_mask_selection":
                attached_layout_entries = collect_attached_layout_entries(
                    layer_meta or {},
                    context_layers or [],
                    target_bbox,
                    img.shape[1],
                    img.shape[0]
                )
                if attached_layout_entries:
                    layout_names = ", ".join([
                        str((entry.get("layer") or {}).get("name") or "unknown")
                        for entry in attached_layout_entries[:6]
                    ])
                    print(
                        f"Attached layout candidates for {layer_ids[i]}: "
                        f"{len(attached_layout_entries)} -> {layout_names}"
                    )
                layout_removed_pixels = 0
                layout_removed_count = 0
                for entry in attached_layout_entries[:6]:
                    layout_mask, layout_quality = segment_attached_layout_mask(
                        img,
                        entry,
                        context_layers or []
                    )
                    if layout_mask is None or not np.any(layout_mask > 0.5):
                        print(
                            f"Attached layout skip for {layer_ids[i]}: "
                            f"{str((entry.get('layer') or {}).get('name') or 'unknown')} no_mask"
                        )
                        continue
                    next_mask, changed, removed_pixels = subtract_attached_layout_from_food_mask(
                        mask,
                        layout_mask,
                        target_bbox,
                        layer_meta=entry.get("layer") or {},
                        entry_bbox=entry.get("bbox")
                    )
                    if not changed:
                        print(
                            f"Attached layout keep for {layer_ids[i]}: "
                            f"{str((entry.get('layer') or {}).get('name') or 'unknown')} removed=0"
                        )
                        continue
                    mask = cleanup_mask(next_mask, target_bbox)
                    layout_removed_count += 1
                    layout_removed_pixels += removed_pixels
                    print(
                        f"Attached layout subtract for {layer_ids[i]}: "
                        f"{str((entry.get('layer') or {}).get('name') or 'unknown')} removed={removed_pixels}"
                    )
                if layout_removed_count > 0:
                    print(
                        f"Attached layout subtraction removed {layout_removed_pixels} px "
                        f"across {layout_removed_count} layout mask(s) for {layer_ids[i]}"
                    )

        matte_cleanup_mask = None
        label_cleanup_mask = None
        flat_cleanup_mask = None
        if quality and quality.get("strategy") == "food_product":
            # Keep food extraction complete first. Cleanup of labels/base should be
            # a separate deterministic pass after we have a stable full subject.
            label_cleanup_mask = None
            flat_cleanup_mask = None
            matte_cleanup_mask = None

        local_refined = False
        initial_mask_area = int(np.count_nonzero(mask > 0.5))
        allow_local_refine = not (
            quality and quality.get("strategy") in {
                "food_product",
                "soft_edge",
                "table",
                *HARD_EDGE_STRATEGIES
            }
        )
        if refine_masks and engine_name.startswith("sam") and allow_local_refine:
            refine_cleanup_mask = None
            if quality and quality.get("strategy") == "food_product":
                refine_cleanup_mask = build_food_label_cleanup_mask(
                    layer_meta or {},
                    context_layers or [],
                    target_bbox,
                    img_w=img.shape[1],
                    img_h=img.shape[0]
                )[0]
            elif quality and quality.get("strategy") in {"hard_product", "layout_embedded_product"}:
                refine_cleanup_mask = build_exclude_mask(
                    build_exclude_bboxes(layer_meta or {}, context_layers or [], target_bbox, img.shape[1], img.shape[0]),
                    img.shape[1],
                    img.shape[0]
                )
            refined_mask, local_refined = refine_mask_with_local_sam(
                img,
                mask,
                target_bbox,
                cleanup_mask=refine_cleanup_mask,
                strategy_type=quality.get("strategy") if quality else None
            )
            if local_refined and np.any(refined_mask > 0.5):
                mask = cleanup_mask(refined_mask, target_bbox, strategy_type=quality.get("strategy") if quality else None)
                mask = constrain_mask_to_bbox(mask, target_bbox)
            if quality and quality.get("strategy") in HARD_EDGE_STRATEGIES:
                refined_area = int(np.count_nonzero(mask > 0.5)) if local_refined else initial_mask_area
                print(
                    f"Hard-edge local refine for {layer_ids[i]}: "
                    f"accepted={bool(local_refined)} area={initial_mask_area}->{refined_area}"
                )

        alpha_mask = generate_alpha_matte(
            img,
            mask,
            target_bbox,
            cleanup_mask=matte_cleanup_mask,
            strategy_type=quality.get("strategy") if quality else None,
            label_cleanup_mask=label_cleanup_mask,
            flat_cleanup_mask=flat_cleanup_mask
        )
        if quality and quality.get("strategy") == "soft_edge":
            alpha_mask = build_soft_edge_alpha(img, mask, target_bbox)
        if quality and quality.get("strategy") == "furniture":
            alpha_mask = build_hard_edge_alpha(mask, target_bbox)
            # Keep antialiasing inside the accepted furniture silhouette only.
            # Rasterizing a contour can otherwise place a fractional pixel on
            # the far side of a small hole or a tight concavity.
            alpha_mask = np.where(mask > 0.5, alpha_mask, 0).astype(np.uint8)
        if quality and quality.get("strategy") == "food_product":
            # GrabCut may classify a bright background patch immediately outside
            # the accepted SAM contour as probable foreground. Keep only a tiny
            # antialias guard around the semantic mask; never let matte restore
            # the layout spill rejected above.
            semantic_guard = cv2.dilate(
                (mask > 0.5).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1
            ) > 0
            alpha_mask = np.where(semantic_guard, alpha_mask, 0).astype(np.uint8)
        alpha_mask = np.asarray(
            constrain_mask_to_bbox(alpha_mask.astype(np.float32), target_bbox),
            dtype=np.uint8
        )
        if quality and quality.get("strategy") in {"table", "furniture"}:
            opaque_pixels = int(np.count_nonzero(alpha_mask >= 245))
            edge_pixels = int(np.count_nonzero((alpha_mask > 0) & (alpha_mask < 245)))
            print(
                f"{strategy_type.capitalize()} SAM safe matte for {layer_ids[i]}: "
                f"preservedMaskPixels={int(np.count_nonzero(mask > 0.5))} "
                f"opaque={opaque_pixels} antialiasedEdge={edge_pixels}"
            )
        elif quality and quality.get("strategy") in HARD_EDGE_STRATEGIES:
            opaque_pixels = int(np.count_nonzero(alpha_mask >= 245))
            edge_pixels = int(np.count_nonzero((alpha_mask > 0) & (alpha_mask < 245)))
            print(
                f"Hard-edge alpha for {layer_ids[i]}: "
                f"opaque={opaque_pixels} antialiasedEdge={edge_pixels}"
            )
        output_img = img
        if quality and quality.get("strategy") == "soft_edge":
            output_img = despill_soft_edge_image(
                img,
                alpha_mask,
                target_bbox,
                context_bbox=expand_bbox(*target_bbox, w, h)
            )
        elif quality and quality.get("strategy") in HARD_EDGE_STRATEGIES and quality.get("strategy") not in {"table", "furniture"}:
            output_img = despill_hard_edge_image(img, alpha_mask, target_bbox)
        if quality is None:
            quality = {}
        quality["postProcess"] = {
            "maskCleanup": True,
            "localRefine": bool(local_refined),
            "matting": (
                "sam_safe_matte"
                if strategy_type in {"table", "furniture"}
                else "opencv_grabcut"
            )
        }

        cutout = build_cutout_entry(
            output_img,
            mask,
            target_bbox,
            w,
            h,
            layer_ids[i],
            engine_name,
            quality,
            alpha_mask=alpha_mask
        )
        if cutout is not None:
            cutouts.append(cutout)

    return cutouts

@app.post("/segment")
async def segment(request: Request):
    try:
        data = await request.json()
        requested_engine = normalize_requested_engine(data.get("engine"))
        image_b64 = data.get("image")
        bboxes_norm = data.get("bboxes", [])
        layer_ids = data.get("layerIds", [])
        layer_metas = data.get("layers", [])
        context_layers = data.get("contextLayers", layer_metas)
        
        if not image_b64:
            return JSONResponse(status_code=400, content={"error": "No image provided"})

        img = base64_to_cv2(image_b64)
        print("Image received, size:", img.shape)
        h, w = img.shape[:2]
        
        # Determine if we should use bounding box prompts
        if bboxes_norm and len(bboxes_norm) > 0:
            print(f"Using {len(bboxes_norm)} bounding box prompts with engine={requested_engine}")
            # Convert 0-1000 normalized bboxes back to pixel coordinates [x1, y1, x2, y2]
            pixel_bboxes = []
            original_target_bboxes = []
            sam_prompt_bboxes = []
            legacy_food_output_count = 0
            for bbox in bboxes_norm:
                ymin_n, xmin_n, ymax_n, xmax_n = bbox
                y1 = int((ymin_n / 1000.0) * h)
                x1 = int((xmin_n / 1000.0) * w)
                y2 = int((ymax_n / 1000.0) * h)
                x2 = int((xmax_n / 1000.0) * w)
                output_bbox = [
                    clamp(x1, 0, w - 1),
                    clamp(y1, 0, h - 1),
                    clamp(x2, 1, w),
                    clamp(y2, 1, h)
                ]
                prompt_bbox = expand_bbox(*output_bbox, w, h)
                original_target_bboxes.append(output_bbox)
                layer_meta = layer_metas[len(pixel_bboxes)] if (
                    isinstance(layer_metas, list) and len(pixel_bboxes) < len(layer_metas)
                ) else {}
                is_food_product = get_layer_strategy(layer_meta or {}).get("type") == "food_product"
                # Restore the previous cloud-parity food path: food products
                # use the expanded bbox for candidate/output parity.
                pixel_bboxes.append(prompt_bbox if is_food_product else output_bbox)
                if is_food_product:
                    legacy_food_output_count += 1
                sam_prompt_bboxes.append(prompt_bbox)

            print(
                f"Using expanded SAM prompts by {int(BBOX_EXPAND_RATIO * 100)}% "
                "with strict original-bbox output clipping"
            )
            if legacy_food_output_count:
                print(
                    f"Food product legacy bbox parity enabled for {legacy_food_output_count} layer(s): "
                    "expanded prompt bbox is also used for output"
                )
            if requested_engine == "sam":
                def sam_mask_provider(target_bbox, layer_meta, index):
                    strategy_type = get_layer_strategy(layer_meta or {}).get("type")
                    prompt_bbox = sam_prompt_bboxes[index]
                    layer_name = layer_meta.get("name") or layer_ids[index] or index
                    if strategy_type == "food_product":
                        print(
                            f"SAM prompts for {layer_meta.get('name') or layer_ids[index] or index}: "
                            f"bbox-only strategy=food_product"
                        )
                        results = run_sam_bbox_inference(
                            img,
                            prompt_bbox,
                            multimask_output=True,
                            imgsz=1024,
                            model_variant="b"
                        )
                        return normalize_result_masks(results, w, h)

                    soft_edge_prompts = None
                    if strategy_type == "soft_edge":
                        soft_edge_prompts = build_soft_edge_prompt_inputs(img, target_bbox)
                        print(
                            f"SAM prompts for {layer_meta.get('name') or layer_ids[index] or index}: "
                            f"+{len(soft_edge_prompts['positive'])} -{len(soft_edge_prompts['negative'])} "
                            "strategy=soft_edge_color_guided"
                        )
                    else:
                        print(
                            f"SAM prompts for {layer_name}: "
                            f"bbox-only strategy={strategy_type}"
                        )
                    results = run_sam_bbox_inference(
                        img,
                        prompt_bbox,
                        multimask_output=True,
                        imgsz=(
                            SOFT_EDGE_SAM_IMGSZ
                            if strategy_type == "soft_edge"
                            else HARD_EDGE_SAM_IMGSZ
                            if strategy_type in HARD_EDGE_STRATEGIES
                            else 1024
                        ),
                        points=soft_edge_prompts["points"] if soft_edge_prompts else None,
                        labels=soft_edge_prompts["labels"] if soft_edge_prompts else None,
                        model_variant="b"
                    )
                    use_subpixel_masks = (
                        strategy_type == "soft_edge" or
                        strategy_type in HARD_EDGE_STRATEGIES
                    )
                    candidate_masks = normalize_result_masks(
                        results,
                        w,
                        h,
                        interpolation=cv2.INTER_LINEAR if use_subpixel_masks else cv2.INTER_NEAREST,
                        debug_label=(
                            f"{layer_meta.get('name') or layer_ids[index] or index} "
                            f"strategy={strategy_type}"
                        )
                    )
                    if soft_edge_prompts:
                        candidate_masks = filter_soft_edge_masks_by_points(candidate_masks, soft_edge_prompts)
                    if strategy_type not in {"food_product", "soft_edge"}:
                        escalate, escalation_reason = should_escalate_sam_to_l(
                            candidate_masks,
                            target_bbox,
                            strategy_type
                        )
                        layer_label = layer_meta.get("name") or layer_ids[index] or index
                        if escalate:
                            print(
                                f"SAM route for {layer_label}: model=B -> L "
                                f"reason={escalation_reason}"
                            )
                            try:
                                l_results = run_sam_bbox_inference(
                                    img,
                                    prompt_bbox,
                                    multimask_output=True,
                                    imgsz=(
                                        SOFT_EDGE_SAM_IMGSZ
                                        if strategy_type == "soft_edge"
                                        else HARD_EDGE_SAM_IMGSZ
                                    ),
                                    model_variant="l"
                                )
                                l_masks = normalize_result_masks(
                                    l_results,
                                    w,
                                    h,
                                    interpolation=cv2.INTER_LINEAR if use_subpixel_masks else cv2.INTER_NEAREST,
                                    debug_label=f"{layer_label} strategy={strategy_type} model=L"
                                )
                                candidate_masks, arbitration = arbitrate_sam_b_l_masks(
                                    candidate_masks,
                                    l_masks,
                                    target_bbox,
                                    strategy_type=strategy_type
                                )
                                print(
                                    f"SAM arbitration for {layer_label}: "
                                    f"{arbitration} candidates={len(candidate_masks)}"
                                )
                            except Exception as error:
                                print(f"SAM-L escalation failed for {layer_label}: {error}")
                        else:
                            print(
                                f"SAM route for {layer_label}: model=B "
                                f"reason={escalation_reason}"
                            )
                    return candidate_masks

                cutouts = process_prompted_cutouts(
                    img,
                    pixel_bboxes,
                    layer_ids,
                    layer_metas,
                    context_layers,
                    sam_mask_provider,
                    "sam_bbox_prompt",
                    refine_masks=True,
                    original_target_bboxes=original_target_bboxes
                )
                return JSONResponse(content={"success": True, "engine": "sam", "cutouts": cutouts})

            # Default FastSAM path: global candidate masks + existing merge logic.
            fastsam = get_fastsam_model()
            results = fastsam(img, retina_masks=True, imgsz=1024, conf=0.25, iou=0.9)
            masks = normalize_result_masks(results, w, h)

            def fastsam_mask_provider(target_bbox, layer_meta, index):
                return masks

            cutouts = process_prompted_cutouts(
                img,
                pixel_bboxes,
                layer_ids,
                layer_metas,
                context_layers,
                fastsam_mask_provider,
                "fastsam_multi_mask",
                refine_masks=False,
                original_target_bboxes=original_target_bboxes
            )
            return JSONResponse(content={"success": True, "engine": "fastsam", "cutouts": cutouts})
        else:
            # Fallback to everything=True segmentation if no bboxes provided
            if requested_engine == "sam":
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "高精 SAM 当前仅支持带 bbox prompt 的分割请求"}
                )

            print("No bounding boxes provided, fallback to segment everything")
            fastsam = get_fastsam_model()
            results = fastsam(img, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            
            layers = []
            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.data.cpu().numpy()
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    layer_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    layer_img[:, :, 3] = mask * 255
                    
                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped_img = layer_img[y1:y2, x1:x2]
                    
                    if cropped_img.size == 0:
                        continue
                        
                    layer_b64 = cv2_to_base64(cropped_img)
                    norm_ymin = int((y1 / h) * 1000)
                    norm_xmin = int((x1 / w) * 1000)
                    norm_ymax = int((y2 / h) * 1000)
                    norm_xmax = int((x2 / w) * 1000)
                    
                    layers.append({
                        "id": f"fastsam-layer-{i}-{int(time.time())}",
                        "name": f"FastSAM 层 {i+1}",
                        "layerType": "OBJECT",
                        "bbox": [norm_ymin, norm_xmin, norm_ymax, norm_xmax],
                        "image": layer_b64,
                        "assetStatus": "idle"
                    })
            
            return JSONResponse(content={"success": True, "engine": "fastsam", "layers": layers})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    print("Starting FastSAM Backend on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
