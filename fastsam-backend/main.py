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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
FASTSAM_MODEL_PATH = os.getenv("FASTSAM_MODEL_PATH", "FastSAM-x.pt")
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", "sam_b.pt")
print(f"Loading FastSAM model: {FASTSAM_MODEL_PATH}")
model = FastSAM(FASTSAM_MODEL_PATH)
print("Model loaded.")
sam_model = None

def get_sam_model():
    global sam_model
    if sam_model is not None:
        return sam_model

    print(f"Loading high precision SAM model: {SAM_MODEL_PATH}")
    sam_model = SAM(SAM_MODEL_PATH)
    print("High precision SAM model loaded.")
    return sam_model


def get_sam_predictor():
    sam = get_sam_model()
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
    masks=None
):
    predictor = get_sam_predictor()
    previous_imgsz = getattr(predictor.args, "imgsz", 1024)
    predictor.args.imgsz = imgsz
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
        predictor.args.imgsz = previous_imgsz


def run_sam_mask_refine_inference(
    img,
    points=None,
    labels=None,
    masks=None,
    imgsz=1024,
    multimask_output=True
):
    predictor = get_sam_predictor()
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
    imgsz=1024
):
    predictor = get_sam_predictor()
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
LOCAL_REFINE_EXPAND_RATIO = 0.22
LOCAL_REFINE_MAX_SIDE = 1280
MASK_COMPONENT_MIN_PIXELS = 36
MASK_KEEP_COMPONENT_MAX_GAP = 36
MASK_HOLE_MIN_AREA = 24
GRABCUT_ITER_COUNT = 2
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
            "max_fill": 0.82,
            "max_merged_fill": 0.78,
            "max_attachment_distance": 18,
            "allow_attachments": True,
            "prefer_rectangular": False,
            "max_masks": 7,
            "require_overlap_for_attachments": False
        }

    if extraction_profile == "multi_part_hard_object":
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
            "max_fill": 0.72,
            "max_merged_fill": 0.62,
            "max_attachment_distance": 8,
            "allow_attachments": False,
            "prefer_rectangular": True,
            "max_masks": 1,
            "require_overlap_for_attachments": True
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


def select_clean_components(mask_binary, target_bbox):
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
        close_to_primary = bbox_distance(candidate["bbox"], primary["bbox"]) <= MASK_KEEP_COMPONENT_MAX_GAP
        overlaps_target = candidate["overlap_ratio"] >= 0.015 or candidate["inside_ratio"] >= 0.20
        if close_to_primary and overlaps_target and candidate["area"] >= primary["area"] * 0.02:
            keep_labels.add(candidate["label"])

    return np.isin(labels, list(keep_labels))


def cleanup_mask(mask, target_bbox):
    mask_binary = mask > 0.5
    if not np.any(mask_binary):
        return mask.astype(np.float32)

    kernel_size = compute_cleanup_kernel(target_bbox)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = select_clean_components(mask_binary, target_bbox).astype(np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    if kernel_size >= 2:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = fill_small_holes(cleaned > 0).astype(np.uint8)
    return cleaned.astype(np.float32)


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

        is_flat_graphic = (
            area_ratio <= 0.48 and
            gray_std <= 32.0 and
            color_std <= 30.0 and
            0.45 <= aspect <= 2.35 and
            overlap_ratio <= 0.32 and
            distance <= max(96.0, (primary_area ** 0.5) * 0.75)
        )

        if is_flat_graphic:
            removed += 1
            continue

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


def build_refine_prompts(coarse_mask, local_bbox, cleanup_mask=None, strategy_type=None):
    coarse_binary = coarse_mask > 0.5
    negative_points = []

    if cleanup_mask is not None:
        cleanup_binary = cleanup_mask > 0
        cleanup_negative_limit = 2 if strategy_type == "food_product" else 4
        negative_points.extend(build_negative_points_from_mask(cleanup_binary, max_points=cleanup_negative_limit))
    if strategy_type != "food_product":
        negative_points.extend(build_boundary_negative_points(coarse_binary, local_bbox, max_points=4))

    if negative_points:
        batched_points = [normalize_points(negative_points)]
        batched_labels = [[0] * len(negative_points)]
    else:
        batched_points = None
        batched_labels = None

    # Use the first-round mask as the main prompt, and only add a few negative
    # points to trim obvious contamination.
    mask_prompt = coarse_binary.astype(np.uint8)
    return batched_points, batched_labels, mask_prompt


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
    refine_imgsz = choose_local_refine_imgsz(crop_w, crop_h)

    # Mask prompt must align with the crop. Ultralytics SAM expects a 2D mask per prompt.
    mask_prompt = np.asarray(mask_prompt, dtype=np.uint8)
    if mask_prompt.shape != (crop_h, crop_w):
        mask_prompt = cv2.resize(mask_prompt, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    mask_prompt = mask_prompt[None, :, :]

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
            masks=mask_prompt,
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
    binary_mask = (mask > 0.5).astype(np.uint8)
    if not np.any(binary_mask):
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

def build_quality_gate(score, primary_score, target_fill_ratio, selected, strategy_type):
    min_score = MIN_RUNTIME_ACCEPT_SCORE
    min_primary_score = MIN_RUNTIME_ACCEPT_PRIMARY_SCORE
    min_fill_ratio = MIN_RUNTIME_ACCEPT_FILL_RATIO
    max_fill_ratio = MAX_RUNTIME_ACCEPT_FILL_RATIO

    if strategy_type == "food_product":
        min_score = 0.28
        min_primary_score = 0.22
        min_fill_ratio = 0.05
        max_fill_ratio = 0.82

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


def collect_food_conflict_entries_for_mask(food_mask, layer_meta, context_layers, target_bbox, img_w, img_h, max_entries=3):
    entries = collect_attached_layout_entries(layer_meta or {}, context_layers or [], target_bbox, img_w, img_h)
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
        if overlap_area <= 0:
            continue

        entry_area = max(1, int(np.count_nonzero(entry_mask)))
        overlap_ratio = overlap_area / entry_area
        if overlap_ratio < 0.02:
            continue

        entry_area_ratio = entry_area / max(1, bbox_area(target_bbox))
        if entry_area_ratio >= 0.58:
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
    return conflicts[:max_entries]


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


def refine_food_mask_with_conflict_sam(img, coarse_mask, target_bbox, layer_meta, context_layers):
    img_h, img_w = img.shape[:2]
    current_mask = (coarse_mask > 0.5).astype(np.float32)
    changed = False
    debug_rows = []

    conflict_entries = collect_food_conflict_entries_for_mask(
        current_mask,
        layer_meta or {},
        context_layers or [],
        target_bbox,
        img_w,
        img_h,
        max_entries=2
    )
    if not conflict_entries:
        return current_mask, False, debug_rows

    for entry in conflict_entries:
        current_binary = current_mask > 0.5
        current_bbox = mask_bbox(current_binary)
        if not current_bbox:
            break

        entry_bbox = entry.get("bbox")
        if not entry_bbox:
            continue

        conflict_mask = entry.get("entryMask")
        if conflict_mask is None or not np.any(conflict_mask & current_binary):
            continue

        conflict_bbox = mask_bbox(conflict_mask)
        local_focus_bbox = merge_bboxes(current_bbox, conflict_bbox or entry_bbox)
        crop, crop_bounds = crop_region_from_bbox(img, local_focus_bbox, 0.12)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
        crop_h, crop_w = crop.shape[:2]
        if crop_h <= 2 or crop_w <= 2:
            continue

        coarse_crop = current_binary[crop_y1:crop_y2, crop_x1:crop_x2]
        conflict_crop = conflict_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        if not np.any(coarse_crop) or not np.any(conflict_crop & coarse_crop):
            continue

        positive_points = build_food_conflict_positive_points(coarse_crop, avoid_mask=conflict_crop)
        negative_points = build_negative_points_from_mask(conflict_crop, max_points=3)
        if not negative_points:
            local_conflict_bbox = mask_bbox(conflict_crop)
            if local_conflict_bbox:
                negative_points = sample_points_in_bbox(local_conflict_bbox, [(0.5, 0.5), (0.35, 0.5), (0.65, 0.5)])

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

        if not prompt_points:
            continue

        try:
            results = run_sam_mask_refine_inference(
                crop,
                imgsz=min(640, choose_local_refine_imgsz(crop_w, crop_h)),
                points=[prompt_points],
                labels=[prompt_labels],
                multimask_output=True
            )
            candidate_masks = normalize_result_masks(results, crop_w, crop_h)
        except Exception as error:
            print(f"Food conflict SAM refine failed for {str((entry.get('layer') or {}).get('name') or 'unknown')}: {error}")
            continue

        best_candidate = None
        best_metrics = None
        for candidate_mask in candidate_masks:
            metrics = score_food_conflict_refine_candidate(candidate_mask, coarse_crop, conflict_crop)
            if not metrics:
                continue
            if metrics["preservedRatio"] < 0.86:
                continue
            if metrics["areaRatio"] < 0.84:
                continue
            if metrics["conflictRemovedRatio"] < 0.18:
                continue
            if best_metrics is None or metrics["score"] > best_metrics["score"]:
                best_candidate = candidate_mask
                best_metrics = metrics

        if not best_candidate or not best_metrics:
            debug_rows.append({
                "name": str((entry.get("layer") or {}).get("name") or "unknown"),
                "status": "rejected",
                "reason": "no_candidate_passed"
            })
            continue

        refined_full = current_binary.copy()
        refined_full[crop_y1:crop_y2, crop_x1:crop_x2] = best_candidate > 0.5
        refined_clean = cleanup_mask(refined_full.astype(np.float32), target_bbox) > 0.5
        if not np.any(refined_clean):
            continue

        original_area = max(1, int(np.count_nonzero(current_binary)))
        refined_area = int(np.count_nonzero(refined_clean))
        preserved_ratio = int(np.count_nonzero(refined_clean & current_binary)) / original_area
        if preserved_ratio < 0.84 or refined_area < original_area * 0.82:
            debug_rows.append({
                "name": str((entry.get("layer") or {}).get("name") or "unknown"),
                "status": "rejected",
                "reason": "preserve_guard"
            })
            continue

        current_mask = refined_clean.astype(np.float32)
        changed = True
        debug_rows.append({
            "name": str((entry.get("layer") or {}).get("name") or "unknown"),
            "status": "accepted",
            "removed": round(float(best_metrics["conflictRemovedRatio"]), 3),
            "preserved": round(float(best_metrics["preservedRatio"]), 3)
        })

    return current_mask, changed, debug_rows


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

def count_mask_in_strong_excludes(mask_binary, exclude_bboxes):
    strong_bboxes = [entry for entry in exclude_bboxes if is_strong_exclude(entry)]
    return count_mask_in_bboxes(mask_binary, strong_bboxes) if strong_bboxes else 0

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
    return (inside * 0.42) + (plausible_fill * 0.28) + (overlap * 0.22) + center_bonus + rectangular_bonus + shape_bonus - background_penalty


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
        candidate_variants = [
            ("raw", raw_binary),
            ("minus_price_text", raw_binary & (~context_priors["textMask"])),
            ("minus_price_only", raw_binary & (~context_priors["priceMask"]))
        ]
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

        is_probable_foreground = (
            target_fill_ratio > 0
            and target_fill_ratio <= strategy["max_fill"]
            and mask_inside_target_ratio >= MIN_MASK_INSIDE_TARGET_RATIO
            and MIN_MASK_AREA_RATIO_IN_BBOX <= mask_area_ratio <= MAX_MASK_AREA_RATIO_IN_BBOX
            and (bbox_overlap_ratio >= MIN_BBOX_OVERLAP_RATIO or center_inside)
            and shape_allowed
            and strong_exclude_mask_ratio < 0.22
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
            "mask_area": mask_area,
            "exclude_mask_ratio": exclude_mask_ratio,
            "strong_exclude_mask_ratio": strong_exclude_mask_ratio,
            "shape_features": shape_features
        }
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
            "candidate": bool(is_probable_foreground),
            "selected": False,
            "rejectReason": "" if is_probable_foreground else (
                "sibling_overlap" if strong_exclude_mask_ratio >= 0.22 else (shape_reject_reason or "gate")
            )
        }
        debug_candidates.append(debug_candidate)

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

    if not candidates:
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
        is_table_shape_part = (
            strategy["type"] == "table" and
            is_table_structure and
            is_table_related
        )
        is_furniture_block_peer = (
            strategy["type"] == "furniture" and
            shape_features["isBlockLike"] and
            primary_shape["isBlockLike"] and
            has_close_bottom_band(shape_features, primary_shape) and
            (is_attachment or distance <= strategy["max_attachment_distance"] * 1.5)
        )
        is_decor_compound_part = (
            strategy["type"] == "decor_arrangement" and
            metrics["target_fill_ratio"] <= 0.42 and
            metrics["mask_area_ratio"] <= 0.48 and
            (
                is_attachment or
                distance <= strategy["max_attachment_distance"] * 2.2 or
                horizontal_overlap >= 0.12 or
                shape_features["isBlockLike"] or
                shape_features["isThinVertical"] or
                is_decor_base_shape(shape_features)
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
        is_background_like = (
            metrics["target_fill_ratio"] >= background_fill_limit or
            metrics["mask_area_ratio"] >= background_area_limit or
            metrics["bbox_touch_count"] >= background_touch_limit or
            is_food_layout_contaminated
        )
        if is_food_complete_candidate:
            is_background_like = False
        if strategy["type"] == "food_product" and is_food_support_shape(shape_features):
            is_background_like = False
        shape_merge_allowed = (
            (strategy["type"] != "food_product" and is_attachment and is_small_part) or
            is_table_shape_part or
            is_furniture_block_peer or
            is_decor_compound_part or
            is_decor_atomic_fragment or
            is_hard_product_part or
            is_food_product_part
        )

        if not shape_merge_allowed or is_background_like:
            if is_food_layout_contaminated:
                candidate["debug"]["rejectReason"] = "product_layout_contaminated"
            elif is_background_like:
                candidate["debug"]["rejectReason"] = "background_like"
            elif not is_attachment and not (is_table_shape_part or is_furniture_block_peer):
                candidate["debug"]["rejectReason"] = "not_attachment"
            else:
                candidate["debug"]["rejectReason"] = "not_shape_peer"
            continue

        trial = np.maximum(merged, candidate["mask"])
        trial_binary = trial > 0.5
        trial_target_area = int(np.count_nonzero(trial_binary[ty1:ty2, tx1:tx2]))
        trial_fill_ratio = trial_target_area / target_area
        if trial_fill_ratio > strategy["max_merged_fill"]:
            candidate["debug"]["rejectReason"] = "merged_fill_limit"
            continue

        merged = trial
        selected.append(candidate)
        candidate["debug"]["selected"] = True
        candidate["debug"]["rejectReason"] = "table_shape_part" if is_table_shape_part else (
            "furniture_block_peer" if is_furniture_block_peer else (
                "decor_compound_part" if is_decor_compound_part else (
                    "decor_atomic_fragment" if is_decor_atomic_fragment else (
                        "hard_product_part" if is_hard_product_part else (
                            "food_product_part" if is_food_product_part else "attachment"
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
    if target_fill_ratio > background_risk_limit and len(selected) > 1:
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
    quality_gate = build_quality_gate(score, primary["score"], target_fill_ratio, selected, strategy["type"])
    strategy_profile = str((layer_meta or {}).get("extractionProfile", "")).lower() or strategy["type"]
    quality_reason = ",".join(quality_gate["issues"]) if quality_gate["issues"] else (
        f"semantic_{strategy['type']}_primary_mask_with_attachments"
    )
    quality = {
        "status": quality_gate["status"] if selected else "failed",
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
        "debugCandidates": sorted(debug_candidates, key=lambda item: item["score"], reverse=True)[:12],
        "reason": quality_reason,
        "runtimeAction": quality_gate["runtimeAction"],
        "shouldGenerateRuntimeLayer": quality_gate["shouldGenerateRuntimeLayer"] if selected else False,
        "needsHigherPrecision": quality_gate["needsHigherPrecision"] or not selected,
        "issues": quality_gate["issues"] if selected else ["no_selected_mask"],
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

def normalize_result_masks(results, img_w, img_h):
    if len(results) == 0 or results[0].masks is None:
        return np.empty((0, img_h, img_w), dtype=np.float32)

    normalized_masks = []
    for mask in results[0].masks.data.cpu().numpy():
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        normalized_masks.append(mask.astype(np.float32))

    if not normalized_masks:
        return np.empty((0, img_h, img_w), dtype=np.float32)
    return np.stack(normalized_masks, axis=0)


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

def process_prompted_cutouts(img, pixel_bboxes, layer_ids, layer_metas, context_layers, mask_provider, engine_name, refine_masks=False):
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
        strategy_name = quality.get("strategyProfile") or quality.get("strategy") if quality else "unknown"
        print(f"Layer {layer_ids[i]} engine={engine_name} strategy={strategy_name} merged {selected_count} candidate masks")
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

        cleanup_candidates = None
        if quality and quality.get("postProcess"):
            cleanup_candidates = None

        cleaned_mask = cleanup_mask(mask, target_bbox)
        if np.any(cleaned_mask > 0.5):
            mask = cleaned_mask

        if quality and quality.get("strategy") == "food_product":
            conflict_refined_mask, conflict_changed, conflict_debug = refine_food_mask_with_conflict_sam(
                img,
                mask,
                target_bbox,
                layer_meta or {},
                context_layers or []
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
                    layout_mask, layout_quality = segment_attached_layout_mask(img, entry, context_layers or [])
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
        allow_local_refine = not (quality and quality.get("strategy") == "food_product")
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
                mask = cleanup_mask(refined_mask, target_bbox)

        alpha_mask = generate_alpha_matte(
            img,
            mask,
            target_bbox,
            cleanup_mask=matte_cleanup_mask,
            strategy_type=quality.get("strategy") if quality else None,
            label_cleanup_mask=label_cleanup_mask,
            flat_cleanup_mask=flat_cleanup_mask
        )
        if quality is None:
            quality = {}
        quality["postProcess"] = {
            "maskCleanup": True,
            "localRefine": bool(local_refined),
            "matting": "opencv_grabcut"
        }

        cutout = build_cutout_entry(
            img,
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
            for bbox in bboxes_norm:
                ymin_n, xmin_n, ymax_n, xmax_n = bbox
                y1 = int((ymin_n / 1000.0) * h)
                x1 = int((xmin_n / 1000.0) * w)
                y2 = int((ymax_n / 1000.0) * h)
                x2 = int((xmax_n / 1000.0) * w)
                pixel_bboxes.append(expand_bbox(x1, y1, x2, y2, w, h))

            print(f"Expanded bbox prompts by {int(BBOX_EXPAND_RATIO * 100)}%")

            if requested_engine == "sam":
                def sam_mask_provider(target_bbox, layer_meta, index):
                    strategy_type = get_layer_strategy(layer_meta or {}).get("type")
                    if strategy_type == "food_product":
                        print(
                            f"SAM prompts for {layer_meta.get('name') or layer_ids[index] or index}: "
                            f"bbox-only strategy=food_product"
                        )
                        results = run_sam_bbox_inference(
                            img,
                            target_bbox,
                            multimask_output=True,
                            imgsz=1024
                        )
                        return normalize_result_masks(results, w, h)

                    prompt_inputs = build_sam_prompt_inputs(
                        layer_meta or {},
                        context_layers or [],
                        target_bbox,
                        w,
                        h
                    )
                    point_count = len(prompt_inputs["points"][0]) if prompt_inputs["points"] else 0
                    negative_count = sum(1 for value in (prompt_inputs["labels"][0] if prompt_inputs["labels"] else []) if value == 0)
                    positive_count = point_count - negative_count
                    if point_count > 0:
                        print(
                            f"SAM prompts for {layer_meta.get('name') or layer_ids[index] or index}: "
                            f"+{positive_count} -{negative_count} strategy={prompt_inputs['strategyType']}"
                        )
                    results = run_sam_bbox_inference(
                        img,
                        target_bbox,
                        multimask_output=True,
                        points=prompt_inputs["points"] if prompt_inputs else None,
                        labels=prompt_inputs["labels"] if prompt_inputs else None
                    )
                    return normalize_result_masks(results, w, h)

                cutouts = process_prompted_cutouts(
                    img,
                    pixel_bboxes,
                    layer_ids,
                    layer_metas,
                    context_layers,
                    sam_mask_provider,
                    "sam_bbox_prompt",
                    refine_masks=True
                )
                return JSONResponse(content={"success": True, "engine": "sam", "cutouts": cutouts})

            # Default FastSAM path: global candidate masks + existing merge logic.
            results = model(img, retina_masks=True, imgsz=1024, conf=0.25, iou=0.9)
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
                refine_masks=False
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
            results = model(img, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            
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
