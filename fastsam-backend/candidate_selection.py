"""Candidate scoring and semantic mask arbitration helpers.

The functions here are deterministic policy code. Model inference is supplied
through sam_runtime, while request handling remains in api.py.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image

from sam_runtime import *
from segmentation_policy import *
from mask_ops import *
from segmentation_primitives import (
    build_food_positive_points_from_mask,
    build_positive_points_from_mask,
    normalize_result_masks,
    sample_points_in_bbox,
    shrink_bbox,
)

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


def build_completion_candidate_preview(mask, target_bbox):
    """Return a compact alpha-only preview for completion diagnostics.

    Candidate previews are intentionally limited to the requested target box
    and contain no source pixels.  This makes the arbitration decision
    inspectable without leaking the whole completed scene or changing any
    extraction result.
    """
    binary = np.asarray(mask > 0.5, dtype=np.uint8)
    if binary.ndim != 2:
        return None
    height, width = binary.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = (binary[y1:y2, x1:x2] * 255).astype(np.uint8)
    rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
    # A saturated fill stays visible against the chat panel's white
    # background while the alpha channel still shows the exact candidate
    # silhouette.
    rgba[:, :, 0] = 255
    rgba[:, :, 1] = 64
    rgba[:, :, 2] = 32
    rgba[:, :, 3] = crop
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        return None
    return 'data:image/png;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')

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

def build_quality_gate(
    score,
    primary_score,
    target_fill_ratio,
    selected,
    strategy_type,
    high_coverage_entity=False,
    policy=None
):
    policy = policy or policy_for_strategy_type(strategy_type)
    effective_strategy_type = policy.get("selectionType", strategy_type)
    min_score = MIN_RUNTIME_ACCEPT_SCORE
    min_primary_score = MIN_RUNTIME_ACCEPT_PRIMARY_SCORE
    min_fill_ratio = MIN_RUNTIME_ACCEPT_FILL_RATIO
    max_fill_ratio = MAX_RUNTIME_ACCEPT_FILL_RATIO

    if policy.get("selector") == "compound_food" or effective_strategy_type == "food_product":
        min_score = 0.28
        min_primary_score = 0.22
        min_fill_ratio = 0.05
        max_fill_ratio = 0.82
    elif effective_strategy_type == "wall_art":
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
            "matting_or_hq_sam" if policy.get("spatial") else "hq_sam"
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
            # A completion foreground context is an independently extracted
            # object left intact in the scene (unlike the foreground
            # occluder, which the completed target is expected to occupy
            # behind).  Keep that distinction through candidate scoring.
            "completion_foreground_context": bool(other.get("completionForegroundContext")),
            "strong": bool(
                other.get("completionOccluder") or
                (
                    bool(other.get("completionForegroundContext")) and
                    bool(get_layer_strategy(layer_meta or {}).get("spatialCanonicalCompletion"))
                ) or
                (same_parent and should_strong_exclude_sibling(layer_meta, other))
            )
        })

    return excludes

def get_exclude_bbox(entry):
    return entry.get("bbox") if isinstance(entry, dict) else entry

def is_strong_exclude(entry):
    return bool(entry.get("strong")) if isinstance(entry, dict) else False

def is_completion_foreground_context(entry):
    return bool(entry.get("completion_foreground_context")) if isinstance(entry, dict) else False

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


def decode_completion_observation_mask(mask_data, img_w, img_h):
    """Decode the first-pass visible target mask for completion ranking."""
    if not isinstance(mask_data, str) or not mask_data:
        return None
    try:
        encoded = mask_data.split(",", 1)[1] if "," in mask_data else mask_data
        raw = base64.b64decode(encoded.replace(" ", "").replace("\n", "").replace("\r", ""))
        source = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if source is None:
            return None
        if source.ndim == 3 and source.shape[2] == 4:
            mask = source[:, :, 3]
        elif source.ndim == 3:
            mask = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            mask = source
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return mask > 48
    except Exception as error:
        print(f"Completion observation mask decode failed: {error}")
        return None


def decode_completion_occlusion_mask(mask_data, img_w, img_h):
    """Decode the foreground region that the completion must recover behind."""
    return decode_completion_observation_mask(mask_data, img_w, img_h)


def decode_completion_foreground_context_mask(mask_data, img_w, img_h):
    """Decode exact alpha for independent foreground objects left in the scene."""
    return decode_completion_observation_mask(mask_data, img_w, img_h)


def completion_observation_is_consistent(
    observation_area,
    observation_recall,
    observation_iou,
    containment,
    base_strategy_type=None,
    spatial=False
):
    """Reject post-inpaint scene leakage without blocking normal completions.

    A first-pass observation is only a visible fragment, so generic subjects
    may legitimately have a modest IoU after inpainting. Tables and furniture
    are different: their hard silhouette is a reliable anchor, and a very low
    IoU with high recall is the characteristic signature of a background
    envelope added around that anchor.
    """
    if observation_area <= 0 or observation_recall is None:
        return True, "no_observation_anchor"
    if observation_recall < 0.72:
        return False, "low_observation_recall"

    hard_spatial_target = spatial and base_strategy_type in {
        "table", "furniture", "lighting"
    }
    if not hard_spatial_target:
        return True, "generic_observation_anchor"
    if containment < 0.88:
        return False, "spatial_completion_outside_target"
    if observation_iou is None or observation_iou < 0.32:
        return False, "spatial_completion_observation_iou"
    return True, "spatial_observation_anchor"



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
    # During canonical completion, overlap with a known foreground bbox is
    # expected: it is the region where the hidden target must be recovered.
    # Exact foreground ownership is resolved by the frontend's real alpha mask
    # after this candidate is returned, so a coarse bbox must not downgrade the
    # complete target candidate here.
    if not metrics.get("completion_recovery"):
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


def recover_completion_hidden_delta(
    merged_binary,
    fallback_candidates,
    primary_index,
    completion_occlusion_mask,
    completion_observation,
    completion_foreground_context_mask,
    target_bbox,
    target_area,
):
    """Recover only evidence-bounded hidden pixels from alternate SAM masks.

    Multimask alternatives are evidence, not independent assets. This helper
    returns a delta rather than an alternate full silhouette: pixels must be
    inside the verified occlusion region, outside known foreground context,
    and supported by the observed/accepted target silhouette.
    """
    merged = np.asarray(merged_binary > 0, dtype=bool).copy()
    if completion_occlusion_mask is None or not fallback_candidates:
        return merged, 0, 0
    occlusion_binary = np.asarray(completion_occlusion_mask > 0.5, dtype=bool)
    observation_binary = (
        np.asarray(completion_observation > 0.5, dtype=bool)
        if completion_observation is not None else
        np.zeros_like(occlusion_binary, dtype=bool)
    )
    context_binary = (
        np.asarray(completion_foreground_context_mask > 0.5, dtype=bool)
        if completion_foreground_context_mask is not None else
        np.zeros_like(occlusion_binary, dtype=bool)
    )
    if merged.shape != occlusion_binary.shape:
        return merged, 0, 0
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    height, width = merged.shape
    tx1 = max(0, min(width - 1, tx1))
    ty1 = max(0, min(height - 1, ty1))
    tx2 = max(tx1 + 1, min(width, tx2))
    ty2 = max(ty1 + 1, min(height, ty2))
    anchor_binary = merged | observation_binary
    rescue_radius = max(12, min(36, int(round(min(tx2 - tx1, ty2 - ty1) * 0.10))))
    anchor_support = cv2.dilate(
        anchor_binary.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (rescue_radius * 2 + 1, rescue_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    allowed_hidden = occlusion_binary.copy()
    allowed_hidden[:ty1] = False
    allowed_hidden[ty2:] = False
    allowed_hidden[:, :tx1] = False
    allowed_hidden[:, tx2:] = False
    rescue_limit = max(256, int(max(1, target_area) * 0.32))
    accepted_pixels = 0
    considered_candidates = 0
    raw_delta_pixels = 0
    context_blocked_pixels = 0
    zone_blocked_pixels = 0
    support_blocked_pixels = 0
    primary_index = int(primary_index) if primary_index is not None else None
    # Only candidates that independently passed the publish foreground gates
    # may contribute hidden structure.  `fallback_candidates` intentionally
    # contains every raw multimask result for diagnostics, including masks
    # rejected as background-like or contaminated.  Treating that diagnostic
    # pool as rescue evidence lets a rejected table/background envelope put
    # its pixels back into the canonical mask (the selected primary may be
    # clean while the final merged mask is not).
    rescue_candidates = [
        item for item in fallback_candidates
        if item.get("metrics", {}).get("index") != primary_index and
        item.get("debug", {}).get("candidate") is True and
        (
            item.get("debug", {}).get("spatialCompletionStructuralPeer") is True or
            item.get("debug", {}).get("completionRecoveryCandidate") is True
        )
    ]
    skipped_rejected_candidates = max(
        0,
        len(fallback_candidates) - 1 - len(rescue_candidates)
    )
    rescue_candidates.sort(
        key=lambda item: item.get("debug", {}).get("completionRecoveryPixels", 0),
        reverse=True
    )
    for candidate in rescue_candidates:
        considered_candidates += 1
        candidate_mask = candidate.get("mask")
        if candidate_mask is None:
            continue
        candidate_binary = np.asarray(candidate_mask > 0.5, dtype=bool)
        if candidate_binary.shape != merged.shape:
            continue
        raw_delta = candidate_binary & (~merged)
        raw_delta_pixels += int(np.count_nonzero(raw_delta))
        context_blocked_pixels += int(np.count_nonzero(raw_delta & context_binary))
        zone_blocked_pixels += int(np.count_nonzero(raw_delta & (~allowed_hidden)))
        support_blocked_pixels += int(np.count_nonzero(raw_delta & allowed_hidden & (~context_binary) & (~anchor_support)))
        # The verified occlusion mask is the ownership boundary.  A hidden
        # leg/support can be disconnected from the visible anchor, so using
        # anchor dilation as a hard gate drops exactly the structure that
        # completion is meant to recover.  Keep the dilation as an audit
        # signal, but accept only the candidate's bounded occlusion delta.
        delta = raw_delta & allowed_hidden & (~context_binary)
        if not np.any(delta):
            continue
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            delta.astype(np.uint8), connectivity=8
        )
        accepted_delta = np.zeros_like(delta, dtype=bool)
        for component_index in range(1, component_count):
            component_area = int(component_stats[component_index, cv2.CC_STAT_AREA])
            if component_area < max(24, int(max(1, target_area) * 0.00035)):
                continue
            accepted_delta |= component_labels == component_index
        delta_pixels = int(np.count_nonzero(accepted_delta))
        if delta_pixels <= 0 or accepted_pixels + delta_pixels > rescue_limit:
            continue
        merged_candidate_area = int(np.count_nonzero(merged | accepted_delta))
        growth = merged_candidate_area / max(1, int(np.count_nonzero(merged)))
        if growth > 2.35:
            continue
        merged |= accepted_delta
        accepted_pixels += delta_pixels
        candidate.setdefault("debug", {})["completionHiddenRescuePixels"] = delta_pixels
        candidate["debug"]["completionHiddenRescue"] = True
    print(
        "Completion hidden candidate rescue audit: "
        f"candidates={considered_candidates} skippedRejected={skipped_rejected_candidates} "
        f"rawDelta={raw_delta_pixels} "
        f"accepted={accepted_pixels} contextBlocked={context_blocked_pixels} "
        f"zoneBlocked={zone_blocked_pixels} supportBlocked={support_blocked_pixels} "
        f"limit={rescue_limit} radius={rescue_radius}"
    )
    return merged, accepted_pixels, rescue_radius


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

def select_and_merge_masks(
    candidate_masks,
    target_bbox,
    img_w,
    img_h,
    layer_meta=None,
    context_layers=None,
    quality_profile="publish"
):
    policy = resolve_mask_policy(layer_meta or {}, quality_profile)
    resolved_strategy = get_layer_strategy(layer_meta or {})
    base_strategy_type = resolved_strategy.get("baseStrategyType", resolved_strategy.get("type"))
    strategy = dict(resolved_strategy)
    if policy["completion"] and not policy["spatialCanonicalCompletion"]:
        # Completion is a phase shared by both profiles. Use one generic
        # candidate selector for flat entities. Spatial structural entities
        # keep their canonical table/furniture strategy instead.
        strategy["type"] = "completion_object"
    if policy["selector"] == "compound_food":
        food_result = select_food_masks_from_candidates(
            candidate_masks,
            target_bbox,
            img_w,
            img_h,
            layer_meta=layer_meta,
            context_layers=context_layers
        )
        if food_result[2] is not None:
            food_result[2].update({
                "baseStrategyType": base_strategy_type,
                "profile": resolved_strategy.get("profile", "flat_design"),
                "features": resolved_strategy.get("features", []),
                "phase": resolved_strategy.get("phase", "initial")
            })
            food_result[2].update({
                "policyVersion": policy["version"],
                "selectionMode": policy["selector"]
            })
        return food_result

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
    # The second-pass layer marker is authoritative. The broad `completion`
    # profile is also used by the initial observation pass, so it must not
    # relax ordinary first-pass extraction for every layer in that batch.
    completion_observation_fallback = bool(
        (layer_meta or {}).get("_completionObservationFallback")
    )
    completion_full_scene_mask = bool(
        (layer_meta or {}).get("_completionFullSceneMask")
    )
    completion_recovery = (
        is_completion_segmentation_layer(layer_meta) and
        not completion_observation_fallback and
        not policy["spatialCanonicalCompletion"]
    )
    # B can pass the completion gates without entering B/L arbitration. It is
    # still a valid full-scene result for flat hard-edge entities; otherwise
    # that route would silently fall back to the old first-observation clip.
    completion_full_scene_candidate = bool(
        completion_recovery and
        policy["profile"] == "flat_design" and
        not policy["spatial"] and
        policy["softEdge"] is False and
        "hard_edge" in policy["features"]
    )
    completion_observation = decode_completion_observation_mask(
        (layer_meta or {}).get("completionObservationMask"),
        img_w,
        img_h
    ) if (completion_recovery or policy["spatialCanonicalCompletion"]) else None
    completion_occlusion_mask = decode_completion_occlusion_mask(
        (layer_meta or {}).get("completionOcclusionMask"),
        img_w,
        img_h
    ) if (completion_recovery or policy["spatialCanonicalCompletion"]) else None
    completion_foreground_context_mask = decode_completion_foreground_context_mask(
        (layer_meta or {}).get("completionForegroundContextMask"),
        img_w,
        img_h
    ) if policy["spatialCanonicalCompletion"] else None
    completion_observation_area = int(np.count_nonzero(completion_observation)) if completion_observation is not None else 0
    # The completion prompt intentionally includes the verified occluder area
    # so SAM can recover the hidden target.  It is therefore not a valid
    # identity box for the target itself.  Use the original visible
    # observation as the anchor box and keep the prompt box only for recovery
    # containment/occlusion checks.
    completion_anchor_bbox = mask_bbox(completion_observation) if completion_observation is not None else None
    completion_anchor_bbox_area = bbox_area(completion_anchor_bbox) if completion_anchor_bbox else 0
    completion_occlusion_target_area = (
        int(np.count_nonzero(completion_occlusion_mask[ty1:ty2, tx1:tx2]))
        if completion_occlusion_mask is not None else 0
    )

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
        if completion_foreground_context_mask is not None:
            completion_foreground_mask_area = int(np.count_nonzero(
                mask_binary & completion_foreground_context_mask
            ))
        else:
            # A context bbox is only a prompt/exclusion hint, not pixel-level
            # ownership evidence.  Do not turn it into a hard candidate
            # rejection when the exact foreground alpha mask is unavailable:
            # the browser reconciliation stage will retain observed pixels
            # and hold the completion because ownership evidence is missing.
            # Treating this coarse rectangle as a real mask caused every
            # multimask candidate to be rejected and the API to return 422
            # before the safe hold path could run.
            completion_foreground_mask_area = 0
        completion_foreground_mask_ratio = completion_foreground_mask_area / mask_area
        target_fill_ratio = target_mask_area / target_area
        mask_inside_target_ratio = target_mask_area / mask_area
        bbox_overlap_ratio = intersection_area(current_bbox, target_bbox) / target_area
        completion_anchor_bbox_overlap_ratio = (
            intersection_area(current_bbox, completion_anchor_bbox) /
            max(1, completion_anchor_bbox_area)
            if completion_anchor_bbox else None
        )
        completion_identity_overlap_ratio = (
            completion_anchor_bbox_overlap_ratio
            if completion_anchor_bbox_overlap_ratio is not None
            else bbox_overlap_ratio
        )
        mask_area_ratio = mask_area / target_area
        center_inside = box_center_inside(current_bbox, target_bbox)
        bbox_touch_count = int(current_bbox[0] <= tx1 + 2) + int(current_bbox[1] <= ty1 + 2) + int(current_bbox[2] >= tx2 - 2) + int(current_bbox[3] >= ty2 - 2)
        completion_observation_recall = None
        completion_observation_iou = None
        if completion_observation is not None and completion_observation_area > 0:
            candidate_observation_overlap = int(np.count_nonzero(mask_binary & completion_observation))
            candidate_observation_union = int(np.count_nonzero(mask_binary | completion_observation))
            completion_observation_recall = candidate_observation_overlap / completion_observation_area
            completion_observation_iou = candidate_observation_overlap / max(1, candidate_observation_union)
        completion_observation_consistent, completion_observation_reason = (
            completion_observation_is_consistent(
                completion_observation_area,
                completion_observation_recall,
                completion_observation_iou,
                mask_inside_target_ratio,
                base_strategy_type=base_strategy_type,
                spatial=policy["spatial"]
        ) if completion_recovery else (True, "spatial_completion_anchor" if policy["spatialCanonicalCompletion"] else "not_completion")
        )
        completion_recovery_pixels = 0
        completion_recovery_ratio = None
        if completion_occlusion_mask is not None:
            completion_region = mask_binary[ty1:ty2, tx1:tx2]
            if completion_observation is not None:
                completion_region = completion_region & (~completion_observation[ty1:ty2, tx1:tx2])
            completion_recovery_pixels = int(np.count_nonzero(
                completion_region & completion_occlusion_mask[ty1:ty2, tx1:tx2]
            ))
            if completion_occlusion_target_area > 0:
                completion_recovery_ratio = completion_recovery_pixels / completion_occlusion_target_area
        spatial_recovery_bonus = (
            min(0.18, max(0.0, float(completion_recovery_ratio or 0.0)) * 0.18)
            if policy["spatialCanonicalCompletion"] else 0.0
        )
        completion_zone_ok = (
            not completion_recovery or
            completion_occlusion_target_area <= 0 or
            completion_recovery_pixels >= max(256, int(completion_occlusion_target_area * 0.03))
        )
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
        completion_anchor_candidate = bool(
        completion_recovery and
            completion_observation_recall is not None and
            completion_observation_consistent and
            completion_observation_recall >= 0.72 and
            mask_inside_target_ratio >= 0.72 and
            0.10 <= target_fill_ratio <= 0.78 and
            mask_area_ratio <= 0.82 and
            completion_identity_overlap_ratio >= 0.70
        )
        completion_recovery_candidate = bool(
            completion_recovery and
            mask_inside_target_ratio >= 0.72 and
            0.10 <= target_fill_ratio <= 0.78 and
            mask_area_ratio <= 0.82 and
            completion_identity_overlap_ratio >= 0.70
        )
        # A completed target may grow only from the hidden/edited region. Its
        # original visible silhouette is the non-negotiable identity anchor;
        # without this check SAM can select a background-shaped candidate that
        # happens to fit the completion bbox.
        completion_observation_candidate = bool(
            completion_recovery_candidate and
            completion_observation_area > 0 and
            completion_observation_recall is not None and
            completion_observation_consistent and
            (
                completion_observation_recall >= 0.72 or
                (
                    completion_observation_recall >= 0.60 and
                    mask_inside_target_ratio >= 0.90 and
                    0.50 <= target_fill_ratio <= 0.78 and
                    mask_area_ratio <= 0.78
                )
            )
            and completion_zone_ok
        )
        # Spatial tables keep the canonical table selector, but their visible
        # first-pass cutout is still a valid identity anchor.  A low-fill SAM-L
        # candidate with that anchor is preferable to returning no cutout when
        # a foreground context bbox overlaps the completed scene.  This gate
        # excludes broad/background candidates by containment, overlap, and
        # edge-touch limits, and is used only for post-inpaint tables.
        spatial_completion_anchor_candidate = bool(
            policy["spatialCanonicalCompletion"] and
            mask_inside_target_ratio >= 0.80 and
            0.06 <= target_fill_ratio <= 0.78 and
            mask_area_ratio <= 0.92 and
            # The prompt bbox may include a chair/stool that must be removed
            # from the generated scene.  Anchor identity to the observed
            # target silhouette instead of requiring the target to fill that
            # union box.
            (
                completion_anchor_bbox_overlap_ratio is None or
                completion_anchor_bbox_overlap_ratio >= 0.70
            ) and
            bbox_touch_count <= 2 and
            (
                completion_observation_recall is None or
                completion_observation_recall >= 0.52
            )
        )
        # SAM-L multimask candidates from a completed table are often
        # alternative structural views of the same anchored object: one mask
        # keeps the tabletop/left support while another keeps a lower or
        # detached support.  They are not ordinary "attachments", so the
        # legacy shape-peer gate can discard a valid leg (the observed
        # `not_shape_peer` failure).  Permit a bounded union only when the
        # candidate independently passes the spatial anchor/containment gates
        # and contributes pixels in the verified completion zone.
        spatial_completion_structural_peer = bool(
            spatial_completion_anchor_candidate and
            completion_recovery_pixels >= max(256, int(completion_occlusion_target_area * 0.03)) and
            mask_inside_target_ratio >= 0.80 and
            bbox_overlap_ratio >= 0.70 and
            mask_area_ratio <= 0.92 and
            completion_foreground_mask_ratio < 0.025
        )

        metrics = {
            "index": index,
            "target_fill_ratio": target_fill_ratio,
            "mask_inside_target_ratio": mask_inside_target_ratio,
            "bbox_overlap_ratio": bbox_overlap_ratio,
            "completion_anchor_bbox_overlap_ratio": completion_anchor_bbox_overlap_ratio,
            "completion_identity_overlap_ratio": completion_identity_overlap_ratio,
            "mask_area_ratio": mask_area_ratio,
            "center_inside": center_inside,
            "bbox_touch_count": bbox_touch_count,
            "bbox": current_bbox,
            "target_bbox": target_bbox,
            "mask_area": mask_area,
            "exclude_mask_ratio": exclude_mask_ratio,
            "strong_exclude_mask_ratio": strong_exclude_mask_ratio,
            "completion_foreground_mask_ratio": completion_foreground_mask_ratio,
            "shape_features": shape_features,
            "multi_entity_coverage": multi_entity_coverage,
            "multi_entity_count": multi_entity_count
        }
        metrics["completion_recovery"] = completion_recovery
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
            and mask_inside_target_ratio >= (
                0.80 if spatial_completion_anchor_candidate else MIN_MASK_INSIDE_TARGET_RATIO
            )
            and MIN_MASK_AREA_RATIO_IN_BBOX <= mask_area_ratio <= MAX_MASK_AREA_RATIO_IN_BBOX
            and (bbox_overlap_ratio >= MIN_BBOX_OVERLAP_RATIO or center_inside)
            and shape_allowed
            and not lighting_background_like
            and (
                not table_background_like or
                completion_observation_candidate or
                spatial_completion_anchor_candidate
            )
            # A bounded high-coverage entity may touch the semantic bbox. The
            # previous gate rejected it as background even when the geometry
            # audit had already marked it as a valid entity.
            and (
                not furniture_background_like or
                multi_entity_coverage or
                (
                    high_coverage_entity and
                    strong_exclude_mask_ratio < 0.22 and
                    mask_inside_target_ratio >= 0.84
                )
            )
            and (not completion_recovery or completion_observation_candidate)
            # For every spatial completion candidate, independently extracted
            # objects still present in the scene are negative ownership
            # evidence.  Anchor overlap proves identity, not purity: an
            # anchored candidate that also contains a vase/stool/ornament
            # must not be promoted to the canonical mask.  The hidden-only
            # reconciler can still recover a clean delta from other candidates
            # or hold the task for review when no clean evidence exists.
            and not (
                policy["spatialCanonicalCompletion"] and
                completion_foreground_mask_ratio >= 0.025
            )
        )
        score = score_candidate(metrics, strategy)
        # When the prompt contains a verified occluder, prefer an otherwise
        # valid anchored candidate that actually covers some generated target
        # area. This prevents a visible-only mask from winning solely because
        # it has a slightly cleaner score than the candidate that recovers a
        # hidden leg/support.
        if spatial_completion_anchor_candidate and spatial_recovery_bonus > 0:
            score += spatial_recovery_bonus
        debug_candidate = {
            "index": index,
            "score": round(float(score), 3),
            "fill": round(float(target_fill_ratio), 3),
            "inside": round(float(mask_inside_target_ratio), 3),
            "area": round(float(mask_area_ratio), 3),
            "bboxOverlap": round(float(bbox_overlap_ratio), 3),
            "anchorBboxOverlap": (
                round(float(completion_anchor_bbox_overlap_ratio), 3)
                if completion_anchor_bbox_overlap_ratio is not None else None
            ),
            "identityOverlap": round(float(completion_identity_overlap_ratio), 3),
            "exclude": round(float(exclude_mask_ratio), 3),
            "strongExclude": round(float(strong_exclude_mask_ratio), 3),
            "foregroundContext": round(float(completion_foreground_mask_ratio), 3),
            "foregroundContextMaskAvailable": bool(completion_foreground_context_mask is not None),
            "touch": int(bbox_touch_count),
            "center": bool(center_inside),
            "shapeFeatures": round_shape_features(shape_features),
            "decorBase": bool(decor_base),
            "highCoverageEntity": bool(high_coverage_entity),
            "multiEntityCoverage": bool(multi_entity_coverage),
            "observationRecall": round(completion_observation_recall, 3) if completion_observation_recall is not None else None,
            "observationIoU": round(completion_observation_iou, 3) if completion_observation_iou is not None else None,
            "completionRecoveryPixels": completion_recovery_pixels,
            "completionRecoveryRatio": round(completion_recovery_ratio, 3) if completion_recovery_ratio is not None else None,
            "spatialRecoveryBonus": round(float(spatial_recovery_bonus), 3),
            "completionObservationReason": completion_observation_reason,
            "completionAnchor": completion_anchor_candidate,
            "spatialCompletionAnchor": spatial_completion_anchor_candidate,
            "completionRecoveryCandidate": completion_recovery_candidate,
            "spatialCompletionStructuralPeer": spatial_completion_structural_peer,
            "completionObservationCandidate": completion_observation_candidate,
            "candidate": bool(is_probable_foreground),
            "selected": False,
            "rejectReason": "" if is_probable_foreground else (
                "lighting_background_like" if lighting_background_like else (
                    "table_background_like" if table_background_like else (
                        "furniture_background_like" if furniture_background_like else (
                            completion_observation_reason if completion_recovery and not completion_observation_consistent else (
                            "completion_occlusion_coverage_low" if completion_recovery and not completion_zone_ok else (
                            "completion_foreground_context_overlap" if (
                                policy["spatialCanonicalCompletion"] and
                                completion_foreground_mask_ratio >= 0.025
                            ) else (
                            "missing_completion_observation_anchor" if completion_recovery and not completion_observation_candidate else (
                                "sibling_overlap" if strong_exclude_mask_ratio >= 0.22 and not completion_recovery_candidate else (
                                high_coverage_reason if high_coverage_entity else (shape_reject_reason or "gate")
                                )
                            )
                            )
                            )
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

        anchor_debug = (
            f"anchorOverlap={completion_anchor_bbox_overlap_ratio:.3f}, "
            if completion_anchor_bbox_overlap_ratio is not None else ""
        )
        debug_rows.append(
            f"fill={target_fill_ratio:.3f}, inside={mask_inside_target_ratio:.3f}, "
            f"area={mask_area_ratio:.3f}, bboxOverlap={bbox_overlap_ratio:.3f}, "
            f"{anchor_debug}"
            f"exclude={exclude_mask_ratio:.3f}, strongExclude={strong_exclude_mask_ratio:.3f}, "
            f"aspect={shape_features['aspectRatio']:.3f}, thinV={shape_features['isThinVertical']}, "
            f"support={shape_features['isTableSupport']}, block={shape_features['isBlockLike']}, "
            f"bottom={shape_features['bottomBand']:.3f}, "
            f"touch={bbox_touch_count}, center={center_inside}, score={score:.3f}, "
            f"candidate={is_probable_foreground} "
            f"completionAudit={completion_observation_reason}"
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
    exclude_reliable = bool(exclude_bboxes) and not completion_recovery and not (
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
            or spatial_completion_structural_peer
        )

        if not shape_merge_allowed or is_background_like:
            if is_food_layout_contaminated:
                candidate["debug"]["rejectReason"] = "product_layout_contaminated"
            elif is_background_like:
                candidate["debug"]["rejectReason"] = "background_like"
            elif not is_attachment and not (is_table_leg_part or is_table_surface_part or is_furniture_block_peer or spatial_completion_structural_peer):
                candidate["debug"]["rejectReason"] = "not_attachment"
            else:
                candidate["debug"]["rejectReason"] = "spatial_completion_structural_peer" if spatial_completion_structural_peer else "not_shape_peer"
            continue

        # A spatial completion peer is evidence for hidden structure, not a
        # second full silhouette.  The generic union below would copy its
        # visible/background pixels wholesale and was the source of both
        # missing-leg and scene-contamination regressions.  Leave it for the
        # bounded hidden-delta pass after arbitration; that pass clips to the
        # verified occlusion mask, foreground ownership, and anchor support.
        if spatial_completion_structural_peer:
            candidate["debug"]["rejectReason"] = "structural_peer_hidden_rescue_only"
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
                "spatial_completion_structural_peer" if spatial_completion_structural_peer else (
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

    # Spatial completion candidates are often multimask variants of the same
    # table.  The primary variant can preserve the observed top and one
    # support while another variant contains a second hidden support.  Do not
    # merge the alternate scene envelope wholesale; recover only its delta in
    # the verified occlusion zone and outside unrelated foreground context.
    # A hidden support may be disconnected from the visible silhouette, so the
    # occlusion boundary—not anchor proximity—is authoritative. This is the hidden-only rule
    # from the completion contract and is independent of whether occluders are
    # represented as one composite layer or several layers.
    completion_hidden_rescue_pixels = 0
    if policy["spatialCanonicalCompletion"] and completion_occlusion_mask is not None:
        # Use every raw candidate as evidence, including candidates rejected by
        # the generic publish selector. Their complete masks are never copied;
        # only a bounded hidden delta can be rescued below.
        merged_binary, completion_hidden_rescue_pixels, rescue_radius = recover_completion_hidden_delta(
            merged_binary,
            fallback_candidates,
            primary["metrics"].get("index"),
            completion_occlusion_mask,
            completion_observation,
            completion_foreground_context_mask,
            target_bbox,
            target_area,
        )
        if completion_hidden_rescue_pixels > 0:
            merged = merged_binary.astype(np.float32)
            merged_target_area = int(np.count_nonzero(merged_binary[ty1:ty2, tx1:tx2]))
            target_fill_ratio = merged_target_area / target_area
            primary["debug"]["completionHiddenRescuePixels"] = completion_hidden_rescue_pixels
            primary["debug"]["fill"] = round(float(target_fill_ratio), 3)
            primary["debug"]["completionRecoveryPixels"] = int(
                primary["debug"].get("completionRecoveryPixels", 0) or 0
            ) + completion_hidden_rescue_pixels
            print(
                f"Completion hidden candidate rescue: pixels={completion_hidden_rescue_pixels} "
                f"fill={target_fill_ratio:.3f} radius={rescue_radius}"
            )

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

    if (
        completion_recovery and
        not policy["spatialCanonicalCompletion"] and
        not (completion_full_scene_mask or completion_full_scene_candidate)
    ):
        # Flat completion context may contain coarse sibling bboxes, but entries
        # explicitly marked as real foreground occluders are safe to remove
        # from the canonical target mask. This prevents detached lemon,
        # label, or other occluder pixels from becoming part of the target
        # without trusting every semantic bbox in the scene. Spatial canonical
        # completion is excluded from this cleanup because its hidden target
        # pixels intentionally overlap the occluder bbox and are later
        # composited under that foreground layer by z-index.
        strong_completion_excludes = [
            entry for entry in exclude_bboxes if is_strong_exclude(entry)
        ]
        if strong_completion_excludes:
            strong_mask = build_exclude_mask(strong_completion_excludes, img_w, img_h)
            cleaned = merged * (~strong_mask).astype(np.float32)
            cleaned_binary = cleaned > 0.5
            cleaned_target_area = int(np.count_nonzero(cleaned_binary[ty1:ty2, tx1:ty2]))
            cleaned_fill_ratio = cleaned_target_area / target_area
            if cleaned_fill_ratio >= max(0.08, target_fill_ratio * 0.45):
                merged = cleaned
                target_fill_ratio = cleaned_fill_ratio

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
        high_coverage_entity=high_coverage_selected,
        policy=policy
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
        "baseStrategyType": base_strategy_type,
        "spatialCanonicalCompletion": policy["spatialCanonicalCompletion"],
        "profile": resolved_strategy.get("profile", "flat_design"),
        "features": resolved_strategy.get("features", []),
        "phase": resolved_strategy.get("phase", "initial"),
        "strategyProfile": strategy_profile,
        "selectedIndexes": [item["metrics"]["index"] for item in selected],
        "highCoverageEntity": high_coverage_selected,
        "debugCandidates": sorted(debug_candidates, key=lambda item: item["score"], reverse=True)[:12],
        "reason": "wall_art_quality_fallback" if forced_wall_art_fallback else quality_reason,
        "selectionMode": "completion_recovery" if completion_recovery else "publish",
        "runtimeAction": "accept" if forced_wall_art_fallback else quality_gate["runtimeAction"],
        "shouldGenerateRuntimeLayer": True if forced_wall_art_fallback else (quality_gate["shouldGenerateRuntimeLayer"] if selected else False),
        "needsHigherPrecision": False if forced_wall_art_fallback else (quality_gate["needsHigherPrecision"] or not selected),
        "issues": ["wall_art_quality_fallback"] if forced_wall_art_fallback else (quality_gate["issues"] if selected else ["no_selected_mask"]),
        "recommendedEngine": quality_gate["recommendedEngine"]
    }
    selected_debug = next(
        (row for row in debug_candidates if isinstance(row, dict) and row.get("selected")),
        None
    )
    quality["completionIdentityVerified"] = bool(
        selected_debug and (
            selected_debug.get("completionObservationCandidate") is True or
            selected_debug.get("completionRecoveryCandidate") is True or
            selected_debug.get("spatialCompletionAnchor") is True
        )
    ) if policy["spatialCanonicalCompletion"] else None
    quality["completionStructureVerified"] = bool(
        selected_debug and
        int(selected_debug.get("completionRecoveryPixels", 0) or 0) >= 256 and
        (
            float(selected_debug.get("completionRecoveryRatio") or 0.0) >= 0.03 or
            int(completion_hidden_rescue_pixels) >= 64 or
            int(selected_debug.get("completionRecoveryPixels", 0) or 0) >= 1024
        )
    ) if policy["spatialCanonicalCompletion"] else None
    # Keep the raw multimask evidence available to the completion debugger.
    # These are alpha-only crops, never used by selection or replacement.
    # Limiting this to spatial completion avoids enlarging ordinary SAM
    # responses and keeps other category extraction untouched.
    if policy["spatialCanonicalCompletion"]:
        preview_rows = []
        debug_by_index = {
            int(row.get("index")): row
            for row in debug_candidates
            if isinstance(row, dict) and row.get("index") is not None
        }
        # Four candidates plus the four chain-level previews keeps the
        # diagnostic conversation bounded while still showing the alternatives
        # that can explain a missing support/leg.
        for item in fallback_candidates[:4]:
            index = int(item.get("metrics", {}).get("index", -1))
            preview = build_completion_candidate_preview(item.get("mask"), target_bbox)
            if not preview:
                continue
            row = debug_by_index.get(index, {})
            preview_rows.append({
                "index": index,
                "dataUrl": preview,
                "selected": bool(row.get("selected")),
                "candidate": bool(row.get("candidate")),
                "rejectReason": row.get("rejectReason", "")
            })
        quality["completionCandidatePreviews"] = preview_rows
    quality["policyVersion"] = policy["version"]
    quality["selectionMode"] = policy["selector"]
    quality["matteType"] = policy["matteType"]
    quality["completionOutputMode"] = (
        "full_scene_sam" if (completion_full_scene_mask or completion_full_scene_candidate) else
        "observation_fallback" if completion_observation_fallback else
        "incremental_recovery" if completion_recovery else
        "not_applicable"
    )
    quality["completionHiddenRescuePixels"] = int(completion_hidden_rescue_pixels)
    quality["completionHiddenRescuePolicy"] = (
        "occlusion_bounded_multimask_delta"
        if policy["spatialCanonicalCompletion"] else None
    )
    quality["maskAudit"] = mask_integrity_audit(merged, target_bbox)

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
