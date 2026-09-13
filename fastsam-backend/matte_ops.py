"""Prompt construction and alpha/matte post-processing."""

import cv2
import numpy as np

from sam_runtime import *
from segmentation_policy import *
from mask_ops import *
from candidate_selection import *
from segmentation_primitives import normalize_result_masks

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
    foreground_context_mask = decode_completion_foreground_context_mask(
        (layer_meta or {}).get("completionForegroundContextMask"),
        img_w,
        img_h
    ) if resolve_mask_policy(layer_meta or {}, "completion")["spatialCanonicalCompletion"] else None

    label_cleanup_mask = np.zeros((img_h, img_w), dtype=bool)
    flat_cleanup_mask = np.zeros((img_h, img_w), dtype=bool)
    negative_mask = strong_exclude_mask.copy()
    if foreground_context_mask is not None:
        negative_mask |= foreground_context_mask
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


def recover_hard_edge_mask_with_points(
    img,
    mask,
    target_bbox,
    layer_name,
    strategy_type=None,
    model_variant="b"
):
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
            imgsz=choose_sam_imgsz(crop, strategy_type, model_variant=model_variant),
            points=[local_points],
            labels=[[1] * len(local_points)],
            model_variant=model_variant
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


def recover_completion_mask_with_points(
    img,
    mask,
    target_bbox,
    layer_name,
    occlusion_mask=None,
    foreground_context_mask=None,
    model_variant="l"
):
    """Recover generated target pixels behind verified occluders.

    The ordinary hard-edge recovery intentionally allows only a small contour
    growth.  Completion needs a different invariant: new pixels are allowed
    to be larger, but only inside the supplied occlusion region and never in
    independently verified foreground context.  This keeps recovery generic
    for tables, furniture, and other spatial objects without relaxing their
    normal extraction gates.
    """
    coarse_binary = mask > 0.5
    if not np.any(coarse_binary) or occlusion_mask is None or not np.any(occlusion_mask):
        return mask, False, {"status": "skipped:no_completion_occlusion_region"}

    positive = build_hard_edge_completion_points(
        img,
        mask,
        target_bbox,
        max_points=6
    )
    if not positive:
        return mask, False, {"status": "skipped:no_completion_boundary_points"}

    crop, crop_bounds = crop_region_from_bbox(img, target_bbox, 0.04)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_h, crop_w = crop.shape[:2]
    if crop.size == 0:
        return mask, False, {"status": "skipped:empty_completion_crop"}

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
            imgsz=choose_sam_imgsz(crop, "table", model_variant=model_variant),
            points=[local_points],
            labels=[[1] * len(local_points)],
            model_variant=model_variant
        )
        candidates = normalize_result_masks(
            results,
            crop_w,
            crop_h,
            interpolation=cv2.INTER_LINEAR
        )
    except Exception as error:
        print(f"Completion recovery SAM failed for {layer_name}: {error}")
        return mask, False, {"status": "failed:sam_error", "error": str(error)}

    coarse_crop = coarse_binary[crop_y1:crop_y2, crop_x1:crop_x2]
    occlusion_crop = (occlusion_mask[crop_y1:crop_y2, crop_x1:crop_x2] > 0.5)
    context_crop = (
        foreground_context_mask[crop_y1:crop_y2, crop_x1:crop_x2] > 0.5
        if foreground_context_mask is not None else
        np.zeros_like(occlusion_crop, dtype=bool)
    )
    coarse_area = max(1, int(np.count_nonzero(coarse_crop)))
    allowed_area = max(1, int(np.count_nonzero(occlusion_crop)))
    best = None
    best_score = None
    best_debug = None

    for index, candidate in enumerate(candidates):
        candidate_binary = candidate > 0.5
        candidate_area = int(np.count_nonzero(candidate_binary))
        if candidate_area <= 0:
            continue
        overlap = int(np.count_nonzero(candidate_binary & coarse_crop))
        new_pixels = candidate_binary & (~coarse_crop)
        allowed_new = new_pixels & occlusion_crop
        spill_pixels = new_pixels & (~occlusion_crop)
        context_pixels = candidate_binary & context_crop
        preserve_ratio = overlap / coarse_area
        new_area = int(np.count_nonzero(new_pixels))
        allowed_new_area = int(np.count_nonzero(allowed_new))
        spill_area = int(np.count_nonzero(spill_pixels))
        inside_ratio = int(np.count_nonzero(
            candidate_binary[local_bbox[1]:local_bbox[3], local_bbox[0]:local_bbox[2]]
        )) / max(1, candidate_area)
        context_ratio = int(np.count_nonzero(context_pixels)) / max(1, candidate_area)
        allowed_ratio = allowed_new_area / max(1, new_area)
        growth_ratio = candidate_area / coarse_area
        # A candidate may recover a complete hidden support, but its growth is
        # still bounded by the actual occlusion area and must not spill across
        # the generated scene.
        max_growth = min(2.8, 1.0 + (allowed_area / coarse_area) * 1.15)
        if (
            preserve_ratio < 0.78 or
            new_area <= 0 or
            allowed_ratio < 0.72 or
            spill_area > max(12, int(allowed_new_area * 0.28)) or
            growth_ratio > max_growth or
            inside_ratio < 0.72 or
            context_ratio > 0.08
        ):
            continue
        score = (
            (preserve_ratio * 0.34) +
            (min(1.0, allowed_new_area / allowed_area) * 0.42) +
            (inside_ratio * 0.18) -
            (context_ratio * 0.24) -
            (spill_area / max(1, candidate_area) * 0.16)
        )
        if best_score is None or score > best_score:
            best_score = score
            best = candidate_binary
            best_debug = {
                "index": index,
                "preserveRatio": round(preserve_ratio, 3),
                "allowedNewPixels": allowed_new_area,
                "allowedRecoveryRatio": round(allowed_new_area / allowed_area, 3),
                "growthRatio": round(growth_ratio, 3),
                "contextRatio": round(context_ratio, 3),
                "spillPixels": spill_area,
            }

    if best is None:
        return mask, False, {
            "status": "rejected:no_safe_completion_candidate",
            "candidateCount": int(len(candidates)),
        }

    recovered = mask.astype(np.float32).copy()
    recovered[crop_y1:crop_y2, crop_x1:crop_x2] = np.maximum(
        recovered[crop_y1:crop_y2, crop_x1:crop_x2],
        best.astype(np.float32)
    )
    recovered = constrain_mask_to_bbox(recovered, target_bbox)
    debug = {"status": "accepted:occlusion_bounded_recovery", **(best_debug or {})}
    print(
        f"Completion mask recovery for {layer_name}: "
        f"status={debug['status']} allowed={debug.get('allowedNewPixels', 0)} "
        f"growth={debug.get('growthRatio', 0):.3f}"
    )
    return recovered, True, debug


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
