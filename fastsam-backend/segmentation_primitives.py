"""Shared low-level segmentation primitives.

These helpers deliberately depend only on NumPy/OpenCV.  They are imported
by mask cleanup, prompt construction, and candidate selection so those
modules do not rely on the old monolithic ``main.py`` global namespace.
"""

import cv2
import numpy as np


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


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

    components.sort(key=lambda item: item["area"], reverse=True)
    return [
        [int(round(item["centroid"][0])), int(round(item["centroid"][1]))]
        for item in components[:max_points]
    ]


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
    probes = [
        (0, region.shape[1] // 2),
        (region.shape[0] - 1, region.shape[1] // 2),
        (region.shape[0] // 2, 0),
        (region.shape[0] // 2, region.shape[1] - 1),
    ]
    for ry, rx in probes:
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
    return [x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y]


def bbox_from_center(cx, cy, half_w, half_h, img_w, img_h):
    return [
        clamp(int(round(cx - half_w)), 0, img_w - 1),
        clamp(int(round(cy - half_h)), 0, img_h - 1),
        clamp(int(round(cx + half_w)), 1, img_w),
        clamp(int(round(cy + half_h)), 1, img_h),
    ]


def sample_points_in_bbox(box, fractions):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return [
        [
            clamp(int(round(x1 + width * fx)), x1, max(x1, x2 - 1)),
            clamp(int(round(y1 + height * fy)), y1, max(y1, y2 - 1)),
        ]
        for fx, fy in fractions
    ]


def point_in_bbox(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px < x2 and y1 <= py < y2


def point_in_any_bbox(point, boxes):
    return any(point_in_bbox(point, box) for box in boxes)


def normalize_result_masks(results, img_w, img_h, interpolation=cv2.INTER_NEAREST, debug_label=None):
    if results is None or len(results) == 0 or results[0].masks is None:
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
