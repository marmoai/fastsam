"""Model-independent mask cleanup, geometry, and prompt primitives."""

import cv2
import numpy as np

from sam_runtime import *
from segmentation_policy import *
from segmentation_primitives import (
    build_boundary_negative_points,
    normalize_result_masks,
)

def mask_integrity_audit(mask, target_bbox):
    """Produce model-independent diagnostics for a selected binary mask."""
    binary = np.asarray(mask > 0.5, dtype=np.uint8)
    bbox = mask_bbox(binary)
    if bbox is None:
        return {
            "status": "empty",
            "pixels": 0,
            "components": 0,
            "enclosedHolePixels": 0,
            "bbox": None,
            "touchesTargetEdges": 0
        }

    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    crop = binary[max(0, y1):min(binary.shape[0], y2), max(0, x1):min(binary.shape[1], x2)]
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    substantial = sum(
        1 for label in range(1, component_count)
        if int(component_stats[label, cv2.CC_STAT_AREA]) >= MASK_COMPONENT_MIN_PIXELS
    )
    inverted = (crop == 0).astype(np.uint8)
    hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    enclosed_holes = 0
    for label in range(1, hole_count):
        hx, hy, hw, hh, area = [int(value) for value in hole_stats[label]]
        if hx > 0 and hy > 0 and hx + hw < inverted.shape[1] and hy + hh < inverted.shape[0]:
            enclosed_holes += area
    bx1, by1, bx2, by2 = bbox
    touches = int(bx1 <= x1 + 2) + int(by1 <= y1 + 2)
    touches += int(bx2 >= x2 - 2) + int(by2 >= y2 - 2)
    return {
        "status": "ok",
        "pixels": int(np.count_nonzero(binary)),
        "components": int(substantial),
        "enclosedHolePixels": int(enclosed_holes),
        "bbox": [int(value) for value in bbox],
        "touchesTargetEdges": int(touches)
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
    is continuous with the immediately adjacent accepted entity mask.
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


def refine_furniture_mask_with_internal_points(
    img,
    mask,
    target_bbox,
    layer_name,
    model_variant="l"
):
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
            labels=[prompt_labels],
            model_variant=model_variant
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


def refine_furniture_mask_with_cross_model_evidence(
    img,
    b_masks,
    l_mask,
    target_bbox,
    layer_name,
    allow_disconnected_components=False
):
    """Ask L to validate regions present in B but absent from the L primary.

    This is intentionally class-agnostic. It does not assume that the missing
    region is a base, leg, arm, or any other named part. A B-only region must
    receive an L positive point and survive L's own mask selection before it
    can be added. Non-composite furniture keeps the proximity requirement;
    verified composite occluders may contribute separate, bbox-contained
    components.
    """
    if b_masks is None or len(b_masks) == 0:
        return l_mask, False, {"status": "skipped:no_b_masks", "components": 0, "pixels": 0}
    # Keep every B multimask as evidence, but never use it as output. A lower
    # confidence peer can contain a real attached part omitted by the primary;
    # L point validation below decides whether that evidence is trustworthy.
    b_binary = np.zeros_like(np.asarray(l_mask > 0.5, dtype=bool), dtype=bool)
    for b_candidate in b_masks:
        b_binary |= np.asarray(b_candidate > 0.5, dtype=bool)
    l_binary = np.asarray(l_mask > 0.5, dtype=bool)
    if not np.any(b_binary) or not np.any(l_binary):
        return l_mask, False, {"status": "skipped:no_masks", "components": 0, "pixels": 0}

    height, width = l_binary.shape
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = clamp(tx1, 0, width - 1)
    ty1 = clamp(ty1, 0, height - 1)
    tx2 = clamp(tx2, tx1 + 1, width)
    ty2 = clamp(ty2, ty1 + 1, height)
    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    min_side = max(1, min(tx2 - tx1, ty2 - ty1))

    # Use B only as a source of disagreement evidence. Large B envelopes are
    # not fed back wholesale. For an explicitly composite occluder, however,
    # disconnected B-supported components are expected: requiring every
    # component to be within a small radius of the L primary was the reason a
    # grouped pair of stools lost one side before GPT inpainting.
    near_radius = max(8, min(28, int(round(min_side * 0.045))))
    near_l = cv2.dilate(
        l_binary.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (near_radius * 2 + 1, near_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    disagreement = b_binary & (~l_binary)
    disagreement_component_count, disagreement_labels, disagreement_stats, _ = cv2.connectedComponentsWithStats(
        disagreement.astype(np.uint8), connectivity=8
    )
    substantial_disagreement_components = sum(
        1 for component_index in range(1, disagreement_component_count)
        if int(disagreement_stats[component_index, cv2.CC_STAT_AREA]) >= max(32, int(target_area * 0.00015))
    )
    disconnected_mode = bool(
        allow_disconnected_components and substantial_disagreement_components >= 2
    )
    if not disconnected_mode:
        disagreement &= near_l
    disagreement[:ty1] = False
    disagreement[ty2:] = False
    disagreement[:, :tx1] = False
    disagreement[:, tx2:] = False
    if not np.any(disagreement):
        return l_mask, False, {
            "status": "skipped:no_near_disagreement",
            "components": 0,
            "pixels": 0,
        }

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        disagreement.astype(np.uint8),
        connectivity=8
    )
    proposal = np.zeros_like(disagreement, dtype=bool)
    proposal_components = 0
    proposal_pixels = 0
    max_component_area = max(
        128,
        int(target_area * (0.34 if disconnected_mode else 0.10))
    )
    max_total_area = max(
        256,
        int(target_area * (0.62 if disconnected_mode else 0.18))
    )
    for label in range(1, count):
        x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
        if area < max(32, int(target_area * 0.00015)) or area > max_component_area:
            continue
        component = labels == label
        proposal |= component
        proposal_components += 1
        proposal_pixels += area
        if proposal_pixels >= max_total_area:
            break

    if not np.any(proposal):
        return l_mask, False, {
            "status": "skipped:disagreement_filtered",
            "components": 0,
            "pixels": 0,
        }

    positive_points = []
    proposal_count, proposal_labels, _, _ = cv2.connectedComponentsWithStats(
        proposal.astype(np.uint8),
        connectivity=8
    )
    for label in range(1, proposal_count):
        component = proposal_labels == label
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        py, px = np.unravel_index(int(np.argmax(distance)), distance.shape)
        if distance[py, px] >= 2:
            positive_points.append([int(px), int(py)])
        if len(positive_points) >= 6:
            break

    if not positive_points:
        return l_mask, False, {
            "status": "skipped:no_disagreement_points",
            "components": proposal_components,
            "pixels": proposal_pixels,
        }

    # Anchor the existing L body and explicitly discourage bbox-edge growth.
    anchor_distance = cv2.distanceTransform(l_binary.astype(np.uint8), cv2.DIST_L2, 5)
    for _ in range(5):
        py, px = np.unravel_index(int(np.argmax(anchor_distance)), anchor_distance.shape)
        if anchor_distance[py, px] < 3:
            break
        positive_points.append([int(px), int(py)])
        cv2.circle(anchor_distance, (int(px), int(py)), max(16, int(min_side * 0.12)), 0, thickness=-1)
    # Do not place a background point on a bbox edge that contains a disputed
    # B/L region. Missing subject parts often touch the semantic bbox edge;
    # marking that same edge as negative makes the validation self-contradictory.
    negative_points = build_boundary_negative_points(
        l_binary | proposal,
        [tx1, ty1, tx2, ty2],
        max_points=4
    )
    points = positive_points + negative_points
    labels_prompt = ([1] * len(positive_points)) + ([0] * len(negative_points))
    evidence_debug = {
        "bEvidencePixels": int(np.count_nonzero(b_binary[ty1:ty2, tx1:tx2])),
        "disagreementPixels": int(np.count_nonzero(disagreement)),
        "positivePoints": len(positive_points),
        "negativePoints": len(negative_points),
        "disconnectedComponentsAllowed": bool(allow_disconnected_components),
    }

    try:
        results = run_sam_bbox_inference(
            img,
            target_bbox,
            multimask_output=True,
            imgsz=HARD_EDGE_SAM_IMGSZ,
            points=[points],
            labels=[labels_prompt],
            model_variant="l"
        )
        candidates = normalize_result_masks(
            results,
            width,
            height,
            interpolation=cv2.INTER_LINEAR,
            debug_label=f"{layer_name} strategy=cross_model_refine model=L"
        )
        del results
    except Exception as error:
        print(f"Furniture cross-model SAM refine failed for {layer_name}: {error}")
        return l_mask, False, {
            "status": "failed:sam_error",
            "components": proposal_components,
            "pixels": proposal_pixels,
            **evidence_debug,
        }

    l_area = max(1, int(np.count_nonzero(l_binary)))
    l_core_radius = max(2, min(10, int(round(min_side * 0.012))))
    l_core = cv2.erode(
        l_binary.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (l_core_radius * 2 + 1, l_core_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    proposal_support = cv2.dilate(
        proposal.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1
    ) > 0
    best = None
    best_score = None
    proposal_area = max(1, int(np.count_nonzero(proposal)))
    for candidate in candidates:
        candidate_binary = constrain_mask_to_bbox(candidate, target_bbox) > 0.5
        if not np.any(candidate_binary):
            continue
        raw_new_pixels = candidate_binary & (~l_binary)
        raw_new_area = int(np.count_nonzero(raw_new_pixels))
        if raw_new_area <= 0:
            continue
        # Treat the existing L silhouette as authoritative. The refinement is
        # allowed to contribute only its validated additions; a point-guided
        # SAM response may otherwise shave a few pixels from the original body.
        # SAM may redraw a wider contour around the prompted region. Only the
        # part supported by the B/L disagreement is eligible for merging.
        new_pixels = raw_new_pixels & proposal_support
        new_area = int(np.count_nonzero(new_pixels))
        evidence_coverage = new_area / proposal_area
        if new_area <= 0:
            continue
        merged_binary = l_binary | new_pixels
        preserved_core = int(np.count_nonzero(merged_binary & l_core)) / max(1, int(np.count_nonzero(l_core)))
        preserved_l = int(np.count_nonzero(merged_binary & l_binary)) / l_area
        expected_coverage = int(np.count_nonzero(new_pixels & proposal)) / max(1, new_area)
        unsupported = int(np.count_nonzero(new_pixels & (~near_l)))
        growth = (l_area + new_area) / l_area
        if (
            preserved_core < 0.985 or
            preserved_l < 0.94 or
            evidence_coverage < 0.10 or
            expected_coverage < 0.55 or
            (
                not disconnected_mode and
                unsupported > max(64, int(new_area * 0.10))
            ) or
            growth > (2.40 if disconnected_mode else 1.30)
        ):
            continue
        score = (
            evidence_coverage * 2.0 +
            expected_coverage * 2.0 +
            preserved_l * 1.5 -
            unsupported / max(1, new_area)
        )
        if best_score is None or score > best_score:
            best = merged_binary.astype(np.float32)
            best_score = score
            best_debug = {
                "rawNewPixels": raw_new_area,
                "validatedPixels": new_area,
                "evidenceCoverage": evidence_coverage,
                "expectedCoverage": expected_coverage,
            }

    if best is None:
        return l_mask, False, {
            "status": "rejected:no_safe_candidate",
            "components": proposal_components,
            "pixels": proposal_pixels,
            **evidence_debug,
        }
    return best, True, {
        "status": "accepted",
        "disconnectedComponentsAllowed": bool(disconnected_mode),
        "disagreementComponents": int(substantial_disagreement_components),
        "components": proposal_components,
        "pixels": proposal_pixels,
        "areaBefore": l_area,
        "areaAfter": int(np.count_nonzero(best > 0.5)),
        **evidence_debug,
        **best_debug,
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
