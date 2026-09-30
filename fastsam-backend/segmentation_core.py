"""SAM candidate routing and cutout materialization pipeline."""

import os
import time

import cv2
import numpy as np

from sam_runtime import *
from segmentation_policy import *
from mask_ops import *
from candidate_selection import *
from matte_ops import *
from sam_diagnostics import sam_diagnostic_layer, sam_diagnostic_event
from boundary_matte import recover_safe_outer_boundary, review_hard_boundary


def audit_local_refine_candidate(base_mask, candidate_mask, target_bbox, strategy_type):
    """Accept only a non-destructive hard-product local refinement.

    Local SAM is useful for a narrow rim, but its multimask selector can
    reinterpret a low-contrast product interior as background.  The selected
    SAM mask is therefore the authority for the subject core; local refinement
    may improve the rim, but it may not trade away established foreground.
    This audit intentionally uses only the source masks and geometry, never a
    hand label or a category-specific image assumption.
    """
    base = np.asarray(base_mask > 0.5, dtype=bool)
    candidate = np.asarray(candidate_mask > 0.5, dtype=bool)
    base_area = int(base.sum())
    candidate_area = int(candidate.sum())
    if base_area <= 0 or candidate_area <= 0:
        return False, {
            "status": "rejected",
            "reason": "empty_mask",
            "basePixels": base_area,
            "candidatePixels": candidate_area,
        }

    bbox_width = max(1, int(target_bbox[2]) - int(target_bbox[0]))
    bbox_height = max(1, int(target_bbox[3]) - int(target_bbox[1]))
    # Keep the accepted interior while allowing changes in a narrow rim.
    band_radius = max(2, min(14, int(round(min(bbox_width, bbox_height) * 0.024))))
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_radius + 1, 2 * band_radius + 1))
    stable_core = cv2.erode(base.astype(np.uint8), band_kernel, iterations=1) > 0
    boundary_band = (cv2.dilate(base.astype(np.uint8), band_kernel, iterations=1) > 0) & ~stable_core
    removed = base & ~candidate
    added = candidate & ~base
    outside_band = int(((removed | added) & ~boundary_band).sum())
    overlap = int((base & candidate).sum()) / max(1, base_area)
    core_area = max(1, int(stable_core.sum()))
    core_preserve = int((candidate & stable_core).sum()) / core_area
    removed_ratio = int(removed.sum()) / max(1, base_area)
    added_ratio = int(added.sum()) / max(1, base_area)
    changed_ratio = int((removed | added).sum()) / max(1, base_area)

    # Hard-product local refinement is a rim operation.  These limits are
    # relative to the accepted subject, so they scale across image sizes.
    max_removed_ratio = 0.015
    max_added_ratio = 0.12
    max_changed_ratio = 0.16
    core_removed_limit = max(8, int(round(base_area * 0.0005)))
    core_removed = int((removed & stable_core).sum())
    accepted = bool(
        strategy_type == "hard_product" and
        outside_band == 0 and
        core_preserve >= 0.995 and
        core_removed <= core_removed_limit and
        overlap >= 0.985 and
        removed_ratio <= max_removed_ratio and
        added_ratio <= max_added_ratio and
        changed_ratio <= max_changed_ratio
    )
    audit = {
        "status": "accepted" if accepted else "rejected",
        "reason": "verified_non_destructive_rim" if accepted else "core_or_pixel_budget_failed",
        "basePixels": base_area,
        "candidatePixels": candidate_area,
        "addedPixels": int(added.sum()),
        "removedPixels": int(removed.sum()),
        "coreRemovedPixels": core_removed,
        "outsideBoundaryBandPixels": outside_band,
        "boundaryBandRadius": band_radius,
        "coreRemovedLimit": core_removed_limit,
        "corePreserve": round(core_preserve, 5),
        "overlap": round(overlap, 5),
        "removedRatio": round(removed_ratio, 5),
        "addedRatio": round(added_ratio, 5),
        "changedRatio": round(changed_ratio, 5),
        "bbox": [int(target_bbox[0]), int(target_bbox[1]), int(target_bbox[2]), int(target_bbox[3])],
        "bboxMinSide": min(bbox_width, bbox_height),
    }
    return accepted, audit


def allow_thin_strict_edge_continuations(core_mask, added_mask, strict_exclude_mask, target_bbox, config=None):
    """Keep a tiny, attached contour run when a strict context box overlaps it.

    Semantic context boxes are useful ownership evidence but can overlap a
    subject's outer contour (for example, a title box crossing a plate rim).
    Only a small, thin component touching the selected core and the target-box
    edge may pass through that negative region. Broad or detached components
    remain subject to strict exclusion.
    """
    core = np.asarray(core_mask > 0.5, dtype=bool)
    added = np.asarray(added_mask, dtype=bool)
    strict = np.asarray(strict_exclude_mask, dtype=bool)
    allowed = np.zeros_like(core, dtype=bool)
    if core.shape != added.shape or core.shape != strict.shape or not np.any(core) or not np.any(added):
        return allowed, {"status": "skipped", "reason": "invalid_masks", "components": []}

    config = config or {}
    height, width = core.shape
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    target_width = max(1, tx2 - tx1)
    target_height = max(1, ty2 - ty1)
    core_area = int(np.count_nonzero(core))
    max_area = max(64, int(round(core_area * float(config.get("maxAreaRatio", 0.01)))))
    max_thickness = max(
        6,
        int(round(min(target_width, target_height) * float(config.get("maxThicknessRatio", 0.05))))
    )
    edge_margin = max(3, int(round(min(target_width, target_height) * 0.03)))
    attach_radius = max(1, int(config.get("attachRadius", 2)))
    attach_kernel = np.ones((2 * attach_radius + 1, 2 * attach_radius + 1), np.uint8)
    near_core = cv2.dilate(core.astype(np.uint8), attach_kernel, iterations=1) > 0
    # Analyze only the portion actually removed by strict ownership. A thin
    # true contour can be attached to a much larger valid proposal outside the
    # text box; measuring that entire proposal would hide its narrow geometry.
    strict_added = added & strict
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        strict_added.astype(np.uint8), connectivity=8
    )
    rows = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == component_id
        strict_ratio = 1.0
        x, y, comp_width, comp_height = [int(value) for value in stats[component_id, :4]]
        thickness = min(comp_width, comp_height)
        aspect = max(comp_width, comp_height) / max(1, thickness)
        edge_adjacent = bool(
            x <= tx1 + edge_margin or y <= ty1 + edge_margin or
            x + comp_width >= tx2 - edge_margin or
            y + comp_height >= ty2 - edge_margin
        )
        attached_ratio = int(np.count_nonzero(component & near_core)) / area
        accepted = bool(
            area <= max_area and
            thickness <= max_thickness and
            aspect >= float(config.get("minAspectRatio", 3.0)) and
            strict_ratio >= float(config.get("minStrictOverlap", 0.5)) and
            attached_ratio >= float(config.get("minAttachedRatio", 0.5)) and
            edge_adjacent
        )
        if accepted:
            allowed |= component
        rows.append({
            "component": int(component_id),
            "area": area,
            "bbox": [x, y, x + comp_width, y + comp_height],
            "strictOverlap": round(strict_ratio, 4),
            "attachedRatio": round(attached_ratio, 4),
            "aspectRatio": round(aspect, 3),
            "status": "allowed" if accepted else "blocked",
            "reason": "thin_attached_target_edge" if accepted else "insufficient_edge_evidence",
        })
    allowed_pixels = int(np.count_nonzero(allowed & strict))
    return allowed, {
        "status": "accepted" if allowed_pixels else "skipped",
        "reason": "thin_attached_target_edge" if allowed_pixels else "no_safe_edge_continuation",
        "allowedPixels": allowed_pixels,
        "components": rows,
    }


def remove_strictly_excluded_detached_components(mask, strict_exclude_mask):
    """Remove only small detached mask islands claimed by explicit strict context.

    Do not subtract a context rectangle from the main silhouette: such boxes
    may overlap real target pixels. A small disconnected component that is
    almost entirely covered by a strict text/badge ownership box is safer to
    reject as a whole.
    """
    binary = np.asarray(mask > 0.5, dtype=bool)
    strict = np.asarray(strict_exclude_mask, dtype=bool)
    if binary.shape != strict.shape or not np.any(binary) or not np.any(strict):
        return np.asarray(mask, dtype=np.float32), {
            "status": "skipped", "reason": "invalid_or_empty_masks", "removed": []
        }
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if count <= 2:
        return np.asarray(mask, dtype=np.float32), {
            "status": "skipped", "reason": "single_component", "removed": []
        }
    main_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    max_area = max(64, int(round(main_area * 0.03)))
    result = binary.copy()
    removed = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 24 or area > max_area or area == main_area:
            continue
        component = labels == component_id
        strict_overlap = int(np.count_nonzero(component & strict)) / max(1, area)
        if strict_overlap < 0.95:
            continue
        x, y, comp_width, comp_height = [int(value) for value in stats[component_id, :4]]
        result[component] = False
        removed.append({
            "component": int(component_id),
            "pixels": area,
            "bbox": [x, y, x + comp_width, y + comp_height],
            "strictOverlap": round(strict_overlap, 4),
            "reason": "small_detached_component_in_strict_context",
        })
    return result.astype(np.float32), {
        "status": "accepted" if removed else "skipped",
        "reason": "strict_context_component_removed" if removed else "no_safe_component",
        "removed": removed,
    }


def recover_strict_bbox_edge_continuations(
    baseline_mask,
    candidate_masks,
    target_bbox,
    strict_exclude_mask=None,
    config=None,
):
    """Recover only thin, attached peer-mask pixels just outside a strict bbox.

    A high-confidence semantic bbox is normally an output contract. If that
    bbox is a couple of pixels tighter than SAM's consistent contour evidence,
    however, clipping it can shave the silhouette. This narrowly permits
    peer-candidate pixels that touch the selected in-box contour, lie in a
    small edge band, and do not overlap strict semantic exclusions.
    """
    baseline = np.asarray(baseline_mask > 0.5, dtype=bool)
    height, width = baseline.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1, y1 = clamp(x1, 0, width - 1), clamp(y1, 0, height - 1)
    x2, y2 = clamp(x2, x1 + 1, width), clamp(y2, y1 + 1, height)
    config = config or {}
    extension = max(
        1,
        min(6, int(round(min(x2 - x1, y2 - y1) * float(config.get("maxExtensionRatio", 0.015)))))
    )
    strict = np.zeros_like(baseline, dtype=bool)
    if strict_exclude_mask is not None:
        strict_candidate = np.asarray(strict_exclude_mask, dtype=bool)
        if strict_candidate.shape == baseline.shape:
            strict = strict_candidate

    in_box = np.zeros_like(baseline, dtype=bool)
    in_box[y1:y2, x1:x2] = True
    core = baseline & in_box
    if not np.any(core) or candidate_masks is None or len(candidate_masks) < 2:
        return baseline.astype(np.float32), {
            "status": "skipped", "reason": "no_peer_candidate_or_core",
            "addedPixels": 0, "effectiveBbox": [x1, y1, x2, y2],
        }

    sides = {
        "left": (slice(y1, y2), slice(max(0, x1 - extension), x1)),
        "right": (slice(y1, y2), slice(x2, min(width, x2 + extension))),
        "top": (slice(max(0, y1 - extension), y1), slice(x1, x2)),
        "bottom": (slice(y2, min(height, y2 + extension)), slice(x1, x2)),
    }
    proposed = np.zeros_like(baseline, dtype=bool)
    for candidate in candidate_masks:
        peer = np.asarray(candidate > 0.5, dtype=bool)
        if peer.shape != baseline.shape:
            continue
        for ys, xs in sides.values():
            if ys.stop <= ys.start or xs.stop <= xs.start:
                continue
            proposed[ys, xs] |= peer[ys, xs]

    proposed &= ~in_box
    proposed &= ~strict
    # Retain only 8-connected peer components that reach the established
    # in-box silhouette. This allows a narrow multi-pixel continuation while
    # rejecting detached badges/text in the same semantic edge band.
    connected = core | proposed
    component_count, component_labels = cv2.connectedComponents(
        connected.astype(np.uint8), connectivity=8
    )
    core_labels = np.unique(component_labels[core])
    keep_components = np.zeros(component_count, dtype=bool)
    keep_components[core_labels] = True
    keep_components[0] = False
    additions = proposed & keep_components[component_labels]

    accepted_sides = {}
    for side, (ys, xs) in sides.items():
        band = np.zeros_like(baseline, dtype=bool)
        if ys.stop > ys.start and xs.stop > xs.start:
            band[ys, xs] = True
        side_pixels = int(np.count_nonzero(additions & band))
        side_length = (y2 - y1) if side in {"left", "right"} else (x2 - x1)
        support = side_pixels / max(1, side_length)
        if support < float(config.get("minBoundarySupportRatio", 0.05)):
            additions &= ~band
            continue
        accepted_sides[side] = {
            "pixels": side_pixels,
            "boundarySupport": round(support, 4),
        }

    base_area = max(1, int(np.count_nonzero(core)))
    added_pixels = int(np.count_nonzero(additions))
    max_added = max(8, int(round(base_area * float(config.get("maxAddedRatio", 0.01)))))
    if added_pixels <= 0 or added_pixels > max_added:
        return baseline.astype(np.float32), {
            "status": "skipped", "reason": "edge_evidence_or_pixel_budget",
            "addedPixels": added_pixels, "maxAddedPixels": max_added,
            "effectiveBbox": [x1, y1, x2, y2], "sides": accepted_sides,
        }

    merged = baseline | additions
    ys, xs = np.where(additions)
    effective_bbox = [
        min(x1, int(xs.min())), min(y1, int(ys.min())),
        max(x2, int(xs.max()) + 1), max(y2, int(ys.max()) + 1),
    ]
    return merged.astype(np.float32), {
        "status": "accepted",
        "reason": "attached_peer_edge_continuation",
        "addedPixels": added_pixels,
        "maxAddedPixels": max_added,
        "extensionLimit": extension,
        "effectiveBbox": effective_bbox,
        "sides": accepted_sides,
    }


def resolve_sibling_pixel_exclusions(image, base_mask, sibling_entries, imgsz=1024):
    """Replace a sibling's overlapping box with a verified SAM silhouette."""
    h, w = base_mask.shape
    resolved = np.zeros((h, w), dtype=bool)
    audits = []
    for entry in sibling_entries:
        x1, y1, x2, y2 = map(int, entry["bbox"])
        box = np.zeros((h, w), dtype=bool)
        box[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        box_area = int(box.sum())
        if box_area < 64:
            resolved |= box
            audits.append({"status": "fallback", "reason": "small_bbox"})
            continue
        try:
            results = run_sam_bbox_inference(
                image, [x1, y1, x2, y2], multimask_output=True,
                imgsz=imgsz, model_variant="b"
            )
            candidates = normalize_result_masks(results, w, h)
            del results
        except Exception as error:
            resolved |= box
            audits.append({"status": "fallback", "reason": "inference_failed", "error": str(error)})
            continue
        eligible = []
        for candidate in candidates if candidates is not None else []:
            full = np.asarray(candidate > 0.5, dtype=bool)
            bounded = full & box
            area = int(bounded.sum())
            if (
                0.25 <= area / box_area <= 0.85 and
                area / max(1, int(full.sum())) >= 0.95 and
                int((bounded & (base_mask > 0.5)).sum()) / area <= 0.20
            ):
                eligible.append(bounded)
        pairs = []
        for first in range(len(eligible)):
            for second in range(first + 1, len(eligible)):
                intersection = eligible[first] & eligible[second]
                agreement = int(intersection.sum()) / max(1, int((eligible[first] | eligible[second]).sum()))
                if agreement >= 0.94:
                    pairs.append((agreement, intersection))
        if not pairs:
            resolved |= box
            audits.append({"status": "fallback", "reason": "no_stable_sibling_pair"})
            continue
        agreement, sibling = max(pairs, key=lambda pair: pair[0])
        sibling = cv2.dilate(
            sibling.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ) > 0
        sibling &= box
        resolved |= sibling
        audits.append({
            "status": "accepted", "reason": "stable_sibling_pixels",
            "agreement": round(agreement, 4), "blockedPixels": int(sibling.sum()),
            "bboxPixels": box_area,
        })
    return resolved, audits


def recover_furniture_candidate_supports(
    image, mask, candidates, target_bbox, layer_meta=None, context_layers=None, imgsz=1024
):
    """Accept only connected support growth backed by SAM and ownership evidence."""
    skipped = {"status": "skipped", "reason": "no_verified_support", "addedPixels": 0}
    if mask is None or candidates is None or len(candidates) < 2:
        return mask, skipped
    base = np.asarray(mask > 0.5, dtype=bool)
    h, w = base.shape
    if image.shape[:2] != (h, w) or not np.any(base):
        return mask, skipped
    x1, y1, x2, y2 = map(int, target_bbox)
    width, height = x2 - x1, y2 - y1
    if width < 100 or height < 100:
        return mask, skipped
    target = np.zeros_like(base)
    target[y1:y2, x1:x2] = True
    base_area = int(base.sum())
    peers = []
    for index, raw in enumerate(candidates):
        peer = np.asarray(raw > 0.5, dtype=bool)
        if peer.shape != base.shape or not np.any(peer):
            continue
        inside = int((peer & target).sum()) / int(peer.sum())
        preserve = int((peer & base).sum()) / base_area
        added = peer & target & ~base
        added_pixels = int(added.sum())
        if preserve >= 0.995 and added_pixels >= max(64, int(base_area * 0.02)):
            peers.append((index, inside, added_pixels, added))

    intent = (layer_meta or {}).get("segmentationIntent") or {}
    excluded = intent.get("excludedAdjacentObjects") or []
    sibling_boxes = []
    if float(intent.get("confidence") or 0) >= 0.9:
        for entry in excluded:
            raw_box = entry.get("bbox") if isinstance(entry, dict) else None
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                continue
            excluded_box = normalize_context_bbox_to_pixel(raw_box, w, h)
            for other in context_layers or []:
                if not isinstance(other, dict) or same_layer(layer_meta or {}, other):
                    continue
                if get_layer_strategy(other).get("type") != "furniture":
                    continue
                other_box = other.get("bbox")
                if not isinstance(other_box, list) or len(other_box) != 4:
                    continue
                other_box = normalize_context_bbox_to_pixel(other_box, w, h)
                overlap = intersection_area(excluded_box, other_box)
                if overlap / max(1, min(bbox_area(excluded_box), bbox_area(other_box))) >= 0.8:
                    sibling_boxes.append({"bbox": other_box})
                    break

    bounded = [
        row for row in peers
        if row[1] >= 0.985 and row[2] <= base_area * 0.06
    ]
    if bounded and sibling_boxes:
        sibling, audits = resolve_sibling_pixel_exclusions(
            image, base, sibling_boxes[:2], imgsz=imgsz
        )
        if all(row.get("status") == "accepted" for row in audits):
            for index, _, _, added in sorted(bounded, key=lambda row: row[2], reverse=True):
                safe_added = added & ~sibling
                if int(safe_added.sum()) < max(48, int(base_area * 0.01)):
                    continue
                trial = (base | safe_added) & target
                count, _, stats, _ = cv2.connectedComponentsWithStats(
                    trial.astype(np.uint8), connectivity=8
                )
                if count != 2 or int(stats[1, cv2.CC_STAT_AREA]) < base_area * 0.95:
                    continue
                return np.maximum(mask, safe_added.astype(np.float32)), {
                    "status": "accepted", "reason": "owned_connected_candidate",
                    "peerIndex": index, "addedPixels": int(safe_added.sum()),
                    "siblingPixelsBlocked": int((added & sibling).sum()),
                }

    if width < 160 or height < 150:
        return mask, skipped
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_median = float(np.median(gray[max(y1, y2 - 25):y2, x1:x2]))
    dark_limit = min(50, max(30, int(lower_median * 0.45)))
    start_y = y2 - max(20, int(round(height * 0.16)))
    for index, inside, added_pixels, added in peers:
        if not (0.78 <= inside < 0.95 and 0.05 <= added_pixels / base_area <= 0.15):
            continue
        dark = added[start_y:y2, x1:x2] & (gray[start_y:y2, x1:x2] < dark_limit)
        row_counts = np.count_nonzero(dark, axis=1)
        peak_offset = int(np.argmax(row_counts))
        if row_counts[peak_offset] < width * 0.22:
            continue
        support_x = np.flatnonzero(added[start_y + peak_offset, x1:x2])
        if support_x[-1] - support_x[0] + 1 < width * 0.45:
            continue
        peak_y = start_y + peak_offset
        top = max(start_y, peak_y - max(4, int(round(height * 0.024))))
        bottom = min(y2, peak_y + max(4, int(round(height * 0.017))))
        band = np.zeros_like(base)
        band[top:bottom, x1:x2] = True
        safe_added = added & band
        if not max(64, int(base_area * 0.008)) <= int(safe_added.sum()) <= base_area * 0.04:
            continue
        trial = (base | safe_added) & target
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            trial.astype(np.uint8), connectivity=8
        )
        if count != 2 or int(stats[1, cv2.CC_STAT_AREA]) < base_area * 0.95:
            continue
        return np.maximum(mask, safe_added.astype(np.float32)), {
            "status": "accepted", "reason": "connected_candidate_rail",
            "peerIndex": index, "addedPixels": int(safe_added.sum()),
            "supportBand": [top, bottom],
        }
    return mask, skipped


def audit_consensus_boundary_review(base_mask, reviewed_mask, peer_support):
    """Do not let color-only matte undo an independently agreed SAM subject."""
    base = np.asarray(base_mask > 0.5, dtype=bool)
    reviewed = np.asarray(reviewed_mask > 0.5, dtype=bool)
    peers = np.asarray(peer_support, dtype=bool)
    added = reviewed & ~base
    removed = base & ~reviewed
    added_pixels = int(added.sum())
    supported_additions = int((added & peers).sum())
    supported_removals = int((removed & peers).sum())
    accepted = (
        (added_pixels < 16 or supported_additions / added_pixels >= 0.5) and
        supported_removals <= max(16, int(base.sum() * 0.002))
    )
    return accepted, {
        "status": "accepted" if accepted else "rejected",
        "reason": "peer_supported_matte" if accepted else "peer_disagreement",
        "addedPixels": added_pixels,
        "peerSupportedAdditions": supported_additions,
        "peerSupportedRemovals": supported_removals,
    }


def reject_unsupported_probe_supports(probe_mask, baseline_masks, layer_meta, target_bbox, image=None):
    """Remove probe-only supports claimed by an explicitly excluded neighbour."""
    intent = (layer_meta or {}).get("segmentationIntent") or {}
    excluded = intent.get("excludedAdjacentObjects") or []
    if not excluded or float(intent.get("confidence") or 0) < 0.9:
        return probe_mask, {"status": "skipped", "reason": "no_explicit_exclusion"}
    probe = np.asarray(probe_mask > 0.5, dtype=bool)
    h, w = probe.shape
    baseline_array = np.asarray(baseline_masks) > 0.5
    baseline = (
        np.any(baseline_array, axis=0)
        if baseline_array.ndim == 3 else baseline_array
    )
    if baseline.shape != probe.shape:
        return probe_mask, {"status": "skipped", "reason": "shape_mismatch"}
    exclusion = np.zeros_like(probe)
    for entry in excluded:
        bbox = entry.get("bbox") if isinstance(entry, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = normalize_context_bbox_to_pixel(bbox, w, h)
        exclusion[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    if not np.any(exclusion):
        return probe_mask, {"status": "skipped", "reason": "no_valid_exclusion"}
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (probe & ~baseline).astype(np.uint8), connectivity=8
    )
    target_width = max(1, int(target_bbox[2]) - int(target_bbox[0]))
    target_height = max(1, int(target_bbox[3]) - int(target_bbox[1]))
    result = probe.copy()
    removed = []
    lab = (
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        if image is not None and image.shape[:2] == probe.shape else None
    )
    for index in range(1, len(stats)):
        x, y, width, height, area = map(int, stats[index])
        if (
            area < max(48, int(baseline.sum() * 0.01)) or
            width > target_width * 0.18 or
            height < target_height * 0.20 or
            height < width * 2.5
        ):
            continue
        component = labels == index
        if int((component & exclusion).sum()) / area < 0.8:
            continue
        result[component] = False
        root_pixels = 0
        if lab is not None:
            color = np.median(lab[component], axis=0)
            root = np.zeros_like(probe)
            root[
                max(0, y - max(4, int(round(width * 1.5)))):min(h, y + height),
                max(0, x - 3):min(w, x + width + 3)
            ] = True
            similar = np.linalg.norm(lab - color, axis=2) <= 22
            root &= result & exclusion & similar
            root_pixels = int(root.sum())
            result[root] = False
        removed.append({
            "bbox": [x, y, x + width, y + height],
            "pixels": area, "rootPixels": root_pixels,
        })
    return result.astype(np.float32), {
        "status": "accepted" if removed else "skipped",
        "reason": "unsupported_adjacent_support" if removed else "no_unsupported_support",
        "removed": removed,
    }


def recover_owned_candidate_consensus(
    image, base_mask, candidate_masks, target_bbox, ownership_excludes,
    blocked_mask=None, relaxed_region=None
):
    """Recover a missing part supported by two fuller SAM candidates.

    A complete candidate may include an adjacent instance. Only the pixels
    both peers propose inside the target's ownership region are considered;
    the selected mask remains immutable. This is an initial-candidate remedy,
    separate from local SAM refinement and its boundary-only acceptance gate.
    """
    base = np.asarray(base_mask > 0.5, dtype=bool)
    if image.shape[:2] != base.shape or not np.any(base) or len(candidate_masks) < 3:
        return base_mask, {"status": "skipped", "reason": "insufficient_candidates"}
    sibling_entries = [
        entry for entry in ownership_excludes
        if isinstance(entry, dict) and entry.get("spatial_ownership_guard")
    ]
    if not sibling_entries:
        return base_mask, {"status": "skipped", "reason": "no_verified_sibling"}

    h, w = base.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target = np.zeros_like(base)
    target[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    blocked = (
        build_exclude_mask(ownership_excludes, w, h)
        if blocked_mask is None else np.asarray(blocked_mask, dtype=bool)
    )
    base_area = int(base.sum())
    target_area = max(1, int(target.sum()))
    peers = []
    for index, raw in enumerate(candidate_masks):
        candidate = np.asarray(raw > 0.5, dtype=bool)
        if candidate.shape != base.shape:
            continue
        candidate_area = int(candidate.sum())
        bounded = candidate & target
        bounded_area = int(bounded.sum())
        preserve = int((bounded & base).sum()) / base_area
        added = bounded & ~base
        added_area = int(added.sum())
        blocked_growth = int((added & blocked).sum()) / max(1, added_area)
        if (
            preserve >= 0.99 and
            bounded_area / max(1, candidate_area) >= 0.97 and
            1.10 <= bounded_area / base_area <= 1.45 and
            blocked_growth <= 0.20
        ):
            peers.append((index, bounded, added_area))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    reference_edge = float(np.median(gradient[target]))
    accepted = []
    for first in range(len(peers)):
        for second in range(first + 1, len(peers)):
            first_index, first_mask, _ = peers[first]
            second_index, second_mask, _ = peers[second]
            consensus = first_mask & second_mask
            agreement = int(consensus.sum()) / max(1, int((first_mask | second_mask).sum()))
            if agreement < 0.985:
                continue
            added = consensus & ~base & ~blocked
            # The selected subject may have an incomplete rim, but a distant
            # island is not evidence that the same entity continues there.
            reach = max(4, min(24, int(round(min(x2 - x1, y2 - y1) * 0.10))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
            near_base = cv2.dilate(base.astype(np.uint8), kernel) > 0
            if relaxed_region is not None:
                extended_reach = max(reach, min(64, int(round(min(x2 - x1, y2 - y1) * 0.30))))
                extended_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * extended_reach + 1, 2 * extended_reach + 1)
                )
                near_base |= (
                    (cv2.dilate(base.astype(np.uint8), extended_kernel) > 0) &
                    np.asarray(relaxed_region, dtype=bool)
                )
            added &= near_base
            added_area = int(added.sum())
            if not 0.08 <= added_area / base_area <= 0.30:
                continue
            trial = base | added
            if int(trial.sum()) / target_area > 0.80:
                continue
            _, components = cv2.connectedComponents(trial.astype(np.uint8), connectivity=8)
            anchored = np.unique(components[base])
            added &= np.isin(components, anchored)
            added_area = int(added.sum())
            if added_area / base_area < 0.08:
                continue
            outer_edge = consensus & ~(cv2.erode(consensus.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            supported_edge = outer_edge & added
            edge_pixels = int(supported_edge.sum())
            edge_strength = float(np.median(gradient[supported_edge])) if edge_pixels else 0.0
            if edge_pixels < 24 or edge_strength < max(32.0, reference_edge * 2.0):
                continue
            accepted.append((added_area, agreement, first_index, second_index, added, edge_strength))

    if not accepted:
        return base_mask, {
            "status": "skipped", "reason": "no_safe_consensus",
            "eligiblePeers": [row[0] for row in peers],
        }
    added_area, agreement, first_index, second_index, added, edge_strength = max(accepted, key=lambda row: row[0])
    result = (base | added).astype(np.float32)
    return result, {
        "status": "accepted", "reason": "bounded_owned_peer_consensus",
        "peerIndexes": [int(first_index), int(second_index)],
        "peerAgreement": round(agreement, 5),
        "basePixels": base_area, "addedPixels": added_area,
        "edgeStrength": round(edge_strength, 2),
        "referenceEdgeStrength": round(reference_edge, 2),
    }


def promote_optional_completion_recovery(quality, completion_recovery_audit):
    """Treat point-prompt recovery as optional after SAM-L selected a safe mask.

    The post-inpaint candidate selector validates identity separately from
    hidden-structure evidence.  The later point-prompt pass is an enhancement
    and may legitimately find no additional pixels, but it cannot promote a
    visible-only fragment: spatial completion still needs a measured hidden
    delta (or bounded multimask rescue).
    """
    if not isinstance(completion_recovery_audit, dict):
        return completion_recovery_audit, False
    recovery_status = str(completion_recovery_audit.get("status", ""))
    if recovery_status.startswith("accepted:"):
        return completion_recovery_audit, False
    selected_candidate = next(
        (
            row for row in (quality or {}).get("debugCandidates", [])
            if isinstance(row, dict) and row.get("selected")
        ),
        None
    )
    if not selected_candidate or selected_candidate.get("candidate") is not True:
        return completion_recovery_audit, False
    selected_recovery_pixels = int(
        selected_candidate.get("completionRecoveryPixels", 0) or 0
    )
    selected_observation_recall = selected_candidate.get("observationRecall")
    try:
        selected_observation_recall = float(selected_observation_recall)
    except (TypeError, ValueError):
        selected_observation_recall = None
    def numeric(value, fallback=0.0):
        try:
            return float(value) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback
    selected_inside = numeric(selected_candidate.get("inside"))
    selected_bbox_overlap = numeric(selected_candidate.get("bboxOverlap"))
    selected_anchor_overlap = numeric(selected_candidate.get("anchorBboxOverlap"))
    selected_geometry_overlap = max(selected_bbox_overlap, selected_anchor_overlap)
    selected_completion_evidence = bool(
        selected_candidate.get("completionObservationCandidate") is True or
        selected_candidate.get("completionRecoveryCandidate") is True or
        selected_candidate.get("spatialCompletionAnchor") is True
    )
    is_spatial_completion = selected_candidate.get("spatialCompletionAnchor") is True
    selected_recovery_ratio = numeric(
        selected_candidate.get("completionRecoveryRatio")
    )
    hidden_rescue_pixels = int(
        (quality or {}).get("completionHiddenRescuePixels", 0) or 0
    )
    # Identity evidence (visible anchor/containment) is not structural
    # evidence.  For a spatial table, promotion is allowed only when the
    # selected mask or the bounded multimask rescue proves that it enters the
    # verified hidden zone.  This prevents an intact-looking but incomplete
    # visible fragment from being promoted merely because the point pass did
    # not find a safe candidate.
    spatial_structure_verified = (
        not is_spatial_completion or
        (
            selected_recovery_pixels >= 256 and
            (
                selected_recovery_ratio >= 0.03 or
                hidden_rescue_pixels >= 64 or
                # Older quality payloads did not include a ratio.  A
                # substantial absolute hidden delta remains valid evidence
                # for those responses.
                selected_recovery_pixels >= 1024
            )
        )
    )
    # The selected SAM-L candidate is already the authoritative multimask
    # decision for this pass.  Point-prompt recovery is only an optional
    # refinement.  In particular, a candidate can prove the hidden delta via
    # its own completionRecoveryPixels while the point prompt finds no safe
    # extra point; that must not downgrade an otherwise safe candidate to a
    # runtime hold.
    selected_candidate_has_bounded_hidden_delta = (
        selected_recovery_pixels >= 256 and
        selected_recovery_ratio >= 0.03
    )
    if not (
        selected_completion_evidence or selected_candidate_has_bounded_hidden_delta
    ) or not (
        selected_inside >= 0.80 and
        # The prompt bbox may include the whole occluder union.  For spatial
        # completion its overlap is not the target-identity test; the observed
        # anchor bbox is.  Use whichever verified geometry is available.
        selected_geometry_overlap >= 0.70 and
        # Observation recall alone proves only the visible identity anchor;
        # it must not authorize a mask that contains no pixels from the
        # generated/occluded zone.  The selected candidate's own recovery
        # accounting (or the stricter completion-observation gate) is the
        # required evidence here.
        (
            selected_recovery_pixels >= 64 or
            selected_candidate.get("completionObservationCandidate") is True
        ) and
        spatial_structure_verified
    ):
        return completion_recovery_audit, False
    return {
        **completion_recovery_audit,
        "status": "accepted:selected_candidate_recovery",
        "verification": "selected_sam_candidate",
        "allowedNewPixels": max(
            int(completion_recovery_audit.get("allowedNewPixels", 0) or 0),
            selected_recovery_pixels
        ),
        "fallback": True,
        "fallbackReason": "optional_point_recovery_missed"
    }, True

def should_escalate_sam_to_l(
    candidate_masks,
    target_bbox,
    strategy_type,
    quality_profile="publish",
    completion_observation=None,
    policy=None
):
    """Route only genuinely unsafe B masks to SAM-L.

    Completion candidates do not need a publish-ready matte before inpainting.
    They only need enough reliable foreground and containment to establish the
    target/occluder relationship. Keep L for structural failures, not ordinary
    holes or small visible-area loss.
    """
    quality_profile = normalize_sam_quality_profile(quality_profile)
    policy = policy or policy_for_strategy_type(strategy_type, quality_profile=quality_profile)
    if not policy.get("allowModelEscalation", True):
        return False, "strategy_prefers_b"
    if candidate_masks is None or len(candidate_masks) == 0:
        return True, "no_b_candidates"

    # The alpha for a foreground occluder becomes the model edit mask. It
    # therefore needs an independent model review even when B produced a
    # plausible-looking partial component. This is intentionally based on the
    # execution role, not an object label such as food, fruit, or furniture.
    if policy.get("completionOccluder") and not policy.get("softEdge"):
        return True, "completion_occluder_independent_l_review"

    multi_entity, multi_entity_reason = detect_multi_entity_candidate_disagreement(
        candidate_masks,
        target_bbox,
        strategy_type
    )
    # A larger alternative can be useful evidence for publish-quality
    # extraction, but it is not by itself a reason to pay for SAM-L when this
    # mask is only the observation used by object completion.
    if multi_entity and quality_profile != "completion":
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

    hard_entity = (
        strategy_type in HARD_EDGE_STRATEGIES or
        strategy_type in {"table", "furniture", "completion_object"}
    )
    if quality_profile == "completion":
        # A partially visible hard object is intentionally allowed here: the
        # completion model will infer the hidden body later. Containment and
        # structural sanity remain stricter than the fill threshold.
        fill_threshold = 0.30 if hard_entity else 0.24
        inside_threshold = 0.82
        hole_ratio = 0.04
        max_holes = 5000
    else:
        fill_threshold = 0.56 if hard_entity else 0.48
        inside_threshold = 0.94
        hole_ratio = 0.006 if hard_entity else 0.012
        max_holes = 1400 if hard_entity else 1200
    # At high resolution, a fixed hole count is too lenient for a large
    # textured object. Conversely, a pure percentage is too aggressive for a
    # small object. The hard-entity threshold catches real internal breaks
    # while leaving ordinary SAM raster noise on the B path.
    hole_threshold = max(
        max_holes,
        int(target_area * hole_ratio)
    )
    if quality_profile == "completion" and strategy_type == "completion_object":
        reference_bbox = mask_bbox(reference)
        if reference_bbox:
            rx1, ry1, rx2, ry2 = reference_bbox
            reference_touch_count = int(rx1 <= x1 + 2) + int(ry1 <= y1 + 2)
            reference_touch_count += int(rx2 >= x2 - 2) + int(ry2 >= y2 - 2)
            # A post-inpaint mask touching several prompt edges is commonly a
            # scene envelope or a partial foreground mask. Give SAM-L a
            # chance to resolve it before the generic completion gate runs.
            if reference_touch_count >= 3 or best_inside < 0.90:
                return True, (
                    f"completion_b_shape_ambiguous_touch={reference_touch_count}"
                )

    # A tightly framed furniture bbox can touch several prompt edges without
    # being a background envelope. If containment is borderline, let SAM-L
    # inspect it instead of trusting B's coarse fill acceptance.
    if strategy_type in {"table", "furniture"}:
        reference_bbox = mask_bbox(reference)
        if reference_bbox:
            rx1, ry1, rx2, ry2 = reference_bbox
            reference_touch_count = (
                int(rx1 <= x1 + 2) + int(ry1 <= y1 + 2) +
                int(rx2 >= x2 - 2) + int(ry2 >= y2 - 2)
            )
            reference_area_ratio = area / target_area
            if (
                reference_touch_count >= 3 and
                0.84 <= best_inside < 0.94 and
                0.96 <= reference_area_ratio <= 1.05
            ):
                return True, (
                    f"{strategy_type}_b_bbox_envelope_inside={best_inside:.3f}"
                )

    if best_fill < fill_threshold:
        return True, f"low_b_fill={best_fill:.3f}_profile={quality_profile}"
    if best_holes > hole_threshold:
        return True, f"b_internal_gaps={best_holes}>{hole_threshold}_profile={quality_profile}"
    if best_inside < inside_threshold:
        return True, f"low_b_inside={best_inside:.3f}_profile={quality_profile}"

    if quality_profile == "completion" and completion_observation is not None:
        observation_area = int(np.count_nonzero(completion_observation > 0.5))
        if observation_area > 0:
            observed_overlap = int(np.count_nonzero(binary & (completion_observation > 0.5)))
            observation_recall = observed_overlap / observation_area
            if observation_recall < 0.72:
                return True, (
                    f"low_completion_observation_recall={observation_recall:.3f}"
                )

    # A highly fragmented mask is not a useful completion observation even if
    # its total fill and containment ratios look acceptable.
    target_crop = binary[y1:y2, x1:x2].astype(np.uint8)
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        target_crop,
        connectivity=8
    )
    substantial_components = [
        int(component_stats[label, cv2.CC_STAT_AREA])
        for label in range(1, component_count)
        if int(component_stats[label, cv2.CC_STAT_AREA]) >= max(64, int(target_area * 0.01))
    ]
    if quality_profile == "completion" and len(substantial_components) >= 5:
        return True, f"fragmented_b_mask={len(substantial_components)}_profile={quality_profile}"

    if quality_profile == "completion" and best_fill > 0.94 and area / target_area > 0.98:
        return True, f"background_like_b_mask=fill:{best_fill:.3f}_area:{area / target_area:.3f}"

    return False, f"b_accepted_fill={best_fill:.3f}_profile={quality_profile}"


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
        if strategy_type in HARD_EDGE_STRATEGIES or strategy_type in {"table", "furniture", "completion_object"}:
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


def resolve_verified_effective_bbox(target_bbox, effective_bbox, img_w, img_h):
    """Return an audited expansion only when it contains the semantic bbox.

    A recovery audit may extend the visible subject beyond its original BBOX.
    B/L arbitration must use that same geometry; otherwise a later candidate
    pass can erase pixels that the recovery already proved.  Never accept a
    purported effective bbox that shrinks or displaces the semantic target.
    """
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    original = [
        clamp(x1, 0, img_w - 1),
        clamp(y1, 0, img_h - 1),
        clamp(x2, 1, img_w),
        clamp(y2, 1, img_h),
    ]
    if (
        not isinstance(effective_bbox, (list, tuple)) or
        len(effective_bbox) != 4 or
        not all(isinstance(value, (int, float)) for value in effective_bbox)
    ):
        return original, False
    ex1, ey1, ex2, ey2 = [int(value) for value in effective_bbox]
    candidate = [
        clamp(ex1, 0, img_w - 1),
        clamp(ey1, 0, img_h - 1),
        clamp(ex2, 1, img_w),
        clamp(ey2, 1, img_h),
    ]
    valid_expansion = bool(
        candidate[2] > candidate[0] and
        candidate[3] > candidate[1] and
        candidate[0] <= original[0] and
        candidate[1] <= original[1] and
        candidate[2] >= original[2] and
        candidate[3] >= original[3]
    )
    return (candidate if valid_expansion else original), valid_expansion


def refine_b_mask_boundary_with_l_evidence(
    b_masks,
    l_masks,
    target_bbox,
    strategy_type=None,
    config=None
):
    """Apply only audited SAM-L contour edits to an accepted SAM-B silhouette.

    This deliberately does not decide which adjacent instance belongs to a
    layer. SAM-B remains the source of identity and coverage.  SAM-L is merely
    a second boundary measurement: all of its changes are clipped to a narrow
    band around B, while B's eroded stable core stays immutable.
    """
    config = config or {}
    if b_masks is None or len(b_masks) == 0:
        return None, False, {"status": "skipped:no_b_masks"}
    if l_masks is None or len(l_masks) == 0:
        return None, False, {"status": "skipped:no_l_masks"}

    reference = choose_b_reference_mask(
        b_masks,
        target_bbox,
        strategy_type=strategy_type
    )
    if reference is None:
        return None, False, {"status": "skipped:no_b_reference"}

    base = np.asarray(reference > 0.5, dtype=bool)
    height, width = base.shape[:2]
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    tx1 = clamp(tx1, 0, width - 1)
    ty1 = clamp(ty1, 0, height - 1)
    tx2 = clamp(tx2, tx1 + 1, width)
    ty2 = clamp(ty2, ty1 + 1, height)
    constrained = np.zeros_like(base, dtype=bool)
    constrained[ty1:ty2, tx1:tx2] = True
    base &= constrained
    base_area = int(np.count_nonzero(base))
    if base_area < 96:
        return None, False, {"status": "skipped:small_b_reference", "basePixels": base_area}

    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    min_side = max(1, min(tx2 - tx1, ty2 - ty1))
    band_radius = max(
        2,
        min(12, int(round(min_side * float(config.get("bandRatio", 0.018)))))
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (band_radius * 2 + 1, band_radius * 2 + 1)
    )
    stable_core = cv2.erode(base.astype(np.uint8), kernel, iterations=1) > 0
    if int(np.count_nonzero(stable_core)) < max(48, int(base_area * 0.12)):
        return None, False, {
            "status": "skipped:insufficient_stable_core",
            "basePixels": base_area,
            "corePixels": int(np.count_nonzero(stable_core)),
            "bandRadius": band_radius,
        }
    outer_support = cv2.dilate(base.astype(np.uint8), kernel, iterations=1) > 0
    boundary_band = (outer_support & (~stable_core)) & constrained
    baseline_components, _, baseline_stats, _ = cv2.connectedComponentsWithStats(
        base.astype(np.uint8), connectivity=8
    )
    substantial_baseline_components = sum(
        1 for label in range(1, baseline_components)
        if int(baseline_stats[label, cv2.CC_STAT_AREA]) >= max(64, int(target_area * 0.004))
    )

    min_core_preserve = float(config.get("minStableCorePreserve", 0.985))
    min_baseline_preserve = float(config.get("minBaselinePreserve", 0.985))
    min_l_coverage = float(config.get("minLCoverageOfBaseline", 0.94))
    max_changed = max(64, int(target_area * float(config.get("maxChangedRatio", 0.075))))
    max_removed = max(32, int(base_area * float(config.get("maxRemovedRatio", 0.015))))
    max_added = max(32, int(base_area * float(config.get("maxAddedRatio", 0.06))))
    core_area = max(1, int(np.count_nonzero(stable_core)))

    best = None
    best_score = None
    rejection_counts = {}
    for candidate in l_masks:
        l_binary = np.asarray(candidate > 0.5, dtype=bool) & constrained
        if not np.any(l_binary):
            rejection_counts["empty_l_candidate"] = rejection_counts.get("empty_l_candidate", 0) + 1
            continue
        l_core_preserve = int(np.count_nonzero(l_binary & stable_core)) / core_area
        l_baseline_coverage = int(np.count_nonzero(l_binary & base)) / base_area
        if l_core_preserve < min_core_preserve or l_baseline_coverage < min_l_coverage:
            rejection_counts["l_disagrees_with_b_identity"] = rejection_counts.get("l_disagrees_with_b_identity", 0) + 1
            continue

        # Outside this band B is immutable.  This prevents a more conservative
        # L candidate from deleting an arm, support, or other already verified
        # portion of the subject, and prevents broad L background from growing
        # the selection beyond the semantic B silhouette.
        refined = (base & (~boundary_band)) | (l_binary & boundary_band)
        added = refined & (~base)
        removed = base & (~refined)
        added_area = int(np.count_nonzero(added))
        removed_area = int(np.count_nonzero(removed))
        changed_area = added_area + removed_area
        baseline_preserve = int(np.count_nonzero(refined & base)) / base_area
        stable_core_preserve = int(np.count_nonzero(refined & stable_core)) / core_area
        if (
            stable_core_preserve < min_core_preserve or
            baseline_preserve < min_baseline_preserve or
            added_area > max_added or
            removed_area > max_removed or
            changed_area > max_changed
        ):
            rejection_counts["boundary_change_out_of_bounds"] = rejection_counts.get("boundary_change_out_of_bounds", 0) + 1
            continue
        if changed_area < max(12, int(target_area * 0.00008)):
            rejection_counts["no_material_boundary_delta"] = rejection_counts.get("no_material_boundary_delta", 0) + 1
            continue

        refined_components, _, refined_stats, _ = cv2.connectedComponentsWithStats(
            refined.astype(np.uint8), connectivity=8
        )
        substantial_refined_components = sum(
            1 for label in range(1, refined_components)
            if int(refined_stats[label, cv2.CC_STAT_AREA]) >= max(64, int(target_area * 0.004))
        )
        if substantial_refined_components > substantial_baseline_components:
            rejection_counts["boundary_change_fragmented_subject"] = rejection_counts.get("boundary_change_fragmented_subject", 0) + 1
            continue

        l_iou = int(np.count_nonzero(l_binary & base)) / max(
            1,
            int(np.count_nonzero(l_binary | base))
        )
        # Prefer a candidate that independently agrees with B's full shape;
        # with that evidence tied, choose the smaller contour adjustment.
        score = (l_core_preserve * 3.0) + (l_iou * 2.0) - (changed_area / target_area)
        if best_score is None or score > best_score:
            best = refined
            best_score = score
            best_debug = {
                "addedPixels": added_area,
                "removedPixels": removed_area,
                "changedPixels": changed_area,
                "baselinePreserve": baseline_preserve,
                "stableCorePreserve": stable_core_preserve,
                "lBaselineCoverage": l_baseline_coverage,
                "lIou": l_iou,
                "substantialComponents": substantial_refined_components,
            }

    audit_base = {
        "basePixels": base_area,
        "corePixels": int(np.count_nonzero(stable_core)),
        "bandPixels": int(np.count_nonzero(boundary_band)),
        "bandRadius": band_radius,
        "lCandidateCount": int(len(l_masks)),
        "baselineComponents": substantial_baseline_components,
        "rejections": rejection_counts,
    }
    if best is None:
        return None, False, {"status": "rejected:no_safe_boundary_candidate", **audit_base}

    output = np.zeros_like(reference, dtype=np.float32)
    output[best] = 1.0
    return output, True, {
        "status": "accepted:bounded_l_boundary",
        "score": round(float(best_score), 4),
        **audit_base,
        **best_debug,
    }


def build_b_boundary_refine_prompt_inputs(reference_mask, target_bbox, band_ratio=0.018):
    """Anchor an L boundary review inside B's verified subject core.

    These are intentionally positive core points only. Boundary negatives can
    land on a valid appendage touching the semantic bbox and make L erase the
    very silhouette B has already established. The later boundary-band audit
    supplies the exclusion constraint deterministically.
    """
    binary = np.asarray(reference_mask > 0.5, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        return None, None, {"status": "skipped:no_reference"}
    height, width = binary.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1 = clamp(x1, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    x2 = clamp(x2, x1 + 1, width)
    y2 = clamp(y2, y1 + 1, height)
    constrained = np.zeros_like(binary, dtype=bool)
    constrained[y1:y2, x1:x2] = True
    binary &= constrained
    min_side = max(1, min(x2 - x1, y2 - y1))
    radius = max(2, min(12, int(round(min_side * float(band_ratio)))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1)
    )
    core = cv2.erode(binary.astype(np.uint8), kernel, iterations=1) > 0
    if int(np.count_nonzero(core)) < 48:
        core = binary
    distances = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    points = []
    separation = max(8, min(32, int(round(min_side * 0.12))))
    for _ in range(4):
        _, maximum, _, location = cv2.minMaxLoc(distances)
        if maximum < 2.0:
            break
        px, py = location
        points.append([int(px), int(py)])
        cv2.circle(distances, (int(px), int(py)), separation, 0, thickness=-1)
    if not points:
        return None, None, {
            "status": "skipped:no_core_points",
            "bandRadius": radius,
        }
    return [points], [[1] * len(points)], {
        "status": "ready",
        "points": len(points),
        "bandRadius": radius,
        "corePixels": int(np.count_nonzero(core)),
    }


def build_positive_probe_inputs(reference_mask, target_bbox):
    """Build a few interior positive points from the B baseline silhouette."""
    binary = np.asarray(reference_mask > 0.5, dtype=np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return None, None

    points = build_positive_points_from_mask(binary > 0, target_bbox) or []
    # Distance-transform maxima are safer than fixed bbox fractions: they land
    # in the known subject core even when the semantic bbox contains poster
    # background. Add at most two spatially separated core points.
    distances = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    for _ in range(2):
        _, maximum, _, maximum_location = cv2.minMaxLoc(distances)
        if maximum < 2.0:
            break
        px, py = maximum_location
        points.append([int(px), int(py)])
        cv2.circle(
            distances,
            (int(px), int(py)),
            max(4, int(round(maximum * 0.65))),
            0,
            thickness=-1
        )

    # The baseline points alone cannot recover an omitted appendage: they all
    # lie inside the already selected mask. Add at most two probes in a narrow
    # ring just outside the baseline, but only inside the semantic bbox. The
    # acceptance gate below decides whether these points produced a real,
    # attached subject extension; unsafe probes are discarded wholesale.
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target_region = np.zeros_like(binary, dtype=np.uint8)
    target_region[y1:y2, x1:x2] = 1
    ring = (
        cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            iterations=1
        ) & (~binary) & target_region
    )
    ring_distances = cv2.distanceTransform(ring, cv2.DIST_L2, 5)
    for _ in range(2):
        _, maximum, _, maximum_location = cv2.minMaxLoc(ring_distances)
        if maximum < 2.0:
            break
        px, py = maximum_location
        points.append([int(px), int(py)])
        cv2.circle(
            ring_distances,
            (int(px), int(py)),
            max(8, int(round(maximum * 1.8))),
            0,
            thickness=-1
        )

    deduped = []
    seen = set()
    for point in points:
        key = (int(point[0]), int(point[1]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([key[0], key[1]])
    if not deduped:
        return None, None
    return [deduped], [[1] * len(deduped)]


def filter_probe_growth_by_image(image, baseline, proposed, core, prompt_bbox):
    """Use GrabCut only to review proposed additions, never to erase baseline."""
    x1, y1, x2, y2 = prompt_bbox
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    base_crop = baseline[y1:y2, x1:x2]
    proposal_crop = proposed[y1:y2, x1:x2]
    core_crop = core[y1:y2, x1:x2]
    allowed = base_crop | proposal_crop
    if crop.size == 0 or np.count_nonzero(core_crop) < 5 or np.count_nonzero(~allowed) < 5:
        return None, {"status": "rejected", "reason": "insufficient_image_seeds"}
    labels = np.full(base_crop.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[allowed] = cv2.GC_PR_FGD
    labels[core_crop] = cv2.GC_FGD
    try:
        cv2.grabCut(
            crop, labels, None, np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64), 2, cv2.GC_INIT_WITH_MASK
        )
    except cv2.error as error:
        return None, {"status": "rejected", "reason": "image_review_failed", "error": str(error)}
    additions = np.zeros_like(baseline)
    additions[y1:y2, x1:x2] = (
        proposal_crop & ~base_crop &
        ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD))
    )
    # Color filtering can sever a previously attached extension. Retain only
    # components that still reach the established silhouette after filtering.
    count, components = cv2.connectedComponents((baseline | additions).astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(components[baseline])] = True
    keep[0] = False
    additions &= keep[components]
    return additions, {
        "status": "reviewed",
        "proposedPixels": int(np.count_nonzero(proposed & ~baseline)),
        "retainedPixels": int(np.count_nonzero(additions))
    }


def positive_probe_is_safe(
    baseline_mask,
    probe_mask,
    target_bbox,
    prompt_bbox=None,
    exclude_mask=None,
    min_preserve=0.98,
    min_fill_gain=0.025,
    min_growth_gain=0.025,
    max_growth=1.45,
    min_connected_growth=0.35,
    max_outside_added_ratio=0.55,
    max_context_conflict_ratio=0.12,
    image=None
):
    """Accept bounded growth that belongs to a component preserving the core."""
    base = np.asarray(baseline_mask > 0.5, dtype=bool)
    probe = np.asarray(probe_mask > 0.5, dtype=bool)
    if base.shape != probe.shape or not np.any(base) or not np.any(probe):
        return False, {"reason": "invalid_masks"}, None
    height, width = base.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1 = clamp(x1, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    x2 = clamp(x2, x1 + 1, width)
    y2 = clamp(y2, y1 + 1, height)
    target = np.zeros_like(base)
    target[y1:y2, x1:x2] = True

    if prompt_bbox is not None:
        px1, py1, px2, py2 = [int(value) for value in prompt_bbox]
        px1 = clamp(px1, 0, width - 1)
        py1 = clamp(py1, 0, height - 1)
        px2 = clamp(px2, px1 + 1, width)
        py2 = clamp(py2, py1 + 1, height)
        prompt_region = np.zeros_like(base)
        prompt_region[py1:py2, px1:px2] = True
        probe &= prompt_region
    else:
        px1, py1, px2, py2 = 0, 0, width, height

    base_area = max(1, int(np.count_nonzero(base)))
    base_core = cv2.erode(
        base.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    ) > 0
    if not np.any(base_core):
        base_core = base

    component_count, labels, _, _ = cv2.connectedComponentsWithStats(
        probe.astype(np.uint8), connectivity=8
    )
    owned_probe = np.zeros_like(probe)
    min_core_overlap = max(8, int(round(base_area * 0.002)))
    for label in range(1, component_count):
        component = labels == label
        core_overlap = int(np.count_nonzero(component & base_core))
        if core_overlap >= min_core_overlap:
            owned_probe |= component

    if not np.any(owned_probe):
        return False, {
            "reason": "no_core_connected_component",
            "probeComponents": int(max(0, component_count - 1))
        }, None

    raw_added = probe & ~base
    connected_added = owned_probe & ~base
    raw_added_area = int(np.count_nonzero(raw_added))
    connected_added_area = int(np.count_nonzero(connected_added))
    connected_growth = connected_added_area / max(1, raw_added_area)
    # The established B mask is retained verbatim. The probe contributes only
    # growth from components that overlap the eroded subject core.
    selected = base | connected_added
    selected_area = int(np.count_nonzero(selected))
    preserve = int(np.count_nonzero(base & owned_probe)) / base_area
    core_preserve = int(np.count_nonzero(base_core & owned_probe)) / max(1, int(np.count_nonzero(base_core)))
    base_fill = int(np.count_nonzero(base & target)) / max(1, int(np.count_nonzero(target)))
    probe_fill = int(np.count_nonzero(selected & target)) / max(1, int(np.count_nonzero(target)))
    if connected_added_area <= 0:
        failed_gates = []
        if core_preserve < min_preserve:
            failed_gates.append("preserve_core")
        if not failed_gates:
            failed_gates.append("no_added_pixels")
        return False, {
            "reason": "+".join(failed_gates),
            "failedGates": failed_gates,
            "preserve": round(preserve, 4),
            "preserveCore": round(core_preserve, 4),
            "baseFill": round(base_fill, 4),
            "probeFill": round(probe_fill, 4)
        }, None
    probe_growth = selected_area / base_area
    growth_gain = connected_added_area / base_area
    outside_added = int(np.count_nonzero(connected_added & ~target))
    outside_ratio = outside_added / connected_added_area
    touch_band = max(2, min(8, int(round(min(x2 - x1, y2 - y1) * 0.015))))
    boundary_continuation = bool(
        (np.any(base[y1:y1 + touch_band, x1:x2]) and np.any(connected_added[:y1, :])) or
        (np.any(base[y2 - touch_band:y2, x1:x2]) and np.any(connected_added[y2:, :])) or
        (np.any(base[y1:y2, x1:x1 + touch_band]) and np.any(connected_added[:, :x1])) or
        (np.any(base[y1:y2, x2 - touch_band:x2]) and np.any(connected_added[:, x2:]))
    )
    context_conflict_ratio = (
        int(np.count_nonzero(connected_added & np.asarray(exclude_mask, dtype=bool))) /
        connected_added_area
        if exclude_mask is not None else 0.0
    )
    evidence_gain = bool(
        probe_fill >= base_fill + min_fill_gain or
        (outside_added > 0 and growth_gain >= min_growth_gain)
    )
    failed_gates = []
    if core_preserve < min_preserve:
        failed_gates.append("preserve_core")
    if not evidence_gain:
        failed_gates.append("insufficient_growth")
    if probe_growth > max_growth:
        failed_gates.append("growth_limit")
    if connected_growth < min_connected_growth:
        failed_gates.append("disconnected_growth")
    if outside_ratio > max_outside_added_ratio and not boundary_continuation:
        failed_gates.append("outside_without_boundary_continuation")
    if context_conflict_ratio > max_context_conflict_ratio:
        failed_gates.append("context_conflict")
    safe = not failed_gates
    # Existing out-of-box speckles are not evidence for enlarging the output.
    # Only newly accepted growth can extend the semantic bbox.
    added_bbox = mask_bbox(connected_added)
    effective_bbox = [x1, y1, x2, y2]
    if added_bbox:
        effective_bbox = [
            max(px1, min(x1, int(added_bbox[0]))),
            max(py1, min(y1, int(added_bbox[1]))),
            min(px2, max(x2, int(added_bbox[2]))),
            min(py2, max(y2, int(added_bbox[3])))
        ]
    audit = {
        "reason": "accepted" if safe else "+".join(failed_gates),
        "failedGates": failed_gates,
        "preserve": round(preserve, 4),
        "preserveCore": round(core_preserve, 4),
        "baseFill": round(base_fill, 4),
        "probeFill": round(probe_fill, 4),
        "fillGain": round(probe_fill - base_fill, 4),
        "addedPixels": connected_added_area,
        "rawAddedPixels": raw_added_area,
        "connectedGrowth": round(connected_growth, 4),
        "growth": round(probe_growth, 4),
        "growthGain": round(growth_gain, 4),
        "outsideAddedRatio": round(outside_ratio, 4),
        "boundaryContinuation": boundary_continuation,
        "contextConflictRatio": round(context_conflict_ratio, 4),
        "effectiveBbox": effective_bbox
    }
    # A SAM probe is intentionally broad.  Its raw growth and ownership
    # overlap are only hypotheses until the image review removes obvious
    # background and detached pixels.  Defer those candidate-dependent gates
    # to the filtered union, while retaining hard stops for an entirely owned
    # addition or an extreme envelope that cannot be safely reviewed.
    hard_review_rejection = bool(
        context_conflict_ratio >= 0.999 or
        probe_growth > (max_growth * 1.25)
    )
    if image is not None and connected_added_area > 0 and not hard_review_rejection:
        additions, image_review = filter_probe_growth_by_image(
            image, base, owned_probe, base_core, [px1, py1, px2, py2]
        )
        audit["imageReview"] = image_review
        if additions is None:
            audit.update(reason=image_review["reason"], failedGates=[image_review["reason"]])
            return False, audit, None
        # Audit the filtered union again.  In particular, semantic ownership
        # is measured here, after image review, rather than against the broad
        # raw SAM candidate.  This keeps unrelated text/panels from vetoing a
        # valid subject extension while still rejecting retained conflicts.
        safe, filtered_audit, filtered = positive_probe_is_safe(
            baseline_mask, (base | additions).astype(np.float32), target_bbox,
            prompt_bbox=prompt_bbox, exclude_mask=exclude_mask,
            min_preserve=min_preserve, min_fill_gain=min_fill_gain,
            min_growth_gain=min_growth_gain, max_growth=max_growth,
            # Image review has already removed detached islands and the
            # baseline is immutable; do not reject a valid boundary extension
            # merely because it does not overlap the eroded core.
            min_connected_growth=0.0,
            max_outside_added_ratio=max_outside_added_ratio,
            max_context_conflict_ratio=max_context_conflict_ratio
        )
        filtered_audit.update(
            preserve=audit["preserve"], preserveCore=audit["preserveCore"],
            imageReview=image_review, rawAddedPixels=raw_added_area,
            connectedGrowth=audit["connectedGrowth"],
            rawContextConflictRatio=audit["contextConflictRatio"],
            rawFailedGates=failed_gates
        )
        return safe, filtered_audit, filtered
    return safe, audit, selected.astype(np.float32) if safe else None


def run_positive_probe(
    img,
    baseline_masks,
    target_bbox,
    prompt_bbox,
    layer_name,
    strategy_type=None,
    layer_meta=None,
    context_layers=None,
    quality_profile="publish"
):
    """Probe B with interior positives and retain it only on strict evidence."""
    policy = resolve_mask_policy(layer_meta or {}, quality_profile)
    config = policy.get("positiveProbe") or {}
    baseline_reference = choose_b_reference_mask(
        baseline_masks,
        target_bbox,
        strategy_type=strategy_type
    )
    if strategy_type == "furniture" and (
        ((layer_meta or {}).get("segmentationIntent") or {}).get("excludedAdjacentObjects")
    ):
        layer_meta["_probeBaselineSupport"] = np.any(
            np.asarray(baseline_masks) > 0.5, axis=0
        )
    if baseline_reference is None:
        return baseline_masks, {"status": "skipped", "reason": "no_baseline_reference"}
    intent_prompt = build_spatial_intent_prompt_inputs(
        layer_meta or {}, target_bbox, prompt_bbox, img.shape[1], img.shape[0]
    )
    if intent_prompt.get("enabled"):
        # A generic probe can inherit a neighbouring object from a contaminated
        # baseline. Multi-component targets instead probe from declared cores.
        points = [list(intent_prompt["positive"])]
        labels = [[1] * len(points[0])]
    else:
        points, labels = build_positive_probe_inputs(baseline_reference, target_bbox)
    if not points:
        return baseline_masks, {"status": "skipped", "reason": "no_positive_points"}
    if intent_prompt.get("enabled") and intent_prompt.get("negative"):
        points[0].extend(intent_prompt["negative"])
        labels[0].extend([0] * len(intent_prompt["negative"]))
    probe_results = run_sam_bbox_inference(
        img,
        prompt_bbox,
        multimask_output=True,
        imgsz=policy["samImgSize"],
        points=points,
        labels=labels,
        model_variant="b"
    )
    try:
        probe_masks = normalize_result_masks(
            probe_results,
            img.shape[1],
            img.shape[0],
            interpolation=cv2.INTER_LINEAR if strategy_type in HARD_EDGE_STRATEGIES else cv2.INTER_NEAREST,
            debug_label=f"{layer_name} strategy={strategy_type} positive_probe"
        )
    finally:
        del probe_results
    sam_diagnostic_event(
        "probe_candidates", masks=probe_masks,
        targetBbox=target_bbox, promptBbox=prompt_bbox, points=points, labels=labels,
        segmentationIntent=intent_prompt
    )
    if probe_masks is None or len(probe_masks) == 0:
        return baseline_masks, {
            "status": "rejected",
            "reason": "no_probe_candidates",
            "probeCandidates": 0
        }

    semantic_excludes = build_semantic_ownership_bboxes(
        layer_meta or {}, context_layers or [], prompt_bbox,
        img.shape[1], img.shape[0]
    )
    exclude_mask = (
        build_exclude_mask(semantic_excludes, img.shape[1], img.shape[0])
        if semantic_excludes else None
    )
    accepted_candidates = []
    rejected = {}
    candidate_audits = []
    sam_diagnostic_event(
        "probe_reference_and_context", masks=[baseline_reference],
        excludeBboxes=semantic_excludes, thresholds=config,
        segmentationIntent=intent_prompt
    )
    for index, probe_candidate in enumerate(probe_masks):
        accepted, candidate_audit, selected_mask = positive_probe_is_safe(
            baseline_reference,
            probe_candidate,
            target_bbox,
            prompt_bbox=prompt_bbox,
            exclude_mask=exclude_mask,
            min_preserve=float(config.get("minPreserveCore", 0.98)),
            min_fill_gain=float(config.get("minFillGain", 0.025)),
            min_growth_gain=float(config.get("minGrowthGain", 0.025)),
            max_growth=float(config.get("maxGrowth", 1.45)),
            min_connected_growth=float(config.get("minConnectedGrowth", 0.35)),
            max_outside_added_ratio=float(config.get("maxOutsideAddedRatio", 0.55)),
            max_context_conflict_ratio=float(config.get("maxContextConflictRatio", 0.12)),
            image=img
        )
        candidate_audit["index"] = int(index)
        sam_diagnostic_event("probe_candidate_audit", audit=candidate_audit)
        candidate_audits.append(candidate_audit)
        if not accepted:
            reason = candidate_audit.get("reason", "rejected")
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        fill_gain = float(candidate_audit.get("fillGain", 0.0))
        growth_gain = float(candidate_audit.get("growthGain", 0.0))
        growth = float(candidate_audit.get("growth", 1.0))
        conflict = float(candidate_audit.get("contextConflictRatio", 0.0))
        score = (
            min(1.0, max(fill_gain, growth_gain) / 0.16) * 0.50 +
            float(candidate_audit.get("preserve", 0.0)) * 0.25 +
            float(candidate_audit.get("connectedGrowth", 0.0)) * 0.20 -
            max(0.0, growth - 1.0) * 0.08 -
            conflict * 0.25
        )
        accepted_candidates.append((score, index, selected_mask, candidate_audit))

    if not accepted_candidates:
        best_audit = max(
            candidate_audits,
            key=lambda row: (
                float(row.get("preserve", 0.0)),
                float(row.get("connectedGrowth", 0.0)),
                float(row.get("fillGain", 0.0))
            ),
            default={}
        )
        return baseline_masks, {
            **best_audit,
            "status": "rejected",
            "reason": "no_safe_probe_candidate",
            "probeCandidates": int(len(probe_masks)),
            "positivePoints": int(len(points[0])),
            "rejectionCounts": rejected
        }

    accepted_candidates.sort(key=lambda row: row[0], reverse=True)
    score, index, selected_mask, audit = accepted_candidates[0]
    audit.update({
        "status": "accepted",
        "reason": "connected_growth_accepted",
        "score": round(float(score), 4),
        "selectedIndex": int(index),
        "probeCandidates": int(len(probe_masks)),
        "positivePoints": int(len(points[0])),
        "rejectionCounts": rejected
    })
    # Keep the original B proposals beside the accepted probe proposal. The
    # September 16 path relied on the narrow original candidate when a probe
    # grew into a neighbouring instance or a scene envelope. The probe is
    # additional evidence, not permission to discard that baseline. The
    # normal candidate selector is responsible for choosing between them.
    baseline_array = np.asarray(baseline_masks, dtype=np.float32)
    selected_array = np.asarray(selected_mask, dtype=np.float32)
    if selected_array.ndim == 2:
        selected_array = selected_array[None, ...]
    combined = [np.asarray(mask, dtype=np.float32) for mask in baseline_array]
    verified_candidate_indexes = []
    for probe_mask in selected_array:
        probe_binary = probe_mask > 0.5
        if not np.any(probe_binary):
            continue
        duplicate = False
        for existing in combined:
            existing_binary = existing > 0.5
            overlap = int(np.count_nonzero(probe_binary & existing_binary))
            union = int(np.count_nonzero(probe_binary | existing_binary))
            if union > 0 and overlap / union >= 0.96:
                duplicate = True
                break
        if not duplicate:
            # Keep provenance for the exact candidate that passed all probe
            # safety gates; this is not a blanket preference for expansions.
            verified_candidate_indexes.append(int(len(combined)))
            combined.append(probe_mask)
    audit["preservedBaselineCandidates"] = int(len(baseline_array))
    audit["probeCandidateAppended"] = bool(len(combined) > len(baseline_array))
    audit["verifiedCandidateIndexes"] = verified_candidate_indexes
    return np.stack(combined, axis=0), audit


def audit_compound_component_discovery(
    core_mask,
    candidate_mask,
    target_bbox,
    prompt_bbox,
    config=None,
    exclude_mask=None,
    strict_exclude_mask=None
):
    """Keep package components while rejecting thin background envelopes.

    The core is immutable.  Only candidate components that preserve that core
    and have bounded, object-like support are added.  This is deliberately
    shape/evidence based; it never treats white or low-chroma pixels as
    background, which is important for plates and bowls.
    """
    config = config or {}
    core = np.asarray(core_mask > 0.5, dtype=bool)
    candidate = np.asarray(candidate_mask > 0.5, dtype=bool)
    if core.shape != candidate.shape or not np.any(core) or not np.any(candidate):
        return core_mask, {"status": "rejected", "reason": "invalid_masks"}
    height, width = core.shape
    tx1, ty1, tx2, ty2 = [int(value) for value in target_bbox]
    px1, py1, px2, py2 = [int(value) for value in prompt_bbox]
    px1, py1 = clamp(px1, 0, width - 1), clamp(py1, 0, height - 1)
    px2, py2 = clamp(px2, px1 + 1, width), clamp(py2, py1 + 1, height)
    prompt_region = np.zeros_like(core)
    prompt_region[py1:py2, px1:px2] = True
    candidate &= prompt_region
    core_area = max(1, int(np.count_nonzero(core)))
    preserve = int(np.count_nonzero(core & candidate)) / core_area
    if preserve < float(config.get("minCorePreserve", 0.97)):
        return core_mask, {
            "status": "rejected",
            "reason": "preserve_core",
            "preserveCore": round(preserve, 4)
        }

    raw_added = candidate & ~core
    raw_added_area = int(np.count_nonzero(raw_added))
    strict_mask = None
    strict_edge_override = np.zeros_like(core, dtype=bool)
    strict_excluded_area = 0
    strict_edge_audit = {"status": "skipped", "reason": "no_strict_ownership"}
    if strict_exclude_mask is not None:
        strict_mask = np.asarray(strict_exclude_mask, dtype=bool)
        if strict_mask.shape == raw_added.shape:
            strict_edge_override, strict_edge_audit = allow_thin_strict_edge_continuations(
                core,
                raw_added,
                strict_mask,
                target_bbox,
                config=config.get("strictEdgeContinuation") if isinstance(config, dict) else None,
            )
            effective_strict_mask = strict_mask & ~strict_edge_override
            strict_excluded_area = int(np.count_nonzero(raw_added & effective_strict_mask))
            # Explicitly owned pixels are forbidden, but their presence must
            # not veto an otherwise valid connected extension in its entirety.
            # Clip owned pixels except a narrowly bounded contour run that
            # directly continues the selected target silhouette at its edge.
            raw_added &= ~effective_strict_mask
            strict_mask = effective_strict_mask
    added = raw_added
    added_area = int(np.count_nonzero(added))
    if added_area <= 0:
        return core_mask, {
            "status": "rejected",
            "reason": "strict_ownership_excluded" if strict_excluded_area else "no_new_components",
            "preserveCore": round(preserve, 4),
            "candidateAddedPixels": raw_added_area,
            "strictOwnershipRemovedPixels": strict_excluded_area,
            "strictEdgeContinuation": strict_edge_audit,
        }

    target = np.zeros_like(core)
    target[max(0, ty1):min(height, ty2), max(0, tx1):min(width, tx2)] = True
    distance_to_core = cv2.distanceTransform(
        (~core).astype(np.uint8), cv2.DIST_L2, 5
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        added.astype(np.uint8), connectivity=8
    )
    min_area = max(
        32,
        int(bbox_area(target_bbox) * float(config.get("minComponentAreaRatio", 0.0025)))
    )
    kept_added = np.zeros_like(core)
    rows = []
    for component_id in range(1, component_count):
        component = labels == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        strict_edge_overlap = (
            int(np.count_nonzero(component & strict_edge_override)) / max(1, area)
            if strict_exclude_mask is not None else 0.0
        )
        is_verified_thin_edge = strict_edge_overlap >= 0.80
        if area < min_area and not is_verified_thin_edge:
            rows.append({"component": component_id, "area": area, "status": "ignored", "reason": "small"})
            continue
        x, y, comp_w, comp_h, _ = [int(value) for value in stats[component_id]]
        component_bbox = [x, y, x + comp_w, y + comp_h]
        fill = area / max(1, comp_w * comp_h)
        distance = float(distance_to_core[component].min())
        inside_target = int(np.count_nonzero(component & target)) / max(1, area)
        context_conflict = (
            int(np.count_nonzero(component & np.asarray(exclude_mask, dtype=bool))) / area
            if exclude_mask is not None else 0.0
        )
        # Added pixels have already been clipped to strict ownership bounds.
        # Keep the ratio as an audit invariant for malformed/mismatched masks.
        strict_conflict = (
            int(np.count_nonzero(component & strict_mask)) / area
            if strict_mask is not None and strict_mask.shape == component.shape else 0.0
        )
        prompt_edges = int(
            x <= px1 + 1 or y <= py1 + 1 or
            x + comp_w >= px2 - 1 or y + comp_h >= py2 - 1
        )
        target_edge_margin = max(
            2,
            int(round(min(tx2 - tx1, ty2 - ty1) * float(config.get("targetEdgeMarginRatio", 0.05))))
        )
        target_edges = int(
            x <= tx1 + target_edge_margin or y <= ty1 + target_edge_margin or
            x + comp_w >= tx2 - target_edge_margin or
            y + comp_h >= ty2 - target_edge_margin
        )
        low_fill = fill < float(config.get("maxEnvelopeFill", 0.40))
        small_context_component = bool(
            not is_verified_thin_edge and
            context_conflict > float(config.get("maxContextConflictRatio", 0.22)) and
            area <= core_area * float(config.get("maxContextComponentAreaRatio", 0.02))
        )
        fully_owned_by_target = inside_target >= float(
            config.get("minContainedEdgeEnvelopeTargetRatio", 0.95)
        )
        edge_envelope = bool(
            low_fill and target_edges > 0 and (
                (x + comp_w >= tx2 - 1 and comp_h >= (ty2 - ty1) * float(config.get("minEdgeEnvelopeHeightRatio", 0.35))) or
                (y + comp_h >= ty2 - 1 and comp_w >= (tx2 - tx1) * float(config.get("minEdgeEnvelopeWidthRatio", 0.35)))
            ) and not fully_owned_by_target
        )
        detached = distance > float(config.get("maxDetachedDistance", 18))
        rejected_reason = None
        if strict_conflict > float(config.get("maxStrictOwnershipConflictRatio", 0.18)):
            rejected_reason = "strict_semantic_ownership"
        elif edge_envelope:
            rejected_reason = "background_edge_envelope"
        elif small_context_component:
            rejected_reason = "context_conflict_small_component"
        elif (
            not is_verified_thin_edge and
            context_conflict > float(config.get("maxContextConflictRatio", 0.22)) and
            low_fill and
            (prompt_edges > 0 or target_edges > 0)
        ):
            rejected_reason = "context_conflict"
        elif detached:
            rejected_reason = "detached_without_support"
        elif prompt_edges > 0 and fill < float(config.get("maxPromptEdgeFill", 0.55)):
            rejected_reason = "prompt_edge_envelope"
        if rejected_reason:
            rows.append({
                "component": component_id,
                "area": area,
                "bbox": component_bbox,
                "fill": round(fill, 4),
                "distanceToCore": round(distance, 2),
                "insideTarget": round(inside_target, 4),
                "contextConflict": round(context_conflict, 4),
                "strictOwnershipConflict": round(strict_conflict, 4),
                "status": "rejected",
                "reason": rejected_reason
            })
            continue
        kept_added |= component
        rows.append({
            "component": component_id,
            "area": area,
            "bbox": component_bbox,
            "fill": round(fill, 4),
            "distanceToCore": round(distance, 2),
            "insideTarget": round(inside_target, 4),
            "contextConflict": round(context_conflict, 4),
            "strictOwnershipConflict": round(strict_conflict, 4),
            "status": "kept",
            "reason": (
                "strict_edge_continuation"
                if is_verified_thin_edge else "bounded_package_component"
            )
        })

    kept_area = int(np.count_nonzero(kept_added))
    if kept_area <= 0:
        return core_mask, {
            "status": "rejected",
            "reason": "no_verified_components",
            "preserveCore": round(preserve, 4),
            "candidateAddedPixels": raw_added_area,
            "effectiveCandidateAddedPixels": added_area,
            "strictOwnershipRemovedPixels": strict_excluded_area,
            "strictEdgeContinuation": strict_edge_audit,
            "components": rows
        }
    if kept_area > int(core_area * float(config.get("maxGrowthRatio", 0.75))):
        return core_mask, {
            "status": "rejected",
            "reason": "growth_limit",
            "preserveCore": round(preserve, 4),
            "candidateAddedPixels": raw_added_area,
            "effectiveCandidateAddedPixels": added_area,
            "strictOwnershipRemovedPixels": strict_excluded_area,
            "verifiedAddedPixels": kept_area,
            "strictEdgeContinuation": strict_edge_audit,
            "components": rows
        }
    merged = np.maximum(core_mask, kept_added.astype(np.float32)).astype(np.float32)
    return merged, {
        "status": "accepted",
        "reason": "verified_package_components",
        "preserveCore": round(preserve, 4),
        "candidateAddedPixels": raw_added_area,
        "effectiveCandidateAddedPixels": added_area,
        "strictOwnershipRemovedPixels": strict_excluded_area,
        "verifiedAddedPixels": kept_area,
        "growth": round((core_area + kept_area) / core_area, 4),
        "strictEdgeContinuation": strict_edge_audit,
        "components": rows
    }


def audit_interior_holes(mask, target_bbox, config=None, strict_exclude_mask=None):
    """Find substantial enclosed mask holes worth an evidence-gated review."""
    config = config or {}
    binary = np.asarray(mask > 0.5, dtype=bool)
    height, width = binary.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1, y1 = clamp(x1, 0, width - 1), clamp(y1, 0, height - 1)
    x2, y2 = clamp(x2, x1 + 1, width), clamp(y2, y1 + 1, height)
    crop = binary[y1:y2, x1:x2]
    subject_area = int(np.count_nonzero(crop))
    if subject_area < 256:
        return {"status": "skipped", "reason": "small_subject", "holes": []}

    inverted = (~crop).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    min_hole = max(
        int(config.get("minHolePixels", 64)),
        int(round(subject_area * float(config.get("minHoleRatio", 0.008))))
    )
    min_component = max(
        4, int(config.get("minHoleComponentPixels", 8))
    )
    strict_exclude_mask = (
        np.asarray(strict_exclude_mask, dtype=bool)
        if strict_exclude_mask is not None else None
    )
    holes = []
    for label in range(1, count):
        hx, hy, hw, hh, area = [int(value) for value in stats[label]]
        if (
            hx <= 0 or hy <= 0 or
            hx + hw >= crop.shape[1] or hy + hh >= crop.shape[0] or
            area < min_component
        ):
            continue
        component = labels == label
        ring = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1
        ) > 0
        ring &= ~component
        ring_support = int(np.count_nonzero(ring & crop)) / max(1, int(np.count_nonzero(ring)))
        if ring_support < float(config.get("minRingSupport", 0.72)):
            continue
        full_component = np.zeros_like(binary)
        full_component[y1:y2, x1:x2] = component
        strict_conflict = (
            int(np.count_nonzero(full_component & strict_exclude_mask)) / area
            if strict_exclude_mask is not None else 0.0
        )
        if strict_conflict > float(config.get("maxStrictOwnershipRatio", 0.18)):
            continue
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        _, _, _, location = cv2.minMaxLoc(distance)
        px, py = int(location[0]) + x1, int(location[1]) + y1
        holes.append({
            "label": int(label),
            "area": int(area),
            "bbox": [x1 + hx, y1 + hy, x1 + hx + hw, y1 + hy + hh],
            "ringSupport": round(ring_support, 4),
            "strictOwnership": round(strict_conflict, 4),
            "point": [px, py],
            "mask": full_component
        })
    holes.sort(key=lambda row: row["area"], reverse=True)
    holes = holes[:int(config.get("maxHoleCount", 24))]
    total_hole_pixels = sum(int(row["area"]) for row in holes)
    if not holes or total_hole_pixels < min_hole:
        return {
            "status": "skipped",
            "reason": "no_substantial_enclosed_holes",
            "subjectPixels": subject_area,
            "holes": []
        }
    return {
        "status": "triggered",
        "reason": "substantial_enclosed_holes",
        "subjectPixels": subject_area,
        "holePixels": int(total_hole_pixels),
        "holeRatio": round(total_hole_pixels / max(1, subject_area), 4),
        "holes": holes
    }


def recover_interior_holes_with_sam(
    img, mask, target_bbox, layer_name, layer_meta=None,
    context_layers=None, policy=None
):
    """Repair only holes that a same-embedding SAM point review verifies."""
    policy = policy or resolve_mask_policy(layer_meta or {}, "completion")
    config = policy.get("interiorHoleRecovery") or {}
    if not config.get("enabled"):
        return mask, False, {"status": "skipped", "reason": "capability_disabled"}
    width, height = img.shape[1], img.shape[0]
    prompt_bbox = expand_bbox(*target_bbox, width, height, ratio=0.12)
    semantic_excludes = build_semantic_ownership_bboxes(
        layer_meta or {}, context_layers or [], target_bbox, width, height
    )
    strict_excludes = [
        entry for entry in semantic_excludes
        if isinstance(entry, dict) and entry.get("strict_ownership")
    ]
    strict_mask = build_exclude_mask(strict_excludes, width, height) if strict_excludes else None
    audit = audit_interior_holes(mask, target_bbox, config=config, strict_exclude_mask=strict_mask)
    if audit.get("status") != "triggered":
        return mask, False, audit
    holes = audit.get("holes") or []
    # Hole masks are needed internally to score candidate additions, but must
    # not be placed into request metadata or diagnostics as giant JSON arrays.
    public_audit = {
        **audit,
        "holes": [
            {key: value for key, value in row.items() if key != "mask"}
            for row in holes
        ]
    }
    points = [row["point"] for row in holes[:int(config.get("maxProbePoints", 4))]]
    try:
        results = run_sam_bbox_inference(
            img,
            prompt_bbox,
            multimask_output=True,
            imgsz=policy.get("samImgSize", 1024),
            points=[points],
            labels=[[1] * len(points)],
            model_variant="b"
        )
        candidates = normalize_result_masks(
            results, width, height, interpolation=cv2.INTER_LINEAR,
            debug_label=f"{layer_name} interior_hole_recovery"
        )
    except Exception as error:
        return mask, False, {**public_audit, "status": "rejected", "reason": "sam_review_failed", "error": str(error)}
    finally:
        try:
            del results
        except UnboundLocalError:
            pass

    base = np.asarray(mask > 0.5, dtype=bool)
    base_area = max(1, int(np.count_nonzero(base)))
    hole_union = np.zeros_like(base)
    for row in holes:
        hole_union |= row["mask"]
    accepted = []
    audits = []
    for index, candidate in enumerate(candidates if candidates is not None else []):
        candidate_binary = np.asarray(candidate > 0.5, dtype=bool)
        preserve = int(np.count_nonzero(candidate_binary & base)) / base_area
        added = candidate_binary & ~base
        hole_added = added & hole_union
        outside_added = added & ~hole_union
        hole_fill = int(np.count_nonzero(hole_added)) / max(1, int(np.count_nonzero(hole_union)))
        outside_ratio = int(np.count_nonzero(outside_added)) / max(1, int(np.count_nonzero(added)))
        added_ratio = int(np.count_nonzero(added)) / base_area
        strict_conflict = (
            int(np.count_nonzero(added & strict_mask)) / max(1, int(np.count_nonzero(added)))
            if strict_mask is not None else 0.0
        )
        hole_strict_conflict = (
            int(np.count_nonzero(hole_added & strict_mask)) /
            max(1, int(np.count_nonzero(hole_added)))
            if strict_mask is not None else 0.0
        )
        # The baseline mask is authoritative. A positive-point candidate can
        # legitimately omit parts of that baseline while proving an interior
        # addition, so evaluate clipped recovery against the baseline union
        # instead of treating those omitted baseline pixels as destruction.
        candidate_core_overlap = (
            int(np.count_nonzero(candidate_binary & base)) / base_area
        )
        raw_safe = (
            preserve >= float(config.get("minPreserveCore", 0.98)) and
            hole_fill >= float(config.get("minHoleFillRatio", 0.25)) and
            added_ratio <= float(config.get("maxAddedRatio", 0.16)) and
            outside_ratio <= float(config.get("maxOutsideHoleRatio", 0.18)) and
            strict_conflict <= float(config.get("maxStrictOwnershipRatio", 0.18))
        )
        # A point-prompt mask is used as evidence, not as the final geometry:
        # SAM can return the whole surrounding subject envelope even when the
        # point proves that an enclosed hole belongs to the subject. If the
        # candidate preserves the established core, fills nearly all audited
        # holes and has no meaningful strict-ownership conflict, retain only
        # its intersection with the audited hole union. This prevents the
        # broad candidate envelope from restoring background outside the hole.
        clipped_safe = (
            candidate_core_overlap >= float(config.get("minClippedCandidateCoreOverlap", 0.90)) and
            hole_fill >= float(config.get("minClippedHoleFillRatio", 0.90)) and
            added_ratio <= float(config.get("maxAddedRatio", 0.16)) and
            hole_strict_conflict <= float(config.get("maxStrictOwnershipRatio", 0.18))
        )
        safe = raw_safe or clipped_safe
        row = {
            "index": int(index),
            "preserveCore": round(preserve, 4),
            "holeFill": round(hole_fill, 4),
            "addedPixels": int(np.count_nonzero(added)),
            "addedRatio": round(added_ratio, 4),
            "outsideHoleRatio": round(outside_ratio, 4),
            "strictOwnership": round(strict_conflict, 4),
            "holeStrictOwnership": round(hole_strict_conflict, 4),
            "candidateCoreOverlap": round(candidate_core_overlap, 4),
            "status": "accepted" if safe else "rejected",
            "reason": (
                "verified_hole_fill" if raw_safe else
                "verified_hole_fill_clipped" if clipped_safe else
                "unsafe_hole_growth"
            ),
            "mergeMode": "candidate" if raw_safe else "audited_hole_intersection" if clipped_safe else None
        }
        audits.append(row)
        if safe:
            score = hole_fill * 0.55 + preserve * 0.25 - outside_ratio * 0.15 - added_ratio * 0.05
            accepted.append((score, candidate_binary, row))
    if not accepted:
        return mask, False, {
            **public_audit,
            "status": "rejected",
            "reason": "no_safe_hole_candidate",
            "points": points,
            "candidates": audits
        }
    accepted.sort(key=lambda row: row[0], reverse=True)
    _, selected, selected_audit = accepted[0]
    selected_added = selected & ~base
    if selected_audit.get("mergeMode") == "audited_hole_intersection":
        selected_added &= hole_union
    recovered = np.maximum(base, selected_added).astype(np.float32)
    selected_audit = {
        **selected_audit,
        "retainedAddedPixels": int(np.count_nonzero(selected_added)),
        "retainedAddedRatio": round(
            int(np.count_nonzero(selected_added)) / base_area, 4
        )
    }
    return recovered, True, {
        **public_audit,
        "status": "accepted",
        "reason": selected_audit.get("reason", "verified_hole_fill"),
        "points": points,
        "selected": selected_audit,
        "candidates": audits
    }


def supports_interior_hole_recovery(strategy_type, policy):
    """Return whether the generic enclosed-hole capability may run.

    ``hard_product`` is a hard-edge SAM route, but it is intentionally not
    part of ``HARD_EDGE_STRATEGIES`` because that set controls the older
    boundary/local-refine behavior. Reusing that set as the gate here made
    the new capability unreachable for food and other hard products.
    """
    config = (policy or {}).get("interiorHoleRecovery") or {}
    return bool(
        config.get("enabled") and
        strategy_type in HARD_EDGE_STRATEGIES.union({"hard_product"})
    )


def run_compound_component_discovery(
    img,
    candidate_masks,
    target_bbox,
    layer_name,
    layer_meta=None,
    context_layers=None,
    quality_profile="publish",
    policy=None
):
    """Discover detached package parts using the current request embedding."""
    policy = policy or resolve_mask_policy(layer_meta or {}, quality_profile)
    config = dict(policy.get("compoundComponentDiscovery") or {})
    if not config.get("enabled") or candidate_masks is None or len(candidate_masks) == 0:
        return candidate_masks, {"status": "skipped", "reason": "capability_disabled"}
    img_h, img_w = img.shape[:2]
    core_mask, _, core_quality = select_and_merge_masks(
        candidate_masks,
        target_bbox,
        img_w,
        img_h,
        layer_meta=layer_meta,
        context_layers=context_layers,
        quality_profile=quality_profile
    )
    if core_mask is None or not np.any(core_mask > 0.5):
        return candidate_masks, {"status": "skipped", "reason": "no_core_mask"}
    primary_index = None
    for row in (core_quality or {}).get("debugCandidates", []):
        if isinstance(row, dict) and row.get("rejectReason") == "primary":
            primary_index = int(row.get("index"))
            break
    if primary_index is None:
        selected_indexes = (core_quality or {}).get("selectedIndexes") or []
        if selected_indexes:
            primary_index = int(selected_indexes[0])
    if primary_index is not None and 0 <= primary_index < len(candidate_masks):
        core_mask = np.asarray(candidate_masks[primary_index], dtype=np.float32)
    prompt_bbox = expand_bbox(
        *target_bbox, img_w, img_h,
        ratio=float(config.get("promptExpandRatio", 0.36))
    )
    semantic_excludes = build_semantic_ownership_bboxes(
        layer_meta or {}, context_layers or [], prompt_bbox, img_w, img_h
    )
    exclude_mask = build_exclude_mask(semantic_excludes, img_w, img_h) if semantic_excludes else None
    strict_excludes = [
        entry for entry in semantic_excludes
        if isinstance(entry, dict) and entry.get("strict_ownership")
    ]
    layer_meta = layer_meta or {}
    strict_exclude_mask = (
        build_exclude_mask(strict_excludes, img_w, img_h)
        if strict_excludes else None
    )

    def finalize_compound_mask(mask, audit):
        recovered, edge_audit = recover_strict_bbox_edge_continuations(
            mask,
            candidate_masks,
            target_bbox,
            strict_exclude_mask=strict_exclude_mask,
            config=config.get("strictBboxEdgeContinuation"),
        )
        layer_meta["_samStrictBboxEdgeExtension"] = edge_audit
        audit["strictBboxEdgeExtension"] = edge_audit
        return np.stack([recovered], axis=0), audit

    baseline_mask = core_mask.copy()
    baseline_core_pixels = int(np.count_nonzero(baseline_mask > 0.5))
    audits = []
    rejected_existing_envelope = False
    for index, existing_candidate in enumerate(candidate_masks):
        if primary_index is not None and index == primary_index:
            continue
        merged_existing, existing_audit = audit_compound_component_discovery(
            baseline_mask,
            np.maximum(baseline_mask, existing_candidate).astype(np.float32),
            target_bbox,
            prompt_bbox,
            config=config,
            # Keep the established B-alternative behavior here. A broad
            # layout ownership box can overlap a real plate; component-level
            # context rejection would remove the entire right plate section.
            exclude_mask=None,
            strict_exclude_mask=strict_exclude_mask
        )
        existing_audit["index"] = int(index)
        existing_audit["source"] = "existing_b_candidate"
        audits.append(existing_audit)
        if existing_audit.get("status") == "accepted":
            baseline_mask = merged_existing
        if any(
            row.get("reason") == "background_edge_envelope"
            for row in existing_audit.get("components", [])
        ):
            rejected_existing_envelope = True

    points = build_positive_points_from_mask(baseline_mask > 0.5, target_bbox) or []
    if not points:
        if rejected_existing_envelope:
            return finalize_compound_mask(baseline_mask, {
                "status": "accepted",
                "reason": "existing_background_envelopes_removed",
                "candidates": audits
            })
        return candidate_masks, {"status": "skipped", "reason": "no_core_points"}
    results = run_sam_bbox_inference(
        img,
        prompt_bbox,
        multimask_output=True,
        imgsz=policy.get("samImgSize", 1024),
        points=[points],
        labels=[[1] * len(points)],
        model_variant="b"
    )
    try:
        discovery_masks = normalize_result_masks(
            results,
            img_w,
            img_h,
            interpolation=cv2.INTER_LINEAR,
            debug_label=f"{layer_name} compound_component_discovery"
        )
    finally:
        del results
    accepted = []
    discovery_iterable = discovery_masks if discovery_masks is not None else []
    for index, discovery_mask in enumerate(discovery_iterable):
        merged, audit = audit_compound_component_discovery(
            baseline_mask,
            np.maximum(baseline_mask, discovery_mask).astype(np.float32),
            target_bbox,
            prompt_bbox,
            config=config,
            exclude_mask=exclude_mask,
            strict_exclude_mask=strict_exclude_mask
        )
        audit["index"] = int(index)
        audits.append(audit)
        if audit.get("status") == "accepted":
            accepted.append((float(audit.get("verifiedAddedPixels", 0)), index, merged, audit))
    if not accepted:
        baseline_changed = int(np.count_nonzero(baseline_mask > 0.5)) > baseline_core_pixels
        if rejected_existing_envelope or baseline_changed:
            return finalize_compound_mask(baseline_mask, {
                "status": "accepted",
                "reason": (
                    "existing_background_envelopes_removed"
                    if rejected_existing_envelope else
                    "no_verified_discovery_keep_baseline"
                ),
                "promptBbox": prompt_bbox,
                "candidateCount": int(len(discovery_masks)),
                "baselinePixels": int(np.count_nonzero(baseline_mask > 0.5)),
                "candidates": audits
            })
        return candidate_masks, {
            "status": "rejected",
            "reason": "no_verified_components",
            "promptBbox": prompt_bbox,
            "candidateCount": int(len(discovery_masks)),
            "candidates": audits
        }
    accepted.sort(key=lambda item: item[0], reverse=True)
    _, selected_index, selected_mask, selected_audit = accepted[0]
    return finalize_compound_mask(selected_mask, {
        **selected_audit,
        "status": "accepted",
        "selectedIndex": int(selected_index),
        "promptBbox": prompt_bbox,
        "candidateCount": int(len(discovery_masks)),
        "baselinePixels": int(np.count_nonzero(baseline_mask > 0.5)),
        "candidates": audits
    })


def has_verified_compound_silhouette(layer_meta):
    """Whether compound discovery accepted at least one supported extension."""
    audit = (layer_meta or {}).get("_compoundComponentDiscovery") or {}
    if audit.get("status") != "accepted":
        return False
    try:
        if int(audit.get("verifiedAddedPixels", 0)) > 0:
            return True
    except (TypeError, ValueError):
        pass
    for candidate in audit.get("candidates", []):
        if not isinstance(candidate, dict) or candidate.get("status") != "accepted":
            continue
        try:
            if int(candidate.get("verifiedAddedPixels", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def merge_furniture_l_peer_masks(primary_mask, l_masks, primary_index, target_bbox):
    """Merge safe attached peers from L multimask output into its primary.

    SAM multimask results can split a physical object into a body candidate and
    a small attached component. The merge is shape-agnostic: it relies on
    overlap, proximity, bbox containment, and bounded component size rather
    than naming a furniture part.
    """
    merged = np.asarray(primary_mask > 0.5, dtype=bool).copy()
    if l_masks is None or len(l_masks) < 2 or primary_index < 0:
        return merged.astype(np.float32), {"peers": 0, "pixels": 0}

    height, width = merged.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1 = clamp(x1, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    x2 = clamp(x2, x1 + 1, width)
    y2 = clamp(y2, y1 + 1, height)
    target_area = max(1, (x2 - x1) * (y2 - y1))
    min_side = max(1, min(x2 - x1, y2 - y1))
    attach_radius = max(8, min(24, int(round(min_side * 0.035))))
    contact_radius = max(2, min(6, int(round(min_side * 0.010))))
    near_merged = cv2.dilate(
        merged.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (attach_radius * 2 + 1, attach_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    contact_merged = cv2.dilate(
        merged.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (contact_radius * 2 + 1, contact_radius * 2 + 1)
        ),
        iterations=1
    ) > 0
    max_total_added = max(256, int(target_area * 0.18))
    total_added = 0
    merged_peers = 0

    peer_rows = []
    for index, candidate in enumerate(l_masks):
        if index == primary_index:
            continue
        candidate_binary = np.asarray(candidate > 0.5, dtype=bool).copy()
        candidate_binary[:y1] = False
        candidate_binary[y2:] = False
        candidate_binary[:, :x1] = False
        candidate_binary[:, x2:] = False
        candidate_area = int(np.count_nonzero(candidate_binary))
        if candidate_area <= 0:
            continue
        overlap = int(np.count_nonzero(candidate_binary & merged)) / candidate_area
        peer_rows.append((overlap, index, candidate_binary))
    peer_rows.sort(reverse=True, key=lambda row: row[0])

    for overlap, index, candidate_binary in peer_rows:
        # A peer may overlap the body, or it may be a detached-but-adjacent
        # part represented separately by SAM. The latter is accepted only when
        # most of its pixels are inside the narrow neighborhood of the body.
        contact_ratio = int(np.count_nonzero(candidate_binary & contact_merged)) / max(
            1,
            int(np.count_nonzero(candidate_binary))
        )
        if overlap < 0.45 and contact_ratio < 0.60:
            continue
        added = candidate_binary & (~merged)
        if not np.any(added):
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            added.astype(np.uint8),
            connectivity=8
        )
        accepted = np.zeros_like(added, dtype=bool)
        peer_pixels = 0
        for label in range(1, count):
            x, y, comp_w, comp_h, area = [int(value) for value in stats[label]]
            if area < max(24, int(target_area * 0.00012)):
                continue
            if area > max(128, int(target_area * 0.10)):
                continue
            component = labels == label
            inside_ratio = int(np.count_nonzero(component[y1:y2, x1:x2])) / max(1, area)
            if inside_ratio < 0.98:
                continue
            if not np.any(component & contact_merged):
                continue
            accepted |= component
            peer_pixels += area

        if peer_pixels <= 0 or total_added + peer_pixels > max_total_added:
            continue
        merged |= accepted
        total_added += peer_pixels
        merged_peers += 1
        near_merged = cv2.dilate(
            merged.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (attach_radius * 2 + 1, attach_radius * 2 + 1)
            ),
            iterations=1
        ) > 0
        contact_merged = cv2.dilate(
            merged.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (contact_radius * 2 + 1, contact_radius * 2 + 1)
            ),
            iterations=1
        ) > 0

    return merged.astype(np.float32), {
        "peers": merged_peers,
        "pixels": total_added,
    }


def merge_composite_instance_masks(primary_mask, l_masks, primary_index, target_bbox):
    """Union independently segmented instances for one semantic composite.

    A composite layer is deliberately kept as one workbench asset, but its
    visible members may be disconnected.  The normal furniture peer merger
    only accepts attached parts, which silently drops a second stool/chair.
    Here we accept a contained, substantial, detached candidate as another
    instance while rejecting bbox-sized envelopes and tiny noise.
    """
    merged = np.asarray(primary_mask > 0.5, dtype=bool).copy()
    if l_masks is None or len(l_masks) < 2 or primary_index < 0:
        return merged.astype(np.float32), {"peers": 0, "pixels": 0, "mode": "composite_union"}

    height, width = merged.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1 = clamp(x1, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    x2 = clamp(x2, x1 + 1, width)
    y2 = clamp(y2, y1 + 1, height)
    target_area = max(1, (x2 - x1) * (y2 - y1))
    primary_area = max(1, int(np.count_nonzero(merged[y1:y2, x1:x2])))
    max_union_area = max(primary_area, int(target_area * 0.90))
    min_peer_area = max(64, int(target_area * 0.015), int(primary_area * 0.12))
    peer_rows = []

    for index, candidate in enumerate(l_masks):
        if index == primary_index:
            continue
        candidate_binary = np.asarray(candidate > 0.5, dtype=bool).copy()
        candidate_binary[:y1] = False
        candidate_binary[y2:] = False
        candidate_binary[:, :x1] = False
        candidate_binary[:, x2:] = False
        candidate_area = int(np.count_nonzero(candidate_binary[y1:y2, x1:x2]))
        if candidate_area < min_peer_area:
            continue
        raw_area = max(1, int(np.count_nonzero(candidate_binary)))
        containment = candidate_area / raw_area
        fill = candidate_area / target_area
        if containment < 0.90 or fill > 0.65:
            continue
        overlap = int(np.count_nonzero(candidate_binary & merged)) / max(1, raw_area)
        # Alternatives for the same instance substantially overlap. A
        # composite candidate may already contain the primary instance plus a
        # detached peer, so permit that higher-overlap case only when it adds
        # a substantial amount of new contained area.
        if overlap > 0.80:
            continue
        added = candidate_binary & (~merged)
        added_area = int(np.count_nonzero(added[y1:y2, x1:x2]))
        if added_area < min_peer_area:
            continue
        if overlap > 0.45 and added_area < int(primary_area * 0.25):
            continue
        peer_rows.append((added_area, index, candidate_binary))

    peer_rows.sort(reverse=True, key=lambda row: row[0])
    accepted_peers = 0
    accepted_pixels = 0
    for _added_area, index, candidate_binary in peer_rows:
        added = candidate_binary & (~merged)
        added_inside = int(np.count_nonzero(added[y1:y2, x1:x2]))
        if added_inside <= 0 or int(np.count_nonzero(merged[y1:y2, x1:x2])) + added_inside > max_union_area:
            continue
        merged |= added
        accepted_peers += 1
        accepted_pixels += added_inside

    return merged.astype(np.float32), {
        "peers": accepted_peers,
        "pixels": accepted_pixels,
        "mode": "composite_union"
    }


def arbitrate_sam_b_l_masks(
    b_masks,
    l_masks,
    target_bbox,
    strategy_type=None,
    b_failure_reason="",
    completion_recovery=False,
    completion_observation=None,
    completion_occlusion_mask=None,
    completion_base_strategy_type=None,
    completion_spatial=False,
    completion_flat_hard_edge=False,
    composite_instance_union=False
):
    """Choose an L recovery mask without allowing background envelopes.

    Furniture and canonical completion use an independent recovery path: when
    B is partial, requiring L to preserve 98.5% of B makes the damaged B mask
    the authority and rejects a genuinely complete silhouette. Ordinary table
    extraction keeps the stricter incremental arbitration because its bbox can
    contain floor or wall.
    """
    if completion_recovery:
        recovery_masks = l_masks if l_masks is not None and len(l_masks) > 0 else b_masks
        selected, reason = choose_completion_recovery_mask(
            recovery_masks,
            target_bbox,
            strategy_type=strategy_type,
            observation_mask=completion_observation,
            occlusion_mask=completion_occlusion_mask,
            base_strategy_type=completion_base_strategy_type,
            spatial=completion_spatial,
            prefer_full_scene=completion_flat_hard_edge
        )
        if selected is not None:
            return np.stack([selected], axis=0), reason

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

    b_failed = str(b_failure_reason).startswith((
        "low_b_",
        "b_internal_gaps=",
        "multi_entity_disagreement=",
        "no_b_",
        "completion_occluder_"
    ))
    # B is the established identity baseline.  A structural escalation (for
    # example internal gaps) does not authorize L to replace a usable B
    # silhouette: L may only add audited pixels in the bounded hybrid path
    # below.  Independent L recovery remains available when B is effectively
    # empty, which is the actual fallback case.
    direct_l_recovery = bool(
        b_failed and
        strategy_type != "soft_edge" and
        base_area < max(256, int(target_area * 0.08))
    )
    if direct_l_recovery:
        recovery_candidates = []
        for index, candidate in enumerate(l_masks):
            raw_binary = np.asarray(candidate > 0.5, dtype=bool)
            raw_area = int(np.count_nonzero(raw_binary))
            if raw_area <= 0:
                continue

            clipped = raw_binary.copy()
            clipped[:y1] = False
            clipped[y2:] = False
            clipped[:, :x1] = False
            clipped[:, x2:] = False
            clipped_area = int(np.count_nonzero(clipped[y1:y2, x1:x2]))
            if clipped_area <= 0:
                continue

            fill = clipped_area / target_area
            raw_inside_ratio = clipped_area / raw_area
            outside_ratio = 1.0 - raw_inside_ratio
            if raw_inside_ratio < 0.86 or outside_ratio > 0.14:
                continue
            # A bbox-filling envelope is more likely background than a
            # complete upholstered object. Keep a margin below full coverage.
            if fill < 0.22 or fill > 0.90:
                continue

            component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
                clipped[y1:y2, x1:x2].astype(np.uint8),
                connectivity=8
            )
            substantial_components = sum(
                1 for component_label in range(1, component_count)
                if int(component_stats[component_label, cv2.CC_STAT_AREA]) >= max(64, int(target_area * 0.01))
            )
            if substantial_components == 0:
                continue

            # L can still return a mostly-contained mask with detached pixels
            # or large internal pockets. Measure those defects before choosing
            # the direct recovery candidate; otherwise the first reasonable
            # fill can win despite producing the broken matte seen downstream.
            crop_binary = clipped[y1:y2, x1:x2]
            component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                crop_binary.astype(np.uint8),
                connectivity=8
            )
            total_pixels = max(1, int(np.count_nonzero(crop_binary)))
            largest_component = 0
            small_component_pixels = 0
            for component_label in range(1, component_count):
                component_area = int(component_stats[component_label, cv2.CC_STAT_AREA])
                largest_component = max(largest_component, component_area)
                if component_area < max(96, int(target_area * 0.008)):
                    small_component_pixels += component_area
            inverted = (~crop_binary).astype(np.uint8)
            hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(
                inverted,
                connectivity=8
            )
            enclosed_hole_pixels = 0
            for hole_label in range(1, hole_count):
                hx, hy, hw, hh, hole_area = [int(value) for value in hole_stats[hole_label]]
                if hx > 0 and hy > 0 and hx + hw < crop_binary.shape[1] and hy + hh < crop_binary.shape[0]:
                    enclosed_hole_pixels += hole_area
            largest_ratio = largest_component / total_pixels
            fragment_ratio = small_component_pixels / total_pixels
            hole_ratio = enclosed_hole_pixels / target_area

            # Prefer a contained, reasonably complete silhouette. The score is
            # intentionally independent of B overlap so damaged B cannot veto L.
            score = (
                raw_inside_ratio * 2.4 +
                (1.0 - min(1.0, abs(fill - 0.58))) -
                max(0.0, fill - 0.72) * 2.5 +
                min(0.12, substantial_components * 0.02) +
                min(0.18, largest_ratio * 0.18) -
                min(0.32, fragment_ratio * 1.8) -
                min(0.36, hole_ratio * 1.8)
            )
            recovery_candidates.append((
                score,
                index,
                clipped,
                fill,
                largest_ratio,
                fragment_ratio,
                hole_ratio
            ))

        if recovery_candidates:
            recovery_candidates.sort(key=lambda item: item[0], reverse=True)
            score, index, selected, fill, largest_ratio, fragment_ratio, hole_ratio = recovery_candidates[0]
            recovered = np.zeros_like(b_masks[0], dtype=np.float32)
            recovered[selected] = 1.0
            peer_debug = {"peers": 0, "pixels": 0}
            if strategy_type == "furniture":
                merger = (
                    merge_composite_instance_masks
                    if composite_instance_union else
                    merge_furniture_l_peer_masks
                )
                recovered, peer_debug = merger(recovered, l_masks, index, target_bbox)
            return np.stack([recovered], axis=0), (
                f"l_direct_{strategy_type}_recovery=index={index} score={score:.3f} "
                f"fill={fill:.3f} largest={largest_ratio:.3f} "
                f"fragments={fragment_ratio:.3f} holes={hole_ratio:.3f} "
                f"unionMode={'composite' if composite_instance_union else 'attached'} "
                f"peers={peer_debug.get('peers', 0)} "
                f"peerPixels={peer_debug.get('pixels', 0)}"
            )

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


def choose_completion_recovery_mask(
    candidate_masks,
    target_bbox,
    strategy_type=None,
    observation_mask=None,
    occlusion_mask=None,
    base_strategy_type=None,
    spatial=False,
    prefer_full_scene=False
):
    """Choose a complete-looking completion mask without preserving bad B geometry.

    The first SAM pass is intentionally a partial observation. Requiring SAM-L
    to preserve it pixel-for-pixel makes the partial mask veto the completed
    object. Completion mode therefore ranks contained candidates by coverage,
    while rejecting bbox-sized scene envelopes.
    """
    if candidate_masks is None or len(candidate_masks) == 0:
        return None, "completion_no_candidates"

    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target_area = max(1, (x2 - x1) * (y2 - y1))
    observation_binary = np.asarray(observation_mask > 0.5, dtype=bool) if observation_mask is not None else None
    occlusion_binary = np.asarray(occlusion_mask > 0.5, dtype=bool) if occlusion_mask is not None else None
    observation_area = int(np.count_nonzero(observation_binary)) if observation_binary is not None else 0
    occlusion_target_area = (
        int(np.count_nonzero(occlusion_binary[y1:y2, x1:x2]))
        if occlusion_binary is not None else 0
    )
    best = None
    rows = []
    for index, candidate in enumerate(candidate_masks):
        binary = np.asarray(candidate > 0.5, dtype=bool)
        area = int(np.count_nonzero(binary))
        if area <= 0:
            continue
        inside = int(np.count_nonzero(binary[y1:y2, x1:x2]))
        bbox = mask_bbox(candidate)
        if not bbox:
            continue
        fill = inside / target_area
        containment = inside / area
        overlap = intersection_area(bbox, target_bbox) / target_area
        area_ratio = area / target_area
        observation_recall = None
        observation_iou = None
        if observation_area > 0:
            observation_overlap = int(np.count_nonzero(binary & observation_binary))
            observation_recall = observation_overlap / observation_area
            observation_iou = observation_overlap / max(
                1, int(np.count_nonzero(binary | observation_binary))
            )
        completion_recovery_pixels = 0
        completion_recovery_ratio = None
        if occlusion_target_area > 0:
            new_pixels = binary[y1:y2, x1:x2]
            if observation_binary is not None:
                new_pixels = new_pixels & (~observation_binary[y1:y2, x1:x2])
            completion_recovery_pixels = int(np.count_nonzero(
                new_pixels & occlusion_binary[y1:y2, x1:x2]
            ))
            completion_recovery_ratio = completion_recovery_pixels / occlusion_target_area
        observation_consistent, observation_reason = completion_observation_is_consistent(
            observation_area,
            observation_recall,
            observation_iou,
            containment,
            base_strategy_type=base_strategy_type,
            spatial=spatial
        )
        # The completion crop includes scene context. A valid target must be
        # mostly contained by the prompt bbox, but it should not be the whole
        # crop or a room/background envelope.
        # The initial observation is a visible-only mask and can be slightly
        # misregistered after the scene inpaint/rescale round-trip. For a
        # bounded, well-contained completion candidate, allow a small recall
        # loss; reconciliation restores any missing original pixels and the
        # final observation audit remains the authoritative safety check.
        observation_anchor_ok = (
            observation_area == 0 or
            observation_recall >= 0.72 or
            (
                observation_recall >= 0.60 and
                containment >= 0.90 and
                0.50 <= fill <= 0.78 and
                area_ratio <= 0.78
            )
        )
        # A completion candidate must contribute pixels in the actual
        # foreground region that was removed for inpainting. Otherwise a
        # visible-only SAM mask can pass the observation checks while silently
        # dropping the completed part of the object.
        minimum_recovery_pixels = max(256, int(occlusion_target_area * 0.03))
        completion_zone_ok = (
            occlusion_target_area <= 0 or
            completion_recovery_pixels >= minimum_recovery_pixels
        )
        accepted = (
            containment >= 0.72 and
            0.10 <= fill <= 0.78 and
            area_ratio <= 0.82 and
            overlap >= 0.70 and
            observation_anchor_ok and
            completion_zone_ok and
            observation_consistent
        )
        row = {
            "index": index,
            "fill": round(fill, 3),
            "inside": round(containment, 3),
            "area": round(area_ratio, 3),
            "bboxOverlap": round(overlap, 3),
            "observationRecall": round(observation_recall, 3) if observation_recall is not None else None,
            "observationIoU": round(observation_iou, 3) if observation_iou is not None else None,
            "completionRecoveryPixels": completion_recovery_pixels,
            "completionRecoveryRatio": round(completion_recovery_ratio, 3) if completion_recovery_ratio is not None else None,
            "reason": observation_reason,
            "accepted": accepted
        }
        rows.append(row)
        if not accepted:
            continue
        # Coverage is the useful signal after inpaint. Containment and bbox
        # overlap break ties against a scene envelope.
        recovery_bonus = min(0.9, (completion_recovery_ratio or 0.0) * 2.0)
        score = (
            fill * 3.0 + containment * 1.2 + overlap * 0.25 + recovery_bonus
            - max(0.0, area_ratio - 0.55) * 0.8
        )
        if best is None or score > best[0]:
            best = (score, index, binary.astype(np.float32), row)

    if best is None:
        # A post-inpaint SAM candidate is not required to be a trustworthy
        # full-object mask. In practice SAM-L may return a scene envelope
        # whose useful information is only the small part inside the region
        # that was occluded during completion. Recover that delta locally and
        # keep the first-pass visible silhouette as the identity anchor.
        incremental_best = None
        if observation_binary is not None and occlusion_binary is not None:
            image_h, image_w = observation_binary.shape[:2]
            anchor_radius = max(8, min(28, int(round(min(image_h, image_w) * 0.04))))
            anchor_support = cv2.dilate(
                observation_binary.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (anchor_radius * 2 + 1, anchor_radius * 2 + 1)
                ),
                iterations=1
            ) > 0
            allowed_zone = occlusion_binary.copy()
            allowed_zone[:y1] = False
            allowed_zone[y2:] = False
            allowed_zone[:, :x1] = False
            allowed_zone[:, x2:] = False
            target_area_float = float(target_area)
            occlusion_area = max(1, int(np.count_nonzero(allowed_zone)))
            minimum_delta_pixels = max(64, int(occlusion_area * 0.005))
            maximum_delta_pixels = max(256, int(target_area_float * 0.22))

            for index, candidate in enumerate(candidate_masks):
                candidate_binary = np.asarray(candidate > 0.5, dtype=bool)
                delta = candidate_binary & (~observation_binary) & allowed_zone
                # Only accept newly generated pixels that are close to the
                # known silhouette. This prevents a broad background mask
                # from turning the complete occluder into the target object.
                delta &= anchor_support
                delta_pixels = int(np.count_nonzero(delta))
                if delta_pixels < minimum_delta_pixels or delta_pixels > maximum_delta_pixels:
                    continue

                component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                    delta.astype(np.uint8),
                    connectivity=8
                )
                substantial_components = 0
                for component_index in range(1, component_count):
                    component_area = int(component_stats[component_index, cv2.CC_STAT_AREA])
                    if component_area >= max(24, int(target_area_float * 0.0005)):
                        substantial_components += 1
                if substantial_components == 0:
                    continue

                merged_binary = observation_binary | delta
                merged_binary[:y1] = False
                merged_binary[y2:] = False
                merged_binary[:, :x1] = False
                merged_binary[:, x2:] = False
                merged_area = int(np.count_nonzero(merged_binary))
                merged_inside = int(np.count_nonzero(merged_binary[y1:y2, x1:x2]))
                merged_fill = merged_inside / target_area_float
                merged_containment = merged_inside / max(1, merged_area)
                if merged_fill < 0.10 or merged_fill > 0.90 or merged_containment < 0.88:
                    continue

                delta_ratio = delta_pixels / target_area_float
                score = (
                    min(0.22, delta_ratio) * 3.0 +
                    min(1.0, merged_containment) * 0.8 -
                    max(0.0, delta_ratio - 0.12) * 2.0
                )
                if incremental_best is None or score > incremental_best[0]:
                    incremental_best = (
                        score,
                        index,
                        merged_binary.astype(np.float32),
                        delta_pixels,
                        merged_fill,
                        merged_containment,
                        int(np.count_nonzero(delta & occlusion_binary)) / occlusion_area
                    )

        if incremental_best is not None:
            (
                score,
                index,
                selected,
                delta_pixels,
                merged_fill,
                merged_containment,
                recovery_ratio
            ) = incremental_best
            print(
                f"Completion candidate selection: incremental_recovery index={index} "
                f"score={score:.3f} deltaPixels={delta_pixels} "
                f"fill={merged_fill:.3f} inside={merged_containment:.3f} "
                f"recoveryRatio={recovery_ratio:.3f}"
            )
            return selected, (
                f"completion_incremental_recovery=index={index} "
                f"deltaPixels={delta_pixels}"
            )

        if observation_binary is not None and observation_area > 0:
            # A failed recovery must not erase a valid first-pass layer. The
            # caller marks this result so normal completion-zone validation
            # does not reject the deliberate observation-only fallback.
            print(
                f"Completion candidate selection: observation_fallback "
                f"pixels={observation_area} rows={rows[:8]}"
            )
            return observation_binary.astype(np.float32), "completion_observation_fallback"

        print(f"Completion candidate selection: no accepted mask rows={rows[:8]}")
        return None, "completion_no_acceptable_candidate"

    score, index, selected, row = best
    # A completed flat hard-edge entity is materially different from a table,
    # chair, or soft edge: when its full-scene candidate passes containment,
    # observation, and occlusion-zone checks above, keep that silhouette as
    # the canonical result.  The original first-pass mask remains an audit
    # reference, not a pixel-level clipping constraint.  Spatial objects keep
    # the established conservative incremental/reconciliation path.
    full_scene_direct = bool(prefer_full_scene and not spatial)
    print(
        f"Completion candidate selection: {'full_scene_sam' if full_scene_direct else 'model_independent'} index={index} "
        f"score={score:.3f} fill={row['fill']:.3f} inside={row['inside']:.3f} "
        f"area={row['area']:.3f} overlap={row['bboxOverlap']:.3f} "
        f"recoveryPixels={row.get('completionRecoveryPixels', 0)} "
        f"recoveryRatio={row.get('completionRecoveryRatio')}"
    )
    output_mode = "completion_full_scene_sam" if full_scene_direct else "completion_direct_recovery"
    return selected, (
        f"{output_mode}=index={index} score={score:.3f} "
        f"fill={row['fill']:.3f} inside={row['inside']:.3f}"
    )


def boundary_recovery_evidence(
    candidate_masks,
    target_bbox,
    strategy_type=None,
    layer_meta=None,
    context_layers=None,
    policy=None
):
    """Collect model-independent evidence for a bounded boundary review."""
    config = (policy or {}).get("boundaryRecovery") or {}
    strategy_type = strategy_type or (policy or {}).get("selectionType") or "default"
    edge_type = config.get("edgeType", "hard_edge")
    evidence = {
        "boundary_truncated": False,
        "candidate_disagreement": False,
        "context_conflict": False,
        "recovery_growth": None,
        "preserve_core": False,
        "edge_type": edge_type,
        "needed": False,
        "reason": "no_b_candidates"
    }
    if candidate_masks is None or len(candidate_masks) == 0:
        return evidence
    reference = choose_b_reference_mask(
        candidate_masks,
        target_bbox,
        strategy_type=strategy_type
    )
    if reference is None:
        evidence["reason"] = "no_b_reference"
        return evidence
    evidence["preserve_core"] = True
    reference_bbox = mask_bbox(reference)
    if not reference_bbox:
        evidence["preserve_core"] = False
        evidence["reason"] = "empty_b_reference"
        return evidence
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    rx1, ry1, rx2, ry2 = [int(value) for value in reference_bbox]
    touch_count = (
        int(rx1 <= x1 + 2) + int(ry1 <= y1 + 2) +
        int(rx2 >= x2 - 2) + int(ry2 >= y2 - 2)
    )
    target_area = max(1, (x2 - x1) * (y2 - y1))
    fill = int(np.count_nonzero((reference > 0.5)[y1:y2, x1:x2])) / target_area
    evidence["boundary_truncated"] = bool(
        touch_count >= int(config.get("minTouchCount", 4)) or
        (fill < float(config.get("lowFillThreshold", 0.42)) and target_area >= 4096)
    )
    valid_fills = []
    for candidate in candidate_masks:
        binary = np.asarray(candidate > 0.5, dtype=bool)
        if not np.any(binary):
            continue
        valid_fills.append(
            int(np.count_nonzero(binary[y1:y2, x1:x2])) / target_area
        )
    evidence["candidate_disagreement"] = bool(
        len(valid_fills) >= 2 and max(valid_fills) - min(valid_fills) >= 0.22
    )
    exclude_bboxes = build_exclude_bboxes(
        layer_meta or {}, context_layers or [], target_bbox,
        reference.shape[1], reference.shape[0]
    )
    if exclude_bboxes:
        exclude_mask = build_exclude_mask(
            exclude_bboxes, reference.shape[1], reference.shape[0]
        )
        evidence["context_conflict"] = bool(
            int(np.count_nonzero((reference > 0.5) & exclude_mask)) /
            max(1, int(np.count_nonzero(reference > 0.5))) > 0.08
        )
    # A bbox edge contact is common for tightly annotated subjects. Escalate
    # only when the selected B mask is also visibly sparse, or when candidate
    # disagreement occurs in a genuinely low-fill case. Context overlap alone
    # is not enough to spend another full image encoding pass.
    evidence["needed"] = bool(
        (
            evidence["boundary_truncated"] and
            fill < float(config.get("recoveryTriggerFill", 0.52))
        ) or
        (
            evidence["candidate_disagreement"] and
            fill < float(config.get("disagreementTriggerFill", 0.52))
        )
    )
    if evidence["needed"] and evidence["boundary_truncated"]:
        evidence["reason"] = f"boundary_truncated touch={touch_count} fill={fill:.3f}"
    elif evidence["needed"] and evidence["candidate_disagreement"]:
        evidence["reason"] = "candidate_disagreement"
    elif evidence["needed"] and evidence["context_conflict"]:
        evidence["reason"] = "context_conflict"
    else:
        evidence["reason"] = f"not_needed touch={touch_count} fill={fill:.3f}"
    return evidence


def _boundary_recovery_prompt_inputs(coarse_mask, target_bbox, prompt_bbox):
    """Build conservative points that preserve the known core."""
    coarse_binary = coarse_mask > 0.5
    positive = build_positive_points_from_mask(coarse_binary, target_bbox) or []
    if not positive:
        positive = sample_points_in_bbox(target_bbox, [(0.5, 0.5)])
    # Negative points are sampled only on the expanded prompt border. Do not
    # mark holes inside the original bbox as background: those holes may be
    # exactly the missing hair/arm evidence this pass is meant to recover.
    negative = build_boundary_negative_points(
        coarse_binary,
        prompt_bbox,
        max_points=4
    )
    points = []
    labels = []
    seen = set()
    for point in positive:
        key = (int(point[0]), int(point[1]), 1)
        if key not in seen:
            seen.add(key)
            points.append([key[0], key[1]])
            labels.append(1)
    for point in negative:
        key = (int(point[0]), int(point[1]), 0)
        if key not in seen:
            seen.add(key)
            points.append([key[0], key[1]])
            labels.append(0)
    return ([points] if points else None), ([labels] if labels else None)


def recover_boundary_with_sam_l(
    img,
    b_masks,
    target_bbox,
    prompt_bbox,
    layer_name,
    strategy_type=None,
    layer_meta=None,
    context_layers=None,
    policy=None
):
    """Review one suspicious entity with full-image SAM-L evidence.

    The returned mask is accepted only when it retains the B silhouette and
    adds a connected, bounded extension. This keeps a broad SAM-L envelope
    from becoming an output simply because it is larger than B.
    """
    strategy_type = strategy_type or (policy or {}).get("selectionType") or "default"
    config = (policy or {}).get("boundaryRecovery") or {}
    reference = choose_b_reference_mask(b_masks, target_bbox, strategy_type=strategy_type)
    if reference is None:
        return None, {"status": "rejected", "reason": "no_b_reference"}
    points, labels = _boundary_recovery_prompt_inputs(reference, target_bbox, prompt_bbox)
    l_results, l_imgsz = run_sam_l_with_retry(
        img,
        prompt_bbox,
        strategy_type,
        layer_name,
        policy=policy,
        points=points,
        labels=labels
    )
    try:
        l_masks = normalize_result_masks(
            l_results,
            img.shape[1],
            img.shape[0],
            interpolation=cv2.INTER_LINEAR,
            debug_label=f"{layer_name} boundary_recovery model=L imgsz={l_imgsz}"
        )
    finally:
        del l_results

    height, width = img.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    px1, py1, px2, py2 = [int(value) for value in prompt_bbox]
    target_area = max(1, (x2 - x1) * (y2 - y1))
    prompt_area = max(1, (px2 - px1) * (py2 - py1))
    coarse = reference > 0.5
    coarse_area = max(1, int(np.count_nonzero(coarse)))
    coarse_dilated = cv2.dilate(
        coarse.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1
    ) > 0

    exclude_mask = None
    exclude_bboxes = build_exclude_bboxes(
        layer_meta or {}, context_layers or [], target_bbox, width, height
    )
    if exclude_bboxes:
        exclude_mask = build_exclude_mask(exclude_bboxes, width, height)

    accepted = []
    rejection_counts = {}

    def reject(reason):
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    for index, candidate in enumerate(l_masks):
        binary = np.asarray(candidate > 0.5, dtype=bool)
        if binary.shape != (height, width) or not np.any(binary):
            reject("invalid_or_empty")
            continue
        prompt_region = np.zeros_like(binary)
        prompt_region[py1:py2, px1:px2] = True
        binary &= prompt_region
        if not np.any(binary):
            reject("outside_prompt")
            continue
        overlap = int(np.count_nonzero(binary & coarse))
        preserve = overlap / coarse_area
        added = binary & (~coarse)
        added_area = int(np.count_nonzero(added))
        target_region = np.zeros_like(binary)
        target_region[y1:y2, x1:x2] = True
        added_outside_target = int(np.count_nonzero(added & ~target_region))
        # Recovery can repair a missing arm/hair/garment inside the semantic
        # bbox. Requiring growth outside the bbox incorrectly rejected exactly
        # those candidates and made every L pass fall back to B.
        min_added_area = max(
            int(config.get("minAddedPixels", 32)),
            int(coarse_area * float(config.get("minAddedRatio", 0.025)))
        )
        if added_area < min_added_area:
            reject("insufficient_growth")
            continue
        if preserve < float(config.get("minPreserveCore", 0.90)):
            reject("preserve_core")
            continue
        attached = int(np.count_nonzero(added & coarse_dilated)) / max(1, added_area)
        if attached < float(config.get("minAttachedGrowth", 0.78)):
            reject("growth_not_attached")
            continue
        growth = int(np.count_nonzero(binary)) / coarse_area
        if (
            growth > float(config.get("maxGrowth", 2.20)) or
            int(np.count_nonzero(binary)) > int(
                prompt_area * float(config.get("maxPromptAreaRatio", 0.92))
            )
        ):
            reject("growth_limit")
            continue
        prompt_containment = int(np.count_nonzero(binary & prompt_region)) / max(1, int(np.count_nonzero(binary)))
        if prompt_containment < 0.98:
            reject("prompt_containment")
            continue
        if exclude_mask is not None:
            excluded_ratio = int(np.count_nonzero(binary & exclude_mask)) / max(1, int(np.count_nonzero(binary)))
            if excluded_ratio > float(config.get("maxContextConflictRatio", 0.16)):
                reject("context_conflict")
                continue
        candidate_bbox = mask_bbox(binary)
        if not candidate_bbox:
            reject("empty_bbox")
            continue
        candidate_bbox_area = max(1, bbox_area(candidate_bbox))
        candidate_fill = int(np.count_nonzero(binary)) / candidate_bbox_area
        # Reject an almost solid prompt rectangle, a common SAM-L background
        # envelope when an entity touches several prompt edges.
        if candidate_fill > 0.94 and candidate_bbox_area > target_area * 1.25:
            reject("background_envelope")
            continue
        score = (
            preserve * 0.42 +
            min(1.0, added_area / max(1.0, coarse_area * 0.22)) * 0.34 +
            attached * 0.16 +
            min(1.0, candidate_fill / 0.72) * 0.08
        )
        accepted.append((score, index, binary.astype(np.float32), preserve, added_outside_target, candidate_bbox))

    if not accepted:
        return None, {
            "status": "rejected",
            "reason": "no_safe_boundary_candidate",
            "candidateCount": int(len(l_masks)),
            "imgsz": int(l_imgsz),
            "rejectionCounts": rejection_counts,
            "edgeType": config.get("edgeType", "hard_edge")
        }
    accepted.sort(key=lambda row: row[0], reverse=True)
    score, index, selected, preserve, added_outside_target, selected_bbox = accepted[0]
    effective_bbox = [
        min(x1, int(selected_bbox[0])),
        min(y1, int(selected_bbox[1])),
        max(x2, int(selected_bbox[2])),
        max(y2, int(selected_bbox[3]))
    ]
    return selected, {
        "status": "accepted",
        "index": int(index),
        "score": round(float(score), 4),
        "preserve": round(float(preserve), 4),
        "addedOutsideTarget": int(added_outside_target),
        "effectiveBbox": effective_bbox,
        "candidateCount": int(len(l_masks)),
        "imgsz": int(l_imgsz),
        "edgeType": config.get("edgeType", "hard_edge"),
        "preserveCore": round(float(preserve), 4),
        "recoveryGrowth": round(float(int(np.count_nonzero(selected)) / coarse_area), 4)
    }


def run_sam_l_with_retry(img, prompt_bbox, strategy_type, layer_name, policy=None, points=None, labels=None):
    """Run L at a bounded size and retry once at a lower size after CUDA OOM."""
    initial_imgsz = choose_sam_imgsz(
        img,
        strategy_type,
        model_variant="l",
        policy=policy
    )
    attempts = [initial_imgsz]
    retry_imgsz = min(initial_imgsz, SAM_L_OOM_RETRY_IMGSZ)
    if retry_imgsz < initial_imgsz:
        attempts.append(retry_imgsz)

    last_error = None
    for attempt_index, imgsz in enumerate(attempts):
        if attempt_index > 0:
            print(
                f"SAM-L retry for {layer_name}: imgsz={imgsz} "
                f"after_cuda_oom={initial_imgsz}"
            )
        if imgsz != HARD_EDGE_SAM_IMGSZ and strategy_type in HARD_EDGE_STRATEGIES:
            print(
                f"SAM-L adaptive imgsz for {layer_name}: "
                f"source={img.shape[:2]} imgsz={imgsz}"
            )
        try:
            results = run_sam_bbox_inference(
                img,
                prompt_bbox,
                multimask_output=True,
                imgsz=imgsz,
                points=points,
                labels=labels,
                model_variant="l"
            )
            return results, imgsz
        except Exception as error:
            last_error = error
            if not is_cuda_oom(error) or attempt_index >= len(attempts) - 1:
                raise
            release_sam_model("l", reason=f"oom_retry_{imgsz}")

    raise last_error or RuntimeError("SAM-L inference failed")


def should_run_local_upscale(candidate_masks, target_bbox, strategy_type):
    """Use local upscaling for small lighting or difficult furniture masks."""
    if not LOCAL_UPSCALE_ENABLED or strategy_type not in LOCAL_UPSCALE_STRATEGIES:
        return False, "strategy_disabled"
    if candidate_masks is None or len(candidate_masks) == 0:
        return True, "no_b_candidates"

    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)
    if (
        strategy_type == "lighting" and
        max(bbox_width, bbox_height) <= LOCAL_UPSCALE_MAX_BBOX_SIDE
    ):
        return True, f"small_{strategy_type}_bbox={bbox_width}x{bbox_height}"

    reference = choose_b_reference_mask(candidate_masks, target_bbox, strategy_type)
    if reference is None:
        return True, "no_b_reference"
    target_area = max(1, bbox_width * bbox_height)
    binary = reference > 0.5
    inside_pixels = int(np.count_nonzero(binary[y1:y2, x1:x2]))
    area = int(np.count_nonzero(binary))
    fill = inside_pixels / target_area
    inside = inside_pixels / max(1, area)
    fill_threshold = 0.56 if strategy_type == "lighting" else 0.68
    inside_threshold = 0.94 if strategy_type == "lighting" else 0.95
    if fill < fill_threshold:
        return True, f"low_{strategy_type}_fill={fill:.3f}"
    if inside < inside_threshold:
        return True, f"low_{strategy_type}_inside={inside:.3f}"
    return False, f"{strategy_type}_primary_accepted={fill:.3f}"


def should_run_l_local_upscale(candidate_masks, target_bbox, strategy_type):
    """Run a local L pass only for a difficult furniture escalation."""
    if strategy_type != "furniture" or candidate_masks is None or len(candidate_masks) == 0:
        return False, "strategy_disabled"

    reference = choose_b_reference_mask(candidate_masks, target_bbox, strategy_type)
    if reference is None:
        return True, "no_b_reference"

    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    target_area = max(1, (x2 - x1) * (y2 - y1))
    binary = reference > 0.5
    inside_pixels = int(np.count_nonzero(binary[y1:y2, x1:x2]))
    area = int(np.count_nonzero(binary))
    fill = inside_pixels / target_area
    inside = inside_pixels / max(1, area)
    if fill < 0.70 or inside < 0.95:
        return True, f"difficult_furniture_fill={fill:.3f}_inside={inside:.3f}"
    return False, f"furniture_local_l_not_needed={fill:.3f}"


def run_upscaled_hard_edge_bbox_inference(
    img,
    prompt_bbox,
    target_bbox,
    img_w,
    img_h,
    layer_name,
    model_variant="b",
    imgsz=LOCAL_UPSCALE_SAM_IMGSZ
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
    max_upscaled_side = LOCAL_UPSCALE_MAX_SOURCE_SIDE
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
            imgsz=imgsz,
            model_variant=model_variant
        )
        upscaled_masks = normalize_result_masks(
            results,
            up_w,
            up_h,
            interpolation=cv2.INTER_LINEAR,
            debug_label=f"{layer_name} strategy=local_upscaled"
        )
        del results
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
    bx1, by1, bx2, by2 = crop_bbox
    alpha = alpha_mask if alpha_mask is not None else dilate_and_feather_mask(mask)
    # Convert only the requested output crop. The previous implementation
    # converted and copied the entire source image for every layer, even
    # though only this crop is returned.
    cropped_img = cv2.cvtColor(img[by1:by2, bx1:bx2], cv2.COLOR_BGR2BGRA)
    if cropped_img.size == 0:
        return None
    cropped_img[:, :, 3] = alpha[by1:by2, bx1:bx2]

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
    original_target_bboxes=None,
    quality_profile="publish"
):
    h, w = img.shape[:2]
    cutouts = []

    for i, target_bbox in enumerate(pixel_bboxes):
        layer_started_at = time.perf_counter()
        if i >= len(layer_ids):
            break

        layer_meta = layer_metas[i] if isinstance(layer_metas, list) and i < len(layer_metas) else {}
        layer_policy = resolve_mask_policy(layer_meta, quality_profile)
        spatial_ownership = get_spatial_ownership_config(layer_meta, layer_policy)
        strict_spatial_output = bool(spatial_ownership.get("strictOutput"))
        sam_diagnostic_layer(i, layer_ids[i])
        completion_layer = is_completion_segmentation_layer(layer_meta)
        if completion_layer:
            print(
                f"Completion SAM selection enabled for {layer_ids[i]}: "
                f"completionSegmentation={bool((layer_meta or {}).get('completionSegmentation'))} "
                f"layerId={layer_meta.get('id') or layer_meta.get('layerId')}"
            )
        candidate_masks = mask_provider(target_bbox, layer_meta, i)
        sam_diagnostic_event(
            "provider_candidates", masks=candidate_masks,
            targetBbox=target_bbox, effectiveBbox=layer_meta.get("_samEffectiveBbox")
        )
        if candidate_masks is None or len(candidate_masks) == 0:
            print(
                f"SAM layer timing layer={layer_ids[i]} "
                f"durationMs={(time.perf_counter() - layer_started_at) * 1000:.1f} "
                "status=no_candidates"
            )
            continue
        # Recovery capabilities run inside mask_provider and can prove that a
        # connected subject continues beyond the original semantic bbox. Read
        # that evidence after inference so selection, alpha and output crop all
        # operate on the same accepted geometry.
        recovery_bbox = (layer_meta or {}).get("_samEffectiveBbox")
        if (
            not strict_spatial_output and
            isinstance(recovery_bbox, list) and
            len(recovery_bbox) == 4 and
            all(isinstance(value, (int, float)) for value in recovery_bbox)
        ):
            effective_bbox = [
                clamp(int(recovery_bbox[0]), 0, w - 1),
                clamp(int(recovery_bbox[1]), 0, h - 1),
                clamp(int(recovery_bbox[2]), 1, w),
                clamp(int(recovery_bbox[3]), 1, h)
            ]
            if (
                effective_bbox[2] > effective_bbox[0] and
                effective_bbox[3] > effective_bbox[1]
            ):
                target_bbox = effective_bbox
                print(
                    f"SAM boundary effective bbox layer={layer_ids[i]} "
                    f"effective={target_bbox}"
                )
        mask, selected_count, quality = select_and_merge_masks(
            candidate_masks,
            target_bbox,
            w,
            h,
            layer_meta,
            context_layers,
            quality_profile=quality_profile
        )
        selected_inside_ratio = 0.0
        if quality is not None:
            quality["qualityProfile"] = normalize_sam_quality_profile(quality_profile)
            quality["policyVersion"] = layer_policy["version"]
            quality["samModelVariant"] = str(
                (layer_meta or {}).get("_samModelVariant", "b")
            ).lower()
            quality["samSelectionRoute"] = str(
                (layer_meta or {}).get("_samSelectionRoute", "b_first")
            )
            quality["spatialOwnership"] = spatial_ownership
            selected_candidate = next(
                (row for row in (quality.get("debugCandidates") or []) if row.get("selected")),
                None
            )
            try:
                selected_inside_ratio = float((selected_candidate or {}).get("inside", 0.0))
            except (TypeError, ValueError):
                selected_inside_ratio = 0.0
            if layer_policy["completionOccluder"]:
                quality["completionOccluderAudit"] = layer_meta.get("_completionOccluderAudit") or {
                    "status": "unverified",
                    "selectedModel": "b",
                    "reason": "l_review_not_completed"
                }
        strategy_name = quality.get("strategy") if quality else "unknown"
        strategy_profile = quality.get("strategyProfile") if quality else "unknown"
        print(
            f"Layer {layer_ids[i]} engine={engine_name} "
            f"semanticStrategy={strategy_name} profile={strategy_profile} "
            f"base={quality.get('baseStrategyType') if quality else 'unknown'} "
            f"domain={quality.get('profile') if quality else 'unknown'} "
            f"phase={quality.get('phase') if quality else 'unknown'} "
            f"features={','.join(quality.get('features', [])) if quality else ''} "
            f"merged {selected_count} candidate masks"
        )
        if quality and quality.get("debugCandidates"):
            debug_summary = " | ".join([
                f"#{row['index']} s={row['score']} fill={row['fill']} excl={row['exclude']} "
                f"spatialExcl={row.get('spatialOwnershipExclude', 0)} "
                f"inside={row['inside']} area={row['area']} ov={row['bboxOverlap']} touch={row['touch']} "
                f"anchorOv={row.get('anchorBboxOverlap')} "
                f"recovery={row.get('completionRecoveryPixels', 0)}/"
                f"{row.get('completionRecoveryRatio')} "
                f"base={row.get('decorBase', False)} bot={row['shapeFeatures']['bottomBand']} "
                f"cy={row['shapeFeatures']['centerY']} block={row['shapeFeatures']['isBlockLike']} "
                f"thin={row['shapeFeatures']['isThinVertical']} "
                f"anchor={row.get('spatialCompletionAnchor', False)} "
                f"structPeer={row.get('spatialCompletionStructuralPeer', False)} "
                f"sel={row['selected']} why={row['rejectReason']}"
                for row in quality["debugCandidates"][:8]
            ])
            print(f"Layer {layer_ids[i]} candidates: {debug_summary}")
        if mask is None:
            print(
                f"SAM layer timing layer={layer_ids[i]} "
                f"durationMs={(time.perf_counter() - layer_started_at) * 1000:.1f} "
                "status=no_selected_mask"
            )
            continue

        strategy_type = quality.get("strategy") if quality else None
        if strategy_type == "furniture" and not completion_layer:
            baseline_support = layer_meta.pop("_probeBaselineSupport", None)
            if baseline_support is not None:
                mask, support_audit = reject_unsupported_probe_supports(
                    mask, baseline_support, layer_meta, target_bbox, image=img
                )
                if quality is not None:
                    quality["adjacentSupportReview"] = support_audit
                print(f"SAM adjacent support review layer={layer_ids[i]}: {support_audit}")
                sam_diagnostic_event(
                    "adjacent_support_review", masks=[mask], audit=support_audit,
                    targetBbox=target_bbox
                )
            if engine_name.startswith("sam"):
                mask, support_audit = recover_furniture_candidate_supports(
                    img, mask, candidate_masks, target_bbox, layer_meta,
                    context_layers, imgsz=layer_policy.get("samImgSize", 1024)
                )
                if quality is not None:
                    quality["candidateSupportRecovery"] = support_audit
                if support_audit["status"] == "accepted":
                    print(f"SAM furniture candidate support layer={layer_ids[i]}: {support_audit}")
                    sam_diagnostic_event(
                        "furniture_candidate_support", masks=[mask], audit=support_audit,
                        targetBbox=target_bbox
                    )
        owned_consensus_base = None
        owned_consensus_blocked = None
        owned_consensus_evidence = None
        sam_diagnostic_event(
            "selected_mask", masks=[mask], targetBbox=target_bbox, quality=quality
        )
        if (
            engine_name.startswith("sam") and
            strategy_type == "hard_product" and
            not completion_layer and
            strict_spatial_output and
            selected_count == 1 and
            len(candidate_masks) >= 3
        ):
            ownership_context = build_semantic_ownership_bboxes(
                layer_meta, context_layers or [], target_bbox, w, h
            )
            recovered_mask, peer_audit = recover_owned_candidate_consensus(
                img, mask, candidate_masks, target_bbox, ownership_context
            )
            if peer_audit["status"] == "accepted":
                sibling_context = [
                    entry for entry in ownership_context
                    if isinstance(entry, dict) and entry.get("spatial_ownership_guard")
                ]
                sibling_pixels, sibling_audit = resolve_sibling_pixel_exclusions(
                    img, mask, sibling_context, imgsz=layer_policy.get("samImgSize", 1024)
                )
                other_context = [entry for entry in ownership_context if entry not in sibling_context]
                pixel_blocked = sibling_pixels | build_exclude_mask(other_context, w, h)
                peer_audit["siblingPixels"] = sibling_audit
                verified_siblings = [
                    entry for entry, audit in zip(sibling_context, sibling_audit)
                    if audit["status"] == "accepted"
                ]
                if verified_siblings:
                    refined_mask, refined_audit = recover_owned_candidate_consensus(
                        img, mask, candidate_masks, target_bbox, ownership_context,
                        blocked_mask=pixel_blocked,
                        relaxed_region=build_exclude_mask(verified_siblings, w, h)
                    )
                else:
                    refined_mask, refined_audit = recovered_mask, {"status": "skipped"}
                if refined_audit["status"] == "accepted":
                    recovered_mask, peer_audit = refined_mask, {
                        **refined_audit, "siblingPixels": sibling_audit
                    }
                    owned_consensus_blocked = pixel_blocked
                else:
                    owned_consensus_blocked = build_exclude_mask(ownership_context, w, h)
                owned_consensus_base = np.asarray(mask > 0.5, dtype=bool)
                first, second = peer_audit["peerIndexes"]
                owned_consensus_evidence = (
                    (candidate_masks[first] > 0.5) & (candidate_masks[second] > 0.5)
                )
                mask = constrain_mask_to_bbox(recovered_mask, target_bbox)
                print(f"SAM owned candidate consensus layer={layer_ids[i]}: {peer_audit}")
                sam_diagnostic_event(
                    "owned_candidate_consensus", masks=[mask],
                    audit=peer_audit, targetBbox=target_bbox
                )
            if quality is not None:
                quality["ownedCandidateConsensus"] = peer_audit
        interior_hole_config = layer_policy.get("interiorHoleRecovery") or {}
        hole_recovered = False
        interior_hole_capable = supports_interior_hole_recovery(
            strategy_type, layer_policy
        )
        print(
            f"SAM interior hole capability for {layer_ids[i]}: "
            f"enabled={bool(interior_hole_config.get('enabled'))} "
            f"strategy={strategy_type} capable={interior_hole_capable}"
        )
        if (
            engine_name.startswith("sam") and
            interior_hole_capable
        ):
            mask, hole_recovered, hole_audit = recover_interior_holes_with_sam(
                img,
                mask,
                target_bbox,
                layer_ids[i],
                layer_meta=layer_meta,
                context_layers=context_layers,
                policy=layer_policy
            )
            layer_meta["_interiorHoleRecovery"] = hole_audit
            sam_diagnostic_event(
                "interior_hole_decision",
                masks=[mask],
                audit=hole_audit,
                targetBbox=target_bbox
            )
            print(
                f"SAM interior hole recovery for {layer_ids[i]}: "
                f"status={hole_audit.get('status')} "
                f"reason={hole_audit.get('reason')} "
                f"holePixels={hole_audit.get('holePixels', 0)} "
                f"holeRatio={hole_audit.get('holeRatio', 0)} "
                f"recovered={bool(hole_recovered)}"
            )
        base_strategy_type = quality.get("baseStrategyType", strategy_type) if quality else strategy_type
        if not layer_meta.get("completionSegmentation") and not strict_spatial_output:
            effective_bbox, bbox_evidence = derive_safe_entity_bbox(
                mask,
                target_bbox,
                strategy_type=strategy_type,
                capability_config=(
                    layer_policy.get("boundaryRecovery")
                    if engine_name.startswith("sam") else None
                )
            )
            if bbox_evidence:
                print(
                    f"Entity bbox evidence extension for {layer_ids[i]}: "
                    f"original={target_bbox} effective={effective_bbox} evidence={bbox_evidence}"
                )
                target_bbox = effective_bbox
        elif strict_spatial_output:
            # Arbitration and probing intentionally ran before this clip, as
            # in the September 16 path. The original semantic BBOX remains
            # the final output contract unless the compound pass independently
            # verified a tiny, attached peer-mask contour beyond its edge.
            edge_extension = (layer_meta or {}).get("_samStrictBboxEdgeExtension") or {}
            proposed_bbox = edge_extension.get("effectiveBbox")
            if edge_extension.get("status") == "accepted" and isinstance(proposed_bbox, list) and len(proposed_bbox) == 4:
                try:
                    ex1, ey1, ex2, ey2 = [int(value) for value in proposed_bbox]
                    ox1, oy1, ox2, oy2 = [int(value) for value in target_bbox]
                    valid_extension = bool(
                        ex1 <= ox1 and ey1 <= oy1 and ex2 >= ox2 and ey2 >= oy2 and
                        ox1 - ex1 <= 6 and oy1 - ey1 <= 6 and
                        ex2 - ox2 <= 6 and ey2 - oy2 <= 6
                    )
                except (TypeError, ValueError):
                    valid_extension = False
                if valid_extension:
                    target_bbox = [ex1, ey1, ex2, ey2]
                    sam_diagnostic_event(
                        "strict_bbox_edge_continuation",
                        masks=[mask], audit=edge_extension, targetBbox=target_bbox
                    )
                    print(
                        f"SAM strict bbox edge continuation layer={layer_ids[i]}: "
                        f"effective={target_bbox} addedPixels={edge_extension.get('addedPixels')} "
                        f"sides={edge_extension.get('sides')}"
                    )
            mask = constrain_mask_to_bbox(mask, target_bbox)
            ownership_context = build_semantic_ownership_bboxes(
                layer_meta, context_layers or [], target_bbox, w, h
            )
            strict_context = [
                entry for entry in ownership_context
                if isinstance(entry, dict) and entry.get("strict_ownership")
            ]
            if strict_context:
                mask, strict_component_audit = remove_strictly_excluded_detached_components(
                    mask, build_exclude_mask(strict_context, w, h)
                )
                sam_diagnostic_event(
                    "strict_excluded_component_cleanup",
                    masks=[mask], audit=strict_component_audit,
                    targetBbox=target_bbox
                )
                if strict_component_audit.get("status") == "accepted":
                    print(
                        f"SAM strict context component cleanup layer={layer_ids[i]}: "
                        f"{strict_component_audit}"
                    )
            print(
                f"SAM spatial ownership output locked layer={layer_ids[i]} "
                f"bbox={target_bbox} inside={selected_inside_ratio:.3f}"
            )

        cleanup_candidates = None
        if quality and quality.get("postProcess"):
            cleanup_candidates = None

        if strategy_type == "completion_object":
            mask = constrain_mask_to_bbox(mask, target_bbox)
            full_scene_completion = bool(
                quality and quality.get("completionOutputMode") == "full_scene_sam"
            )
            if full_scene_completion:
                # A flat hard-edge completion has already passed the complete
                # silhouette gate. Morphology can bridge printed poster gaps
                # or erase intentional line-art details, so preserve SAM's
                # selected contour verbatim.
                print(
                    f"Completion full-scene mask preserved for {layer_ids[i]}: "
                    "topology_repair=skipped micro_gap_repair=skipped"
                )
            else:
                # Incremental/spatial completion may contain narrow raster
                # cracks. Repair only enclosed, geometry-supported gaps.
                target_area = max(1, bbox_area(target_bbox))
                mask_fill = int(np.count_nonzero(mask > 0.5)) / target_area
                if mask_fill < 0.84:
                    mask, topology_repair = recover_entity_topology_gaps(
                        mask,
                        target_bbox
                    )
                    mask = constrain_mask_to_bbox(mask, target_bbox)
                    print(
                        f"Completion topology recovery for {layer_ids[i]}: "
                        f"status={topology_repair['status']} "
                        f"components={topology_repair['components']} "
                        f"pixels={topology_repair['pixels']} "
                        f"fill={mask_fill:.3f}"
                    )
                mask, micro_repair = recover_micro_entity_gaps(mask, target_bbox)
                mask = constrain_mask_to_bbox(mask, target_bbox)
                if micro_repair["components"]:
                    print(
                        f"Completion micro-gap recovery for {layer_ids[i]}: "
                        f"components={micro_repair['components']} "
                        f"pixels={micro_repair['pixels']}"
                    )
        elif strategy_type in HARD_EDGE_STRATEGIES:
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
        # Flat completion uses its own canonical recovery. Spatial structural
        # completion keeps the established table/furniture repair path because
        # it is the policy that selected this object in the first pass.
        if (
            (not completion_layer or layer_policy["spatialCanonicalCompletion"]) and
            base_strategy_type in {"table", "furniture"}
        ):
            # Tables and textured furniture use the accepted SAM silhouette as
            # the source of truth. Do not run another completion pass before
            # matting: it can reinterpret floor, seams, or surface texture.
            mask = constrain_mask_to_bbox(mask, target_bbox)
            if completion_layer and base_strategy_type == "table":
                completion_occlusion = decode_completion_occlusion_mask(
                    layer_meta.get("completionOcclusionMask"),
                    w,
                    h
                )
                completion_foreground_context = decode_completion_foreground_context_mask(
                    layer_meta.get("completionForegroundContextMask"),
                    w,
                    h
                )
                mask, completion_recovered, completion_recovery_debug = recover_completion_mask_with_points(
                    img,
                    mask,
                    target_bbox,
                    layer_ids[i],
                    occlusion_mask=completion_occlusion,
                    foreground_context_mask=completion_foreground_context,
                    model_variant="l"
                )
                layer_meta["_completionRecoveryAudit"] = completion_recovery_debug
                print(
                    f"Completion table recovery for {layer_ids[i]}: "
                    f"status={completion_recovery_debug.get('status')} "
                    f"accepted={bool(completion_recovered)}"
                )
                if completion_recovered:
                    mask = constrain_mask_to_bbox(mask, target_bbox)
                # SAM-L can return a geometrically correct low-fill table with
                # tiny enclosed transparent islands.  For the post-inpaint
                # table only, repair supported cracks/holes before the safe
                # matte rasterizes the silhouette.  This is deliberately
                # bounded and does not run on initial extraction, furniture,
                # or any flat-object category.
                table_before_audit = mask_integrity_audit(mask, target_bbox)
                mask, table_topology_repair = recover_entity_topology_gaps(
                    mask,
                    target_bbox
                )
                mask = constrain_mask_to_bbox(mask, target_bbox)
                mask, table_micro_repair = recover_micro_entity_gaps(
                    mask,
                    target_bbox
                )
                mask = constrain_mask_to_bbox(mask, target_bbox)
                mask, table_residual_repair = recover_residual_entity_gaps(
                    mask,
                    target_bbox
                )
                mask = constrain_mask_to_bbox(mask, target_bbox)
                table_after_audit = mask_integrity_audit(mask, target_bbox)
                if (
                    table_topology_repair["components"] or
                    table_micro_repair["components"] or
                    table_residual_repair["components"] or
                    table_before_audit.get("enclosedHolePixels", 0) != table_after_audit.get("enclosedHolePixels", 0)
                ):
                    print(
                        f"Table completion gap repair for {layer_ids[i]}: "
                        f"holes={table_before_audit.get('enclosedHolePixels', 0)}->"
                        f"{table_after_audit.get('enclosedHolePixels', 0)} "
                        f"topology={table_topology_repair.get('pixels', 0)} "
                        f"micro={table_micro_repair.get('pixels', 0)} "
                        f"residual={table_residual_repair.get('pixels', 0)}"
                    )
            if strategy_type == "furniture":
                local_l_config = layer_policy.get("localLRefine") or {}
                local_l_refined = False
                if (
                    local_l_config.get("enabled") and
                    str(layer_meta.get("_samModelVariant", "b")).lower() == "b" and
                    not completion_layer
                ):
                    local_l_mask, local_l_refined = refine_mask_with_local_sam(
                        img,
                        mask,
                        target_bbox,
                        cleanup_mask=None,
                        strategy_type="furniture",
                        model_variant="l",
                        boundary_only=True,
                        refine_config=local_l_config
                    )
                    print(
                        f"Furniture local L ROI refine for {layer_ids[i]}: "
                        f"accepted={bool(local_l_refined)}"
                    )
                    if local_l_refined and np.any(local_l_mask > 0.5):
                        mask = constrain_mask_to_bbox(local_l_mask, target_bbox)
                if str(layer_meta.get("_samModelVariant", "b")).lower() == "l":
                    refined_mask, internal_refined, internal_debug = refine_furniture_mask_with_internal_points(
                        img,
                        mask,
                        target_bbox,
                        layer_ids[i],
                        model_variant="l"
                    )
                    print(
                        f"Furniture internal SAM refine for {layer_ids[i]}: "
                        f"status={internal_debug.get('status')} "
                        f"candidates={internal_debug.get('candidates', 0)} "
                        f"growth={internal_debug.get('candidatePixels', 0)} "
                        f"accepted={bool(internal_refined)}"
                    )
                    if internal_refined and np.any(refined_mask > 0.5):
                        mask = constrain_mask_to_bbox(refined_mask, target_bbox)
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
                layer_ids[i],
                strategy_type=strategy_type,
                model_variant=str(layer_meta.get("_samModelVariant", "b")).lower()
            )
            if completion_changed:
                mask = constrain_mask_to_bbox(mask, target_bbox)
        else:
            ab_skip_step = (
                os.environ.get("SAM_AB_HARD_PRODUCT_SKIP_STEP")
                if strategy_type == "hard_product" else None
            )
            if strategy_type == "hard_product" and hole_recovered:
                # The hole pass accepted an image-reviewed SAM continuation
                # clipped to the audited subject interior. Morphological
                # cleanup immediately afterward can undo that verified fill
                # and shave supported outer-contour pixels, so preserve this
                # validated silhouette for alpha generation.
                print(
                    f"SAM hard-product cleanup skipped for {layer_ids[i]}: "
                    "verified interior-hole recovery"
                )
            elif (
                strategy_type == "hard_product" and
                os.environ.get("SAM_AB_SKIP_HARD_PRODUCT_CLEANUP") == "1"
            ):
                print(f"SAM A/B hard-product cleanup skipped for {layer_ids[i]}")
            elif (
                strategy_type == "hard_product" and
                has_verified_compound_silhouette(layer_meta) and
                ab_skip_step not in {"components", "close", "open", "holes"}
            ):
                # Compound discovery has already compared B peers and SAM
                # growth against target ownership. Re-running generic cleanup
                # can shave off exactly the low-fill outer contour that passed
                # that review, so retain the verified silhouette for alpha.
                print(
                    f"SAM hard-product cleanup skipped for {layer_ids[i]}: "
                    "verified compound silhouette"
                )
            else:
                if ab_skip_step in {"components", "close", "open", "holes"}:
                    print(f"SAM A/B hard-product cleanup skip={ab_skip_step} for {layer_ids[i]}")
                else:
                    ab_skip_step = None
                cleaned_mask = cleanup_mask(mask, target_bbox, skip_step=ab_skip_step)
                if np.any(cleaned_mask > 0.5):
                    mask = cleaned_mask

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
        full_scene_completion = bool(
            completion_layer and quality and
            quality.get("completionOutputMode") == "full_scene_sam"
        )
        allow_local_refine = (
            layer_policy["allowLocalRefine"] and
            not full_scene_completion and
            not layer_policy.get("boundaryRecovery", {}).get("enabled")
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
            protected_hard_product = strategy_type == "hard_product" and not completion_layer
            refined_mask, local_refined = refine_mask_with_local_sam(
                img,
                mask,
                target_bbox,
                cleanup_mask=refine_cleanup_mask,
                strategy_type=quality.get("strategy") if quality else None,
                boundary_only=protected_hard_product,
                refine_config={
                    "bandRatio": 0.024,
                    "minCorePreserve": 0.995,
                    "minBaselinePreserve": 0.985,
                    "minCandidateOverlap": 0.94,
                    "maxChangedRatio": 0.16,
                    "maxRemovedRatio": 0.015,
                    "maxAddedRatio": 0.12,
                } if protected_hard_product else None,
            )
            if local_refined and np.any(refined_mask > 0.5):
                candidate = cleanup_mask(refined_mask, target_bbox, strategy_type=quality.get("strategy") if quality else None)
                candidate = constrain_mask_to_bbox(candidate, target_bbox)
                if protected_hard_product:
                    accepted, audit = audit_local_refine_candidate(mask, candidate, target_bbox, strategy_type)
                    local_refined = accepted
                    if quality is not None:
                        quality["localRefineGuard"] = audit
                    print(f"SAM local refine guard layer={layer_ids[i]}: {audit}")
                if local_refined:
                    mask = candidate
            if quality and quality.get("strategy") in HARD_EDGE_STRATEGIES:
                refined_area = int(np.count_nonzero(mask > 0.5)) if local_refined else initial_mask_area
                print(
                    f"Hard-edge local refine for {layer_ids[i]}: "
                    f"accepted={bool(local_refined)} area={initial_mask_area}->{refined_area}"
                )

        # Completion and initial extraction share the same matte policy. The
        # phase changes candidate selection, not the alpha treatment of a
        # proven table, furniture, or soft-edge silhouette.
        matte_strategy_type = layer_policy["matteType"]
        ownership_boundary_config = layer_policy.get("ownershipBoundaryRepair") or {}
        if engine_name.startswith("sam") and ownership_boundary_config.get("enabled"):
            ownership_context = build_semantic_ownership_bboxes(
                layer_meta, context_layers or [], target_bbox, w, h
            )
            sibling_context = [
                entry for entry in ownership_context
                if isinstance(entry, dict) and entry.get("spatial_ownership_guard")
            ]
            repaired_mask, ownership_boundary_audit = recover_safe_outer_boundary(
                img,
                mask,
                target_bbox,
                blocked_additions_mask=(
                    owned_consensus_blocked if owned_consensus_blocked is not None
                    else build_exclude_mask(sibling_context, w, h)
                ),
                config=ownership_boundary_config
            )
            if ownership_boundary_audit.get("status") == "accepted":
                mask = constrain_mask_to_bbox(repaired_mask, target_bbox)
            if quality is not None:
                quality["ownershipBoundaryRepair"] = ownership_boundary_audit
            print(
                f"SAM ownership boundary repair layer={layer_ids[i]}: "
                f"{ownership_boundary_audit}"
            )
            sam_diagnostic_event(
                "ownership_boundary_repair",
                masks=[mask], audit=ownership_boundary_audit,
                targetBbox=target_bbox, contextBboxes=sibling_context
            )
        sam_diagnostic_event("postprocess_mask", masks=[mask], targetBbox=target_bbox)
        boundary_matte_accepted = False
        boundary_matte_config = layer_policy.get("boundaryMatte") or {}
        if engine_name.startswith("sam") and boundary_matte_config.get("enabled"):
            context_boxes = build_semantic_ownership_bboxes(
                layer_meta, context_layers or [], target_bbox, w, h
            )
            reviewed_mask, boundary_matte_audit = review_hard_boundary(
                img, mask, target_bbox,
                context_mask=build_exclude_mask(context_boxes, w, h),
                config=boundary_matte_config
            )
            boundary_matte_accepted = boundary_matte_audit["status"] == "accepted"
            if boundary_matte_accepted and owned_consensus_evidence is not None:
                boundary_matte_accepted, peer_review = audit_consensus_boundary_review(
                    mask, reviewed_mask, owned_consensus_evidence
                )
                boundary_matte_audit["candidateConsensusReview"] = peer_review
                if not boundary_matte_accepted:
                    boundary_matte_audit["status"] = "rejected"
                    boundary_matte_audit["reason"] = "candidate_consensus_disagreement"
            if boundary_matte_accepted:
                mask = reviewed_mask
            if quality is not None:
                quality["boundaryMatte"] = boundary_matte_audit
            print(f"SAM boundary matte layer={layer_ids[i]}: {boundary_matte_audit}")
            if owned_consensus_base is not None:
                mask = np.where(
                    owned_consensus_blocked,
                    owned_consensus_base,
                    mask > 0.5,
                ).astype(np.float32)
            sam_diagnostic_event(
                "boundary_matte", masks=[mask], audit=boundary_matte_audit,
                targetBbox=target_bbox, contextBboxes=context_boxes
            )
        if (
            engine_name.startswith("sam") and
            matte_strategy_type == "hard_product" and
            str(layer_meta.get("semanticType", "")).lower() == "decor_plant"
        ):
            mask, gap_audit = remove_flat_background_gaps(img, mask, target_bbox)
            if quality is not None:
                quality["backgroundGapCleanup"] = gap_audit
            print(f"SAM background gap cleanup layer={layer_ids[i]}: {gap_audit}")
            sam_diagnostic_event(
                "background_gap_cleanup", masks=[mask], audit=gap_audit,
                targetBbox=target_bbox
            )
        if boundary_matte_accepted:
            # The reviewed contour owns alpha. Running generic GrabCut again
            # would expand it and potentially restore the rejected background.
            alpha_mask = build_hard_edge_alpha(
                mask, target_bbox,
                preserve_accepted_mask=matte_strategy_type != "food_product"
            )
            alpha_mask = np.where(mask > .5, alpha_mask, 0).astype(np.uint8)
        elif full_scene_completion:
            # GrabCut is a color-model classifier. On a poster completion it
            # mistakes headline fills and strong ink outlines for foreground
            # while cutting holes in skin/shirt gradients. The accepted SAM-L
            # mask is the authoritative alpha in this narrowly gated route.
            alpha_mask = build_hard_edge_alpha(mask, target_bbox)
            alpha_mask = np.where(mask > 0.5, alpha_mask, 0).astype(np.uint8)
        elif (
            engine_name.startswith("sam") and
            bool((layer_policy.get("compoundComponentDiscovery") or {}).get("enabled")) and
            matte_strategy_type in HARD_EDGE_STRATEGIES
        ):
            # Discovery has already gated ownership and components. A second
            # color-model matte can cut plate rims or restore poster panels,
            # so the accepted SAM silhouette is authoritative here only.
            alpha_mask = build_hard_edge_alpha(mask, target_bbox)
            alpha_mask = np.where(mask > 0.5, alpha_mask, 0).astype(np.uint8)
        else:
            alpha_mask = generate_alpha_matte(
                img,
                mask,
                target_bbox,
                cleanup_mask=matte_cleanup_mask,
                strategy_type=matte_strategy_type,
                label_cleanup_mask=label_cleanup_mask,
                flat_cleanup_mask=flat_cleanup_mask
            )
        if matte_strategy_type == "soft_edge":
            alpha_mask = build_soft_edge_alpha(img, mask, target_bbox)
        if matte_strategy_type == "furniture":
            alpha_mask = build_hard_edge_alpha(mask, target_bbox, preserve_accepted_mask=True)
            # Keep antialiasing inside the accepted furniture silhouette only.
            # Rasterizing a contour can otherwise place a fractional pixel on
            # the far side of a small hole or a tight concavity.
            alpha_mask = np.where(mask > 0.5, alpha_mask, 0).astype(np.uint8)
        if matte_strategy_type == "food_product":
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
        if (
            engine_name.startswith("sam") and
            bool((layer_policy.get("compoundComponentDiscovery") or {}).get("enabled")) and
            matte_strategy_type in HARD_EDGE_STRATEGIES
        ):
            # The supersampled contour rasterizer can create fractional pixels
            # just outside the accepted SAM silhouette. Compound plates often
            # sit on a similarly colored panel, so those pixels become a
            # visible background halo. Keep alpha strictly inside SAM's
            # ownership mask while preserving all accepted plate pixels.
            alpha_before_guard = int(np.count_nonzero(alpha_mask > 0))
            alpha_mask = np.where(mask > 0.5, alpha_mask, 0).astype(np.uint8)
            alpha_removed = alpha_before_guard - int(np.count_nonzero(alpha_mask > 0))
            if alpha_removed > 0:
                print(
                    f"SAM compound alpha ownership guard layer={layer_ids[i]} "
                    f"removedOutsideMask={alpha_removed}"
                )
        sam_diagnostic_event("alpha_before_clip", alpha=alpha_mask, targetBbox=target_bbox)
        alpha_mask = np.asarray(
            constrain_mask_to_bbox(alpha_mask.astype(np.float32), target_bbox),
            dtype=np.uint8
        )
        if owned_consensus_base is not None:
            alpha_mask = np.where(
                owned_consensus_blocked & ~owned_consensus_base,
                0,
                alpha_mask,
            ).astype(np.uint8)
        sam_diagnostic_event("final_alpha", alpha=alpha_mask, targetBbox=target_bbox)
        if matte_strategy_type == "table":
            final_mask_bbox = mask_bbox(mask > 0.5)
            alpha_bbox = mask_bbox(alpha_mask > 0)
            print(
                f"Table cutout geometry for {layer_ids[i]}: "
                f"output_bbox={target_bbox} mask_bbox={final_mask_bbox} "
                f"alpha_bbox={alpha_bbox} output_size="
                f"({target_bbox[2] - target_bbox[0]},{target_bbox[3] - target_bbox[1]})"
            )
        if matte_strategy_type in {"table", "furniture"}:
            opaque_pixels = int(np.count_nonzero(alpha_mask >= 245))
            edge_pixels = int(np.count_nonzero((alpha_mask > 0) & (alpha_mask < 245)))
            print(
                f"{strategy_type.capitalize()} SAM safe matte for {layer_ids[i]}: "
                f"preservedMaskPixels={int(np.count_nonzero(mask > 0.5))} "
                f"opaque={opaque_pixels} antialiasedEdge={edge_pixels}"
            )
        elif matte_strategy_type in HARD_EDGE_STRATEGIES:
            opaque_pixels = int(np.count_nonzero(alpha_mask >= 245))
            edge_pixels = int(np.count_nonzero((alpha_mask > 0) & (alpha_mask < 245)))
            print(
                f"Hard-edge alpha for {layer_ids[i]}: "
                f"opaque={opaque_pixels} antialiasedEdge={edge_pixels}"
            )
        output_img = img
        if matte_strategy_type == "soft_edge":
            output_img = despill_soft_edge_image(
                img,
                alpha_mask,
                target_bbox,
                context_bbox=expand_bbox(*target_bbox, w, h)
            )
        elif (
            matte_strategy_type in HARD_EDGE_STRATEGIES and
            matte_strategy_type not in {"table", "furniture"} and
            not full_scene_completion
        ):
            output_img = despill_hard_edge_image(img, alpha_mask, target_bbox)
        if quality is None:
            quality = {}
        completion_recovery_audit = layer_meta.get("_completionRecoveryAudit")
        if (
            completion_layer and
            layer_policy["spatialCanonicalCompletion"] and
            isinstance(completion_recovery_audit, dict)
        ):
            completion_recovery_audit, optional_recovery_promoted = (
                promote_optional_completion_recovery(
                    quality,
                    completion_recovery_audit
                )
            )
            if optional_recovery_promoted:
                # The selected SAM-L candidate has already passed the
                # completion-specific structural gates.  Make the runtime
                # decision explicit as well; leaving an earlier generic
                # low-fill/score hold in place would reproduce the same
                # frontend rejection under a different issue name.
                quality["runtimeAction"] = "accept"
                quality["shouldGenerateRuntimeLayer"] = True
                quality["needsHigherPrecision"] = False
                quality["status"] = "ok"
                quality["issues"] = [
                    issue for issue in (quality.get("issues") or [])
                    if issue not in {
                        "completion_recovery_not_verified",
                        "low_quality_status"
                    }
                ]
                print(
                    f"Completion table recovery for {layer_ids[i]}: "
                    "optional point recovery missed; selected SAM-L candidate "
                    "verified, continuing"
                )
            quality["completionRecoveryAudit"] = completion_recovery_audit
            if not str(completion_recovery_audit.get("status", "")).startswith("accepted:"):
                quality["runtimeAction"] = "hold"
                quality["shouldGenerateRuntimeLayer"] = False
                quality["issues"] = list(quality.get("issues") or [])
                if "completion_recovery_not_verified" not in quality["issues"]:
                    quality["issues"].append("completion_recovery_not_verified")
        quality["postProcess"] = {
            "maskCleanup": True,
            "localRefine": bool(local_refined),
            "matting": (
                "sam_safe_matte"
                if boundary_matte_accepted or full_scene_completion or matte_strategy_type in {"table", "furniture"}
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
            if quality is not None:
                quality["finalMaskAudit"] = mask_integrity_audit(mask, target_bbox)
                quality["layerTimingMs"] = round(
                    (time.perf_counter() - layer_started_at) * 1000,
                    1
                )
            cutouts.append(cutout)
            print(
                f"SAM layer timing layer={layer_ids[i]} "
                f"durationMs={(time.perf_counter() - layer_started_at) * 1000:.1f} "
                "status=ok"
            )
        else:
            print(
                f"SAM layer timing layer={layer_ids[i]} "
                f"durationMs={(time.perf_counter() - layer_started_at) * 1000:.1f} "
                "status=empty_cutout"
            )

    return cutouts
