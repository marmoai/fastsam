"""SAM candidate routing and cutout materialization pipeline."""

import cv2
import numpy as np

from sam_runtime import *
from segmentation_policy import *
from mask_ops import *
from candidate_selection import *
from matte_ops import *


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
    # A hard-product mask may be a printed illustration or a complex product
    # silhouette. When B explicitly escalated because of structural failure,
    # do not make that damaged B result veto a contained SAM-L candidate. The
    # same conservative containment/fill/hole checks used for furniture still
    # reject bbox-sized scene masks.
    direct_l_recovery = b_failed and strategy_type != "soft_edge"
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


def run_sam_l_with_retry(img, prompt_bbox, strategy_type, layer_name, policy=None):
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
                points=None,
                labels=None,
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
    original_target_bboxes=None,
    quality_profile="publish"
):
    h, w = img.shape[:2]
    cutouts = []

    for i, target_bbox in enumerate(pixel_bboxes):
        if i >= len(layer_ids):
            break

        layer_meta = layer_metas[i] if isinstance(layer_metas, list) and i < len(layer_metas) else {}
        layer_policy = resolve_mask_policy(layer_meta, quality_profile)
        completion_layer = is_completion_segmentation_layer(layer_meta)
        if completion_layer:
            print(
                f"Completion SAM selection enabled for {layer_ids[i]}: "
                f"completionSegmentation={bool((layer_meta or {}).get('completionSegmentation'))} "
                f"layerId={layer_meta.get('id') or layer_meta.get('layerId')}"
            )
        candidate_masks = mask_provider(target_bbox, layer_meta, i)
        if candidate_masks is None or len(candidate_masks) == 0:
            continue
        mask, selected_count, quality = select_and_merge_masks(
            candidate_masks,
            target_bbox,
            w,
            h,
            layer_meta,
            context_layers,
            quality_profile=quality_profile
        )
        if quality is not None:
            quality["qualityProfile"] = normalize_sam_quality_profile(quality_profile)
            quality["policyVersion"] = layer_policy["version"]
            quality["samModelVariant"] = str(
                (layer_meta or {}).get("_samModelVariant", "b")
            ).lower()
            quality["samSelectionRoute"] = str(
                (layer_meta or {}).get("_samSelectionRoute", "b_first")
            )
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
            continue

        strategy_type = quality.get("strategy") if quality else None
        base_strategy_type = quality.get("baseStrategyType", strategy_type) if quality else strategy_type
        if not layer_meta.get("completionSegmentation"):
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
        full_scene_completion = bool(
            completion_layer and quality and
            quality.get("completionOutputMode") == "full_scene_sam"
        )
        allow_local_refine = layer_policy["allowLocalRefine"] and not full_scene_completion
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

        # Completion and initial extraction share the same matte policy. The
        # phase changes candidate selection, not the alpha treatment of a
        # proven table, furniture, or soft-edge silhouette.
        matte_strategy_type = layer_policy["matteType"]
        if full_scene_completion:
            # GrabCut is a color-model classifier. On a poster completion it
            # mistakes headline fills and strong ink outlines for foreground
            # while cutting holes in skin/shirt gradients. The accepted SAM-L
            # mask is the authoritative alpha in this narrowly gated route.
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
            alpha_mask = build_hard_edge_alpha(mask, target_bbox)
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
        alpha_mask = np.asarray(
            constrain_mask_to_bbox(alpha_mask.astype(np.float32), target_bbox),
            dtype=np.uint8
        )
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
                if full_scene_completion or matte_strategy_type in {"table", "furniture"}
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
            cutouts.append(cutout)

    return cutouts
