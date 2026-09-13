"""Deterministic, model-free backend regression checks."""

import base64
import cv2
import numpy as np

from sam_runtime import *
from segmentation_policy import *
from mask_ops import *
from candidate_selection import *
from matte_ops import *
from segmentation_core import *
from segmentation_primitives import build_boundary_negative_points

def run_deterministic_self_tests():
    """Run model-free regression checks for policy, gates, and mask audits."""
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    # Model configuration must be owned by sam_runtime rather than the
    # process entrypoint. This catches the most common regression when the
    # backend is imported by uvicorn instead of launched as main.py.
    check("runtime_model_cache_configured", bool(MODEL_CACHE_DIR))
    check("runtime_sam_b_path_configured", bool(SAM_B_MODEL_PATH))
    check("runtime_sam_l_path_configured", bool(SAM_L_MODEL_PATH))
    check(
        "runtime_oss_model_url_resolution",
        resolve_model_url("oss://bucket/models/sam_l.pt").endswith("/models/sam_l.pt")
    )
    probe_mask = np.zeros((32, 32), dtype=np.uint8)
    probe_mask[4:28, 4:28] = 1
    check(
        "cross_module_boundary_prompt_dependency",
        len(build_boundary_negative_points(probe_mask, [2, 2, 30, 30])) > 0
    )

    table_meta = {
        "name": "大理石玄关桌",
        "semanticType": "other",
        "designRole": "scene_object",
        "extractionProfile": "multi_part_hard_object"
    }
    table = get_layer_strategy(table_meta)
    check("table_spatial_profile", table["profile"] == "spatial_design")
    check("table_hard_edge_feature", "hard_edge" in table["features"])
    check("table_support_feature", "supports" in table["features"])
    check("table_initial_phase", table["phase"] == "initial")
    table_policy = resolve_mask_policy(table_meta, "publish")
    check("table_policy_is_spatial", table_policy["profile"] == "spatial_design")
    check("table_policy_keeps_matte", table_policy["matteType"] == "table")
    check("table_policy_preserves_table_size", table_policy["samImgSize"] == 1024)

    transparent_table_meta = {
        **table_meta,
        "name": "透明玻璃玄关桌"
    }
    transparent_table = get_layer_strategy(transparent_table_meta)
    transparent_table_policy = resolve_mask_policy(transparent_table_meta, "publish")
    check("transparent_table_stays_hard_edge", "soft_edge" not in transparent_table["features"])
    check("transparent_table_keeps_safe_matte", transparent_table_policy["matteType"] == "table")

    curtain_meta = {"name": "半透明窗纱", "semanticType": "other"}
    curtain = get_layer_strategy(curtain_meta)
    check("soft_edge_spatial_profile", curtain["profile"] == "spatial_design")
    check("soft_edge_feature", "soft_edge" in curtain["features"])
    check("soft_edge_not_hard", "hard_edge" not in curtain["features"])
    curtain_policy = resolve_mask_policy(curtain_meta, "publish")
    check("soft_edge_policy_uses_soft_matte", curtain_policy["matteType"] == "soft_edge")
    check("soft_edge_policy_keeps_high_res", curtain_policy["samImgSize"] == SOFT_EDGE_SAM_IMGSZ)

    food_meta = {
        "name": "一盘复合食物",
        "semanticType": "product_food",
        "designRole": "product_image"
    }
    food = get_layer_strategy(food_meta)
    check("food_flat_profile", food["profile"] == "flat_design")
    check("food_compound_feature", "compound" in food["features"])
    food_policy = resolve_mask_policy(food_meta, "publish")
    check("food_policy_uses_compound_selector", food_policy["selector"] == "compound_food")
    check("food_policy_is_not_completion", not food_policy["completion"])
    check("food_policy_keeps_b_first", not food_policy["allowModelEscalation"])

    verified_occluder_policy = resolve_mask_policy({
        **food_meta,
        "completionOccluder": True
    }, "completion")
    check("completion_occluder_policy_is_initial", not verified_occluder_policy["completion"])
    check("completion_occluder_policy_allows_l_review", verified_occluder_policy["allowModelEscalation"])
    check("completion_occluder_policy_marks_execution_role", verified_occluder_policy["completionOccluder"])

    composite_occluder_policy = resolve_mask_policy({
        **verified_occluder_policy,
        "compositeRole": "composite_group",
        "childLayerIds": ["instance_a", "instance_b"]
    }, "completion")
    check("composite_occluder_keeps_one_layer_contract", composite_occluder_policy["compositeInstanceUnion"])
    composite_primary = np.zeros((100, 160), dtype=np.float32)
    composite_primary[30:55, 25:55] = 1
    composite_peer = np.zeros((100, 160), dtype=np.float32)
    composite_peer[30:55, 90:120] = 1
    composite_union, composite_debug = merge_composite_instance_masks(
        composite_primary,
        [composite_primary, composite_peer],
        0,
        [10, 10, 140, 140]
    )
    check("composite_occluder_unions_detached_instances", composite_debug["peers"] == 1 and np.count_nonzero(composite_union) > np.count_nonzero(composite_primary))

    soft_occluder_policy = resolve_mask_policy({
        "name": "半透明窗纱",
        "semanticType": "other",
        "completionOccluder": True
    }, "completion")
    check("soft_occluder_keeps_existing_route", not soft_occluder_policy["completionOccluder"])

    poster_policy = resolve_mask_policy({
        "name": "主视觉波普人物",
        "semanticType": "other",
        "designRole": "product_image"
    }, "publish")
    check("poster_policy_is_flat", poster_policy["profile"] == "flat_design")
    check("poster_policy_allows_quality_escalation", poster_policy["allowModelEscalation"])

    mislabeled_person_policy = resolve_mask_policy({
        "name": "酸表情人物插画",
        "semanticType": "product_food",
        "designRole": "product_image",
        "extractionProfile": "layout_embedded_product"
    }, "completion")
    check("person_overrides_coarse_food_label", mislabeled_person_policy["baseStrategyType"] == "hard_product")
    check("person_avoids_compound_food_selector", mislabeled_person_policy["selector"] != "compound_food")

    completion_meta = {
        "id": "completion-scene-sam-person-1",
        "name": "插画人物",
        "semanticType": "product_food",
        "designRole": "product_image",
        "completionSegmentation": True
    }
    completion = get_layer_strategy(completion_meta)
    check("completion_phase", completion["phase"] == "completion")
    check("completion_generic_selector", completion["selectionType"] == "completion_object")
    check("completion_person_uses_hard_entity_mode", completion["baseStrategyType"] == "hard_product")
    check("completion_uses_canonical_threshold", completion["max_fill"] == 0.90)
    check("completion_uses_canonical_attachments", completion["max_masks"] == 8)
    completion_policy = resolve_mask_policy(completion_meta, "completion")
    check("completion_policy_is_generic", completion_policy["selector"] == "generic_completion")
    check("completion_policy_has_occlusion", "occluded" in completion_policy["features"])
    check("completion_is_not_spatial_canonical", not completion_policy["spatialCanonicalCompletion"])
    check("completion_policy_allows_model_escalation", completion_policy["allowModelEscalation"])
    check("completion_policy_allows_local_refine", completion_policy["allowLocalRefine"])
    check("completion_policy_uses_canonical_matte", completion_policy["matteType"] == "completion_object")
    check("completion_policy_uses_stable_imgsz", completion_policy["samImgSize"] == 1024)

    furniture_completion_meta = {
        "id": "completion-scene-sam-table-1",
        "name": "被遮挡的玄关桌",
        "semanticType": "other",
        "designRole": "scene_object",
        "extractionProfile": "multi_part_hard_object",
        "completionSegmentation": True
    }
    furniture_completion = get_layer_strategy(furniture_completion_meta)
    furniture_completion_policy = resolve_mask_policy(furniture_completion_meta, "completion")
    check("furniture_completion_keeps_table_strategy", furniture_completion["type"] == "table")
    check("furniture_completion_keeps_table_selector", furniture_completion["selectionType"] == "table")
    check("furniture_completion_keeps_table_threshold", furniture_completion["max_fill"] == 0.66)
    check("furniture_completion_marks_spatial_canonical", furniture_completion["spatialCanonicalCompletion"])
    check("furniture_completion_policy_is_spatial_canonical", furniture_completion_policy["selector"] == "spatial_canonical_completion")
    check("furniture_completion_keeps_safe_matte", furniture_completion_policy["matteType"] == "table")
    check("furniture_completion_skips_generic_local_refine", not furniture_completion_policy["allowLocalRefine"])

    ambiguous_console_meta = {
        **furniture_completion_meta,
        "name": "透明石材玄关端景台",
        "semanticType": "hard_product"
    }
    ambiguous_console = get_layer_strategy(ambiguous_console_meta)
    ambiguous_console_policy = resolve_mask_policy(ambiguous_console_meta, "completion")
    check("ambiguous_console_keeps_table_base", ambiguous_console["baseStrategyType"] == "table")
    check("ambiguous_console_stays_hard_edge", "soft_edge" not in ambiguous_console["features"])
    check("ambiguous_console_uses_table_matte", ambiguous_console_policy["matteType"] == "table")

    furniture_item_meta = {
        **furniture_completion_meta,
        "id": "completion-scene-sam-chair-1",
        "name": "被遮挡的休闲椅"
    }
    furniture_item_policy = resolve_mask_policy(furniture_item_meta, "completion")
    check("furniture_item_keeps_furniture_selector", furniture_item_policy["selectionType"] == "furniture")
    check("furniture_item_keeps_furniture_matte", furniture_item_policy["matteType"] == "furniture")
    check("furniture_item_keeps_initial_imgsz", furniture_item_policy["samImgSize"] == HARD_EDGE_SAM_IMGSZ)

    consistent, reason = completion_observation_is_consistent(
        100,
        0.985,
        0.67,
        0.96,
        base_strategy_type="table",
        spatial=True
    )
    check("spatial_completion_accepts_consistent_observation", consistent and reason == "spatial_observation_anchor")
    consistent, reason = completion_observation_is_consistent(
        100,
        0.966,
        0.194,
        0.96,
        base_strategy_type="table",
        spatial=True
    )
    check("spatial_completion_rejects_background_expansion", not consistent and reason == "spatial_completion_observation_iou")

    image_size = 100
    target_bbox = [20, 20, 80, 80]
    observed = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    observed[25:75, 25:75, 3] = 255
    ok, encoded = cv2.imencode(".png", observed)
    check("observation_mask_encoded", bool(ok))
    completion_meta["completionObservationMask"] = (
        "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    )
    candidate = np.zeros((image_size, image_size), dtype=np.float32)
    candidate[24:76, 24:76] = 1.0
    selected_mask, selected_count, quality = select_and_merge_masks(
        [candidate],
        target_bbox,
        image_size,
        image_size,
        completion_meta,
        [],
        quality_profile="completion"
    )
    check("completion_candidate_selected", selected_mask is not None and selected_count == 1)
    check("completion_quality_phase", quality and quality["phase"] == "completion")
    check("completion_quality_generic_strategy", quality and quality["strategy"] == "completion_object")
    check("completion_quality_base_mode", quality and quality["baseStrategyType"] == "hard_product")
    check("completion_quality_policy_version", quality and quality["policyVersion"] == MASK_POLICY_VERSION)
    check("completion_quality_mask_audit", quality and quality["maskAudit"]["status"] == "ok")

    direct_completion_meta = {
        **completion_meta,
        "_completionFullSceneMask": True
    }
    _, _, direct_quality = select_and_merge_masks(
        [candidate],
        target_bbox,
        image_size,
        image_size,
        direct_completion_meta,
        [],
        quality_profile="completion"
    )
    check(
        "flat_completion_exposes_full_scene_output_mode",
        direct_quality and direct_quality["completionOutputMode"] == "full_scene_sam"
    )

    occlusion_mask = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    occlusion_mask[70:75, 45:75, 3] = 255
    ok, encoded = cv2.imencode(".png", occlusion_mask)
    check("completion_occlusion_mask_encoded", bool(ok))
    occluded_completion_meta = {
        **completion_meta,
        "completionOcclusionMask": (
            "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        )
    }
    visible_only_candidate = np.zeros((image_size, image_size), dtype=np.float32)
    visible_only_candidate[25:70, 25:75] = 1.0
    selected_mask, selected_count, _ = select_and_merge_masks(
        [visible_only_candidate],
        target_bbox,
        image_size,
        image_size,
        occluded_completion_meta,
        [],
        quality_profile="completion"
    )
    check("completion_rejects_visible_only_candidate", selected_mask is None and selected_count == 0)

    spatial_completion_meta = {
        **furniture_completion_meta,
        "completionObservationMask": completion_meta["completionObservationMask"]
    }
    spatial_candidate = np.zeros((image_size, image_size), dtype=np.float32)
    spatial_candidate[30:70, 30:70] = 1.0
    selected_mask, selected_count, spatial_quality = select_and_merge_masks(
        [spatial_candidate],
        target_bbox,
        image_size,
        image_size,
        spatial_completion_meta,
        [],
        quality_profile="completion"
    )
    check("spatial_completion_accepts_consistent_candidate", selected_mask is not None and selected_count == 1)
    check("spatial_completion_keeps_table_selection", spatial_quality and spatial_quality["strategy"] == "table")
    check(
        "spatial_completion_does_not_expose_full_scene_output_mode",
        spatial_quality and spatial_quality["completionOutputMode"] != "full_scene_sam"
    )
    # A separately extracted foreground object remaining in the completed
    # scene must not be absorbed by a spatial table's canonical alpha.
    foreground_context = [{
        "id": "completion-foreground-vase",
        "name": "花瓶",
        "completionForegroundContext": True,
        "bbox": [300, 300, 700, 700]
    }]
    selected_mask, selected_count, foreground_quality = select_and_merge_masks(
        [spatial_candidate],
        target_bbox,
        image_size,
        image_size,
        spatial_completion_meta,
        foreground_context,
        quality_profile="completion"
    )
    check(
        "spatial_completion_rejects_foreground_context_contamination",
        selected_mask is None and selected_count == 0
    )
    anchored_spatial_candidate = np.zeros((image_size, image_size), dtype=np.float32)
    anchored_spatial_candidate[24:76, 24:76] = 1.0
    anchored_mask, anchored_count, anchored_quality = select_and_merge_masks(
        [anchored_spatial_candidate],
        target_bbox,
        image_size,
        image_size,
        spatial_completion_meta,
        foreground_context,
        quality_profile="completion"
    )
    check(
        "spatial_completion_anchor_survives_context_overlap",
        anchored_mask is not None and anchored_count == 1 and
        anchored_quality and
        anchored_quality["debugCandidates"][0]["spatialCompletionAnchor"] is True
    )

    # The prompt box may deliberately include an occluder and therefore be
    # wider than the visible target.  Candidate identity must remain anchored
    # to the observed silhouette rather than requiring high overlap with that
    # union prompt box.
    wide_observation = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    wide_observation[30:55, 20:60, 3] = 255
    ok, wide_encoded = cv2.imencode(".png", wide_observation)
    check("wide_prompt_observation_encoded", bool(ok))
    wide_spatial_meta = {
        **spatial_completion_meta,
        "completionObservationMask": (
            "data:image/png;base64," + base64.b64encode(wide_encoded.tobytes()).decode("ascii")
        )
    }
    wide_prompt_candidate = np.zeros((image_size, image_size), dtype=np.float32)
    wide_prompt_candidate[30:80, 20:60] = 1.0
    wide_occlusion = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    wide_occlusion[55:80, 20:60, 3] = 255
    ok, wide_occlusion_encoded = cv2.imencode(".png", wide_occlusion)
    check("wide_prompt_occlusion_encoded", bool(ok))
    wide_spatial_meta["completionOcclusionMask"] = (
        "data:image/png;base64," + base64.b64encode(wide_occlusion_encoded.tobytes()).decode("ascii")
    )
    wide_selected, wide_count, wide_quality = select_and_merge_masks(
        [wide_prompt_candidate],
        [10, 10, 95, 90],
        image_size,
        image_size,
        wide_spatial_meta,
        [],
        quality_profile="completion"
    )
    check(
        "spatial_completion_uses_observed_anchor_for_wide_prompt",
        wide_selected is not None and wide_count == 1 and
        wide_quality and
        wide_quality["debugCandidates"][0]["spatialCompletionAnchor"] is True and
        wide_quality["debugCandidates"][0]["bboxOverlap"] < 0.70 and
        wide_quality["debugCandidates"][0]["anchorBboxOverlap"] >= 0.70 and
        wide_quality["debugCandidates"][0]["completionRecoveryPixels"] > 0
    )

    cracked = candidate.copy()
    cracked[48:52, 48:52] = 0
    crack_audit = mask_integrity_audit(cracked, target_bbox)
    check("mask_audit_detects_enclosed_hole", crack_audit["enclosedHolePixels"] > 0)
    rejected = build_quality_gate(
        0.0,
        -1.0,
        0.0,
        [],
        "completion_object",
        policy=completion_policy
    )
    check("quality_gate_rejects_empty_candidate", not rejected["shouldGenerateRuntimeLayer"])
    check("quality_gate_reports_empty_mask", "no_selected_mask" in rejected["issues"])

    promoted_audit, promoted = promote_optional_completion_recovery(
        {
            "debugCandidates": [{
                "selected": True,
                "candidate": True,
                "spatialCompletionAnchor": True,
                "inside": 1.0,
                "bboxOverlap": 0.64,
                "anchorBboxOverlap": 0.991,
                "observationRecall": 0.92,
                "completionRecoveryPixels": 1800
            }]
        },
        {"status": "rejected:no_safe_completion_candidate"}
    )
    check(
        "optional_point_recovery_does_not_hold_verified_candidate",
        promoted and promoted_audit["status"] == "accepted:selected_candidate_recovery"
    )
    rejected_audit, not_promoted = promote_optional_completion_recovery(
        {
            "debugCandidates": [{
                "selected": True,
                "candidate": True,
                "inside": 0.71,
                "bboxOverlap": 0.28,
                "observationRecall": 0.31,
                "completionRecoveryPixels": 0
            }]
        },
        {"status": "rejected:no_safe_completion_candidate"}
    )
    check(
        "optional_point_recovery_keeps_unverified_candidate_held",
        not not_promoted and rejected_audit["status"].startswith("rejected:")
    )

    # Hidden-only multimask rescue is model-independent: an alternate mask may
    # contribute a sufficiently large target component only inside the verified
    # occlusion band and outside known foreground ownership context.  A broad
    # scene/background component must remain excluded.
    rescue_size = 96
    rescue_bbox = [16, 16, 80, 80]
    rescue_merged = np.zeros((rescue_size, rescue_size), dtype=bool)
    rescue_merged[24:50, 24:72] = True
    rescue_occlusion = np.zeros_like(rescue_merged)
    rescue_occlusion[48:74, 20:76] = True
    rescue_context = np.zeros_like(rescue_merged)
    rescue_context[54:70, 44:56] = True
    rescue_alternate = rescue_merged.copy()
    rescue_alternate[48:66, 28:42] = True
    rescue_alternate[54:70, 44:56] = True
    rescue_alternate[60:82, 76:92] = True
    rescue_result, rescue_pixels, rescue_radius = recover_completion_hidden_delta(
        rescue_merged,
        [{
            "metrics": {"index": 0},
            "mask": rescue_merged,
            "debug": {"completionRecoveryPixels": 0}
        }, {
            "metrics": {"index": 1},
            "mask": rescue_alternate,
            "debug": {
                "candidate": True,
                "spatialCompletionStructuralPeer": True,
                "completionRecoveryPixels": 1200
            }
        }],
        0,
        rescue_occlusion,
        rescue_merged,
        rescue_context,
        rescue_bbox,
        64 * 64,
    )
    check("hidden_rescue_accepts_occluded_target_delta", rescue_pixels >= 200)
    check("hidden_rescue_keeps_context_owned_pixels_out", not np.any(rescue_result & rescue_context))
    check("hidden_rescue_keeps_delta_inside_bbox", not np.any(
        rescue_result[:rescue_bbox[0]]
    ) and not np.any(rescue_result[rescue_bbox[2]:]) and not np.any(
        rescue_result[:, :rescue_bbox[1]]
    ) and not np.any(rescue_result[:, rescue_bbox[3]:]))
    check("hidden_rescue_reports_bounded_radius", 12 <= rescue_radius <= 36)

    rejected_result, rejected_pixels, _ = recover_completion_hidden_delta(
        rescue_merged,
        [{
            "metrics": {"index": 0},
            "mask": rescue_merged,
            "debug": {
                "candidate": True,
                "spatialCompletionStructuralPeer": True
            }
        }, {
            "metrics": {"index": 1},
            "mask": rescue_alternate,
            "debug": {
                "candidate": False,
                "rejectReason": "table_background_like",
                "completionRecoveryPixels": 1200
            }
        }],
        0,
        rescue_occlusion,
        rescue_merged,
        np.zeros_like(rescue_context),
        rescue_bbox,
        64 * 64,
    )
    check("hidden_rescue_rejects_background_like_alternate", rejected_pixels == 0 and np.array_equal(rejected_result, rescue_merged))

    print(f"SAM deterministic self-tests passed: {len(checks)} checks")
    return True
