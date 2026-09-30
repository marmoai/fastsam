import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fastsam-backend"))

from candidate_selection import build_spatial_ownership_guard, build_semantic_ownership_bboxes
from matte_ops import build_sam_prompt_inputs


def atomic_layer(layer_id, name, bbox, parent="coffee_table_group", profile="layout_embedded_product"):
    return {
        "id": layer_id,
        "name": name,
        "bbox": bbox,
        "parentLayerId": parent,
        "extractionProfile": profile,
        "segmentationIntent": {
            "targetKind": "atomic_object",
            "bboxRole": "tight_subject",
            "confidence": 0.93,
            "excludedAdjacentObjects": [],
        },
    }


def test_guard_enables_only_for_named_nearby_spatial_atomic_sibling():
    target = atomic_layer("terrazzo", "Terrazzo coffee table", [716, 672, 831, 1000])
    sibling = atomic_layer("marble", "White marble coffee table", [784, 781, 871, 1000])
    target["segmentationIntent"]["excludedAdjacentObjects"] = [{
        "id": "marble", "name": "White marble coffee table", "bbox": sibling["bbox"],
    }]
    target_bbox = [463, 476, 690, 589]
    guard = build_spatial_ownership_guard(target, [target, sibling], target_bbox, 690, 709)
    assert guard["enabled"] is True
    assert guard["entries"][0]["strong"] is True
    assert guard["entries"][0]["spatial_ownership_guard"] is True
    target["_spatialOwnershipGuard"] = guard
    exclusions = build_semantic_ownership_bboxes(
        target, [target, sibling], target_bbox, 690, 709
    )
    assert any(item.get("spatial_ownership_guard") for item in exclusions)
    prompts = build_sam_prompt_inputs(
        target, [target, sibling], target_bbox, 690, 709
    )
    # Legacy guard helpers remain testable for diagnostics, but API routing
    # removes them.  An incomplete explicit intent must not reactivate the
    # ownership prompt path.
    assert prompts["segmentationIntentAudit"]["reason"] == "no_safe_ownership_evidence"


def test_guard_does_not_enable_for_flat_or_unmatched_context():
    target = atomic_layer("woman", "Screaming woman", [0, 0, 1000, 1000], parent="poster", profile="standard_object")
    target["segmentationIntent"]["excludedAdjacentObjects"] = [{"name": "Badge", "bbox": [20, 20, 80, 80]}]
    badge = atomic_layer("badge", "Badge", [20, 20, 80, 80], parent="poster", profile="standard_object")
    guard = build_spatial_ownership_guard(target, [target, badge], [0, 0, 690, 709], 690, 709)
    assert guard["enabled"] is False


def test_legacy_spatial_atomic_still_locks_output_without_prompting():
    chair = {
        "id": "front_chair", "name": "Front chair", "bbox": [629, 39, 893, 386],
        "parentLayerId": None, "compositeRole": "atomic_object",
        "semanticType": "furniture_chair", "extractionProfile": "multi_part_hard_object",
    }
    guard = build_spatial_ownership_guard(
        chair, [chair], [26, 412, 303, 633], 690, 709
    )
    assert guard["enabled"] is True
    assert guard["entries"] == []
