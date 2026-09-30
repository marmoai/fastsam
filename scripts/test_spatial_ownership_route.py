import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fastsam-backend"))

from boundary_matte import recover_safe_outer_boundary
from matte_ops import build_sam_prompt_inputs
import segmentation_core
from segmentation_core import run_positive_probe
from segmentation_policy import get_spatial_ownership_config, resolve_mask_policy
from candidate_selection import (
    build_spatial_ownership_guard,
    select_and_merge_masks,
    verified_positive_probe_priority,
)


def spatial_table_intent():
    return {
        "id": "marble_table",
        "name": "White marble table tier",
        "semanticType": "furniture_table",
        "designRole": "scene_object",
        "extractionProfile": "multi_part_hard_object",
        "bbox": [700, 700, 900, 1000],
        "segmentationIntent": {
            "targetKind": "atomic_object",
            "bboxRole": "tight_subject",
            "confidence": 0.94,
            "includedComponents": [{
                "name": "marble surface", "bbox": [700, 700, 900, 1000], "reason": "target body"
            }],
            "excludedAdjacentObjects": [{
                "name": "stone tray", "bbox": [730, 780, 810, 900], "reason": "separate object"
            }],
            "relations": ["The tray rests on the table but is separately editable."]
        }
    }


def test_single_component_with_covered_exclusion_uses_bbox_only_baseline():
    layer = spatial_table_intent()
    policy = resolve_mask_policy(layer, "completion")
    config = get_spatial_ownership_config(layer, policy)
    assert config["enabled"] is False
    assert config["strictOutput"] is False

    prompts = build_sam_prompt_inputs(
        layer, [layer], [483, 496, 690, 638], 690, 709, prompt_bbox=[450, 470, 690, 660]
    )
    audit = prompts["segmentationIntentAudit"]
    assert audit["enabled"] is False
    # The coarse excluded BBOX sits inside the included subject BBOX. It is
    # therefore unsafe as a negative prompt and must not be forced.
    assert len(audit["negative"]) == 0
    # A sibling rectangle is never made into a whole-image mask or a strict
    # route when it is fully covered by the target component.
    assert prompts["strongExcludeMask"].sum() == 0


def test_flat_single_component_with_covered_exclusion_keeps_bbox_only_baseline():
    layer = spatial_table_intent()
    layer.update({
        "name": "Poster food photo",
        "semanticType": "product_food",
        "designRole": "product_image",
        "extractionProfile": "layout_embedded_product"
    })
    policy = resolve_mask_policy(layer, "completion")
    config = get_spatial_ownership_config(layer, policy)
    assert policy["profile"] == "flat_design"
    assert config["enabled"] is False
    assert config["strictOutput"] is False


def test_missing_component_anchor_cannot_enable_legacy_spatial_lock():
    layer = spatial_table_intent()
    layer["segmentationIntent"]["includedComponents"] = []
    policy = resolve_mask_policy(layer, "completion")
    config = get_spatial_ownership_config(layer, policy)
    assert config["enabled"] is False
    assert config["strictOutput"] is False


def test_single_component_without_safe_exclusion_keeps_bbox_only_prompts():
    layer = spatial_table_intent()
    prompts = build_sam_prompt_inputs(
        layer, [layer], [483, 496, 690, 638], 690, 709, prompt_bbox=[450, 470, 690, 660]
    )
    assert prompts["segmentationIntentAudit"]["reason"] == "no_safe_ownership_evidence"
    assert prompts["points"] is not None
    assert prompts["labels"] is not None


def test_single_component_exclusion_keeps_strict_bbox_and_bbox_only_prompts():
    layer = spatial_table_intent()
    layer["segmentationIntent"]["includedComponents"][0]["bbox"] = [720, 720, 860, 860]
    layer["segmentationIntent"]["excludedAdjacentObjects"][0]["bbox"] = [720, 885, 810, 960]
    config = get_spatial_ownership_config(layer, resolve_mask_policy(layer, "completion"))
    assert config["enabled"] is True
    assert config["strictOutput"] is True
    assert config["promptEnabled"] is False
    prompts = build_sam_prompt_inputs(
        layer, [layer], [483, 496, 690, 638], 690, 709, prompt_bbox=[450, 470, 690, 660]
    )
    audit = prompts["segmentationIntentAudit"]
    assert audit["enabled"] is False
    assert audit["reason"] == "single_component_bbox_baseline"
    assert prompts["points"] is not None


def test_verified_atomic_sibling_enables_one_safe_negative_prompt():
    target = {
        "id": "front-instance",
        "name": "Front atomic instance",
        "semanticType": "furniture_chair",
        "extractionProfile": "multi_part_hard_object",
        "bbox": [626, 38, 892, 389],
        "segmentationIntent": {
            "targetKind": "atomic_object",
            "bboxRole": "tight_subject",
            "confidence": 0.99,
            "includedComponents": [{"bbox": [626, 38, 892, 389]}],
            "excludedAdjacentObjects": [{
                "id": "rear-instance", "bbox": [580, 155, 775, 439]
            }]
        }
    }
    sibling = {
        "id": "rear-instance",
        "name": "Rear atomic instance",
        "semanticType": "furniture_chair",
        "extractionProfile": "multi_part_hard_object",
        "bbox": [580, 155, 775, 439],
        "segmentationIntent": {
            "targetKind": "atomic_object",
            "bboxRole": "tight_subject",
            "confidence": 0.96,
            "includedComponents": [{"bbox": [580, 155, 775, 439]}]
        }
    }
    target_bbox = [26, 443, 268, 632]
    target["_spatialOwnershipGuard"] = build_spatial_ownership_guard(
        target, [target, sibling], target_bbox, 690, 709
    )

    config = get_spatial_ownership_config(target, resolve_mask_policy(target, "completion"))
    assert config["promptEnabled"] is False
    assert config["atomicInstancePromptEligible"] is True
    prompts = build_sam_prompt_inputs(
        target, [target, sibling], target_bbox, 690, 709,
        prompt_bbox=[0, 409, 311, 666]
    )
    audit = prompts["segmentationIntentAudit"]
    assert audit["enabled"] is True
    assert audit["mode"] == "atomic_instance"
    assert len(audit["positive"]) == 1
    assert len(audit["negative"]) == 1
    assert prompts["labels"] == [[1, 0]]
    assert audit["negative"][0][1] < target_bbox[1]


def test_verified_atomic_sibling_enables_outer_boundary_repair_only():
    target = {
        "semanticType": "furniture_chair",
        "extractionProfile": "multi_part_hard_object",
        "segmentationIntent": {
            "targetKind": "atomic_object",
            "bboxRole": "tight_subject",
            "confidence": 0.99,
            "includedComponents": [{"bbox": [500, 100, 800, 500]}],
            "excludedAdjacentObjects": [{"bbox": [500, 450, 800, 800]}]
        },
        "_spatialOwnershipGuard": {"entries": [{"bbox": [50, 50, 80, 80]}]}
    }
    policy = resolve_mask_policy(target, "completion")
    assert policy["ownershipBoundaryRepair"]["enabled"] is True


def test_outer_boundary_repair_never_removes_or_adds_inside_sibling_guard(monkeypatch):
    import boundary_matte

    base = np.zeros((32, 32), dtype=np.float32)
    base[8:24, 8:20] = 1.0
    proposed = base.copy()
    proposed[12, 20] = 1.0
    proposed[12, 21] = 1.0
    proposed[12, 22] = 1.0
    proposed[12, 8] = 0.0
    blocked = np.zeros_like(base, dtype=bool)
    blocked[10:16, 21:24] = True

    monkeypatch.setattr(
        boundary_matte,
        "review_hard_boundary",
        lambda *_args, **_kwargs: (proposed, {"status": "accepted", "reason": "test"})
    )
    repaired, audit = recover_safe_outer_boundary(
        np.zeros((32, 32, 3), dtype=np.uint8), base, [0, 0, 32, 32],
        blocked_additions_mask=blocked, config={"maxBandPixels": 3, "maxAddedRatio": .05}
    )

    assert audit["status"] == "accepted"
    assert audit["removedPixels"] == 0
    assert repaired[12, 8] == 1.0
    assert repaired[12, 20] == 1.0
    assert repaired[12, 21] == 0.0
    assert repaired[12, 22] == 0.0


def test_atomic_intent_without_matched_sibling_keeps_bbox_only_prompts():
    layer = spatial_table_intent()
    layer["segmentationIntent"]["includedComponents"][0]["bbox"] = [720, 720, 860, 860]
    layer["segmentationIntent"]["excludedAdjacentObjects"][0]["bbox"] = [720, 885, 810, 960]
    prompts = build_sam_prompt_inputs(
        layer, [layer], [483, 496, 690, 638], 690, 709,
        prompt_bbox=[450, 470, 690, 660]
    )
    assert prompts["segmentationIntentAudit"]["reason"] == "single_component_bbox_baseline"


def test_multi_component_intent_uses_declared_positive_anchors_only():
    layer = spatial_table_intent()
    layer["segmentationIntent"]["targetKind"] = "composite_assembly"
    layer["segmentationIntent"]["includedComponents"] = [
        {"name": "surface", "bbox": [700, 700, 760, 1000]},
        {"name": "support", "bbox": [780, 760, 900, 900]},
    ]
    layer["segmentationIntent"]["excludedAdjacentObjects"] = [{
        "name": "stool", "bbox": [700, 500, 780, 650]
    }]
    prompts = build_sam_prompt_inputs(
        layer, [layer], [483, 496, 690, 638], 690, 709, prompt_bbox=[450, 470, 690, 660]
    )
    audit = prompts["segmentationIntentAudit"]
    assert audit["enabled"] is True
    assert prompts["points"][0][:2] == audit["positive"]
    assert prompts["labels"][0][:2] == [1, 1]
    assert prompts["labels"][0] == [1, 1]


def test_strict_output_does_not_clip_candidates_before_arbitration():
    layer = spatial_table_intent()
    layer["segmentationIntent"]["includedComponents"][0]["bbox"] = [250, 250, 750, 750]
    layer["segmentationIntent"]["excludedAdjacentObjects"][0]["bbox"] = [100, 100, 300, 300]
    target = [10, 10, 30, 30]
    broad = np.zeros((40, 40), dtype=np.float32)
    broad[5:35, 5:35] = 1.0
    bounded = np.zeros((40, 40), dtype=np.float32)
    bounded[12:28, 12:28] = 1.0

    selected, count, quality = select_and_merge_masks(
        [broad, bounded], target, 40, 40, layer, [], quality_profile="completion"
    )

    assert selected is not None
    assert count >= 1
    assert any(row["inside"] < 1.0 for row in quality["debugCandidates"])


def test_tight_subject_contract_keeps_probe_inside_bbox_contract(monkeypatch):
    import numpy as np

    layer = spatial_table_intent()
    baseline = np.zeros((24, 24), dtype=np.float32)
    baseline[5:19, 5:19] = 1.0
    class FakeTensor:
        def cpu(self):
            return self

        def numpy(self):
            return baseline[None, ...]

    class FakeMasks:
        data = FakeTensor()

    class FakeResult:
        masks = FakeMasks()

    monkeypatch.setattr(
        segmentation_core,
        "run_sam_bbox_inference",
        lambda *args, **kwargs: [FakeResult()]
    )
    masks, audit = run_positive_probe(
        np.zeros((24, 24, 3), dtype=np.uint8), baseline[None, ...],
        [5, 5, 19, 19], [3, 3, 21, 21], "atomic-layer",
        strategy_type="table", layer_meta=layer, context_layers=[], quality_profile="completion"
    )
    assert audit["reason"] != "tight_subject_output_contract"


def test_accepted_probe_preserves_original_b_candidates(monkeypatch):
    layer = spatial_table_intent()
    baseline = np.zeros((24, 24), dtype=np.float32)
    baseline[7:17, 7:17] = 1.0
    probe = np.zeros((24, 24), dtype=np.float32)
    probe[5:19, 5:19] = 1.0

    class FakeTensor:
        def cpu(self):
            return self

        def numpy(self):
            return probe[None, ...]

    class FakeMasks:
        data = FakeTensor()

    class FakeResult:
        masks = FakeMasks()

    monkeypatch.setattr(
        segmentation_core,
        "run_sam_bbox_inference",
        lambda *args, **kwargs: [FakeResult()]
    )
    monkeypatch.setattr(
        segmentation_core,
        "positive_probe_is_safe",
        lambda *args, **kwargs: (
            True,
            {
                "preserve": 1.0,
                "preserveCore": 1.0,
                "fillGain": 0.20,
                "growthGain": 0.20,
                "growth": 1.4,
                "connectedGrowth": 1.0,
                "contextConflictRatio": 0.0,
                "effectiveBbox": [5, 5, 19, 19]
            },
            probe
        )
    )

    masks, audit = run_positive_probe(
        np.zeros((24, 24, 3), dtype=np.uint8), baseline[None, ...],
        [5, 5, 19, 19], [3, 3, 21, 21], "atomic-layer",
        strategy_type="table", layer_meta=layer, context_layers=[], quality_profile="completion"
    )

    assert masks.shape[0] == 2
    assert audit["preservedBaselineCandidates"] == 1
    assert audit["probeCandidateAppended"] is True
    assert audit["verifiedCandidateIndexes"] == [1]
    assert np.count_nonzero(masks[0]) == np.count_nonzero(baseline)


def test_verified_positive_probe_priority_requires_full_safety_audit():
    layer = {
        "_positiveProbeAudit": {
            "status": "accepted",
            "verifiedCandidateIndexes": [3],
            "preserveCore": 0.996,
            "connectedGrowth": 0.54,
            "contextConflictRatio": 0.015,
            "imageReview": {"status": "reviewed"},
        }
    }
    assert verified_positive_probe_priority(layer, 3) == 0.12
    assert verified_positive_probe_priority(layer, 1) == 0.0

    layer["_positiveProbeAudit"]["contextConflictRatio"] = 0.13
    assert verified_positive_probe_priority(layer, 3) == 0.0
