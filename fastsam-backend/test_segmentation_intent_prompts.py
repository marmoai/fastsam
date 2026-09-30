"""Unit tests for high-confidence semantic ownership prompt compilation."""

import unittest

import numpy as np

from candidate_selection import (
    build_segmentation_intent_exclude_mask,
    build_segmentation_intent_tight_subject_contract,
)
from matte_ops import build_segmentation_intent_prompt_inputs
from segmentation_core import positive_probe_is_safe


class SegmentationIntentPromptTests(unittest.TestCase):
    def test_uses_component_anchor_and_safe_exclusion_point(self):
        meta = {
            "segmentationIntent": {
                "targetKind": "atomic_object",
                "confidence": 0.95,
                "includedComponents": [{"name": "stone tray", "bbox": [400, 400, 620, 620]}],
                # This support overlaps the tray, but its upper-left corner is
                # outside the included component and remains a safe negative.
                "excludedAdjacentObjects": [{"name": "marble support", "bbox": [300, 300, 700, 700]}]
            }
        }
        audit = build_segmentation_intent_prompt_inputs(
            meta, [30, 30, 90, 90], [20, 20, 95, 95], 100, 100
        )
        self.assertTrue(audit["enabled"])
        self.assertEqual(audit["componentCount"], 1)
        self.assertEqual(len(audit["positive"]), 1)
        self.assertEqual(len(audit["negative"]), 1)
        self.assertNotEqual(audit["positive"][0], audit["negative"][0])

    def test_does_not_enable_uncertain_or_low_confidence_intent(self):
        meta = {
            "segmentationIntent": {
                "targetKind": "uncertain",
                "confidence": 0.99,
                "includedComponents": [{"name": "unknown", "bbox": [300, 300, 700, 700]}]
            }
        }
        audit = build_segmentation_intent_prompt_inputs(
            meta, [20, 20, 80, 80], [20, 20, 80, 80], 100, 100
        )
        self.assertFalse(audit["enabled"])
        self.assertEqual(audit["reason"], "intent_not_confident")

    def test_skips_an_exclusion_when_every_safe_sample_is_a_component(self):
        meta = {
            "segmentationIntent": {
                "targetKind": "atomic_object",
                "confidence": 0.91,
                "includedComponents": [{"name": "subject", "bbox": [200, 200, 800, 800]}],
                "excludedAdjacentObjects": [{"name": "ambiguous overlap", "bbox": [200, 200, 800, 800]}]
            }
        }
        audit = build_segmentation_intent_prompt_inputs(
            meta, [20, 20, 80, 80], [20, 20, 80, 80], 100, 100
        )
        self.assertFalse(audit["enabled"])
        self.assertEqual(audit["negative"], [])

    def test_safe_exclude_mask_preserves_component_projection_overlap(self):
        meta = {
            "segmentationIntent": {
                "targetKind": "atomic_object",
                "confidence": 0.95,
                "includedComponents": [{"bbox": [300, 300, 700, 700]}],
                "excludedAdjacentObjects": [{"bbox": [100, 100, 500, 500]}]
            }
        }
        exclude_mask, audit = build_segmentation_intent_exclude_mask(meta, 100, 100)
        self.assertTrue(audit["enabled"])
        self.assertTrue(exclude_mask[15, 15])
        self.assertFalse(exclude_mask[35, 35])
        self.assertFalse(exclude_mask[49, 49])

    def test_probe_rejects_growth_into_safe_intent_exclusion(self):
        baseline = np.zeros((30, 30), dtype=np.float32)
        baseline[10:20, 10:20] = 1
        probe = baseline.copy()
        probe[10:20, 20:25] = 1
        intent_exclude = np.zeros((30, 30), dtype=bool)
        intent_exclude[10:20, 22:25] = True

        accepted, audit, selected = positive_probe_is_safe(
            baseline,
            probe,
            [8, 8, 22, 22],
            prompt_bbox=[0, 0, 30, 30],
            intent_exclude_mask=intent_exclude,
            image=None
        )

        self.assertFalse(accepted)
        self.assertIsNone(selected)
        self.assertIn("intent_ownership_conflict", audit["failedGates"])
        self.assertGreater(audit["intentConflictPixels"], 0)

    def test_tight_subject_rejects_large_unproven_external_growth(self):
        meta = {
            "segmentationIntent": {
                "targetKind": "atomic_object",
                "bboxRole": "tight_subject",
                "confidence": 0.95,
                "includedComponents": [{"bbox": [333, 333, 667, 667]}]
            }
        }
        baseline = np.zeros((30, 30), dtype=np.float32)
        baseline[10:20, 10:20] = 1
        probe = baseline.copy()
        probe[10:20, 20:25] = 1
        contract = build_segmentation_intent_tight_subject_contract(
            meta, [10, 10, 20, 20], 30, 30
        )

        accepted, audit, _ = positive_probe_is_safe(
            baseline,
            probe,
            [10, 10, 20, 20],
            prompt_bbox=[0, 0, 30, 30],
            tight_subject_contract=contract,
            max_growth=2.0,
            image=None
        )

        self.assertFalse(accepted)
        self.assertIn("tight_subject_external_growth", audit["failedGates"])
        self.assertIn("tight_subject_side_extension", audit["failedGates"])

    def test_tight_subject_keeps_small_bounded_annotation_correction(self):
        meta = {
            "segmentationIntent": {
                "targetKind": "atomic_object",
                "bboxRole": "tight_subject",
                "confidence": 0.95,
                "includedComponents": [{"bbox": [250, 250, 750, 750]}]
            }
        }
        baseline = np.zeros((80, 80), dtype=np.float32)
        baseline[20:60, 20:60] = 1
        probe = baseline.copy()
        probe[20:60, 60:63] = 1
        contract = build_segmentation_intent_tight_subject_contract(
            meta, [20, 20, 60, 60], 80, 80
        )

        accepted, audit, selected = positive_probe_is_safe(
            baseline,
            probe,
            [20, 20, 60, 60],
            prompt_bbox=[0, 0, 80, 80],
            tight_subject_contract=contract,
            min_growth_gain=0.01,
            image=None
        )

        self.assertTrue(accepted)
        self.assertIsNotNone(selected)
        self.assertEqual(audit["tightSubject"]["failedGates"], [])


if __name__ == "__main__":
    unittest.main()
