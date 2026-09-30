"""Safety regressions for B-led, L-bounded contour refinement."""

import unittest

import numpy as np

from segmentation_core import (
    build_b_boundary_refine_prompt_inputs,
    refine_b_mask_boundary_with_l_evidence,
    resolve_verified_effective_bbox,
)


class CrossModelBoundaryRefineTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((140, 140), np.float32)
        self.base[30:110, 30:110] = 1
        self.target = [10, 10, 130, 130]

    def test_l_can_adjust_only_the_b_boundary_band(self):
        l_candidate = self.base.copy()
        l_candidate[29:111, 29:111] = 1

        refined, accepted, audit = refine_b_mask_boundary_with_l_evidence(
            [self.base], [l_candidate], self.target, strategy_type="furniture"
        )

        self.assertTrue(accepted, audit)
        self.assertEqual(audit["status"], "accepted:bounded_l_boundary")
        self.assertGreater(audit["addedPixels"], 0)
        self.assertEqual(audit["removedPixels"], 0)
        # L's outline is accepted at the contour, but cannot grow beyond the
        # narrow B-derived support band.
        self.assertTrue(np.all(refined[20:29, :] == 0))
        self.assertTrue(np.all(refined[:, 20:29] == 0))
        self.assertTrue(np.all(refined[40:100, 40:100] == 1))

    def test_l_boundary_prompts_are_inside_b_stable_core(self):
        points, labels, audit = build_b_boundary_refine_prompt_inputs(
            self.base, self.target
        )

        self.assertEqual(audit["status"], "ready")
        self.assertEqual(len(points), 1)
        self.assertEqual(labels, [[1] * len(points[0])])
        for x, y in points[0]:
            self.assertEqual(self.base[y, x], 1)
            self.assertGreaterEqual(x, 32)
            self.assertLessEqual(x, 107)
            self.assertGreaterEqual(y, 32)
            self.assertLessEqual(y, 107)

    def test_verified_effective_bbox_expands_l_arbitration_geometry(self):
        bbox, accepted = resolve_verified_effective_bbox(
            [30, 30, 110, 110], [20, 24, 122, 118], 140, 140
        )

        self.assertTrue(accepted)
        self.assertEqual(bbox, [20, 24, 122, 118])

    def test_effective_bbox_cannot_shrink_semantic_target(self):
        bbox, accepted = resolve_verified_effective_bbox(
            [30, 30, 110, 110], [35, 30, 110, 108], 140, 140
        )

        self.assertFalse(accepted)
        self.assertEqual(bbox, [30, 30, 110, 110])

    def test_l_candidate_that_loses_b_core_is_rejected(self):
        l_candidate = self.base.copy()
        l_candidate[52:88, 52:88] = 0

        refined, accepted, audit = refine_b_mask_boundary_with_l_evidence(
            [self.base], [l_candidate], self.target, strategy_type="hard_product"
        )

        self.assertFalse(accepted)
        self.assertIsNone(refined)
        self.assertEqual(audit["status"], "rejected:no_safe_boundary_candidate")
        self.assertGreater(audit["rejections"].get("l_disagrees_with_b_identity", 0), 0)

    def test_l_cannot_delete_more_than_the_bounded_edge_allowance(self):
        l_candidate = self.base.copy()
        l_candidate[30:32, 30:110] = 0

        refined, accepted, audit = refine_b_mask_boundary_with_l_evidence(
            [self.base], [l_candidate], self.target, strategy_type="table"
        )

        self.assertFalse(accepted)
        self.assertIsNone(refined)
        self.assertEqual(audit["status"], "rejected:no_safe_boundary_candidate")
        self.assertGreater(audit["rejections"].get("boundary_change_out_of_bounds", 0), 0)


if __name__ == "__main__":
    unittest.main()
