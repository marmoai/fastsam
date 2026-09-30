import unittest
from unittest.mock import patch

import numpy as np

from segmentation_core import (
    audit_consensus_boundary_review,
    recover_owned_candidate_consensus,
    resolve_sibling_pixel_exclusions,
)


class OwnedCandidateConsensusTest(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((160, 220), dtype=np.float32)
        self.base[50:140, 40:160] = 1
        self.full = self.base.copy()
        self.full[38:50, 50:150] = 1
        self.image = np.full((160, 220, 3), 20, dtype=np.uint8)
        self.image[self.full > 0.5] = 210
        self.box = [20, 20, 180, 140]
        self.sibling = [{"bbox": [45, 25, 65, 65], "spatial_ownership_guard": True}]

    def test_restores_consensus_without_adding_sibling_pixels(self):
        result, audit = recover_owned_candidate_consensus(
            self.image, self.base, [self.full, self.full.copy(), self.base],
            self.box, self.sibling
        )
        self.assertEqual(audit["status"], "accepted", audit)
        self.assertGreater(audit["addedPixels"], 900)
        self.assertTrue(np.all(result[self.base > 0.5] > 0.5))
        self.assertFalse(np.any(result[38:50, 50:65] > 0.5))
        self.assertTrue(np.all(result[38:50, 70:140] > 0.5))

    def test_requires_verified_sibling(self):
        result, audit = recover_owned_candidate_consensus(
            self.image, self.base, [self.full, self.full.copy(), self.base], self.box, []
        )
        self.assertEqual(audit["reason"], "no_verified_sibling")
        self.assertTrue(np.array_equal(result, self.base))

    def test_requires_peer_agreement(self):
        disagreeing = self.full.copy()
        disagreeing[38:50, 100:150] = 0
        result, audit = recover_owned_candidate_consensus(
            self.image, self.base, [self.full, disagreeing, self.base],
            self.box, self.sibling
        )
        self.assertEqual(audit["reason"], "no_safe_consensus")
        self.assertTrue(np.array_equal(result, self.base))

    def test_requires_image_edge(self):
        blank = np.full_like(self.image, 70)
        result, audit = recover_owned_candidate_consensus(
            blank, self.base, [self.full, self.full.copy(), self.base],
            self.box, self.sibling
        )
        self.assertEqual(audit["reason"], "no_safe_consensus")
        self.assertTrue(np.array_equal(result, self.base))

    def test_rejects_peer_that_drops_selected_core(self):
        damaged = self.full.copy()
        damaged[90:115, 70:110] = 0
        result, audit = recover_owned_candidate_consensus(
            self.image, self.base, [damaged, damaged.copy(), self.base],
            self.box, self.sibling
        )
        self.assertEqual(audit["reason"], "no_safe_consensus")
        self.assertTrue(np.array_equal(result, self.base))

    def test_verified_sibling_pixels_do_not_block_whole_bbox(self):
        sibling_box = [{"bbox": [45, 25, 85, 75], "spatial_ownership_guard": True}]
        sibling = np.zeros_like(self.base)
        sibling[25:55, 45:75] = 1
        candidates = np.stack([sibling, sibling.copy(), np.zeros_like(sibling)])
        with patch("segmentation_core.run_sam_bbox_inference", return_value=object()), patch(
            "segmentation_core.normalize_result_masks", return_value=candidates
        ):
            blocked, audit = resolve_sibling_pixel_exclusions(
                self.image, self.base, sibling_box
            )
        self.assertEqual(audit[0]["status"], "accepted", audit)
        self.assertTrue(blocked[35, 55])
        self.assertFalse(blocked[65, 55])

    def test_unstable_sibling_falls_back_to_bbox(self):
        sibling_box = [{"bbox": [45, 25, 85, 75], "spatial_ownership_guard": True}]
        candidates = np.zeros((1, *self.base.shape), dtype=np.float32)
        with patch("segmentation_core.run_sam_bbox_inference", return_value=object()), patch(
            "segmentation_core.normalize_result_masks", return_value=candidates
        ):
            blocked, audit = resolve_sibling_pixel_exclusions(
                self.image, self.base, sibling_box
            )
        self.assertEqual(audit[0]["status"], "fallback", audit)
        self.assertTrue(blocked[65, 55])

    def test_matte_requires_candidate_support(self):
        base = np.zeros((50, 50), dtype=np.float32)
        base[10:40, 10:40] = 1
        reviewed = base.copy()
        reviewed[10:40, 40:43] = 1
        accepted, audit = audit_consensus_boundary_review(base, reviewed, base > 0.5)
        self.assertFalse(accepted, audit)
        accepted, audit = audit_consensus_boundary_review(base, base.copy(), base > 0.5)
        self.assertTrue(accepted, audit)

    def test_extended_reach_is_limited_to_verified_overlap(self):
        base = np.zeros((160, 220), dtype=np.float32)
        base[80:140, 40:160] = 1
        full = base.copy()
        full[55:80, 50:120] = 1
        image = np.full((160, 220, 3), 20, dtype=np.uint8)
        image[full > 0.5] = 210
        sibling = [{"bbox": [45, 25, 90, 90], "spatial_ownership_guard": True}]
        unblocked = np.zeros(base.shape, dtype=bool)
        overlap = np.zeros(base.shape, dtype=bool)
        overlap[25:90, 45:90] = True
        result, audit = recover_owned_candidate_consensus(
            image, base, [full, full.copy(), base], [20, 20, 180, 140],
            sibling, blocked_mask=unblocked, relaxed_region=overlap
        )
        self.assertEqual(audit["status"], "accepted", audit)
        self.assertTrue(result[57, 70] > 0.5)
        self.assertFalse(result[57, 100] > 0.5)


if __name__ == "__main__":
    unittest.main()
