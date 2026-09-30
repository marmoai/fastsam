import unittest

import numpy as np

from segmentation_core import audit_local_refine_candidate


class LocalRefineGuardTest(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((120, 120), dtype=np.float32)
        self.base[20:100, 20:100] = 1
        self.bbox = [15, 15, 105, 105]

    def test_accepts_small_boundary_change(self):
        candidate = self.base.copy()
        candidate[25:95, 19] = 1
        accepted, audit = audit_local_refine_candidate(self.base, candidate, self.bbox, "hard_product")
        self.assertTrue(accepted, audit)

    def test_rejects_missing_core(self):
        candidate = self.base.copy()
        candidate[40:70, 40:70] = 0
        accepted, audit = audit_local_refine_candidate(self.base, candidate, self.bbox, "hard_product")
        self.assertFalse(accepted)
        self.assertGreater(audit["coreRemovedPixels"], 0)

    def test_rejects_excessive_boundary_loss(self):
        candidate = self.base.copy()
        candidate[20:24, 20:100] = 0
        accepted, audit = audit_local_refine_candidate(self.base, candidate, self.bbox, "hard_product")
        self.assertFalse(accepted)
        self.assertGreater(audit["removedRatio"], 0.015)

    def test_rejects_interior_cleanup_change(self):
        candidate = self.base.copy()
        candidate[50, 50] = 0
        accepted, audit = audit_local_refine_candidate(self.base, candidate, self.bbox, "hard_product")
        self.assertFalse(accepted)
        self.assertEqual(audit["outsideBoundaryBandPixels"], 1)

    def test_does_not_apply_to_other_strategies(self):
        candidate = self.base.copy()
        candidate[25:95, 19] = 1
        accepted, _ = audit_local_refine_candidate(self.base, candidate, self.bbox, "food_product")
        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()
