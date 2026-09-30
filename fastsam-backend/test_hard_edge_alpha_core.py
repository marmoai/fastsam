import unittest

import numpy as np

from matte_ops import build_hard_edge_alpha, generate_table_safe_matte


class HardEdgeAlphaCoreTests(unittest.TestCase):
    def test_preserves_accepted_edge_pixels_without_adding_outside(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[5:25, 6:26] = 1
        mask[5:12, 16:20] = 0
        mask[18:25, 21:26] = 0
        mask[12:25, 24] = 1
        target_bbox = [4, 3, 28, 28]

        alpha = build_hard_edge_alpha(mask, target_bbox, preserve_accepted_mask=True)

        self.assertTrue(np.all(alpha[mask > 0.5] >= 128))
        self.assertTrue(np.all(alpha[mask <= 0.5] == 0))
        self.assertEqual(int(alpha[7, 18]), 0)

    def test_table_safe_matte_preserves_subject_and_bbox(self):
        mask = np.zeros((24, 28), dtype=np.float32)
        mask[5:17, 3:23] = 1
        mask[11:17, 18:23] = 0
        bbox = [2, 4, 20, 19]

        alpha = generate_table_safe_matte(mask, bbox)
        expected = np.zeros_like(mask, dtype=bool)
        expected[4:19, 2:20] = mask[4:19, 2:20] > 0.5

        self.assertTrue(np.all(alpha[expected] >= 128))
        self.assertTrue(np.all(alpha[~expected] == 0))


if __name__ == "__main__":
    unittest.main()
