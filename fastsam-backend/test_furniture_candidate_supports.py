import unittest
from unittest.mock import patch

import cv2
import numpy as np

from segmentation_core import recover_furniture_candidate_supports


class FurnitureCandidateSupportTests(unittest.TestCase):
    def test_verified_sibling_blocks_only_foreign_growth(self):
        image = np.full((300, 300, 3), 180, np.uint8)
        base = np.zeros((300, 300), np.float32)
        base[80:190, 80:220] = 1
        peer = base.copy()
        peer[190:230, 100:105] = 1
        peer[190:230, 205:210] = 1
        sibling = np.zeros((300, 300), dtype=bool)
        sibling[190:230, 205:210] = True
        intent = {
            "segmentationIntent": {
                "confidence": 0.95,
                "excludedAdjacentObjects": [{"bbox": [633, 666, 800, 800]}],
            }
        }
        context = [{
            "id": "stool", "bbox": [633, 666, 800, 800],
            "semanticType": "furniture_stool",
            "extractionProfile": "multi_part_hard_object",
        }]
        with patch(
            "segmentation_core.resolve_sibling_pixel_exclusions",
            return_value=(sibling, [{"status": "accepted"}]),
        ):
            result, audit = recover_furniture_candidate_supports(
                image, base, np.stack([base, peer]), [50, 40, 240, 240],
                intent, context,
            )
        self.assertEqual(audit["reason"], "owned_connected_candidate")
        self.assertEqual(int(result[190:230, 100:105].sum()), 200)
        self.assertEqual(int(result[190:230, 205:210].sum()), 0)

    def test_floor_peer_requires_connected_horizontal_support(self):
        image = np.full((400, 400, 3), 180, np.uint8)
        image[310:314, 90:320] = 25
        base = np.zeros((400, 400), np.float32)
        base[80:308, 90:320] = 1
        peer = base.copy()
        peer[305:330, 90:320] = 1
        peer[80:280, 350:380] = 1

        result, audit = recover_furniture_candidate_supports(
            image, base, np.stack([base, peer]), [50, 50, 350, 350]
        )
        self.assertEqual(audit["reason"], "connected_candidate_rail")
        self.assertEqual(int(result[325:345, 90:320].sum()), 0)
        count, _, _, _ = cv2.connectedComponentsWithStats(
            (result > 0.5).astype(np.uint8), connectivity=8
        )
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
