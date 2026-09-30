"""Model-free recovery safety regressions."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from segmentation_core import (
    positive_probe_is_safe, filter_probe_growth_by_image,
    reject_unsupported_probe_supports,
)


class ProbeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((120, 120), np.float32)
        self.base[20:80, 20:80] = 1
        self.probe = self.base.copy()
        self.probe[35:65, 80:95] = 1
        self.target = [10, 10, 110, 110]

    def test_core_preservation_is_not_whole_boundary_preservation(self):
        self.probe[20:22, 20:80] = 0
        accepted, audit, mask = positive_probe_is_safe(self.base, self.probe, self.target)
        self.assertTrue(accepted, audit)
        self.assertLess(audit['preserve'], 0.98)
        self.assertEqual(audit['preserveCore'], 1.0)
        self.assertTrue(np.all(mask[self.base > .5] == 1))

    def test_missing_interior_core_is_rejected(self):
        self.probe[30:50, 30:50] = 0
        accepted, audit, mask = positive_probe_is_safe(self.base, self.probe, self.target)
        self.assertFalse(accepted)
        self.assertIn('preserve_core', audit['failedGates'])
        self.assertIsNone(mask)

    def test_old_outside_speck_does_not_expand_output(self):
        self.base[1, 1] = 1
        accepted, audit, _ = positive_probe_is_safe(self.base, self.probe, self.target)
        self.assertTrue(accepted, audit)
        self.assertEqual(audit['effectiveBbox'], self.target)

    def test_actual_new_growth_can_expand_output(self):
        accepted, audit, _ = positive_probe_is_safe(
            self.base, self.probe, [10, 10, 80, 110], prompt_bbox=[0, 0, 105, 120]
        )
        self.assertTrue(accepted, audit)
        self.assertEqual(audit['effectiveBbox'], [10, 10, 95, 110])

    def test_image_review_removes_background_only_from_additions(self):
        img = np.full((120, 120, 3), 230, np.uint8)
        img[self.probe > .5] = [20, 40, 190]
        self.probe[80:95, 30:60] = 1
        before_base, before_probe, before_image = self.base.copy(), self.probe.copy(), img.copy()
        accepted, audit, mask = positive_probe_is_safe(self.base, self.probe, self.target, image=img)
        self.assertTrue(accepted, audit)
        self.assertEqual(audit['imageReview']['status'], 'reviewed')
        self.assertTrue(np.all(mask[self.base > .5] == 1))
        self.assertTrue(np.all(mask[35:65, 80:95] == 1))
        self.assertFalse(np.any(mask[80:95, 30:60]))
        np.testing.assert_array_equal(self.base, before_base)
        np.testing.assert_array_equal(self.probe, before_probe)
        np.testing.assert_array_equal(img, before_image)

    def test_conflict_and_excessive_growth_are_rejected_before_image_review(self):
        img = np.zeros((120, 120, 3), np.uint8)
        with patch('segmentation_core.filter_probe_growth_by_image') as review:
            accepted, audit, _ = positive_probe_is_safe(
                self.base, self.probe, self.target, image=img, exclude_mask=self.probe > .5
            )
            self.assertFalse(accepted)
            self.assertIn('context_conflict', audit['failedGates'])
            self.probe[:, :] = 1
            accepted, audit, _ = positive_probe_is_safe(self.base, self.probe, self.target, image=img)
            self.assertFalse(accepted)
            self.assertIn('growth_limit', audit['failedGates'])
            review.assert_not_called()

    def test_mixed_context_conflict_is_checked_after_image_review(self):
        img = np.zeros((120, 120, 3), np.uint8)
        exclude = np.zeros((120, 120), bool)
        # The raw SAM extension overlaps an explicit context box, but the
        # reviewed addition is the valid half outside that ownership region.
        exclude[35:45, 80:95] = True
        reviewed_additions = (
            (self.probe > 0.5) & ~(self.base > 0.5) & ~exclude
        )
        with patch(
            'segmentation_core.filter_probe_growth_by_image',
            return_value=(reviewed_additions, {'status': 'reviewed'})
        ) as review:
            accepted, audit, mask = positive_probe_is_safe(
                self.base, self.probe, self.target,
                exclude_mask=exclude, image=img
            )
        self.assertTrue(accepted, audit)
        self.assertGreater(audit['rawContextConflictRatio'], 0.12)
        self.assertLessEqual(audit['contextConflictRatio'], 0.12)
        self.assertIn('context_conflict', audit['rawFailedGates'])
        self.assertTrue(np.all(mask[self.base > 0.5] == 1))
        review.assert_called_once()

    def test_image_failure_rejects_recovery(self):
        with patch('segmentation_core.cv2.grabCut', side_effect=cv2.error('failed')):
            accepted, audit, mask = positive_probe_is_safe(
                self.base, self.probe, self.target, image=np.zeros((120, 120, 3), np.uint8)
            )
        self.assertFalse(accepted)
        self.assertEqual(audit['reason'], 'image_review_failed')
        self.assertIsNone(mask)

    def test_color_filter_cannot_leave_a_detached_island(self):
        core = cv2.erode(self.base.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        proposal = self.probe > .5
        proposal[45:55, 100:110] = True
        proposal[49:51, 95:101] = True

        def remove_bridge(image, labels, *_):
            labels[49:51, 95:100] = cv2.GC_BGD

        with patch('segmentation_core.cv2.grabCut', side_effect=remove_bridge):
            additions, _ = filter_probe_growth_by_image(
                np.zeros((120, 120, 3), np.uint8), self.base > .5, proposal, core, [0, 0, 120, 120]
            )
        self.assertFalse(np.any(additions[45:55, 100:110]))
        self.assertTrue(np.any(additions[35:65, 80:95]))

    def test_explicit_adjacent_support_removes_only_unsupported_thin_growth(self):
        base = np.zeros((200, 200), np.float32)
        base[30:90, 20:160] = 1
        base[90:175, 40:48] = 1
        probe = base.copy()
        probe[90:170, 95:107] = 1
        probe[90:100, 130:145] = 1
        meta = {"segmentationIntent": {
            "confidence": 0.95,
            "excludedAdjacentObjects": [{"bbox": [400, 450, 900, 650]}],
        }}
        result, audit = reject_unsupported_probe_supports(
            probe, np.stack([base]), meta, [10, 20, 170, 180]
        )
        self.assertEqual(audit["status"], "accepted", audit)
        self.assertFalse(np.any(result[110:160, 95:107]))
        self.assertTrue(np.all(result[90:100, 130:145]))
        self.assertTrue(np.all(result[90:175, 40:48]))

    def test_probe_support_requires_explicit_exclusion(self):
        result, audit = reject_unsupported_probe_supports(
            self.probe, np.stack([self.base]), {}, self.target
        )
        self.assertEqual(audit["status"], "skipped")
        np.testing.assert_array_equal(result, self.probe)

    def test_adjacent_support_root_is_color_limited(self):
        base = np.zeros((200, 200), np.float32)
        base[30:90, 20:160] = 1
        base[90:105, 95:107] = 1
        base[90:175, 112:120] = 1
        probe = base.copy()
        probe[105:170, 95:107] = 1
        image = np.full((200, 200, 3), 180, np.uint8)
        image[90:170, 95:107] = (60, 100, 150)
        meta = {"segmentationIntent": {
            "confidence": 0.95,
            "excludedAdjacentObjects": [{"bbox": [400, 450, 900, 650]}],
        }}
        result, audit = reject_unsupported_probe_supports(
            probe, base, meta, [10, 20, 170, 180], image=image
        )
        self.assertEqual(audit["status"], "accepted", audit)
        self.assertGreater(audit["removed"][0]["rootPixels"], 0)
        self.assertFalse(np.any(result[90:105, 95:107]))
        self.assertTrue(np.all(result[90:175, 112:120]))


if __name__ == '__main__':
    unittest.main()
