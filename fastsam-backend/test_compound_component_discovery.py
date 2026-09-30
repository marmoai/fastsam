import unittest

import numpy as np

from segmentation_core import (
    audit_compound_component_discovery,
    allow_thin_strict_edge_continuations,
    has_verified_compound_silhouette,
    recover_strict_bbox_edge_continuations,
    remove_strictly_excluded_detached_components,
)


class CompoundComponentDiscoveryTest(unittest.TestCase):
    def test_only_accepted_discovery_with_added_support_protects_silhouette(self):
        self.assertTrue(has_verified_compound_silhouette({
            "_compoundComponentDiscovery": {
                "status": "accepted",
                "verifiedAddedPixels": 40,
            }
        }))
        self.assertTrue(has_verified_compound_silhouette({
            "_compoundComponentDiscovery": {
                "status": "accepted",
                "candidates": [{"status": "accepted", "verifiedAddedPixels": 40}],
            }
        }))
        self.assertFalse(has_verified_compound_silhouette({
            "_compoundComponentDiscovery": {
                "status": "accepted",
                "reason": "no_verified_discovery_keep_baseline",
                "candidates": [{"status": "rejected", "verifiedAddedPixels": 0}],
            }
        }))
        self.assertFalse(has_verified_compound_silhouette({
            "_compoundComponentDiscovery": {
                "status": "rejected",
                "verifiedAddedPixels": 40,
            }
        }))

    def test_strict_ownership_clips_pixels_without_vetoing_whole_attachment(self):
        core = np.zeros((120, 140), dtype=np.float32)
        core[30:80, 30:70] = 1
        candidate = core.copy()
        candidate[42:67, 69:100] = 1
        strict = np.zeros_like(core, dtype=bool)
        strict[38:72, 84:100] = True

        merged, audit = audit_compound_component_discovery(
            core,
            candidate,
            [20, 20, 100, 100],
            [10, 10, 120, 110],
            config={"minComponentAreaRatio": 0.0025},
            strict_exclude_mask=strict,
        )

        self.assertEqual(audit["status"], "accepted", audit)
        self.assertEqual(audit["strictOwnershipRemovedPixels"], 400)
        self.assertTrue(np.all(merged[45:60, 75:83] > 0.5))
        self.assertFalse(np.any(merged[strict]))

    def test_thin_attached_target_edge_can_cross_overlapping_strict_box(self):
        core = np.zeros((120, 200), dtype=np.float32)
        core[22:82, 30:170] = 1
        candidate = core.copy()
        candidate[18:22, 70:86] = 1
        strict = np.zeros_like(core, dtype=bool)
        strict[18:23, 68:90] = True

        allowed, audit = allow_thin_strict_edge_continuations(
            core, (candidate > 0.5) & ~(core > 0.5), strict, [20, 18, 180, 90]
        )

        self.assertEqual(audit["status"], "accepted", audit)
        self.assertEqual(int(allowed.sum()), 64)
        merged, merge_audit = audit_compound_component_discovery(
            core,
            candidate,
            [20, 18, 180, 90],
            [10, 10, 190, 100],
            config={"minComponentAreaRatio": 0.02},
            strict_exclude_mask=strict,
        )
        self.assertEqual(merge_audit["status"], "accepted", merge_audit)
        self.assertTrue(np.all(merged[18:22, 70:86] > 0.5))
        self.assertEqual(merge_audit["strictEdgeContinuation"]["allowedPixels"], 64)

    def test_small_detached_component_inside_strict_context_is_removed_only(self):
        mask = np.zeros((120, 150), dtype=np.float32)
        mask[25:85, 25:85] = 1
        mask[50:60, 95:105] = 1
        mask[95:103, 35:43] = 1
        strict = np.zeros_like(mask, dtype=bool)
        strict[48:62, 93:107] = True

        cleaned, audit = remove_strictly_excluded_detached_components(mask, strict)

        self.assertEqual(audit["status"], "accepted", audit)
        self.assertEqual(int((mask > 0.5).sum() - (cleaned > 0.5).sum()), 100)
        self.assertFalse(np.any(cleaned[50:60, 95:105] > 0.5))
        self.assertTrue(np.all(cleaned[25:85, 25:85] > 0.5))
        self.assertTrue(np.all(cleaned[95:103, 35:43] > 0.5))

    def test_contained_low_fill_target_edge_extension_is_not_called_background(self):
        core = np.zeros((120, 140), dtype=np.float32)
        core[45:65, 30:60] = 1
        candidate = core.copy()
        candidate[45:50, 60:100] = 1
        candidate[50:80, 95:100] = 1

        merged, audit = audit_compound_component_discovery(
            core,
            candidate,
            [20, 20, 100, 100],
            [10, 10, 120, 110],
            config={"minComponentAreaRatio": 0.0025},
        )

        self.assertEqual(audit["status"], "accepted", audit)
        self.assertTrue(np.all(merged[55:75, 97:100] > 0.5))

    def test_low_fill_edge_envelope_outside_target_is_still_rejected(self):
        core = np.zeros((120, 140), dtype=np.float32)
        core[45:65, 30:60] = 1
        candidate = core.copy()
        candidate[45:50, 60:105] = 1
        candidate[50:80, 100:105] = 1

        merged, audit = audit_compound_component_discovery(
            core,
            candidate,
            [20, 20, 100, 100],
            [10, 10, 120, 110],
            config={"minComponentAreaRatio": 0.0025},
        )

        self.assertEqual(audit["status"], "rejected", audit)
        self.assertEqual(audit["components"][0]["reason"], "background_edge_envelope")
        self.assertTrue(np.array_equal(merged, core))

    def test_strict_bbox_recovers_only_attached_peer_edge_pixels(self):
        baseline = np.zeros((60, 70), dtype=np.float32)
        baseline[15:35, 20:40] = 1
        peer = baseline.copy()
        peer[19:31, 40:42] = 1
        peer[35:37, 27:35] = 1
        peer[10:14, 40:42] = 1  # detached extension noise
        strict = np.zeros_like(baseline, dtype=bool)
        strict[19:21, 40] = True

        recovered, audit = recover_strict_bbox_edge_continuations(
            baseline,
            [baseline, peer],
            [20, 15, 40, 35],
            strict_exclude_mask=strict,
            config={"maxExtensionRatio": 0.1, "maxAddedRatio": 0.1},
        )

        self.assertEqual(audit["status"], "accepted", audit)
        self.assertEqual(audit["effectiveBbox"], [20, 15, 42, 37])
        self.assertTrue(np.all(recovered[21:31, 40:42] > 0.5))
        self.assertFalse(np.any(recovered[strict] > 0.5))
        self.assertFalse(np.any(recovered[10:14, 40:42] > 0.5))

    def test_strict_bbox_does_not_expand_for_unattached_or_excluded_peer_noise(self):
        baseline = np.zeros((60, 70), dtype=np.float32)
        baseline[15:35, 20:40] = 1
        peer = baseline.copy()
        peer[10:14, 40:42] = 1
        strict = np.zeros_like(baseline, dtype=bool)
        strict[15:35, 40:42] = True

        recovered, audit = recover_strict_bbox_edge_continuations(
            baseline,
            [baseline, peer],
            [20, 15, 40, 35],
            strict_exclude_mask=strict,
            config={"maxExtensionRatio": 0.1},
        )

        self.assertEqual(audit["status"], "skipped", audit)
        self.assertTrue(np.array_equal(recovered > 0.5, baseline > 0.5))


if __name__ == "__main__":
    unittest.main()
