import unittest

import numpy as np

from sam_runtime import _SamEmbeddingCache


class FakePredictor:
    def __init__(self):
        self.features = None
        self.im = None
        self.encoded = []

    def reset_image(self):
        self.features = None
        self.im = None

    def set_image(self, image):
        self.encoded.append(image)
        self.features = object()


class SamEmbeddingCacheTest(unittest.TestCase):
    def test_restores_source_features_after_crop(self):
        cache = _SamEmbeddingCache()
        predictor = FakePredictor()
        source = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = np.zeros((25, 25, 3), dtype=np.uint8)

        cache.prepare(predictor, source, 1024, "b")
        source_features = predictor.features
        cache.prepare(predictor, crop, 1024, "b")
        crop_features = predictor.features
        self.assertIsNot(source_features, crop_features)

        cache.prepare(predictor, source, 1024, "b")
        self.assertIs(predictor.features, source_features)
        self.assertEqual(len(predictor.encoded), 2)

        cache.prepare(predictor, crop, 1024, "b")
        self.assertIs(predictor.features, crop_features)
        self.assertEqual(len(predictor.encoded), 2)
        cache.close()
        self.assertIsNone(predictor.features)


if __name__ == "__main__":
    unittest.main()
