import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sam_ab_regression import request_metadata, semantic_signature


def test_semantic_signature_ignores_embedded_image_but_detects_intent_change():
    base = {
        "image": "data:image/png;base64,old-raster",
        "bboxes": [[0, 0, 1000, 1000]],
        "layerIds": ["chair"],
        "layers": [{"id": "chair", "bbox": [0, 0, 1000, 1000]}],
        "contextLayers": [],
    }
    same_semantics = {**base, "image": "data:image/png;base64,new-raster"}
    changed = json.loads(json.dumps(base))
    changed["layers"][0]["segmentationIntent"] = {
        "targetKind": "atomic_object",
        "includedComponents": [{"name": "chair", "bbox": [1, 1, 999, 999]}],
    }
    assert semantic_signature(base) == semantic_signature(same_semantics)
    assert semantic_signature(base) != semantic_signature(changed)


def test_request_metadata_summarizes_ownership_contract():
    request = {
        "layers": [{
            "id": "tray", "name": "Stone tray", "bbox": [1, 2, 3, 4],
            "parentLayerId": "table", "compositeRole": "atomic_object",
            "segmentationIntent": {
                "targetKind": "atomic_object",
                "includedComponents": [{"name": "tray"}],
                "excludedAdjacentObjects": [{"name": "table"}],
            },
        }],
        "contextLayers": [{}, {}],
    }
    assert request_metadata(request) == {
        "layerId": "tray", "layerName": "Stone tray", "bbox": [1, 2, 3, 4],
        "contextCount": 2, "hasSegmentationIntent": True,
        "intentKind": "atomic_object", "intentComponents": 1,
        "intentExclusions": 1, "parentLayerId": "table",
        "compositeRole": "atomic_object",
    }
