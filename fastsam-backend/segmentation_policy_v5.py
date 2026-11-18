"""V5 capability resolution: no semantic pipeline switching.

The September policy resolved an object label into an end-to-end pipeline
(food -> compound discovery, chair -> furniture cleanup, glass -> soft matte).
That made extraction depend on wording: a small semantic change moved a layer
into a different chain with different thresholds, and the chains disagreed.

V5 keeps exactly one pipeline and derives every pixel decision from image
evidence. This module therefore resolves only request *capabilities*:

- which phase the layer is in (initial vs post-inpaint completion),
- how much context the prompt box needs,
- which user-supplied regions are hints (include) and which are priors
  (exclude),
- whether the caller is allowed to ask for translucent alpha at all.

It deliberately does not choose an edge class, a matte algorithm, a cleanup
stage, or per-category thresholds. Those are measured from the image in
``segmentation_core_v5``.

Semantics may only act as *hints*. A hint can widen a search window or seed a
completion candidate; it can never authorize a pixel edit by itself.
"""

import os
import math

POLICY_VERSION_V5 = "v5-image-evidence-2"

_STRUCTURED_ENTITY_MIN_CONFIDENCE = 0.85
_STRUCTURED_ATTRIBUTE_MIN_CONFIDENCE = 0.80


def _as_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _bbox(value, width=None, height=None):
    """Normalize a bbox to [x1, y1, x2, y2] without assuming a source space."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        y1, x1, y2, x2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def is_completion_layer_v5(layer_meta):
    """Completion is a phase marker, not an object category."""
    meta = layer_meta or {}
    return bool(
        meta.get("completionSegmentation") or
        str(meta.get("id") or "").startswith("completion-scene-sam-") or
        str(meta.get("layerId") or "").startswith("completion-scene-sam-")
    )


def intent_hints(layer_meta):
    """Return the trusted include/exclude hints declared by the caller.

    The caller's segmentation intent is advisory geometry: included components
    describe where the subject is expected, excluded objects describe what must
    not be absorbed. Neither is treated as a mask.
    """
    meta = layer_meta or {}
    intent = meta.get("segmentationIntent")
    if not isinstance(intent, dict):
        return {"available": False, "confidence": 0.0}

    confidence = _as_float(intent.get("confidence"))
    if not 0.85 <= confidence <= 1.0:
        return {"available": False, "confidence": 0.0}
    included = []
    for entry in intent.get("includedComponents") or []:
        box = _bbox((entry or {}).get("bbox"))
        if box is not None:
            included.append(box)

    excluded = []
    for entry in intent.get("excludedAdjacentObjects") or []:
        box = _bbox((entry or {}).get("bbox"))
        if box is not None:
            excluded.append(box)

    return {
        "available": True,
        "confidence": confidence,
        "targetKind": str(intent.get("targetKind") or "unknown"),
        "bboxRole": str(intent.get("bboxRole") or "unknown"),
        "included": included,
        "excluded": excluded,
        "relations": list(intent.get("relations") or []),
    }


def material_prior(layer_meta):
    """Resolve whether the caller *asserts* a translucent material.

    The assertion is a prior for image evidence, not a route. A layer may claim
    glass and still resolve to opaque alpha if the pixels do not support it.
    """
    meta = layer_meta or {}
    semantics = meta.get("entitySemantics")
    if not isinstance(semantics, dict):
        return {"asserted": False, "reason": "no_semantics"}

    target = semantics.get("target")
    if not isinstance(target, dict):
        return {"asserted": False, "reason": "no_target"}
    if _as_float(target.get("confidence")) < _STRUCTURED_ENTITY_MIN_CONFIDENCE:
        return {"asserted": False, "reason": "low_target_confidence"}

    attributes = semantics.get("attributes")
    if not isinstance(attributes, dict):
        return {"asserted": False, "reason": "no_attributes"}

    transparency = attributes.get("transparency")
    if not isinstance(transparency, dict):
        return {"asserted": False, "reason": "no_transparency"}
    if _as_float(transparency.get("confidence")) < _STRUCTURED_ATTRIBUTE_MIN_CONFIDENCE:
        return {"asserted": False, "reason": "low_transparency_confidence"}

    value = str(transparency.get("value") or "").strip().lower()
    if value in {"transparent", "translucent", "semi_transparent"}:
        return {"asserted": True, "reason": "structured_transparency", "value": value}
    return {"asserted": False, "reason": "opaque_attribute"}


def _box_overlap_share(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return (width * height) / area_a


def _exclusion_context(layer_meta, context_layers, hint_bbox):
    """Collect exclusion boxes that may veto an addition.

    Layers here receive normalized (0-1000) boxes from the client, exactly like
    the hint. Two things are not priors: a full-frame context box (a background
    plate covers the whole scene, so its rect says nothing about ownership) and
    any box that contains the subject hint. Everything else stays a prior
    because a sibling box can overlap the subject, but V5 never turns another
    layer's rectangle into a hard background label.
    """
    priors = []
    for entry in context_layers or []:
        if not isinstance(entry, dict):
            continue
        target_id = layer_meta.get("id") or layer_meta.get("layerId")
        if target_id and target_id == (entry.get("id") or entry.get("layerId")):
            continue
        box = _bbox(entry.get("bboxNormalized") or entry.get("bbox"))
        if box is None:
            continue
        if box[2] - box[0] >= 980 and box[3] - box[1] >= 980:
            continue
        if hint_bbox is not None and _box_overlap_share(hint_bbox, box) >= 0.95:
            continue
        source = (
            "context_strict"
            if entry.get("strictOwnership") or entry.get("excludedAdjacent")
            else "context_soft"
        )
        priors.append({"bbox": box, "source": source})
    for entry in intent_hints(layer_meta).get("excluded") or []:
        priors.append({"bbox": entry, "source": "intent_excluded"})
    return priors


def resolve_policy_v5(layer_meta=None, quality_profile="publish"):
    """Resolve request capabilities. No object category enters this result."""
    meta = dict(layer_meta or {})
    phase = "completion" if is_completion_layer_v5(meta) else "initial"
    hints = intent_hints(meta)
    material = material_prior(meta)

    # Prompt context. A tight-subject intent is the case most likely to cut a
    # protruding part (a leg, a rim, a garnish), so it gets slightly more
    # search context. This value widens the *prompt*, never the output canvas.
    hint_expand = 0.18
    if hints.get("bboxRole") == "tight_subject" and hints.get("confidence", 0) >= 0.85:
        hint_expand = 0.26
    if phase == "completion":
        hint_expand = min(hint_expand, 0.14)

    max_masks = 3
    if hints.get("targetKind") == "composite_assembly":
        max_masks = 4

    return {
        "version": POLICY_VERSION_V5,
        "phase": phase,
        "completion": phase == "completion",
        "hintExpandRatio": hint_expand,
        "maxMasks": max_masks,
        "intent": hints,
        # A high-confidence tight-subject intent allows the "fill the hinted
        # subject region" evidence pass. It still requires edge and background
        # evidence per region inside the core pipeline.
        "intentFill": {
            "enabled": bool(
                phase == "initial" and
                hints.get("available") and
                hints.get("confidence", 0) >= 0.85 and
                hints.get("targetKind") in {"atomic_object", "composite_assembly"} and
                hints.get("bboxRole") in {"tight_subject", "context_region"}
            ),
            "reason": (
                "trusted_tight_subject" if hints.get("available") else "no_intent"
            ),
        },
        "materialPrior": material,
        # Translucent alpha is evaluated from pixels in the core pipeline. This
        # flag only records that the caller asserted the material; the image
        # still has to agree.
        "allowTranslucent": bool(material.get("asserted")),
        "exclusionPriors": _exclusion_context(
            meta, meta.get("_contextLayers") or [],
            hints.get("included")[0] if hints.get("included") else None,
        ),
        "qualityProfile": str(quality_profile or "publish").lower(),
        # Diagnostics are opt-in so production logs stay readable.
        "verbose": str(os.environ.get("SAMV5_VERBOSE", "1")).lower() not in {
            "0", "false", "no", "off"
        },
    }


def expand_hint_bbox_v5(hint_bbox, image_width, image_height, ratio):
    """Add search context around a semantic hint box.

    The output canvas is derived later from the accepted alpha, so this padding
    cannot inflate the exported crop.
    """
    x1, y1, x2, y2 = [float(v) for v in hint_bbox]
    pad_x = max(2.0, (x2 - x1) * ratio)
    pad_y = max(2.0, (y2 - y1) * ratio)
    # Edge-to-edge hints keep the axis that already spans the frame.
    if x1 <= 1 and image_width - x2 <= 1:
        pad_x = 0.0
    if y1 <= 1 and image_height - y2 <= 1:
        pad_y = 0.0
    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width), x2 + pad_x),
        min(float(image_height), y2 + pad_y),
    ]


def policy_for_strategy_type(strategy_type=None, layer_meta=None, quality_profile="publish"):
    """Compatibility shim for callers that only pass a strategy string."""
    return resolve_policy_v5(layer_meta or {}, quality_profile)
