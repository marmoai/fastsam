"""V5 extraction pipeline: one image-evidence pipeline for every layer.

Design rules
------------
The semantic box is a search hint. Accepted subject pixels are exported at
full extent unless SAMV5_EXPORT_BBOX=hint explicitly requests a crop. One
pipeline serves every category. Neighbour boxes require independent colour
and connectivity evidence; ambiguous overlaps are reported, never cut as
rectangles. Background removal is bounded by image edges and removal budgets.
Removed pixels are forbidden to completion, hole repair and alpha. Completion
requires SAM-candidate consensus, while enclosed omissions additionally need
foreground colour evidence. Fine structure is measured from topology; channel
alpha requires per-pixel backdrop evidence and cannot clear a dense core.
Quality decisions and diagnostics are preserved by the HTTP transport.

The module is pure NumPy/OpenCV so it can be evaluated offline against saved
SAM candidates without loading a model.
"""

import math
import os
import time
import sys

import cv2
import numpy as np
from ownership_v5 import build_ownership_evidence, review_ownership_candidates, restore_owned_contour
from boundary_refine_v5 import boundary_evidence
from matting_v5 import backdrop_matte
from boundary_audit_v5 import BoundaryAudit, diagnostic_color_image
from alpha_guard_v5 import constrain_alpha
from sam_diagnostics import sam_diagnostic_event

from segmentation_policy_v5 import (
    POLICY_VERSION_V5,
    resolve_policy_v5,
)

CORE_VERSION_V5 = "v5-image-evidence-core-24-source-boundary-control"

# Generic, non-category thresholds. They are expressed relative to image and
# window scale wherever possible so one constant serves every image size.
_WINDOW_MARGIN_RATIO = 0.06          # analysis context beyond hint + mask
_EDGE_BAND_RATIO = 0.0028            # half-width of the edge measurement band
_BACKGROUND_REACH_RATIO = 0.004      # flood reach for background seeds
_COMPLETION_GAP_RATIO = 0.005        # adjacency gap for completion candidates
# Detached parts are thin and anti-aliased, so their boundary rarely sits
# fully on a strong gradient; a quarter of it is enough.
_COMPLETION_EDGE_SUPPORT = 0.25
# A detached part should be closer to the subject palette than to the
# background palette, not merely similar to the subject's hue.
_COMPLETION_COLOR_MARGIN = 4.0
_MIN_COMPONENT_PIXELS = 48
_MAX_REMOVAL_FRACTION = 0.45
_MAX_ADDITION_FRACTION = 0.60
_EXCLUSION_VETO_SHARE = 0.55
# A caller-declared exclusion may remove pixels, but not most of the layer: at
# that point the prior and the proposal disagree about what the subject is, and
# cutting would delete the layer rather than refine it.
_EXCLUSION_TRIM_LIMIT = 0.50
# Below this the two palettes are the same colour and no colour decision is
# evidence. It is a floor, not a quality bar: a white subject on a white
# backdrop separates by ten levels and still has a real, walkable boundary.
_MIN_PALETTE_SEPARATION = 6.0
_ALPHA_MIN = 0.08
# A hole enclosed by the subject shares its palette; it is told apart from a
# missing part by the background-coloured pixels separating it from the mask.
_BRIDGE_BACKGROUND_SHARE = 0.65
# Accepted subject pixels may extend beyond the semantic hint. Explicit crop
# mode is retained for callers that deliberately request a rectangular cut.
_EXPORT_BBOX_SOURCE = os.environ.get("SAMV5_EXPORT_BBOX", "alpha")
_HOLE_MAX_SHARE = 0.03
_HOLE_FILL_BUDGET = 0.08
# A perforated subject (foliage, mesh, lace) is recognised by spread-out
# enclosed openings, and refined with local channel levels rather than with the
# hole rules: for foliage the openings *are* the subject's structure, so filling
# them is the wrong repair.
_CHANNEL_MIN_OPENINGS = 2
_CHANNEL_MIN_OPENING_SHARE = 0.015
_CHANNEL_CONTRAST = 15.0
_CHANNEL_COLOR_MATCH = 18.0
# Narrow antialias coverage preserves thin stems and original sharp corners.
_SILHOUETTE_BLUR_RADIUS = 0.45


def _truthy(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).lower() not in {"0", "false", "no", "off"}


_VERBOSE = _truthy("SAMV5_VERBOSE", True)


def _log(policy, stage, message, **fields):
    if not (policy or {}).get("verbose", _VERBOSE):
        return
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"SAMV5 {stage} policy={POLICY_VERSION_V5} {message} {suffix}".rstrip())


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _clamp(value, low, high):
    return max(low, min(high, value))


def _odd(value):
    value = max(1, int(round(value)))
    return value if value % 2 == 1 else value + 1


def _as_bbox(value):
    if value is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_intersection(a, b):
    return [
        max(a[0], b[0]), max(a[1], b[1]),
        min(a[2], b[2]), min(a[3], b[3]),
    ]


def _bbox_union(a, b):
    return [
        min(a[0], b[0]), min(a[1], b[1]),
        max(a[2], b[2]), max(a[3], b[3]),
    ]


def _bbox_area(box):
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _mask_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    ys = np.flatnonzero(rows)
    xs = np.flatnonzero(cols)
    return [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]


def _apply_box_edges_to_alpha(alpha, box):
    """Antialias the crop border, but only where a subject pixel is cut.

    A crop line crossing solid subject shows the cut, which looks like a hard
    slice; a one-pixel ramp hides it. A crop line crossing already-transparent
    pixels is untouched, so no background leaks in at the border.
    """
    if alpha is None or box is None:
        return alpha
    height, width = alpha.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    ramp = 0.55
    if x1 > 0:
        column = alpha[y1:y2, x1]
        column[:] = np.where(column > 0.05, np.minimum(column, ramp), column)
    if x2 < width:
        column = alpha[y1:y2, x2 - 1]
        column[:] = np.where(column > 0.05, np.minimum(column, ramp), column)
    if y1 > 0:
        row = alpha[y1, x1:x2]
        row[:] = np.where(row > 0.05, np.minimum(row, ramp), row)
    if y2 < height:
        row = alpha[y2 - 1, x1:x2]
        row[:] = np.where(row > 0.05, np.minimum(row, ramp), row)
    return alpha


def _window_bbox(clamp_box, mask, shape, margin):
    height, width = shape[:2]
    mask_box = _mask_bbox(mask)
    if mask_box is not None:
        clamp_box = _bbox_union(clamp_box, mask_box) if clamp_box else mask_box
    diag = math.hypot(width, height)
    pad = max(4.0, diag * margin)
    x1 = int(_clamp(math.floor(clamp_box[0] - pad), 0, width - 1))
    y1 = int(_clamp(math.floor(clamp_box[1] - pad), 0, height - 1))
    x2 = int(_clamp(math.ceil(clamp_box[2] + pad), x1 + 2, width))
    y2 = int(_clamp(math.ceil(clamp_box[3] + pad), y1 + 2, height))
    return [x1, y1, x2, y2]


def _window_mask(window, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = window
    mask[y1:y2, x1:x2] = 1
    return mask.astype(bool)


def _border_band(window, shape, thickness):
    band = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = window
    t = max(1, int(thickness))
    band[y1:min(y2, y1 + t), x1:x2] = 1
    band[max(y1, y2 - t):y2, x1:x2] = 1
    band[y1:y2, x1:min(x2, x1 + t)] = 1
    band[y1:y2, max(x1, x2 - t):x2] = 1
    return band.astype(bool)


def _disc(radius):
    radius = max(1, int(round(radius)))
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )


def _dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), _disc(radius)) > 0


def _erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), _disc(radius)) > 0


# ---------------------------------------------------------------------------
# Image evidence
# ---------------------------------------------------------------------------


def _lab_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)


def _gradient_magnitude(lab):
    """Edge strength in lightness plus chroma, so colour-only borders count."""
    luma = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    luma_grad = cv2.magnitude(gx, gy)

    chroma = lab[:, :, 1:3]
    chroma_grad = np.zeros_like(luma_grad)
    for channel in range(2):
        plane = cv2.GaussianBlur(chroma[:, :, channel], (3, 3), 0)
        cx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        cy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        chroma_grad = np.maximum(chroma_grad, cv2.magnitude(cx, cy))
    # Luma edges dominate; chroma adds colour-only boundaries at half weight.
    return luma_grad + 0.5 * chroma_grad


def _kmeans_palette(lab, pixels, clusters, seed=1234):
    samples = lab[pixels]
    if samples.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(samples) > 16000:
        rng = np.random.RandomState(seed)
        samples = samples[rng.choice(len(samples), 16000, replace=False)]
    clusters = int(min(clusters, max(1, len(samples) // 24)))
    if clusters < 2:
        return np.median(samples, axis=0)[None, :].astype(np.float32)
    cv2.setRNGSeed(seed)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0
    )
    _, _, centers = cv2.kmeans(
        samples.astype(np.float32), clusters, None, criteria, 2,
        cv2.KMEANS_PP_CENTERS
    )
    return centers.astype(np.float32)


def _palette_distance(lab, palette):
    if palette is None or len(palette) == 0:
        return np.full(lab.shape[:2], 1e6, dtype=np.float32)
    distance = np.full(lab.shape[:2], 1e6, dtype=np.float32)
    for center in palette:
        delta = lab - center.reshape(1, 1, 3)
        distance = np.minimum(distance, np.linalg.norm(delta, axis=2))
    return distance


def _weighted_mean(lab, weights, ksize):
    """Local mean of ``lab`` using non-negative weights, normalised by weight."""
    weights = weights.astype(np.float32)
    ksize = _odd(ksize)
    numerator = cv2.boxFilter(
        lab * weights[:, :, None], -1, (ksize, ksize), normalize=False
    )
    denominator = cv2.boxFilter(weights, -1, (ksize, ksize), normalize=False)
    result = numerator / np.maximum(denominator, 1e-3)[:, :, None]
    # Pixels with no support fall back to the global mean of the weighted set.
    total = float(weights.sum())
    if total > 0:
        fallback = (lab * weights[:, :, None]).sum(axis=(0, 1)) / total
    else:
        fallback = np.median(lab.reshape(-1, 3), axis=0)
    empty = denominator <= 1e-3
    result[empty] = fallback
    return result, denominator


def _color_models(lab, mask, window, scale, exclusion_mask=None):
    """Build local foreground/background palettes from the analysis window.

    Background comes from the ring just outside the accepted mask. Foreground
    is then built in a second pass from mask pixels that are *dissimilar* to
    that background: a proposal that included a slab of backdrop would
    otherwise contribute it to the foreground palette and collapse the
    palette separation, disabling cleanup. Both models stay local to the
    window, so a global scene statistic cannot decide a small object's fate.
    """
    x1, y1, x2, y2 = window
    crop = lab[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2].astype(bool)
    if not crop_mask.any():
        return None

    core_radius = max(1, int(round(scale * 0.002)))
    inner = _erode(crop_mask, core_radius)
    if int(inner.sum()) < 32:
        inner = crop_mask.copy()
    # Neighbours may not teach the target's foreground palette. Their boxes
    # still cannot delete pixels without independent image evidence.
    if exclusion_mask is not None:
        owned = inner & ~exclusion_mask[y1:y2, x1:x2]
        if int(owned.sum()) >= max(32, int(inner.sum() * 0.20)):
            inner = owned

    # Background is sampled from the window border, not from everything the
    # proposal left out. A subject part the proposal missed - a plate rim, a
    # garnish, the second half of a cabinet - also sits outside the mask, so
    # "outside the mask" is not the same as "backdrop". Feeding those pixels to
    # the background model teaches it that part of the subject *is* backdrop,
    # and from then on no stage can recover that part: cleanup deletes it,
    # completion treats it as background, and hole repair reads a missed part as
    # a real opening. The window border is the one region that is context by
    # construction, because the analysis window is always padded past the mask.
    border_thickness = max(2, int(round(min(crop.shape[:2]) * 0.08)))
    border = np.zeros(crop.shape[:2], dtype=bool)
    border[:border_thickness] = True
    border[-border_thickness:] = True
    border[:, :border_thickness] = True
    border[:, -border_thickness:] = True
    outer = border & ~_dilate(crop_mask, max(1, core_radius))
    if exclusion_mask is not None:
        outer &= ~exclusion_mask[y1:y2, x1:x2]
    if int(outer.sum()) < 32:
        # The subject fills the window and leaves no border context; fall back
        # to the ring just outside the mask.
        outer = ~_dilate(crop_mask, max(2, core_radius * 2))
        if exclusion_mask is not None:
            outer &= ~exclusion_mask[y1:y2, x1:x2]
    if int(outer.sum()) < 32:
        outer = ~_dilate(crop_mask, max(3, core_radius * 3))

    bg_palette = _kmeans_palette(crop, outer, 3)
    bg_dist = _palette_distance(crop, bg_palette)

    # Second pass: keep only mask pixels that do not resemble the local
    # background. The threshold is a percentile of the inner distribution, so
    # the split follows the image rather than a fixed colour distance.
    fg_palette = np.empty((0, 3), dtype=np.float32)
    for fraction in (0.35, 0.15, 0.0):
        if int(inner.sum()) < 32:
            break
        threshold = float(np.percentile(bg_dist[inner], fraction * 100.0))
        seed = inner & (bg_dist >= threshold) if threshold > 0 else inner
        if int(seed.sum()) < max(32, int(inner.sum() * 0.25)):
            continue
        fg_palette = _kmeans_palette(crop, seed, 8)
        break
    if len(fg_palette) == 0:
        fg_palette = _kmeans_palette(crop, inner, 8)

    fg_dist = _palette_distance(crop, fg_palette)

    # Palette separation tells the stages how much colour evidence exists.
    separation = 0.0
    if len(fg_palette) and len(bg_palette):
        best = 1e6
        for fg in fg_palette:
            for bg in bg_palette:
                best = min(best, float(np.linalg.norm(fg - bg)))
        separation = best

    shape = mask.shape
    return {
        "window": window,
        "fgPalette": fg_palette,
        "bgPalette": bg_palette,
        # Crop-local maps for measurement stages that work in window space.
        "fgDist": fg_dist,
        "bgDist": bg_dist,
        # Full-image maps for stages that reason about whole-mask geometry.
        "fgDistFull": _full_map(fg_dist, window, shape),
        "bgDistFull": _full_map(bg_dist, window, shape),
        "separation": separation,
        "corePixels": int(inner.sum()),
        "backgroundPixels": int(outer.sum()),
    }


def _full_map(crop_map, window, shape, fill=1e6):
    out = np.full(shape[:2], fill, dtype=np.float32)
    x1, y1, x2, y2 = window
    out[y1:y2, x1:x2] = crop_map
    return out


def _models_full(models, shape):
    return {
        "fgDist": _full_map(models["fgDist"], models["window"], shape),
        "bgDist": _full_map(models["bgDist"], models["window"], shape),
    }


def _classify_edge(lab, mask, models, gradient, window, scale):
    """Measure edge softness from colour mixture across the boundary.

    A hard edge has a narrow transition: band pixels are clearly foreground or
    clearly background. A soft edge has many pixels that are a measurable
    mixture. The measurement also sets the band radius used by alpha
    estimation, so the same rule adapts to resolution.
    """
    x1, y1, x2, y2 = window
    crop = lab[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2].astype(bool)
    band_radius = max(4, int(round(scale * _EDGE_BAND_RATIO)))
    band = _dilate(crop_mask, band_radius) & ~_erode(crop_mask, band_radius)
    if int(band.sum()) < 16:
        return {
            "class": "hard", "radius": 2, "fractionMid": 0.0,
            "contrast": 0.0, "bandPixels": 0, "reason": "empty_band",
        }

    core = _erode(crop_mask, band_radius)
    if int(core.sum()) < 16:
        core = crop_mask
    outside = _dilate(crop_mask, band_radius * 2) & ~_dilate(crop_mask, band_radius)

    foreground, fg_support = _weighted_mean(crop, core, band_radius * 2 + 1)
    background, bg_support = _weighted_mean(crop, outside, band_radius * 2 + 1)

    vector = foreground - background
    denom = np.sum(vector * vector, axis=2)
    confident = (denom > 64) & (fg_support > 0) & (bg_support > 0)
    mix = np.zeros(crop.shape[:2], dtype=np.float32)
    safe = np.where(denom > 1e-3, denom, 1.0)
    mix = np.clip(
        np.sum((crop - background) * vector, axis=2) / safe, 0.0, 1.0
    )

    band_pixels = int(band.sum())
    mid = band & confident & (mix > 0.15) & (mix < 0.85)
    fraction_mid = float(mid.sum()) / max(1, band_pixels)
    contrast = float(np.median(np.sqrt(denom[band & confident]))) \
        if int((band & confident).sum()) else 0.0

    # Transition width in pixels per unit of boundary, not as a share of the
    # measurement band. A share depends on how wide the aperture happens to be:
    # the same one-pixel antialias reads as "soft" when measured narrowly and as
    # "hard" when measured widely, while a real feather - a ramp spread over
    # several pixels - collapses towards zero as soon as the aperture widens.
    # Counting mixture pixels per boundary pixel recovers the ramp width itself,
    # which is what alpha estimation needs and what "feather" actually means.
    boundary = crop_mask & ~_erode(crop_mask, 1)
    perimeter = max(1, int(boundary.sum()))
    width = float(mid.sum()) / perimeter
    # More lenient soft edge detection for feathers, glass, etc.
    # Original: width >= 2.0 or contrast < 16.0
    # New: width >= 1.0 or contrast < 20.0
    if width >= 1.0 or contrast < 20.0:
        edge_class = "soft"
        radius = int(_clamp(round(width) + 1, 3, 10))
    else:
        # A hard edge keeps a fixed narrow refine radius. The band radius above
        # is a measurement aperture only; letting it scale with the image would
        # spread partial alpha several pixels beyond the silhouette.
        edge_class = "hard"
        radius = 2 if width >= 0.8 else 1

    return {
        "class": edge_class,
        "radius": int(radius),
        "fractionMid": round(fraction_mid, 4),
        "transitionWidth": round(width, 2),
        "contrast": round(contrast, 2),
        "bandPixels": band_pixels,
        "bandRadius": band_radius,
        "reason": "colour_mixture_width",
    }


def _classify_material(lab, mask, models, gradient, edge, policy, scale):
    """Decide whether translucent alpha is supported by pixels.

    Translucency shows up as a large interior share whose colour is closer to
    the local background than to the foreground palette, while the object still
    has its own boundary. A semantic assertion only lowers the bar; it cannot
    enable translucent alpha on its own.
    """
    x1, y1, x2, y2 = models["window"]
    crop_mask = mask[y1:y2, x1:x2].astype(bool)
    core = _erode(crop_mask, max(2, int(round(scale * 0.003))))
    if int(core.sum()) < 64:
        core = crop_mask
    if int(core.sum()) == 0:
        return {"mode": "opaque", "bgShare": 0.0, "reason": "empty_core"}

    fg_dist = models["fgDist"]
    bg_dist = models["bgDist"]
    bg_share = float(np.mean(bg_dist[core] < fg_dist[core]))
    core_contrast = float(np.median(fg_dist[core]))
    asserted = bool(policy.get("allowTranslucent"))
    separation = float(models.get("separation") or 0.0)

    # A translucent interior is one where a large share of the core reads closer
    # to the backdrop than to the subject *and* the two models are actually
    # distinguishable: without palette separation, "closer to background" is
    # noise, and a white cabinet against a white wall would be painted half
    # transparent. The outline condition that used to sit here required a wide
    # colour ramp, which a clean pane framed by a hard edge never has - that is
    # why glass fell back to fully opaque no matter what it looked like.
    minimum_share = 0.30 if asserted else 0.55
    supported = bg_share >= minimum_share and separation >= 12.0
    return {
        "mode": "translucent" if supported else "opaque",
        "bgShare": round(bg_share, 4),
        "coreContrast": round(core_contrast, 2),
        "asserted": asserted,
        "reason": (
            "interior_matches_background" if supported
            else "asserted_not_supported_by_pixels" if asserted
            else "opaque_interior"
        ),
    }


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def normalize_candidates_v5(candidate_masks, width, height):
    """Normalize arbitrary SAM/FastSAM candidate arrays to full-size bool."""
    normalized = []
    for candidate in ([] if candidate_masks is None else candidate_masks):
        array = np.asarray(candidate)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 2:
            continue
        if array.shape != (height, width):
            array = cv2.resize(
                array.astype(np.float32), (width, height),
                interpolation=cv2.INTER_NEAREST
            )
        if array.ndim != 2 or not np.isfinite(array).all():
            continue
        binary = array > 0.5
        if binary.any():
            normalized.append(binary)
    return normalized


def _boundary_alignment(mask, gradient, radius=2):
    """Boundary strength of a candidate (raw contrast, not yet normalized).

    Each candidate is normalized against the same image reference, so the value
    is comparable across proposals of different size.
    """
    band = _dilate(mask, radius) & ~_erode(mask, radius)
    if int(band.sum()) == 0:
        return 0.0
    return float(np.mean(gradient[band]))


def _frame_contact(mask, thickness=2):
    """Share of the mask boundary that lies on (or within) the image border.

    A proposal that runs to the frame edge is usually the subject merged with
    its surroundings; the scene's own frame cut is rarely the object boundary.
    """
    height, width = mask.shape
    t = max(1, int(thickness))
    edges = np.zeros_like(mask, dtype=bool)
    edges[:t, :] = True
    edges[-t:, :] = True
    edges[:, :t] = True
    edges[:, -t:] = True
    contact = _dilate(mask, 1) & ~_erode(mask, 1) & edges
    boundary = _dilate(mask, 1) & ~_erode(mask, 1)
    total = int(boundary.sum())
    if total == 0:
        return 0.0
    return float(contact.sum()) / total


def _detect_intrusion_in_candidate(mask, hint_bbox, shape):
    """检测候选是否包含边界侵入物。

    两种方法：
    1. 旧方法：薄物体跨越hint边界
    2. 新方法：直接查找hint框外的细小/长条区域

    返回：(入侵像素数, 入侵像素掩码)
    """
    try:
        import cv2
        import numpy as np

        hint = [int(round(v)) for v in hint_bbox]
        height, width = shape[:2]
        shape = (height, width)

        # 创建提示框区域
        hint_region = np.zeros(shape, dtype=bool)
        hx1 = max(0, hint[0])
        hy1 = max(0, hint[1])
        hx2 = min(width, hint[2])
        hy2 = min(height, hint[3])
        if hx2 <= hx1 or hy2 <= hy1:
            return 0, None

        hint_region[hy1:hy2, hx1:hx2] = True

        # 方法1：快速直接检查 - hint框外的所有mask像素
        mask_outside_hint = mask & ~hint_region
        outside_hint_pixels = int(mask_outside_hint.sum())

        # 如果hint框外没有像素，直接返回
        if outside_hint_pixels == 0:
            return 0, None

        # 找出hint框外的连通组件
        outside_uint8 = mask_outside_hint.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(outside_uint8, connectivity=8)

        intrusion_mask = np.zeros(shape, dtype=bool)
        total_intrusion_pixels = 0

        for label in range(1, num_labels):
            component_mask = labels == label
            area = int(component_mask.sum())

            # 检查组件是否是我们要移除的类型
            # 条件1：面积较小（小于主体面积的5%）
            total_mask_area = int(mask.sum())
            if area > total_mask_area * 0.05:
                # 太大，可能不是临近物而是主体的一部分
                continue

            # 条件2：检查组件的形状（长宽比高的很可能是临近物）
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            # 长宽比判断：如果是细长的，很可能是临近物
            aspect_ratio = max(w, h) / max(1, min(w, h))
            is_thin_and_long = aspect_ratio > 3.0

            # 条件3：检查组件是否与hint框接触
            component_touching_hint = np.zeros_like(component_mask)
            # 扩展hint边界1像素
            expanded_hint = hint_region.copy()
            expanded_hint = cv2.dilate(expanded_hint.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

            touches_hint = (component_mask & expanded_hint).any()

            # 条件4：检查组件是否"薄"（形态学开运算会移除很多）
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(component_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            opened_area = int(opened.sum())
            is_thin = opened_area < area * 0.5  # 开运算移除超过一半

            # 决定是否移除：
            # 如果是细长的且接触hint框，或者是薄的且接触hint框
            if (touches_hint and (is_thin_and_long or is_thin)) or area < 100:
                intrusion_mask |= component_mask
                total_intrusion_pixels += area
                print(f"DEBUG INTRUSION: component {label}: area={area}, aspect={aspect_ratio:.1f}, "
                      f"thin={is_thin}, touches_hint={touches_hint}, will remove", file=sys.stderr)

        # 此外，也检查hint框边缘附近的薄物体
        if total_intrusion_pixels < 20:  # 如果没找到足够的入侵，尝试边缘检查
            # 创建边缘带（hint框内外各2像素）
            edge_band = np.zeros(shape, dtype=bool)
            edge_width = 2
            # 上边缘
            edge_band[max(0, hy1-edge_width):min(height, hy1+edge_width), max(0, hx1):min(width, hx2)] = True
            # 下边缘
            edge_band[max(0, hy2-edge_width):min(height, hy2+edge_width), max(0, hx1):min(width, hx2)] = True
            # 左边缘
            edge_band[max(0, hy1):min(height, hy2), max(0, hx1-edge_width):min(width, hx1+edge_width)] = True
            # 右边缘
            edge_band[max(0, hy1):min(height, hy2), max(0, hx2-edge_width):min(width, hx2+edge_width)] = True

            mask_in_edge = mask & edge_band
            if mask_in_edge.any():
                # 边缘上的像素全部当作入侵（很可能是临近物）
                edge_intrusion = mask_in_edge & ~hint_region  # 只取边缘上的hint框外部分
                edge_intrusion_count = int(edge_intrusion.sum())
                if edge_intrusion_count > 0:
                    intrusion_mask |= edge_intrusion
                    total_intrusion_pixels += edge_intrusion_count
                    print(f"DEBUG INTRUSION: added {edge_intrusion_count} edge pixels", file=sys.stderr)

        total_intrusion_pixels = int(intrusion_mask.sum())
        return total_intrusion_pixels, intrusion_mask if total_intrusion_pixels > 0 else None

    except Exception as e:
        print(f"DEBUG: Error detecting intrusion: {e}", file=sys.stderr)
        return 0, None


def _score_candidates(image, candidates, hint_bbox, policy, shape, detail=False):
    """Rank candidates by hint coverage, size agreement and edge alignment.

    This is the only selection rule: no object category participates. It is
    deliberately sensitive to over-large proposals because accepting one is the
    main way background leaks into a layer.

    新增：检测是否有边界侵入（临近物），优先选择没有侵入的候选
    """
    lab = _lab_image(image)
    gradient = _gradient_magnitude(lab)
    hint_area = max(1.0, _bbox_area(hint_bbox))
    height, width = shape[:2]
    frame_area = float(width * height)
    hint_x1, hint_y1, hint_x2, hint_y2 = [int(v) for v in hint_bbox]
    center = ((hint_x1 + hint_x2) / 2, (hint_y1 + hint_y2) / 2)
    # One contrast reference for the whole image, so alignment scores are
    # comparable between a tight candidate and a flooded one.
    contrast_scale = max(1e-3, float(np.percentile(gradient, 95)))
    rows = []
    for index, mask in enumerate(candidates):
        area = int(mask.sum())
        if area == 0:
            continue
        crop = mask[hint_y1:hint_y2, hint_x1:hint_x2]
        coverage = float(crop.sum()) / hint_area
        includes_center = bool(
            0 <= int(center[1]) < height and 0 <= int(center[0]) < width and
            mask[int(center[1]), int(center[0])]
        )
        mask_box = _mask_bbox(mask) or [0, 0, width, height]
        filled = area / max(1.0, _bbox_area(mask_box))
        contact = _frame_contact(mask)
        # Size agreement: area may not greatly exceed the hinted region. The
        # subject may legitimately extend beyond the hint, so the ratio is
        # capped softly rather than rejected outright.
        oversize = max(0.0, area / hint_area - 1.0)
        oversize_penalty = min(1.0, oversize / 1.5)
        alignment = _boundary_alignment(mask, gradient)
        contrast = min(1.0, alignment / contrast_scale)
        # A candidate that is nearly all pixels inside its own box is almost
        # always a merged slab; the subject silhouette has concavities.
        slab_penalty = 0.0
        if filled > 0.90:
            slab_penalty = min(1.0, (filled - 0.90) / 0.10)
        # 检查是否有边界侵入（薄物体跨越hint边界）
        # _detect_intrusion_in_candidate 返回 (intrusion_count, intrusion_mask)
        intrusion_result = _detect_intrusion_in_candidate(mask, hint_bbox, shape[:2])
        if isinstance(intrusion_result, tuple) and len(intrusion_result) == 2:
            intrusion_pixels, _ = intrusion_result
            intrusion_pixels = int(intrusion_pixels)
        else:
            # 向后兼容：如果返回单个值
            intrusion_pixels = int(intrusion_result) if not isinstance(intrusion_result, tuple) else 0
        intrusion_ratio = intrusion_pixels / max(1, area) if intrusion_pixels > 0 else 0.0
        intrusion_penalty = 0.0

        if intrusion_pixels > max(20, area * 0.02):  # 至少有20像素或掩码面积的2%
            # 有显著侵入，大惩罚
            import sys
            intrusion_penalty = 0.6 + min(0.4, intrusion_ratio * 10)
            print(f"DEBUG V5 INTRUSION CHECK: candidate {index} has {intrusion_pixels} intrusion pixels "
                  f"({intrusion_ratio:.2%}), penalty={intrusion_penalty:.2f}", file=sys.stderr)

        score = (
            1.6 * min(1.0, coverage) +
            0.6 * (1.0 if includes_center else 0.0) +
            1.8 * contrast -
            1.2 * oversize_penalty -
            1.0 * slab_penalty -
            0.8 * min(1.0, contact / 0.10) -
            intrusion_penalty
        )
        inside = float(crop.sum()) / max(1, area)
        issues = []
        if score < 0.35:
            issues.append("low_candidate_score")
        if inside < 0.55:
            issues.append("candidate_outside_target")
        if coverage < 0.005:
            issues.append("candidate_misses_target")
        if contact > 0.80 and filled > 0.98 and contrast < 0.10:
            issues.append("unbounded_scene_candidate")
        rows.append({
            "index": index,
            "area": area,
            "bbox": mask_box,
            "coverage": round(coverage, 4),
            "fill": round(filled, 4),
            "frameContact": round(contact, 4),
            "includesCenter": includes_center,
            "alignment": round(alignment, 4),
            "score": round(score, 4),
            "intrusionPixels": intrusion_pixels,
            "intrusionRatio": round(intrusion_ratio, 4),
            "insideRatio": round(inside, 4),
            "issues": issues,
        })
    # A sparse subject can span its entire hint without filling it or crossing
    # its centre. Prefer image-supported, spatially complete nested proposals
    # over a weaker contour that annexes large regions beyond that silhouette.
    # Equal-extent part/whole masks (e.g. an outline versus its interior) do
    # not provide this evidence and must retain the normal completeness rules.
    hw, hh = max(1, hint_x2-hint_x1), max(1, hint_y2-hint_y1)
    for row in rows:
        penalty = 0.0
        witness = None
        for other in rows:
            if other['issues'] or other['area']*1.5 > row['area']:
                continue
            x, y, X, Y = other['bbox']
            extent_x = max(0, min(X,hint_x2)-max(x,hint_x1))/hw
            extent_y = max(0, min(Y,hint_y2)-max(y,hint_y1))/hh
            if min(extent_x, extent_y) < .85:
                continue
            a, b, A, B = row['bbox']
            extension = max((x-a)/hw, (A-X)/hw, (y-b)/hh, (B-Y)/hh)
            if extension < .10 or other['alignment'] < row['alignment']*1.35:
                continue
            small, large = candidates[other['index']], candidates[row['index']]
            overlap = (small & large).sum()/max(1, other['area'])
            if overlap < .90:
                # Opposite-polarity masks may surround the same supported
                # contour instead of containing its foreground. Require an
                # almost disjoint interior and shared contour, not merely a
                # different object somewhere inside the same broad rectangle.
                rim = small & ~_erode(small, 1)
                uncertainty = int(np.clip(round(math.hypot(height,width)*.005), 2, 8))
                adjacent = (rim & _dilate(large, uncertainty)).sum()/max(1, rim.sum())
                if overlap > .10 or adjacent < .75:
                    continue
            penalty = max(penalty, 1.2)
            witness = other['index']
        row['backgroundExpansionPenalty'] = penalty
        row['extentWitnessIndex'] = witness
        row['score'] = round(row['score']-penalty, 4)
    rows.sort(key=lambda row: (not row["issues"], row["score"], row['alignment']), reverse=True)
    _log(policy, "candidates", "ranked", count=len(rows),
         top=(rows[0] if rows else None))
    if not detail:
        # Best-only information is enough for the acceptance report.
        top = rows[0] if rows else None
        return [dict(top, acceptance="selected")] + [
            dict(row, acceptance="rejected") for row in rows[1:]
        ]
    return rows


def score_candidates_v5(image, candidates, hint_bbox, policy, shape):
    """Public ranking used by the offline harness and the API."""
    return _score_candidates(image, candidates, hint_bbox, policy, shape, detail=True)


def select_candidate_v5(image, candidates, hint_bbox, policy, shape):
    rows = score_candidates_v5(image, candidates, hint_bbox, policy, shape)
    if not rows:
        return None, []
    return candidates[rows[0]["index"]], rows


# ---------------------------------------------------------------------------
# Region schedule
# ---------------------------------------------------------------------------


class EvidenceSchedule:
    """Pixels that edits may not touch, and which edits were applied."""

    def __init__(self, protected, forbidden, policy, soft_boxes=None):
        self.protected = protected.astype(bool)
        self.forbidden = forbidden.astype(bool)
        # Soft sibling boxes never remove pixels. They can only reject a whole
        # candidate component that sits almost entirely inside one of them.
        self.soft_boxes = list(soft_boxes or [])
        self.policy = policy
        self.audit = []

    def record(self, stage, action="applied", **fields):
        entry = {"stage": stage, "action": action}
        entry.update(fields)
        self.audit.append(entry)
        _log(self.policy, stage, "applied", **fields)

    def record_skip(self, stage, reason, **fields):
        entry = {"stage": stage, "skipped": reason}
        entry.update(fields)
        self.audit.append(entry)
        _log(self.policy, stage, "skipped", reason=reason, **fields)

    def veto_share(self, pixels, exclusion_mask):
        if exclusion_mask is None or not pixels.any():
            return 0.0
        count = int(pixels.sum())
        if count == 0:
            return 0.0
        return float((pixels & exclusion_mask).sum()) / count

    def soft_share(self, pixels):
        """Largest share of ``pixels`` that falls inside one soft sibling box."""
        if not self.soft_boxes or not pixels.any():
            return 0.0
        ys, xs = np.where(pixels)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        best = 0.0
        for box in self.soft_boxes:
            ix1, iy1 = max(x1, box[0]), max(y1, box[1])
            ix2, iy2 = min(x2, box[2]), min(y2, box[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            count = int(pixels[iy1:iy2, ix1:ix2].sum())
            best = max(best, count / max(1, int(pixels.sum())))
        return best

    def vetoed(self, pixels):
        """A component is vetoed only by explicit or near-total sibling overlap."""
        if self.veto_share(pixels, self.forbidden) > _EXCLUSION_VETO_SHARE:
            return True
        return self.soft_share(pixels) >= 0.90


def _prior_box_pixels(box, shape):
    """Convert a 0-1000 normalized prior box to pixel coordinates."""
    height, width = shape[:2]
    x1 = int(_clamp(box[0] / 1000.0 * width, 0, width))
    y1 = int(_clamp(box[1] / 1000.0 * height, 0, height))
    x2 = int(_clamp(box[2] / 1000.0 * width, 0, width))
    y2 = int(_clamp(box[3] / 1000.0 * height, 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


_HARD_PRIOR_SOURCES = {"intent_excluded", "context_strict"}


def _split_priors(policy, shape):
    """Split caller priors into a hard veto map and soft sibling boxes.

    A hard prior is an explicit exclusion (a badge the user marked, a strict
    ownership box). Its pixels are never added to the subject.

    A soft prior is just another layer's rectangle. Background plates and
    sibling objects routinely cover most of the frame, and vetoing every pixel
    they touch would make completion impossible; instead a soft box can only
    reject a candidate component that is almost entirely inside it.
    """
    forbidden = np.zeros(shape[:2], dtype=bool)
    soft_boxes = []
    frame_area = float(shape[0] * shape[1])
    for prior in policy.get("exclusionPriors") or []:
        box = _as_bbox(prior.get("bbox"))
        if box is None:
            continue
        pixels = _prior_box_pixels(box, shape)
        if pixels is None:
            continue
        source = str(prior.get("source") or "")
        area = float((pixels[2] - pixels[0]) * (pixels[3] - pixels[1]))
        if source in _HARD_PRIOR_SOURCES:
            forbidden[pixels[1]:pixels[3], pixels[0]:pixels[2]] = True
            continue
        # A box that spans most of the frame cannot tell ownership apart.
        if area >= frame_area * 0.25:
            continue
        soft_boxes.append(pixels)
    return forbidden, soft_boxes


def _build_protected(mask, policy, scale):
    """Protect the accepted mask core from removal stages."""
    return _erode(mask, max(1, int(round(scale * 0.0015))))


def _reachable_from(passable, seeds):
    """Connected components of ``passable`` that contain a seed pixel."""
    if not passable.any() or not seeds.any():
        return np.zeros_like(passable)
    count, labels = cv2.connectedComponents(passable.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(passable)
    seeded = np.unique(labels[seeds & passable])
    seeded = seeded[seeded > 0]
    if seeded.size == 0:
        return np.zeros_like(passable)
    return np.isin(labels, seeded)


def _components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    rows = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        rows.append({
            "label": label,
            "area": area,
            "x": int(stats[label, cv2.CC_STAT_LEFT]),
            "y": int(stats[label, cv2.CC_STAT_TOP]),
            "w": int(stats[label, cv2.CC_STAT_WIDTH]),
            "h": int(stats[label, cv2.CC_STAT_HEIGHT]),
        })
    return labels, rows


def _component_edge_support(component, gradient, threshold):
    if not component.any():
        return 0.0
    boundary = _dilate(component, 1) & ~_erode(component, 1)
    if int(boundary.sum()) == 0:
        return 0.0
    return float(np.mean(gradient[boundary] >= threshold))


# ---------------------------------------------------------------------------
# Stage: background cleanup
# ---------------------------------------------------------------------------


def review_background_v5_v1_style(image, mask, hint_bbox, schedule, policy):
    """Compatibility entry point for the same category-independent V5 cleanup."""
    lab = _lab_image(image)
    scale = math.hypot(*image.shape[:2])
    window = _window_bbox(hint_bbox, mask, image.shape, _WINDOW_MARGIN_RATIO)
    models = _color_models(lab, mask, window, scale, schedule.forbidden)
    if models is None:
        return mask
    gradient = _gradient_magnitude(lab)
    edge = _classify_edge(lab, mask, models, gradient, window, scale)
    return review_background_v5(lab, mask, models, gradient, window, schedule, edge, policy)

def review_background_v5(lab, mask, models, gradient, window, schedule, edge, policy):
    """Remove backdrop that the proposal absorbed inside the subject outline.

    This is the only background cleanup stage. It floods outward from the
    backdrop and keeps only what that flood reaches inside the mask, so colour
    alone never deletes a pixel: strong image edges are walls, and the subject's
    own palette stops the flood. Negative space seen through a chair frame is
    removed because it opens to the backdrop through the frame's gaps, while a
    leg resting on a similar rug is kept because the silhouette edge between
    them separates the two.

    The protected core is deliberately not consulted here. Absorbed backdrop
    routinely sits deep inside the silhouette, which is exactly the region that
    erosion protects, so protecting it would restore the bug this stage exists
    to fix. The subject model (``fg_like``), the barriers and the per-component
    guard below carry that job instead.
    """
    shape = mask.shape
    window_pixels = _window_mask(window, shape)
    x1, y1, x2, y2 = window
    crop_gradient = gradient[y1:y2, x1:x2]
    scale = float(np.percentile(crop_gradient, 75)) if crop_gradient.size else 0.0
    edge_threshold = float(_clamp(scale, 14.0, 40.0))

    separation = float(models.get("separation") or 0.0)
    fg_dist = models["fgDistFull"]
    bg_dist = models["bgDistFull"]
    # Without palette separation there is no colour evidence to act on. The bar
    # is deliberately low: a white cabinet against a white wall separates by
    # barely ten levels, and refusing to work at all left the absorbed backdrop
    # completely untouched. Local thresholds do the real filtering below, and
    # the flood can only remove what is reachable from the backdrop anyway.
    if separation < _MIN_PALETTE_SEPARATION:
        schedule.record_skip("background", "low_palette_separation",
                             separation=round(separation, 2))
        return mask

    bg_threshold = float(_clamp(0.35 * separation, 6.0, 12.0))
    bg_like = (bg_dist < bg_threshold) & window_pixels
    # Pixels that are themselves close to the foreground palette are never
    # classified as background, even if they also resemble it slightly.
    fg_like = (fg_dist < max(4.0, 0.45 * separation)) & window_pixels

    barrier = gradient >= edge_threshold
    # Evidence maps are local to the window; outside it no colour decision is
    # trustworthy, so restrict every classification to the window.
    local = window_pixels
    bg_like &= local
    fg_like &= local
    # Only unambiguous backdrop is passable: close to the background palette,
    # far from the subject palette, and not across a strong image edge.
    passable = local & bg_like & ~fg_like & ~barrier

    # Backdrop the proposal absorbed is still connected to the backdrop
    # *outside* the subject through pixels of its own kind, so the search is a
    # flood from the outside rather than a per-pixel classification. Absorbed
    # tablecloth, floor and wall are reachable; so is negative space seen
    # through a chair frame, because it opens to the outside through the
    # frame's own gaps. A subject region that merely resembles the backdrop is
    # not reachable across the silhouette's gradient wall, which is what keeps
    # a white plate on a white cloth and a dark leg on a dark rug.
    #
    # The previous version looked only at the subject's own boundary rim
    # (``mask & ~erode(mask)``) and deleted background-coloured pixels there.
    # That cannot clean a backdrop: the absorbed region is deep inside the
    # silhouette, and the eroded core excluded it by construction, so the stage
    # ran, reported a removal, and left the background in place. Deleting every
    # background-coloured pixel inside the mask would be worse in the other
    # direction - it is exactly how the plate of a plated dish disappears.
    seeds = passable & ~mask
    if not seeds.any():
        schedule.record_skip("background", "no_outside_backdrop_seed",
                             edgeThreshold=round(edge_threshold, 1))
        return mask
    reachable = _reachable_from(passable, seeds)
    # The flood may *pass* through the silhouette rim but may not delete it. An
    # anti-aliased or shadowed outline is the one place a subject's pixels
    # genuinely do resemble the backdrop, and without this the stage nibbles a
    # gash out of the subject's own edge - the failure mode that made the result
    # look worse than doing nothing. Absorbed backdrop sits behind that rim, and
    # the flood still reaches it.
    # Reachability and foreground distance already protect the true contour.
    # Keeping the proposal rim left a visible frame of absorbed backdrop.
    removal = reachable & mask
    # Do not turn surface noise/highlights into scattered pinholes.
    removal_labels, removal_rows = _components(removal, max(16, int(mask.sum() * 0.0005)))
    removal &= np.isin(removal_labels, [row["label"] for row in removal_rows])
    if not removal.any():
        schedule.record_skip("background", "no_interior_background",
                             edgeThreshold=round(edge_threshold, 1))
        return mask

    total = int(mask.sum())
    removed = int(removal.sum())
    if removed == 0:
        schedule.record_skip("background", "only_protected_pixels")
        return mask
    if removed > total * _MAX_REMOVAL_FRACTION:
        schedule.record_skip(
            "background", "removal_budget_exceeded",
            removed=removed, maskPixels=total,
            fraction=round(removed / max(1, total), 4),
        )
        return mask

    # Per-component guard: never erase an entire accepted component. A thin
    # part that is genuinely background-coloured but edge-separated was kept
    # by the barrier; anything else losing most of its pixels is suspicious.
    labels, rows = _components(mask, 1)
    kept = 0
    for row in rows:
        component = labels == row["label"]
        removed_here = int((component & removal).sum())
        if removed_here == 0:
            continue
        share = removed_here / max(1, row["area"])
        if share > 0.85 and row["area"] > 32:
            removal &= ~component
            continue
        kept += removed_here
    if kept == 0:
        schedule.record_skip("background", "component_guard_rejected_all")
        return mask

    updated = mask & ~removal
    schedule.record(
        "background", "removed", pixels=int(removal.sum()),
        fraction=round(removal.sum() / max(1, total), 4),
        threshold=round(bg_threshold, 2),
        edgeThreshold=round(edge_threshold, 1),
        reason="reachable_background_negative_space",
    )
    return updated


def review_cleanup_deletions_v5(before, proposed, candidates, lab, gradient,
                                schedule, stage, authoritative=None):
    """Adjudicate connected cleanup edits before they become permanent negatives.

    SAM agreement alone is not ownership: absorbed backdrop can be unanimous.
    Reject a cut only with independent local surface/contour continuity too.
    Verified neighbour interiors and negative anchors always take precedence.
    No pixels outside the incoming mask may be added by this review.
    """
    removed = before & ~proposed
    if not removed.any() or len(candidates) < 2:
        return proposed
    authority = (np.zeros_like(before) if authoritative is None
                 else np.asarray(authoritative, dtype=bool))
    votes = np.sum(candidates, axis=0)
    stable = votes >= max(2, math.ceil(len(candidates) * .8))
    labels, regions = _components(removed & ~authority, 8)
    updated = proposed.copy()
    details = []
    for row in regions:
        region = labels == row['label']
        if float(stable[region].mean()) < .95:
            continue
        # Inspect both sides locally, rather than using the global subject
        # palette, which often omits a dark side, reflection or small part.
        band = _dilate(region, 3)
        kept = band & proposed & ~authority
        exterior = band & ~before
        cut = (proposed & ~_erode(proposed, 1)) & band
        rim = (before & ~_erode(before, 1)) & band
        if kept.sum() < 8 or cut.sum() < 4:
            continue
        centre = np.median(lab[region], axis=0)
        kept_distance = float(np.linalg.norm(centre - np.median(lab[kept], axis=0)))
        outside_distance = (float(np.linalg.norm(centre - np.median(lab[exterior], axis=0)))
                            if exterior.sum() >= 4 else None)
        cut_strength = float(np.median(gradient[cut]))
        rim_strength = float(np.median(gradient[rim])) if rim.any() else 0.
        surface_continuity = (kept_distance <= 10. and
                              (outside_distance is None or kept_distance + 3. < outside_distance))
        stronger_original = (rim_strength >= max(24., cut_strength * 1.5)
                             and cut_strength < 32.)
        # A flood entering a narrow opening can carve a long interior gash.
        # It needs local same-surface evidence as well as candidate agreement.
        interior_gash = (cut.sum() >= max(12, rim.sum() * 3)
                         and cut_strength < 24. and kept_distance <= 5.)
        if surface_continuity and (stronger_original or interior_gash):
            updated |= region
            details.append(dict(area=row['area'],
                                keptDistance=round(kept_distance, 2),
                                outsideDistance=(round(outside_distance, 2)
                                                 if outside_distance is not None else None),
                                cutGradient=round(cut_strength, 2),
                                rimGradient=round(rim_strength, 2)))
    restored = int((updated & ~proposed).sum())
    schedule.record('cleanup_deletion_review', 'local_surface_and_contour',
                    sourceStage=stage, proposedPixels=int(removed.sum()),
                    rejectedPixels=restored,
                    acceptedPixels=int((before & ~updated).sum()), details=details)
    return updated


def _max_side(shape):
    return float(max(shape[0], shape[1]))


def exclude_prior_pixels_v5(mask, models, schedule, policy):
    """Review explicit neighbour regions using a separate observed palette.

    An overlapping rectangle is not a cut mask. The target palette excludes
    declared neighbours, then each neighbour must have distinguishable colour
    evidence. Same-coloured overlap is left unchanged and reported as uncertain.
    """
    lab = models.get("lab")
    forbidden = schedule.forbidden & ~models.get('reviewedNeighbour', np.zeros_like(mask))
    if lab is None or not forbidden.any():
        schedule.record_skip("exclusion", "no_explicit_neighbour_evidence")
        return mask
    removed = np.zeros_like(mask)
    uncertain = 0
    labels, rows = _components(forbidden, 1)
    for row in rows:
        region = labels == row["label"]
        overlap = mask & region
        if not overlap.any():
            continue
        seeds = _erode(region, 2)
        hint = models.get("hint")
        if hint is not None:
            subject_box = np.zeros_like(mask)
            x1, y1, x2, y2 = [int(round(v)) for v in hint]
            subject_box[y1:y2, x1:x2] = True
            exterior_seeds = seeds & ~subject_box
            if int(exterior_seeds.sum()) >= 16:
                seeds = exterior_seeds
            else:
                seeds &= mask
        else:
            seeds &= mask
        if int(seeds.sum()) < 8:
            uncertain += int(overlap.sum())
            continue
        palette = _kmeans_palette(lab, seeds, 3)
        distance = _palette_distance(lab, palette)
        fg = models["fgDistFull"]
        evidence_region = region & (distance < 12.0) & (distance + 10.0 < fg)
        # Only components reached from a reliable neighbour interior may go.
        evidence = _reachable_from(evidence_region, seeds & evidence_region) & overlap
        labels_evidence, evidence_rows = _components(evidence, 8)
        evidence &= np.isin(labels_evidence, [r["label"] for r in evidence_rows])
        # Interior lettering/highlights belong to an enclosing neighbour too.
        # This repairs enclosed negative islands only, never its convex hull
        # or the full rectangle (which could swallow a nearby plate or leg).
        enclosure = cv2.morphologyEx(evidence.astype(np.uint8), cv2.MORPH_CLOSE, _disc(2)) > 0
        enclosure &= region
        pad = np.pad(enclosure.astype(np.uint8), 1)
        _, free = cv2.connectedComponents((pad == 0).astype(np.uint8), connectivity=4)
        enclosed = ((free != 0) & (free != free[0, 0]))[1:-1, 1:-1]
        hole_labels, hole_rows = _components(enclosed & overlap, 1)
        for hole in hole_rows:
            if hole["area"] <= max(4, int(evidence.sum() * 0.40)):
                evidence |= hole_labels == hole["label"]
        removed |= evidence
        uncertain += int((overlap & ~evidence).sum())
    share = float(removed.sum()) / max(1, int(mask.sum()))
    protected = models.get('contactProtected')
    protected_pixels = 0
    if protected is not None:
        protected_pixels = int((removed & protected).sum())
        removed &= ~protected
        share = float(removed.sum()) / max(1,int(mask.sum()))
    if share > _EXCLUSION_TRIM_LIMIT:
        schedule.record_skip("exclusion", "prior_covers_most_of_subject", share=round(share, 4))
        return mask
    # Remove newly detached lettering islands surrounded by verified neighbour
    # removal. Original detached subject parts are not eligible.
    remaining = mask & ~removed
    _, original_labels = cv2.connectedComponents(mask.astype(np.uint8))
    owners, fragments = _components(remaining, 1)
    for fragment in fragments:
        component = owners == fragment["label"]
        if fragment["area"] > max(16, int(removed.sum() * 0.15)):
            continue
        if not np.all(forbidden[component]):
            continue
        parent = original_labels[component][0]
        if not (removed & (original_labels == parent)).any():
            continue
        ring = _dilate(component, 2) & ~component
        if ring.any() and float(np.mean(removed[ring])) >= 0.50:
            removed |= component
    schedule.record("exclusion", "removed" if removed.any() else "unchanged",
                    pixels=int(removed.sum()), unresolvedPixels=uncertain,
                    protectedContactPixels=protected_pixels,
                    reason="explicit_neighbour_colour_and_connected_seed")
    return mask & ~removed


def prune_border_intrusions_v5(mask, hint, schedule, policy):
    """Remove thin objects that enter the layer across the requested box edge.

    A chair photographed at a table has a table leg running behind it, and a
    cabinet on a floor can absorb the skirting it stands against. Those parts
    share the subject's material, so no colour rule separates them and the
    proposal keeps them. Geometry can: the intruder is a thin appendage that
    crosses the edge of the box the user asked for and continues *outside* it,
    which means it belongs to something this layer does not own. A subject that
    is legitimately cut by its own box is thick where it crosses, so thickness
    at the border is the discriminator.
    """
    # 确保 mask 是布尔类型，避免 bitwise_and 错误
    if mask.dtype != bool:
        if mask.dtype in [np.float32, np.float64]:
            mask = mask > 0.5
        else:
            mask = mask.astype(bool)

    shape = mask.shape
    total = int(mask.sum())
    if total == 0:
        return mask

    # Separate the subject's bulk from every slender piece attached to it.
    # Erosion was the wrong tool: ``mask - erode(mask)`` is the subject's own
    # silhouette ring, one component that runs all the way around the body and
    # therefore merges the intruder into an oversized blob the size cap rejects.
    # An opening removes slender structures outright, so the intruder comes out
    # as its own component while the body stays whole.
    core_radius = max(2, int(round(_max_side(shape) * 0.012)))
    core = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, _disc(core_radius)
    ) > 0
    if not core.any():
        schedule.record_skip("intrusion", "no_core")
        return mask
    slender = mask & ~_dilate(core, 2)
    if not slender.any():
        schedule.record_skip("intrusion", "no_thin_appendages")
        return mask

    hint_region = np.zeros(shape[:2], dtype=bool)
    hx1, hy1, hx2, hy2 = [int(round(v)) for v in hint]
    hint_region[max(0, hy1):hy2, max(0, hx1):hx2] = True
    border = _border_band([hx1, hy1, hx2, hy2], shape, 2)

    labels, rows = _components(slender, max(24, int(total * 0.0004)))
    removal = np.zeros(shape[:2], dtype=bool)
    removed = []
    for row in rows:
        component = labels == row["label"]
        area = int(row["area"])
        if area > total * 0.12:
            continue
        # The appendage must cross the edge of the box the user asked for...
        if not (component & border).any():
            continue
        # ...and continue past it. Material that merely ends at the box edge is
        # the subject being cropped by its own box, which is legitimate.
        beyond = int((component & ~hint_region).sum())
        if beyond < max(8, area * 0.2):
            continue
        # ...and be narrow along its whole length, so a thick part that happens
        # to poke out of the box is not mistaken for an intruder.
        depth = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        if float(depth.max(initial=0.0)) > core_radius:
            continue
        if schedule.vetoed(component):
            continue
        removal |= component
        removed.append({"area": area, "beyondBox": beyond})

    if not removed:
        schedule.record_skip("intrusion", "no_border_crossing")
        return mask
    updated = mask & ~removal
    schedule.record(
        "intrusion", "removed", pixels=int(removal.sum()),
        components=len(removed), details=removed[:8],
        reason="thin_object_crosses_requested_box",
    )
    return updated


# ---------------------------------------------------------------------------
# Stage: subject completion
# ---------------------------------------------------------------------------


def complete_from_hint_v5(
    lab, mask, models, gradient, hint_bbox, window, schedule, policy, edge
):
    """Fill subject regions inside a trusted hint box.

    Gemini's tight-subject box says "the subject lives here". Inside that box,
    anything that is not background-coloured and not reachable as background
    is a subject region candidate; it still needs edge support and must not
    fall inside an exclusion prior.
    """
    intent_fill = policy.get("intentFill") or {}
    if not intent_fill.get("enabled"):
        schedule.record_skip("hint_completion", intent_fill.get("reason") or "disabled")
        return mask

    shape = mask.shape
    hint = [int(v) for v in hint_bbox]
    hint[0] = int(_clamp(hint[0], 0, shape[1]))
    hint[2] = int(_clamp(hint[2], 0, shape[1]))
    hint[1] = int(_clamp(hint[1], 0, shape[0]))
    hint[3] = int(_clamp(hint[3], 0, shape[0]))
    if _bbox_area(hint) <= 0:
        schedule.record_skip("hint_completion", "empty_hint")
        return mask

    window_pixels = _window_mask(window, shape)
    region = np.zeros(shape[:2], dtype=bool)
    region[hint[1]:hint[3], hint[0]:hint[2]] = True
    region &= window_pixels

    separation = float(models.get("separation") or 0.0)
    if separation < _MIN_PALETTE_SEPARATION:
        schedule.record_skip("hint_completion", "low_palette_separation")
        return mask

    bg_like = (
        (models["bgDistFull"] < float(_clamp(0.6 * separation, 6.0, 34.0))) &
        window_pixels
    )
    fg_like = (models["fgDistFull"] < max(4.0, 0.45 * separation)) & window_pixels
    crop_gradient = gradient[window[1]:window[3], window[0]:window[2]]
    edge_threshold = max(14.0, float(np.percentile(crop_gradient, 70)))
    barrier = gradient >= edge_threshold

    # Background reachability, independent of mask membership, so background
    # the SAM candidate missed is still recognised as background.
    passable = window_pixels & bg_like & ~fg_like & ~barrier
    reach = max(2, int(round(_max_side(shape) * _BACKGROUND_REACH_RATIO)))
    seeds = passable & _border_band(window, shape, reach)
    background = _reachable_from(passable, seeds)

    unassigned = region & ~mask & ~schedule.protected & ~schedule.forbidden
    candidate = unassigned & ~background
    # Inside a trusted hint, "not backdrop" is the evidence that matters, not
    # "looks like the subject". A missed plate rim, a white napkin or a steel
    # fork is nowhere near the meat's palette, and demanding foreground
    # similarity is precisely what made a missing plate impossible to recover -
    # the subject's own palette can only describe what the proposal already
    # found. What disqualifies a region is *background* evidence: closeness to
    # the backdrop palette sampled at the window border.
    # The margin is deliberately wide: "a bit unlike the backdrop" is how a
    # neighbouring object's own edge gets admitted once the hint box is trimmed
    # tightly around the subject. Only material the backdrop model cannot account
    # for at all qualifies, and the attachment test below still has to pass.
    far_from_backdrop = models["bgDistFull"] > max(
        1.40 * separation, 0.70 * separation + 10.0
    )
    candidate &= (fg_like | far_from_backdrop)
    if not candidate.any():
        schedule.record_skip("hint_completion", "no_candidate_regions")
        return mask

    labels, rows = _components(candidate, max(_MIN_COMPONENT_PIXELS, int(region.sum() * 0.002)))
    if not rows:
        schedule.record_skip("hint_completion", "no_components_above_minimum")
        return mask

    addition = np.zeros(shape[:2], dtype=bool)
    accepted = []
    region_area = float(region.sum())
    component_gap = max(2, int(round(_max_side(shape) * _COMPLETION_GAP_RATIO)))
    for row in rows:
        component = labels == row["label"]
        area = row["area"]
        if area > region_area * 0.5:
            continue
        support = _component_edge_support(component, gradient, edge_threshold)
        # A region that does not look like the subject still has to be attached
        # to it, or surrounded by it, to be admitted. "Not backdrop" alone is
        # not enough: a same-coloured patch floating in a corner of the box is
        # the backdrop's, not the subject's.
        fg_share = float(np.mean(fg_like[component]))
        ring = _dilate(component, component_gap) & ~component
        attached = float(np.mean(mask[ring])) if ring.any() else 0.0
        if fg_share < 0.35 and attached < 0.60:
            continue
        # An enclosed region (not touching the hint border) with foreground
        # pull is subject matter; a region on the border needs real edge
        # support to avoid absorbing a smooth backdrop.
        touches_border = bool(
            (component & _border_band(hint, shape, 2)).any()
        )
        # An enclosed component sharing the frame's palette can be a missing
        # part or a hole. A hole is separated from the subject by
        # background-coloured pixels; a part is attached or sits directly on
        # it. Colour alone cannot tell them apart.
        if not touches_border:
            bridge = _dilate(component, component_gap) & ~component & ~mask
            if bridge.any() and float(np.mean(bg_like[bridge])) >= _BRIDGE_BACKGROUND_SHARE:
                schedule.record(
                    "hint_completion", "skipped_component",
                    reason="background_separated_hole", area=int(area),
                )
                continue
        if touches_border and support < 0.30:
            continue
        if not touches_border and support < 0.12:
            continue
        if schedule.vetoed(component):
            continue
        addition |= component
        accepted.append({
            "area": int(area),
            "edgeSupport": round(support, 3),
            "enclosed": not touches_border,
        })

    if not accepted:
        schedule.record_skip("hint_completion", "no_component_passed_evidence")
        return mask

    base_total = int(mask.sum())
    add_total = int(addition.sum())
    if add_total > base_total * _MAX_ADDITION_FRACTION:
        schedule.record_skip(
            "hint_completion", "addition_budget_exceeded",
            added=add_total, maskPixels=base_total,
        )
        return mask

    updated = mask | addition
    schedule.record(
        "hint_completion", "added", pixels=add_total,
        components=len(accepted), details=accepted[:8],
        reason="hinted_subject_region",
    )
    return updated


def repair_holes_v5(mask, models, schedule, edge, material, perforated=False):
    """Fill only small enclosed omissions supported by foreground colour.

    Size alone never establishes ownership: a four-pixel gap can be a real
    opening. Ambiguous/low-contrast gaps are kept for review, including mesh.
    """
    if not mask.any() or material.get("mode") == "translucent":
        schedule.record_skip("hole_repair", "empty_or_translucent")
        return mask
    padded = np.pad(mask.astype(np.uint8), 1)
    _, labels = cv2.connectedComponents((padded == 0).astype(np.uint8), connectivity=4)
    holes = ((labels != 0) & (labels != labels[0, 0]))[1:-1, 1:-1]
    owners, rows = _components(holes, 1)
    fill = np.zeros_like(mask)
    total = int(mask.sum())
    details = []
    for row in rows:
        region = owners == row["label"]
        area = row["area"]
        if area > max(16, total * _HOLE_MAX_SHARE) or schedule.vetoed(region):
            continue
        fg = models["fgDistFull"][region]
        bg = models["bgDistFull"][region]
        # Distances are smaller for better matches; require a positive margin
        # in favour of foreground, never the reverse.
        supported = (fg + 4.0 < bg) & (fg < 18.0)
        if float(np.mean(supported)) < 0.90:
            continue
        fill |= region
        details.append({"area": int(area), "reason": "foreground_colour_evidence"})
    if int(fill.sum()) > total * _HOLE_FILL_BUDGET:
        schedule.record_skip("hole_repair", "fill_budget_exceeded", pixels=int(fill.sum()))
        return mask
    schedule.record("hole_repair", "filled" if fill.any() else "unchanged",
                    pixels=int(fill.sum()), perforated=bool(perforated), details=details[:8])
    return mask | fill


def complete_by_color_v5(lab, mask, models, gradient, window, schedule, policy):
    """Add nearby regions that match the subject's own colour model.

    This recovers detached same-material parts (a garnish that shares the
    subject palette, a detached base) without any category logic.
    """
    shape = mask.shape
    window_pixels = _window_mask(window, shape)
    separation = float(models.get("separation") or 0.0)
    if separation < 12.0:
        schedule.record_skip("color_completion", "low_palette_separation")
        return mask

    threshold = float(_clamp(0.70 * separation, 9.0, 30.0))
    similar = models["fgDistFull"] < threshold
    # A detached part is similar to the subject but not necessarily the same
    # colour: it must still be clearly closer to the subject palette than to the
    # background palette, which rules out a backdrop that merely leans toward
    # the subject's hue.
    margin = similar & (models["bgDistFull"] > models["fgDistFull"] + _COMPLETION_COLOR_MARGIN)
    diag = math.hypot(shape[1], shape[0])
    gap = max(3, int(round(diag * _COMPLETION_GAP_RATIO)))
    adjacent = _dilate(mask, gap) & ~mask
    candidate = (
        window_pixels & margin & adjacent &
        ~schedule.protected & ~schedule.forbidden
    )
    if not candidate.any():
        schedule.record_skip("color_completion", "no_adjacent_similar_region")
        return mask

    crop_gradient = gradient[window[1]:window[3], window[0]:window[2]]
    edge_threshold = max(14.0, float(np.percentile(crop_gradient, 70)))
    labels, rows = _components(candidate, _MIN_COMPONENT_PIXELS)
    if not rows:
        schedule.record_skip("color_completion", "no_components_above_minimum")
        return mask

    base_total = int(mask.sum())
    addition = np.zeros(shape[:2], dtype=bool)
    accepted = []
    for row in rows:
        component = labels == row["label"]
        if row["area"] > base_total * 0.5:
            continue
        support = _component_edge_support(component, gradient, edge_threshold)
        if support < _COMPLETION_EDGE_SUPPORT:
            continue
        if schedule.vetoed(component):
            continue
        addition |= component
        accepted.append({"area": int(row["area"]), "edgeSupport": round(support, 3)})

    if not accepted:
        schedule.record_skip("color_completion", "no_component_passed_edge_support")
        return mask
    if int(addition.sum()) > base_total * _MAX_ADDITION_FRACTION:
        schedule.record_skip("color_completion", "addition_budget_exceeded",
                             added=int(addition.sum()), maskPixels=base_total)
        return mask

    updated = mask | addition
    schedule.record("color_completion", "added", pixels=int(addition.sum()),
                    components=len(accepted), details=accepted[:8],
                    reason="adjacent_subject_color")
    return updated


# ---------------------------------------------------------------------------
# Stage: channel levels for perforated subjects
# ---------------------------------------------------------------------------


def _perforation_stats(mask):
    """Count the enclosed openings the proposal left inside its own support.

    A proposal drawn for a leafy subject closes the gaps between leaves: at the
    mask's working resolution those gaps are smaller than the object, so the
    model treats them as interior. Counting them is the evidence that this
    happened, and it is topology, not a category name.
    """
    padded = np.pad(mask.astype(np.uint8), 1)
    _, free = cv2.connectedComponents((padded == 0).astype(np.uint8), connectivity=4)
    outside = int(free[0, 0])
    # As in repair_holes_v5: label 0 is the subject, not free space.
    holes = (free != 0) & (free != outside)
    holes = holes[1:-1, 1:-1]
    if not holes.any():
        return 0, 0
    _, rows = _components(holes, 16)
    return len(rows), int(sum(row["area"] for row in rows))


def channel_coverage_v5(image, mask, window, schedule):
    """Channel levels for perforated, fine-grained subjects (foliage, mesh, lace).

    A silhouette cannot describe a leafy subject: the gaps between leaves are
    interior as far as the mask is concerned, so the cutout comes back as a
    solid slab with the backdrop painted in. Colour levels can describe it,
    because the subject is whatever stands out from a locally uniform backdrop
    and the gaps are whatever is locally flat. This is the operation Photoshop's
    channel-based cutouts use, and the one V1 applied to plants; here it is
    gated on measured topology (enclosed openings or an open fine silhouette)
    and on exterior colour agreement, so surface texture alone cannot trigger it.

    The result is a multiplicative coverage map: it can only take alpha away,
    never invent a pixel the proposal did not contain.
    """
    x1, y1, x2, y2 = window
    crop = image[y1:y2, x1:x2]
    base = mask[y1:y2, x1:x2].astype(bool)
    if crop.size == 0 or not base.any():
        return None

    mask_pixels = max(1, int(mask.sum()))
    openings, opening_pixels = _perforation_stats(mask)
    # Open comb-like gaps can connect to the outside, so they do not register
    # as enclosed holes. Admit them through the same measured channel path.
    eroded = cv2.erode(base.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary_ratio = float((base & ~eroded).sum() / max(1, int(base.sum())))
    fill_ratio = float(base.sum() / max(1, crop.shape[0] * crop.shape[1]))
    enclosed_structure = (
        openings >= _CHANNEL_MIN_OPENINGS and
        opening_pixels >= max(128, int(mask_pixels * _CHANNEL_MIN_OPENING_SHARE))
    )
    open_structure = not enclosed_structure and fill_ratio <= 0.68 and boundary_ratio >= 0.08
    if not (enclosed_structure or open_structure):
        schedule.record_skip("channel", "no_perforated_structure", openings=openings,
                             fillRatio=round(fill_ratio, 4), boundaryRatio=round(boundary_ratio, 4))
        return None

    # Local median background at two scales, so illumination gradients and both
    # light and dark backdrops are followed rather than assumed.
    radius = max(5, min(15, int(round(min(crop.shape[:2]) * 0.04))))
    contrasts = []
    for step in (radius, max(radius + 1, int(round(radius * 1.5)))):
        background = cv2.medianBlur(crop, step * 2 + 1)
        contrasts.append(
            np.max(background.astype(np.float32) - crop.astype(np.float32), axis=2)
        )
    contrast = np.maximum(contrasts[0], contrasts[1])

    ridge = (contrast > _CHANNEL_CONTRAST) & base
    if int(ridge.sum()) == 0:
        schedule.record_skip("channel", "no_channel_contrast")
        return None
    depth = cv2.distanceTransform(ridge.astype(np.uint8), cv2.DIST_L2, 5)
    thin_ratio = float(((depth <= 2) & ridge).sum() / max(1, ridge.sum()))
    ridge_ratio = float(ridge.sum() / max(1, base.sum()))
    # Open branch/leaf silhouettes are thicker than enclosed mesh holes, so
    # their thin-ridge ratio is a little lower.  Keep the stricter threshold
    # for ordinary masks and relax it only after the independent open-structure
    # gate above has passed.
    thin_floor = 0.40 if open_structure else 0.50
    if thin_ratio < thin_floor or not 0.2 < ridge_ratio < 0.85:
        schedule.record_skip(
            "channel", "no_fine_channel_separation",
            thinRatio=round(thin_ratio, 4), ridgeRatio=round(ridge_ratio, 4),
        )
        return None

    # Fabric weave can satisfy the fine-ridge gate too. A true branching
    # structure must retain most of its contrast after suppressing pixel-scale
    # texture. This evidence controls interior edits, independently of names.
    coarse = cv2.GaussianBlur(crop, (0, 0), 1.5)
    coarse_contrasts = [
        np.max(cv2.medianBlur(coarse, step*2+1).astype(np.float32)-coarse, axis=2)
        for step in (radius, max(radius+1, int(round(radius*1.5))))
    ]
    coarse_agreement = float(((np.maximum(*coarse_contrasts)>_CHANNEL_CONTRAST)&ridge).sum()
                             /max(1,int(ridge.sum())))

    noise = contrast[~base]
    floor = max(3.0, float(np.median(noise))) if noise.size else 3.0
    level = max(floor + 12.0, float(np.percentile(contrast[ridge], 30)))
    coverage = np.clip((contrast - floor) / (level - floor), 0.0, 1.0)

    # Dense local foreground has no reliable background estimate; keep it,
    # rather than letting the median turn a dense dark canopy into holes.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    local_median = cv2.medianBlur(gray, radius * 2 + 1)
    background_values = local_median[~base]
    if background_values.size:
        dark_floor = float(np.percentile(background_values, 5))
        coverage[(local_median < dark_floor) & base] = 1.0

    # Texture on a solid surface also produces thin contrast ridges. Require the
    # pixels this pass would clear to share a colour actually observed outside
    # the subject; otherwise it is surface detail, not backdrop.
    smooth = np.max(
        np.abs(cv2.medianBlur(crop, 5).astype(np.int16) - crop.astype(np.int16)), axis=2
    ) <= 5
    exterior = cv2.morphologyEx(
        ((~base) & smooth).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ) > 0
    if int(exterior.sum()) < 64:
        schedule.record_skip("channel", "insufficient_exterior_evidence")
        return None
    lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab_crop[exterior]
    quantized = (samples // 12).astype(np.int32)
    _, ids, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    nearest = np.full(coverage.shape, np.inf, dtype=np.float32)
    for index in np.argsort(-counts):
        if counts[index] < max(16, len(samples) * 0.001):
            continue
        color = np.median(samples[ids == index], axis=0)
        nearest = np.minimum(nearest, np.linalg.norm(lab_crop - color, axis=2))
    clear = base & (coverage <= 0.1)
    agreement = (
        float(np.mean(nearest[clear] <= _CHANNEL_COLOR_MATCH)) if clear.any() else 0.0
    )
    # Open gaps need not dominate all low-contrast pixels: pale foreground
    # surfaces may share the crop. Accept only locally connected exterior
    # evidence in that case, leaving unsupported surfaces fully opaque.
    open_evidence = open_structure and coarse_agreement >= .60
    if agreement < (0.5 if open_evidence else 0.8):
        schedule.record_skip(
            "channel", "surface_texture_without_background_evidence",
            agreement=round(agreement, 4),
        )
        return None
    # Colours with no observed background counterpart keep their opacity even
    # when the regional gate passed.
    coverage[base & (nearest > _CHANNEL_COLOR_MATCH)] = 1.0
    if open_evidence and not enclosed_structure:
        background_path = (nearest <= _CHANNEL_COLOR_MATCH) & (coverage < .5)
        reachable = _reachable_from(background_path, background_path & ~base)
        coverage[base & ~reachable] = 1.0
        # A broad pale surface can be connected to the exterior and share its
        # colour (e.g. upholstery). Open-gap evidence authorizes narrow spaces,
        # not wholesale surface removal. Reject this route if it proposes a
        # broad interior patch; enclosed-hole handling remains unchanged.
        proposed_clear = base & (coverage < .5)
        gap_depth = cv2.distanceTransform(proposed_clear.astype(np.uint8), cv2.DIST_L2, 5)
        _, _, subject_w, subject_h = cv2.boundingRect(base.astype(np.uint8))
        max_depth = max(3.0, .06 * min(subject_w, subject_h))
        if float(gap_depth.max()) > max_depth:
            schedule.record_skip('channel', 'broad_surface_in_open_structure',
                                 gapDepth=round(float(gap_depth.max()), 2),
                                 allowedDepth=round(max_depth, 2))
            return None

    retained = int((coverage > 0.5).sum())
    if retained < base.sum() * 0.2:
        schedule.record_skip("channel", "insufficient_structure_preserved",
                             retained=retained)
        return None

    full = np.zeros(mask.shape[:2], dtype=np.float32)
    full[y1:y2, x1:x2] = coverage
    schedule.record(
        "channel", "applied", openings=openings, openingPixels=opening_pixels,
        structureMode="enclosed" if enclosed_structure else "open_boundary",
        fillRatio=round(fill_ratio, 4), boundaryRatio=round(boundary_ratio, 4),
        thinRatio=round(thin_ratio, 4), ridgeRatio=round(ridge_ratio, 4),
        backgroundAgreement=round(agreement, 4),
        coarseStructureAgreement=round(coarse_agreement, 4),
        interiorEvidence=coarse_agreement >= .60,
        clearedPixels=int(((coverage <= 0.1) & base).sum()),
        reason="perforated_channel_levels" if enclosed_structure else "open_structure_channel_levels",
    )
    return full


# ---------------------------------------------------------------------------
# Stage: boundary alpha
# ---------------------------------------------------------------------------


def _supersampled_alpha(mask):
    """Fallback antialias: 4x contour rasterization without colour evidence."""
    height, width = mask.shape
    small = (mask.astype(np.uint8) * 255)
    upscaled = cv2.resize(
        small, (width * 4, height * 4), interpolation=cv2.INTER_LINEAR
    )
    downscaled = cv2.resize(
        upscaled, (width, height), interpolation=cv2.INTER_AREA
    )
    return downscaled.astype(np.float32) / 255.0


def _smooth_mask_boundary(mask):
    """Subpixel antialias without opening away thin branches or closing holes."""
    if mask is None:
        return None
    # A narrow ramp preserves one-pixel lines, corners and disconnected parts.
    # Thick cores are restored by alpha estimation; no binary morphology is
    # allowed to decide which detail belongs to the object.
    soft = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), _SILHOUETTE_BLUR_RADIUS)
    return np.clip(soft, 0.0, 1.0)


def _color_coverage_alpha(models, base, radius, shape):
    """Alpha from colour distance to the backdrop, as V1's soft matte did it.

    The silhouette says *where* the boundary is; it cannot say *how opaque* a
    pixel is. Colour can: a feather barb, a gauze veil and a glass pane are all
    pixels whose colour lies between the subject's and the backdrop's, and how
    far along that line they sit is the coverage. ``low`` is the distance at
    which a pixel is indistinguishable from the backdrop (measured on the ring
    outside the silhouette); ``high`` is the distance the subject's own core
    reaches. Between them the ramp is smoothstepped, which is exactly the
    transfer curve V1 used for soft edges.
    """
    x1, y1, x2, y2 = models["window"]
    crop_base = base[y1:y2, x1:x2]
    if not crop_base.any():
        return None
    bg_dist = models["bgDist"]
    inner = _erode(crop_base, max(2, radius))
    if int(inner.sum()) < 32:
        inner = crop_base
    outer = _dilate(crop_base, max(3, radius + 2)) & ~crop_base
    if int(outer.sum()) < 32:
        outer = ~_dilate(crop_base, max(4, radius + 3))
    if int(inner.sum()) == 0 or int(outer.sum()) == 0:
        return None
    low = max(6.0, float(np.percentile(bg_dist[outer], 98)))
    high = max(low + 18.0, float(np.percentile(bg_dist[inner], 60)))
    if high - low < 10.0:
        return None
    ramp = np.clip((bg_dist - low) / (high - low), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    full = np.zeros(shape[:2], dtype=np.float32)
    full[y1:y2, x1:x2] = ramp
    return full


def estimate_band_alpha_v5(lab, mask, models, edge, material, window, policy):
    """Estimate alpha by local foreground/background colour mixture.

    For every pixel in the measured band, alpha is the projection of its colour
    onto the segment between the local foreground and background colours. This
    is the same model Photoshop's refine-edge uses for a band of radius r; the
    difference is that our r comes from measured transition width.
    """
    shape = mask.shape
    # Measure on a de-staircased outline, otherwise every edge inherits the
    # proposal's pixel steps no matter what the colour evidence says.
    soft = _smooth_mask_boundary(mask)
    base = soft >= 0.5
    radius = int(edge.get("radius") or 2)
    height, width = shape[:2]

    band = _dilate(base, radius) & ~_erode(base, radius)
    core = _erode(base, radius)
    if int(core.sum()) == 0:
        core = base
    outside_ring = _dilate(base, radius * 2) & ~_dilate(base, radius)

    foreground, fg_support = _weighted_mean(lab, core, radius * 2 + 1)
    background, bg_support = _weighted_mean(lab, outside_ring, radius * 2 + 1)

    vector = foreground - background
    denom = np.sum(vector * vector, axis=2)
    safe = np.where(denom > 1e-3, denom, 1.0)
    mix = np.clip(
        np.sum((lab - background) * vector, axis=2) / safe, 0.0, 1.0
    )
    confident = (denom > 64.0) & (fg_support > 0) & (bg_support > 0)

    translucent = material.get("mode") == "translucent"

    # Anti-aliased silhouette: a sub-pixel ramp whose width follows the measured
    # transition. A hard edge stays crisp, a soft edge stays soft, and neither
    # shows the proposal's pixel steps. The ramp comes from the smoothed
    # coverage map itself; thresholding it and blurring the binary again would
    # rebuild the staircase one pixel away from where it was removed.
    alpha = np.clip(soft.astype(np.float32), 0.0, 1.0)
    sigma = float(_clamp(0.6 * max(1, radius), 0.7, 2.2))
    if edge.get("class") == "soft" and sigma > _SILHOUETTE_BLUR_RADIUS + 0.2:
        # A measured soft edge is wider than the base smoothing; extend it.
        alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigma), 0.0, 1.0)

    # How far partial alpha may reach. A hard edge keeps the one-pixel
    # antialias; a measured ramp opens the band to its own width, which is what
    # lets feather, gauze and hair recover instead of being cut at the
    # silhouette. The ramp itself comes from colour, not from the mask, so a
    # wing tip that the proposal rounded off is still described by the pixels.
    width = float(edge.get("transitionWidth") or 0.0)
    reach = int(_clamp(round(max(2.0, width)) + 1, 2, 12))
    coverage = _color_coverage_alpha(models, base, radius, shape)
    if coverage is not None:
        ring = _dilate(base, reach) & ~_erode(base, reach)
        if edge.get("class") == "hard":
            ring &= base
        alpha[ring] = np.maximum(alpha[ring], coverage[ring])

    if translucent:
        # Translucent interiors keep a mixture-based alpha instead of being
        # forced opaque. Floor at 0.15 so glass is visible but not erased.
        fg_norm = np.clip(models["fgDistFull"], 0, None)
        bg_norm = np.clip(models["bgDistFull"], 0, None)
        scale = max(6.0, float(np.median(fg_norm[core])))
        fg_sim = np.exp(-fg_norm / scale)
        bg_sim = np.exp(-bg_norm / scale)
        ratio = fg_sim / np.maximum(fg_sim + bg_sim, 1e-4)
        alpha[base] = np.clip(ratio[base], 0.15, 1.0)
        # Preserve the outermost silhouette so the layer still has an edge.
        alpha[core] = np.maximum(alpha[core], 0.35)
    else:
        # Inside the band, refine the ramp with the measured colour mixture.
        # Blending with the silhouette keeps a noisy fg/bg pair from carving
        # into the subject.
        inside_band = band & base
        blended = 0.5 * mix + 0.5 * alpha
        alpha[inside_band] = np.where(
            confident[inside_band], blended[inside_band], alpha[inside_band]
        )
        # Fill in the sub-pixel ring just outside the outline where the image
        # agrees the pixel is a subject/backdrop mixture.
        halo_ring = _dilate(base, 1) & ~base
        alpha[halo_ring] = np.where(
            confident[halo_ring] & (mix[halo_ring] >= 0.35),
            mix[halo_ring], alpha[halo_ring],
        )

        # The interior is opaque; only the transition carries partial alpha.
        distance_inside = cv2.distanceTransform(
            base.astype(np.uint8), cv2.DIST_L2, 3
        )
        alpha[base & (distance_inside >= radius + 1.0)] = 1.0

    # Containment: nothing beyond the antialias/feather band, which is what
    # stops the background halo. The band is as wide as the measured ramp, so a
    # feather is not clipped back to a one-pixel ring.
    support = _dilate(base, max(2, reach))
    alpha[~support] = 0.0

    alpha[core] = np.maximum(alpha[core], 1.0 if not translucent else 0.35)

    if int((alpha > _ALPHA_MIN).sum()) == 0:
        _log(policy, "alpha", "fallback", reason="fallback_supersample")
        return _supersampled_alpha(mask)
    return alpha


def decontaminate_edge_colors_v5(image, alpha, background, limit=0.92, floor=0.08):
    """Remove background-coloured fringe from partially transparent pixels."""
    out = image.astype(np.float32)
    a = alpha[:, :, None]
    mixed = (a > floor) & (a < limit)
    safe = np.where(mixed, np.maximum(a, floor), 1.0)
    corrected = (out - (1.0 - a) * background) / safe
    out = np.where(mixed[:, :, 0][:, :, None], corrected, out)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_quality_v5(mask, alpha, hint, candidate, schedule, original, clipped_pixels=0):
    """Acceptance is independent of object category and transported unchanged."""
    issues = list(candidate.get("issues") or [])
    warnings = []
    total = int(mask.sum())
    x1, y1, x2, y2 = [int(round(v)) for v in hint]
    inside = int(mask[y1:y2, x1:x2].sum()) / max(1, total)
    if total < 4 or not np.isfinite(alpha).all():
        issues.append("empty_or_invalid_subject")
    if inside < 0.55:
        issues.append("subject_outside_target")
    retained = int((mask & original).sum()) / max(1, int(original.sum()))
    if retained < 0.50:
        issues.append("excessive_subject_removal")
    for entry in schedule.audit:
        if entry.get("stage") == "exclusion" and entry.get("unresolvedPixels", 0) > max(16, total * 0.02):
            # A neighbour's rectangle also overlaps correct subject pixels.
            # Lack of deletion evidence is a warning, not proof of a bad mask.
            warnings.append("ambiguous_neighbour_overlap")
        if entry.get("skipped") in {"removal_budget_exceeded", "prior_covers_most_of_subject"}:
            issues.append(entry["skipped"])
        if entry.get("stage") == "detached_review" and entry.get("disputedPixels", 0):
            issues.append("unowned_detached_regions")
    if clipped_pixels:
        warnings.append("explicit_hint_crop_clips_subject")
    issues = list(dict.fromkeys(issues))
    accepted = not issues
    return {"status": "ok" if accepted else "low_quality",
            "runtimeAction": "accept" if accepted else "hold",
            "shouldGenerateRuntimeLayer": accepted, "needsHigherPrecision": not accepted,
            "issues": issues, "warnings": warnings,
            "validation": {"insideRatio": round(inside, 4),
                           "retainedOriginalRatio": round(retained, 4),
                           "candidateScore": candidate.get("score")}}


def _quality_summary(policy, edge, material, schedule, hint_bbox, alpha_box,
                     mask_box, selected_row, timings):
    delta = None
    if hint_bbox is not None and alpha_box is not None:
        delta = {
            "left": int(alpha_box[0] - hint_bbox[0]),
            "top": int(alpha_box[1] - hint_bbox[1]),
            "right": int(alpha_box[2] - hint_bbox[2]),
            "bottom": int(alpha_box[3] - hint_bbox[3]),
        }
    return {
        "version": CORE_VERSION_V5,
        "policyVersion": POLICY_VERSION_V5,
        "phase": policy.get("phase"),
        "qualityProfile": policy.get("qualityProfile", "publish"),
        "edgeClass": edge.get("class"),
        "edgeRadius": edge.get("radius"),
        "edgeFractionMid": edge.get("fractionMid"),
        "edgeContrast": edge.get("contrast"),
        "material": material.get("mode"),
        "materialShare": material.get("bgShare"),
        "canvas": {
            "hint": [int(round(v)) for v in hint_bbox] if hint_bbox else None,
            "mask": mask_box,
            "alpha": alpha_box,
            "deltaFromHint": delta,
        },
        "stages": schedule.audit,
        "candidate": selected_row,
        "intentFill": policy.get("intentFill"),
        "timingsMs": {key: round(value * 1000, 2) for key, value in timings.items()},
    }


def _to_uint8_alpha(alpha):
    return np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)


def review_detached_regions_v5(mask, hint, policy, schedule):
    """Drop small, detached out-of-hint fragments without clipping attached parts.

    Explicit included components can preserve disconnected subject pieces.
    Large disputed pieces are retained for the quality gate to review.
    """
    owned = np.zeros_like(mask)
    x1, y1, x2, y2 = [int(round(v)) for v in hint]
    owned[y1:y2, x1:x2] = True
    for box in (policy.get("intent") or {}).get("included") or []:
        pixels = _prior_box_pixels(box, mask.shape)
        if pixels:
            bx1, by1, bx2, by2 = pixels
            owned[by1:by2, bx1:bx2] = True
    labels, rows = _components(mask, 1)
    removed = np.zeros_like(mask)
    total = int(mask.sum())
    disputed = 0
    for row in rows:
        component = labels == row["label"]
        owned_share = int((component & owned).sum()) / max(1, row["area"])
        if owned_share >= 0.20:
            continue
        if row["area"] <= max(48, total * 0.08):
            removed |= component
        else:
            disputed += row["area"]
    schedule.record("detached_review", "removed" if removed.any() else "unchanged",
                    pixels=int(removed.sum()), disputedPixels=disputed)
    return mask & ~removed


def process_layer_v5(image, hint_bbox, candidate_masks, layer_meta=None,
                     context_layers=None, quality_profile="publish", policy=None,
                     context_exclusion_mask=None):
    """Run category-independent evidence refinement and report acceptance.

    The hint selects an object; accepted mask pixels may extend past it.
    Explicit SAMV5_EXPORT_BBOX=hint remains available as an opt-in crop.
    """
    started = time.perf_counter()
    timings = {}
    height, width = image.shape[:2]
    hint = _as_bbox(hint_bbox)
    if hint is None or not all(math.isfinite(v) for v in hint):
        return None
    hint = [_clamp(hint[0], 0, width), _clamp(hint[1], 0, height),
            _clamp(hint[2], 0, width), _clamp(hint[3], 0, height)]
    if _bbox_area(hint) <= 0:
        return None
    if policy is None:
        meta = dict(layer_meta or {})
        meta["_contextLayers"] = context_layers or []
        policy = resolve_policy_v5(meta, quality_profile)
    candidates = normalize_candidates_v5(candidate_masks, width, height)
    if not candidates:
        return None
    ownership = policy.get('_ownershipEvidence')
    if ownership is None and not policy.get('_ownershipEvidenceFrozen'):
        reference, _ = select_candidate_v5(image, candidates, hint, dict(policy, verbose=False), image.shape)
        ownership = build_ownership_evidence(hint, candidates, policy, image.shape, reference=reference)
    eligible, ownership_review = review_ownership_candidates(candidates, ownership)
    ownership_unresolved = ownership is not None and not eligible
    # Preserve every anchor-valid proposal for evidence, even if completeness
    # ranking later prefers one of them. Ranking is not independent agreement.
    evidence_candidates = candidates
    if ownership is not None:
        valid_indices = [row['index'] for row in ownership_review.get('candidates', [])
                         if row['accepted']]
        evidence_candidates = [candidates[i] for i in valid_indices]
    if eligible:
        candidates = eligible
    selected, ranking = select_candidate_v5(image, candidates, hint, policy, image.shape)
    if selected is None:
        return None
    selected_row = ranking[0]
    lab = _lab_image(image)
    gradient = _gradient_magnitude(lab)
    boundary_audit = BoundaryAudit(image, selected,
        reference=policy.get('_hqDefaultBoundaryReference'), candidates=evidence_candidates)
    boundary_audit.record('selected_candidate', selected)
    diag = math.hypot(width, height)
    window = _window_bbox(hint, selected, image.shape, _WINDOW_MARGIN_RATIO)
    exclusion_mask, soft_boxes = _split_priors(policy, image.shape)
    if context_exclusion_mask is not None:
        explicit = np.asarray(context_exclusion_mask, dtype=bool)
        if explicit.shape != selected.shape:
            raise ValueError("context_exclusion_mask must match image height/width")
        exclusion_mask |= explicit
    schedule = EvidenceSchedule(_build_protected(selected, policy, diag),
                                exclusion_mask, policy, soft_boxes)
    schedule.audit.append({"stage": "window", "hint": hint, "window": window,
                           "selectedArea": int(selected.sum())})
    mask = selected.copy()
    verified_negative_anchors = np.zeros_like(mask)
    if ownership is not None and not ownership_unresolved and not policy.get('_ownershipRetryUnresolved'):
        for x,y in ownership['negative']:
            if not selected[y,x]:
                verified_negative_anchors[y,x] = True
    verified_neighbour = policy.get('_ownershipNeighbourMask')
    ownership_conflict = False
    if verified_neighbour is not None:
        verified_neighbour = np.asarray(verified_neighbour, dtype=bool)
        if verified_neighbour.shape != mask.shape:
            raise ValueError('ownership neighbour mask must match image height/width')
        votes = np.sum(evidence_candidates, axis=0) if evidence_candidates else np.zeros_like(mask, int)
        # A shared one-pixel rim is uncertain when target proposals agree on
        # it too. Only the neighbour interior may override that agreement.
        disputed_rim = (verified_neighbour & ~_erode(verified_neighbour, 2) &
                        _erode(mask, 2) & (votes >= 2))
        verified_neighbour = verified_neighbour & ~disputed_rim
        schedule.record('ownership_contact', 'target_agreement_protected',
                        pixels=int((disputed_rim & mask).sum()))
        removal = mask & verified_neighbour
        ownership_conflict = int(removal.sum()) > max(16, int(mask.sum() * .25))
        if not ownership_conflict:
            mask &= ~verified_neighbour
            # Consensus can leave a few disconnected edge pixels of the
            # rejected instance. Remove only tiny fragments next to its
            # verified silhouette; never extend the cut into the main object.
            near_neighbour = _dilate(verified_neighbour, 3)
            labels, fragments = _components(mask, 1)
            for fragment in fragments:
                if fragment['area'] > max(12, int(selected.sum() * .0005)):
                    continue
                region = labels == fragment['label']
                if ownership is not None and (region & ownership['safeCore']).any():
                    continue
                if np.mean(near_neighbour[region]) >= .75:
                    mask &= ~region
            schedule.record('ownership_neighbour', 'verified_consensus', pixels=int(removal.sum()))
    baseline = policy.get('_ownershipBaselineMasks')
    original_reference = None
    original_ranking = []
    if ownership is not None and baseline:
        original_reference, original_ranking = select_candidate_v5(
            image, baseline, hint, dict(policy, verbose=False), image.shape)
    if (ownership_unresolved or ownership_conflict or policy.get('_ownershipRetryUnresolved')):
        original_recovery = np.zeros_like(mask)
        contour_review = {'restoredPixels': 0, 'reason': 'ownership_not_verified'}
    else:
        original_recovery, contour_review = restore_owned_contour(
            mask, baseline, ownership,
            verified_neighbour if not ownership_conflict else None, reference=original_reference)
    if original_ranking:
        contour_review['referenceIndex'] = original_ranking[0]['index']
    mask |= original_recovery
    boundary_audit.record('ownership', mask, authority=verified_neighbour)
    schedule.record('ownership_contour_restore', 'original_contour_reference', **contour_review)
    models = _color_models(lab, mask, window, diag, exclusion_mask)
    if models is None:
        return None
    models["lab"] = lab
    models["hint"] = hint
    if ownership is not None and not ownership_unresolved and not policy.get('_ownershipRetryUnresolved'):
        # A color palette alone cannot revoke pixels grounding a validated
        # declared component. Independent neighbor masks still take priority.
        regions = ownership.get('componentRegions') or []
        if regions:
            models['contactProtected'] = np.logical_or.reduce(regions) & mask
    confirmed_support=policy.get('_componentVerifiedSupport')
    if confirmed_support is not None:
        models['contactProtected'] = (models.get('contactProtected',np.zeros_like(mask))
                                      |(np.asarray(confirmed_support,dtype=bool)&mask))
    t0 = time.perf_counter()
    if verified_neighbour is not None and not ownership_conflict:
        # A locally verified neighbour does not certify every other pixel in
        # the exclusion priors. Review the remaining region independently.
        models['reviewedNeighbour'] = verified_neighbour
    cleanup_authority = verified_negative_anchors.copy()
    if verified_neighbour is not None and not ownership_conflict:
        cleanup_authority |= verified_neighbour
    before_exclusion = mask.copy()
    mask = exclude_prior_pixels_v5(mask, models, schedule, policy)
    mask = review_cleanup_deletions_v5(
        before_exclusion, mask, evidence_candidates, lab, gradient, schedule,
        'exclusion', cleanup_authority)
    mask, connected_background = boundary_audit.cleanup(
        before_exclusion, mask, lab, cleanup_authority, models, 'color_exclusion')
    verified_background_deletions = connected_background & before_exclusion & ~mask
    boundary_audit.record('exclusion', mask, authority=cleanup_authority, background=connected_background)
    blocked = (selected & ~mask) | verified_negative_anchors
    mask &= ~verified_negative_anchors
    if verified_neighbour is not None and not ownership_conflict:
        blocked |= verified_neighbour
    if not mask.any():
        return None
    models = _color_models(lab, mask, window, diag, exclusion_mask) or models
    edge = _classify_edge(lab, mask, models, gradient, window, diag)
    material = _classify_material(lab, mask, models, gradient, edge, policy, diag)
    material_alpha, matte_background, matte_review = backdrop_matte(image, mask, hint, layer_meta)
    schedule.audit.append(dict(stage='material_matte', **matte_review))
    _log(policy, 'material_matte', matte_review['status'],
         **{k:v for k,v in matte_review.items() if k!='status'})
    if material_alpha is not None:
        material = dict(material, mode='translucent', asserted=True,
                        reason=matte_review['reason'], matteMode=matte_review['mode'])
        if matte_review['mode']=='soft_fringe':
            edge = dict(edge, **{'class':'soft'})
    # Absorbed backdrop is not proof of transparency. Only an asserted,
    # image-supported translucent interior bypasses opaque cleanup.
    if material["mode"] == "translucent" and not material.get("asserted"):
        material = dict(material, mode="opaque", reason="unasserted_background_requires_cleanup")
    if material["mode"] != "translucent":
        before = mask.copy()
        mask = review_background_v5(lab, mask, models, gradient, window, schedule, edge, policy)
        mask = review_cleanup_deletions_v5(
            before, mask, evidence_candidates, lab, gradient, schedule,
            'background', cleanup_authority)
        mask, connected_background = boundary_audit.cleanup(
            before, mask, lab, cleanup_authority, models, 'connected_background')
        verified_background_deletions |= connected_background & before & ~mask
        blocked |= before & ~mask
    else:
        schedule.record_skip("background", "supported_translucent_interior")
    boundary_audit.record('background', mask, authority=cleanup_authority,
                          background=connected_background)
    timings["cleanup"] = time.perf_counter() - t0
    if not mask.any():
        return None
    # Removed regions are permanent negative evidence for every later stage.
    schedule.forbidden |= blocked
    models = _color_models(lab, mask, window, diag, exclusion_mask) or models
    t0 = time.perf_counter()
    completion_evidence = None
    accepted_additions = original_recovery.copy()
    if material["mode"] != "translucent":
        before_completion = mask.copy()
        mask = complete_from_hint_v5(lab, mask, models, gradient, hint, window,
                                     schedule, policy, edge)
        mask = complete_by_color_v5(lab, mask, models, gradient, window, schedule, policy)
        proposed_completion = mask & ~before_completion & ~exclusion_mask & ~blocked
        # A proposal rejected by the vote gate can still request a separate
        # SAM verification pass. It cannot authorize a pixel edit itself.
        intent = policy.get('intent') or {}
        if intent.get('available') and intent.get('bboxRole') == 'tight_subject':
            labels_completion, parts_completion = _components(proposed_completion, 24)
            points = []
            for part in sorted(parts_completion, key=lambda row: -row['area'])[:3]:
                region = labels_completion == part['label']
                depth = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
                y, x = np.unravel_index(int(depth.argmax()), depth.shape)
                if depth[y, x] >= 1.5:
                    points.append([int(x), int(y)])
            # A small visible omission can be an entire small part. Request a
            # model verification based on region depth/size, not whole-object
            # area. This changes prompts only; vote-rejected pixels stay out.
            if points and int(proposed_completion.sum()) >= 48:
                completion_evidence = {'points': points, 'mask': proposed_completion}
        # Colour resemblance cannot invent ownership. Outside-hole recovery
        # needs agreement from independent SAM proposals as well.
        votes = np.zeros(mask.shape, dtype=np.uint16)
        for proposal in evidence_candidates:
            votes += proposal
        supported = votes >= 2
        unsupported = mask & ~before_completion & ~supported
        mask &= ~unsupported
        accepted_additions |= mask & ~before_completion
        schedule.record("completion_consensus", "reviewed",
                        rejectedPixels=int(unsupported.sum()),
                        addedPixels=int((mask & ~before_completion).sum()))
    boundary_audit.record('completion_consensus', mask)
    openings, opening_pixels = _perforation_stats(mask)
    perforated = openings >= _CHANNEL_MIN_OPENINGS and opening_pixels >= max(
        16, int(mask.sum() * _CHANNEL_MIN_OPENING_SHARE))
    before_holes = mask.copy()
    mask = repair_holes_v5(mask, models, schedule, edge, material, perforated)
    verified_hole_additions = mask & ~before_holes
    accepted_additions |= verified_hole_additions
    mask &= ~blocked
    boundary_audit.record('hole_repair', mask)
    before = mask.copy()
    mask = review_detached_regions_v5(mask, hint, policy, schedule)
    blocked |= before & ~mask
    boundary_audit.record('detached_cleanup', mask)
    if material['mode'] != 'translucent':
        # This narrow contour repair uses local foreground/background pixels,
        # not a two-proposal vote. It therefore works with a single valid SAM
        # proposal without permitting broad same-colour neighbour completion.
        boundary_forbidden = blocked.copy()
        thin_guard = np.zeros_like(mask)
        for box in (policy.get('intent') or {}).get('excluded',[]):
            x,y,X,Y=box
            if X-x > 2*(Y-y) and (Y-y)*height/1000 < (hint[3]-hint[1])*.5:
                continue
            from ownership_v5 import _box_mask
            thin_guard |= _box_mask(box,mask.shape,True)
        added, repair = boundary_evidence(lab, mask, boundary_forbidden,thin_guard=thin_guard)
        mask |= added
        schedule.record('boundary_repair', 'local_image_evidence', **repair)
    boundary_audit.record('boundary_repair', mask)
    timings["completion"] = time.perf_counter() - t0
    models = _color_models(lab, mask, window, diag, exclusion_mask) or models
    edge = _classify_edge(lab, mask, models, gradient, window, diag)
    if material_alpha is not None and matte_review['mode']=='soft_fringe':
        edge = dict(edge, **{'class':'soft'})
    schedule.audit.append({"stage": "edge_measurement", "edge": edge,
                           "material": material, "perforated": perforated})
    t0 = time.perf_counter()
    alpha = estimate_band_alpha_v5(lab, mask, models, edge, material, window, policy)
    boundary_audit.record('alpha_estimate', alpha >= .5, alpha=alpha)
    if (edge.get('class') == 'hard' and material.get('mode') == 'opaque'
            and not perforated and material_alpha is None):
        from boundary_refine_v5 import refine_opaque_contour
        refined = refine_opaque_contour(lab, mask, alpha)
        schedule.record('contour_antialias', 'coherent_image_edge',
                        pixels=int((np.abs(refined-alpha)>1/255).sum()))
        alpha = refined
    boundary_audit.record('contour_antialias', alpha >= .5, alpha=alpha)
    if material_alpha is not None:
        alpha = material_alpha.copy()
        # Material coverage was estimated before detached-region review.
        # Deleted fragments cannot leave an orphaned translucent halo behind.
        support_radius = max(3, min(16, round(min(window[2]-window[0],
                                                window[3]-window[1])*.0175)+1))
        alpha[~_dilate(mask, support_radius)] = 0.
    boundary_audit.record('material_alpha', alpha >= .5, alpha=alpha)
    channel = None
    if material["mode"] != "translucent":
        channel = channel_coverage_v5(image, mask, window, schedule)
    if channel is not None:
        # Texture contrast is not transparency. Even an accepted regional
        # channel needs independent per-pixel backdrop agreement; otherwise
        # woven upholstery is mistaken for lace and its whole seat disappears.
        channel_review = next((r for r in reversed(schedule.audit)
                               if r.get('stage')=='channel'), {})
        if channel_review.get('interiorEvidence'):
            # Channel coverage has its own observed-exterior color test.
            # A foreground palette contaminated by absorbed backdrop cannot
            # veto that test throughout every interior gap.
            channel_pixels = mask
        else:
            channel_pixels = (mask & ~_erode(mask,2) &
                              (models["bgDistFull"]<12.) &
                              (models["bgDistFull"]+6.<models["fgDistFull"]))
        alpha[channel_pixels] = np.minimum(alpha[channel_pixels], channel[channel_pixels])
        schedule.record("channel_blend", "background_verified_only",
                        pixels=int(channel_pixels.sum()))
    boundary_audit.record('channel_alpha', alpha >= .5, alpha=alpha)
    alpha, alpha_review = constrain_alpha(image, boundary_audit.selected,
        boundary_audit.protected, boundary_audit.exterior, mask, alpha, blocked,
        material, edge, material_alpha, perforated)
    boundary_audit.record('alpha_boundary_guard',alpha >= .5,alpha=alpha)
    schedule.audit.append(dict(stage='alpha_boundary_guard',**alpha_review))
    from boundary_control_v5 import constrain_global_boundary
    mask, alpha, restored, global_review = constrain_global_boundary(
        image, boundary_audit.selected, mask, alpha, boundary_audit.protected,
        cleanup_authority, verified_background_deletions, accepted_additions & mask,
        material, edge, material_alpha, perforated)
    # Only this explicitly reviewed, non-authoritative boundary restoration
    # can revoke an earlier soft deletion. Hard ownership remains immutable.
    blocked &= ~restored
    boundary_audit.record('global_boundary_guard', alpha >= .5,
                          authority=cleanup_authority, background=connected_background, alpha=alpha)
    schedule.audit.append(dict(stage='global_boundary_guard', **global_review))
    # Real openings remain transparent even when narrower than the alpha band.
    padded = np.pad(mask.astype(np.uint8), 1)
    _, free = cv2.connectedComponents((padded == 0).astype(np.uint8), connectivity=4)
    holes = ((free != 0) & (free != free[0, 0]))[1:-1, 1:-1]
    alpha[(holes if material_alpha is None else np.zeros_like(holes)) | blocked] = 0.0
    if ownership is not None:
        # A source-supported negative anchor excluded by the accepted binary
        # proposal cannot be recreated by the subsequent antialias halo.
        for x,y in ownership['negative']:
            if not mask[y,x]:
                alpha[y,x] = 0.0
    # A rectangular prior is not negative pixel evidence. Verified neighbour
    # and cleanup pixels are already blocked above; cutting the entire prior
    # here used to cancel antialias along otherwise valid slanted target legs.
    timings["alpha"] = time.perf_counter() - t0
    clipped_pixels = 0
    export_limit = hint if _EXPORT_BBOX_SOURCE == "hint" else None
    if export_limit is not None:
        allowed = np.zeros(mask.shape, bool)
        x1, y1, x2, y2 = [int(round(v)) for v in hint]
        allowed[y1:y2, x1:x2] = True
        clipped_pixels = int(((alpha > 0) & ~allowed).sum())
        alpha[~allowed] = 0.0
        mask &= allowed
        schedule.record("export", "explicit_hint_crop", pixels=clipped_pixels)
    alpha_box = _mask_bbox(alpha > 0)
    if alpha_box is None:
        return None
    quality = _quality_summary(policy, edge, material, schedule, hint, alpha_box,
                               _mask_bbox(mask), selected_row, timings)
    quality.update(maskPixels=int(mask.sum()), alphaPixels=int((alpha > 0).sum()),
                   exportBboxSource="hint" if export_limit is not None else "alpha",
                   clippedPixels=clipped_pixels, candidateRanking=ranking)
    boundary_audit.record('final_alpha', alpha >= .5, authority=cleanup_authority,
                          background=connected_background, alpha=alpha)
    quality['boundaryAudit'] = boundary_audit.summary()
    quality['alphaBoundaryGuard'] = alpha_review
    quality['globalBoundaryGuard'] = global_review
    quality.update(evaluate_quality_v5(mask, alpha, hint, selected_row, schedule,
                                       selected, clipped_pixels))
    if global_review.get('unresolved'):
        quality.setdefault('warnings', []).append('global_boundary_deviation_review_required')
        quality['needsBoundaryReview'] = True
    if ownership is not None:
        _, final_review = review_ownership_candidates(
            [mask if material_alpha is not None else alpha >= .5], ownership)
        if material_alpha is not None:
            final_review['validationDomain'] = 'geometric_support_before_transmission'
        quality['ownershipReview'] = dict(policy.get('_ownershipReview') or ownership_review,
                                          final=final_review)
        if (ownership_unresolved or ownership_conflict or policy.get('_ownershipRetryUnresolved') or
                final_review['status'] == 'unresolved'):
            quality.update(status='low_quality', runtimeAction='hold',
                           shouldGenerateRuntimeLayer=False, needsHigherPrecision=True,
                           issues=[*(quality.get('issues') or []), 'instance_ownership_unresolved'])
    quality["timingsMs"]["total"] = round((time.perf_counter() - started) * 1000, 2)
    return {"mask": mask, "alpha": _to_uint8_alpha(alpha), "bbox": alpha_box,
            "_boundaryAudit": boundary_audit,
            "mattingBackground": matte_background, "materialMatte": matte_review,
            "completionEvidence": completion_evidence,
            "foreground": models["fgPalette"], "backgroundPalette": models["bgPalette"],
            "quality": quality, "policy": {"version": policy.get("version"),
            "phase": policy.get("phase"), "edge": edge["class"], "material": material["mode"]}}


def build_cutout_entry_v5(
    image, result, layer_id, extract_engine="sam_bbox_prompt_v5",
    decontaminate=True,
):
    """Build the transport cutout from a processed layer.

    The crop is the alpha bbox. Edge colour decontamination is applied to
    partially transparent pixels so dark backgrounds do not leave a fringe.
    """
    if not result:
        return None
    x1, y1, x2, y2 = [int(v) for v in result["bbox"]]
    height, width = image.shape[:2]
    x1 = int(_clamp(x1, 0, width - 1))
    y1 = int(_clamp(y1, 0, height - 1))
    x2 = int(_clamp(x2, x1 + 1, width))
    y2 = int(_clamp(y2, y1 + 1, height))

    crop = image[y1:y2, x1:x2]
    alpha_crop = result["alpha"][y1:y2, x1:x2]
    if crop.size == 0:
        return None

    bgr = crop
    if decontaminate:
        try:
            lab = _lab_image(image)
            palette = result.get("backgroundPalette")
            matte_background = result.get("mattingBackground")
            if matte_background is not None:
                bgr = decontaminate_edge_colors_v5(crop, alpha_crop.astype(np.float32)/255.,
                                                   np.asarray(matte_background).reshape(1,1,3))
            elif palette is not None and len(palette):
                background = lab.copy()
                # Background estimate: use the nearest palette colour, which is
                # stable outside the subject and avoids a global average.
                nearest = np.argmin(
                    np.stack([
                        np.linalg.norm(lab - center.reshape(1, 1, 3), axis=2)
                        for center in palette
                    ], axis=0), axis=0
                )
                for index, center in enumerate(palette):
                    background[nearest == index] = center
                # The compositing equation operates in BGR, not Lab.
                background = cv2.cvtColor(np.clip(background, 0, 255).astype(np.uint8),
                                          cv2.COLOR_LAB2BGR)
                bgr = decontaminate_edge_colors_v5(
                    crop, alpha_crop.astype(np.float32) / 255.0,
                    background[y1:y2, x1:x2],
                )
        except Exception as error:  # pragma: no cover - defensive
            print(f"SAMV5 export decontamination skipped: {error}")
            bgr = crop

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha_crop
    rgb_change = np.any(bgr != crop, axis=2)
    result['quality']['rgbAudit'] = dict(changedPixels=int(rgb_change.sum()),
        changedOpaquePixels=int((rgb_change & (alpha_crop == 255)).sum()),
        changedTransparentPixels=int((rgb_change & (alpha_crop == 0)).sum()),
        geometryChanged=False)
    diagnostic_color_image('export_source_rgb', crop, crop=[x1,y1,x2,y2])
    diagnostic_color_image('export_png_rgba', bgra, crop=[x1,y1,x2,y2],
                         rgbReview=result['quality']['rgbAudit'])
    ok, buffer = cv2.imencode(".png", bgra)
    if not ok:
        return None
    import base64
    # The client feeds cutout.image straight into dataURLToFile, and the
    # original transport contract is a full data URL, not a bare payload.
    encoded = "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")

    return {
        "layerId": layer_id,
        "bbox": [
            int(y1 / height * 1000), int(x1 / width * 1000),
            int(y2 / height * 1000), int(x2 / width * 1000),
        ],
        "image": encoded,
        "width": int(x2 - x1),
        "height": int(y2 - y1),
        "extractEngine": extract_engine,
        "quality": result.get("quality"),
    }
