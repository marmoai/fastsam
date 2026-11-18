"""Narrow, image-supported contour repairs independent of category or SAM count."""
import cv2
import numpy as np


def refine_opaque_contour(lab, mask, alpha):
    """Subpixel coverage on thick exterior silhouettes with coherent image edges.

    Never edits the binary ownership mask, holes, or unsupported thin details.
    Local colour mixture must explain the source before replacing coverage.
    """
    kernel = np.ones((3, 3), np.uint8)
    inside = cv2.erode(mask.astype(np.uint8), kernel, iterations=3) > 0
    if not inside.any():
        return alpha
    outside = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3) == 0
    fg, bg = _nearest_color(lab, inside), _nearest_color(lab, outside)
    if fg is None or bg is None:
        return alpha
    vector = fg-bg
    denominator = np.sum(vector*vector, axis=2)
    coverage = np.clip(np.sum((lab-bg)*vector, axis=2)/np.maximum(denominator, 1), 0, 1)
    residual = np.linalg.norm(lab-(bg+coverage[..., None]*vector), axis=2)
    # A smooth, directed source edge differs from texture or an isolated barb.
    gray = cv2.GaussianBlur(lab[:, :, 0], (3, 3), .7)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    xx = cv2.boxFilter(gx*gx, -1, (5, 5))
    yy = cv2.boxFilter(gy*gy, -1, (5, 5))
    xy = cv2.boxFilter(gx*gy, -1, (5, 5))
    coherence = np.sqrt((xx-yy)**2+4*xy*xy)/np.maximum(xx+yy, 1)
    near_core = cv2.dilate(inside.astype(np.uint8), kernel, iterations=4) > 0
    band = ((cv2.dilate(mask.astype(np.uint8), kernel)>0) &
            ~(cv2.erode(mask.astype(np.uint8), kernel, iterations=2)>0))
    padded = np.pad((~mask).astype(np.uint8), 1, constant_values=1)
    _, labels = cv2.connectedComponents(padded, connectivity=4)
    holes = ((labels != labels[0, 0]) & (labels != 0))[1:-1, 1:-1]
    protected = cv2.dilate(holes.astype(np.uint8), kernel, iterations=4)>0
    silhouette = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), .85)
    eligible = (band & near_core & ~protected & (denominator>900) &
                (residual<8) & (coherence>.75) & (xx+yy>100) &
                (np.abs(coverage-silhouette)<.9))
    # Source-derived coverage supplies the ramp; blur alone never adds pixels.
    proposed = .8*coverage + .2*silhouette
    result = alpha.copy()
    result[eligible] = proposed[eligible]
    return result


def _nearest_color(lab, seeds):
    if not seeds.any():
        return None
    _, labels = cv2.distanceTransformWithLabels(
        (~seeds).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    palette = np.zeros((int(labels.max()) + 1, 3), np.float32)
    palette[labels[seeds]] = lab[seeds]
    return palette[labels]


def boundary_evidence(lab, mask, forbidden=None, thin_guard=None):
    """Repair shallow notches only when nearby image colours support foreground.

    Morphology proposes pixels; it cannot authorize edits. The exterior sample
    is beyond the entire repair band so a missed rim cannot teach background.
    No foreground is eroded. Structural openings remain intact; tiny enclosed
    omissions require foreground colour evidence just like shallow notches.
    """
    radius = int(np.clip(round(np.hypot(*mask.shape) * .0022), 2, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    # Deeper recovery is confined to concavities. Only one pixel of free
    # outward recovery is allowed: a wider colour-only band grows shadows and
    # similarly coloured floor along a base instead of recovering its edge.
    expanded = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    rim = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    proposed = (closed | rim) & ~mask
    # Keep structural holes/slots. Tiny omissions still need the same local
    # image test; a white real opening will fail foreground colour evidence.
    padded = np.pad((~mask).astype(np.uint8), 1, constant_values=1)
    _, labels = cv2.connectedComponents(padded, connectivity=4)
    exterior = (labels == labels[0, 0])[1:-1, 1:-1]
    tiny_holes = np.zeros_like(mask)
    count, holes, stats, _ = cv2.connectedComponentsWithStats(
        ((~mask) & ~exterior).astype(np.uint8), 8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= min(16, radius*radius):
            tiny_holes |= holes == label
    proposed = (proposed & exterior) | tiny_holes
    if forbidden is not None:
        proposed &= ~forbidden
    if not proposed.any():
        return np.zeros_like(mask), {'radius': radius, 'proposedPixels': 0, 'addedPixels': 0}
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not inner.any():
        return np.zeros_like(mask), {'radius': radius, 'proposedPixels': int(proposed.sum()), 'addedPixels': 0}
    outer = ~(cv2.dilate(expanded.astype(np.uint8), kernel) > 0)
    fg, bg = _nearest_color(lab, inner), _nearest_color(lab, outer)
    if bg is None:
        return np.zeros_like(mask), {'radius': radius, 'proposedPixels': int(proposed.sum()), 'addedPixels': 0}
    fg_dist = np.linalg.norm(lab-fg, axis=2)
    bg_dist = np.linalg.norm(lab-bg, axis=2)
    separation = np.linalg.norm(fg-bg, axis=2)
    supported = ((separation >= 12) & (fg_dist + 5 < bg_dist) &
                 (fg_dist < np.maximum(12, separation*.45)))
    support_fraction = cv2.boxFilter(supported.astype(np.float32), -1, (3, 3))
    supported &= support_fraction >= .55
    added = proposed & supported
    # Thin supports already carry the HQ contour. Color resemblance alone
    # must not thicken them or grow parallel floor/shadow streaks. Repair
    # remains available beside a locally thick foreground core.
    depth = cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    thick = depth >= max(2.5,float(radius))
    reach = 2*radius+3
    near_thick = cv2.dilate(thick.astype(np.uint8),np.ones((reach,reach),np.uint8)) > 0
    guarded = np.zeros_like(mask) if thin_guard is None else thin_guard
    thin_rejected = int((added & ~near_thick & guarded).sum())
    added &= near_thick | ~guarded
    return added, {'radius': radius, 'proposedPixels': int(proposed.sum()),
                   'addedPixels': int(added.sum()),'thinUnsupportedPixels':thin_rejected}
