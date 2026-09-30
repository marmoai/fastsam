"""Bounded, image-guided review of hard mask boundaries; no model inference."""

import time

import cv2
import numpy as np


def review_hard_boundary(image, mask, target_bbox, context_mask=None, config=None):
    """Refine an uncertain edge only when two trimaps agree on the result.

    Context boxes enlarge uncertainty near the edge. They never become hard
    background labels: an overlapping box can also contain the target.
    Rejection returns the original mask, including its subpixel values.
    """
    config = config or {}
    started = time.perf_counter()

    def finish(result, status, reason, **evidence):
        return result, {
            "status": status, "reason": reason, **evidence,
            "durationMs": round((time.perf_counter() - started) * 1000, 1)
        }

    if image.shape[:2] != mask.shape or mask.ndim != 2 or not np.all(np.isfinite(mask)):
        return finish(mask, "skipped", "invalid_input")
    h, w = mask.shape
    x1, y1, x2, y2 = map(int, target_bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return finish(mask, "skipped", "invalid_bbox")
    if (x2 - x1) * (y2 - y1) > int(config.get("maxReviewPixels", 1500000)):
        return finish(mask, "skipped", "review_pixel_budget")
    base = mask[y1:y2, x1:x2] > .5
    area = int(np.count_nonzero(base))
    if area < 32:
        return finish(mask, "skipped", "insufficient_foreground")
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    context = np.zeros_like(base) if context_mask is None else context_mask[y1:y2, x1:x2].astype(bool)
    side = min(base.shape)
    radius = max(2, min(32, int(round(side * float(config.get("bandRatio", .03))))))
    radii = [radius, max(radius + 1, int(round(radius * 1.5)))]
    distances = cv2.distanceTransform(base.astype(np.uint8), cv2.DIST_L2, 5)
    stable_core = (distances > radii[-1]) & (~context | (distances > radii[-1] * 2))
    # Background-colored pockets can already be inside SAM's boundary. Use
    # learned color frequencies to make near-exterior pixels uncertain, never
    # force them to background or reclassify deeply enclosed subject details.
    broad_support = cv2.dilate(
        base.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radii[-1] * 2 + 1, radii[-1] * 2 + 1)
        )
    ) > 0
    if np.count_nonzero(~broad_support) < 32 or np.count_nonzero(stable_core) < 32:
        return finish(mask, "skipped", "insufficient_image_seeds")
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.int32) // 16
    bins = lab[:, :, 0] * 256 + lab[:, :, 1] * 16 + lab[:, :, 2]
    fg_frequency = np.bincount(bins[stable_core], minlength=4096).astype(float) + 1
    bg_frequency = np.bincount(bins[~broad_support], minlength=4096).astype(float) + 1
    fg_frequency /= fg_frequency.sum()
    bg_frequency /= bg_frequency.sum()
    _, holes = cv2.connectedComponents((~base).astype(np.uint8), connectivity=8)
    edge_ids = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    exterior = np.isin(holes, edge_ids[edge_ids != 0])
    outer_distance = cv2.distanceTransform((~exterior).astype(np.uint8), cv2.DIST_L2, 5)
    uncertain_color = (
        (bg_frequency[bins] > fg_frequency[bins] * float(config.get("backgroundOdds", 4))) &
        (outer_distance <= radii[-1] * 2)
    )
    stable_core &= ~uncertain_color
    core_ratio = np.count_nonzero(stable_core) / area
    if core_ratio < float(config.get("minCoreRatio", .40)):
        return finish(mask, "skipped", "insufficient_stable_core", coreRatio=round(core_ratio, 4))

    results = []
    try:
        for band in radii:
            core = (distances > band) & (~context | (distances > band * 2)) & ~uncertain_color
            support = cv2.dilate(
                base.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
            ) > 0
            if np.count_nonzero(~support) < 32:
                return finish(mask, "skipped", "insufficient_background")
            labels = np.full(base.shape, cv2.GC_BGD, np.uint8)
            labels[support] = cv2.GC_PR_BGD
            labels[base] = cv2.GC_PR_FGD
            labels[core] = cv2.GC_FGD
            cv2.grabCut(crop, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_MASK)
            proposed = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
            count, components = cv2.connectedComponents(proposed.astype(np.uint8), connectivity=8)
            anchored = np.zeros(count, bool)
            anchored[np.unique(components[core])] = True
            anchored[0] = False
            results.append(proposed & anchored[components])
    except cv2.error as error:
        return finish(mask, "rejected", "image_review_failed", error=str(error))

    small, proposed = results
    agreement = np.count_nonzero(small & proposed) / max(1, np.count_nonzero(small | proposed))
    added = int(np.count_nonzero(proposed & ~base))
    removed = int(np.count_nonzero(base & ~proposed))
    evidence = {
        "radii": radii, "agreement": round(agreement, 4),
        "addedPixels": added, "removedPixels": removed, "coreRatio": round(core_ratio, 4)
    }
    if agreement < float(config.get("minAgreement", .98)):
        return finish(mask, "rejected", "unstable_boundary", **evidence)
    if added > area * float(config.get("maxAddedRatio", .10)) or removed > area * float(config.get("maxRemovedRatio", .10)):
        return finish(mask, "rejected", "change_budget", **evidence)
    if np.any(stable_core & ~proposed):
        return finish(mask, "rejected", "core_loss", **evidence)
    # Protect substantial separate parts. A color-model pass must not erase a
    # small instance just because the main instance dominates its statistics.
    count, components, stats, _ = cv2.connectedComponentsWithStats(base.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        component_area = int(stats[index, cv2.CC_STAT_AREA])
        if component_area < max(32, area * .001):
            continue
        preserved = np.count_nonzero(proposed & (components == index)) / component_area
        if preserved < float(config.get("minComponentPreserve", .90)):
            return finish(mask, "rejected", "component_loss", component=index, **evidence)
    if added + removed == 0:
        return finish(mask, "skipped", "unchanged", **evidence)
    result = np.zeros_like(mask, dtype=np.float32)
    result[y1:y2, x1:x2] = proposed
    return finish(result, "accepted", "stable_image_boundary", **evidence)


def recover_safe_outer_boundary(
    image, mask, target_bbox, blocked_additions_mask=None, config=None
):
    """Add only image-verified exterior edge pixels to an owned SAM mask.

    A normal boundary review may trim foreground as well as add it.  That is
    unsuitable for a tightly owned instance beside a same-material sibling:
    identity is already settled by SAM's positive/negative prompts.  This
    adapter keeps that silhouette immutable, accepts only attached additions,
    and freezes additions inside the verified sibling region.
    """
    config = config or {}
    base = np.asarray(mask, dtype=np.float32) > 0.5
    if image.shape[:2] != base.shape or not np.any(base):
        return mask, {"status": "skipped", "reason": "invalid_input"}

    h, w = base.shape
    x1, y1, x2, y2 = [int(value) for value in target_bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return mask, {"status": "skipped", "reason": "invalid_bbox"}

    blocked = (
        np.zeros_like(base, dtype=bool)
        if blocked_additions_mask is None
        else np.asarray(blocked_additions_mask, dtype=bool)
    )
    if blocked.shape != base.shape:
        return mask, {"status": "skipped", "reason": "invalid_blocked_mask"}

    reviewed, review = review_hard_boundary(
        image,
        base.astype(np.float32),
        [x1, y1, x2, y2],
        context_mask=blocked,
        config=config
    )
    if review.get("status") != "accepted":
        return mask, {
            "status": "skipped",
            "reason": f"image_review_{review.get('status', 'unavailable')}",
            "review": review
        }

    proposal = np.asarray(reviewed, dtype=np.float32) > 0.5
    additions = proposal & ~base
    raw_additions = int(np.count_nonzero(additions))
    blocked_additions = int(np.count_nonzero(additions & blocked))
    additions &= ~blocked

    # The first review already uses a narrow support band. Keep an independent
    # geometric envelope here so a future review implementation cannot create
    # a distant island in this ownership-preserving route.
    band_radius = max(1, int(config.get("maxBandPixels", 8)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_radius * 2 + 1, band_radius * 2 + 1)
    )
    support = cv2.dilate(base.astype(np.uint8), kernel, iterations=1) > 0
    additions &= support
    target = np.zeros_like(base, dtype=bool)
    target[y1:y2, x1:x2] = True
    additions &= target

    merged = base | additions
    count, labels = cv2.connectedComponents(merged.astype(np.uint8), connectivity=8)
    anchored = np.zeros(count, dtype=bool)
    anchored[np.unique(labels[base])] = True
    anchored[0] = False
    additions &= anchored[labels]

    added = int(np.count_nonzero(additions))
    base_pixels = max(1, int(np.count_nonzero(base)))
    max_added = max(8, int(round(base_pixels * float(config.get("maxAddedRatio", .025)))))
    audit = {
        "status": "accepted" if added else "skipped",
        "reason": "verified_outer_boundary" if added else "no_safe_additions",
        "basePixels": base_pixels,
        "rawAddedPixels": raw_additions,
        "blockedAddedPixels": blocked_additions,
        "addedPixels": added,
        "removedPixels": 0,
        "maxAddedPixels": max_added,
        "review": review
    }
    if added > max_added:
        audit.update({"status": "skipped", "reason": "addition_budget"})
        return mask, audit
    if not added:
        return mask, audit

    result = base | additions
    return result.astype(np.float32), audit
