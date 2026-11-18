"""Disconnected geometry is ambiguous until instance ownership supports a cut."""
import cv2
import numpy as np
from ownership_v5 import _box_mask


def uncertain_components(reference, hint, policy):
    """Do not turn small excluded-overlap islands into mandatory part anchors.

    This is a candidate-retention classification, never deletion permission.
    Unknown or explicitly verified components retain the established protection.
    """
    empty = np.zeros(reference.shape, bool)
    intent = policy.get('intent') or {}
    if (not intent.get('available') or intent.get('confidence', 0) < .85 or
            intent.get('bboxRole') != 'tight_subject' or not intent.get('excluded')):
        return empty
    excluded = np.logical_or.reduce([_box_mask(b, reference.shape, True)
                                    for b in intent['excluded']])
    verified = np.asarray(policy.get('_componentVerifiedSupport', empty), bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(reference.astype(np.uint8), 8)
    if count < 3:
        return empty
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    uncertain = empty.copy()
    limit = max(12, int(reference.sum() * .005))
    for i in range(1, count):
        if i == largest or stats[i, cv2.CC_STAT_AREA] > limit:
            continue
        region = labels == i
        if not (region & verified).any() and float(excluded[region].mean()) >= .85:
            uncertain |= region
    return uncertain


def review_neighbour_residue(reference, cut, agreement, safe, policy, positive):
    """Complete a verified local cut through small disconnected neighbour rims.

    Requires two neighbour queries to claim the fragment, a nearby established
    target-rejection cut, and no exclusive/verified subject support. A box or
    color match by itself cannot delete anything, including in-hint fragments.
    """
    added = np.zeros(reference.shape, bool)
    if not cut.any():
        return added, []
    remaining = reference & ~cut
    n, labels, stats, _ = cv2.connectedComponentsWithStats(remaining.astype(np.uint8), 8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
    distance = cv2.distanceTransform((~cut).astype(np.uint8), cv2.DIST_L2, 5)
    radius = max(2, min(5, int(round(np.hypot(*reference.shape) * .0025))))
    verified = np.asarray(policy.get('_componentVerifiedSupport', added), bool)
    reviews = []
    for i in range(1, n):
        if i == largest or stats[i, cv2.CC_STAT_AREA] > max(12, reference.sum() * .005):
            continue
        region = labels == i
        if ((region & (safe | verified)).any() or
                any(region[y, x] for x, y in positive)):
            continue
        support = float(agreement[region].mean())
        proximity = float((distance[region] <= radius).mean())
        accepted = support >= .85 and proximity >= .75
        reviews.append(dict(pixels=int(region.sum()), neighbourAgreement=support,
                            nearVerifiedCut=proximity, radius=radius, accepted=accepted))
        if accepted:
            added |= region
    return added, reviews
