"""Category-independent instance ownership constraints for SAM proposals.

The target and neighbour rectangles can overlap. Only proposal-supported pixels
in their exclusive regions seed prompts; overlapping rectangles are never cut.
"""
import cv2
import numpy as np


def _box_mask(box, shape, normalized=False):
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    if normalized:
        x1, x2 = x1 * width / 1000, x2 * width / 1000
        y1, y2 = y1 * height / 1000, y2 * height / 1000
    x1, x2 = np.clip(np.rint([x1, x2]).astype(int), 0, width)
    y1, y2 = np.clip(np.rint([y1, y2]).astype(int), 0, height)
    mask = np.zeros((height, width), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _interior_points(region, limit):
    """Use interior distance and spatial diversity, never rectangle centres."""
    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    points = []
    radius = max(4, int(round(np.sqrt(max(1, int(region.sum()))) * .20)))
    yy, xx = np.ogrid[:region.shape[0], :region.shape[1]]
    for _ in range(limit):
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        if distance[y, x] < 1.5:
            break
        points.append([int(x), int(y)])
        distance[(xx-x)**2 + (yy-y)**2 <= radius**2] = 0
    return points


def build_ownership_evidence(hint, candidates, policy, shape, reference=None):
    """Build image/proposal-grounded anchors from trusted explicit intent only."""
    intent = policy.get('intent') or {}
    if (not intent.get('available') or intent.get('confidence', 0) < .85 or
            intent.get('bboxRole') != 'tight_subject' or not intent.get('excluded')):
        return None
    included = intent.get('included') or []
    if not included or not candidates:
        return None
    candidates = [np.asarray(mask) > .5 for mask in candidates]
    includes = [_box_mask(box, shape, True) for box in included]
    target = np.logical_or.reduce(includes)
    target |= _box_mask(hint, shape)
    neighbours = [_box_mask(box, shape, True) for box in intent['excluded']]
    excluded = np.logical_or.reduce(neighbours)
    union = np.logical_or.reduce(candidates)
    votes = np.sum(candidates, axis=0)
    agreement = votes >= min(2, len(candidates))
    # Multimask outputs often describe a part and a whole, not independent
    # votes for the same object. A ranked whole-target reference defines what
    # must be retained; agreement on a tiny part cannot stand in for it.
    reference_mask = None if reference is None else np.asarray(reference) > .5
    anchor_support = agreement if reference_mask is None else reference_mask
    # A declared component may overlap a neighbour rectangle. Reference
    # pixels connected to the exclusive core can still ground its anchors.
    exclusive_core = anchor_support & target & ~excluded
    _, component_labels = cv2.connectedComponents(
        anchor_support.astype(np.uint8), connectivity=8)
    owners = np.unique(component_labels[exclusive_core])
    connected_reference = anchor_support & np.isin(component_labels, owners[owners > 0])
    positive_groups = []
    component_regions = []
    for component in includes:
        safe = component & ~excluded & anchor_support
        points = _interior_points(safe, 2 if reference_mask is None else 4)
        if not points:
            safe = component & connected_reference
            points = _interior_points(safe, 2)
        if not points:
            safe = component & ~excluded & union
            points = _interior_points(safe, 1)
        if points:
            positive_groups.append(points)
            component_regions.append(safe)
    negative = []
    neighbour_groups = []
    for neighbour in neighbours:
        group_points = []
        # Use the union of proposals for negative anchors. A neighbouring
        # instance may appear in only one multimask head while the other heads
        # describe the target at a different scale. Requiring agreement here
        # can suppress the later whole-object completion retry; the candidate
        # and final ownership gates still validate these anchors.
        exclusive = neighbour & ~target & union
        count, labels, stats, _ = cv2.connectedComponentsWithStats(exclusive.astype(np.uint8), 8)
        for owner in sorted(range(1, count), key=lambda n: -stats[n, cv2.CC_STAT_AREA])[:3]:
            if stats[owner, cv2.CC_STAT_AREA] < 12:
                continue
            group_points.extend(_interior_points(labels == owner, 2))
        neighbour_groups.append({'region':neighbour,'negative':group_points[:8],
                                 'supportPixels':int(exclusive.sum())})
        negative.extend(group_points)
    # If a neighbour never extends outside the target/component rectangles,
    # geometry alone cannot identify which overlapping pixels it owns.
    if not positive_groups or not negative:
        return None
    positive = [point for group in positive_groups for point in group]
    negative = negative[:8]
    safe = anchor_support & target & ~excluded
    safe = cv2.erode(safe.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return {'positive': positive, 'positiveGroups': positive_groups, 'negative': negative,
            'points': [positive + negative],
            'labels': [[1]*len(positive) + [0]*len(negative)],
            'safeCore': safe, 'excluded': excluded, 'targetRegion': target,
            'componentRegions': component_regions, 'neighbourGroups': neighbour_groups,
            'negativeSupport': int(sum(g['supportPixels'] for g in neighbour_groups))}


def restore_owned_contour(selected, baseline, evidence, neighbour=None, reference=None):
    """Recover original contour only in independently owned target regions.

    Original masks can merge instances. Their ranked outline is a contour
    reference, not a new whole-object proposal. Only nearby pixels outside all declared
    neighbour regions may return, and detached fragments cannot seed recovery.
    """
    restored = np.zeros_like(selected)
    audit = {'restoredPixels': 0, 'reason': 'no_original_contour_reference'}
    if evidence is None or not baseline or len(baseline) < 2:
        return restored, audit
    target = evidence.get('targetRegion')
    if target is None:
        return restored, audit
    consensus = np.sum([np.asarray(mask) > .5 for mask in baseline], axis=0) >= 2
    # Use the independently ranked original outline when available. Voting
    # away its fringe repeats the contour loss we are trying to prevent.
    contour = consensus if reference is None else np.asarray(reference) > .5
    # More than an antialias rim, bounded relative to the actual target size.
    _, _, width, height = cv2.boundingRect(target.astype(np.uint8))
    reach = int(np.clip(round(np.hypot(width, height)*.015), 3, 24))
    distance = cv2.distanceTransform((~selected).astype(np.uint8), cv2.DIST_L2, 5)
    excluded = cv2.dilate(evidence['excluded'].astype(np.uint8),
                          np.ones((3, 3), np.uint8)) > 0
    allowed = contour & target & ~excluded & (distance <= reach)
    if neighbour is not None:
        allowed &= ~neighbour
    # Flood only within the old outline starting at the verified target. This
    # excludes nearby disconnected original fragments and isolated specks.
    _, labels = cv2.connectedComponents((allowed | selected).astype(np.uint8), connectivity=8)
    owners = np.unique(labels[selected])
    restored = allowed & ~selected & np.isin(labels, owners[owners != 0])
    audit = {'restoredPixels': int(restored.sum()), 'reach': reach,
             'reason': 'original_reference_in_exclusive_target_region' if reference is not None
                       else 'original_consensus_in_exclusive_target_region'}
    return restored, audit


def review_ownership_candidates(candidates, evidence):
    """Require target anchors, negative anchors and protected-core retention."""
    if evidence is None:
        return list(candidates), {'status': 'not_needed', 'reason': 'no_exclusive_ownership_anchors'}
    safe = evidence['safeCore']
    safe_area = max(1, int(safe.sum()))
    accepted, rows = [], []
    for index, mask in enumerate(candidates):
        # A tiny detached false positive must not disqualify an otherwise
        # complete object and force selection of a part-only proposal.
        # Remove only negative-anchored components wholly outside the target,
        # with no positive anchor; never cut the connected subject here.
        mask = np.asarray(mask, dtype=bool).copy()
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        removed = 0
        target = evidence.get('targetRegion')
        for owner in {int(labels[y, x]) for x, y in evidence['negative']} - {0}:
            region = labels == owner
            if (target is not None and not (region & target).any()
                    and stats[owner, cv2.CC_STAT_AREA] <= max(12, int(mask.sum() * .005))
                    and not any(region[y, x] for x, y in evidence['positive'])):
                removed += int(region.sum())
                mask[region] = False
        pos = sum(bool(mask[y, x]) for x, y in evidence['positive'])
        neg = sum(bool(mask[y, x]) for x, y in evidence['negative'])
        retention = int((mask & safe).sum()) / safe_area if safe.any() else 1.0
        groups = all(any(mask[y, x] for x, y in group) for group in evidence['positiveGroups'])
        component_retention = [float((mask & region).sum()) / max(1, int(region.sum()))
                               for region in evidence.get('componentRegions', [])]
        valid = (groups and pos == len(evidence['positive']) and neg == 0 and retention >= .90
                 and all(value >= .90 for value in component_retention))
        rows.append({'index': index, 'positiveHits': pos, 'negativeHits': neg,
                     'safeCoreRetention': round(retention, 4),
                     'componentRetention': [round(v, 4) for v in component_retention],
                     'detachedNegativePixels': removed,
                     'accepted': bool(valid)})
        if valid:
            accepted.append(mask)
    # A proposal missing an entire material/component (e.g. the base) must
    # not beat a more complete proposal merely by fitting the hint tightly.
    if accepted:
        best = max(row['safeCoreRetention'] for row in rows if row['accepted'])
        accepted = [mask for mask, row in zip(accepted, [r for r in rows if r['accepted']])
                    if row['safeCoreRetention'] >= best - .01]
    return accepted, {'status': 'accepted' if accepted else 'unresolved',
                      'reason': 'instance_ownership_anchors', 'candidates': rows,
                      'positivePoints': evidence['positive'], 'negativePoints': evidence['negative']}


def neighbour_consensus(candidates, evidence):
    """Only independently prompted, anchor-consistent neighbour agreement cuts.

    Rectangles restrict the review, but cannot themselves authorize removal.
    Disconnected visible pieces of an occluded neighbour can be included.
    """
    valid = []
    for proposal in candidates:
        mask = np.asarray(proposal) > .5
        if (all(mask[y, x] for x, y in evidence['negative']) and
                not any(mask[y, x] for x, y in evidence['positive'])):
            valid.append(mask)
    if len(valid) < 2:
        return np.zeros_like(evidence['excluded'])
    return (np.sum(valid, axis=0) >= 2) & evidence['excluded']
