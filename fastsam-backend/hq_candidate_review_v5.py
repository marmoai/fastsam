"""Conditional HQ fusion selection; no pixel edits or logit reweighting."""
import cv2
import numpy as np
from ownership_v5 import build_ownership_evidence, review_ownership_candidates
from segmentation_core_v5 import score_candidates_v5


def _retains(candidate, reference):
    if reference.any() and candidate[reference].mean() < .94:
        return False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(reference.astype(np.uint8), 8)
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] >= 8 and candidate[labels == k].mean() < .90:
            return False
    return True


def review_hq_candidates(image, hint, defaults, evidence, policy):
    audit = dict(status='kept_default', reason='no_internal_evidence')
    if not defaults or not evidence or 'allFullResolutionLogits' not in evidence:
        return defaults, audit
    full = np.asarray(evidence['allFullResolutionLogits'])
    if full.shape != (9, *image.shape[:2]) or not np.isfinite(full).all():
        audit['reason'] = 'invalid_branch_evidence'; return defaults, audit
    threshold = evidence['maskThreshold']; head = evidence['defaultHead']
    default = np.asarray(defaults[0]) > .5
    if not np.array_equal(default, full[5+head] > threshold):
        audit['reason'] = 'default_parity_failed'; return defaults, audit
    hq = full[4] > threshold
    core = cv2.erode(hq.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not core.any():
        audit['reason'] = 'no_hq_core'; return defaults, audit
    retention = float(default[core].mean())
    cancelled = core & (full[head] < -full[4]) & ~default
    audit.update(defaultHead=int(head), defaultPixels=int(default.sum()),
                 hqCoreRetention=retention, cancellationFraction=float(cancelled[core].mean()),
                 fusion='SAM head logits + HQ logits; official unweighted sum',
                 scoreMeaning=evidence['scoreMeaning'])
    if retention >= .65 or audit['cancellationFraction'] < .30:
        audit['reason'] = 'default_supported_by_hq'; return defaults, audit
    candidates = [default] + [full[5+i] > threshold for i in range(1, 4) if i != head]
    heads = [head] + [i for i in range(1, 4) if i != head]
    ranking = score_candidates_v5(image, candidates, hint, dict(policy, verbose=False), image.shape)
    audit['ranking'] = ranking; audit['candidateHeads'] = heads
    base = next(row for row in ranking if row['index'] == 0)
    supported = default & core
    support_region = cv2.dilate((hq | default).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    checks = []; audit['checks'] = checks
    for row in ranking:
        i = row['index']
        if i == 0:
            continue
        candidate = candidates[i]; failures = []
        if row['issues'] or row['insideRatio'] < .80 or row['score'] < base['score'] + .25:
            failures.append('insufficient_image_support')
        if candidate[core].mean() < .90:
            failures.append('hq_core_incomplete')
        if not _retains(candidate, supported):
            failures.append('supported_default_part_lost')
        if int((candidate & ~support_region).sum()) > max(8, int(core.sum()*.005)):
            failures.append('unsupported_expansion')
        ownership = build_ownership_evidence(hint, candidates, policy, image.shape, reference=candidate)
        owner_review = None
        if ownership is not None:
            valid, owner_review = review_ownership_candidates([candidate], ownership)
            if not valid:
                # This is a proposal for the API's SAME-model ownership retry,
                # not an authorization to bypass negative-point verification.
                audit['ownershipPending'] = True
        checks.append(dict(head=heads[i], pixels=int(candidate.sum()),
                           hqCoreRetention=float(candidate[core].mean()), failures=failures))
        if not failures:
            audit.update(status='accepted', reason='default_hq_conflict_verified',
                         selectedHead=heads[i], selectedPixels=int(candidate.sum()))
            if owner_review is not None and not owner_review['status'] == 'accepted':
                audit.update(status='proposed', reason='hq_conflict_requires_ownership_retry',
                             ownershipReview=owner_review)
                # Retain rejected default as negative-anchor evidence for the
                # established V2 ownership retry; never union its pixels.
                return [candidate, default], audit
            return [candidate], audit
    audit['reason'] = 'no_safe_alternative'
    return defaults, audit
