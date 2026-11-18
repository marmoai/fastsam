"""Compare prompt/branch alternatives against fixed incumbent ownership.

All masks remain native model outputs. No unions, logit weighting, or new
negative anchors derived from the alternative pool are permitted here.
"""
import os
import cv2
import numpy as np
from ownership_v5 import build_ownership_evidence, review_ownership_candidates
from local_completeness_v5 import local_witnesses
from fragment_ownership_v5 import uncertain_components
from segmentation_core_v5 import score_candidates_v5, select_candidate_v5


def enabled(name):
    return os.environ.get(name, '1').strip().lower() not in {'0', 'false', 'off'}


def branch_candidates(evidence, shape, source):
    if not evidence or not evidence.get('binaryParity'):
        return []
    full = np.asarray(evidence.get('allFullResolutionLogits', []))
    threshold = evidence.get('maskThreshold')
    if full.shape != (9, *shape[:2]) or threshold is None or not np.isfinite(full).all():
        return []
    rows = [(full[5+i] > threshold, dict(prompt=source, branch='fusion', head=i))
            for i in range(1, 4)]
    if enabled('SAMV5_HQ_ONLY_CANDIDATES'):
        rows.append((full[4] > threshold, dict(prompt=source, branch='hq_only', head=None)))
    return rows


def choose_candidate_pool(image, hint, incumbents, expanded, tight, policy,
                          verify_uncertain=None):
    if not enabled('SAMV5_CANDIDATE_POOL'):
        return incumbents,dict(status='disabled',selectedSource='incumbent',candidates=[])
    binary_incumbents = [np.asarray(mask) > .5 for mask in incumbents]
    incumbent, _ = select_candidate_v5(image, binary_incumbents, hint,
                                      dict(policy, verbose=False), image.shape)
    audit = dict(status='kept_incumbent', selectedSource='incumbent', candidates=[])
    if incumbent is None:
        audit['reason'] = 'no_incumbent'; return incumbents, audit
    masks = [incumbent]
    sources = [dict(prompt='expanded', branch='incumbent', head=None)]
    for mask, source in (branch_candidates(expanded, image.shape, 'expanded') +
                         branch_candidates(tight, image.shape, 'original')):
        duplicate = next((i for i,m in enumerate(masks) if np.array_equal(mask,m)), None)
        if duplicate is not None:
            audit['candidates'].append(dict(source=source, duplicateOf=duplicate))
            continue
        masks.append(mask); sources.append(source)
    ranking = score_candidates_v5(image, masks, hint, dict(policy, verbose=False), image.shape)
    rows = {r['index']:r for r in ranking}
    base = rows[0]
    # Freeze evidence before evaluating alternatives: another mask must not
    # move the target's positive anchors or manufacture neighbour negatives.
    ownership = build_ownership_evidence(hint, binary_incumbents, policy, image.shape, reference=incumbent)
    uncertain = uncertain_components(incumbent, hint, policy)
    retained_subject = incumbent & ~uncertain
    audit['uncertainDetachedPixels'] = int(uncertain.sum())
    audit['uncertainPolicy'] = 'not mandatory local anchors; no deletion authority'
    witnesses = local_witnesses(retained_subject, ownership, limit=9)
    core = cv2.erode(retained_subject.astype(np.uint8), np.ones((3,3),np.uint8)) > 0
    count,components,stats,_=cv2.connectedComponentsWithStats(retained_subject.astype(np.uint8),8)
    part_ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 8)
    part_ids = part_ids[part_ids != 0]
    best = None
    verified_uncertain = None
    for row in ranking:
        i = row['index']
        if i == 0: continue
        mask = masks[i]; source = sources[i]; failures=[]
        retention = float(mask[incumbent].mean())
        core_retention = float(mask[core].mean()) if core.any() else retention
        parts = [float(mask[region].mean()) for region,_ in witnesses]
        growth = int((mask & ~incumbent).sum())
        if row['issues'] or row['insideRatio'] < max(.80, base['insideRatio']-.02):
            failures.append('image_or_target_support')
        if retention < .97 or core_retention < .99:
            failures.append('incumbent_core_or_contour_lost')
        if any(not mask[y,x] for _,(x,y) in witnesses) or any(v<.95 for v in parts):
            failures.append('local_part_lost')
        kept = np.bincount(components[mask], minlength=count)
        if np.any(kept[part_ids] / stats[part_ids, cv2.CC_STAT_AREA] < .95):
            failures.append('disconnected_part_lost')
        if mask.sum() > incumbent.sum()*1.25 or growth > max(32, incumbent.sum()*.20):
            failures.append('excessive_new_extent')
        if row.get('backgroundExpansionPenalty',0) > base.get('backgroundExpansionPenalty',0)+.01:
            failures.append('background_expansion')
        owner_review=None
        if ownership is not None:
            valid,owner_review=review_ownership_candidates([mask],ownership)
            if not valid: failures.append('fixed_ownership_not_verified')
        # Improvement must be measured, never inferred from model score, head
        # number, prompt tightness, HQ label, or largest area.
        edge_gain = (row['alignment'] >= max(8.,base['alignment']*1.08)
                     and row['score'] >= base['score']-.02)
        supported_gain = False
        if expanded:
            full=np.asarray(expanded.get('allFullResolutionLogits',[]))
            if full.shape == (9,*image.shape[:2]):
                hq=full[4]>expanded['maskThreshold']
                added=mask & ~incumbent
                supported_gain = (growth>=max(12,incumbent.sum()*.005)
                    and float(hq[added].mean())>=.95
                    and row['alignment']>=base['alignment']*.98
                    and row['score']>=base['score']+.15)
        if not (edge_gain or supported_gain): failures.append('no_measured_improvement')
        lost_uncertain = uncertain & ~mask
        if not failures and lost_uncertain.any():
            # A neighbour rectangle only raises ambiguity. Dropping that
            # geometry still needs independent instance ownership evidence.
            if verified_uncertain is None:
                verified_uncertain = np.zeros(incumbent.shape, bool)
                if verify_uncertain is not None:
                    try:
                        verified_uncertain, review = verify_uncertain(incumbent)
                        verified_uncertain = np.asarray(verified_uncertain, bool)
                        audit['uncertainOwnershipReview'] = review
                    except Exception as error:
                        audit['uncertainOwnershipError'] = str(error)
            if (lost_uncertain & ~verified_uncertain).any():
                failures.append('uncertain_part_not_independently_verified')
        # HQ-only may retain an attractive outer boundary but hollow out a
        # local material. Above whole/core/cell/ownership gates apply equally.
        check=dict(source=source,pixels=int(mask.sum()),score=row['score'],
                   alignment=row['alignment'],retention=round(retention,5),
                   coreRetention=round(core_retention,5),localRetention=parts,
                   failures=failures,accepted=not failures)
        audit['candidates'].append(check)
        if not failures and best is None: best=i
    if best is None:
        audit['reason']='no_verified_improvement';return incumbents,audit
    audit.update(status='accepted',reason='fixed_core_parts_ownership_and_image_improvement',
                 selectedSource=sources[best],selectedPixels=int(masks[best].sum()))
    return [masks[best]],audit
