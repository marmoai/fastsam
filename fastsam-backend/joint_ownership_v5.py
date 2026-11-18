"""Reconcile visible ownership using validated sibling pixels, never box cuts."""
import cv2
import numpy as np
from ownership_v5 import _box_mask, _interior_points, review_ownership_candidates


def _trusted_intent(record):
    intent=record['policy'].get('intent') or {}
    return (intent.get('available') and intent.get('confidence',0)>=.85
            and intent.get('bboxRole')=='tight_subject')


def _sync_geometry(result):
    ys, xs = np.where(result['alpha'] > 0)
    if not len(xs):
        return
    result['bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    x, y, w, h = cv2.boundingRect(result['mask'].astype(np.uint8))
    quality = result['quality']
    quality.update(maskPixels=int(result['mask'].sum()), alphaPixels=int((result['alpha']>0).sum()))
    canvas = quality.get('canvas')
    if canvas is not None:
        canvas.update(alpha=result['bbox'][:], mask=[x,y,x+w,y+h])
        hint = canvas.get('hint')
        if hint:
            canvas['deltaFromHint'] = dict(zip(('left','top','right','bottom'),
                [v-h for v,h in zip(result['bbox'],hint)]))


def _hold(result):
    quality = result['quality']
    issue = 'joint_ownership_unresolved'
    quality.update(status='low_quality',runtimeAction='hold',shouldGenerateRuntimeLayer=False,
                   needsHigherPrecision=True,issues=list(dict.fromkeys([*(quality.get('issues') or []),issue])))


def _anchored_support(record):
    result = record['result']
    if (not _trusted_intent(record) or result['quality'].get('runtimeAction') != 'accept'
            or result.get('policy',{}).get('material') != 'opaque'):
        return None
    support = result['mask'].copy()
    reference = record.get('reference')
    verified_extension=any(result['quality'].get(key,{}).get('status')=='accepted'
                           for key in ('completionReview','componentRecovery'))
    if reference is not None and not verified_extension:
        support &= np.asarray(reference,dtype=bool)
    evidence = record.get('evidence')
    points = evidence['positive'] if evidence is not None else _interior_points(
        support & _box_mask(record['hint'],support.shape),4)
    if not points or not all(support[y,x] for x,y in points):
        return None
    _, labels = cv2.connectedComponents(support.astype(np.uint8),connectivity=8)
    owners = {int(labels[y,x]) for x,y in points}-{0}
    return support & np.isin(labels,list(owners))


def _box_iou(first, second):
    a,b,c,d=first; x,y,X,Y=second
    intersection=max(0,min(c,X)-max(a,x))*max(0,min(d,Y)-max(b,y))
    return intersection/max(1,(c-a)*(d-b)+(X-x)*(Y-y)-intersection)


def reconcile_layers(records):
    """Order-independent review of explicit exclusions and atomic siblings.

    Broadly nested platforms cannot claim the small object on top: a whole
    neighbour claim must fit the declared exclusion and have its own anchors.
    Sibling evidence can also remove detached, unanchored fragments locally.
    """
    if len(records)<2:
        return
    supports=[_anchored_support(r) for r in records]
    source_alphas={id(r):r['result']['alpha'].copy() for r in records}
    updates=[]
    for index, target in enumerate(records):
        result=target['result']; mask=result['mask']; evidence=target.get('evidence')
        if (not _trusted_intent(target) or result['quality'].get('runtimeAction')!='accept'
                or result.get('policy',{}).get('material')!='opaque'):
            continue
        positives=evidence['positive'] if evidence is not None else []
        _, components=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
        anchored={int(components[y,x]) for x,y in positives}-{0}
        if not positives:
            anchored={int(components[y,x]) for x,y in _interior_points(mask,4)}-{0}
        detached=mask & ~np.isin(components,list(anchored))
        exclusions=target['policy'].get('intent',{}).get('excluded') or []
        claims=[]
        for j,source in enumerate(records):
            support=supports[j]
            if j==index or support is None or not (support & mask).any():
                continue
            explicit=False
            for box in exclusions:
                region=_box_mask(box,mask.shape,True)
                x,y,w,h=cv2.boundingRect(region.astype(np.uint8))
                if (_box_iou([x,y,x+w,y+h],source['hint'])>=.5
                        and (support & region).sum()/max(1,support.sum())>=.8):
                    explicit=True
                    break
            parent=target['meta'].get('parentLayerId')
            siblings=(parent and parent==source['meta'].get('parentLayerId')
                      and target['meta'].get('compositeRole')=='atomic_object'
                      and source['meta'].get('compositeRole')=='atomic_object')
            if not explicit and not siblings:
                continue
            claim=support if explicit else support & detached
            overlap=claim & mask
            if not overlap.any():
                continue
            # A target anchor is independent contrary evidence; do not decide
            # by area or processing order when both objects claim that pixel.
            if any(overlap[y,x] for x,y in positives):
                claims.append((source,None,'anchor_conflict'))
            else:
                claims.append((source,claim,'explicit_neighbour' if explicit else 'detached_sibling'))
        if claims:
            updates.append((target,claims))
    conflicted=set()
    for target,claims in updates:
        union=np.zeros_like(target['result']['mask'])
        for _,claim,_ in claims:
            if claim is not None:
                union |= claim
        old=target['result']['mask']
        if (any(claim is None for _,claim,_ in claims)
                or (union & old).sum()>max(16,old.sum()*.25)
                or not (old & ~union).any()):
            conflicted.add(id(target))
    # Propagate uncertainty before editing anything, so a layer that will be
    # held cannot become deletion authority merely by being processed first.
    changed=True
    while changed:
        changed=False
        for target,claims in updates:
            if id(target) not in conflicted and any(id(source) in conflicted for source,_,_ in claims):
                conflicted.add(id(target));changed=True
    for target,claims in updates:
        result=target['result']; old=result['mask'].copy(); removal=np.zeros_like(old)
        opacity=np.zeros_like(result['alpha'],dtype=np.float32);reviews=[];unresolved=False
        for source,claim,reason in claims:
            if id(source) in conflicted:
                claim=None;reason='source_ownership_conflict'
            if claim is None:
                unresolved=True;reviews.append({'layerId':source['id'],'status':'unresolved','reason':reason});continue
            # Use the frozen source matte only in the independently supported
            # silhouette. Rectangles never become deletion masks.
            removal |= claim
            opacity=np.maximum(opacity,np.where(claim,source_alphas[id(source)]/255.,0))
            reviews.append({'layerId':source['id'],'status':'verified','reason':reason,
                            'pixels':int((claim & old).sum())})
        cuts=removal & old
        # Preserve original completeness outside the disputed neighbour pixels.
        evidence=target.get('evidence')
        protected=(evidence['safeCore'] & old & ~removal) if evidence is not None else old & ~removal
        if cuts.sum()>max(16,old.sum()*.25) or not (old & ~removal).any():
            unresolved=True
        if unresolved:
            _hold(result)
            result['quality']['jointOwnership']={'status':'unresolved','reviews':reviews}
            continue
        candidate=result['mask'] & ~(removal & (opacity>=.5))
        final_review=None
        if evidence is not None:
            # Independently verified foreign pixels are disputed, not a
            # mandatory part of the target's original completeness baseline.
            adjusted=dict(evidence,safeCore=evidence['safeCore'] & ~removal,
                          componentRegions=[region & ~removal for region in evidence.get('componentRegions',[])])
            valid,final_review=review_ownership_candidates([candidate],adjusted)
            if not valid:
                _hold(result)
                result['quality']['jointOwnership']={'status':'unresolved','reason':'target_retention_failed','reviews':reviews}
                continue
        result['mask'] = candidate
        result['alpha']=np.rint(result['alpha']*(1-opacity)).astype(np.uint8)
        if (result['mask'] & protected).sum()<protected.sum():
            raise AssertionError('joint ownership modified undisputed target support')
        _sync_geometry(result)
        result['quality']['jointOwnership']={'status':'accepted','reviews':reviews,
                                             'removedPixels':int(cuts.sum()),
                                             'undisputedRetention':1.0}
        result['quality'].setdefault('validation',{})['jointOwnership']='verified_pixels'
        if final_review is not None:
            final_review['validationDomain']='target_support_excluding_verified_neighbours'
            result['quality']['ownershipReview']=dict(result['quality'].get('ownershipReview') or {},final=final_review)
