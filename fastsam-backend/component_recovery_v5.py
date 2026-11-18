"""Image-grounded proposals for missing declared thin components, never fills."""
import cv2
import numpy as np
from ownership_v5 import _box_mask


def merge_verified_completion(original, replacement, proposal):
    """Only import connected new model support touching the image proposal."""
    added = (replacement['alpha'] >= 128) & (original['alpha'] < 128)
    nearby = cv2.dilate(proposal['mask'].astype(np.uint8),np.ones((7,7),np.uint8)) > 0
    n, labels = cv2.connectedComponents(added.astype(np.uint8),8)
    region = proposal['mask'].copy()
    for owner in range(1,n):
        part = labels == owner
        if (part & nearby).any():
            region |= part
    return merge_component_alpha(original,replacement,{'region':region},preserve_threshold=128)


def merge_component_alpha(original, replacement, proposal, preserve_threshold=1):
    """Add a verified missing part without rewriting any existing alpha.

    This is completion, not a second matting pass. In particular a new local
    decode must not erode or harden the original contour inside the ROI.
    """
    region = proposal['region'].astype(np.uint8)
    region = cv2.dilate(region, np.ones((7, 7), np.uint8)) > 0
    weight = cv2.GaussianBlur(region.astype(np.float32), (7, 7), 1.0)
    weight[~region] = 0
    merged = np.rint(
        original['alpha'] * (1-weight) + replacement['alpha'] * weight
    ).astype(original['alpha'].dtype)
    replacement['alpha'] = np.where(original['alpha'] >= preserve_threshold,
                                    original['alpha'],np.maximum(original['alpha'],merged))
    replacement['mask'] = original['mask'] | (region & replacement['mask'])
    ys, xs = np.where(replacement['alpha'] > 0)
    if len(xs):
        replacement['bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    replacement['quality'].update(maskPixels=int(replacement['mask'].sum()),
                                  alphaPixels=int((replacement['alpha'] > 0).sum()))
    canvas = replacement['quality'].get('canvas')
    if isinstance(canvas, dict):
        canvas['alpha'] = replacement['bbox'][:]
        x, y, w, h = cv2.boundingRect(replacement['mask'].astype(np.uint8))
        canvas['mask'] = [x, y, x+w, y+h]
        hint = canvas.get('hint')
        if hint:
            canvas['deltaFromHint'] = dict(zip(
                ('left', 'top', 'right', 'bottom'),
                [value-prior for value, prior in zip(replacement['bbox'], hint)]))
    return replacement


def component_proposal(image, mask, policy):
    intent=policy.get('intent') or {}
    if (not intent.get('available') or intent.get('confidence',0)<.85
            or intent.get('bboxRole')!='tight_subject'):
        return None
    height,width=mask.shape
    proposals=[]
    for box in intent.get('included') or []:
        region=_box_mask(box,mask.shape,True)
        x,y,w,h=cv2.boundingRect(region.astype(np.uint8))
        if min(w,h)<6 or max(w,h)/min(w,h)<2.5:
            continue
        support=mask[y:y+h,x:x+w]
        if support.sum()<16:
            continue
        crop=image[y:y+h,x:x+w]
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(3,3),.7)
        size=max(7,min(41,round(min(w,h)*.8)))
        size+=1-size%2
        background=cv2.medianBlur(gray,size)
        contrast=np.abs(gray.astype(np.float32)-background)
        noise=contrast[~support]
        floor=max(14.,float(np.median(noise))+8) if len(noise) else 14.
        ridge=(contrast>floor).astype(np.uint8)
        lines=cv2.HoughLinesP(ridge,1,np.pi/180,threshold=max(12,round(max(w,h)*.10)),
                              minLineLength=max(16,round(max(w,h)*.22)),maxLineGap=3)
        if lines is None:
            continue
        for x1,y1,x2,y2 in lines[:,0]:
            dx,dy=int(x2-x1),int(y2-y1)
            if (w>h and abs(dx)<abs(dy)*3) or (h>w and abs(dy)<abs(dx)*3):
                continue
            corridor=np.zeros_like(ridge)
            cv2.line(corridor,(x1,y1),(x2,y2),1,max(5,round(min(w,h)*.35)))
            # The observed line also bounds where an independent local SAM
            # may bridge low-contrast gaps. This corridor only validates a
            # model proposal; it never becomes foreground by itself.
            envelope=np.zeros_like(ridge)
            if w>h:
                slope=dy/max(1,dx) if dx>=0 else dy/min(-1,dx)
                start=(0,round(y1-x1*slope));end=(w-1,round(y1+(w-1-x1)*slope))
            else:
                slope=dx/max(1,dy) if dy>=0 else dx/min(-1,dy)
                start=(round(x1-y1*slope),0);end=(round(x1+(h-1-y1)*slope),h-1)
            cv2.line(envelope,start,end,1,max(5,round(min(w,h)*.35)))
            structure=(ridge>0)&(corridor>0)
            # Require the proposed line to attach to observed subject support.
            near=cv2.dilate(support.astype(np.uint8),np.ones((5,5),np.uint8))>0
            if (structure & near).sum()<max(8,structure.sum()*.10):
                continue
            missing=structure & ~support
            n,labels,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),8)
            for owner in range(1,n):
                area=int(stats[owner,4])
                if area<max(24,round(support.sum()*.02)):
                    continue
                part=labels==owner
                depth=cv2.distanceTransform(part.astype(np.uint8),cv2.DIST_L2,5)
                py,px=np.unravel_index(int(depth.argmax()),depth.shape)
                if depth[py,px]<1.5:
                    continue
                full=np.zeros_like(mask);full[y:y+h,x:x+w]=part
                full_envelope=np.zeros_like(mask);full_envelope[y:y+h,x:x+w]=envelope>0
                proposals.append((area,[int(x+px),int(y+py)],full,[x,y,x+w,y+h],region,full_envelope))
    if not proposals:
        return None
    proposals.sort(key=lambda row:-row[0])
    points=[];missing=np.zeros_like(mask);regions=np.zeros_like(mask);envelope=np.zeros_like(mask);seen_boxes=set()
    for area,point,part,box,region,line_envelope in proposals:
        key=tuple(box)
        if key in seen_boxes or any(np.linalg.norm(np.subtract(point,p))<8 for p in points):
            continue
        seen_boxes.add(key)
        points.append(point);missing|=part;regions|=region;envelope|=line_envelope
        if len(points)>=3:
            break
    ys,xs=np.where(regions)
    return {'points':points,'mask':missing,'region':regions,'envelope':envelope,
            'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
            'reason':'declared_component_coherent_image_ridge'}


def review_component_candidate(candidate, baseline, proposal, positive, negative, evidence=None, audit=None):
    """Whole-target validation and a local growth budget, not area alone."""
    from ownership_v5 import review_ownership_candidates
    def reject(reason):
        if audit is not None:
            audit.update(accepted=False, reason=reason)
        return False
    candidate=np.asarray(candidate)>.5
    if evidence is not None:
        valid,_=review_ownership_candidates([candidate],evidence)
        if not valid:
            return reject('ownership_anchors_failed')
        candidate=valid[0]
    if (candidate&baseline).sum()/max(1,baseline.sum())<.97:
        return reject('baseline_retention_below_97_percent')
    if not all(candidate[y,x] for x,y in positive+proposal['points']):
        return reject('positive_anchor_missing')
    if any(candidate[y,x] for x,y in negative):
        return reject('negative_anchor_included')
    added=candidate&~baseline
    # Model confirmation may refine around the proposal, but cannot annex a
    # floor/neighbor merely because the requested component sits on it.
    allowed=cv2.dilate(proposal['mask'].astype(np.uint8),np.ones((9,9),np.uint8))>0
    allowed|=proposal.get('envelope',np.zeros_like(baseline))
    local_support=proposal.get('_localBoundarySupport')
    if local_support is not None:
        allowed|=local_support
    # Independently decoded contours may shift by an antialias rim; this is
    # separate from permission to grow a new component into its surroundings.
    allowed|=cv2.dilate(baseline.astype(np.uint8),np.ones((5,5),np.uint8))>0
    if audit is not None:
        audit.update(addedPixels=int(added.sum()),
                     outsideAllowedPixels=int((added&~allowed).sum()),
                     outsideLimit=float(max(16,baseline.sum()*.001)),
                     sparseBudget=int(max(64,proposal['mask'].sum()*3)))
    if (added&~allowed).sum()>max(16,baseline.sum()*.001):
        return reject('growth_outside_image_supported_region')
    sparse_budget=max(64,proposal['mask'].sum()*3)
    if added.sum()>sparse_budget:
        # Ridge pixels describe high-contrast evidence, not the full width of
        # a real part. Beyond that budget require almost all growth inside the
        # declared image-line envelope and attachment to existing foreground.
        envelope=proposal.get('envelope',np.zeros_like(baseline)) & proposal['region']
        if local_support is not None:
            envelope|=local_support
        near=cv2.dilate(baseline.astype(np.uint8),np.ones((5,5),np.uint8))>0
        grounded=envelope & ~baseline
        if (not grounded.any() or added.sum()>int(grounded.sum())+max(16,int(baseline.sum()*.001))
                or (added & ~(envelope|near)).sum()>max(16,baseline.sum()*.001)):
            return reject('extended_growth_outside_line_envelope')
        n,components,stats,_=cv2.connectedComponentsWithStats(added.astype(np.uint8),8)
        for owner in range(1,n):
            if stats[owner,cv2.CC_STAT_AREA]<8:
                continue
            part=components==owner
            extension=part if local_support is None else part & ~near
            low_support=(extension & envelope).sum()/max(1,extension.sum())<.90
            if local_support is not None:
                low_support=low_support and extension.sum()>max(16,baseline.sum()*.001)
            if not (part & near).any() or low_support:
                return reject('extended_growth_detached_or_low_envelope_support')
    if audit is not None:
        audit.update(accepted=True,reason='component_constraints_verified')
    return True


def review_local_component(candidate,baseline,proposal,positive,negative,evidence,
                           local_masks,crop,audit=None):
    """Retry a failed growth gate with bounded, multihead-supported edge tolerance."""
    first={}
    if review_component_candidate(candidate,baseline,proposal,positive,negative,evidence,audit=first):
        if audit is not None:audit.update(first)
        return True
    if audit is not None:audit.update(first)
    if first.get('reason') not in {'growth_outside_image_supported_region',
                                  'extended_growth_outside_line_envelope',
                                  'extended_growth_detached_or_low_envelope_support'}:
        return False
    if len(local_masks)<2 or 'envelope' not in proposal:return False
    x,y,X,Y=crop
    consensus=np.zeros_like(baseline)
    consensus[y:Y,x:X]=np.sum([np.asarray(m)>.5 for m in local_masks],axis=0)>=2
    # Same four-pixel image-space tolerance as the existing ridge review,
    # restricted to the declared component and exact fused-head agreement.
    margin=cv2.dilate(proposal['envelope'].astype(np.uint8),np.ones((9,9),np.uint8))>0
    support=margin & proposal['region'] & consensus
    retry=dict(proposal,_localBoundarySupport=support)
    second={}
    accepted=review_component_candidate(candidate,baseline,retry,positive,negative,evidence,audit=second)
    if audit is not None:
        audit.update(second,boundaryConsensusRetry=True,initialReason=first['reason'],
                     initialOutsideAllowedPixels=first.get('outsideAllowedPixels'),boundaryMargin=4)
    return accepted


def local_fusion_proposals(defaults, evidence):
    """Default first, then exact same-decode fused heads for local validation."""
    result=[(np.asarray(m)>.5,'default') for m in defaults]
    if not evidence or 'allFullResolutionLogits' not in evidence:
        return result
    full=np.asarray(evidence['allFullResolutionLogits'])
    if (not result or full.shape!=(9,*result[0][0].shape)
            or not np.isfinite(full).all() or evidence.get('binaryParity') is not True):
        return result
    for head in range(1,4):
        mask=full[5+head]>evidence['maskThreshold']
        if not any(np.array_equal(mask,old) for old,_ in result):
            result.append((mask,f'fused{head}'))
    return result


def local_component_inputs(image, baseline, proposal, evidence=None):
    """Crop context around the component; keep all coordinates explicit."""
    from ownership_v5 import _interior_points
    x1,y1,x2,y2=proposal['bbox']
    pad=max(24,round(min(x2-x1,y2-y1)*1.5))
    x,y,X,Y=max(0,x1-pad),max(0,y1-pad),min(image.shape[1],x2+pad),min(image.shape[0],y2+pad)
    positive=[p for p in ((evidence or {}).get('positive') or [])
              if x1<=p[0]<x2 and y1<=p[1]<y2]
    if not positive:
        positive=_interior_points(baseline & proposal['region'],2)
    positive=positive+proposal['points']
    safe=~cv2.dilate((baseline|proposal['region']).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    negative=_interior_points(safe[y:Y,x:X],4)
    points=[[px-x,py-y] for px,py in positive]+negative
    return {'image':image[y:Y,x:X].copy(),'crop':[x,y,X,Y],
            'bbox':[x1-x,y1-y,x2-x,y2-y],
            'points':[points],'labels':[[1]*len(positive)+[0]*len(negative)],
            'positive':positive,'negative':[[px+x,py+y] for px,py in negative]}
