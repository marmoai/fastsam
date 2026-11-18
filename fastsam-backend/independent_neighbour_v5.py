"""Independent neighbor prompts; rectangles never authorize a cut."""
import cv2
import numpy as np
from ownership_v5 import _box_mask, _interior_points
from fragment_ownership_v5 import review_neighbour_residue

def verify_neighbours(image,hint,reference,policy,decode,decode_validated=None):
    intent=policy.get('intent') or {}
    empty=np.zeros(reference.shape,bool)
    audit=[]
    def ask(box, points, labels, validator, purpose):
        if decode_validated is not None:
            return decode_validated(box, points, labels, validator, purpose)
        return decode(box, points, labels)
    if (not intent.get('available') or intent.get('confidence',0)<.85
            or intent.get('bboxRole')!='tight_subject'):
        return empty,audit
    neighbours=[_box_mask(b,reference.shape,True) for b in intent.get('excluded',[])]
    if not neighbours:return empty,audit
    excluded=np.logical_or.reduce(neighbours)
    target=_box_mask(hint,reference.shape)
    for b in intent.get('included',[]):target|=_box_mask(b,reference.shape,True)
    safe=reference & target & ~excluded
    positive=_interior_points(safe,4)
    if len(positive)<2:return empty,audit
    confirmed=empty.copy()
    for region in neighbours[:3]:
        row=dict(status='unresolved',removedPixels=0);audit.append(row)
        if (reference & region).sum()<24:row['reason']='no_overlap';continue
        x,y,w,h=cv2.boundingRect(region.astype(np.uint8))
        if w>h*2 and h<(hint[3]-hint[1])*.5:
            row['reason']='broad_shallow_support_surface';continue
        # A separate box query can observe neighbor pixels that the target
        # proposal never contained. Foreground points are negative here.
        first=ask([x,y,x+w,y+h],[positive],[[0]*len(positive)],
                  lambda m:not any(m[py,px] for px,py in positive), 'neighbour_discovery')
        valid=[np.asarray(m)>.5 for m in first if not any(m[py,px] for px,py in positive)]
        if not valid:row['reason']='neighbor_query_unresolved';continue
        support=np.logical_or.reduce(valid) & region & ~target
        seeds=_interior_points(cv2.erode(support.astype(np.uint8),np.ones((3,3),np.uint8))>0,3)
        if not seeds:row['reason']='no_independent_exclusive_seed';continue
        second=ask([x,y,x+w,y+h],[seeds+positive],[[1]*len(seeds)+[0]*len(positive)],
                   lambda m:all(m[py,px] for px,py in seeds) and not any(m[py,px] for px,py in positive),
                   'neighbour_confirmation')
        valid2=[np.asarray(m)>.5 for m in second
                if all(m[py,px] for px,py in seeds) and not any(m[py,px] for px,py in positive)]
        if not valid2:row['reason']='neighbor_confirmation_failed';continue
        agree=np.logical_or.reduce(valid) & np.logical_or.reduce(valid2) & region
        negative=_interior_points(cv2.erode(agree.astype(np.uint8),np.ones((3,3),np.uint8))>0,3)
        if not negative:row['reason']='no_neighbor_interior';continue
        guided=ask(hint,[positive+negative],[[1]*len(positive)+[0]*len(negative)],
                   lambda m: (m & safe).sum()/max(1,safe.sum())>=.98,
                   'neighbour_target_rejection')
        good=[]
        for m in guided:
            m=np.asarray(m)>.5
            if (all(m[py,px] for px,py in positive)
                    and not any(m[py,px] for px,py in negative)
                    and (m & safe).sum()/max(1,safe.sum())>=.98):good.append(m)
        if not good:row['reason']='target_confirmation_failed';continue
        # Only pixels the neighbor claims AND every valid target rejects.
        cut=reference & agree & ~np.logical_or.reduce(good)
        residue,residue_review=review_neighbour_residue(
            reference,cut,agree,safe,policy,positive)
        cut |= residue
        if cut.sum()>reference.sum()*.20:row['reason']='excessive_overlap';continue
        confirmed|=cut
        row.update(status='confirmed',reason='two_neighbor_prompts_and_target_rejection',
                   removedPixels=int(cut.sum()),positive=positive,negative=negative,seeds=seeds,
                   residuePixels=int(residue.sum()),residueReview=residue_review)
    return confirmed,audit
