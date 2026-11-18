"""Source-aware edit accounting and bounded native-contour fallback for V5."""
import os
import cv2
import numpy as np


class ModificationLedger:
    """Exact sparse deltas allow inspection/undo without a full image per stage."""
    def __init__(self, baseline):
        self.baseline = np.asarray(baseline, bool).copy()
        self.previous = self.baseline.copy()
        self.alpha = None
        self.entries = []

    def record(self, source, mask, authority=None, background=None, alpha=None):
        mask = np.asarray(mask, bool)
        removed = np.flatnonzero(self.previous & ~mask).astype(np.uint32)
        added = np.flatnonzero(mask & ~self.previous).astype(np.uint32)
        entry = dict(source=source, removed=removed, added=added)
        for name, support in [('verifiedOwnership',authority),('connectedBackground',background)]:
            entry[name] = (np.flatnonzero(np.asarray(support,bool) & (self.previous ^ mask)).astype(np.uint32)
                           if support is not None else np.empty(0,np.uint32))
        if alpha is not None:
            values=np.asarray(alpha,dtype=np.float32)
            if values.max(initial=0)<=1: values=values*255
            values=np.clip(np.rint(values),0,255).astype(np.uint8)
            changed=(np.flatnonzero(values!=self.alpha).astype(np.uint32)
                     if self.alpha is not None else np.empty(0,np.uint32))
            entry['alphaIndices']=changed
            entry['alphaBefore']=self.alpha.ravel()[changed].copy() if self.alpha is not None else np.empty(0,np.uint8)
            entry['alphaAfter']=values.ravel()[changed].copy()
            self.alpha=values.copy()
        self.entries.append(entry);self.previous=mask.copy()

    def undo_geometry(self, mask, entry_index):
        result=np.asarray(mask,bool).copy();entry=self.entries[entry_index]
        result.ravel()[entry['added']]=False;result.ravel()[entry['removed']]=True
        return result

    def summary(self):
        return dict(version='v5-source-ledger-1', stages=[dict(
            source=e['source'],removedPixels=len(e['removed']),addedPixels=len(e['added']),
            verifiedOwnershipPixels=len(e['verifiedOwnership']),connectedBackgroundPixels=len(e['connectedBackground']),
            changedAlphaPixels=len(e.get('alphaIndices',[]))) for e in self.entries],
            scope='one core transaction; adopted transport merges recorded separately')


def boundary_metrics(reference, current):
    def rim(mask):
        return mask & ~(cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),
                                  borderType=cv2.BORDER_CONSTANT,borderValue=0)>0)
    first,second=rim(reference),rim(current)
    if not first.any() or not second.any():
        return dict(chamferPx=None,boundaryIoU=0.,p95DistancePx=None)
    a=cv2.distanceTransform((~first).astype(np.uint8),cv2.DIST_L2,5)
    b=cv2.distanceTransform((~second).astype(np.uint8),cv2.DIST_L2,5)
    distances=np.concatenate([a[second],b[first]])
    return dict(chamferPx=float((a[second].mean()+b[first].mean())/2),
                boundaryIoU=float(((a<=2)&(b<=2)).sum()/max(1,((a<=2)|(b<=2)).sum())),
                p95DistancePx=float(np.percentile(distances,95)))


def constrain_global_boundary(image, selected, mask, alpha, protected, authority,
                              background, allowed_additions, material, edge,
                              material_alpha, perforated):
    active=os.environ.get('SAMV5_GLOBAL_BOUNDARY_GUARD','1').lower() not in {'0','false','off'}
    review=dict(enabled=active,status='not_applicable',reference='accepted_native_candidate',
                restoredPixels=0,trimmedPixels=0)
    if material.get('mode')!='opaque' or edge.get('class')!='hard' or material_alpha is not None or perforated:
        return mask,alpha,np.zeros_like(mask),review
    # Authorized geometry stays in the reference, so a correct neighbour cut
    # or independently verified part can never trigger rollback to a bad default.
    permission=np.asarray(authority,bool)|np.asarray(background,bool)
    allowed=np.asarray(allowed_additions,bool)
    # Only accepted cuts alter the reference. A neighbour query can include
    # pixels never actually deleted and must not demand their removal here.
    expected=(selected & ~(permission & ~mask))|allowed
    current=np.asarray(alpha)>=.5
    metrics=boundary_metrics(expected,current)
    review.update(before=metrics,status='within_budget',
                  thresholds=dict(chamferPx=1.5,boundaryIoU=.75,p95DistancePx=3.))
    breached=(metrics['chamferPx'] is None or metrics['chamferPx']>1.5
              or (metrics['boundaryIoU']<.75 and metrics['p95DistancePx']>3))
    if not breached:return mask,alpha,np.zeros_like(mask),review
    # Restore only image-supported native boundary. Never fill arbitrary
    # interiors or resurrect a verified neighbour/background deletion.
    depth=cv2.distanceTransform(selected.astype(np.uint8),cv2.DIST_L2,5)
    near=cv2.dilate(protected.astype(np.uint8),np.ones((7,7),np.uint8))>0
    restore=selected & near & (depth<=6) & ~permission & (~mask | (alpha<.5))
    # A strong image edge can also outline an unrelated native fragment.
    # Rollback may repair an attached contour, never resurrect a component
    # that cleanup removed in full merely to improve a global distance score.
    _, components=cv2.connectedComponents((mask | restore).astype(np.uint8),8)
    retained=np.unique(components[mask]);retained=retained[retained!=0]
    restore &= np.isin(components,retained)
    # Unverified expansion beyond the native 3px envelope is reverted locally.
    distance=cv2.distanceTransform((~selected).astype(np.uint8),cv2.DIST_L2,5)
    trim=(~selected)&(~allowed)&(distance>3)&(alpha>0)
    revised=mask.copy();values=np.asarray(alpha).copy()
    if active:
        revised[restore]=True;values[restore]=1.
        revised[trim]=False;values[trim]=0.
    review.update(status='local_fallback' if active and (restore.any() or trim.any()) else 'review_required',
                  restoredPixels=int(restore.sum()) if active else 0,
                  trimmedPixels=int(trim.sum()) if active else 0,
                  after=boundary_metrics(expected,values>=.5))
    after=review['after']
    review['unresolved']=bool(after['chamferPx'] is None or after['chamferPx']>1.5
                            or (after['boundaryIoU']<.75 and after['p95DistancePx']>3))
    return revised,values,restore if active else np.zeros_like(mask),review
