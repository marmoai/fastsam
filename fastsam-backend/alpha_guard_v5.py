"""Constrain opaque HQ boundary alpha using observed local color mixtures."""
import cv2
import numpy as np
import os


def constrain_alpha(image, selected, protected, exterior, mask, alpha, blocked,
                    material, edge, material_alpha=None, perforated=False):
    active=os.environ.get('SAMV5_ALPHA_BOUNDARY_GUARD','1').strip().lower() not in {'0','false','off'}
    review=dict(enabled=active,status='skipped',
                reason='no_supported_opaque_boundary',raisedPixels=0,clearedHaloPixels=0)
    if (not review['enabled'] or material.get('mode')!='opaque' or
            edge.get('class')!='hard' or material_alpha is not None or perforated):
        return alpha,review
    # Two observed, locally populated color populations are required. HQ
    # logits are not alpha and do not justify a constant opaque floor.
    foreground=(cv2.erode((selected & mask & ~blocked).astype(np.uint8),
                         np.ones((5,5),np.uint8))>0) & (alpha>=.98)
    background=exterior & ~mask
    values=image.astype(np.float32)
    def population(region):
        count=cv2.boxFilter(region.astype(np.float32),-1,(9,9),normalize=False)
        total=cv2.boxFilter(values*region[...,None],-1,(9,9),normalize=False)
        return count,total/np.maximum(count[...,None],1.)
    nf,fg=population(foreground);nb,bg=population(background)
    direction=fg-bg;delta=values-bg
    contrast=np.sum(direction*direction,axis=2)
    coverage=np.clip(np.sum(delta*direction,axis=2)/np.maximum(contrast,1.),0.,1.)
    error=np.linalg.norm(delta-coverage[...,None]*direction,axis=2)
    reliable=(nf>=3)&(nb>=3)&(contrast>=30.**2)&(error<=6.)
    target=np.clip(coverage-.03,0.,.98)
    raise_pixels=(reliable & protected & selected & mask & ~blocked &
                  (target>=alpha+.10))
    near=cv2.dilate(protected.astype(np.uint8),np.ones((5,5),np.uint8))>0
    clear=(reliable & near & exterior & ~mask & ~blocked &
           (coverage<=.02) & (alpha>0))
    result=alpha.copy()
    result[raise_pixels]=target[raise_pixels]
    result[clear]=0.
    review.update(status='applied' if raise_pixels.any() or clear.any() else 'unchanged',
        reason='local_color_mixture_verified',raisedPixels=int(raise_pixels.sum()),
        clearedHaloPixels=int(clear.sum()),
        maxIncrease=float(np.max(result-alpha,initial=0)),
        maxDecrease=float(np.max(alpha-result,initial=0)),
        scope='image-supported original opaque boundary; no binary mask edits')
    return result,review
