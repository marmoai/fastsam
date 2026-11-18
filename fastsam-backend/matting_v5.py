"""Evidence gates for backdrop matting; no object names or category routing."""
import cv2
import numpy as np


def backdrop_evidence(image, hint):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, hint)
    pad = max(24, round(max(x2-x1, y2-y1)*.15))
    x1, y1, x2, y2 = max(0,x1-pad), max(0,y1-pad), min(width,x2+pad), min(height,y2+pad)
    crop = image[y1:y2,x1:x2]
    if min(crop.shape[:2], default=0) < 8:
        return None
    border = np.zeros(crop.shape[:2], bool)
    r = max(2, round(min(crop.shape[:2])*.06))
    border[:r] = border[-r:] = True
    border[:,:r] = border[:,-r:] = True
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[border]
    # A minority table/contact surface must not set the background noise floor.
    color = np.median(samples,axis=0)
    distance = np.linalg.norm(lab-color,axis=2)
    inliers = distance[border] <= 12
    share = float(inliers.mean())
    if share < .70:
        return None
    noise = distance[border][inliers]
    return {'window':[x1,y1,x2,y2], 'distance':distance,
            'background':np.median(crop[border][inliers],axis=0).astype(np.float32),
            'low':max(8.,float(np.percentile(noise,98))+3), 'stableShare':share}


def polarity_prompt(image, hint, candidates):
    """Detect a background proposal using independent color AND geometry."""
    evidence = backdrop_evidence(image,hint)
    if evidence is None or not candidates:
        return None
    x1,y1,x2,y2 = evidence['window']
    high = evidence['distance'] > max(24,evidence['low']*2)
    yy,xx = np.indices(image.shape[:2])
    target = (xx>=hint[0]) & (xx<hint[2]) & (yy>=hint[1]) & (yy<hint[3])
    for raw in candidates:
        mask = np.asarray(raw)>.5
        frame = np.concatenate([mask[0],mask[-1],mask[:,0],mask[:,-1]])
        opposite = ~mask
        # Polarity changes require a compact, non-frame-touching alternative.
        if frame.mean()<.25 or opposite.sum()<max(64,mask.size*.01) or opposite.sum()>mask.size*.65:
            continue
        local=mask[y1:y2,x1:x2]; inv=~local
        if (high[local].mean()>.20 or high[inv].mean()<.65
                or (opposite & target).sum()/max(1,opposite.sum())<.90):
            continue
        from ownership_v5 import _interior_points
        foreground=np.zeros_like(mask); foreground[y1:y2,x1:x2]=inv & high
        background=np.zeros_like(mask)
        background[y1:y2,x1:x2] = local & (evidence['distance']<=evidence['low'])
        pos=_interior_points(foreground,4); neg=_interior_points(background,4)
        if pos and neg:
            return {'positive':pos,'negative':neg,'points':[pos+neg],
                    'labels':[[1]*len(pos)+[0]*len(neg)],'reference':opposite,
                    'reason':'uniform_backdrop_candidate_polarity'}
    return None


def backdrop_matte(image, mask, hint, layer_meta=None):
    """V1-style color coverage, enabled only by observed backdrop separation."""
    audit={'status':'skipped','reason':'no_supported_material_evidence'}
    evidence=backdrop_evidence(image,hint)
    if evidence is None:
        return None,None,audit
    semantics=(layer_meta or {}).get('entitySemantics') or {}
    attrs=semantics.get('attributes') or {}
    def attribute(key,values):
        item=attrs.get(key) or {}
        confidence=item.get('confidence')
        return (isinstance(confidence,(int,float)) and not isinstance(confidence,bool)
                and .8<=confidence<=1 and item.get('value') in values)
    contract=semantics.get('transparencyExtraction') or {}
    from transparency_contract_v5 import transmission_request
    transmission=transmission_request(contract)
    transparent=attribute('transparency',{'transparent','translucent','semi_transparent'})
    soft=attribute('boundary',{'soft','fine','hairlike'})
    x1,y1,x2,y2=evidence['window']; base=mask[y1:y2,x1:x2]
    distance=evidence['distance']; low=evidence['low']
    core=cv2.erode(base.astype(np.uint8),np.ones((3,3),np.uint8))>0
    values=distance[core if core.any() else base]
    if values.size<32:
        return None,None,audit
    backdrop_share=float((values<=low+6).mean())
    high=max(low+24,float(np.percentile(values,72)))
    contrast=float(np.percentile(values,95))
    # Explicit surface preservation vetoes transmission; uncertain contracts
    # do not veto an independently measured soft fringe.
    confidence=contract.get('confidence')
    preserve=(contract.get('mode')=='preserve_surface'
              and isinstance(confidence,(int,float)) and not isinstance(confidence,bool)
              and .8<=confidence<=1)
    mode=None
    if (not preserve and (transmission or transparent) and backdrop_share>=.25
            and contrast>low+30):
        mode='transmission'
    elif soft and contrast>low+30:
        # Fractional color coverage must actually be present, not just a label.
        fractional=float(((values>low+3)&(values<high*.85)).mean())
        if fractional>=.04:
            mode='soft_fringe'
    audit.update(backdropShare=round(backdrop_share,4),stableBackdropShare=round(evidence['stableShare'],4),
                 contrast=round(contrast,2))
    if mode is None:
        return None,None,audit
    normalized=np.clip((distance-low)/max(1,high-low),0,1)
    coverage=normalized*normalized*(3-2*normalized)
    radius=max(2,min(15,round(min(base.shape)*.0175)))
    support=cv2.dilate(base.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2))>0
    alpha=np.where(support,coverage,0).astype(np.float32)
    alpha=cv2.GaussianBlur(alpha,(3,3),0)
    if mode=='soft_fringe':
        # Soft contours do not imply a transmitting interior. Colour alone
        # cannot distinguish a pale highlight from the backdrop. Keep colour
        # coverage on thin strands and the contour transition, while ramping
        # back to opaque inside broad surfaces. Padding handles frame contact.
        depth=cv2.distanceTransform(np.pad(base.astype(np.uint8),1),
                                    cv2.DIST_L2,5)[1:-1,1:-1]
        interior_floor=np.clip((depth-1.)/max(1.,float(radius)),0.,1.)
        protected=base & (interior_floor>alpha)
        alpha[base]=np.maximum(alpha[base],interior_floor[base])
        audit.update(protectedInteriorPixels=int(protected.sum()),
                     softFringeRadius=int(radius))
        from soft_backdrop_alpha import soft_colour_line
        alpha, colour_review = soft_colour_line(image[y1:y2,x1:x2],base,
                                                evidence['background'],support,alpha)
        audit['softColourLine'] = colour_review
    # The estimator must retain a real foreground, not return an empty matte.
    if (alpha>.5).sum()<max(16,base.sum()*.03):
        audit['reason']='insufficient_foreground_signal'
        return None,None,audit
    full=np.zeros(mask.shape,np.float32); full[y1:y2,x1:x2]=alpha
    audit.update(status='accepted',reason='observed_backdrop_color_coverage',mode=mode,
                 fractionalPixels=int(((full>0)&(full<.98)).sum()),
                 clearedInteriorPixels=int((mask&(full<.05)).sum()))
    return full,evidence['background'],audit
