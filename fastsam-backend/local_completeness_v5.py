"""Spatial witnesses for existing foreground, not inferred object semantics."""
import cv2
import numpy as np
from ownership_v5 import _interior_points


def local_witnesses(reference, evidence, limit=6):
    safe = np.asarray(reference, dtype=bool).copy()
    if evidence is not None:
        safe &= evidence['targetRegion'] & ~evidence['excluded']
    ys, xs = np.where(safe)
    if not len(xs):
        return []
    # Spatial cells stop a thick central body monopolising every positive.
    xcuts = np.linspace(xs.min(), xs.max()+1, 4).astype(int)
    ycuts = np.linspace(ys.min(), ys.max()+1, 4).astype(int)
    anchors = (evidence or {}).get('positive', [])
    regions = []
    for y1, y2 in zip(ycuts, ycuts[1:]):
        for x1, x2 in zip(xcuts, xcuts[1:]):
            region = np.zeros_like(safe); region[y1:y2, x1:x2] = safe[y1:y2, x1:x2]
            if region.sum() < 32 or any(region[y,x] for x,y in anchors):
                continue
            points = _interior_points(region, 1)
            if points:
                regions.append((region, points[0]))
    # Largest well-supported regions first, with a bounded decode prompt.
    regions.sort(key=lambda r: int(r[0].sum()), reverse=True)
    return regions[:limit]


def augment_ownership(evidence, reference):
    if evidence is None:
        return None
    witnesses = local_witnesses(reference, evidence)
    if not witnesses:
        return evidence
    result = dict(evidence)
    result['positive'] = evidence['positive'] + [p for _,p in witnesses]
    result['positiveGroups'] = evidence['positiveGroups'] + [[p] for _,p in witnesses]
    result['componentRegions'] = list(evidence.get('componentRegions', [])) + [r for r,_ in witnesses]
    result['points'] = [result['positive'] + result['negative']]
    result['labels'] = [[1]*len(result['positive']) + [0]*len(result['negative'])]
    return result
