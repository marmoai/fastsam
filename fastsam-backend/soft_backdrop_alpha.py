"""Recover soft foreground coverage on a verified uniform backdrop.

Only pixels explained by an observed foreground/background colour line use
fractional alpha. Nonmatching colours retain the existing conservative matte.
"""
import cv2
import numpy as np


def soft_colour_line(crop, base, background, support, fallback):
    values=crop.astype(np.float32)
    bg=np.asarray(background,np.float32).reshape(1,1,3)
    delta=values-bg
    contrast=np.linalg.norm(delta,axis=2)
    samples=contrast[base]
    if samples.size<32:
        return fallback, {'status':'skipped','reason':'insufficient_soft_foreground'}
    # Brightness alone is not a foreground criterion: separation from the
    # measured backdrop supplies the seed, independent of backdrop polarity.
    seeds=base & (contrast>=np.percentile(samples,85))
    if seeds.sum()<16:
        return fallback, {'status':'skipped','reason':'insufficient_colour_line_seeds'}
    _,nearest=cv2.distanceTransformWithLabels((~seeds).astype(np.uint8),cv2.DIST_L2,5,
                                            labelType=cv2.DIST_LABEL_PIXEL)
    foreground=values[seeds][np.clip(nearest-1,0,int(seeds.sum())-1)]
    direction=foreground-bg
    denominator=(direction*direction).sum(axis=2)
    coverage=np.clip((delta*direction).sum(axis=2)/np.maximum(denominator,1),0,1)
    residual=np.linalg.norm(delta-coverage[...,None]*direction,axis=2)
    reliable=support & (denominator>=40**2) & (residual<=12)
    if reliable.sum()<32:
        return fallback, {'status':'skipped','reason':'no_supported_colour_line'}
    alpha=fallback.copy()
    alpha[reliable]=coverage[reliable]
    alpha[~support]=0
    return alpha, {'status':'accepted','reason':'observed_soft_foreground_colour_line',
                   'seedPixels':int(seeds.sum()),'verifiedPixels':int(reliable.sum()),
                   'meanResidual':round(float(residual[reliable].mean()),3)}
