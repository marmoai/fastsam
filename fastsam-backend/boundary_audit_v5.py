"""V5 geometry accounting and conservative, local cleanup review.

Distances are image pixels. Candidate agreement is correlated evidence, never
independent ownership. Raster diagnostics are opt-in via SAM_DEBUG_DUMP.
"""
import os
import cv2
import numpy as np
from sam_diagnostics import sam_diagnostic_event
from boundary_control_v5 import ModificationLedger


def diagnostic_color_image(stage, image, **metadata):
    """Save BGR/BGRA without passing color arrays to the shared mask writer."""
    from sam_diagnostics import _trace
    state = _trace.get()
    if state is None:
        return
    try:
        filename = f"{len(state['events']):03d}-{stage}.png"
        ok, png = cv2.imencode('.png', image)
        if not ok:
            raise OSError('Could not encode color diagnostic')
        png.tofile(str(state['directory'] / filename))
        sam_diagnostic_event(stage, artifact=filename, **metadata)
    except Exception as error:
        print(f'SAMV5 color diagnostics unavailable: {error}')


def rim(mask):
    return mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)


def distance_to(points):
    return cv2.distanceTransform((~points).astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)


class BoundaryAudit:
    def __init__(self, image, selected, reference=None, candidates=()):
        self.selected = np.asarray(selected, bool).copy()
        self.reference = self.selected if reference is None else np.asarray(reference, bool)
        self.previous = self.selected.copy()
        self.reference_rim = rim(self.reference)
        self.reference_distance = distance_to(self.reference_rim)
        self.reference_band = self.reference_distance <= 3
        self.rows = []
        self.provenance = {}
        self.last_alpha = None
        self.ledger = ModificationLedger(self.selected)
        self.mode = os.environ.get('SAMV5_HQ_BOUNDARY_GUARD', '1').lower() not in {'0', 'false', 'off'}
        # Signed mask normals and luminance/color gradients must be parallel.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
        strength = gx * gx + gy * gy
        channel = strength.argmax(axis=2)[..., None]
        ix = np.take_along_axis(gx, channel, axis=2)[..., 0]
        iy = np.take_along_axis(gy, channel, axis=2)[..., 0]
        soft = cv2.GaussianBlur(self.selected.astype(np.float32), (5, 5), 1.)
        nx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
        ny = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)
        self.strength = np.sqrt(ix * ix + iy * iy)
        alignment = np.abs(ix * nx + iy * ny) / np.maximum(1e-6, self.strength * np.hypot(nx, ny))
        stable = self.selected.copy()
        if len(candidates) > 1:
            stable &= np.mean(candidates, axis=0) >= .8
        padded = np.pad((~self.selected).astype(np.uint8), 1, constant_values=1)
        _, free = cv2.connectedComponents(padded, connectivity=4)
        exterior = (free == free[0, 0])[1:-1, 1:-1]
        self.exterior = exterior
        outer_band = cv2.dilate(exterior.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        self.protected = (self.selected & outer_band &
                          (cv2.distanceTransform(self.selected.astype(np.uint8), cv2.DIST_L2, 5) <= 3) &
                          stable & (alignment >= .85) & (self.strength >= 32))

    def record(self, stage, mask, authority=None, background=None, alpha=None):
        mask = np.asarray(mask, bool)
        self.ledger.record(stage, mask, authority, background, alpha)
        deleted = self.previous & ~mask
        added = mask & ~self.previous
        current_rim = rim(mask)
        current_distance = distance_to(current_rim)
        a = self.reference_distance[current_rim]
        b = current_distance[self.reference_rim]
        row = dict(stage=stage, deletedPixels=int(deleted.sum()), addedPixels=int(added.sum()),
                   deletedInHQBoundaryBand=int((deleted & self.reference_band).sum()),
                   deletedProtectedPixels=int((deleted & self.protected).sum()),
                   deletedWithAuthority=int((deleted & authority).sum()) if authority is not None else 0,
                   deletedWithConnectedBackground=int((deleted & background).sum()) if background is not None else 0,
                   inwardPixels=int((self.reference & ~mask).sum()),
                   outwardPixels=int((mask & ~self.reference).sum()),
                   chamferPx=round(float((a.mean() + b.mean()) / 2), 4) if a.size and b.size else None,
                   boundaryIoU=round(float(((self.reference_distance <= 2) & (current_distance <= 2)).sum() /
                                           max(1, ((self.reference_distance <= 2) | (current_distance <= 2)).sum())), 5),
                   inwardBoundaryMeanPx=round(float(a[(self.reference & ~self.reference_rim)[current_rim]].mean()), 4)
                   if (self.reference & ~self.reference_rim)[current_rim].any() else 0.,
                   outwardBoundaryMeanPx=round(float(a[(mask & ~self.reference)[current_rim]].mean()), 4)
                   if (mask & ~self.reference)[current_rim].any() else 0.)
        if alpha is not None:
            values = np.asarray(alpha, np.float32)
            if values.max(initial=0) > 1: values = values / 255.
            row.update(alphaBelowHalfOnProtected=int((self.protected & (values < .5)).sum()),
                       alphaMassOnProtected=round(float(values[self.protected].sum()), 3))
            if self.last_alpha is not None:
                row.update(alphaDecreasedPixels=int((values < self.last_alpha - 1/255).sum()),
                           alphaIncreasedPixels=int((values > self.last_alpha + 1/255).sum()))
            self.last_alpha = values.copy()
        self.rows.append(row)
        self.previous = mask.copy()
        sam_diagnostic_event('boundary_audit', masks=[mask], boundary=row)

    def cleanup(self, before, proposed, lab, authority, models, stage):
        """Veto unsupported color cuts on the image-supported selected contour.

        Strongly background-like pixels reachable from original exterior and
        independent ownership always veto restoration. No new pixels outside
        the incoming/selected mask can be created.
        """
        removed = before & ~proposed
        bg = np.zeros_like(before)
        if not removed.any():
            self.provenance[stage] = dict(proposedPixels=0, restoredPixels=0)
            return proposed, bg
        if 'bgDistFull' in models and 'fgDistFull' in models:
            passable = ((models['bgDistFull'] < 10) &
                        (models['bgDistFull'] + 8 < models['fgDistFull']) &
                        (self.strength < 24))
            count, owners = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
            touched = np.unique(owners[passable & self.exterior])
            touched = touched[touched != 0]
            if len(touched): bg = np.isin(owners, touched)
        eligible = removed & self.protected & ~authority & ~bg
        # Color alone cannot overrule the image-supported HQ contour. Apply
        # this locally before accepted deletions enter blocked. Independent
        # ownership and connected background retain precedence.
        protected_veto = eligible if stage == 'color_exclusion' else np.zeros_like(before)
        restored = protected_veto.copy()
        selected_band = self.selected & ~(cv2.erode(self.selected.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        count, owners, stats, _ = cv2.connectedComponentsWithStats(removed.astype(np.uint8), 8)
        protected_distance = distance_to(self.protected) if self.protected.any() else None
        selected_depth = cv2.distanceTransform(self.selected.astype(np.uint8), cv2.DIST_L2, 5)
        for owner in range(1, count):
            part = owners == owner
            # Keep a whole unsupported shallow strip, rather than retaining
            # its strong outer rim but cutting holes immediately behind it.
            # Evidence must cover half the strip and every pixel must remain
            # within 3px of that supported contour and 6px of the selected edge.
            if (stage == 'color_exclusion' and protected_distance is not None
                    and not (part & (authority | bg)).any()
                    and float(eligible[part].mean()) >= .5
                    and np.all(protected_distance[part] <= 3)
                    and np.all(selected_depth[part] <= 6)):
                protected_veto |= part
                restored |= part
            # Do not resurrect a rim around a genuinely deleted interior or
            # an authoritative cut. Only an entire shallow edge edit qualifies.
            if (part & (authority | bg)).any() or (part & ~selected_band).any(): continue
            if float(eligible[part].mean()) < .9: continue
            local = cv2.dilate(part.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            kept = local & proposed
            outside = local & ~self.selected
            cut = rim(proposed) & local
            if kept.sum() < 4 or outside.sum() < 4 or cut.sum() < 3: continue
            center = np.median(lab[part], axis=0)
            fg = float(np.linalg.norm(center - np.median(lab[kept], axis=0)))
            bd = float(np.linalg.norm(center - np.median(lab[outside], axis=0)))
            if (fg <= 8 and fg + 6 < bd and np.median(self.strength[cut]) < 24 and
                    np.median(self.strength[part]) > max(32, 1.5 * np.median(self.strength[cut]))):
                restored |= part
        self.provenance[stage] = dict(proposedPixels=int(removed.sum()),
                                      authoritativePixels=int((removed & authority).sum()),
                                      connectedBackgroundPixels=int((removed & bg).sum()),
                                      protectedConflictPixels=int((removed & self.protected & ~authority & ~bg).sum()),
                                      protectedVetoPixels=int(protected_veto.sum()),
                                      restorablePixels=int(restored.sum()),
                                      restoredPixels=int(restored.sum()) if self.mode else 0)
        sam_diagnostic_event('boundary_cleanup_evidence', masks=[removed, authority & removed, bg & removed, restored],
                             sourceStage=stage, review=self.provenance[stage])
        return proposed | restored if self.mode else proposed, bg

    def summary(self):
        return dict(version='v5-hq-boundary-audit-1', guardEnabled=self.mode,
                    reference='initial_default_fusion' if not np.array_equal(self.reference, self.selected)
                    else 'selected_fusion', protectedPixels=int(self.protected.sum()),
                    boundaryBandPx=3, distanceUnits='source_image_pixels',
                    stages=self.rows, deletionReview=self.provenance,
                    reviewWarnings=(['protected_boundary_alpha_below_half_review_required']
                        if self.rows and self.rows[-1].get('alphaBelowHalfOnProtected',0) else []),
                    modificationLedger=self.ledger.summary(),
                    uncertainPolicy='purpose-validated native candidates; evidence-bounded geometry and alpha edits')
