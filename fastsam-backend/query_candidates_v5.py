"""Purpose-validated HQ query fallback; heads never count as independent queries."""
import numpy as np
from candidate_pool_v5 import branch_candidates


def validated_candidates(defaults, evidence, shape, points, labels, validator):
    """Keep a passing official result; expose native alternatives on failure only.

    The caller supplies ownership/completion-specific constraints. There is no
    whole-incumbent retention gate here: legitimate neighbour removal is allowed.
    """
    xy = np.asarray(points if points is not None else [], dtype=int).reshape(-1, 2)
    kinds = np.asarray(labels if labels is not None else [], dtype=int).reshape(-1)
    if len(xy) != len(kinds):
        raise ValueError('query points and labels must match')
    rows = []

    def acceptable(mask, source):
        mask = np.asarray(mask) > .5
        valid = (mask.shape == tuple(shape[:2]) and
                 all(0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
                     and bool(mask[y, x]) == bool(label)
                     for (x, y), label in zip(xy, kinds)))
        if valid:
            valid = bool(validator(mask))
        rows.append(dict(source=source, accepted=valid))
        return valid

    established = [m for m in defaults if acceptable(m, dict(branch='official_default'))]
    if established:
        return established, dict(status='kept_default', independentQueryCount=1, candidates=rows)
    alternatives = []
    for mask, source in branch_candidates(evidence, shape, 'guided'):
        if any(np.array_equal(mask, old) for old in alternatives):
            continue
        if acceptable(mask, source):
            alternatives.append(mask)
    return alternatives, dict(status='fallback' if alternatives else 'unresolved',
                              independentQueryCount=1, candidates=rows)
