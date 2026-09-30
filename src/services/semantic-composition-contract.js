function normalizedBox(box) {
    if (!Array.isArray(box) || box.length !== 4) return null;
    const values = box.map(Number);
    if (!values.every(Number.isFinite)) return null;
    const clamped = values.map(value => Math.max(0, Math.min(1000, value)));
    return clamped[2] > clamped[0] && clamped[3] > clamped[1] ? clamped : null;
}

function area(box) {
    return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

export function requiredCompositionReplacementCount(existingChildCount) {
    const count = Math.max(0, Number(existingChildCount) || 0);
    return Math.max(2, count + (count > 0 ? 1 : 0));
}

export function shouldReviewCompositionGroup(layer) {
    const isGroup = layer?.compositeRole === 'composite_group' ||
        layer?.semanticType === 'composite_group';
    const groupBox = normalizedBox(layer?.bbox);
    if (!isGroup || !groupBox) return false;

    const children = Array.isArray(layer?.children) ? layer.children : [];
    if (children.length === 0 || layer?.compositionReview?.required === true) return true;

    const groupArea = area(groupBox);
    if (groupArea <= 0) return false;
    return children.some(child => {
        const childBox = normalizedBox(child?.bbox);
        return childBox && area(childBox) / groupArea >= 0.52;
    });
}

export function acceptsMoreAtomicReplacement(existingChildCount, reviewedChildCount) {
    return Number(reviewedChildCount) >= requiredCompositionReplacementCount(existingChildCount);
}
