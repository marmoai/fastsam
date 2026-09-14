// Shared, side-effect-free ownership helpers for Magic Layers completion.
// A semantic entity may be represented by one layer, a composite parent with
// child layers, or several independent layers carrying the same entityId.  All
// completion stages must resolve that representation in the same way.

function layerIdOf(layer) {
    const value = layer?.layerId || layer?.id;
    return value == null || value === '' ? null : String(value);
}

function entityIdOf(layer) {
    const value = layer?.entityId || layer?.semanticEntityId || layer?.entity?.id;
    return value == null || value === '' ? null : String(value);
}

function isCompositeLayer(layer) {
    return layer?.compositeRole === 'composite_group' ||
        layer?.semanticType === 'composite_group' ||
        layer?.runtimeType === 'semantic_group';
}

function validBbox(bbox) {
    return Array.isArray(bbox) && bbox.length === 4 && bbox.every(value => Number.isFinite(Number(value)));
}

function unionBbox(current, next) {
    if (!validBbox(next)) return current;
    if (!current) return next.map(Number);
    return [
        Math.min(current[0], Number(next[0])),
        Math.min(current[1], Number(next[1])),
        Math.max(current[2], Number(next[2])),
        Math.max(current[3], Number(next[3]))
    ];
}

export function createCompletionLayerIndex(layers = []) {
    const byId = new Map();
    const childrenByParent = new Map();
    const byEntity = new Map();
    for (const layer of Array.isArray(layers) ? layers : []) {
        const id = layerIdOf(layer);
        if (!id) continue;
        byId.set(id, layer);
        const parentId = layer?.parentLayerId == null ? null : String(layer.parentLayerId);
        if (parentId) {
            const children = childrenByParent.get(parentId) || [];
            children.push(id);
            childrenByParent.set(parentId, children);
        }
        const entityId = entityIdOf(layer);
        if (entityId) {
            const entityLayers = byEntity.get(entityId) || [];
            entityLayers.push(id);
            byEntity.set(entityId, entityLayers);
        }
    }
    // childLayerIds are authoritative when present, but parent links are also
    // indexed because older graph snapshots only persisted one direction.
    for (const layer of Array.isArray(layers) ? layers : []) {
        const parentId = layerIdOf(layer);
        if (!parentId || !Array.isArray(layer?.childLayerIds)) continue;
        const children = childrenByParent.get(parentId) || [];
        for (const childId of layer.childLayerIds.map(String)) {
            if (byId.has(childId) && !children.includes(childId)) children.push(childId);
        }
        childrenByParent.set(parentId, children);
    }
    return { byId, childrenByParent, byEntity };
}

function normalizeIndex(layersOrIndex) {
    if (layersOrIndex?.byId instanceof Map) return layersOrIndex;
    return createCompletionLayerIndex(layersOrIndex);
}

export function resolveEntityParts(layersOrIndex, rootId) {
    const index = normalizeIndex(layersOrIndex);
    const requestedId = rootId == null ? null : String(rootId);
    const requested = requestedId ? index.byId.get(requestedId) : null;
    if (!requested) {
        return {
            entityId: requestedId,
            rootLayerId: requestedId,
            layerIds: [],
            layers: [],
            unionBbox: null,
            composite: false
        };
    }

    const root = (() => {
        let candidate = requested;
        const seen = new Set();
        while (candidate?.parentLayerId != null && !seen.has(layerIdOf(candidate))) {
            seen.add(layerIdOf(candidate));
            const parent = index.byId.get(String(candidate.parentLayerId));
            if (!parent || !isCompositeLayer(parent)) break;
            candidate = parent;
        }
        return candidate;
    })();
    const rootIdResolved = layerIdOf(root) || requestedId;
    const entityId = entityIdOf(root) || entityIdOf(requested);
    const collected = [];
    const visited = new Set();
    const visit = id => {
        const normalizedId = String(id);
        if (visited.has(normalizedId)) return;
        const layer = index.byId.get(normalizedId);
        if (!layer) return;
        visited.add(normalizedId);
        collected.push(layer);
        for (const childId of index.childrenByParent.get(normalizedId) || []) visit(childId);
    };
    visit(rootIdResolved);

    // Some semantic parsers emit multiple atomic layers for one entity without
    // parent links. Include those siblings only when an explicit entityId binds
    // them; never merge merely because names or boxes happen to overlap.
    if (entityId) {
        for (const id of index.byEntity.get(entityId) || []) visit(id);
    }
    let union = null;
    for (const layer of collected) union = unionBbox(union, layer.bbox);
    return {
        entityId: entityId || rootIdResolved,
        rootLayerId: rootIdResolved,
        layerIds: collected.map(layerIdOf).filter(Boolean),
        layers: collected,
        unionBbox: union,
        composite: collected.length > 1 || isCompositeLayer(root)
    };
}

export function expandEntityLayerIds(layersOrIndex, ids = []) {
    const index = normalizeIndex(layersOrIndex);
    const expanded = [];
    const seen = new Set();
    for (const id of Array.isArray(ids) ? ids : []) {
        const parts = resolveEntityParts(index, id);
        for (const layerId of parts.layerIds) {
            if (seen.has(layerId)) continue;
            seen.add(layerId);
            expanded.push(layerId);
        }
    }
    return expanded;
}

export function resolveCompletionTaskEntities(task, layersOrIndex) {
    const index = normalizeIndex(layersOrIndex);
    const target = resolveEntityParts(index, task?.targetLayerId);
    const occluderLayerIds = expandEntityLayerIds(index, task?.occluderLayerIds || [])
        .filter(id => !target.layerIds.includes(id));
    const occluders = occluderLayerIds.map(id => index.byId.get(id)).filter(Boolean);
    let occluderBbox = null;
    for (const layer of occluders) occluderBbox = unionBbox(occluderBbox, layer.bbox);
    return {
        target,
        occluders,
        occluderLayerIds,
        occluderBbox
    };
}
