const MAX_RECENT_ACTIONS = 30;
const MAX_LAYER_SUMMARY = 40;

function cloneValue(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function getLayers(item) {
    if (Array.isArray(item?.scene?.layers) && item.scene.layers.length > 0) return item.scene.layers;
    if (Array.isArray(item?.semanticViews?.editableSceneLayers) && item.semanticViews.editableSceneLayers.length > 0) {
        return item.semanticViews.editableSceneLayers;
    }
    return Array.isArray(item?.layers) ? item.layers : [];
}

function isBackgroundLayer(layer) {
    const text = [
        layer?.name,
        layer?.semanticType,
        layer?.runtimeType,
        layer?.category,
        layer?.designRole
    ].filter(Boolean).join(' ').toLowerCase();

    return layer?.category === 'background' ||
        layer?.layerType === 'background_plate' ||
        layer?.runtimeType === 'background_master' ||
        text.includes('background') ||
        text.includes('背景') ||
        text.includes('底板') ||
        text.includes('底色');
}

function summarizeLayer(layer, index) {
    return {
        id: layer?.id || `layer-${index + 1}`,
        name: String(layer?.name || `图层 ${index + 1}`).slice(0, 120),
        semanticType: layer?.semanticType || layer?.category || 'unknown',
        runtimeType: layer?.runtimeType || null,
        bbox: Array.isArray(layer?.bbox) ? cloneValue(layer.bbox) : null,
        visible: layer?.visible !== false,
        extracted: !!(layer?.cutoutUrl || layer?.previewUrl),
        isBackground: isBackgroundLayer(layer)
    };
}

function isLayerSelected(item, index) {
    if (item?.layerStates instanceof Map) {
        return item.layerStates.get(index)?.selected === true;
    }
    return item?.layerStates?.[index]?.selected === true;
}

function summarizeItem(item, itemId) {
    if (!item) return null;
    const layers = getLayers(item);
    const layerSummary = layers.slice(0, MAX_LAYER_SUMMARY).map((layer, index) => ({
        ...summarizeLayer(layer, index),
        selected: isLayerSelected(item, index)
    }));

    return {
        id: itemId,
        name: String(item.name || item.label || '未命名图片').slice(0, 160),
        type: item.type || 'image',
        layerCount: layers.length,
        hasFullSemanticAnalysis: !!item.hasFullSemanticAnalysis,
        cleanPlateStatus: item.cleanPlateStatus || 'idle',
        hasCleanPlate: !!item.cleanPlateDataUrl,
        layers: layerSummary,
        extractedLayerCount: layerSummary.filter(layer => layer.extracted).length,
        selectedLayers: layerSummary.filter(layer => layer.selected && !layer.isBackground),
        selectedLayerCount: layerSummary.filter(layer => layer.selected && !layer.isBackground).length,
        editableLayerNames: layerSummary
            .filter(layer => !layer.isBackground)
            .map(layer => layer.name)
            .slice(0, 20)
    };
}

function normalizeAction(action = {}) {
    return {
        actionName: String(action.actionName || action.type || 'workbench_action').slice(0, 120),
        prompt: String(action.prompt || '').slice(0, 320),
        itemId: action.itemId || null,
        layerId: action.layerId || null,
        layerName: action.layerName || null,
        status: action.status || 'completed',
        hasResult: !!action.hasResult,
        timestamp: Number(action.timestamp) || Date.now()
    };
}

export function getWorkbenchLayers(item) {
    return getLayers(item);
}

export function syncWorkspaceContext(appState, patch = {}) {
    if (!appState) return null;

    const selectedItemIds = patch.selectedItemIds !== undefined
        ? [...new Set(patch.selectedItemIds || [])]
        : Array.from(appState.selectedWorkbenchItems || []);
    const inferredActiveItemId = appState.currentActiveWorkbenchItemId
        || (selectedItemIds.length === 1 ? selectedItemIds[0] : null);
    const activeItemId = patch.activeItemId !== undefined
        ? patch.activeItemId
        : inferredActiveItemId;
    const current = appState.workspaceContext || {};
    const activeItem = activeItemId ? appState.workbenchItems?.get(activeItemId) : null;
    const activeItemChanged = current.activeItemId !== activeItemId;
    const selectedItems = selectedItemIds
        .map(id => ({ id, item: appState.workbenchItems?.get(id) }))
        .filter(entry => entry.item)
        .map(entry => summarizeItem(entry.item, entry.id));

    appState.workspaceContext = {
        ...current,
        activeItemId,
        activeItem: patch.activeItem !== undefined ? cloneValue(patch.activeItem) : summarizeItem(activeItem, activeItemId),
        activeLayerId: patch.activeLayerId !== undefined
            ? patch.activeLayerId
            : activeItemChanged ? null : (current.activeLayerId || null),
        activeLayerName: patch.activeLayerName !== undefined
            ? patch.activeLayerName
            : activeItemChanged ? null : (current.activeLayerName || null),
        selectedItemIds,
        selectedItems,
        lastUpdatedAt: Date.now(),
        ...(patch.status ? { status: patch.status } : {}),
        ...(patch.sceneMode ? { sceneMode: patch.sceneMode } : {})
    };

    return appState.workspaceContext;
}

export function recordWorkspaceAction(appState, action = {}) {
    if (!appState) return null;
    const normalized = normalizeAction(action);
    const selectionPatch = Object.prototype.hasOwnProperty.call(action, 'activeLayerId')
        ? { activeLayerId: action.activeLayerId, activeLayerName: action.activeLayerName || null }
        : action.actionName === 'semantic_layer_selection_changed'
            ? { activeLayerId: action.layerId || null, activeLayerName: action.layerName || null }
            : {};
    const context = syncWorkspaceContext(appState, {
        activeItemId: action.itemId !== undefined
            ? action.itemId
            : appState.currentActiveWorkbenchItemId || undefined,
        status: action.status || 'ready',
        ...selectionPatch
    });
    context.lastAction = normalized;
    context.recentActions = [
        ...(Array.isArray(context.recentActions) ? context.recentActions : []),
        normalized
    ].slice(-MAX_RECENT_ACTIONS);
    context.lastUpdatedAt = Date.now();
    return context;
}

export function getAssistantWorkspaceContext(appState) {
    const context = syncWorkspaceContext(appState);
    if (!context) return null;

    return cloneValue({
        activeItemId: context.activeItemId,
        activeItem: context.activeItem,
        activeLayerId: context.activeLayerId || null,
        activeLayerName: context.activeLayerName || null,
        selectedItemIds: context.selectedItemIds,
        selectedItems: context.selectedItems,
        sceneMode: context.sceneMode || 'unknown',
        status: context.status || 'ready',
        lastAction: context.lastAction || null,
        recentActions: (context.recentActions || []).slice(-8)
    });
}
