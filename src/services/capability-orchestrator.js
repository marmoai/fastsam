import {
    getAssistantWorkspaceContext,
    getWorkbenchLayers,
    recordWorkspaceAction
} from './workspace-context.js';

const SEARCH_STOP_WORDS = new Set([
    '搜索', '搜', '查找', '找', '寻找', '同款', '相似', '类似', '商品', '产品',
    '一下', '帮我', '请', '看看', '的', '那件', '那个', '这个'
]);

export function looksLikeLensSearchRequest(text = '') {
    const normalized = String(text || '').trim();
    if (!normalized) return false;
    return /搜索|搜一下|搜同款|找同款|寻找同款|相似款|类似款|商品推荐|找类似/.test(normalized)
        && /同款|相似|类似|商品|产品|推荐|搜|搜索|找/.test(normalized);
}

function getLayerState(item, index) {
    if (item?.layerStates instanceof Map) return item.layerStates.get(index) || {};
    return item?.layerStates?.[index] || {};
}

function isBackgroundLayer(layer) {
    return layer?.category === 'background'
        || layer?.layerType === 'background_plate'
        || layer?.runtimeType === 'background_master';
}

const STANDALONE_ASSET_TYPES = new Set([
    'layer-explode',
    'layer-extract',
    'isolated-edit',
    'extraction'
]);

function isStandaloneExtractedAsset(item) {
    return !!item
        && !!(item.parentId || item.sourceParentId || item.metadata?.parentId)
        && STANDALONE_ASSET_TYPES.has(String(item.type || '').toLowerCase())
        && !!(item.file || item.dataUrl || item.originalDataUrl);
}

function buildStandaloneAssetLayer(item, itemId) {
    return {
        id: itemId ? `asset-${itemId}` : 'standalone-asset',
        name: item.layerName || item.label || item.name || '当前提取图层',
        // The asset is already cropped to the selected object. Lens should use
        // the whole child image rather than the parent's coordinate system.
        bbox: [0, 0, 1000, 1000],
        semanticType: 'standalone_asset'
    };
}

function getTargetLayerId(target) {
    return target?.layer?.id || (target?.layerIndex != null ? `layer-${target.layerIndex}` : null);
}

function getSelectableLayers(item) {
    // layerStates indexes are maintained against the rendered scene layers.
    // Prefer that source so a semantic view cannot shift selection indexes.
    if (Array.isArray(item?.scene?.layers) && item.scene.layers.length > 0) return item.scene.layers;
    if (Array.isArray(item?.layers) && item.layers.length > 0) return item.layers;
    return getWorkbenchLayers(item);
}

function getObjectHints(text = '') {
    return String(text || '')
        .replace(/[“”"'‘’]/g, '')
        .split(/[\s,，。！？!?、:：]+/)
        .flatMap(part => part.split(/(同款|相似|类似|搜索|搜一下|搜索一下|寻找|查找|找|商品|产品|推荐|帮我|请|一下)/))
        .map(part => part.trim())
        .filter(part => part.length >= 2 && !SEARCH_STOP_WORDS.has(part));
}

function scoreLayerByText(layer, text) {
    const name = String(layer?.name || '').toLowerCase();
    if (!name) return 0;
    const normalizedText = String(text || '').toLowerCase();
    if (normalizedText.includes(name)) return 100;

    return getObjectHints(text).reduce((score, hint) => {
        return score + (name.includes(hint.toLowerCase()) ? hint.length : 0);
    }, 0);
}

export function resolveLensSearchTarget(appState, requestText = '', workspaceContext = null) {
    const context = workspaceContext || getAssistantWorkspaceContext(appState);
    const itemId = context?.activeItemId || context?.selectedItemIds?.[0] || null;
    const item = itemId ? appState?.workbenchItems?.get(itemId) : null;
    if (!item) {
        return { ok: false, reason: '当前没有可用于 Lens 搜索的活动图片。' };
    }

    // Extracted layers are first-class Workbench assets, not semantic layers
    // on the parent image. Their transparent data is already the search crop.
    if (isStandaloneExtractedAsset(item)) {
        return {
            ok: true,
            itemId,
            item,
            layer: buildStandaloneAssetLayer(item, itemId),
            layerIndex: null,
            source: 'standalone_asset'
        };
    }

    const layers = getSelectableLayers(item);
    const editableLayers = layers
        .map((layer, index) => ({ layer, index }))
        .filter(({ layer }) => !isBackgroundLayer(layer) && Array.isArray(layer?.bbox));

    if (context?.activeLayerId) {
        const activeLayer = editableLayers.find(({ layer, index }) => {
            return layer?.id === context.activeLayerId || `layer-${index}` === context.activeLayerId;
        });
        if (activeLayer) {
            return {
                ok: true,
                itemId,
                item,
                layer: activeLayer.layer,
                layerIndex: activeLayer.index,
                source: 'active_layer_id'
            };
        }
    }

    const selectedLayers = editableLayers.filter(({ index }) => getLayerState(item, index).selected);

    if (selectedLayers.length > 1) {
        return { ok: false, reason: '当前选中了多个图层，请只保留一个目标后再搜索同款。' };
    }

    if (selectedLayers.length === 1) {
        const { layer, index } = selectedLayers[0];
        return { ok: true, itemId, item, layer, layerIndex: index, source: 'selected_layer' };
    }

    const scored = editableLayers
        .map(candidate => ({ ...candidate, score: scoreLayerByText(candidate.layer, requestText) }))
        .filter(candidate => candidate.score > 0)
        .sort((a, b) => b.score - a.score);

    if (scored.length > 0 && (scored.length === 1 || scored[0].score > scored[1].score)) {
        const { layer, index } = scored[0];
        return { ok: true, itemId, item, layer, layerIndex: index, source: 'semantic_name' };
    }

    if (editableLayers.length === 1) {
        const { layer, index } = editableLayers[0];
        return { ok: true, itemId, item, layer, layerIndex: index, source: 'single_editable_layer' };
    }

    return { ok: false, reason: '我找到了多个可编辑图层，但无法确定要搜索哪一个。请先选中目标图层，或直接使用 Lens 框选。' };
}

export async function executeLensSearch(appState, requestText = '', workspaceContext = null) {
    const target = resolveLensSearchTarget(appState, requestText, workspaceContext);
    if (!target.ok) {
        const context = workspaceContext || getAssistantWorkspaceContext(appState);
        console.warn('[LensDispatch] target unresolved', {
            request: requestText,
            reason: target.reason,
            activeItemId: context?.activeItemId || null,
            activeLayerId: context?.activeLayerId || null,
            activeLayerName: context?.activeLayerName || null,
            selectedLayerNames: context?.activeItem?.selectedLayers?.map(layer => layer.name) || [],
            layerNames: context?.activeItem?.layers?.map(layer => layer.name) || []
        });
        return { handled: true, success: false, text: target.reason };
    }

    const lens = globalThis.window?.marmoLens;
    if (!lens || typeof lens.performVisualSearch !== 'function') {
        return { handled: true, success: false, text: 'Lens 当前还没有完成初始化，请先打开或刷新工作台后重试。' };
    }

    const label = target.layer?.name || '当前图层';
    const targetLayerId = getTargetLayerId(target);
    console.log(`[LensDispatch] target item=${target.itemId} layer=${targetLayerId} label=${label} source=${target.source} bbox=${JSON.stringify(target.layer.bbox)}`);
    const result = await lens.performVisualSearch(target.item, {
        box: target.layer.bbox,
        label,
        layerId: targetLayerId
    });

    if (!result?.success) {
        recordWorkspaceAction(appState, {
            actionName: 'lens_search_failed',
            itemId: target.itemId,
            layerId: targetLayerId,
            layerName: label,
            status: 'failed',
            hasResult: false
        });
        return {
            handled: true,
            success: false,
            text: result?.error ? `Lens 搜索失败：${result.error}` : 'Lens 搜索没有返回结果。'
        };
    }

    return {
        handled: true,
        success: true,
        text: `已根据“${label}”打开 Lens 同款搜索结果。`,
        target: {
            itemId: target.itemId,
            layerId: targetLayerId,
            source: target.source
        }
    };
}
