import { fileToDataURL, isInvalidImageSrc } from '../core/utils.js';

const LOCAL_FASTSAM_BACKEND_URL = 'http://127.0.0.1:8000';
const FASTSAM_BACKEND_STORAGE_KEY = 'marmo_fastsam_backend_url';
const SEGMENTATION_ENGINE_STORAGE_KEY = 'marmo_segmentation_engine';
const MIN_RUNTIME_ACCEPT_SCORE = 0.45;
const MIN_RUNTIME_ACCEPT_FILL_RATIO = 0.06;
const MAX_RUNTIME_ACCEPT_FILL_RATIO = 0.76;
const MIN_RUNTIME_ACCEPT_PRIMARY_SCORE = -0.05;
const FOOD_RUNTIME_ACCEPT_SCORE = 0.28;
const FOOD_RUNTIME_ACCEPT_FILL_RATIO = 0.05;
const FOOD_MAX_RUNTIME_ACCEPT_FILL_RATIO = 0.82;
const FOOD_RUNTIME_ACCEPT_PRIMARY_SCORE = 0.22;

function normalizeSegmentationEngine(engine) {
    const value = String(engine || '').trim().toLowerCase();
    if ([
        'sam',
        'hq_sam',
        'high_precision_sam',
        'ultralytics_sam',
        'sam_b'
    ].includes(value)) {
        return 'sam';
    }
    return 'sam';
}

function normalizeBackendUrl(url) {
    if (!url || typeof url !== 'string') return '';
    return url.trim().replace(/\/+$/, '');
}

export function getSegmentationBackendUrl() {
    const runtimeValue = typeof window !== 'undefined' ? window.__MARMO_FASTSAM_BACKEND_URL__ : '';
    if (runtimeValue) {
        return normalizeBackendUrl(runtimeValue);
    }

    try {
        const stored = localStorage.getItem(FASTSAM_BACKEND_STORAGE_KEY);
        if (stored) return normalizeBackendUrl(stored);
    } catch (error) {
        console.warn('[segmentation-service] Failed to read backend URL from localStorage:', error);
    }

    return LOCAL_FASTSAM_BACKEND_URL;
}

export function getSegmentationEngine() {
    const runtimeValue = typeof window !== 'undefined' ? window.__MARMO_SEGMENTATION_ENGINE__ : '';
    if (runtimeValue) {
        return normalizeSegmentationEngine(runtimeValue);
    }

    try {
        const stored = localStorage.getItem(SEGMENTATION_ENGINE_STORAGE_KEY);
        if (stored) return normalizeSegmentationEngine(stored);
    } catch (error) {
        console.warn('[segmentation-service] Failed to read segmentation engine from localStorage:', error);
    }

    return 'sam';
}

async function resolveImagePayload(item) {
    const sourceCandidates = [
        item?.cleanPlateDataUrl,
        item?.originalDataUrl,
        item?.dataUrl
    ].filter(value => typeof value === 'string' && !isInvalidImageSrc(value));

    for (const candidate of sourceCandidates) {
        if (candidate.startsWith('data:')) {
            return candidate;
        }
    }

    if (item?.file) {
        return fileToDataURL(item.file);
    }

    for (const candidate of sourceCandidates) {
        if (candidate.startsWith('http://') || candidate.startsWith('https://')) {
            return fileToDataURL(candidate);
        }
    }

    throw new Error('未找到可用于 FastSAM 分割的图片源');
}

function getContextLayers(item, requestLayers) {
    const candidates = [
        item?.semanticViews?.editableSceneLayers,
        item?.scene?.layers,
        item?.layers,
        requestLayers
    ];
    const layers = candidates.find(value => Array.isArray(value) && value.length > 0) || requestLayers;
    return layers
        .filter(layer => Array.isArray(layer?.bbox) && layer.bbox.length === 4)
        .map(layer => ({
            id: layer.id,
            name: layer.name || '',
            semanticType: layer.semanticType || '',
            designRole: layer.designRole || '',
            renderMode: layer.renderMode || '',
            extractionProfile: layer.extractionProfile || '',
            category: layer.category || '',
            runtimeType: layer.runtimeType || '',
            parentLayerId: layer.parentLayerId || null,
            compositeRole: layer.compositeRole || '',
            bbox: layer.bbox
        }));
}

function buildRequestPayload(imageDataUrl, layers, item, engine) {
    return {
        engine: normalizeSegmentationEngine(engine),
        image: imageDataUrl,
        bboxes: layers.map(layer => layer.bbox),
        layerIds: layers.map(layer => layer.id),
        layers: layers.map(layer => ({
            id: layer.id,
            name: layer.name || '',
            semanticType: layer.semanticType || '',
            designRole: layer.designRole || '',
            renderMode: layer.renderMode || '',
            extractionProfile: layer.extractionProfile || '',
            category: layer.category || '',
            runtimeType: layer.runtimeType || '',
            parentLayerId: layer.parentLayerId || null,
            compositeRole: layer.compositeRole || '',
            bbox: layer.bbox
        })),
        contextLayers: getContextLayers(item, layers)
    };
}

function isFoodProductQuality(quality, sourceLayer) {
    const strategy = String(quality?.strategy || '').toLowerCase();
    const strategyProfile = String(quality?.strategyProfile || '').toLowerCase();
    const semanticType = String(sourceLayer?.semanticType || '').toLowerCase();
    const extractionProfile = String(sourceLayer?.extractionProfile || '').toLowerCase();
    const designRole = String(sourceLayer?.designRole || '').toLowerCase();
    return (
        strategy === 'food_product' ||
        strategyProfile.includes('food') ||
        extractionProfile.includes('product') ||
        semanticType === 'product_food' ||
        semanticType === 'product_drink' ||
        designRole === 'product_image'
    );
}

function getQualityThresholds(quality, sourceLayer) {
    if (isFoodProductQuality(quality, sourceLayer)) {
        return {
            minScore: FOOD_RUNTIME_ACCEPT_SCORE,
            minPrimaryScore: FOOD_RUNTIME_ACCEPT_PRIMARY_SCORE,
            minFillRatio: FOOD_RUNTIME_ACCEPT_FILL_RATIO,
            maxFillRatio: FOOD_MAX_RUNTIME_ACCEPT_FILL_RATIO
        };
    }

    return {
        minScore: MIN_RUNTIME_ACCEPT_SCORE,
        minPrimaryScore: MIN_RUNTIME_ACCEPT_PRIMARY_SCORE,
        minFillRatio: MIN_RUNTIME_ACCEPT_FILL_RATIO,
        maxFillRatio: MAX_RUNTIME_ACCEPT_FILL_RATIO
    };
}

function normalizeQualityGate(quality, sourceLayer = null) {
    if (!quality) {
        return {
            status: 'unknown',
            runtimeAction: 'hold',
            shouldGenerateRuntimeLayer: false,
            needsHigherPrecision: true,
            issues: ['missing_quality'],
            message: 'FastSAM 未返回质量评估，需要高精度模型复核。'
        };
    }

    const issues = Array.isArray(quality.issues) ? [...quality.issues] : [];
    const score = Number(quality.score);
    const primaryScore = Number(quality.primaryScore);
    const targetFillRatio = Number(quality.targetFillRatio);
    const explicitAccept = quality.runtimeAction === 'accept' || quality.shouldGenerateRuntimeLayer === true;
    const explicitHold = quality.runtimeAction === 'hold' || quality.shouldGenerateRuntimeLayer === false;
    const { minScore, minPrimaryScore, minFillRatio, maxFillRatio } = getQualityThresholds(quality, sourceLayer);

    if (explicitAccept && quality.status !== 'failed') {
        return {
            ...quality,
            status: 'ok',
            runtimeAction: 'accept',
            shouldGenerateRuntimeLayer: true,
            needsHigherPrecision: false,
            issues: issues.filter(issue => issue !== 'low_quality_status'),
            message: ''
        };
    }

    if (quality.status === 'failed') issues.push('failed');
    if (quality.status === 'low_quality' || explicitHold) issues.push('low_quality_status');
    if (Number.isFinite(score) && score < minScore) issues.push('low_quality_score');
    if (Number.isFinite(primaryScore) && primaryScore < minPrimaryScore) issues.push('low_primary_score');
    if (Number.isFinite(targetFillRatio) && targetFillRatio < minFillRatio) issues.push('low_target_fill');
    if (Number.isFinite(targetFillRatio) && targetFillRatio > maxFillRatio) issues.push('high_target_fill');

    const uniqueIssues = [...new Set(issues)];
    const shouldGenerateRuntimeLayer =
        quality.runtimeAction === 'accept' ||
        quality.shouldGenerateRuntimeLayer === true ||
        (quality.status === 'ok' && uniqueIssues.length === 0);
    const accepted = shouldGenerateRuntimeLayer && uniqueIssues.length === 0;

    return {
        ...quality,
        status: accepted ? 'ok' : (quality.status || 'low_quality'),
        runtimeAction: accepted ? 'accept' : 'hold',
        shouldGenerateRuntimeLayer: accepted,
        needsHigherPrecision: !accepted,
        issues: uniqueIssues,
        message: accepted
            ? ''
            : `FastSAM 质量不足，建议切换高精度模型处理（${uniqueIssues.join(', ') || 'quality_hold'}）。`
    };
}

function normalizeSegmentResponse(responseJson, layers, requestedEngine) {
    const resultMap = new Map();
    const cutouts = Array.isArray(responseJson?.cutouts) ? responseJson.cutouts : [];
    const requestLayerMap = new Map(layers.map(layer => [layer.id, layer]));

    cutouts.forEach((cutout, index) => {
        let sourceLayer = requestLayerMap.get(cutout.layerId);

        // Some validation backends may omit/mutate layerId. Keep matching strict by
        // default, but allow deterministic index fallback for the same request batch.
        if (!sourceLayer && layers.length === cutouts.length) {
            sourceLayer = layers[index];
        }
        if (!sourceLayer && layers.length === 1 && cutouts.length === 1) {
            sourceLayer = layers[0];
        }
        if (!sourceLayer) return;

        const quality = normalizeQualityGate(cutout.quality, sourceLayer);
        resultMap.set(sourceLayer, {
            success: true,
            dataUrl: cutout.image,
            bbox: cutout.bbox || sourceLayer.bbox,
            width: cutout.width || null,
            height: cutout.height || null,
            extractEngine: cutout.extractEngine || responseJson?.engine || normalizeSegmentationEngine(requestedEngine),
            quality,
            runtimeAction: quality.runtimeAction,
            shouldGenerateRuntimeLayer: quality.shouldGenerateRuntimeLayer
        });
    });

    return resultMap;
}

export async function segmentLayers({ item, layers = [], onProgress = () => {}, engine = null }) {
    const validLayers = layers.filter(layer => Array.isArray(layer?.bbox) && layer.bbox.length === 4);
    if (validLayers.length === 0) return new Map();

    const backendUrl = getSegmentationBackendUrl();
    const requestedEngine = normalizeSegmentationEngine(engine || getSegmentationEngine());
    const engineLabel = requestedEngine === 'sam' ? '高精 SAM 本地后端' : 'FastSAM 本地后端';
    const imageDataUrl = await resolveImagePayload(item);
    const requestLayers = validLayers.map((layer, index) => {
        if (!layer.id) {
            layer.id = `segment-layer-${index}-${Date.now()}`;
        }
        return layer;
    });

    onProgress(`🧩 **${engineLabel}**: 正在提交 ${requestLayers.length} 个图层...`);

    const response = await fetch(`${backendUrl}/segment`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(buildRequestPayload(imageDataUrl, requestLayers, item, requestedEngine))
    });

    const responseJson = await response.json();
    if (!response.ok || !responseJson?.success) {
        throw new Error(responseJson?.error || `${engineLabel} 请求失败: HTTP ${response.status}`);
    }

    const resultMap = normalizeSegmentResponse(responseJson, requestLayers, requestedEngine);
    const failedLayers = requestLayers.filter(layer => !resultMap.has(layer));
    if (failedLayers.length > 0) {
        const names = failedLayers.map(layer => layer.name || '未命名图层').join('、');
        throw new Error(`${engineLabel} 未返回这些图层的 cutout: ${names}`);
    }

    onProgress(`✅ **${engineLabel}**: 已返回 ${resultMap.size} 个透明图层`);
    return resultMap;
}

export async function segmentSingleLayer({ item, layer, onProgress = () => {}, engine = null }) {
    const requestLayer = {
        ...layer,
        id: layer.id || `segment-layer-single-${Date.now()}`
    };
    const resultMap = await segmentLayers({
        item,
        layers: [requestLayer],
        onProgress,
        engine
    });
    const result = resultMap.get(requestLayer);
    if (!result?.dataUrl) {
        throw new Error(`分割引擎未返回单图层 cutout: ${requestLayer.name || requestLayer.id}`);
    }
    return result;
}
