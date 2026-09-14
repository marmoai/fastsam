import {
    RECOGNIZE_BACKEND_URL,
    buildAuthorizedHeaders,
    fileToDataURL,
    isInvalidImageSrc
} from '../core/utils.js';

const LOCAL_FASTSAM_BACKEND_URL = 'http://127.0.0.1:8000';
const FASTSAM_BACKEND_STORAGE_KEY = 'marmo_fastsam_backend_url';
const FASTSAM_BACKEND_TOKEN_STORAGE_KEY = 'marmo_fastsam_backend_token';
const FASTSAM_BACKEND_MODE_STORAGE_KEY = 'marmo_fastsam_backend_mode';
const SEGMENTATION_ENGINE_STORAGE_KEY = 'marmo_segmentation_engine';
const SEGMENTATION_PROXY_MODE = 'segment_proxy';
const LOCAL_BACKEND_PROBE_TIMEOUT_MS = 1200;
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

function normalizeBackendToken(token) {
    if (!token || typeof token !== 'string') return '';
    return token.trim();
}

function normalizeBackendMode(mode) {
    const value = String(mode || '').trim().toLowerCase();
    if (['cloud', 'remote', 'proxy'].includes(value)) return 'cloud';
    if (['local', 'direct'].includes(value)) return 'local';
    if (value === 'auto') return 'auto';
    return '';
}

function getExplicitSegmentationBackendUrl() {
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

    // An empty value means no explicit override. The transport resolver can
    // then probe the local backend and fall back to the cloud proxy.
    return '';
}

export function getSegmentationBackendUrl() {
    return getExplicitSegmentationBackendUrl() || LOCAL_FASTSAM_BACKEND_URL;
}

function getExplicitSegmentationBackendToken() {
    const runtimeValue = typeof window !== 'undefined' ? window.__MARMO_FASTSAM_BACKEND_TOKEN__ : '';
    if (runtimeValue) {
        return normalizeBackendToken(runtimeValue);
    }

    try {
        const stored = localStorage.getItem(FASTSAM_BACKEND_TOKEN_STORAGE_KEY);
        if (stored) return normalizeBackendToken(stored);
    } catch (error) {
        console.warn('[segmentation-service] Failed to read backend token from localStorage:', error);
    }

    return '';
}

export function getSegmentationBackendToken() {
    return getExplicitSegmentationBackendToken();
}

function getExplicitSegmentationBackendMode() {
    const runtimeValue = typeof window !== 'undefined' ? window.__MARMO_FASTSAM_BACKEND_MODE__ : '';
    if (runtimeValue) {
        return normalizeBackendMode(runtimeValue);
    }

    try {
        const stored = localStorage.getItem(FASTSAM_BACKEND_MODE_STORAGE_KEY);
        if (stored) return normalizeBackendMode(stored);
    } catch (error) {
        console.warn('[segmentation-service] Failed to read backend mode from localStorage:', error);
    }

    return '';
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
        item?.segmentationSourceUrl,
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

function isLocalBackendUrl(url) {
    return url.startsWith('http://127.0.0.1') || url.startsWith('http://localhost');
}

async function isLocalBackendAvailable() {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), LOCAL_BACKEND_PROBE_TIMEOUT_MS);

    try {
        const response = await fetch(`${LOCAL_FASTSAM_BACKEND_URL}/healthz`, {
            method: 'GET',
            signal: controller.signal
        });
        return response.ok;
    } catch (_error) {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

async function resolveSegmentationTransport() {
    const explicitMode = getExplicitSegmentationBackendMode();
    const explicitBackendUrl = getExplicitSegmentationBackendUrl();

    // A concrete URL is the strongest override so the original local testing
    // workflow remains valid even if a previous cloud mode is still stored.
    if (explicitBackendUrl) {
        return {
            mode: 'direct',
            backendUrl: explicitBackendUrl,
            backendToken: getExplicitSegmentationBackendToken()
        };
    }

    if (explicitMode === 'cloud') {
        return {
            mode: 'proxy',
            backendUrl: RECOGNIZE_BACKEND_URL,
            backendToken: ''
        };
    }

    if (explicitMode === 'local') {
        return {
            mode: 'direct',
            backendUrl: LOCAL_FASTSAM_BACKEND_URL,
            backendToken: ''
        };
    }

    if (await isLocalBackendAvailable()) {
        return {
            mode: 'direct',
            backendUrl: LOCAL_FASTSAM_BACKEND_URL,
            backendToken: ''
        };
    }

    return {
        mode: 'proxy',
        backendUrl: RECOGNIZE_BACKEND_URL,
        backendToken: ''
    };
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
            id: layer.id || layer.layerId,
            name: layer.name || '',
            semanticType: layer.semanticType || '',
            designRole: layer.designRole || '',
            renderMode: layer.renderMode || '',
            extractionProfile: layer.extractionProfile || '',
            category: layer.category || '',
            runtimeType: layer.runtimeType || '',
            entityId: layer.entityId || layer.semanticEntityId || null,
            parentLayerId: layer.parentLayerId || null,
            childLayerIds: Array.isArray(layer.childLayerIds) ? [...layer.childLayerIds] : [],
            compositeRole: layer.compositeRole || '',
            completionOccluder: layer.completionOccluder === true,
            // This is intentionally scoped to the post-inpaint spatial
            // completion request.  It tells SAM that an already-extracted
            // overlapping scene object is not part of the target silhouette.
            completionForegroundContext: layer.completionForegroundContext === true,
            completionSegmentation: layer.completionSegmentation === true,
            samModelVariant: layer.samModelVariant || null,
            completionObservationMask: layer.completionObservationMask || null,
            completionOcclusionMask: layer.completionOcclusionMask || null,
            completionForegroundContextMask: layer.completionForegroundContextMask || null,
            bbox: layer.bbox
        }));
}

function normalizeSegmentationQualityProfile(profile) {
    const value = String(profile || '').trim().toLowerCase();
    return ['completion', 'completion_candidate', 'completion_review', 'canonical_completion'].includes(value)
        ? 'completion'
        : 'publish';
}

function buildRequestPayload(imageDataUrl, layers, item, engine, qualityProfile = 'publish') {
    return {
        engine: normalizeSegmentationEngine(engine),
        qualityProfile: normalizeSegmentationQualityProfile(qualityProfile),
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
            entityId: layer.entityId || layer.semanticEntityId || null,
            parentLayerId: layer.parentLayerId || null,
            childLayerIds: Array.isArray(layer.childLayerIds) ? [...layer.childLayerIds] : [],
            compositeRole: layer.compositeRole || '',
            completionOccluder: layer.completionOccluder === true,
            completionForegroundContext: layer.completionForegroundContext === true,
            completionSegmentation: layer.completionSegmentation === true,
            samModelVariant: layer.samModelVariant || null,
            completionObservationMask: layer.completionObservationMask || null,
            completionOcclusionMask: layer.completionOcclusionMask || null,
            completionForegroundContextMask: layer.completionForegroundContextMask || null,
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

function isSpatialTableCompletionQuality(quality, sourceLayer) {
    const strategy = String(quality?.strategy || '').toLowerCase();
    const profile = String(quality?.qualityProfile || quality?.phase || '').toLowerCase();
    const descriptor = [
        sourceLayer?.name,
        sourceLayer?.semanticType,
        sourceLayer?.extractionProfile,
        sourceLayer?.designRole
    ].join(' ').toLowerCase();
    const tableName = /table|desk|console|sideboard|cabinet|counter|桌|案|几|台/.test(descriptor);
    const spatialCompletion = Boolean(
        sourceLayer?.completionSegmentation &&
        (quality?.spatialCanonicalCompletion === true || strategy === 'table') &&
        (profile.includes('completion') || quality?.phase === 'completion' || quality?.qualityProfile === 'completion')
    );
    if (!spatialCompletion || (!tableName && strategy !== 'table')) return false;

    const selected = Array.isArray(quality?.debugCandidates)
        ? quality.debugCandidates.find(candidate => candidate?.selected)
        : null;
    const shape = selected?.shapeFeatures || {};
    const spatialCompletionAnchor = selected?.spatialCompletionAnchor === true;
    const inside = Number(selected?.inside ?? selected?.maskInsideTargetRatio);
    const bboxOverlap = Number(selected?.bboxOverlap ?? selected?.bboxOverlapRatio);
    const touch = Number(selected?.touch ?? quality?.maskAudit?.touchesTargetEdges);
    const fill = Number(quality?.targetFillRatio);
    const components = Number(quality?.maskAudit?.components);
    const enclosedHoles = Number(quality?.maskAudit?.enclosedHolePixels);
    const pixels = Number(quality?.maskAudit?.pixels);
    const hasStructure = Boolean(
        shape.isHorizontalSurface ||
        shape.isTableSupport ||
        shape.isBlockLike ||
        Number(shape.bottomBand) >= 0.35
    );

    // A table can have a very sparse alpha fill because most of its bbox is
    // open space between the top and the supports. Accept only a bounded,
    // structurally plausible low-fill result; this exception is never used
    // for furniture, products, people, food, or the initial extraction pass.
    return (
        Number.isFinite(fill) && fill >= 0.08 && fill < 0.18 &&
        Number.isFinite(inside) && (
            inside >= 0.90 ||
            (spatialCompletionAnchor && inside >= 0.80)
        ) &&
        Number.isFinite(bboxOverlap) && bboxOverlap >= 0.70 &&
        Number.isFinite(touch) && touch <= 2 &&
        hasStructure &&
        (!Number.isFinite(components) || components <= 4) &&
        (!Number.isFinite(enclosedHoles) || !Number.isFinite(pixels) || enclosedHoles <= pixels * 0.35)
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
    const spatialTableLowFill = isSpatialTableCompletionQuality(quality, sourceLayer);

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

    let uniqueIssues = [...new Set(issues)];
    if (spatialTableLowFill) {
        uniqueIssues = uniqueIssues.filter(issue => ![
            'low_target_fill',
            'fragmented_table',
            'completion_scene_segmentation_fragmented_table',
            'low_quality_status'
        ].includes(issue));
        console.info('[segmentation-service] spatial table completion accepted by structural quality gate', {
            targetFillRatio,
            inside: quality?.debugCandidates?.find(candidate => candidate?.selected)?.inside ?? null,
            bboxOverlap: quality?.debugCandidates?.find(candidate => candidate?.selected)?.bboxOverlap ?? null,
            touchesTargetEdges: quality?.maskAudit?.touchesTargetEdges ?? null,
            selectedIndexes: quality?.selectedIndexes || []
        });
    }
    const shouldGenerateRuntimeLayer =
        spatialTableLowFill ||
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

export async function segmentLayers({ item, layers = [], onProgress = () => {}, engine = null, qualityProfile = 'publish' }) {
    const validLayers = layers.filter(layer => Array.isArray(layer?.bbox) && layer.bbox.length === 4);
    if (validLayers.length === 0) return new Map();

    const transport = await resolveSegmentationTransport();
    const requestedEngine = normalizeSegmentationEngine(engine || getSegmentationEngine());
    const backendMode = transport.mode === 'proxy'
        ? '云端后端'
        : isLocalBackendUrl(transport.backendUrl)
        ? '本地后端'
        : '云端后端';
    const engineLabel = requestedEngine === 'sam'
        ? `高精 SAM ${backendMode}`
        : `FastSAM ${backendMode}`;
    console.info('[segmentation-service] resolved transport', {
        mode: transport.mode,
        endpoint: transport.mode === 'proxy'
            ? transport.backendUrl
            : `${transport.backendUrl}/segment`
    });
    const imageDataUrl = await resolveImagePayload(item);
    const requestLayers = validLayers.map((layer, index) => {
        if (!layer.id) {
            layer.id = `segment-layer-${index}-${Date.now()}`;
        }
        return layer;
    });

    onProgress(`🧩 **${engineLabel}**: 正在提交 ${requestLayers.length} 个图层...`);

    let response;
    if (transport.mode === 'proxy') {
        response = await fetch(transport.backendUrl, {
            method: 'POST',
            headers: buildAuthorizedHeaders({
                'Content-Type': 'application/json'
            }),
            body: JSON.stringify({
                mode: SEGMENTATION_PROXY_MODE,
            payload: buildRequestPayload(imageDataUrl, requestLayers, item, requestedEngine, qualityProfile)
            })
        });
    } else {
        const headers = {
            'Content-Type': 'application/json'
        };
        if (transport.backendToken) {
            headers.Authorization = `Bearer ${transport.backendToken}`;
        }

        response = await fetch(`${transport.backendUrl}/segment`, {
            method: 'POST',
            headers,
            body: JSON.stringify(buildRequestPayload(imageDataUrl, requestLayers, item, requestedEngine, qualityProfile))
        });
    }

    const responseJson = await response.json();
    console.info('[segmentation-service] response summary', {
        requestedLayerIds: requestLayers.map(layer => layer.id),
        returnedLayerIds: Array.isArray(responseJson?.cutouts)
            ? responseJson.cutouts.map(cutout => cutout?.layerId || null)
            : [],
        success: responseJson?.success === true,
        httpStatus: response.status,
        qualityProfile
    });
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

export async function segmentSingleLayer({ item, layer, onProgress = () => {}, engine = null, qualityProfile = 'publish' }) {
    const requestLayer = {
        ...layer,
        id: layer.id || `segment-layer-single-${Date.now()}`
    };
    const resultMap = await segmentLayers({
        item,
        layers: [requestLayer],
        onProgress,
        engine,
        qualityProfile
    });
    const result = resultMap.get(requestLayer);
    if (!result?.dataUrl) {
        throw new Error(`分割引擎未返回单图层 cutout: ${requestLayer.name || requestLayer.id}`);
    }
    return result;
}
