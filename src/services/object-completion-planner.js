import { getProxiedUrl } from '../core/utils.js';
import { resolveCompletionTaskEntities } from './completion-contract.js';

const NORMALIZED_SCENE_SIZE = 1000;
const COMPLETION_KEY_COLOR = '#ff00ff';
const COMPLETION_KEY_COLOR_RGB = [255, 0, 255];

function cloneSerializable(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function isBbox(value) {
    return Array.isArray(value) && value.length === 4 && value.every(Number.isFinite);
}

function clamp(value, min = 0, max = NORMALIZED_SCENE_SIZE) {
    return Math.max(min, Math.min(max, value));
}

function expandBbox(bbox, paddingX, paddingY) {
    const [ymin, xmin, ymax, xmax] = bbox;
    return [
        clamp(ymin - paddingY),
        clamp(xmin - paddingX),
        clamp(ymax + paddingY),
        clamp(xmax + paddingX)
    ];
}

function ensureMinimumCropSize(bbox, minWidth = 180, minHeight = 180) {
    const [ymin, xmin, ymax, xmax] = bbox;
    const width = xmax - xmin;
    const height = ymax - ymin;
    const centerY = (ymin + ymax) / 2;
    const centerX = (xmin + xmax) / 2;
    const nextWidth = Math.min(NORMALIZED_SCENE_SIZE, Math.max(width, minWidth));
    const nextHeight = Math.min(NORMALIZED_SCENE_SIZE, Math.max(height, minHeight));
    let nextYmin = clamp(centerY - nextHeight / 2);
    let nextXmin = clamp(centerX - nextWidth / 2);
    let nextYmax = nextYmin + nextHeight;
    let nextXmax = nextXmin + nextWidth;

    if (nextYmax > NORMALIZED_SCENE_SIZE) {
        nextYmin = NORMALIZED_SCENE_SIZE - nextHeight;
        nextYmax = NORMALIZED_SCENE_SIZE;
    }
    if (nextXmax > NORMALIZED_SCENE_SIZE) {
        nextXmin = NORMALIZED_SCENE_SIZE - nextWidth;
        nextXmax = NORMALIZED_SCENE_SIZE;
    }
    return [nextYmin, nextXmin, nextYmax, nextXmax].map(value => Number(value.toFixed(3)));
}

function ensureSupportedCompletionAspectRatio(bbox, maxAspectRatio = 2.9) {
    const [ymin, xmin, ymax, xmax] = bbox;
    const width = Math.max(1, xmax - xmin);
    const height = Math.max(1, ymax - ymin);
    const ratio = width / height;
    if (ratio <= maxAspectRatio && ratio >= 1 / maxAspectRatio) return bbox;

    const centerY = (ymin + ymax) / 2;
    const centerX = (xmin + xmax) / 2;
    const requiredWidth = ratio < 1 / maxAspectRatio ? height * maxAspectRatio : width;
    const requiredHeight = ratio > maxAspectRatio ? width / maxAspectRatio : height;
    const nextWidth = Math.min(NORMALIZED_SCENE_SIZE, Math.max(width, requiredWidth));
    const nextHeight = Math.min(NORMALIZED_SCENE_SIZE, Math.max(height, requiredHeight));
    let nextYmin = clamp(centerY - nextHeight / 2);
    let nextXmin = clamp(centerX - nextWidth / 2);
    let nextYmax = nextYmin + nextHeight;
    let nextXmax = nextXmin + nextWidth;

    if (nextYmax > NORMALIZED_SCENE_SIZE) {
        nextYmin = NORMALIZED_SCENE_SIZE - nextHeight;
        nextYmax = NORMALIZED_SCENE_SIZE;
    }
    if (nextXmax > NORMALIZED_SCENE_SIZE) {
        nextXmin = NORMALIZED_SCENE_SIZE - nextWidth;
        nextXmax = NORMALIZED_SCENE_SIZE;
    }
    return [nextYmin, nextXmin, nextYmax, nextXmax].map(value => Number(value.toFixed(3)));
}

function getGraphLayer(graph, layerId) {
    return graph?.layers?.find(layer => layer.layerId === layerId) || null;
}

function bboxArea(bbox) {
    if (!isBbox(bbox)) return 0;
    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    return Math.max(0, ymax - ymin) * Math.max(0, xmax - xmin);
}

function calculateCompletionDifficulty(targetBbox, observedBbox, task, occluders) {
    const targetArea = Math.max(1, bboxArea(targetBbox));
    const observedArea = Math.min(targetArea, Math.max(0, bboxArea(observedBbox)));
    const visibleRatio = observedArea / targetArea;
    const occlusionRatio = Math.max(0, 1 - visibleRatio);
    const occluderCount = Array.isArray(occluders) ? occluders.length : 0;
    const semanticText = (occluders || [])
        .map(layer => [layer?.name, layer?.semanticType, layer?.extractionProfile].join(' '))
        .join(' ')
        .toLowerCase();
    const compoundOccluder = Boolean(
        (occluders || []).some(layer => (
            layer?.compositeRole === 'composite_group' ||
            (Array.isArray(layer?.childLayerIds) && layer.childLayerIds.length > 1)
        )) || /compound|组合|arrangement|group/.test(semanticText)
    );
    const structuralComplexity = compoundOccluder ? 1 : 0;
    const score = Math.min(
        1,
        (occlusionRatio * 0.62) +
        (Math.min(1, occluderCount / 3) * 0.20) +
        (structuralComplexity * 0.12) +
        (Number(task?.maxOverlapRatio) >= 0.35 ? 0.06 : 0)
    );
    const level = score >= 0.62 || visibleRatio < 0.42
        ? 'hard'
        : score >= 0.32 || visibleRatio < 0.68
        ? 'medium'
        : 'easy';
    return {
        level,
        score: Number(score.toFixed(4)),
        occlusionRatio: Number(occlusionRatio.toFixed(4)),
        visibleRatio: Number(visibleRatio.toFixed(4)),
        occluderCount,
        compoundOccluder,
        policy: 'advisory_until_evidence_gate'
    };
}

export function getCompletionSourceImage(item) {
    const candidates = [
        item?.originalDataUrl,
        item?.segmentationSourceUrl,
        item?.sourceImage,
        item?.dataUrl
    ];
    return candidates.find(source => typeof source === 'string' && source.length > 0) || '';
}

function getTaskLockedScene(item) {
    const sceneLock = item?.__magicLayersSceneLock;
    return sceneLock?.dataUrl && sceneLock?.id ? sceneLock : null;
}

export function buildObjectCompletionPreflight(item, asset, graph = item?.semanticViews?.layerGraph) {
    if (!asset || !graph) return null;
    const targetLayer = getGraphLayer(graph, asset.targetLayerId);
    const task = graph.completionTasks?.find(candidate => candidate.id === asset.taskId) || null;
    const observedBbox = asset.observedCutout?.bbox || targetLayer?.bbox;
    const completionBbox = targetLayer?.bbox || observedBbox;
    if (!isBbox(observedBbox) || !isBbox(completionBbox) || !task || !getCompletionSourceImage(item)) return null;

    const [ymin, xmin, ymax, xmax] = completionBbox;
    const targetWidth = xmax - xmin;
    const targetHeight = ymax - ymin;
    // GPT-image-2 rejects edit canvases wider or taller than 3:1. Keep a
    // margin under that limit by expanding the short side with scene context.
    // The target geometry itself remains unchanged.
    const cropBbox = ensureSupportedCompletionAspectRatio(
        ensureMinimumCropSize(
            expandBbox(completionBbox, targetWidth * 0.2, targetHeight * 0.2)
        )
    );
    const resolvedTask = resolveCompletionTaskEntities(task, graph.layers || []);
    task.occluderLayerIds = [...resolvedTask.occluderLayerIds];
    const occluders = resolvedTask.occluders;
    const difficulty = calculateCompletionDifficulty(
        completionBbox,
        observedBbox,
        task,
        occluders
    );

    const sceneLock = getTaskLockedScene(item);
    return {
        version: 'completion-preflight-v2',
        status: 'ready',
        source: {
            // The bitmap itself stays in an ephemeral task lock rather than in
            // persisted session JSON. All completion stages resolve this ID to
            // the exact same pixels during one Magic Layers run.
            kind: sceneLock ? 'task_locked_scene' : (item?.originalDataUrl ? 'original_image' : 'scene_source'),
            imageUrl: getCompletionSourceImage(item),
            taskLockId: sceneLock?.id || null,
            dimensions: sceneLock ? {
                width: sceneLock.width,
                height: sceneLock.height
            } : null,
            coordinateSpace: 'normalized_1000'
        },
        contract: {
            schemaVersion: 'completion-contract-v1',
            taskId: task.id,
            targetEntityId: resolvedTask.target.entityId,
            targetLayerId: task.targetLayerId,
            targetEntityLayerIds: [...resolvedTask.target.layerIds],
            occluderEntityIds: [...new Set(resolvedTask.occluders.map(layer =>
                layer.entityId || layer.semanticEntityId || layer.layerId
            ))],
            occluderLayerIds: [...resolvedTask.occluderLayerIds],
            targetObservedMask: asset.observedCutout?.cutoutUrl || targetLayer?.mask?.cutoutUrl || null,
            sourceSceneUrl: getCompletionSourceImage(item),
            zIndex: targetLayer?.zIndex ?? null,
            state: 'preflight_ready'
        },
        geometry: {
            cropBbox,
            targetVisibleBbox: cloneSerializable(observedBbox),
            targetCompletionBbox: cloneSerializable(completionBbox),
            missingRegionBbox: cloneSerializable(task.missingRegionBbox),
            occluderBboxes: occluders.map(layer => ({
                layerId: layer.layerId,
                bbox: cloneSerializable(layer.bbox)
            })),
            preserveScenePlacement: true,
            outputSizePolicy: 'preserve_crop_aspect_ratio',
            preferredLongEdge: 1024
        },
        difficulty,
        references: {
            observedCutoutUrl: asset.observedCutout?.cutoutUrl || null,
            observedMaskUrl: asset.observedCutout?.maskUrl || null,
            targetName: targetLayer?.name || asset.targetLayerId,
            targetSemanticType: targetLayer?.semanticType || 'unknown',
            occluderNames: occluders.map(layer => layer.name || layer.layerId),
            targetEntityLayerIds: resolvedTask.target.layerIds,
            occluderEntityLayerIds: resolvedTask.occluderLayerIds
        },
        prompt: {
            task: 'canonical_object_completion',
            text: [
                `Complete the partially occluded object "${targetLayer?.name || asset.targetLayerId}" into one complete standalone object.`,
                `The input contains only the partially visible target object on a uniform ${COMPLETION_KEY_COLOR} magenta background.`,
                'The edit mask marks the missing region only. Treat every unmasked visible target pixel as locked: do not repaint, move, resize, restyle, sharpen, or reinterpret it.',
                'Infer only the hidden parts that connect seamlessly to the locked target pixels, preserving the original pose, perspective, proportions, materials, lighting direction, and visible silhouette.',
                'Do not include the occluders, scene background, text, cards, or unrelated objects.',
                `Return exactly one complete target object on the same uniform ${COMPLETION_KEY_COLOR} magenta background. Do not crop the object and do not create a new scene.`
            ].join(' '),
            negative: 'Do not repaint visible object pixels. Do not add people, furniture, decorations, text, shadows detached from the object, or a new background.'
        },
        matte: {
            backdropStrategy: 'fixed_magenta_key',
            keyColor: COMPLETION_KEY_COLOR,
            keyColorRgb: COMPLETION_KEY_COLOR_RGB,
            required: true,
            preserveObservedPixels: true,
            samRefinement: 'quality_gate_only'
        },
        qualityGate: {
            version: 'canonical-asset-quality-v1',
            requireTransparentResult: true,
            minOpaqueRatio: 0.025,
            maxOpaqueRatio: 0.82,
            minTransparentBorderRatio: 0.9,
            requireObservedAssetReference: true,
            rejectOnSceneLikeOutput: true,
            routeAmbiguousResultTo: 'manual_review'
        },
        createdAt: Date.now()
    };
}

export function prepareCompletionAssetPreflight(item, assetId) {
    const asset = item?.semanticViews?.completionAssets?.find(candidate => candidate.id === assetId);
    if (!asset) return null;
    const preflight = buildObjectCompletionPreflight(item, asset);
    if (!preflight) {
        asset.generation = {
            ...(asset.generation || {}),
            state: 'blocked',
            lastError: 'completion_source_or_geometry_unavailable'
        };
        asset.updatedAt = Date.now();
        return null;
    }
    asset.preflight = preflight;
    if (asset.canonicalAsset?.status === 'ready') {
        asset.updatedAt = Date.now();
        return preflight;
    }
    asset.generation = {
        ...(asset.generation || {}),
        state: 'preflight_ready',
        lastError: null
    };
    asset.updatedAt = Date.now();
    return preflight;
}

function loadImage(source) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error('completion_source_load_failed'));
        image.src = getProxiedUrl(source);
    });
}

// This creates transient model input only. Callers must not persist the data URL
// into the completion contract or the session state.
export async function materializeCompletionInputCrop(item, assetId) {
    if (typeof document === 'undefined' || typeof Image === 'undefined') {
        throw new Error('completion_crop_requires_browser');
    }
    const asset = item?.semanticViews?.completionAssets?.find(candidate => candidate.id === assetId);
    const preflight = asset?.preflight || prepareCompletionAssetPreflight(item, assetId);
    if (!asset || !preflight) throw new Error('completion_preflight_unavailable');

    const observedCutoutUrl = asset.observedCutout?.cutoutUrl || preflight.references?.observedCutoutUrl;
    if (!observedCutoutUrl) throw new Error('completion_observed_cutout_unavailable');

    const image = await loadImage(observedCutoutUrl);
    const [ymin, xmin, ymax, xmax] = preflight.geometry.cropBbox;
    const [observedYmin, observedXmin, observedYmax, observedXmax] =
        (asset.observedCutout?.bbox || preflight.geometry.targetVisibleBbox).map(Number);
    const cropWidth = Math.max(1, xmax - xmin);
    const cropHeight = Math.max(1, ymax - ymin);
    const cropAspectRatio = cropWidth / cropHeight;
    const longEdge = 1024;
    const width = cropAspectRatio >= 1
        ? longEdge
        : Math.max(1, Math.round(longEdge * cropAspectRatio));
    const height = cropAspectRatio >= 1
        ? Math.max(1, Math.round(longEdge / cropAspectRatio))
        : longEdge;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.fillStyle = COMPLETION_KEY_COLOR;
    context.fillRect(0, 0, width, height);

    // The model sees no scene or foreground occluders. The source cutout is
    // projected into the same crop coordinates used later for in-place output.
    const observedLeft = ((observedXmin - xmin) / cropWidth) * width;
    const observedTop = ((observedYmin - ymin) / cropHeight) * height;
    const observedWidth = ((observedXmax - observedXmin) / cropWidth) * width;
    const observedHeight = ((observedYmax - observedYmin) / cropHeight) * height;
    if (observedWidth <= 0 || observedHeight <= 0) throw new Error('completion_observed_cutout_geometry_invalid');
    context.drawImage(image, observedLeft, observedTop, observedWidth, observedHeight);

    // White marks the editable area for every supported image-edit adapter.
    // Keep pixels covered by the observed cutout black so the model only
    // infers the missing object regions around it.
    const editMaskCanvas = document.createElement('canvas');
    editMaskCanvas.width = width;
    editMaskCanvas.height = height;
    const editMaskContext = editMaskCanvas.getContext('2d');
    editMaskContext.fillStyle = '#ffffff';
    editMaskContext.fillRect(0, 0, width, height);
    editMaskContext.save();
    editMaskContext.globalCompositeOperation = 'destination-out';
    editMaskContext.drawImage(image, observedLeft, observedTop, observedWidth, observedHeight);
    editMaskContext.restore();

    console.info('[Object Completion] isolated input', {
        targetLayerId: asset.targetLayerId,
        cropBbox: preflight.geometry.cropBbox,
        observedBbox: [observedYmin, observedXmin, observedYmax, observedXmax],
        canvas: { width, height },
        keyColor: COMPLETION_KEY_COLOR,
        editPolicy: 'missing_regions_only'
    });

    return {
        preflight: cloneSerializable(preflight),
        referenceCropDataUrl: canvas.toDataURL('image/png'),
        editMaskDataUrl: editMaskCanvas.toDataURL('image/png'),
        observedCutoutUrl,
        sourceSize: { width: image.naturalWidth || image.width, height: image.naturalHeight || image.height },
        cropSize: { width, height }
    };
}

export function evaluateCanonicalAssetQuality(metrics = {}, qualityGate = {}) {
    const gate = qualityGate || {};
    const opaqueRatio = Number(metrics.opaqueRatio);
    const transparentBorderRatio = Number(metrics.transparentBorderRatio);
    const reasons = [];

    if (gate.requireTransparentResult && metrics.hasTransparency !== true) reasons.push('missing_transparency');
    if (!Number.isFinite(opaqueRatio) || opaqueRatio < Number(gate.minOpaqueRatio ?? 0)) reasons.push('subject_too_small');
    if (Number.isFinite(opaqueRatio) && opaqueRatio > Number(gate.maxOpaqueRatio ?? 1)) reasons.push('scene_like_opaque_coverage');
    if (!Number.isFinite(transparentBorderRatio) || transparentBorderRatio < Number(gate.minTransparentBorderRatio ?? 0)) {
        reasons.push('insufficient_transparent_border');
    }

    return {
        accepted: reasons.length === 0,
        status: reasons.length === 0 ? 'accepted' : 'manual_review',
        reasons,
        metrics: cloneSerializable(metrics)
    };
}
