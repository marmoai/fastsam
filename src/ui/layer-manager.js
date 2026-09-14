import { state } from '../core/state.js';
import { runtime } from '../runtime/CoreRuntime';
import { addMessage } from './chat-panel.js';
import { 
    analyzeImageLayers, 
    editOrQueryImageWithGemini, 
    generatePreciseEditImage,
    extractTextFromImage
} from '../ai-services/skills-engine.js';
import { 
    dataURLToFile, 
    fileToBase64, 
    isRemovalRequest, 
    isMaterialRequest, 
    getImageDimensions,
    getClosestSupportedAspectRatio,
    getProxiedUrl
} from '../core/utils.js';
import { 
    addImageToWorkbench, 
    selectWorkbenchItem, 
    calculateSmartPosition,
    waitForWorkbenchItemPersistence
} from './workbench-core.js';
import { renderLayerList, getLayerState, updateLayerState, showLayerManagerModal, updateFusionUI } from './modals.js';

import { renderCanvasLayers } from './workbench/layers.js';
import { renderSceneToCanvas } from './workbench/renderer.js';
import { dilateAlphaChannel } from '../graphics/mask-utils.js';
import { cleanBackground, cleanMultipleBackgrounds } from './workbench/layer-assets.js';
import { globalMatteTaskSystem } from '../graphics/matte-task-system.js';
import { StrategyDispatcher } from '../ai-services/strategy-dispatcher.js';
import { buildSemanticLayerViews, applySemanticLayerViewsToItem, getBackgroundSemanticHint, getCleanupLayerForEditableLayer, updateLayerExtractionMetadata, validateCompletionCandidatesWithMasks } from '../services/semantic-layer-views.js';
import { executeObjectCompletion } from '../services/object-completion-executor.js';
import { segmentLayers, segmentSingleLayer } from '../services/segmentation-service.js';
import { buildExtractedTextState } from './text-style-utils.js';
import { filterTextLinesToBbox, getCachedTextExtraction, normalizeOcrTextLines } from '../services/text-extraction-cache.js';
import { recordWorkspaceAction } from '../services/workspace-context.js';
import { uploadImageToOSS } from '../services/ossService.js';
import { prepareTextContainerCandidates, restoreTextContainerShapes } from './text-container-restore.js';
import { ensureLayerPanelFooter } from './modals.js';
import { beginDiagnosticTask, updateDiagnosticTask, finishDiagnosticTask, recordDiagnosticBreadcrumb } from '../core/crash-diagnostics.js';
import { isAgentRuntimeFeatureEnabled } from '../core/config.js';
import { startLayerExtractionJob } from './agent-task-controller.js';
import { expandEntityLayerIds } from '../services/completion-contract.js';
import { isCanonicalEvidenceAccepted } from '../services/completion-evidence.js';

const DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST = true;
// Agent extraction stages Magic Layers as one transaction. Normal UI calls
// keep their existing persistence/history behavior.
let activeAgentLayerExtraction = 0;
const agentLayerExtractionOperations = new Map();
const agentCapabilityOperations = new Map();

function captureAgentLayerRuntimeMetadata(asset) {
    if (!asset) return null;
    return {
        layers: asset.layers,
        scene: asset.scene,
        semanticViews: asset.semanticViews,
        hasFullSemanticAnalysis: asset.hasFullSemanticAnalysis,
        originalDataUrl: asset.originalDataUrl,
        cleanPlateDataUrl: asset.cleanPlateDataUrl,
        cleanPlateStatus: asset.cleanPlateStatus
    };
}

function captureAgentLayerLegacyMetadata(item) {
    return {
        dataUrl: item?.dataUrl,
        originalDataUrl: item?.originalDataUrl,
        cleanPlateDataUrl: item?.cleanPlateDataUrl,
        cleanPlateStatus: item?.cleanPlateStatus
    };
}

function cloneSerializable(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function yieldToBrowser() {
    return new Promise(resolve => {
        if (typeof requestIdleCallback === 'function') {
            requestIdleCallback(() => resolve(), { timeout: 100 });
        } else {
            setTimeout(resolve, 0);
        }
    });
}

// Image extraction creates several decoded canvases and can briefly hold both
// the source and generated PNG in memory. Keep only one fallback extraction
// alive at a time so a large Magic Layers batch cannot exhaust the renderer.
async function runLimitedExtractionJobs(jobs, concurrency = 1) {
    const results = new Array(jobs.length);
    let nextIndex = 0;
    const worker = async () => {
        while (nextIndex < jobs.length) {
            const index = nextIndex++;
            results[index] = await jobs[index]();
            await yieldToBrowser();
        }
    };
    const workerCount = Math.min(Math.max(1, concurrency), jobs.length);
    await Promise.all(Array.from({ length: workerCount }, () => worker()));
    return results;
}

function loadImageForFusion(source) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error('融合图片加载失败'));
        image.src = source;
    });
}

// Keep the source scene immutable and accept generated pixels only inside the
// target mask. This also restores the exact source canvas dimensions.
async function compositeMaskedFusionResult(baseDataUrl, generatedDataUrl, maskDataUrl) {
    const [baseImage, generatedImage, maskImage] = await Promise.all([
        loadImageForFusion(baseDataUrl),
        loadImageForFusion(generatedDataUrl),
        loadImageForFusion(maskDataUrl)
    ]);

    const canvas = document.createElement('canvas');
    canvas.width = baseImage.naturalWidth;
    canvas.height = baseImage.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);

    const generatedLayer = document.createElement('canvas');
    generatedLayer.width = canvas.width;
    generatedLayer.height = canvas.height;
    const generatedCtx = generatedLayer.getContext('2d');
    generatedCtx.imageSmoothingEnabled = true;
    generatedCtx.imageSmoothingQuality = 'high';
    generatedCtx.drawImage(generatedImage, 0, 0, canvas.width, canvas.height);
    generatedCtx.globalCompositeOperation = 'destination-in';
    generatedCtx.drawImage(maskImage, 0, 0, canvas.width, canvas.height);

    ctx.drawImage(generatedLayer, 0, 0);
    return canvas.toDataURL('image/png');
}

function getItemSceneLayers(item) {
    return item?.scene?.layers || item?.layers || [];
}

function createMagicLayersSceneLock(item, sourceImage, sourceUrl) {
    const width = Math.max(1, sourceImage.naturalWidth || sourceImage.width);
    const height = Math.max(1, sourceImage.naturalHeight || sourceImage.height);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').drawImage(sourceImage, 0, 0, width, height);
    const sceneLock = {
        id: `magic-layers-scene-${Date.now()}`,
        dataUrl: canvas.toDataURL('image/png'),
        width,
        height,
        sourceUrl,
        createdAt: Date.now()
    };
    // This bitmap is deliberately process-local. Persisting it would duplicate
    // an entire original scene into every session JSON snapshot.
    Object.defineProperty(item, '__magicLayersSceneLock', {
        value: sceneLock,
        configurable: true,
        writable: true,
        enumerable: false
    });
    console.info('[Magic Layers] locked scene source', {
        taskLockId: sceneLock.id,
        dimensions: `${width}x${height}`,
        source: item.originalDataUrl ? 'originalDataUrl' : 'dataUrl'
    });
    return sceneLock;
}

function createLockedSceneSegmentationItem(item, sceneLock) {
    return {
        ...item,
        // Initial SAM must not silently prefer a stale clean plate over the
        // scene captured for this Magic Layers transaction.
        cleanPlateDataUrl: null,
        segmentationSourceUrl: sceneLock.dataUrl,
        originalDataUrl: sceneLock.dataUrl,
        dataUrl: sceneLock.dataUrl
    };
}

function releaseMagicLayersSceneLock(item, parentImage = null) {
    const sceneLock = item?.__magicLayersSceneLock;
    if (sceneLock) {
        // The lock is transaction-local. Keep the durable originalDataUrl, but
        // release the duplicate encoded scene and its decoded image reference.
        sceneLock.dataUrl = null;
        delete item.__magicLayersSceneLock;
    }
    if (parentImage) parentImage.src = '';
}

function getDeferredCompletionTargetIds(item, selectedLayers = []) {
    const graph = item?.semanticViews?.layerGraph;
    if (!graph?.completionTasks?.length) return new Set();
    const selectedById = new Map(selectedLayers.map(layer => [layer.id, layer]));
    const selectedEntityLayerIds = new Set(expandEntityLayerIds(
        graph.layers || [],
        selectedLayers.map(layer => layer.id)
    ));
    const deferred = graph.completionTasks
        .filter(task => {
            if (task?.eligibility !== 'auto' || !selectedById.has(task.targetLayerId)) return false;
            // Deferred scene completion is only for an actual occluded region.
            // Bbox near-contact can describe a legitimate layout adjacency
            // (for example wall art above a console), not missing geometry.
            if (!Array.isArray(task.missingRegionBbox) || task.missingRegionBbox.length !== 4) return false;
            // A semantic graph can include decor or an unselected sibling as a
            // possible occluder. Do not force a defective target extraction
            // merely because one such relation has no mask. One verified,
            // selected foreground hard entity is sufficient to defer it.
            return Array.isArray(task.occluderLayerIds) && task.occluderLayerIds.some(layerId => {
                const occluder = selectedById.get(layerId) || graph.layers?.find(layer => layer.layerId === layerId);
                return selectedEntityLayerIds.has(String(layerId)) &&
                    occluder && isRasterSegmentationRuntimeLayer(occluder);
            });
        })
        .map(task => task.targetLayerId);
    return new Set(deferred);
}

const GENERATIVE_BACKDROPS = [
    { name: 'pure, solid green', hex: '#00FF00', rgb: [0, 255, 0], rule: 'Use pure green (#00FF00) ONLY.' },
    { name: 'pure, solid magenta', hex: '#FF00FF', rgb: [255, 0, 255], rule: 'Use pure magenta (#FF00FF) ONLY.' },
    { name: 'pure, solid blue', hex: '#0000FF', rgb: [0, 0, 255], rule: 'Use pure blue (#0000FF) ONLY.' },
    { name: 'pure, solid red', hex: '#FF0000', rgb: [255, 0, 0], rule: 'Use pure red (#FF0000) ONLY.' }
];

function selectAdaptiveBackdrop(imageData, layerName = '') {
    const { width, height, data } = imageData || {};
    const visibleColors = [];
    if (width && height && data) {
        for (let index = 0; index < data.length; index += 16) {
            if (data[index + 3] >= 32) {
                visibleColors.push([data[index], data[index + 1], data[index + 2]]);
            }
        }
    }

    const withSampleCount = (backdrop) => ({
        ...backdrop,
        sampleCount: visibleColors.length
    });

    if (visibleColors.length === 0) {
        const name = String(layerName).toLowerCase();
        if (/绿|green|草|叶|树/.test(name)) return withSampleCount(GENERATIVE_BACKDROPS[1]);
        if (/蓝|blue|天空|海/.test(name)) return withSampleCount(GENERATIVE_BACKDROPS[3]);
        if (/红|red|橙|orange/.test(name)) return withSampleCount(GENERATIVE_BACKDROPS[2]);
        return withSampleCount(GENERATIVE_BACKDROPS[0]);
    }

    const scored = GENERATIVE_BACKDROPS.map(backdrop => {
        let distanceSum = 0;
        let conflictCount = 0;
        visibleColors.forEach(([r, g, b]) => {
            const distance = Math.sqrt(
                (r - backdrop.rgb[0]) ** 2 +
                (g - backdrop.rgb[1]) ** 2 +
                (b - backdrop.rgb[2]) ** 2
            );
            distanceSum += distance;
            if (distance < 100) conflictCount += 1;
        });
        return {
            backdrop,
            score: distanceSum / visibleColors.length - (conflictCount / visibleColors.length) * 180
        };
    });

    scored.sort((left, right) => right.score - left.score);
    return withSampleCount(scored[0].backdrop);
}

function sampleGeneratedBackdrop(imageData, fallbackRgb) {
    const { width, height, data } = imageData || {};
    const fallback = Array.isArray(fallbackRgb) ? fallbackRgb.map(value => Number(value) || 0) : [0, 255, 0];
    if (!width || !height || !data) {
        return { rgb: fallback, sampledRgb: fallback, stable: false, sampleCount: 0, spread: 0, source: 'requested' };
    }

    const samples = [];
    const border = Math.max(2, Math.round(Math.min(width, height) * 0.04));
    const step = Math.max(1, Math.floor(Math.min(width, height) / 256));
    for (let y = 0; y < height; y += step) {
        for (let x = 0; x < width; x += step) {
            if (x >= border && x < width - border && y >= border && y < height - border) continue;
            const index = (y * width + x) * 4;
            if (data[index + 3] < 220) continue;
            samples.push([data[index], data[index + 1], data[index + 2]]);
        }
    }

    if (samples.length < 12) {
        return { rgb: fallback, sampledRgb: fallback, stable: false, sampleCount: samples.length, spread: 0, source: 'requested' };
    }

    const median = channel => {
        const values = samples.map(sample => sample[channel]).sort((a, b) => a - b);
        return values[Math.floor(values.length / 2)];
    };
    const sampledRgb = [median(0), median(1), median(2)];
    const percentile = (channel, ratio) => {
        const values = samples.map(sample => sample[channel]).sort((a, b) => a - b);
        return values[Math.min(values.length - 1, Math.floor(values.length * ratio))];
    };
    const spread = Math.max(
        percentile(0, 0.9) - percentile(0, 0.1),
        percentile(1, 0.9) - percentile(1, 0.1),
        percentile(2, 0.9) - percentile(2, 0.1)
    );
    const stable = spread <= 90;

    return {
        rgb: stable ? sampledRgb : fallback,
        sampledRgb,
        stable,
        sampleCount: samples.length,
        spread,
        source: stable ? 'generated_border' : 'requested_unstable_border'
    };
}

function isSplitLayerAsset(item) {
    return !!(
        item &&
        item.parentId &&
        Array.isArray(item.originalBbox) &&
        item.originalBbox.length === 4 &&
        ['layer-explode', 'layer-extract', 'isolated-edit', 'extraction'].includes(String(item.type || '').toLowerCase())
    );
}

function hasSplitLayerPlacement(item) {
    return !!(
        item &&
        item.parentId &&
        Array.isArray(item.originalBbox) &&
        item.originalBbox.length === 4
    );
}

function resolveParentLayerForAsset(parentItem, childItem) {
    const layers = getItemSceneLayers(parentItem);
    if (!Array.isArray(layers) || layers.length === 0) return null;

    const childLayerId = childItem.sourceLayerId || childItem.layerId || null;
    if (childLayerId) {
        const byId = layers.find(layer => layer?.id === childLayerId);
        if (byId) return byId;
    }

    const childName = String(childItem.layerName || '').trim();
    if (childName) {
        const byName = layers.find(layer => String(layer?.name || '').trim() === childName);
        if (byName) return byName;
    }

    const childBbox = childItem.originalBbox;
    return layers.find(layer => Array.isArray(layer?.bbox) &&
        layer.bbox.length === 4 &&
        layer.bbox.every((value, index) => Math.abs(Number(value) - Number(childBbox[index])) <= 2)
    ) || null;
}

function drawImageCoveringTarget(ctx, image, targetWidth, targetHeight) {
    const sourceWidth = image.naturalWidth || image.width;
    const sourceHeight = image.naturalHeight || image.height;
    if (!sourceWidth || !sourceHeight) return;

    // Cover the target without changing the source aspect ratio. Negative
    // offsets crop only the excess canvas/transparent margins.
    const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
    const drawWidth = sourceWidth * scale;
    const drawHeight = sourceHeight * scale;
    const offsetX = (targetWidth - drawWidth) / 2;
    const offsetY = (targetHeight - drawHeight) / 2;
    ctx.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
}

async function fitAssetToCanvas(dataUrl, width, height) {
    const image = await loadImageForFusion(dataUrl);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width));
    canvas.height = Math.max(1, Math.round(height));
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    drawImageCoveringTarget(ctx, image, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
}

function getAlphaBottom(image, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width));
    canvas.height = Math.max(1, Math.round(height));
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let y = canvas.height - 1; y >= 0; y -= 1) {
        for (let x = 0; x < canvas.width; x += 1) {
            if (pixels[(y * canvas.width + x) * 4 + 3] > 16) return y + 1;
        }
    }
    return 0;
}

async function alignAssetBottomToSource(dataUrl, sourceImage, width, height) {
    const editedImage = await loadImageForFusion(dataUrl);
    const targetWidth = Math.max(1, Math.round(width));
    const targetHeight = Math.max(1, Math.round(height));
    const sourceBottom = getAlphaBottom(sourceImage, targetWidth, targetHeight);
    const editedBottom = getAlphaBottom(editedImage, targetWidth, targetHeight);
    if (!sourceBottom || !editedBottom || sourceBottom === editedBottom) return dataUrl;

    const deltaY = sourceBottom - editedBottom;
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(editedImage, 0, deltaY, targetWidth, targetHeight);
    return canvas.toDataURL('image/png');
}

async function inspectAssetAlpha(dataUrl) {
    const image = await loadImageForFusion(dataUrl);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, image.naturalWidth);
    canvas.height = Math.max(1, image.naturalHeight);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let transparentPixels = 0;
    let visiblePixels = 0;
    let partialAlphaPixels = 0;
    let edgeTransitionPixels = 0;
    let borderPixels = 0;
    let transparentBorderPixels = 0;
    const borderWidth = Math.max(1, Math.round(Math.min(canvas.width, canvas.height) * 0.03));
    for (let index = 3; index < pixels.length; index += 4) {
        const alpha = pixels[index];
        if (alpha <= 8) transparentPixels += 1;
        if (alpha >= 32) visiblePixels += 1;
        if (alpha > 8 && alpha < 245) partialAlphaPixels += 1;

        const pixelIndex = Math.floor((index - 3) / 4);
        const x = pixelIndex % canvas.width;
        const y = Math.floor(pixelIndex / canvas.width);
        const isBorder = x < borderWidth || y < borderWidth ||
            x >= canvas.width - borderWidth || y >= canvas.height - borderWidth;
        if (isBorder) {
            borderPixels += 1;
            if (alpha <= 8) transparentBorderPixels += 1;
        }

        const state = alpha < 32 ? 0 : 1;
        const neighbors = [];
        if (x > 0) neighbors.push(pixels[index - 4]);
        if (x + 1 < canvas.width) neighbors.push(pixels[index + 4]);
        if (y > 0) neighbors.push(pixels[index - canvas.width * 4]);
        if (y + 1 < canvas.height) neighbors.push(pixels[index + canvas.width * 4]);
        if (neighbors.some(neighborAlpha => (neighborAlpha < 32 ? 0 : 1) !== state)) {
            edgeTransitionPixels += 1;
        }
    }

    const totalPixels = canvas.width * canvas.height;
    const transparentRatio = totalPixels ? transparentPixels / totalPixels : 0;
    const visibleRatio = totalPixels ? visiblePixels / totalPixels : 0;
    const partialAlphaRatio = totalPixels ? partialAlphaPixels / totalPixels : 0;
    const edgeTransitionRatio = totalPixels ? edgeTransitionPixels / totalPixels : 0;
    const transparentBorderRatio = borderPixels ? transparentBorderPixels / borderPixels : 0;
    const hasTransparency = transparentRatio >= 0.01 || transparentBorderRatio >= 0.04;
    const hasForeground = visibleRatio >= 0.05;
    const hasUsableEdge = partialAlphaRatio >= 0.0005 || edgeTransitionRatio >= 0.001;
    const isNearlyOpaque = transparentRatio < 0.005 && partialAlphaRatio < 0.002;

    return {
        width: canvas.width,
        height: canvas.height,
        transparentRatio,
        visibleRatio,
        partialAlphaRatio,
        edgeTransitionRatio,
        transparentBorderRatio,
        hasReliableAlpha: hasTransparency && hasForeground && hasUsableEdge && !isNearlyOpaque
    };
}

function despillMatteImageData(imageData, backgroundRgb) {
    const { width, height, data } = imageData;
    const [bgR, bgG, bgB] = backgroundRgb || [0, 255, 0];
    let edgePixels = 0;
    let correctedPixels = 0;
    let spillStrength = 0;
    let unmixedPixels = 0;
    let unmixStrengthSum = 0;

    const alphaAt = (x, y) => {
        if (x < 0 || y < 0 || x >= width || y >= height) return 0;
        return data[(y * width + x) * 4 + 3];
    };

    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            const index = (y * width + x) * 4;
            const alpha = data[index + 3];
            if (alpha < 8) continue;

            const edge = alpha < 245 ||
                alphaAt(x - 1, y) < 32 || alphaAt(x + 1, y) < 32 ||
                alphaAt(x, y - 1) < 32 || alphaAt(x, y + 1) < 32;
            if (!edge) continue;
            edgePixels += 1;

            const r = data[index];
            const g = data[index + 1];
            const b = data[index + 2];
            const edgeFactor = Math.max(0.15, Math.min(1, (255 - alpha) / 180));
            let contamination = 0;
            let channels = [];

            if (bgG > 180 && bgG > bgR + 80 && bgG > bgB + 80 && g > Math.max(r, b) + 6) {
                contamination = g - Math.max(r, b);
                channels = [1];
            } else if (bgR > 180 && bgB > 180 && bgG < 100 && r > g + 6 && b > g + 6) {
                contamination = Math.min(r, b) - g;
                channels = [0, 2];
            } else if (bgB > 180 && bgB > bgR + 80 && bgB > bgG + 80 && b > Math.max(r, g) + 6) {
                contamination = b - Math.max(r, g);
                channels = [2];
            } else if (bgR > 180 && bgR > bgG + 80 && bgR > bgB + 80 && r > Math.max(g, b) + 6) {
                contamination = r - Math.max(g, b);
                channels = [0];
            }

            if (channels.length > 0 && contamination > 0) {
                const correction = Math.min(contamination, contamination * edgeFactor * 0.9);
                channels.forEach(channel => {
                    data[index + channel] = Math.max(0, Math.round(data[index + channel] - correction));
                });
                correctedPixels += 1;
                spillStrength += contamination;
            }

            // Recover the foreground contribution from semi-transparent edge
            // pixels instead of only subtracting the dominant key channel.
            // Limit the solve to pixels close to the measured background so
            // saturated foreground colors are not globally recolored.
            const alphaNorm = alpha / 255;
            if (alphaNorm > 0.12 && alphaNorm < 0.96) {
                const bgDistance = Math.sqrt(
                    (r - bgR) ** 2 +
                    (g - bgG) ** 2 +
                    (b - bgB) ** 2
                );
                const blendConfidence = Math.max(0.15, Math.min(1, 1 - bgDistance / 320));
                const solveAlpha = Math.max(0.28, alphaNorm);
                const solveStrength = Math.max(0.08, Math.min(0.82, (1 - alphaNorm) * 1.35)) * blendConfidence;
                const current = [data[index], data[index + 1], data[index + 2]];
                const background = [bgR, bgG, bgB];
                let changed = false;

                for (let channel = 0; channel < 3; channel += 1) {
                    const estimate = Math.max(0, Math.min(255,
                        (current[channel] - background[channel] * (1 - alphaNorm)) / solveAlpha
                    ));
                    const fromBackground = current[channel] - background[channel];
                    const towardForeground = estimate - current[channel];
                    if (fromBackground * towardForeground <= 0) continue;
                    const value = current[channel] + towardForeground * solveStrength;
                    data[index + channel] = Math.max(0, Math.min(255, Math.round(value)));
                    changed = true;
                }

                if (changed) {
                    unmixedPixels += 1;
                    unmixStrengthSum += solveStrength;
                }
            }
        }
    }

    return {
        edgePixels,
        correctedPixels,
        averageSpill: correctedPixels ? spillStrength / correctedPixels : 0,
        unmixedPixels,
        averageUnmixStrength: unmixedPixels ? unmixStrengthSum / unmixedPixels : 0
    };
}

async function tryRefineEditedAssetWithSam(dataUrl, childItem, width, height, tempMsg) {
    try {
        const alphaInfo = await inspectAssetAlpha(dataUrl);
        if (alphaInfo.hasReliableAlpha) {
            console.info('[Layer Asset Edit] skipping SAM refinement: generated asset already has alpha', {
                childId: childItem.id || null,
                width: alphaInfo.width,
                height: alphaInfo.height,
                transparentRatio: Number(alphaInfo.transparentRatio.toFixed(4)),
                visibleRatio: Number(alphaInfo.visibleRatio.toFixed(4)),
                partialAlphaRatio: Number(alphaInfo.partialAlphaRatio.toFixed(4)),
                edgeTransitionRatio: Number(alphaInfo.edgeTransitionRatio.toFixed(4)),
                transparentBorderRatio: Number(alphaInfo.transparentBorderRatio.toFixed(4))
            });
            return await fitAssetToCanvas(dataUrl, width, height);
        }

        console.info('[Layer Asset Edit] SAM refinement required: generated asset has no reliable alpha', {
            childId: childItem.id || null,
            width: alphaInfo.width,
            height: alphaInfo.height,
            transparentRatio: Number(alphaInfo.transparentRatio.toFixed(4)),
            visibleRatio: Number(alphaInfo.visibleRatio.toFixed(4)),
            partialAlphaRatio: Number(alphaInfo.partialAlphaRatio.toFixed(4)),
            edgeTransitionRatio: Number(alphaInfo.edgeTransitionRatio.toFixed(4)),
            transparentBorderRatio: Number(alphaInfo.transparentBorderRatio.toFixed(4))
        });
        tempMsg?.update?.('🧩 **正在恢复透明边界**: 用 SAM 检查新资产轮廓...');
        const segmentation = await segmentSingleLayer({
            item: {
                dataUrl,
                originalDataUrl: dataUrl,
                segmentationSourceUrl: dataUrl
            },
                layer: {
                    id: `asset-edit-${childItem.id || Date.now()}`,
                    name: childItem.layerName || 'edited asset',
                    semanticType: childItem.semanticType || childItem.category || 'standalone_asset',
                    extractionProfile: childItem.extractionProfile || '',
                    bbox: [0, 0, alphaInfo.width, alphaInfo.height]
                },
            onProgress: (message) => tempMsg?.update?.(message),
            qualityProfile: 'completion_review'
        });

        if (!segmentation?.dataUrl) return null;
        if (segmentation.quality?.runtimeAction === 'hold' || segmentation.quality?.shouldGenerateRuntimeLayer === false) {
            console.warn('[Layer Asset Edit] SAM quality gate held the generated asset; keeping matte result.');
            return null;
        }
        return await fitAssetToCanvas(segmentation.dataUrl, width, height);
    } catch (error) {
        // SAM is a refinement step, not a reason to lose a successful edit.
        console.warn('[Layer Asset Edit] SAM refinement unavailable; keeping matte result:', error);
        return null;
    }
}

async function replaceExistingLayerAsset(childId, childItem, parentItemId, parentItem, dataUrl, prompt, options = {}) {
    const parentLayer = resolveParentLayerForAsset(parentItem, childItem);
    if (!parentLayer) {
        throw new Error(`未找到“${childItem.layerName || '当前图层'}”对应的父图层`);
    }

    const assetFile = await dataURLToFile(dataUrl, `layer-version-${Date.now()}.png`);
    let storedUrl = dataUrl;
    try {
        storedUrl = await uploadImageToOSS(assetFile);
    } catch (error) {
        console.warn('[Layer Asset Edit] OSS upload failed; keeping local asset:', error);
    }

    if (!Array.isArray(parentLayer.versions)) parentLayer.versions = [];
    if (parentLayer.versions.length === 0) {
        parentLayer.versions.push({
            id: 'base',
            cutoutUrl: parentLayer.cutoutUrl || childItem.dataUrl || null,
            maskUrl: parentLayer.maskUrl || null,
            previewUrl: parentLayer.previewUrl || parentLayer.cutoutUrl || childItem.dataUrl || null,
            prompt: '原始提取',
            createdAt: Date.now(),
            source: 'extraction'
        });
    }

    const versionId = `v${Date.now()}`;
    const version = {
        id: versionId,
        cutoutUrl: storedUrl,
        maskUrl: null,
        previewUrl: storedUrl,
        prompt,
        createdAt: Date.now(),
        source: 'asset_sync_edit',
        sourceChildId: childId || null
    };
    parentLayer.versions.push(version);
    parentLayer.activeVersionId = versionId;
    parentLayer.cutoutUrl = storedUrl;
    parentLayer.previewUrl = storedUrl;
    parentLayer.assetStatus = 'ready';

    // The visible Workbench child is the rendered instance of this semantic layer.
    // Update it in place so its position, size, rotation, and z-index stay intact.
    childItem.dataUrl = storedUrl;
    childItem.originalDataUrl = storedUrl;
    childItem.previewUrl = storedUrl;
    childItem.file = assetFile;
    childItem.assetStatus = 'ready';
    childItem.activeVersionId = versionId;
    childItem.versions = parentLayer.versions;
    if (childItem.el) {
        const imageElement = childItem.el.querySelector('.crop-container > img') || childItem.el.querySelector('img');
        if (imageElement) imageElement.src = dataUrl;
    }

    const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
    const childAsset = workspace?.currentState.assetRegistry.get(childId);
    if (workspace && childAsset) {
        workspace.dispatcher.dispatch({
            type: 'UPDATE_ASSET_METADATA',
            // Metadata persistence is already followed by an explicit session
            // save. Do not create a full undo snapshot containing every PNG.
            meta: { silent: true },
            payload: {
                uid: childId,
                sourceImage: storedUrl,
                originalDataUrl: storedUrl,
                version: versionId
            }
        });
    }

    const parentLayers = getItemSceneLayers(parentItem);
    const parentLayerIndex = parentLayers.indexOf(parentLayer);
    if (parentLayerIndex >= 0) {
        renderLayerList(parentLayers, parentItemId);
    }
    renderCanvasLayers(parentItemId);
    if (options.persist !== false) {
        await persistLayerStateToRuntime(parentItemId, parentItem);
    }
    if (options.recordHistory !== false && window.historyManager) window.historyManager.pushState();

    if (options.recordHistory !== false) {
        recordWorkspaceAction(state, {
            actionName: 'layer_asset_version_replaced',
            itemId: parentItemId,
            layerId: parentLayer.id || null,
            layerName: parentLayer.name || childItem.layerName || null,
            status: 'completed',
            hasResult: true,
            metadata: { versionId, sourceChildId: childId || null, prompt }
        });
    }

    return { storedUrl, versionId, parentLayer };
}

function isTextRuntimeLayer(layer) {
    if (!layer) return false;
    const name = String(layer.name || '').toLowerCase();
    return (
        layer.renderMode === 'text_css' ||
        layer.runtimeType === 'text_node' ||
        layer.semanticType === 'element_text' ||
        !!layer.sourceTextLayerId ||
        (layer.cleanPlateLayerId && name.startsWith('文字:')) ||
        name.includes('文字') ||
        name.includes('文案') ||
        name.includes('排版') ||
        name.includes('字') ||
        name.includes('text')
    );
}

function isSegmentationRuntimeLayer(layer) {
    return !isTextRuntimeLayer(layer) &&
        layer?.renderMode !== 'semantic_group' &&
        layer?.runtimeType !== 'semantic_group' &&
        layer?.compositeRole !== 'composite_group';
}

function isFlatDesignRuntimeLayer(layer) {
    if (!layer) return false;
    const semanticType = String(layer.semanticType || '').toLowerCase();
    const renderMode = String(layer.renderMode || '').toLowerCase();
    const designRole = String(layer.designRole || '').toLowerCase();
    const name = String(layer.name || '').toLowerCase();
    const isComplexRasterDecoration =
        /柠檬|水果|果片|糖果|棒棒糖|贴纸|吉祥物|角色|人物|女子|主视觉|插画|邮戳|印章|邮票|lemon|fruit|candy|sticker|mascot|character|woman|hero|illustration|stamp|seal/i.test(name);
    if (isComplexRasterDecoration) return false;
    return (
        ['vector_shape', 'background_plate', 'deferred'].includes(renderMode) ||
        ['base_background', 'local_panel', 'price_badge'].includes(designRole) ||
        (designRole === 'decor_shape' && renderMode !== 'raster_cutout') ||
        ['shape_panel', 'price_badge', 'ad_background', 'flat_ad_layout', 'cta_button', 'logo_mark'].includes(semanticType) ||
        name.includes('背景') ||
        name.includes('色块') ||
        name.includes('面板') ||
        name.includes('底板') ||
        name.includes('底色') ||
        name.includes('标签底板') ||
        name.includes('价格') ||
        name.includes('价签') ||
        name.includes('badge') ||
        name.includes('panel') ||
        name.includes('card') ||
        name.includes('background') ||
        name.includes('label')
    );
}

function isMagicLayersAutoSelectable(layer) {
    if (!layer) return false;

    const name = String(layer.name || '').toLowerCase();
    const isGroupLayer = layer.runtimeType === 'semantic_group' || layer.compositeRole === 'composite_group';
    const isBackgroundLayer =
        layer.category === 'background' ||
        layer.layerType === 'background_plate' ||
        layer.runtimeType === 'background_master' ||
        layer.semanticGroup === 'background' ||
        layer.semanticType === 'ad_background' ||
        layer.designRole === 'base_background' ||
        name.includes('background') ||
        name.includes('背景') ||
        name.includes('底板') ||
        name.includes('底色');

    return !isGroupLayer && !isBackgroundLayer;
}

function selectAllMagicLayers(itemId) {
    const item = workbenchItems.get(itemId);
    if (!item) return [];

    const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    if (!Array.isArray(layers) || layers.length === 0) return [];

    const selectedIndices = [];
    layers.forEach((layer, index) => {
        const layerState = getLayerState(itemId, index);
        const shouldSelect = isMagicLayersAutoSelectable(layer) && layerState.visible !== false && !layerState.locked;
        updateLayerState(itemId, index, { selected: shouldSelect });
        if (shouldSelect) selectedIndices.push(index);
    });

    renderLayerList(layers, itemId);
    updateFusionUI(itemId);
    return selectedIndices;
}

function createMagicLayersOverlay(item) {
    if (!item?.el) return null;

    const existing = document.getElementById('magicLayersOverlay');
    if (existing) existing.remove();

    const styleId = 'magicLayersOverlayHybridStyles';
    if (!document.getElementById(styleId)) {
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            @keyframes magicLayersHybridSweep {
                from { transform: translate3d(0, 0, 0); }
                to { transform: translate3d(var(--ml-sweep-distance, 1200px), 0, 0); }
            }

            @keyframes magicLayersHybridPulse {
                0%, 100% { opacity: 0.84; }
                50% { opacity: 1; }
            }

            @keyframes magicLayersHybridFlicker {
                0%, 100% { opacity: 0.78; }
                50% { opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }

    const overlay = document.createElement('div');
    overlay.id = 'magicLayersOverlay';
    overlay.style.cssText = `
        position: absolute;
        inset: 0;
        z-index: 5000;
        display: block;
        pointer-events: none;
        overflow: hidden;
        background: rgba(8, 12, 18, 0.08);
    `;

    const veil = document.createElement('div');
    veil.style.cssText = `
        position: absolute;
        inset: 0;
        background:
            linear-gradient(180deg, rgba(8, 14, 24, 0.26) 0%, rgba(6, 10, 18, 0.10) 50%, rgba(5, 8, 14, 0.28) 100%),
            radial-gradient(circle at 50% 50%, rgba(125, 211, 252, 0.05), rgba(125, 211, 252, 0) 58%);
    `;

    const scanRail = document.createElement('div');
    scanRail.style.cssText = `
        position: absolute;
        top: 0;
        bottom: 0;
        left: -180px;
        width: 160px;
        z-index: 1;
        will-change: transform;
        animation: magicLayersHybridSweep 2.6s linear infinite;
        mix-blend-mode: screen;
        pointer-events: none;
    `;

    const scanCanvas = document.createElement('canvas');
    scanCanvas.style.cssText = `
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        display: block;
    `;

    const coreLine = document.createElement('div');
    coreLine.style.cssText = `
        position: absolute;
        top: 0;
        bottom: 0;
        left: 50%;
        width: 2px;
        transform: translateX(-50%);
        background: linear-gradient(180deg,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.92) 12%,
            rgba(255,255,255,1) 50%,
            rgba(255,255,255,0.92) 88%,
            rgba(255,255,255,0) 100%);
        box-shadow:
            0 0 14px rgba(255,255,255,0.42),
            0 0 26px rgba(125,211,252,0.28);
        animation: magicLayersHybridFlicker 1.3s ease-in-out infinite;
    `;

    const edgeLine = document.createElement('div');
    edgeLine.style.cssText = `
        position: absolute;
        top: 2%;
        bottom: 2%;
        left: calc(50% + 7px);
        width: 5px;
        transform: translateX(-50%);
        border-radius: 999px;
        background: linear-gradient(180deg,
            rgba(125,211,252,0) 0%,
            rgba(125,211,252,0.38) 16%,
            rgba(125,211,252,0.64) 50%,
            rgba(125,211,252,0.38) 84%,
            rgba(125,211,252,0) 100%);
        filter: blur(1px);
        animation: magicLayersHybridPulse 1.6s ease-in-out infinite;
    `;

    const label = document.createElement('div');
    label.style.cssText = `
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        z-index: 2;
        min-width: 170px;
        padding: 14px 28px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.30);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.05) inset, 0 18px 60px rgba(15, 23, 42, 0.30);
        text-align: center;
        color: #f8fafc;
    `;
    label.innerHTML = `
        <div class="magic-layers-stage" style="font-size:13px;font-weight:700;letter-spacing:0.2px;margin-bottom:4px;">图层拆分进度</div>
        <div class="magic-layers-percent" style="font-size:16px;font-weight:700;letter-spacing:0.4px;">0%</div>
    `;

    scanRail.appendChild(scanCanvas);
    scanRail.appendChild(coreLine);
    scanRail.appendChild(edgeLine);
    overlay.appendChild(veil);
    overlay.appendChild(scanRail);
    overlay.appendChild(label);
    item.el.appendChild(overlay);

    const stageEl = label.querySelector('.magic-layers-stage');
    const percentEl = label.querySelector('.magic-layers-percent');
    const ctx = scanCanvas.getContext('2d');
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    let disposed = false;
    let currentSize = { w: 0, h: 0 };
    let currentRenderDpr = 1;

    const getRenderDpr = (w, h, bandW) => {
        const area = w * h;
        let cap = Math.min(1.25, dpr);
        if (area >= 1400000 || bandW >= 180) cap = Math.min(cap, 0.72);
        else if (area >= 900000 || bandW >= 160) cap = Math.min(cap, 0.84);
        else if (area >= 480000) cap = Math.min(cap, 0.96);
        return Math.max(0.6, cap);
    };

    const drawScanBand = (bandW, h) => {
        ctx.clearRect(0, 0, bandW, h);

        const trail = ctx.createLinearGradient(0, 0, bandW, 0);
        trail.addColorStop(0, 'rgba(56,189,248,0)');
        trail.addColorStop(0.28, 'rgba(56,189,248,0.03)');
        trail.addColorStop(0.56, 'rgba(125,211,252,0.09)');
        trail.addColorStop(0.82, 'rgba(255,255,255,0.06)');
        trail.addColorStop(1, 'rgba(56,189,248,0)');
        ctx.fillStyle = trail;
        ctx.fillRect(0, h * 0.10, bandW * 0.62, h * 0.80);

        const sweep = ctx.createLinearGradient(0, 0, bandW, 0);
        sweep.addColorStop(0, 'rgba(56,189,248,0)');
        sweep.addColorStop(0.20, 'rgba(56,189,248,0.08)');
        sweep.addColorStop(0.42, 'rgba(125,211,252,0.22)');
        sweep.addColorStop(0.5, 'rgba(255,255,255,0.28)');
        sweep.addColorStop(0.62, 'rgba(167,139,250,0.18)');
        sweep.addColorStop(0.84, 'rgba(56,189,248,0.08)');
        sweep.addColorStop(1, 'rgba(56,189,248,0)');
        ctx.fillStyle = sweep;
        ctx.fillRect(0, 0, bandW, h);

        ctx.save();
        ctx.filter = 'blur(8px)';
        const beamCore = ctx.createLinearGradient(bandW * 0.42, 0, bandW * 0.58, 0);
        beamCore.addColorStop(0, 'rgba(255,255,255,0)');
        beamCore.addColorStop(0.28, 'rgba(125,211,252,0.16)');
        beamCore.addColorStop(0.5, 'rgba(255,255,255,0.30)');
        beamCore.addColorStop(0.72, 'rgba(196,181,253,0.16)');
        beamCore.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = beamCore;
        ctx.fillRect(bandW * 0.34, 0, bandW * 0.32, h);
        ctx.restore();

        const accentGlow = ctx.createLinearGradient(0, 0, bandW, 0);
        accentGlow.addColorStop(0, 'rgba(56,189,248,0)');
        accentGlow.addColorStop(0.30, 'rgba(56,189,248,0.03)');
        accentGlow.addColorStop(0.5, 'rgba(125,211,252,0.08)');
        accentGlow.addColorStop(0.72, 'rgba(168,85,247,0.04)');
        accentGlow.addColorStop(1, 'rgba(56,189,248,0)');
        ctx.fillStyle = accentGlow;
        ctx.fillRect(bandW * 0.14, h * 0.14, bandW * 0.72, h * 0.72);
    };

    const resizeScanBand = () => {
        const rect = item.el.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        const bandW = Math.max(96, Math.min(176, Math.round(w * 0.22)));
        const nextRenderDpr = getRenderDpr(w, h, bandW);
        if (w === currentSize.w && h === currentSize.h && Math.abs(nextRenderDpr - currentRenderDpr) < 0.001) return;
        currentSize = { w, h };
        currentRenderDpr = nextRenderDpr;
        scanRail.style.width = `${bandW}px`;
        scanRail.style.left = `${Math.round(-bandW * 0.32)}px`;
        scanRail.style.setProperty('--ml-sweep-distance', `${Math.round(w + bandW * 0.64)}px`);
        scanCanvas.width = Math.max(1, Math.floor(bandW * currentRenderDpr));
        scanCanvas.height = Math.max(1, Math.floor(h * currentRenderDpr));
        ctx.setTransform(currentRenderDpr, 0, 0, currentRenderDpr, 0, 0);
        drawScanBand(bandW, h);
    };

    const resizeObserver = typeof ResizeObserver !== 'undefined'
        ? new ResizeObserver(() => resizeScanBand())
        : null;
    resizeObserver?.observe(item.el);
    window.addEventListener('resize', resizeScanBand);
    resizeScanBand();

    return {
        element: overlay,
        update(input, opts = {}) {
            const data = typeof input === 'string' ? { detail: input, ...opts } : { ...(input || {}) };
            const progress = Math.max(0, Math.min(100, Number(data.progress ?? 0)));
            if (percentEl) percentEl.textContent = `${Math.round(progress)}%`;
            if (stageEl) stageEl.textContent = '图层拆分进度';
        },
        close() {
            disposed = true;
            window.removeEventListener('resize', resizeScanBand);
            resizeObserver?.disconnect();
            overlay.remove();
        }
    };
}

function isRasterSegmentationRuntimeLayer(layer) {
    return isSegmentationRuntimeLayer(layer) && !isFlatDesignRuntimeLayer(layer);
}

function shouldForceRuntimeAssetForHeldSegmentation(layer, result = null) {
    if (!layer || !result?.dataUrl) return false;
    if (result?.isText || result?.isFlatDesignLayer) return false;
    return isRasterSegmentationRuntimeLayer(layer);
}

function bboxToWorkbenchRect(bbox, baseX, baseY, itemWidth, itemHeight, minWidth = 1, minHeight = 1) {
    const safeBbox = Array.isArray(bbox) && bbox.length === 4 ? bbox : [0, 0, 1000, 1000];
    const [ymin, xmin, ymax, xmax] = safeBbox.map(value => Number(value));
    const left = baseX + (xmin / 1000) * itemWidth;
    const top = baseY + (ymin / 1000) * itemHeight;
    const width = Math.max(((xmax - xmin) / 1000) * itemWidth, minWidth);
    const height = Math.max(((ymax - ymin) / 1000) * itemHeight, minHeight);
    return { left, top, width, height };
}

function getExtractedLayerZIndex(parentItem, layer, fallbackOrder = 0) {
    const parentZ = parseInt(parentItem?.el?.style?.zIndex || 0);
    const semanticZ = Number(layer?.zIndex);
    return parentZ + (Number.isFinite(semanticZ) ? semanticZ : fallbackOrder + 1) + 1;
}

async function persistLayerStateToRuntime(itemId, item, options = {}) {
    if (activeAgentLayerExtraction > 0) return;
    recordDiagnosticBreadcrumb('runtime:persist_start', {
        itemId,
        hasRuntime: Boolean(window.mvrRuntime),
        hasCleanPlate: Boolean(item?.cleanPlateDataUrl),
        layerCount: getItemSceneLayers(item).length
    });
    const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
    const asset = workspace ? workspace.currentState.assetRegistry.get(itemId) : null;

    if (workspace && asset) {
        workspace.dispatcher.dispatch({
            type: 'UPDATE_ASSET_METADATA',
            // This synchronization is persisted explicitly below; avoid
            // cloning the whole image registry into an undo snapshot.
            meta: { silent: true },
            payload: {
                uid: itemId,
                layers: cloneSerializable(item.layers) || [],
                scene: cloneSerializable(item.scene) || null,
                semanticViews: cloneSerializable(item.semanticViews) || null,
                hasFullSemanticAnalysis: !!item.hasFullSemanticAnalysis,
                cleanPlateDataUrl: item.cleanPlateDataUrl || asset.cleanPlateDataUrl || null,
                cleanPlateStatus: item.cleanPlateStatus || asset.cleanPlateStatus || 'idle'
            }
        });
    }

    if (options.persistSession !== false && state.currentSessionId && window.dbHelper?.saveSession) {
        const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
        if (currentSession) {
            await window.dbHelper.saveSession(currentSession);
        }
    }
    recordDiagnosticBreadcrumb('runtime:persist_done', { itemId });
}

function findSplitChildForSemanticLayer(itemId, layer) {
    if (!layer) return null;
    return findSplitChildrenForSemanticLayer(itemId, layer)[0] || null;
}

function getWorkbenchItemId(item) {
    if (!item) return null;
    if (item.id && workbenchItems.get(item.id) === item) return item.id;
    for (const [id, candidate] of workbenchItems.entries()) {
        if (candidate === item) return id;
    }
    return null;
}

function normalizeSplitLayerName(value) {
    return String(value || '')
        .trim()
        .toLowerCase()
        .replace(/^(拆解|提取|补全|完成|同步更新|独立编辑)[:：\-\s]*/u, '')
        .replace(/[\s_\-:：]/g, '');
}

function getSplitLayerIdentity(layer) {
    if (!layer) return null;
    const sourceLayerId = String(layer.sourceLayerId || '').trim();
    if (sourceLayerId) return sourceLayerId;
    const layerId = String(layer.layerId || '').trim();
    return layerId || null;
}

function bboxOverlapRatio(first, second) {
    if (!Array.isArray(first) || !Array.isArray(second) || first.length !== 4 || second.length !== 4) return 0;
    const top = Math.max(Number(first[0]), Number(second[0]));
    const left = Math.max(Number(first[1]), Number(second[1]));
    const bottom = Math.min(Number(first[2]), Number(second[2]));
    const right = Math.min(Number(first[3]), Number(second[3]));
    const overlap = Math.max(0, bottom - top) * Math.max(0, right - left);
    const firstArea = Math.max(0, Number(first[2]) - Number(first[0])) * Math.max(0, Number(first[3]) - Number(first[1]));
    const secondArea = Math.max(0, Number(second[2]) - Number(second[0])) * Math.max(0, Number(second[3]) - Number(second[1]));
    return overlap / Math.max(1, Math.min(firstArea, secondArea));
}

function findSplitChildrenForSemanticLayer(itemId, layer) {
    if (!layer) return [];
    const children = [...workbenchItems.values()].filter(candidate => candidate?.parentId === itemId);
    const targetName = normalizeSplitLayerName(layer.name);
    const targetIdentity = getSplitLayerIdentity(layer) || (layer.id ? String(layer.id) : null);
    const matches = children.filter(candidate => {
        const candidateIdentity = getSplitLayerIdentity(candidate);
        const exactId = Boolean(targetIdentity && candidateIdentity && candidateIdentity === targetIdentity);
        // An explicit identity is authoritative. A bbox can contain another
        // layer entirely (for example a lemon inside a person's bbox), so it
        // must never reassign an identified child during completion replace.
        if (candidateIdentity && targetIdentity && candidateIdentity !== targetIdentity) return false;
        const sameName = targetName && normalizeSplitLayerName(candidate.layerName) === targetName;
        const sameBbox = bboxOverlapRatio(candidate.originalBbox, layer.bbox) >= 0.82;
        return exactId || sameName || (!candidateIdentity && sameBbox);
    });
    return matches.sort((left, right) => {
        const leftExact = getSplitLayerIdentity(left) === targetIdentity;
        const rightExact = getSplitLayerIdentity(right) === targetIdentity;
        return Number(rightExact) - Number(leftExact);
    });
}

function getCompletionPlacementBbox(asset, targetLayer, child) {
    // The split child's original bbox is the stable scene projection. Never
    // resize it from a generative model's silhouette, which is not geometry.
    return asset?.canonicalAsset?.placementBbox ||
        asset?.canonicalAsset?.observedPlacementBbox ||
        asset?.preflight?.geometry?.cropBbox ||
        asset?.preflight?.geometry?.targetCompletionBbox ||
        targetLayer?.bbox ||
        targetLayer?.extractedBbox ||
        child?.extractionBbox || child?.originalBbox ||
        asset?.observedCutout?.bbox ||
        [0, 0, 1000, 1000];
}

function validateCanonicalReplacement(asset, targetLayer, canonicalUrl) {
    const canonical = asset?.canonicalAsset;
    if (!targetLayer || !canonical || canonical.status !== 'ready' || !canonicalUrl) {
        throw new Error('completion_canonical_asset_not_ready');
    }
    const placementBbox = getCompletionPlacementBbox(asset, targetLayer, null);
    if (!Array.isArray(placementBbox) || placementBbox.length !== 4 ||
        placementBbox.some(value => !Number.isFinite(Number(value)))) {
        throw new Error('completion_canonical_placement_invalid');
    }
    const [ymin, xmin, ymax, xmax] = placementBbox.map(Number);
    if (ymax <= ymin || xmax <= xmin || ymin < 0 || xmin < 0 || ymax > 1000 || xmax > 1000) {
        throw new Error('completion_canonical_placement_out_of_bounds');
    }
    const quality = canonical.quality || {};
    const canonicalEvidence = quality.canonicalEvidence || null;
    const evidenceAccepted = isCanonicalEvidenceAccepted(canonicalEvidence);
    const qualityHeld = quality.status === 'failed' ||
        quality.runtimeAction === 'hold' ||
        quality.shouldGenerateRuntimeLayer === false;
    if (qualityHeld && !evidenceAccepted) {
        throw new Error('completion_canonical_quality_hold');
    }
    if (qualityHeld && evidenceAccepted) {
        console.info('[Completion] canonical replacement released by final evidence', {
            targetLayerId: targetLayer.id,
            observedRecall: canonicalEvidence.observedRecall,
            hiddenKeptPixels: canonicalEvidence.hiddenKeptPixels,
            hiddenCoverage: canonicalEvidence.hiddenCoverage
        });
    }
    return { placementBbox: placementBbox.map(value => Number(value)) };
}

async function replaceSplitChildWithCanonicalAsset(itemId, item, asset, canonicalFile = null, options = {}) {
    const targetLayer = item?.scene?.layers?.find(layer => layer.id === asset.targetLayerId) ||
        item?.layers?.find(layer => layer.id === asset.targetLayerId);
    const canonicalUrl = asset?.canonicalAsset?.cutoutUrl;
    if (!targetLayer || !canonicalUrl) throw new Error('completion_target_or_canonical_asset_missing');
    const replacementPreflight = validateCanonicalReplacement(asset, targetLayer, canonicalUrl);
    const observedTargetBbox = cloneSerializable(
        targetLayer.observedBbox || targetLayer.originalBbox || targetLayer.bbox || null
    );

    const matchingChildren = findSplitChildrenForSemanticLayer(itemId, targetLayer);
    const targetIdentity = getSplitLayerIdentity(targetLayer) || (targetLayer.id ? String(targetLayer.id) : null);
    const completionAssetChildren = [...workbenchItems.values()].filter(candidate => {
        if (candidate?.parentId !== itemId || candidate.completionAssetId !== asset.id) return false;
        const candidateIdentity = getSplitLayerIdentity(candidate);
        return !candidateIdentity || !targetIdentity || candidateIdentity === targetIdentity;
    });
    const allMatchingChildren = [...new Map(
        [...matchingChildren, ...completionAssetChildren]
            .map(candidate => [getWorkbenchItemId(candidate), candidate])
            .filter(([childId]) => Boolean(childId))
    ).values()];
    let child = allMatchingChildren[0] || findSplitChildForSemanticLayer(itemId, targetLayer);
    let childId = getWorkbenchItemId(child);
    console.info('[Completion] replacement representations resolved', {
        itemId,
        targetLayerId: targetLayer.id,
        targetLayerName: targetLayer.name,
        matchingChildIds: allMatchingChildren.map(candidate => getWorkbenchItemId(candidate)).filter(Boolean),
        childId,
        parentHasCleanPlate: Boolean(item.cleanPlateDataUrl),
        parentDataIsCleanPlate: item.dataUrl === item.cleanPlateDataUrl,
        parentLayerVisibleBeforeReplace: targetLayer.visible !== false
    });
    if (!child || !childId) {
        // Deferred targets intentionally have no defective initial cutout.
        // Create their first and only workbench representation from the
        // scene-resegmented canonical asset.
        const placementBbox = replacementPreflight.placementBbox;
        const baseX = parseFloat(item.el?.style?.left) || 0;
        const baseY = parseFloat(item.el?.style?.top) || 0;
        const itemWidth = parseFloat(item.el?.style?.width) || item.el?.offsetWidth || 300;
        const itemHeight = parseFloat(item.el?.style?.height) || item.el?.offsetHeight || 300;
        const placement = bboxToWorkbenchRect(placementBbox, baseX, baseY, itemWidth, itemHeight, 1, 1);
        const file = canonicalFile || await dataURLToFile(canonicalUrl, `canonical-${targetLayer.id}-${Date.now()}.png`);
        childId = await addImageToWorkbench(file, `拆解-${targetLayer.name}`, {
            x: placement.left,
            y: placement.top,
            initialWidth: placement.width,
            initialHeight: placement.height,
            parentId: itemId,
            sourceLayerId: targetLayer.id,
            originalBbox: cloneSerializable(placementBbox),
            extractionBbox: cloneSerializable(placementBbox),
            canonicalBbox: cloneSerializable(placementBbox),
            completionAssetId: asset.id,
            layerName: targetLayer.name,
            type: 'layer-explode',
            zIndex: getExtractedLayerZIndex(item, targetLayer, 0),
            extractEngine: 'scene_inpaint_sam',
            quality: asset.canonicalAsset?.quality || null,
            // Do not open the fusion drawer while the completion transaction
            // is still handing off a large image. The child remains fully
            // interactive and can be opened manually after the batch.
            autoOpenDecisionPanel: false,
            // The completion batch emits one runtime notification after all
            // canonical replacements are applied.
            skipBackgroundUpload: true,
            skipRuntimeSnapshot: true,
            skipRuntimeNotify: true
        });
        child = workbenchItems.get(childId);
        if (!child) throw new Error(`completion_deferred_split_child_create_failed:${targetLayer.name || targetLayer.id}`);
        console.info('[Completion] created deferred split child from canonical asset', {
            itemId,
            targetLayerId: targetLayer.id,
            childId,
            placementBbox
        });
    }

    // There must be exactly one workbench representation for a semantic layer.
    // Clean up duplicates left by earlier completion runs before replacing the
    // surviving child, otherwise the old cutout remains visibly overlapped.
    const duplicateChildren = allMatchingChildren.filter(candidate => getWorkbenchItemId(candidate) !== childId);
    const workspace = window.mvrRuntime?.getCurrentWorkspace?.();
    for (const duplicate of duplicateChildren) {
        const duplicateId = getWorkbenchItemId(duplicate);
        if (!duplicateId) continue;
        duplicate.el?.remove();
        workbenchItems.delete(duplicateId);
        state.selectedWorkbenchItems.delete(duplicateId);
        if (workspace?.currentState?.assetRegistry?.get(duplicateId)) {
            workspace.dispatcher.dispatch({
                type: 'REMOVE_ASSET',
                payload: { uid: duplicateId }
            });
        }
        console.warn('[Completion] Removed duplicate split child before replacement', {
            duplicateId,
            targetLayerId: targetLayer.id
        });
    }

    const placementBbox = replacementPreflight.placementBbox;
    const baseX = parseFloat(item.el?.style?.left) || 0;
    const baseY = parseFloat(item.el?.style?.top) || 0;
    const itemWidth = parseFloat(item.el?.style?.width) || item.el?.offsetWidth || 300;
    const itemHeight = parseFloat(item.el?.style?.height) || item.el?.offsetHeight || 300;
    const placement = bboxToWorkbenchRect(placementBbox, baseX, baseY, itemWidth, itemHeight, 1, 1);
    console.info('[Completion] replacing split child in place', {
        itemId,
        targetLayerId: targetLayer.id,
        childId,
        placementBbox,
        placement,
        zIndex: child.el?.style?.zIndex || child.zIndex || null,
        canonicalUrl: canonicalUrl.slice(0, 80)
    });
    const originalZIndex = child.el?.style?.zIndex || child.zIndex || getExtractedLayerZIndex(item, targetLayer, 0);

    // Keep the durable OSS URL separate from the page-local preview. Upload
    // can succeed while the OSS image domain still times out in the browser.
    const displayUrl = canonicalFile ? URL.createObjectURL(canonicalFile) : canonicalUrl;
    if (child.runtimeDisplayUrl?.startsWith('blob:') && child.runtimeDisplayUrl !== displayUrl) {
        try {
            URL.revokeObjectURL(child.runtimeDisplayUrl);
        } catch (error) {
            console.warn('[Completion] Failed to revoke stale preview URL:', error);
        }
    }

    // Replace the existing split child in place. Keep stacking order and
    // rotation; only source pixels and the completion footprint change.
    child.dataUrl = displayUrl;
    child.originalDataUrl = canonicalUrl;
    child.previewUrl = canonicalUrl;
    child.runtimeDisplayUrl = displayUrl;
    // Use the generated File from this completion attempt. Do not fetch the
    // uploaded OSS URL again: an image can be publicly displayable while a
    // programmatic cross-origin fetch is still rejected by the browser.
    child.file = canonicalFile || null;
    // Keep the first extraction geometry immutable. A generated silhouette
    // must never redefine the scene placement used by later replacements.
    child.extractionBbox = cloneSerializable(child.extractionBbox || placementBbox);
    child.originalBbox = cloneSerializable(child.extractionBbox);
    child.canonicalBbox = cloneSerializable(placementBbox);
    child.completionAssetId = asset.id;
    child.assetStatus = 'ready';
    child.left = placement.left;
    child.top = placement.top;
    child.width = placement.width;
    child.height = placement.height;
    if (child.el) {
        child.el.style.left = `${placement.left}px`;
        child.el.style.top = `${placement.top}px`;
        child.el.style.width = `${placement.width}px`;
        child.el.style.height = `${placement.height}px`;
        child.el.style.zIndex = `${originalZIndex}`;
        const image = child.el.querySelector('.crop-container > img') || child.el.querySelector('img');
        if (image) {
            image.style.visibility = 'visible';
            image.onerror = () => {
                if (image.src === displayUrl && canonicalUrl !== displayUrl) {
                    image.src = canonicalUrl;
                    return;
                }
                image.style.visibility = 'hidden';
                console.warn('[Completion] Preview image unavailable:', childId);
            };
            image.src = displayUrl;
        }
    }

    const runtimeAsset = workspace?.currentState?.assetRegistry?.get(childId);
    if (runtimeAsset) {
        workspace.dispatcher.dispatch({
            type: 'UPDATE_ASSET_METADATA',
            // Canonical replacement is persisted below; avoid another full
            // workspace snapshot while replacing a large PNG asset.
            meta: { silent: true },
            payload: {
                uid: childId,
                sourceImage: canonicalUrl,
                originalDataUrl: canonicalUrl,
                runtimeDisplayUrl: displayUrl,
                completionAssetId: asset.id,
                originalBbox: cloneSerializable(child.extractionBbox),
                extractionBbox: cloneSerializable(child.extractionBbox)
            }
        });
        workspace.currentState.assetRegistry.updateAssetTransform(childId, {
            x: placement.left,
            y: placement.top,
            width: placement.width,
            height: placement.height,
            zIndex: Number(originalZIndex)
        });
    }

    // Keep the defective observed cutout as provenance only. The active layer
    // and its publishable version are now the canonical completed asset.
    targetLayer.cutoutUrl = canonicalUrl;
    targetLayer.previewUrl = canonicalUrl;
    // Keep the visible semantic geometry immutable. Generated geometry belongs
    // to the canonical representation and must not become the next task's
    // identity/occlusion anchor.
    targetLayer.observedBbox = observedTargetBbox;
    targetLayer.extractedBbox = observedTargetBbox || cloneSerializable(targetLayer.extractedBbox);
    targetLayer.canonicalBbox = cloneSerializable(placementBbox);
    targetLayer.completionAssetId = asset.id;
    targetLayer.assetStatus = 'ready';
    targetLayer.completionState = 'canonical';
    targetLayer.visible = false;
    const parentLayers = getItemSceneLayers(item);
    const targetLayerIndex = parentLayers.indexOf(targetLayer);
    if (targetLayerIndex >= 0) {
        updateLayerState(itemId, targetLayerIndex, { visible: false, selected: false });
    }
    const versionId = `completion-${Date.now()}`;
    targetLayer.versions = [{
        id: versionId,
        cutoutUrl: canonicalUrl,
        previewUrl: canonicalUrl,
        prompt: 'Magic Layers 自动补全遮挡物',
        createdAt: Date.now(),
        source: 'occlusion_completion',
        publishable: true
    }];
    targetLayer.activeVersionId = versionId;

    asset.status = 'canonical_ready';
    if (asset.preflight?.contract) asset.preflight.contract.state = 'replaced';
    asset.composition = {
        ...(asset.composition || {}),
        activeRepresentation: 'canonical_asset',
        preserveOriginalOcclusion: true,
        sourceLayerVisibility: 'replaced',
        originalZIndex: Number(originalZIndex),
        placementBbox: cloneSerializable(placementBbox),
        placementPolicy: 'replace_split_child_preserve_z_index'
    };
    asset.updatedAt = Date.now();
    // The completed child uses a transient blob preview in this page, but its
    // registry source is the durable OSS URL. Batch callers persist only after
    // every replacement has been applied, so large image references are not
    // serialized once per target during the completion handoff.
    if (options.persist !== false) {
        await persistLayerStateToRuntime(itemId, item);
    }
    console.info('[Completion] replacement committed', {
        itemId,
        targetLayerId: targetLayer.id,
        childId,
        removedDuplicateCount: duplicateChildren.length,
        remainingMatchingChildren: [...new Set([
            ...findSplitChildrenForSemanticLayer(itemId, targetLayer),
            ...[...workbenchItems.values()].filter(candidate => candidate?.completionAssetId === asset.id)
        ].map(candidate => getWorkbenchItemId(candidate)).filter(Boolean))],
        parentLayerVisible: targetLayer.visible,
        parentHasCleanPlate: Boolean(item.cleanPlateDataUrl)
    });
    return { child, targetLayer, placementBbox };
}

async function autoCompleteVerifiedOccludedAssets(itemId, item, progressMessage, options = {}) {
    const targetLayerIds = options.targetLayerIds instanceof Set ? options.targetLayerIds : null;
    const assets = Array.isArray(item?.semanticViews?.completionAssets)
        ? item.semanticViews.completionAssets.filter(asset =>
            asset.status === 'ready_for_completion' &&
            (!targetLayerIds || targetLayerIds.has(asset.targetLayerId))
        )
        : [];
    if (!assets.length) return { completed: 0, skipped: 0, completedTargetLayerIds: [], skippedTargetLayerIds: [] };

    let completed = 0;
    let skipped = 0;
    const completedTargetLayerIds = [];
    const skippedTargetLayerIds = [];
    for (const asset of assets) {
        try {
            progressMessage?.(`正在自动补全被遮挡物体: ${asset.targetLayerId}...`);
            const result = await executeObjectCompletion(item, asset.id, {
                onProgress: message => progressMessage?.(message)
            });
            if (!result?.success) {
                skipped += 1;
                skippedTargetLayerIds.push(asset.targetLayerId);
                continue;
            }
            await replaceSplitChildWithCanonicalAsset(itemId, item, asset, result.canonicalFile, {
                persist: false
            });
            completed += 1;
            completedTargetLayerIds.push(asset.targetLayerId);
            console.info(`[Completion] Replaced incomplete split layer: ${asset.targetLayerId}`);
        } catch (error) {
            skipped += 1;
            skippedTargetLayerIds.push(asset.targetLayerId);
            asset.status = 'manual_review';
            asset.generation = {
                ...(asset.generation || {}),
                state: 'failed',
                lastError: error?.message || 'automatic_completion_failed'
            };
            console.warn('[Completion] automatic completion state trace', {
                targetLayerId: asset.targetLayerId,
                assetStatus: asset.status,
                generationState: asset.generation?.state,
                canonicalStatus: asset.canonicalAsset?.status || null,
                canonicalRuntimeAction: asset.canonicalAsset?.quality?.runtimeAction || null,
                canonicalShouldGenerate: asset.canonicalAsset?.quality?.shouldGenerateRuntimeLayer,
                canonicalEvidence: asset.canonicalAsset?.quality?.canonicalEvidence || null,
                error: error?.message || 'automatic_completion_failed'
            });
            console.warn(`[Completion] Automatic completion skipped for ${asset.targetLayerId}:`, error);
        }
    }

    if (completed > 0) {
        // Let the browser commit the final image-source changes before the
        // single full render and durable save for this completion batch.
        await yieldToBrowser();
        renderLayerList(item.scene?.layers || item.layers || [], itemId);
        renderCanvasLayers(itemId);
        const workspace = window.mvrRuntime?.getCurrentWorkspace?.();
        // Notify the runtime exactly once for the completed batch. Individual
        // child creation/replacement above is intentionally notification-free.
        workspace?.currentState?.notify?.();
        await yieldToBrowser();
        await persistLayerStateToRuntime(itemId, item);
    }
    return { completed, skipped, completedTargetLayerIds, skippedTargetLayerIds };
}

// Both full Magic Layers and a single layer extraction enter here. A target is
// completed only after its own mask and every required foreground mask exist.
export async function runAutomaticCompletionForItem(itemId, options = {}) {
    const item = workbenchItems.get(itemId);
    if (!item) {
        console.warn('[Completion] coordinator skipped: workbench item not found', { itemId });
        return { completed: 0, skipped: 0, eligible: 0, reason: 'item_not_found' };
    }

    const progressMessage = typeof options.onProgress === 'function' ? options.onProgress : null;
    console.info('[Completion] coordinator entered', {
        itemId,
        semanticLayerCount: item.semanticViews?.layerGraph?.layers?.length || 0,
        prescreenTaskCount: item.semanticViews?.layerGraph?.completionTasks?.length || 0,
        extractedLayers: (item.semanticViews?.editableSceneLayers || []).map(layer => ({
            id: layer.id,
            name: layer.name,
            bbox: layer.extractedBbox || layer.bbox || null,
            zIndex: layer.zIndex,
            hasCutout: Boolean(layer.cutoutUrl),
            semanticType: layer.semanticType
        }))
    });
    const graph = await validateCompletionCandidatesWithMasks(item, {
        allowDeferredTarget: options.allowDeferredTarget === true
    });
    if (!graph) {
        console.warn('[Completion] coordinator skipped: semantic graph unavailable', { itemId });
        return { completed: 0, skipped: 0, eligible: 0, reason: 'graph_unavailable' };
    }

    const eligible = graph.stats?.autoCompletionCandidateCount || 0;
    console.info(
        `[Completion Candidates] validated auto=${eligible} ` +
        `manual=${graph.stats?.manualCompletionReviewCount || 0} ` +
        `rejected=${graph.stats?.rejectedCompletionCandidateCount || 0}`
    );
    graph.completionTasks?.forEach(task => {
        const detail = {
            targetLayerId: task.targetLayerId,
            occluderLayerIds: task.occluderLayerIds,
            targetZIndex: graph.layers?.find(layer => layer.layerId === task.targetLayerId)?.zIndex ?? null,
            eligibility: task.eligibility,
            status: task.status,
            reason: task.reason,
            nearContactRelations: task.nearContactRelations || [],
            maskValidation: task.maskValidation,
            targetHasMask: Boolean(graph.layers?.find(layer => layer.layerId === task.targetLayerId)?.mask?.cutoutUrl),
            occluderMaskAvailability: task.occluderLayerIds.map(occluderLayerId => {
                const layer = graph.layers?.find(candidate => candidate.layerId === occluderLayerId);
                return {
                    id: occluderLayerId,
                    name: layer?.name || null,
                    zIndex: layer?.zIndex ?? null,
                    hasMask: Boolean(layer?.mask?.cutoutUrl)
                };
            })
        };
        console.info('[Completion Candidate Detail]', JSON.stringify(detail));
    });
    if (!graph.completionTasks?.length) {
        console.warn('[Completion] no occlusion candidates after bbox prescreen', {
            itemId,
            layers: graph.layers?.map(layer => ({
                id: layer.layerId,
                name: layer.name,
                bbox: layer.bbox,
                zIndex: layer.zIndex,
                semanticGroup: layer.semanticGroup
            })) || []
        });
    }
    const targetLayerIds = Array.isArray(options.targetLayerIds)
        ? new Set(options.targetLayerIds)
        : null;
    if (targetLayerIds) {
        console.info('[Completion] scoped automatic completion batch', {
            itemId,
            targetLayerIds: [...targetLayerIds]
        });
    }
    const result = await autoCompleteVerifiedOccludedAssets(itemId, item, progressMessage, { targetLayerIds });
    return { ...result, eligible };
}

function trimTransparentCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    let top = 0, bottom = height, left = 0, right = width;

    // Find top
    for (let y = 0; y < height; y++) {
        let found = false;
        for (let x = 0; x < width; x++) {
            if (data[(y * width + x) * 4 + 3] > 0) {
                top = y;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    // Find bottom
    for (let y = height - 1; y >= 0; y--) {
        let found = false;
        for (let x = 0; x < width; x++) {
            if (data[(y * width + x) * 4 + 3] > 0) {
                bottom = y + 1;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    // Find left
    for (let x = 0; x < width; x++) {
        let found = false;
        for (let y = top; y < bottom; y++) {
            if (data[(y * width + x) * 4 + 3] > 0) {
                left = x;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    // Find right
    for (let x = width - 1; x >= 0; x--) {
        let found = false;
        for (let y = top; y < bottom; y++) {
            if (data[(y * width + x) * 4 + 3] > 0) {
                right = x + 1;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    const trimmedWidth = right - left;
    const trimmedHeight = bottom - top;
    
    if (trimmedWidth <= 0 || trimmedHeight <= 0) return canvas;

    const trimmedCanvas = document.createElement('canvas');
    trimmedCanvas.width = trimmedWidth;
    trimmedCanvas.height = trimmedHeight;
    trimmedCanvas.getContext('2d').putImageData(ctx.getImageData(left, top, trimmedWidth, trimmedHeight), 0, 0);

    return trimmedCanvas;
}

const { workbenchItems } = state;

/**
 * 核心修改：基于语义图层的物理拆解
 */
export async function triggerLayerExplosion(itemId, options = {}) {
    console.log(`[Explosion] Triggered for item: ${itemId}`);
    const item = workbenchItems.get(itemId);
    if (!item) {
        alert("找不到图片对象");
        return;
    }
    beginDiagnosticTask('magic_layers', { itemId, mode: 'layer_explosion' });

    // 1. 如果还没有图层数据，先自动分析
    const layersToUse = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    const requestedLayerIds = new Set(Array.isArray(options.layerIds) ? options.layerIds.filter(Boolean) : []);

    if (!layersToUse || layersToUse.length === 0) {
        addMessage({ sender: 'bot', type: 'text', visibility: 'progress', persist: false, content: '🔍 正在分析图层结构以进行拆解...' });
        try {
            updateDiagnosticTask('semantic_analysis_start', { itemId });
            const analysisResult = await analyzeImageLayers(item.file || item.dataUrl);
            updateDiagnosticTask('semantic_analysis_done', {
                itemId,
                layerCount: analysisResult?.rawLayers?.length || analysisResult?.scene?.layers?.length || 0
            });
            item.scene = analysisResult.scene;
            const semanticViews = await buildSemanticLayerViews(
                item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl,
                analysisResult.rawLayers,
                { expandText: true }
            );
            applySemanticLayerViewsToItem(item, semanticViews);
            item.hasFullSemanticAnalysis = true;
            workbenchItems.set(itemId, item);
            await persistLayerStateToRuntime(itemId, item);
            // 顺便更新下 UI 面板（虽然现在是隐藏的，但为了数据一致性）
            renderLayerList(item.scene.layers, itemId);
            renderCanvasLayers(itemId);
        } catch (e) {
            finishDiagnosticTask('failed', { itemId, stage: 'semantic_analysis', message: e?.message || String(e) });
            addMessage({ sender: 'bot', type: 'text', content: '❌ 无法识别图层，拆解中止。' });
            return;
        }
    }

    // 2. 准备拆解任务
    const executeLayerExplosion = async (customPrompt) => {
        const pipelineStartedAt = Date.now();
        const allLayers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
        const currentLayers = [];
        
        allLayers.forEach((layer, index) => {
            const state = getLayerState(itemId, index);
            const isRequested = requestedLayerIds.size > 0 && requestedLayerIds.has(layer.id);
            const isAutoSelected = options.selectAll === true && isMagicLayersAutoSelectable(layer) && state.visible !== false;
            if (!state.locked && (isRequested || (requestedLayerIds.size === 0 && (state.selected || isAutoSelected)))) {
                currentLayers.push(layer);
            }
        });

        if (currentLayers.length === 0) {
            finishDiagnosticTask('aborted', { itemId, stage: 'selection', reason: 'no_selected_layers' });
            addMessage({ sender: 'bot', type: 'text', content: '❌ 拆解中止：未能找到被勾选且未锁定的图层。' });
            return;
        }

        const layerNames = currentLayers.map(l => l.name);
        const batchAutoOpenToken = `explode-batch-${itemId}-${Date.now()}`;
        const tempMsg = addMessage({ 
            sender: 'bot', 
            type: 'text', 
            visibility: 'progress',
            persist: false,
            content: `💣 **启动语义拆解模式**\n正在批量提取 **${layerNames.length}** 个独立图层：[${layerNames.join(', ')}]...` 
        });

        try {
            console.log(`[Explosion] Stage: prepare image (${itemId})`);
            updateDiagnosticTask('prepare_image_start', { itemId });
            const parentImg = new Image();
            parentImg.crossOrigin = "anonymous";
            const lockedSourceUrl = item.originalDataUrl || item.dataUrl;
            const itemSrc = getProxiedUrl(lockedSourceUrl);
            if (!itemSrc) {
                throw new Error('无效的原图地址，无法执行图层拆解');
            }
            parentImg.src = itemSrc;
            await new Promise(r => parentImg.onload = r);
            const sceneLock = createMagicLayersSceneLock(item, parentImg, lockedSourceUrl);
            const lockedSegmentationItem = createLockedSceneSegmentationItem(item, sceneLock);
            updateDiagnosticTask('prepare_image_done', {
                itemId,
                taskLockId: sceneLock.id,
                sceneSize: `${sceneLock.width}x${sceneLock.height}`
            });

            let fastSamResults = new Map();
            const objectLayers = currentLayers.filter(layer => isRasterSegmentationRuntimeLayer(layer));
            const segmentationSkippedLayers = currentLayers
                .filter(layer => !objectLayers.includes(layer))
                .map(layer => {
                    const reason = isTextRuntimeLayer(layer)
                        ? 'text_runtime_layer'
                        : layer?.runtimeType === 'semantic_group' || layer?.compositeRole === 'composite_group'
                            ? 'semantic_group'
                            : isFlatDesignRuntimeLayer(layer)
                                ? `flat_design:${layer.renderMode || layer.designRole || layer.semanticType || 'unknown'}`
                                : 'not_raster_segmentable';
                    return {
                        id: layer.id,
                        name: layer.name,
                        semanticType: layer.semanticType || null,
                        designRole: layer.designRole || null,
                        renderMode: layer.renderMode || null,
                        reason
                    };
                });
            console.info('[Explosion] SAM routing', {
                selectedLayerCount: currentLayers.length,
                submittedLayerCount: objectLayers.length,
                submittedLayers: objectLayers.map(layer => ({
                    id: layer.id,
                    name: layer.name,
                    semanticType: layer.semanticType || null,
                    renderMode: layer.renderMode || null,
                    forcedRasterDecoration: /柠檬|水果|果片|糖果|棒棒糖|贴纸|吉祥物|角色|人物|女子|主视觉|插画|邮戳|印章|邮票|lemon|fruit|candy|sticker|mascot|character|woman|hero|illustration|stamp|seal/i.test(String(layer.name || '').toLowerCase())
                })),
                skippedLayers: segmentationSkippedLayers
            });
            recordDiagnosticBreadcrumb('sam:routing', {
                itemId,
                selectedLayerCount: currentLayers.length,
                submittedLayerCount: objectLayers.length,
                submittedLayerIds: objectLayers.map(layer => layer.id),
                skippedLayers: segmentationSkippedLayers,
                selectedLayers: currentLayers.map(layer => ({
                    index: allLayers.findIndex(candidate => candidate === layer || candidate?.id === layer?.id),
                    id: layer.id,
                    name: layer.name,
                    selected: getLayerState(itemId, allLayers.findIndex(candidate => candidate === layer || candidate?.id === layer?.id)).selected,
                    locked: getLayerState(itemId, allLayers.findIndex(candidate => candidate === layer || candidate?.id === layer?.id)).locked,
                    visible: getLayerState(itemId, allLayers.findIndex(candidate => candidate === layer || candidate?.id === layer?.id)).visible,
                    semanticType: layer.semanticType || null,
                    renderMode: layer.renderMode || null,
                    runtimeType: layer.runtimeType || null,
                    designRole: layer.designRole || null,
                    isFlatDesign: isFlatDesignRuntimeLayer(layer),
                    isText: isTextRuntimeLayer(layer)
                }))
            });
            const deferredCompletionTargetIds = getDeferredCompletionTargetIds(item, objectLayers);
            // Deferred targets still receive an original-scene SAM pass. Its
            // cutout is observation-only: it anchors the visible silhouette
            // for the post-inpaint quality gate and is never published as a
            // Workbench asset or included in the first clean plate request.
            // Foreground masks are destructive input to scene inpainting. Mark
            // only the selected hard occluders for an independent B/L review;
            // this is request-local and never changes the semantic layer.
            const completionOccluderLayerIds = new Set(
                (item.semanticViews?.layerGraph?.completionTasks || [])
                    .filter(task => (
                        task?.eligibility === 'auto' &&
                        deferredCompletionTargetIds.has(task.targetLayerId)
                    ))
                    .flatMap(task => expandEntityLayerIds(
                        item.semanticViews?.layerGraph?.layers || [],
                        task.occluderLayerIds || []
                    ))
            );
            const initialSegmentationLayers = objectLayers.map(layer => (
                completionOccluderLayerIds.has(layer.id)
                    ? { ...layer, completionOccluder: true }
                    : layer
            ));
            console.info('[Explosion] completion deferral decision', {
                itemId,
                taskLockId: sceneLock.id,
                lockedSceneSize: `${sceneLock.width}x${sceneLock.height}`,
                completionTasks: (item.semanticViews?.layerGraph?.completionTasks || []).map(task => ({
                    targetLayerId: task.targetLayerId,
                    occluderLayerIds: task.occluderLayerIds,
                    eligibility: task.eligibility,
                    deferred: deferredCompletionTargetIds.has(task.targetLayerId),
                    targetZIndex: item.semanticViews?.layerGraph?.layers?.find(layer => layer.layerId === task.targetLayerId)?.zIndex ?? null,
                    occluders: (task.occluderLayerIds || []).map(occluderId => {
                        const layer = item.semanticViews?.layerGraph?.layers?.find(candidate => candidate.layerId === occluderId);
                        return { id: occluderId, zIndex: layer?.zIndex ?? null };
                    })
                })),
                deferredTargetLayerIds: [...deferredCompletionTargetIds]
            });
            if (deferredCompletionTargetIds.size > 0) {
                console.info('[Explosion] deferring occluded targets until foreground masks are ready', {
                    itemId,
                    targetLayerIds: [...deferredCompletionTargetIds],
                    initialSegmentationLayerIds: initialSegmentationLayers.map(layer => layer.id),
                    observationOnlyLayerIds: [...deferredCompletionTargetIds]
                });
            }
            if (initialSegmentationLayers.length > 0) {
                try {
                    console.log(`[Explosion] Stage: segmentation start (${itemId}) objects=${initialSegmentationLayers.length}`);
                    updateDiagnosticTask('sam_start', {
                        itemId,
                        count: initialSegmentationLayers.length,
                        layerIds: initialSegmentationLayers.map(layer => layer.id)
                    });
                    const segmentationStartedAt = Date.now();
                    const requestedSamResults = await segmentLayers({
                        item: lockedSegmentationItem,
                        layers: initialSegmentationLayers,
                        onProgress: (message) => tempMsg.update(message),
                        qualityProfile: 'completion'
                    });
                    // segmentLayers keys its result Map by request object.
                    // Completion-occluder requests are shallow clones so the
                    // marker stays transaction-local; remap them back to the
                    // canonical layer objects used by all later stages.
                    const sourceLayersById = new Map(objectLayers.map(layer => [layer.id, layer]));
                    fastSamResults = new Map(
                        [...requestedSamResults].map(([requestLayer, result]) => [
                            sourceLayersById.get(requestLayer.id) || requestLayer,
                            result
                        ])
                    );
                    console.log(`[Explosion] Stage: segmentation done (${itemId}) duration=${Date.now() - segmentationStartedAt}ms`);
                    updateDiagnosticTask('sam_done', {
                        itemId,
                        durationMs: Date.now() - segmentationStartedAt,
                        resultCount: fastSamResults.size
                    });
                } catch (error) {
                    if (DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST) {
                        throw new Error(`FastSAM 批量分割失败，已阻止进入 Gemini 兜底: ${error.message || error}`);
                    }
                    console.warn('[FastSAM] Batch segmentation unavailable, fallback to legacy extraction:', error);
                }
            }

            // 3. Prepare extraction jobs. The jobs are executed through a
            // bounded queue below so decoded canvases do not accumulate.
            const originalTextSource = item.originalDataUrl || item.file || item.dataUrl;
            console.log(`[Explosion] Stage: extraction start (${itemId}) layers=${currentLayers.length}`);
            updateDiagnosticTask('extraction_start', { itemId, count: currentLayers.length });
            const extractionStartedAt = Date.now();
            const extractionJobs = currentLayers.map((layer, index) => () => (async () => {
                const isTextLayer = isTextRuntimeLayer(layer);
                
                if (isTextLayer) {
                    try {
                        tempMsg.update(`🔍 **文字提取**: 正在解析 "${layer.name}"...`);
                        const baseBg = originalTextSource;
                        let textLines = getCachedTextExtraction(layer, {
                            allowSemanticFallback: false,
                            allowTextContentFallback: false
                        });

                        if (textLines.length === 0) {
                            const ocrResult = await extractTextFromImage(baseBg, layer.bbox, { mode: 'strict' });
                            textLines = normalizeOcrTextLines(ocrResult, layer);
                        }

                        textLines = filterTextLinesToBbox(textLines, layer.bbox, 0.45);
                        
                        if (textLines.length === 0) {
                            console.warn(`[layer-manager] 未检测到文本内容或提取失败 (层: ${layer.name})，将继续执行背景净化`);
                        }
                        
                        return { status: 'fulfilled', value: { success: true, isText: true, textLines, layerBbox: layer.bbox }, layerName: layer.name };
                    } catch (err) {
                        console.error(`[layer-manager] 文本提取抛出异常 (层: ${layer.name}):`, err);
                        return { status: 'fulfilled', value: { success: true, isText: true, textLines: [], layerBbox: layer.bbox }, layerName: layer.name };
                    }
                }

                if (isFlatDesignRuntimeLayer(layer)) {
                    return {
                        status: 'fulfilled',
                        value: {
                            success: true,
                            isFlatDesignLayer: true,
                            layerBbox: layer.bbox,
                            extractEngine: 'runtime_vector_pending',
                            quality: {
                                status: 'deferred',
                                runtimeAction: 'hold',
                                shouldGenerateRuntimeLayer: false,
                                needsHigherPrecision: false,
                                reason: 'flat_design_layer_not_raster_segmented',
                                recommendedEngine: 'runtime_vector_or_css'
                            }
                        },
                        layerName: layer.name
                    };
                }

                if (deferredCompletionTargetIds.has(layer.id)) {
                    const observed = fastSamResults.get(layer);
                    if (observed?.dataUrl) {
                        updateLayerExtractionMetadata(item, {
                            id: layer.id,
                            name: layer.name,
                            cleanPlateLayerId: layer.cleanPlateLayerId,
                            sourceTextLayerId: layer.sourceTextLayerId
                        }, {
                            extractEngine: observed.extractEngine || 'sam_observed_reference',
                            quality: observed.quality || null,
                            bbox: observed.bbox || layer.bbox,
                            cutoutUrl: observed.dataUrl,
                            previewUrl: observed.dataUrl
                        });
                        console.info('[Explosion] captured deferred target observation mask', {
                            itemId,
                            layerId: layer.id,
                            name: layer.name,
                            bbox: observed.bbox || layer.bbox,
                            quality: observed.quality || null,
                            policy: 'reference_only_not_runtime_asset'
                        });
                    } else {
                        console.warn('[Explosion] deferred target observation mask unavailable', {
                            itemId,
                            layerId: layer.id,
                            name: layer.name
                        });
                    }
                    return {
                        status: 'fulfilled',
                        value: {
                            success: true,
                            completionDeferred: true,
                            layerBbox: layer.bbox,
                            extractEngine: observed?.extractEngine || 'scene_inpaint_sam_pending',
                            observedReferenceReady: Boolean(observed?.dataUrl),
                            observedReferenceBbox: observed?.bbox || layer.bbox
                        },
                        layerName: layer.name
                    };
                }

                const segmented = fastSamResults.get(layer);
                if (segmented?.dataUrl) {
                    const quality = segmented.quality || null;
                    const runtimeAction = segmented.runtimeAction || quality?.runtimeAction || 'accept';
                    const shouldGenerateRuntimeLayer = segmented.shouldGenerateRuntimeLayer !== false;
                    const forceRuntimeChild = shouldForceRuntimeAssetForHeldSegmentation(layer, {
                        dataUrl: segmented.dataUrl,
                        isText: false,
                        isFlatDesignLayer: false
                    });
                    console.log(
                        `[Explosion] Segmentation result (${itemId}) layer="${layer.name}" ` +
                        `hasDataUrl=${!!segmented.dataUrl} runtimeAction=${runtimeAction} ` +
                        `shouldGenerateRuntimeLayer=${shouldGenerateRuntimeLayer} forceRuntimeChild=${forceRuntimeChild} ` +
                        `quality=${quality?.status || 'unknown'}`
                    );
                    return {
                        status: 'fulfilled',
                        value: {
                            success: true,
                            dataUrl: segmented.dataUrl,
                            width: segmented.width,
                            height: segmented.height,
                            segmentedBbox: segmented.bbox || layer.bbox,
                            extractEngine: segmented.extractEngine || 'fastsam',
                            quality,
                            shouldGenerateRuntimeLayer,
                            runtimeAction
                        },
                        layerName: layer.name
                    };
                }
                
                const [ymin, xmin, ymax, xmax] = layer.bbox;
                const pxMinX = (xmin / 1000) * parentImg.naturalWidth;
                const pxMinY = (ymin / 1000) * parentImg.naturalHeight;
                const pxMaxX = (xmax / 1000) * parentImg.naturalWidth;
                const pxMaxY = (ymax / 1000) * parentImg.naturalHeight;
                const w = pxMaxX - pxMinX;
                const h = pxMaxY - pxMinY;

                const isManualBox = layer.id && layer.id.startsWith('box-layer');
                const paddingFactor = 0.15; // Uniformly use 15% padding to ensure strategy-dispatcher has enough background pixels
                const padX = Math.floor(w * paddingFactor);
                const padY = Math.floor(h * paddingFactor);
                const cropX = Math.max(0, Math.floor(pxMinX - padX));
                const cropY = Math.max(0, Math.floor(pxMinY - padY));
                const rawCropW = Math.min(Math.floor(parentImg.naturalWidth) - cropX, Math.floor(w + padX * 2));
                const rawCropH = Math.min(Math.floor(parentImg.naturalHeight) - cropY, Math.floor(h + padY * 2));

                // ==========================================
                // AI SUPER-RESOLUTION UPSCALE (Local Preparation)
                // ==========================================
                let cropW = rawCropW;
                let cropH = rawCropH;
                const maxDim = Math.max(rawCropW, rawCropH);
                if (maxDim < 1024 && maxDim > 0) {
                    const scaleFactor = 1024 / maxDim;
                    cropW = Math.floor(rawCropW * scaleFactor);
                    cropH = Math.floor(rawCropH * scaleFactor);
                }

                const cropCanvas = document.createElement('canvas');
                cropCanvas.width = cropW;
                cropCanvas.height = cropH;
                const ctx = cropCanvas.getContext('2d');
                
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                ctx.drawImage(parentImg, cropX, cropY, rawCropW, rawCropH, 0, 0, cropW, cropH);
                
                // --- 动态背景色逻辑 (Dynamic Backdrop Selection) ---
                const cropImageData = ctx.getImageData(0, 0, cropW, cropH);
                
                tempMsg.update(`🧠 **正在决策**: "${layer.name}"... 执行智能化场景分析调度...`);
                const strategy = StrategyDispatcher.analyze(cropImageData, layer.name);
                console.log(`[Strategy Dispatcher ${layer.name}] Path: ${strategy.path} | Reason: ${strategy.reason}`);
                
                try {
                    let processedAlphaData;
                    let transparentDataUrl;
                    const taskId = `matte_task_${Date.now()}`;
                    
                    if (strategy.path === 'cv') {
                        // FAST PATH: Pure CV
                        tempMsg.update(`⚡ **纯前台秒抠**: "${layer.name}"...`);
                        const bgMeanRGB = strategy.metrics?.bgMean ? 
                            [strategy.metrics.bgMean.r, strategy.metrics.bgMean.g, strategy.metrics.bgMean.b] : 
                            [255, 255, 255];
                        
                        processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                            taskId, 
                            cropImageData, // ORIGINAL PIXELS
                            null, 
                            layer.name, 
                            (msg) => tempMsg.update(msg),
                            { type: 'cv_euclidean', bgColor: bgMeanRGB } 
                        );
                        
                        const finalCanvas = document.createElement('canvas');
                        finalCanvas.width = cropW;
                        finalCanvas.height = cropH;
                        const finalCtx = finalCanvas.getContext('2d');
                        
                        const finalImageData = new ImageData(processedAlphaData, finalCanvas.width, finalCanvas.height);
                        finalCtx.putImageData(finalImageData, 0, 0);
                        const trimmed = trimTransparentCanvas(finalCanvas);
                        cropW = trimmed.width;
                        cropH = trimmed.height;
                        transparentDataUrl = trimmed.toDataURL('image/png');
                        
                    } else {
                        const isCloudOrSmoke = layer.name.includes('云') || layer.name.includes('烟') || layer.name.includes('雾') || layer.name.includes('火') || layer.name.includes('光') || layer.name.includes('冰') || layer.name.includes('水') || layer.name.includes('纱') || layer.name.includes('玻璃') || layer.name.includes('透明');
                        const isFineDetail = (layer.name.includes('发') && !layer.name.includes('沙发')) || layer.name.includes('毛') || layer.name.includes('羽') || layer.name.includes('树') || layer.name.includes('草') || layer.name.includes('叶') || layer.name.includes('线') || layer.name.includes('网');

                        // SLOW PATH: Gemini Hybrid
                        const useMaskFormat = isCloudOrSmoke || isFineDetail;
                        
                        const fgChroma = strategy.metrics?.fgChroma || { r: 0, g: 0, b: 0, m: 0 };
                        
                        // predefined optimal backdrops
                        // We select the color with the LEAST maximum chroma conflict in the foreground, 
                        // ensuring we don't accidentally key out glowing edges (like gold -> red conflict).
                        const backdrops = [
                            { name: 'pure, solid green', hex: '#00FF00', rgb: [0, 255, 0], rule: 'Use pure green (#00FF00) ONLY.', conflict: fgChroma.g },
                            { name: 'pure, solid magenta', hex: '#FF00FF', rgb: [255, 0, 255], rule: 'Use pure magenta (#FF00FF) ONLY.', conflict: fgChroma.m },
                            { name: 'pure, solid blue', hex: '#0000FF', rgb: [0, 0, 255], rule: 'Use pure blue (#0000FF) ONLY.', conflict: fgChroma.b },
                            { name: 'pure, solid red', hex: '#FF0000', rgb: [255, 0, 0], rule: 'Use pure red (#FF0000) ONLY.', conflict: fgChroma.r }
                        ];
                        
                        let bestBackdrop = backdrops[0];
                        let minConflict = Infinity;
                        for (const bd of backdrops) {
                            if (bd.conflict < minConflict) {
                                minConflict = bd.conflict;
                                bestBackdrop = bd;
                            }
                        }
                        
                        // Hardcode logic for specific name hints just in case
                        if (layer.name.includes('云') || layer.name.includes('雾') || layer.name.includes('光')) {
                            // For bright soft things, Black backdrop creates highest contrast and works flawlessly with Luminance opacity
                            bestBackdrop = backdrops[1]; // Black / Magenta in original V5 fallback
                        } else if (layer.name.includes('皮') || layer.name.includes('木') || layer.name.includes('人')) {
                            bestBackdrop = backdrops[2]; // Green
                        } else if (layer.name.includes('树') || layer.name.includes('草') || layer.name.includes('叶')) {
                            bestBackdrop = backdrops[3]; // Magenta
                        }
                        
                        const bgColorHex = bestBackdrop.hex;
                        const bgColorName = bestBackdrop.name;
                        const bgColorRule = '4. BACKGROUND RULE: ' + bestBackdrop.rule;
                        
                        let finalPrompt = "";

                        if (useMaskFormat) {
                            // For soft particles/vapors or fine details, use the Mask format
                            if (!customPrompt || customPrompt.includes('optimal background color')) {
                                finalPrompt = `Act as a precise computer vision masking tool.
Your task is to generate a semantic segmentation grayscale mask for the object "${layer.name}" from the provided cropped image.

CRITICAL REQUIREMENTS:
1. OUTPUT FORMAT: Return ONLY a grayscale mask image.
  - Pure White (#FFFFFF) = opaque parts of the object "${layer.name}"
  - Gradients of Gray = semi-transparent edges, fine details (like hair/feathers), or motion blur
  - Pure Black (#000000) = background and everything else
2. NO RENDERING: DO NOT render, composite, or redraw the object. DO NOT cut the object out.
3. PRESERVE DETAILS: Faithfully map every fine tip, hair, and gradient of transparency into grayscale values.
4. STRICT BOUNDARIES: Everything that is not the object MUST be pure black.`;
                            } else {
                                finalPrompt = `${customPrompt}
                                
CRITICAL REQUIREMENTS:
1. OUTPUT FORMAT: Return ONLY a grayscale mask image.
  - Pure White (#FFFFFF) = opaque parts of the object "${layer.name}"
  - Gradients of Gray = semi-transparent edges, fine details, or motion blur
  - Pure Black (#000000) = background
2. NO RENDERING: DO NOT render, composite, or redraw the object. DO NOT cut the object out.
3. PRESERVE DETAILS: You must output smooth grayscale gradients for fuzzy/semi-transparent edges.`;
                            }
                        } else {
                            // Generative Green Screen for hard edge objects
                            finalPrompt = customPrompt || `Act as a precise image extraction tool.
Your task is to extract the object "${layer.name}" from the provided cropped image and place it on a perfectly solid, uniform ${bgColorName} background (${bgColorHex}).

CRITICAL REQUIREMENTS:
1. ZERO HALLUCINATION: You MUST NOT reconstruct, redraw, or alter the object in any way.
2. PIXEL FIDELITY: Strictly preserve the original pixels, material, texture, color, and lighting of the object itself.
3. BACKGROUND ONLY: Your ONLY task is to change the background surrounding the object to a mathematically solid ${bgColorName} color (${bgColorHex}). Do not touch the object itself.
${bgColorRule}`;
                        }
                        // ----------------------

                        const croppedDataUrl = cropCanvas.toDataURL('image/png');
                        const croppedFile = await dataURLToFile(croppedDataUrl, `cropped-${Date.now()}.png`);
                        
                        // 1. 使用 Gemini 进行独立资产重建
                        tempMsg.update(`✨ **正在重建**: "${layer.name}"... AI 正在根据实体状态补全遮挡并净化背景。`);
                        const geminiResult = await editOrQueryImageWithGemini(`[LAYER_EXPLOSION_OBJECT_RECONSTRUCT:${layer.name}] ${finalPrompt}`, croppedFile);
                        if (!geminiResult.success || !geminiResult.imageData) {
                            throw new Error("Gemini 资产重建失败");
                        }
                        const reconstructedDataUrl = `data:${geminiResult.mimeType};base64,${geminiResult.imageData}`;
                        
                        // 2. 自定义抠图 Pipeline (Matte + Despill + Fallback)
                        tempMsg.update(`✨ **正在提取**: "${layer.name}"... 执行纯色背景抹除计算。`);
                        
                        // 加载 Gemini 返回的图片
                        const reconstructedImg = new Image();
                        reconstructedImg.src = reconstructedDataUrl;
                        await new Promise(r => {
                            reconstructedImg.onload = r;
                            reconstructedImg.onerror = () => r();
                        });
                        
                        const finalCanvas = document.createElement('canvas');
                        finalCanvas.width = cropW;
                        finalCanvas.height = cropH;
                        const finalCtx = finalCanvas.getContext('2d');
                        finalCtx.imageSmoothingEnabled = true;
                        finalCtx.imageSmoothingQuality = 'high';
                        
                        // IMPORTANT: We scale Gemini's returned image to match our upscaled bounds!
                        finalCtx.drawImage(reconstructedImg, 0, 0, cropW, cropH);
                        
                        // Fetch ORIGINAL pixel data to preserve authentic colors without AI compositing!
                        const originalImageDataObj = ctx.getImageData(0, 0, cropW, cropH);
                        
                        // Fetch Gemini Generated pixel data
                        const generatedImageDataObj = finalCtx.getImageData(0, 0, cropW, cropH);
                        
                        if (useMaskFormat) {
                            processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                                taskId, 
                                originalImageDataObj, // ORIGINAL PIXELS
                                reconstructedDataUrl, // Gemini's grayscale mask
                                layer.name, 
                                (msg) => tempMsg.update(msg)
                            );
                        } else {
                            processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                                taskId, 
                                generatedImageDataObj, // Use Gemini's clean output for keys
                                null, 
                                layer.name, 
                                (msg) => tempMsg.update(msg),
                                { type: 'solid', bgColor: bestBackdrop.rgb }
                            );
                        }
                        
                        // Construct the final output image from the returned Alpha array
                        const finalImageData = new ImageData(processedAlphaData, finalCanvas.width, finalCanvas.height);
                        finalCtx.putImageData(finalImageData, 0, 0);
                        
                        const trimmed = trimTransparentCanvas(finalCanvas);
                        cropW = trimmed.width;
                        cropH = trimmed.height;
                        transparentDataUrl = trimmed.toDataURL('image/png');
                    }
                    
                    return { status: 'fulfilled', value: { success: true, dataUrl: transparentDataUrl, width: cropW, height: cropH }, layerName: layer.name };
                } catch (err) {
                    return { status: 'rejected', reason: err, layerName: layer.name };
                }
            })());

            // 4. 等待所有提取完成
            const results = await runLimitedExtractionJobs(extractionJobs, 1);
            console.log(`[Explosion] Stage: extraction done (${itemId}) duration=${Date.now() - extractionStartedAt}ms`);
            updateDiagnosticTask('extraction_done', {
                itemId,
                durationMs: Date.now() - extractionStartedAt,
                fulfilled: results.filter(result => result.status === 'fulfilled').length,
                rejected: results.filter(result => result.status === 'rejected').length
            });
            
            // Collect layers for clean plate, passing the whole layer bounding box for text layers for stable mask generation
            const cleanupLayers = [];
            const textCleanupLayers = [];
            let hasTextLayers = false;
            for (let i = 0; i < results.length; i++) {
                const res = results[i];
                if (res.status === 'fulfilled' && res.value.success) {
                    if (res.value.isText) {
                        hasTextLayers = true;
                        if (Array.isArray(res.value.textLines) && res.value.textLines.length > 0) {
                            res.value.textLines.forEach(line => {
                                textCleanupLayers.push({
                                    bbox: line.bbox || res.value.layerBbox,
                                    name: `Text: "${line.textContent || res.layerName}"`,
                                    semanticType: 'element_text',
                                    isText: true
                                });
                            });
                        } else {
                            const cleanupLayer = getCleanupLayerForEditableLayer(item, currentLayers[i], { preferEditableTextBbox: true });
                            textCleanupLayers.push({
                                bbox: cleanupLayer?.bbox || res.value.layerBbox,
                                name: `Text Container: "${res.layerName}"`,
                                semanticType: 'element_text',
                                isText: true
                            });
                        }
                    } else if (res.value.completionDeferred || res.value.isFlatDesignLayer || res.value.shouldGenerateRuntimeLayer === false || res.value.runtimeAction === 'hold') {
                        continue;
                    } else {
                        cleanupLayers.push(getCleanupLayerForEditableLayer(item, currentLayers[i], { preferEditableTextBbox: false }));
                    }
                }
            }
            if (textCleanupLayers.length > 0) {
                const seenTextBboxes = new Set();
                const dedupedTextCleanupLayers = textCleanupLayers.filter(layer => {
                    const key = Array.isArray(layer.bbox)
                        ? layer.bbox.map(value => Math.round(value)).join(',')
                        : '';
                    if (!key || seenTextBboxes.has(key)) return false;
                    seenTextBboxes.add(key);
                    return true;
                });
                cleanupLayers.unshift(...dedupedTextCleanupLayers);
            }
            
            // 5. 执行背景净化 (Clean Plate)
            if (cleanupLayers.length > 0 && deferredCompletionTargetIds.size === 0) {
                tempMsg?.update?.(`🧹 **正在净化底板**：联合抹除已提取的 ${cleanupLayers.length} 个元素...`);
                const baseBg = item.originalDataUrl || item.dataUrl;
                const backgroundHint = getBackgroundSemanticHint(item);
                
                const customPrompt = undefined;
                console.log(`[Explosion] Stage: clean plate start (${itemId}) cleanupLayers=${cleanupLayers.length} backgroundHint="${backgroundHint}"`);
                updateDiagnosticTask('clean_plate_start', { itemId, count: cleanupLayers.length });
                const cleanPlateStartedAt = Date.now();
                const cleanedBg = await cleanMultipleBackgrounds(baseBg, cleanupLayers, customPrompt, {
                    preserveBackgroundOnly: true,
                    backgroundHint,
                    useSimpleSceneCleanPlatePrompt: true
                });
                console.log(`[Explosion] Stage: clean plate done (${itemId}) duration=${Date.now() - cleanPlateStartedAt}ms`);
                updateDiagnosticTask('clean_plate_done', {
                    itemId,
                    durationMs: Date.now() - cleanPlateStartedAt,
                    hasResult: Boolean(cleanedBg)
                });
                
                if (cleanedBg) {
                    item.cleanPlateDataUrl = cleanedBg;
                    item.dataUrl = cleanedBg;
                    item.cleanPlateStatus = 'ready';
                    recordWorkspaceAction(state, {
                        actionName: 'clean_plate_completed',
                        itemId,
                        status: 'completed',
                        hasResult: true
                    });
                    
                    // Hide only layers that actually produced standalone runtime assets.
                    // Low-quality/hold candidates must stay visible so motion can still use semantic fallback.
                    currentLayers.forEach((layerObj, index) => {
                        const res = results[index];
                        const shouldHideSemanticLayer = !!(
                            res?.status === 'fulfilled' &&
                            res?.value?.success &&
                            (
                                res.value.isText ||
                                (
                                    res.value.dataUrl &&
                                    (
                                        (
                                            res.value.shouldGenerateRuntimeLayer !== false &&
                                            res.value.runtimeAction !== 'hold'
                                        ) ||
                                        shouldForceRuntimeAssetForHeldSegmentation(layerObj, res.value)
                                    ) &&
                                    !res.value.isFlatDesignLayer
                                )
                            )
                        );
                        if (!shouldHideSemanticLayer) return;
                        const layerIndex = allLayers.findIndex(l => (l.name || l) === layerObj.name);
                        if (layerIndex >= 0) {
                            updateLayerState(itemId, layerIndex, { visible: false, selected: false });
                        }
                    });
                    renderLayerList(allLayers, itemId);
                    renderCanvasLayers(itemId);
                    await persistLayerStateToRuntime(itemId, item);
                    if (window.historyManager && activeAgentLayerExtraction === 0) window.historyManager.pushState();

                    // Update the base image element in place
                    if (item.el) {
                        const imgEl = item.el.querySelector('.crop-container > img');
                        if (imgEl) {
                            imgEl.src = cleanedBg;
                        }
                    }
                } else {
                    console.warn("[Background Purification] Could not clean the backgrounds. Using original plate.");
                    item.cleanPlateStatus = 'failed';
                    recordWorkspaceAction(state, {
                        actionName: 'clean_plate_completed',
                        itemId,
                        status: 'failed',
                        hasResult: false
                    });
                    addMessage({
                        sender: 'bot',
                        type: 'text',
                        content: '⚠️ 底板净化失败，原图未被修改；已提取的透明图层仍可继续使用。'
                    });
                }
            } else if (deferredCompletionTargetIds.size > 0) {
                console.info('[Explosion] clean plate deferred until scene-completed targets are extracted', {
                    itemId,
                    deferredTargetLayerIds: [...deferredCompletionTargetIds]
                });
            } else {
                console.warn("[Background Purification] No successfully extracted layers to clean up.");
            }
            console.log(`[Explosion] Pipeline complete (${itemId}) duration=${Date.now() - pipelineStartedAt}ms`);
            
            // 6. 将结果添加到工作台
            const baseX = parseFloat(item.el.style.left) || 0;
            const baseY = parseFloat(item.el.style.top) || 0;
            const itemWidth = parseFloat(item.el.style.width) || 300;
            const itemHeight = parseFloat(item.el.style.height) || 300;
            
            let successCount = 0;
            let heldCount = 0;
            updateDiagnosticTask('runtime_render_start', { itemId, successCount, heldCount });

            const { addTextNoteToWorkbench } = await import('./workbench/notes.js');
            const allExtractedTextLines = results
                .filter(res => res.status === 'fulfilled' && res.value?.success && res.value?.isText)
                .flatMap(res => Array.isArray(res.value.textLines) ? res.value.textLines : []);
            const containerCandidates = allExtractedTextLines.length > 0
                ? await prepareTextContainerCandidates(item, allExtractedTextLines, originalTextSource)
                : [];
            if (allExtractedTextLines.length > 0) {
                await restoreTextContainerShapes({
                    item,
                    itemId,
                    textLines: allExtractedTextLines,
                    baseX,
                    baseY,
                    parentWidth: itemWidth,
                    parentHeight: itemHeight,
                    sourceImage: originalTextSource,
                    zIndexBase: parseInt(item.el.style.zIndex || 0) + 1
                });
            }

            for (let i = 0; i < results.length; i++) {
                const res = results[i];
                const layerObj = currentLayers[i];
                
                if (res.status === 'fulfilled' && res.value.success) {
                    if (res.value.isText) {
                        for (const [index, lineObj] of res.value.textLines.entries()) {
                            const textState = await buildExtractedTextState({
                                lineObj,
                                layerObj,
                                fallbackBbox: res.value.layerBbox,
                                containerCandidates,
                                allTextLines: res.value.textLines,
                                index,
                                baseX,
                                baseY,
                                parentWidth: itemWidth,
                                parentHeight: itemHeight,
                                zIndex: getExtractedLayerZIndex(item, layerObj, i),
                                parentId: itemId,
                                sourceImage: originalTextSource,
                                minWidth: 20,
                                minHeight: 10
                            });
                            const textStates = Array.isArray(textState) ? textState : [textState];
                            textStates.forEach(state => addTextNoteToWorkbench(0, 0, state));
                        }
                        successCount++;
                    } else if (res.value.isFlatDesignLayer) {
                        heldCount++;
                        updateLayerExtractionMetadata(item, {
                            id: layerObj.id,
                            name: layerObj.name,
                            cleanPlateLayerId: layerObj.cleanPlateLayerId,
                            sourceTextLayerId: layerObj.sourceTextLayerId
                        }, {
                            extractEngine: res.value.extractEngine || 'runtime_vector_pending',
                            quality: res.value.quality,
                            bbox: res.value.layerBbox || layerObj.bbox
                        });
                        continue;
                    } else if (res.value.dataUrl) {
                    const isHeld = res.value.shouldGenerateRuntimeLayer === false || res.value.runtimeAction === 'hold';
                    if (isHeld) {
                        heldCount++;
                        updateLayerExtractionMetadata(item, {
                            id: layerObj.id,
                            name: layerObj.name,
                            cleanPlateLayerId: layerObj.cleanPlateLayerId,
                            sourceTextLayerId: layerObj.sourceTextLayerId
                        }, {
                            extractEngine: res.value.extractEngine || 'fastsam',
                            quality: res.value.quality || {
                                status: 'low_quality',
                                runtimeAction: 'hold',
                                reason: 'quality_gate_hold'
                            },
                            bbox: res.value.segmentedBbox || layerObj.bbox,
                            cutoutUrl: res.value.dataUrl || null,
                            previewUrl: res.value.dataUrl || null
                        });

                        if (shouldForceRuntimeAssetForHeldSegmentation(layerObj, res.value)) {
                            const f = await dataURLToFile(res.value.dataUrl, `explode-${res.layerName}-${Date.now()}.png`);
                            const originalBbox = res.value.segmentedBbox || layerObj.bbox;
                            const layerRect = bboxToWorkbenchRect(originalBbox, baseX, baseY, itemWidth, itemHeight, 1, 1);

                            await addImageToWorkbench(f, `拆解-${res.layerName}`, {
                                x: layerRect.left,
                                y: layerRect.top,
                                initialWidth: layerRect.width,
                                initialHeight: layerRect.height,
                                parentId: itemId,
                                sourceLayerId: layerObj.id,
                                originalBbox,
                                layerName: res.layerName,
                                type: 'layer-explode',
                                zIndex: getExtractedLayerZIndex(item, layerObj, i),
                                extractEngine: res.value.extractEngine || 'fastsam',
                                quality: res.value.quality || null,
                                skipRuntimeSnapshot: true,
                                autoOpenDecisionPanel: true,
                                autoOpenDecisionPanelBatchToken: batchAutoOpenToken,
                                autoOpenDecisionPanelBatchFinal: i === currentLayers.length - 1
                            });
                            successCount++;
                            console.log(`[Explosion] Forced runtime child created for held raster layer: ${res.layerName}`);
                        }
                        continue;
                    }

                    const f = await dataURLToFile(res.value.dataUrl, `explode-${res.layerName}-${Date.now()}.png`);
                    const originalBbox = res.value.segmentedBbox || layerObj.bbox;
                    const layerRect = bboxToWorkbenchRect(originalBbox, baseX, baseY, itemWidth, itemHeight, 1, 1);
                    
                    await addImageToWorkbench(f, `拆解-${res.layerName}`, { 
                        x: layerRect.left,
                        y: layerRect.top,
                        initialWidth: layerRect.width,
                        initialHeight: layerRect.height,
                        parentId: itemId,
                        sourceLayerId: layerObj.id,
                        originalBbox,
                        layerName: res.layerName,
                        type: 'layer-explode',
                        zIndex: getExtractedLayerZIndex(item, layerObj, i),
                        extractEngine: res.value.extractEngine || 'fastsam',
                        quality: res.value.quality || null,
                        skipRuntimeSnapshot: true,
                        autoOpenDecisionPanel: true,
                        autoOpenDecisionPanelBatchToken: batchAutoOpenToken,
                        autoOpenDecisionPanelBatchFinal: i === currentLayers.length - 1
                    });

                    updateLayerExtractionMetadata(item, {
                        id: layerObj.id,
                        name: layerObj.name,
                        cleanPlateLayerId: layerObj.cleanPlateLayerId,
                        sourceTextLayerId: layerObj.sourceTextLayerId
                    }, {
                        extractEngine: res.value.extractEngine || 'fastsam',
                        quality: res.value.quality || {
                            status: 'ok',
                            score: null,
                            reason: 'fastsam_batch_extracted'
                        },
                        bbox: res.value.segmentedBbox || layerObj.bbox,
                        cutoutUrl: res.value.dataUrl,
                        previewUrl: res.value.dataUrl
                    });
                    
                    successCount++;
                }
            } else {
                console.warn(`Layer extraction failed for ${res.layerName}:`, res.reason);
            }
            }

            // Completion is fully automatic for verified hard-object occlusions.
            // The replacement keeps the original child z-index, so foreground
            // occluders remain in front of the completed asset.
            let completionResult = { completed: 0, skipped: 0 };
            try {
                updateDiagnosticTask('completion_start', { itemId, targetCount: deferredCompletionTargetIds.size });
                completionResult = await runAutomaticCompletionForItem(itemId, {
                    onProgress: message => tempMsg?.update?.(`✨ **自动补全**：${message}`),
                    allowDeferredTarget: deferredCompletionTargetIds.size > 0,
                    targetLayerIds: [...deferredCompletionTargetIds]
                });
                updateDiagnosticTask('completion_done', { itemId, ...completionResult });
            } catch (error) {
                recordDiagnosticBreadcrumb('completion:error', { itemId, message: error?.message || String(error) });
                console.warn('[Automatic Completion] validation or execution skipped:', error);
            }

            if (deferredCompletionTargetIds.size > 0 && (completionResult.completed > 0 || cleanupLayers.length > 0)) {
                updateDiagnosticTask('final_clean_plate_start', { itemId });
                const completedTargetLayerIds = new Set(completionResult.completedTargetLayerIds || []);
                const completedTargetLayers = currentLayers.filter(layer => completedTargetLayerIds.has(layer.id));
                const finalCleanupLayers = [...cleanupLayers];
                completedTargetLayers.forEach(layer => {
                    const cleanupLayer = getCleanupLayerForEditableLayer(item, layer, { preferEditableTextBbox: false });
                    if (cleanupLayer?.bbox) finalCleanupLayers.push(cleanupLayer);
                });
                const dedupedCleanupLayers = finalCleanupLayers.filter((layer, index, layers) => {
                    const key = Array.isArray(layer?.bbox) ? layer.bbox.map(value => Math.round(value)).join(',') : '';
                    return key && layers.findIndex(candidate =>
                        Array.isArray(candidate?.bbox) && candidate.bbox.map(value => Math.round(value)).join(',') === key
                    ) === index;
                });

                if (dedupedCleanupLayers.length > 0) {
                    tempMsg?.update?.(`🧹 **正在净化底板**：联合抹除 ${dedupedCleanupLayers.length} 个完整实体...`);
                    const backgroundHint = getBackgroundSemanticHint(item);
                    console.info('[Explosion] final clean plate after deferred completion', {
                        itemId,
                        partial: completionResult.completed < deferredCompletionTargetIds.size,
                        completedTargetLayerIds: [...completedTargetLayerIds],
                        skippedTargetLayerIds: completionResult.skippedTargetLayerIds || [],
                        cleanupLayerIds: dedupedCleanupLayers.map(layer => layer.id || layer.name)
                    });
                    const cleanedBg = await cleanMultipleBackgrounds(item.originalDataUrl || item.dataUrl, dedupedCleanupLayers, undefined, {
                        preserveBackgroundOnly: true,
                        backgroundHint,
                        useSimpleSceneCleanPlatePrompt: true
                    });
                    if (cleanedBg) {
                        item.cleanPlateDataUrl = cleanedBg;
                        item.dataUrl = cleanedBg;
                        item.cleanPlateStatus = 'ready';
                        currentLayers.forEach((layerObj, index) => {
                            const included = dedupedCleanupLayers.some(candidate => candidate.id === layerObj.id);
                            if (included) updateLayerState(itemId, index, { visible: false, selected: false });
                        });
                        if (item.el) {
                            const image = item.el.querySelector('.crop-container > img');
                            if (image) image.src = cleanedBg;
                        }
                        recordWorkspaceAction(state, {
                            actionName: 'clean_plate_completed',
                            itemId,
                            status: 'completed',
                            hasResult: true
                        });
                        renderLayerList(allLayers, itemId);
                        renderCanvasLayers(itemId);
                        await persistLayerStateToRuntime(itemId, item);
                        updateDiagnosticTask('final_clean_plate_done', { itemId, hasResult: true });
                    } else {
                        item.cleanPlateStatus = 'failed';
                        console.warn('[Explosion] final clean plate failed after deferred completion', { itemId });
                    }
                }
            } else if (deferredCompletionTargetIds.size > 0) {
                console.warn('[Explosion] final clean plate skipped because deferred completion did not fully succeed', {
                    itemId,
                    expected: deferredCompletionTargetIds.size,
                    completed: completionResult.completed,
                    skipped: completionResult.skipped
                });
            }

            if (tempMsg) tempMsg.remove();
            updateDiagnosticTask('runtime_render_done', { itemId, successCount, heldCount });
            if (successCount > 0) {
                const heldText = heldCount > 0 ? `，${heldCount} 个低质量候选已保留为待高精度处理` : '';
                const completionText = completionResult.completed > 0
                    ? `，已自动补全 ${completionResult.completed} 个被遮挡实体`
                    : '';
                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat('语义拆解', `拆解了 ${successCount}/${layerNames.length} 个图层${heldText}${completionText}: ${layerNames.join(', ')}`, item.dataUrl || item.file, executeLayerExplosion);
                } else {
                    addMessage({ sender: 'bot', type: 'text', content: `✅ **拆解完成**！成功提取并还原了 ${successCount}/${layerNames.length} 个完整图层${heldText}${completionText}。` });
                }
            } else if (heldCount > 0) {
                addMessage({ sender: 'bot', type: 'text', content: `⚠️ **拆解已暂停生成图层**：${heldCount} 个候选质量不足，已标记为需要高精度模型处理。` });
            } else {
                addMessage({ sender: 'bot', type: 'text', content: `❌ 拆解失败，未能提取任何图层。` });
            }
            releaseMagicLayersSceneLock(item, parentImg);
            finishDiagnosticTask('completed', { itemId, successCount, heldCount, completion: completionResult });
            return {
                success: successCount > 0,
                sourceAssetId: itemId,
                extractedLayerIds: currentLayers
                    .filter((layer, index) => results[index]?.status === 'fulfilled' && results[index]?.value?.dataUrl)
                    .map(layer => layer.id)
                    .filter(Boolean),
                successCount,
                heldCount
            };

        } catch (e) {
            console.error("[Explosion] Fatal error:", e);
            releaseMagicLayersSceneLock(item);
            finishDiagnosticTask('failed', { itemId, message: e?.message || String(e) });
            recordDiagnosticBreadcrumb('magic_layers:fatal_error', { itemId, message: e?.message || String(e), stack: e?.stack || '' });
            if (tempMsg) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `❌ 拆解过程出错: ${e.message}` });
            return {
                success: false,
                sourceAssetId: itemId,
                extractedLayerIds: [],
                successCount: 0,
                heldCount: 0,
                error: e?.message || String(e)
            };
        }
    };

    return executeLayerExplosion();
}

/**
 * Runtime adapter for the existing Magic Layers implementation. Commands
 * receive only IDs and result references; the pixel pipeline remains here.
 */
export async function executeAgentLayerExtraction({ workspace, jobId, assetId, layerIds = [] }) {
    if (!workspace || workspace !== window.mvrRuntime?.getCurrentWorkspace?.()) {
        throw new Error('Magic Layers extraction requires the current workspace.');
    }
    const item = workbenchItems.get(assetId);
    if (!item) throw new Error(`Workbench source asset ${assetId} is unavailable.`);
    if (!workspace.currentState.assetRegistry.get(assetId)) {
        throw new Error(`Runtime source asset ${assetId} is unavailable.`);
    }

    const beforeIds = new Set(workbenchItems.keys());
    const beforeLayerStates = item.layerStates instanceof Map
        ? new Map([...item.layerStates.entries()].map(([index, value]) => [index, { ...value }]))
        : null;
    const beforeLegacyMetadata = captureAgentLayerLegacyMetadata(item);
    const operationId = `magic_layers_${jobId}_${Date.now()}`;
    const beforeRuntimeMetadata = captureAgentLayerRuntimeMetadata(workspace.currentState.assetRegistry.get(assetId));
    activeAgentLayerExtraction += 1;
    let result;
    try {
        result = await triggerLayerExplosion(assetId, {
            layerIds,
            selectAll: layerIds.length === 0
        });
    } catch (error) {
        agentLayerExtractionOperations.set(operationId, {
            operationId,
            jobId,
            sourceAssetId: assetId,
            extractedAssetIds: [...workbenchItems.keys()].filter(id => !beforeIds.has(id)),
            item,
            beforeLayerStates,
            beforeRuntimeMetadata,
            beforeLegacyMetadata
        });
        await undoAgentLayerExtraction({ workspace, operationId });
        throw error;
    } finally {
        activeAgentLayerExtraction = Math.max(0, activeAgentLayerExtraction - 1);
    }

    if (!result?.success) {
        agentLayerExtractionOperations.set(operationId, {
            operationId,
            jobId,
            sourceAssetId: assetId,
            extractedAssetIds: [...workbenchItems.keys()].filter(id => !beforeIds.has(id)),
            item,
            beforeLayerStates,
            beforeRuntimeMetadata,
            beforeLegacyMetadata
        });
        await undoAgentLayerExtraction({ workspace, operationId });
        throw new Error(result?.error || 'Magic Layers produced no standalone asset.');
    }

    const extractedAssetIds = [...workbenchItems.keys()].filter(id => !beforeIds.has(id));
    agentLayerExtractionOperations.set(operationId, {
        operationId,
        jobId,
        sourceAssetId: assetId,
        extractedAssetIds,
        item,
        beforeLayerStates,
        beforeRuntimeMetadata,
        beforeLegacyMetadata
    });

    return {
        operationId,
        sourceAssetId: assetId,
        extractedAssetIds,
        extractedLayerIds: result.extractedLayerIds || [],
        extractedCount: extractedAssetIds.length,
        heldCount: result.heldCount || 0
    };
}

export async function commitAgentLayerExtraction({ workspace, operationIds = [] }) {
    for (const operationId of operationIds) {
        const operation = agentLayerExtractionOperations.get(operationId);
        if (!operation) continue;
        const item = workbenchItems.get(operation.sourceAssetId) || operation.item;
        if (!item) throw new Error(`Magic Layers source asset ${operation.sourceAssetId} is unavailable at commit.`);

        await Promise.all(operation.extractedAssetIds.map(assetId => waitForWorkbenchItemPersistence(assetId)));
        const extractedAssets = operation.extractedAssetIds
            .map(assetId => workspace?.currentState?.assetRegistry?.get(assetId))
            .filter(Boolean);
        if (extractedAssets.some(asset => /^(data:|blob:)/i.test(String(asset.sourceImage || '')))) {
            throw new Error('Magic Layers extracted asset has no durable OSS image reference.');
        }

        if (typeof item.cleanPlateDataUrl === 'string' && item.cleanPlateDataUrl.startsWith('data:')) {
            const cleanPlateFile = await dataURLToFile(item.cleanPlateDataUrl, `agent-clean-plate-${operation.sourceAssetId}.png`);
            const durableCleanPlateUrl = await uploadImageToOSS(cleanPlateFile);
            item.cleanPlateDataUrl = durableCleanPlateUrl;
            item.dataUrl = durableCleanPlateUrl;
        }
        await persistLayerStateToRuntime(operation.sourceAssetId, item, { persistSession: false });
    }
}

export async function releaseAgentLayerExtraction({ operationIds = [] }) {
    operationIds.forEach(operationId => agentLayerExtractionOperations.delete(operationId));
}

export async function undoAgentLayerExtraction({ workspace, operationId }) {
    const operation = agentLayerExtractionOperations.get(operationId);
    if (!operation) return;
    const previousAgentFlag = window.__marmoAgentLayerExtractionActive;
    window.__marmoAgentLayerExtractionActive = true;
    try {
        for (const assetId of [...operation.extractedAssetIds].reverse()) {
            if (typeof window.deleteWorkbenchItem === 'function') {
                await window.deleteWorkbenchItem(assetId, true, true);
            }
        }

        // The adapter deliberately withheld parent persistence. Restore the
        // in-memory legacy item from the untouched Runtime metadata.
        const parentItem = workbenchItems.get(operation.sourceAssetId);
        const parentAsset = workspace.currentState.assetRegistry.get(operation.sourceAssetId);
        if (parentItem && parentAsset) {
            Object.assign(parentItem, {
                layers: parentAsset.layers,
                scene: parentAsset.scene,
                semanticViews: parentAsset.semanticViews,
                hasFullSemanticAnalysis: parentAsset.hasFullSemanticAnalysis,
                originalDataUrl: parentAsset.originalDataUrl,
                cleanPlateDataUrl: parentAsset.cleanPlateDataUrl,
                cleanPlateStatus: parentAsset.cleanPlateStatus,
                ...(operation.beforeLegacyMetadata || {}),
                layerStates: operation.beforeLayerStates
                    ? new Map([...operation.beforeLayerStates.entries()].map(([index, value]) => [index, { ...value }]))
                    : undefined
            });
            if (operation.beforeRuntimeMetadata) {
                workspace.dispatcher.dispatch({
                    type: 'UPDATE_ASSET_METADATA',
                    meta: { silent: true, skipSnapshot: true, skipNotify: true },
                    payload: {
                        uid: operation.sourceAssetId,
                        ...operation.beforeRuntimeMetadata
                    }
                });
            }
            renderLayerList(getItemSceneLayers(parentItem), operation.sourceAssetId);
            renderCanvasLayers(operation.sourceAssetId);
            const parentImage = parentItem.el?.querySelector('.crop-container > img');
            if (parentImage) parentImage.src = parentItem.cleanPlateDataUrl || parentItem.dataUrl || '';
        }
        if (typeof window.reconcileAllAssets === 'function') window.reconcileAllAssets();
    } finally {
        window.__marmoAgentLayerExtractionActive = previousAgentFlag;
        agentLayerExtractionOperations.delete(operationId);
    }
}

function captureAgentCapabilitySnapshot({ workspace, itemId, childId }) {
    const item = itemId ? workbenchItems.get(itemId) : null;
    const child = childId ? workbenchItems.get(childId) : null;
    const assetIds = [...new Set([itemId, childId].filter(Boolean))];
    return {
        itemId,
        childId,
        item,
        child,
        beforeWorkbenchIds: new Set(workbenchItems.keys()),
        itemSnapshot: item ? {
            dataUrl: item.dataUrl,
            originalDataUrl: item.originalDataUrl,
            cleanPlateDataUrl: item.cleanPlateDataUrl,
            cleanPlateStatus: item.cleanPlateStatus,
            layers: cloneSerializable(item.layers),
            scene: cloneSerializable(item.scene),
            semanticViews: cloneSerializable(item.semanticViews),
            hasFullSemanticAnalysis: item.hasFullSemanticAnalysis
        } : null,
        childSnapshot: child ? cloneSerializable({
            dataUrl: child.dataUrl,
            originalDataUrl: child.originalDataUrl,
            previewUrl: child.previewUrl,
            runtimeDisplayUrl: child.runtimeDisplayUrl,
            file: null,
            assetStatus: child.assetStatus,
            activeVersionId: child.activeVersionId,
            versions: child.versions,
            left: child.left,
            top: child.top,
            width: child.width,
            height: child.height,
            zIndex: child.zIndex,
            originalBbox: child.originalBbox,
            extractionBbox: child.extractionBbox,
            canonicalBbox: child.canonicalBbox,
            completionAssetId: child.completionAssetId
        }) : null,
        runtimeAssets: new Map(assetIds
            .map(uid => [uid, workspace.currentState.assetRegistry.get(uid)])
            .filter(([, asset]) => Boolean(asset))
            .map(([uid, asset]) => [uid, cloneSerializable(asset)]))
    };
}

function summarizeCapabilityQuality(quality) {
    if (!quality) return null;
    return {
        status: quality.status || null,
        runtimeAction: quality.runtimeAction || null,
        score: Number.isFinite(Number(quality.score)) ? Number(quality.score) : null,
        reason: quality.reason || null,
        issues: Array.isArray(quality.issues) ? quality.issues.slice(0, 8).map(String) : []
    };
}

function restoreAgentCapabilitySnapshot(operation, workspace) {
    const currentIds = [...workbenchItems.keys()];
    for (const assetId of currentIds) {
        if (!operation.beforeWorkbenchIds.has(assetId)) {
            const item = workbenchItems.get(assetId);
            item?.el?.remove();
            workbenchItems.delete(assetId);
            state.selectedWorkbenchItems.delete(assetId);
            if (workspace.currentState.assetRegistry.get(assetId)) {
                workspace.dispatcher.dispatch({
                    type: 'REMOVE_ASSET',
                    meta: { silent: true, skipSnapshot: true, skipNotify: true },
                    payload: { uid: assetId }
                });
            }
        }
    }

    if (operation.item && operation.itemSnapshot) {
        Object.assign(operation.item, operation.itemSnapshot);
        const image = operation.item.el?.querySelector('.crop-container > img');
        if (image) image.src = operation.item.cleanPlateDataUrl || operation.item.dataUrl || '';
    }
    if (operation.child && operation.childSnapshot && workbenchItems.has(operation.childId)) {
        Object.assign(operation.child, operation.childSnapshot);
        const image = operation.child.el?.querySelector('.crop-container > img') || operation.child.el?.querySelector('img');
        if (image) image.src = operation.child.runtimeDisplayUrl || operation.child.dataUrl || '';
    }

    operation.runtimeAssets.forEach((asset, uid) => {
        if (!workspace.currentState.assetRegistry.get(uid)) return;
        workspace.dispatcher.dispatch({
            type: 'UPDATE_ASSET_METADATA',
            meta: { silent: true, skipSnapshot: true, skipNotify: true },
            payload: asset
        });
    });
    if (operation.itemId && operation.item) {
        renderLayerList(getItemSceneLayers(operation.item), operation.itemId);
        renderCanvasLayers(operation.itemId);
    }
}

/**
 * Runtime adapter for AI capabilities. Existing completion/edit algorithms
 * remain the source of truth; this adapter only provides transaction staging,
 * verification references, and one commit boundary.
 */
export async function executeAgentCapability({ workspace, jobId, capabilityType, targetAssetIds = [], params = {} }) {
    if (!workspace || workspace !== window.mvrRuntime?.getCurrentWorkspace?.()) {
        throw new Error('Agent capability requires the current workspace.');
    }

    const itemId = params.itemId || params.parentItemId || workbenchItems.get(params.childId)?.parentId;
    const childId = params.childId;
    const item = itemId ? workbenchItems.get(itemId) : null;
    if (!item && capabilityType === 'object_completion') throw new Error(`Completion source asset ${itemId} is unavailable.`);
    if (capabilityType === 'replace_material' && !childId) throw new Error('Material replacement requires a child asset.');

    const operationId = `capability_${jobId}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const operation = captureAgentCapabilitySnapshot({ workspace, itemId, childId });
    operation.operationId = operationId;
    operation.jobId = jobId;
    operation.capabilityType = capabilityType;

    try {
        if (capabilityType === 'object_completion') {
            const completionAssetId = params.completionAssetId;
            if (!completionAssetId) throw new Error('Object completion requires a completion asset.');
            const completion = await executeObjectCompletion(item, completionAssetId, {
                onProgress: message => console.info('[Agent capability]', message)
            });
            if (!completion?.success || !completion.canonicalFile) {
                throw new Error('Object completion did not produce a canonical asset.');
            }
            const replacement = await replaceSplitChildWithCanonicalAsset(
                itemId,
                item,
                completion.asset,
                completion.canonicalFile,
                { persist: false }
            );
            const resultChildId = replacement?.childId || getWorkbenchItemId(replacement?.child);
            if (!resultChildId) throw new Error('Object completion produced no Workspace asset.');
            operation.targetAssetIds = [resultChildId];
            agentCapabilityOperations.set(operationId, operation);
            return {
                operationId,
                targetAssetIds: [resultChildId],
                outputRefs: {
                    capabilityType,
                    sourceAssetId: itemId,
                    completionAssetId,
                    childAssetId: resultChildId,
                    parentLayerId: replacement.targetLayer?.id || null,
                    versionId: replacement.targetLayer?.activeVersionId || null,
                    quality: summarizeCapabilityQuality(completion.quality)
                }
            };
        }

        if (capabilityType === 'replace_material') {
            const result = await handleIsolatedAssetEdit(childId, params.prompt, {
                replaceCurrent: true,
                persist: false,
                recordHistory: false,
                suppressMessages: true,
                throwOnError: true,
                source: 'agent_capability'
            });
            if (!result?.success) throw new Error(result?.error || 'Material replacement produced no result.');
            operation.targetAssetIds = [childId];
            agentCapabilityOperations.set(operationId, operation);
            return {
                operationId,
                targetAssetIds: [childId],
                outputRefs: {
                    capabilityType,
                    childAssetId: childId,
                    parentLayerId: result.parentLayerId || null,
                    versionId: result.versionId || null
                }
            };
        }

        if (capabilityType === 'edit_asset') {
            const editItemId = params.itemId || targetAssetIds[0];
            const editItem = editItemId ? workbenchItems.get(editItemId) : null;
            if (!editItem) throw new Error('Object editing requires a Workspace asset.');
            const mode = params.mode === 'current_layer' ? 'current_layer' : 'variant';
            if (mode === 'current_layer' && (!editItem.parentId || !hasSplitLayerPlacement(editItem) || !resolveParentLayerForAsset(workbenchItems.get(editItem.parentId), editItem))) {
                throw new Error('当前对象没有可回写的父图层，只能生成新版本。');
            }
            const result = await handleIsolatedAssetEdit(editItemId, params.prompt, {
                replaceCurrent: mode === 'current_layer',
                persist: false,
                recordHistory: false,
                suppressMessages: true,
                throwOnError: true,
                source: 'agent_object_editor'
            });
            if (!result?.success) throw new Error(result?.error || '对象编辑没有生成有效结果。');
            const resultAssetId = result.assetId || result.childId || editItemId;
            operation.targetAssetIds = [resultAssetId];
            agentCapabilityOperations.set(operationId, operation);
            return {
                operationId,
                targetAssetIds: [resultAssetId],
                outputRefs: {
                    capabilityType,
                    mode,
                    sourceAssetId: editItemId,
                    assetId: resultAssetId,
                    parentLayerId: result.parentLayerId || null,
                    versionId: result.versionId || null
                }
            };
        }
        throw new Error(`Unsupported Agent capability: ${capabilityType}`);
    } catch (error) {
        restoreAgentCapabilitySnapshot(operation, workspace);
        throw error;
    }
}

export async function undoAgentCapability({ workspace, operationId }) {
    const operation = agentCapabilityOperations.get(operationId);
    if (!operation) return;
    restoreAgentCapabilitySnapshot(operation, workspace);
    agentCapabilityOperations.delete(operationId);
}

export async function commitAgentCapabilities({ workspace, operationIds = [] }) {
    for (const operationId of operationIds) {
        const operation = agentCapabilityOperations.get(operationId);
        if (!operation) continue;
        const ids = [...new Set([operation.itemId, ...(operation.targetAssetIds || [])].filter(Boolean))];
        for (const itemId of ids) {
            const item = workbenchItems.get(itemId);
            if (item) await persistLayerStateToRuntime(itemId, item, { persistSession: false });
        }
        if (ids.length === 0) throw new Error(`Capability source ${operation.itemId} is unavailable at commit.`);
    }
}

export async function releaseAgentCapabilities({ operationIds = [] }) {
    operationIds.forEach(operationId => agentCapabilityOperations.delete(operationId));
}

export async function triggerMagicLayers(itemId) {
    console.log(`[Magic Layers] Triggered for item: ${itemId}`);
    const item = workbenchItems.get(itemId);
    if (!item) {
        alert("找不到图片对象");
        return;
    }

    const overlay = createMagicLayersOverlay(item);
    overlay?.update({ progress: 4 });

    const existingLayers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    const hasAutoDetectedLayers = Array.isArray(existingLayers) && existingLayers.some(l => l.layerType || (l.id && !l.id.startsWith('box-layer-')));
    if (!item.hasFullSemanticAnalysis && !hasAutoDetectedLayers) {
        overlay?.update({ progress: 8 });
        addMessage({ sender: 'bot', type: 'text', visibility: 'progress', persist: false, content: '🔍 Magic Layers 正在先执行全图语义分析...' });
        try {
            const analysisResult = await analyzeImageLayers(item.file || item.dataUrl);
            overlay?.update({ progress: 22 });
            let newLayers = analysisResult.rawLayers;
            let scene = analysisResult.scene;

            const existingManualLayers = Array.isArray(existingLayers)
                ? existingLayers.filter(l => l.id && l.id.startsWith('box-layer-'))
                : [];

            if (existingManualLayers.length > 0) {
                const calculateOverlapPercent = (box1, box2) => {
                    const [y1min, x1min, y1max, x1max] = box1;
                    const [y2min, x2min, y2max, x2max] = box2;
                    const x_overlap = Math.max(0, Math.min(x1max, x2max) - Math.max(x1min, x2min));
                    const y_overlap = Math.max(0, Math.min(y1max, y2max) - Math.max(y1min, y2min));
                    if (x_overlap <= 0 || y_overlap <= 0) return 0;
                    const overlapArea = x_overlap * y_overlap;
                    const area1 = (x1max - x1min) * (y1max - y1min);
                    const area2 = (x2max - x2min) * (y2max - y2min);
                    const smallestArea = Math.min(area1, area2);
                    return smallestArea === 0 ? 0 : overlapArea / smallestArea;
                };

                newLayers = newLayers.filter(newLayer => {
                    if (newLayer.layerType === 'background_plate') return true;
                    const newBox = newLayer.bbox;
                    for (const manualLayer of existingManualLayers) {
                        const overlap = calculateOverlapPercent(newBox, manualLayer.bbox);
                        if (overlap > 0.6) return false;
                    }
                    return true;
                });

                newLayers = [...existingManualLayers, ...newLayers];
                newLayers.forEach((layer, idx) => {
                    if (layer.id && layer.id.startsWith('box-layer-')) {
                        layer.zIndex = 100 - idx;
                    }
                });
                scene.layers = newLayers;
            }

            const semanticViews = await buildSemanticLayerViews(
                item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl,
                newLayers,
                { expandText: true }
            );
            item.scene = scene;
            applySemanticLayerViewsToItem(item, semanticViews);
            item.hasFullSemanticAnalysis = true;
            workbenchItems.set(itemId, item);
            await persistLayerStateToRuntime(itemId, item);
        } catch (e) {
            overlay?.close();
            addMessage({ sender: 'bot', type: 'text', content: `❌ Magic Layers 语义分析失败: ${e.message}` });
            return;
        }
    }

    const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    if (!Array.isArray(layers) || layers.length === 0) {
        addMessage({ sender: 'bot', type: 'text', content: '❌ Magic Layers 失败：未识别到可编辑图层。' });
        return;
    }

    const selectedIndices = selectAllMagicLayers(itemId);
    if (selectedIndices.length === 0) {
        overlay?.close();
        addMessage({ sender: 'bot', type: 'text', content: '❌ Magic Layers 失败：没有可自动提取的非背景图层。' });
        return;
    }

    overlay?.update({ progress: 35 });
    addMessage({
        sender: 'bot',
        type: 'text',
        visibility: 'progress',
        persist: false,
        content: `✨ **Magic Layers 启动**\n已自动选中 ${selectedIndices.length} 个非背景图层，开始语义提取与一次性背景净化。`
    });

    if (isAgentRuntimeFeatureEnabled('magicLayersCommand')) {
        const layerIds = selectedIndices
            .map(index => layers[index]?.id)
            .filter(Boolean);
        await startLayerExtractionJob({
            runtime: window.mvrRuntime,
            itemId,
            layerIds,
            goal: `提取当前图片的 ${layerIds.length} 个可编辑图层`
        });
        await showLayerManagerModal(itemId, false);
        ensureLayerPanelFooter(itemId);
        return;
    }

    try {
        overlay?.update({ progress: 55 });
        await triggerLayerExplosion(itemId);
        overlay?.update({ progress: 88 });
    } finally {
        overlay?.close();
    }
    await showLayerManagerModal(itemId, false);
    ensureLayerPanelFooter(itemId);
}

/**
 * 执行快捷融合逻辑
 * 逻辑：根据子图的 Bbox 在原图上生成蒙版并进行局部重绘
 */
export async function handleQuickFusionSync(childId, promptText, options = {}) {
    const childItem = workbenchItems.get(childId);
    const parentItem = childItem ? workbenchItems.get(childItem.parentId) : null;
    // Only an explicit sync request may replace an existing decomposed layer.
    // Legacy callers without this flag retain the old full-scene sync behavior.
    const requestedReplace = options.replaceCurrent === true;
    const linkedToSplitLayer = hasSplitLayerPlacement(childItem);
    const parentLayer = parentItem && linkedToSplitLayer
        ? resolveParentLayerForAsset(parentItem, childItem)
        : null;

    console.info('[Layer Asset Sync] route check', {
        childId,
        operationMode: options.operationMode || 'fusion',
        requestedReplace,
        childFound: !!childItem,
        childType: childItem?.type || null,
        parentId: childItem?.parentId || null,
        parentFound: !!parentItem,
        hasOriginalBbox: Array.isArray(childItem?.originalBbox),
        splitTypeRecognized: isSplitLayerAsset(childItem),
        parentLayerMatched: !!parentLayer
    });

    if (!childItem) {
        const message = '同步失败：找不到当前图层';
        console.error('[Layer Asset Sync]', message, { childId });
        addMessage({ sender: 'bot', type: 'text', content: `❌ ${message}` });
        return null;
    }

    // A decomposed layer is already an isolated asset. Do not send the whole
    // scene back to the image model: edit the asset and append a layer version.
    if (requestedReplace && linkedToSplitLayer) {
        if (!parentItem) {
            const message = `同步失败：图层“${childItem.layerName || childId}”的父图已不存在`;
            console.error('[Layer Asset Sync]', message, { childId });
            addMessage({ sender: 'bot', type: 'text', content: `❌ ${message}` });
            return null;
        }
        if (!parentLayer) {
            const message = `同步失败：无法将“${childItem.layerName || childId}”匹配到原 Magic Layers 图层`;
            console.error('[Layer Asset Sync]', message, { childId, parentId: childItem.parentId });
            addMessage({ sender: 'bot', type: 'text', content: `❌ ${message}` });
            return null;
        }
        console.log(`[Layer Asset Edit] routing sync to in-place version replacement: child=${childId}`);
        return handleIsolatedAssetEdit(childId, promptText, {
            replaceCurrent: true,
            parentItemId: childItem.parentId,
            parentItem,
            source: 'sync_to_current_layer'
        });
    }

    if (requestedReplace) {
        const message = '同步到当前图层仅支持 Magic Layers 拆分后的独立图层';
        console.error('[Layer Asset Sync]', message, { childId, childType: childItem.type || null });
        addMessage({ sender: 'bot', type: 'text', content: `❌ ${message}` });
        return null;
    }
    if (!parentItem) return;

    const executeQuickFusionSync = async (customPrompt) => {
        const promptToUse = customPrompt || promptText;
        const tempMsg = addMessage({ sender: 'bot', type: 'text', visibility: 'progress', persist: false, content: `🔄 **正在同步至原图**\n正在将针对“${childItem.layerName || '子图'}”的修改（${promptToUse}）同步回主场景中...` });

        try {
            // 1. 利用子图自带的 originalBbox 生成精确蒙版
            const bgSrc = parentItem.cleanPlateDataUrl || parentItem.originalDataUrl || parentItem.dataUrl;
            const parentImg = new Image();
            parentImg.crossOrigin = "anonymous";
            const parentSrc = getProxiedUrl(bgSrc);
            if (!parentSrc) {
                throw new Error('无效的背景图片地址，无法同步至原图');
            }
            parentImg.src = parentSrc;
            await new Promise(r => parentImg.onload = r);

            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = parentImg.naturalWidth;
            maskCanvas.height = parentImg.naturalHeight;
            const ctx = maskCanvas.getContext('2d');
            
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
            
            ctx.fillStyle = 'white';
            
            // 尝试获取当前图层的最新 bbox，如果找不到则回退到 originalBbox
            const parentLayers = parentItem.scene && parentItem.scene.layers ? parentItem.scene.layers : parentItem.layers;
            const currentLayer = parentLayers ? parentLayers.find(l => l.name === childItem.layerName) : null;
            
            // 计算子图在工作台上的相对位置、缩放和旋转
            const parentEl = parentItem.el;
            const parentWidth = parentEl.offsetWidth || 1;
            const parentHeight = parentEl.offsetHeight || 1;
            const parentLeft = parentEl.offsetLeft || 0;
            const parentTop = parentEl.offsetTop || 0;

            const childEl = childItem.el;
            const childWidth = childEl.offsetWidth || 1;
            const childHeight = childEl.offsetHeight || 1;
            const childLeft = childEl.offsetLeft || 0;
            const childTop = childEl.offsetTop || 0;

            const [ymin, xmin, ymax, xmax] = childItem.originalBbox;
            
            const origX = (xmin / 1000) * maskCanvas.width;
            const origY = (ymin / 1000) * maskCanvas.height;
            const origW = ((xmax - xmin) / 1000) * maskCanvas.width;
            const origH = ((ymax - ymin) / 1000) * maskCanvas.height;

            const relDx = (childLeft - parentLeft) - (childItem.spawnLeft - (parentItem.spawnLeft || 0));
            const relDy = (childTop - parentTop) - (childItem.spawnTop - (parentItem.spawnTop || 0));

            const scaleX = maskCanvas.width / parentWidth;
            const scaleY = maskCanvas.height / parentHeight;

            const sx = childItem.spawnWidth ? (childWidth / childItem.spawnWidth) : 1;
            const sy = childItem.spawnHeight ? (childHeight / childItem.spawnHeight) : 1;

            const drawX = origX + relDx * scaleX;
            const drawY = origY + relDy * scaleY;
            const drawW = origW * sx;
            const drawH = origH * sy;

            const transform = childEl.style.transform || '';
            let rotation = 0;
            const match = transform.match(/rotate\(([-\d.]+)deg\)/);
            if (match) {
                rotation = parseFloat(match[1]) * Math.PI / 180;
            }

            // Prefer the extracted layer's alpha silhouette over a solid bbox.
            // Fall back to the bbox only when the child has no usable cutout.
            let childMaskImage = null;
            const childMaskSource = childItem.dataUrl || childItem.cutoutUrl || null;
            if (childMaskSource) {
                try {
                    childMaskImage = await loadImageForFusion(getProxiedUrl(childMaskSource));
                } catch (error) {
                    console.warn('[Fusion Sync] child alpha mask unavailable; using bbox mask:', error);
                }
            }

            ctx.save();
            ctx.translate(drawX + drawW / 2, drawY + drawH / 2);
            if (rotation !== 0) ctx.rotate(rotation);
            ctx.fillRect(-drawW / 2, -drawH / 2, drawW, drawH);
            if (childMaskImage) {
                ctx.globalCompositeOperation = 'destination-in';
                ctx.drawImage(childMaskImage, -drawW / 2, -drawH / 2, drawW, drawH);
            }
            ctx.restore();
            
            const preciseMask = maskCanvas.toDataURL('image/png');

            // 1.5. Prepare the scene at source resolution. The mask above is
            // built in source pixels, so CSS-sized rendering would misalign it.
            const renderScale = parentImg.naturalWidth / Math.max(1, parentEl.offsetWidth || parentImg.naturalWidth);
            const baseCanvas = await renderSceneToCanvas(parentItem.id, renderScale);
            if (!baseCanvas) throw new Error("无法按原图分辨率渲染场景");

            const baseDataUrl = baseCanvas.toDataURL('image/png');
            const baseFile = await dataURLToFile(baseDataUrl, `base-${Date.now()}.png`);

            // Preserve the source framing instead of forcing this edit to 1:1.
            const sourceAspectRatio = getClosestSupportedAspectRatio(parentImg.naturalWidth, parentImg.naturalHeight);
            console.log('[Fusion Sync] source=', `${parentImg.naturalWidth}x${parentImg.naturalHeight}`, 'rendered=', `${baseCanvas.width}x${baseCanvas.height}`, 'aspect=', sourceAspectRatio, 'mask=', `${maskCanvas.width}x${maskCanvas.height}`);
            const syncPrompt = `${promptToUse}. Return one single final scene, not a collage, grid, panel, or duplicated image. Keep the exact original framing and aspect ratio. Change only the object inside the masked area, keep its original position, scale, floor contact, and perspective, and preserve every pixel outside the masked area.`;
            
            let result = await editOrQueryImageWithGemini(syncPrompt, baseFile, [], preciseMask, sourceAspectRatio);

            if (result && result.success && result.imageData) {
                const generatedDataUrl = `data:${result.mimeType};base64,${result.imageData}`;
                const imgSrc = await compositeMaskedFusionResult(baseDataUrl, generatedDataUrl, preciseMask);
                const newParentFile = await dataURLToFile(imgSrc, `synced-parent-${Date.now()}.png`);
                
                // 3. 将新的原图添加到工作台（放在原图位置附近）
                const smartPos = calculateSmartPosition(parentItem.el, 1); // 1=右侧
                const newParentId = await addImageToWorkbench(newParentFile, `同步更新: ${childItem.layerName || '子图'}`, {
                    x: smartPos.x,
                    y: smartPos.y,
                    parentId: parentItem.id,
                    type: 'sync-update',
                    generationParams: {
                        prompt: promptToUse,
                        sourceChildId: childId,
                        sourceLayerName: childItem.layerName
                    }
                });

                // 4. 选中新生成的原图
                setTimeout(() => {
                    selectWorkbenchItem(newParentId);
                    const newItem = workbenchItems.get(newParentId);
                    if (newItem && newItem.el) {
                        newItem.el.style.boxShadow = "0 0 20px #4CAF50";
                        setTimeout(() => newItem.el.style.boxShadow = "", 2000);
                    }
                }, 100);

                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat(`同步融合 [${childItem.layerName || '子图'}]`, promptToUse, imgSrc, executeQuickFusionSync);
                } else {
                    addMessage({ sender: 'bot', type: 'text', content: `✅ **同步融合完成**！已为您生成新的主场景，${childItem.layerName || '子图'}已更新为：${promptToUse}` });
                }
            }
        } catch (e) {
            console.error("Fusion Sync Failed:", e);
            if (tempMsg && tempMsg.parentNode) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `❌ 同步失败: ${e.message}` });
        }
    };

    await executeQuickFusionSync();
}

/**
 * 独立资产编辑逻辑 (Isolated Asset Edit)
 * 逻辑：给透明图层垫底色 -> Gemini 修改 -> 再次抠图 -> 生成新透明图层
 */
export async function handleIsolatedAssetEdit(childId, promptText, options = {}) {
    const childItem = workbenchItems.get(childId);
    if (!childItem) return;
    const parentItem = options.parentItem || workbenchItems.get(childItem.parentId);

    console.info('[Layer Asset Edit] start', {
        childId,
        childType: childItem.type || null,
        parentId: childItem.parentId || null,
        replaceCurrent: options.replaceCurrent === true,
        source: options.source || 'isolated_asset'
    });
    if (options.replaceCurrent && (!parentItem || !hasSplitLayerPlacement(childItem) || !resolveParentLayerForAsset(parentItem, childItem))) {
        const message = '原位替换失败：当前资产缺少可回写的父图层';
        console.error('[Layer Asset Edit]', message, {
            childId,
            parentId: childItem.parentId || null,
            childType: childItem.type || null,
            originalBbox: childItem.originalBbox || null
        });
        addMessage({ sender: 'bot', type: 'text', content: `❌ ${message}` });
        return;
    }

    const executeIsolatedEdit = async (customPrompt) => {
        const promptToUse = customPrompt || promptText;
        const tempMsg = addMessage({ sender: 'bot', type: 'text', visibility: 'progress', persist: false, content: `🎨 **独立资产编辑中**\n正在对“${childItem.layerName || '子图'}”进行独立修改（${promptToUse}）...` });

        try {
            // 1. 获取当前透明图层
            const childImg = new Image();
            childImg.crossOrigin = "anonymous";
            const childSrc = getProxiedUrl(childItem.dataUrl);
            if (!childSrc) {
                throw new Error('无效的子图地址，无法执行独立资产编辑');
            }
            childImg.src = childSrc;
            await new Promise(r => childImg.onload = r);

            // Keep the generated asset visually comparable to its source item
            // when it is pushed beside the source in the Workbench. The file
            // itself still keeps the source asset's natural pixel dimensions.
            const sourceDisplayWidth = Math.max(
                1,
                parseFloat(childItem.el?.style?.width) || childItem.el?.offsetWidth || childImg.naturalWidth
            );
            const sourceDisplayHeight = Math.max(
                1,
                parseFloat(childItem.el?.style?.height) || childItem.el?.offsetHeight || childImg.naturalHeight
            );
            const sourceRuntimeAsset = window.mvrRuntime?.getCurrentWorkspace?.()?.currentState?.assetRegistry?.get(childId);
            const runtimeZIndex = Number(sourceRuntimeAsset?.transform?.zIndex);
            const domZIndex = Number.parseInt(childItem.el?.style?.zIndex || '', 10);
            const sourceZIndex = Number.isFinite(runtimeZIndex)
                ? runtimeZIndex
                : Number.isFinite(domZIndex)
                    ? domZIndex
                    : 0;
            const sourceLayerId = childItem.sourceLayerId || childItem.layerId || null;
            console.info('[Layer Asset Edit] source stacking captured', {
                childId,
                sourceLayerId,
                sourceZIndex,
                sourceDisplaySize: `${sourceDisplayWidth}x${sourceDisplayHeight}`
            });

            // --- 专属逻辑：独立资产的前端秒改（仅限换色） ---
            function detectFastColorChange(prompt) {
                const p = prompt.toLowerCase();
                const colorMatch = p.match(/(红|黄|蓝|绿|紫|粉|橙|青|黑|白|灰|red|blue|green|yellow|purple|pink|orange|cyan)/);
                if (!colorMatch) return null;

                const isStructural = /(去|加|删|大|小|长|短|带|穿|戴|换上|帽子|衣服|眼镜|猫|狗|人|背景|文字|字|特效|add|remove|delete|make|larger|smaller|wear|put on|take off)/.test(p);
                if (isStructural) return null;

                const isChangeColor = /(改|变|换|调|色相|转|色|color|turn|change|make)/.test(p);
                if (!isChangeColor && p.length > 8) return null; 

                const color = colorMatch[0];
                let targetRgb = null;
                switch(color) {
                    case '红': case 'red': targetRgb = [190, 70, 75]; break; // 莫兰迪红 (Dusty Rose / Terracotta)
                    case '橙': case 'orange': targetRgb = [210, 120, 70]; break; // 焦糖橙 (Burnt Orange)
                    case '黄': case 'yellow': targetRgb = [220, 190, 100]; break; // 芥末黄 (Mustard)
                    case '绿': case 'green': targetRgb = [100, 150, 110]; break; // 鼠尾草绿 (Sage Green)
                    case '青': case 'cyan': targetRgb = [80, 160, 170]; break; // 灰雾蓝青 (Dusty Teal)
                    case '蓝': case 'blue': targetRgb = [90, 130, 190]; break; // 莫兰迪海蓝 (Slate Blue)
                    case '紫': case 'purple': targetRgb = [140, 100, 160]; break; // 丁香紫 (Dusty Plum)
                    case '粉': case 'pink': targetRgb = [220, 130, 150]; break; // 藕粉色 (Dusty Pink)
                    case '黑': case 'black': targetRgb = [50, 50, 55]; break; // 碳烟灰黑 (Charcoal) - 避免死黑
                    case '白': case 'white': targetRgb = [245, 245, 240]; break; // 珍珠白 (Pearl)
                    case '灰': case 'gray': case 'grey': targetRgb = [150, 150, 145]; break; // 暖岩灰 (Warm Gray)
                }
                return targetRgb;
            }

            const targetRgbColor = detectFastColorChange(promptToUse);
            if (targetRgbColor) {
                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                addMessage({ sender: 'bot', type: 'text', visibility: 'progress', persist: false, content: `⚡ **前端增强秒改触发**\n检测到纯颜色编辑指令，启动针对半透明材质(如头纱)的高光保护与渐变映射渲染算法...` });
                
                const fastCanvas = document.createElement('canvas');
                fastCanvas.width = childImg.naturalWidth;
                fastCanvas.height = childImg.naturalHeight;
                const fastCtx = fastCanvas.getContext('2d');
                fastCtx.drawImage(childImg, 0, 0);
                
                const imgData = fastCtx.getImageData(0, 0, fastCanvas.width, fastCanvas.height);
                const d = imgData.data;

                // 核心算法：带高光保留的 Color Blend (参考 PS 颜色混合模式 + 正片叠底增强阴影)
                for (let i = 0; i < d.length; i += 4) {
                    const alpha = d[i+3];
                    if (alpha < 5) continue; 
                    
                    const r = d[i];
                    const g = d[i+1];
                    const b = d[i+2];
                    
                    // 1. 提取原始亮度 0.0 ~ 1.0 (Luminance)
                    const luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                    
                    // 2. 目标色归一化
                    const tr = targetRgbColor[0] / 255;
                    const tg = targetRgbColor[1] / 255;
                    const tb = targetRgbColor[2] / 255;

                    // 3. 颜色混合逻辑
                    // - 乘法 (Multiply) 决定了基础色彩附着，Luma 越低，颜色越重
                    // - 如果 Luma 很高（接近白色高光），需要提亮它，模拟反光
                    
                    // 计算基础染色 (Multiply 效果)
                    let outR = luma * tr;
                    let outG = luma * tg;
                    let outB = luma * tb;

                    // 高光保护与提亮 (Screen 效果的变体)
                    // 当原本像素非常亮时，按比例恢复白色
                    const highlightBoost = Math.max(0, luma - 0.5) * 2; // 0.5 以上开始提亮，1.0 时为全白
                    
                    outR = outR * (1 - highlightBoost) + highlightBoost;
                    outG = outG * (1 - highlightBoost) + highlightBoost;
                    outB = outB * (1 - highlightBoost) + highlightBoost;

                    // 4. Alpha 通道耦合调节 (厚度调节)
                    // 纱越透明 (Alpha小)，染色应该越浅 (越接近环境本身，但在此处作为独立层，我们增强它的亮度视错觉)
                    const normalizedAlpha = alpha / 255;
                    // 对极其透明的边缘进行一定的提亮补偿，防止变脏
                    const edgeCompensation = (1 - normalizedAlpha) * 0.2;
                    outR = Math.min(1, outR + edgeCompensation);
                    outG = Math.min(1, outG + edgeCompensation);
                    outB = Math.min(1, outB + edgeCompensation);

                    d[i]   = Math.round(outR * 255);
                    d[i+1] = Math.round(outG * 255);
                    d[i+2] = Math.round(outB * 255);
                }
                fastCtx.putImageData(imgData, 0, 0);

                const finalAssetDataUrl = fastCanvas.toDataURL('image/png');

                if (options.replaceCurrent) {
                    const refinedAssetDataUrl = await tryRefineEditedAssetWithSam(
                        finalAssetDataUrl,
                        childItem,
                        childImg.naturalWidth,
                        childImg.naturalHeight,
                        tempMsg
                    );
                    const replacementDataUrl = await alignAssetBottomToSource(
                        refinedAssetDataUrl || finalAssetDataUrl,
                        childImg,
                        childImg.naturalWidth,
                        childImg.naturalHeight
                    );
                    const replacement = await replaceExistingLayerAsset(childId, childItem, childItem.parentId, parentItem, replacementDataUrl, promptToUse, {
                        persist: options.persist !== false,
                        recordHistory: options.recordHistory !== false
                    });
                    if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                    if (!options.suppressMessages) addMessage({ sender: 'bot', type: 'text', content: `✅ **图层版本已替换**！“${childItem.layerName || '当前图层'}”已更新，位置、比例和透视保持不变。` });
                    return { success: true, childId, versionId: replacement.versionId, parentLayerId: replacement.parentLayer?.id || null };
                }

                const finalAssetFile = await dataURLToFile(finalAssetDataUrl, `colorized-asset-${Date.now()}.png`);

                const smartPos = calculateSmartPosition(childItem.el, 1); // 生成资产放在当前图层右侧
                const newAssetId = await addImageToWorkbench(finalAssetFile, `颜色秒改: ${childItem.layerName || '子图'}`, {
                    x: smartPos.x,
                    y: smartPos.y,
                    initialWidth: sourceDisplayWidth,
                    initialHeight: sourceDisplayHeight,
                    zIndex: sourceZIndex,
                    sourceZIndex,
                    sourceChildId: childId,
                    sourceLayerId,
                    parentId: childItem.parentId, 
                    layerName: childItem.layerName,
                    type: 'isolated-edit',
                    originalBbox: childItem.originalBbox,
                    generationParams: {
                        prompt: promptToUse,
                        sourceChildId: childId
                    }
                });

                setTimeout(() => {
                    selectWorkbenchItem(newAssetId);
                    const newItem = workbenchItems.get(newAssetId);
                    if (newItem && newItem.el) {
                        newItem.el.style.boxShadow = "0 0 20px #2196F3";
                        setTimeout(() => newItem.el.style.boxShadow = "", 2000);
                    }
                }, 100);

                if (window.addWorkbenchActionToChat && options.source !== 'agent_object_editor') {
                    await window.addWorkbenchActionToChat(`独立编辑 [${childItem.layerName || '子图'}]`, promptToUse, finalAssetDataUrl, executeIsolatedEdit);
                } else if (options.source !== 'agent_object_editor') {
                    addMessage({ sender: 'bot', type: 'text', content: `✅ **秒改完成**！已为您生成新的调整目标图。`});
                }
                return { success: true, assetId: newAssetId, childId: newAssetId };
            }
            // --- 快速通道结束 ---

            // 2. 为 Gemini 准备动态底板与分流策略 (Dynamic Generative Green Screen / Mask Generation)
            const analysisCanvas = document.createElement('canvas');
            analysisCanvas.width = childImg.naturalWidth;
            analysisCanvas.height = childImg.naturalHeight;
            const analysisCtx = analysisCanvas.getContext('2d');
            analysisCtx.drawImage(childImg, 0, 0);

            const imgDataObj = analysisCtx.getImageData(0, 0, analysisCanvas.width, analysisCanvas.height);
            
            if (tempMsg && tempMsg.update) {
                tempMsg.update(`🧠 **正在分析材质**: "${childItem.layerName || '子图'}"...`);
            }
            const strategy = StrategyDispatcher.analyze(imgDataObj, childItem.layerName || 'Asset');
            console.log(`[Isolated Edit Strategy ${childItem.layerName}] Path: ${strategy.path} | Reason: ${strategy.reason}`);

            const isCloudOrSmoke = (childItem.layerName || '').includes('云') || (childItem.layerName || '').includes('烟') || (childItem.layerName || '').includes('雾') || (childItem.layerName || '').includes('火') || (childItem.layerName || '').includes('光') || (childItem.layerName || '').includes('冰') || (childItem.layerName || '').includes('水') || (childItem.layerName || '').includes('纱') || (childItem.layerName || '').includes('玻璃') || (childItem.layerName || '').includes('透明');
            const isFineDetail = ((childItem.layerName || '').includes('发') && !(childItem.layerName || '').includes('沙发')) || (childItem.layerName || '').includes('毛') || (childItem.layerName || '').includes('羽') || (childItem.layerName || '').includes('树') || (childItem.layerName || '').includes('草') || (childItem.layerName || '').includes('叶') || (childItem.layerName || '').includes('线') || (childItem.layerName || '').includes('网');
            const useMaskFormat = isCloudOrSmoke || isFineDetail;

            const bestBackdrop = selectAdaptiveBackdrop(imgDataObj, childItem.layerName || 'Asset');
            console.info('[Layer Asset Edit] adaptive backdrop selected', {
                layerName: childItem.layerName || 'Asset',
                backdrop: bestBackdrop.name,
                rgb: bestBackdrop.rgb,
                strategyPath: strategy.path,
                visibleSampleCount: bestBackdrop.sampleCount || 0
            });

            const bgColorHex = bestBackdrop.hex;
            const bgColorName = bestBackdrop.name;
            const bgColorRule = '4. BACKGROUND RULE: ' + bestBackdrop.rule;

            if (tempMsg && tempMsg.update) {
                tempMsg.update(`🎨 **独立资产编辑中**\n执行材质策略: ${useMaskFormat ? '掩码抽取' : '动态绿幕'}...`);
            }

            // 边缘保护隔离带 (Anti-Edge-Bleed Padding)
            // 避免电商服装直接顶格接触边界，导致 MatteWorker.sampleBackground 误把衣服当背景采样
            const padX = Math.max(20, Math.floor(childImg.naturalWidth * 0.1));
            const padY = Math.max(20, Math.floor(childImg.naturalHeight * 0.1));

            const bgCanvas = document.createElement('canvas');
            bgCanvas.width = childImg.naturalWidth + padX * 2;
            bgCanvas.height = childImg.naturalHeight + padY * 2;
            const ctx = bgCanvas.getContext('2d');
            
            // 实体与云雾皆采用动态纯色底板 (避免边缘像素截断问题)
            ctx.fillStyle = bestBackdrop.hex;
            ctx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);
            ctx.drawImage(childImg, padX, padY);
            
            const paddedDataUrl = bgCanvas.toDataURL('image/png');
            const paddedFile = await dataURLToFile(paddedDataUrl, `padded-${Date.now()}.png`);
            const assetAspectRatio = getClosestSupportedAspectRatio(
                childImg.naturalWidth,
                childImg.naturalHeight
            );
            console.info('[Layer Asset Edit] generation aspect selected', {
                sourceSize: `${childImg.naturalWidth}x${childImg.naturalHeight}`,
                aspectRatio: assetAspectRatio,
                outputPolicy: 'restore_original_asset_pixels'
            });

            // 3. 调用 Gemini 进行独立资产重绘材质与造型
            // FORCE standard generation in edit mode. Overrides useMaskFormat.
            let isolatedPrompt = `Act as a precise image asset editor.
Your task is to modify the object "${childItem.layerName || 'object'}" in the image according to this instruction: "${promptToUse}".

CRITICAL REQUIREMENTS:
1. ZERO HALLUCINATION: You MUST NOT redraw or alter the bounding box shape of the object unless instructed.
2. BACKGROUND RULE: Your ONLY task regarding the background is to keep it a perfectly solid, uniform ${bgColorName} color (${bgColorHex}). ${bgColorRule} Do not add shadows, gradients, or environment.
3. OBJECT ISOLATION: Only modify the object itself based on the instruction. Keep it inside the original object's canvas footprint.
4. PLACEMENT: Preserve the original viewing angle, plane orientation, center position, visual scale, and contact direction. Do not zoom, recenter, rotate, or expand the object to fill the canvas.
5. HIGH QUALITY: Ensure the modified object has realistic textures and lighting.`;

            let result = await editOrQueryImageWithGemini(
                isolatedPrompt,
                paddedFile,
                [],
                null,
                assetAspectRatio
            );

            if (!result || !result.success || !result.imageData) {
                throw new Error("Gemini 资产编辑失败");
            }

            const reconstructedDataUrl = `data:${result.mimeType};base64,${result.imageData}`;

            // 4. 调用本地 matte-task-system 执行基于固体底色的 Alpha 通道抽取
            if (tempMsg && tempMsg.update) {
                tempMsg.update(`✨ **正在提取**: "${childItem.layerName || '子图'}"... AI 正在执行背景抹除计算。`);
            }

            const reconstructedImg = new Image();
            reconstructedImg.src = reconstructedDataUrl;
            await new Promise(r => {
                reconstructedImg.onload = r;
                reconstructedImg.onerror = () => r();
            });

            const tempWorkspaceCanvas = document.createElement('canvas');
            // 关键修复：不要用 reconstructedImg.width，必须强制缩回原有的带 Padding 尺寸
            tempWorkspaceCanvas.width = bgCanvas.width;
            tempWorkspaceCanvas.height = bgCanvas.height;
            const tempCtx = tempWorkspaceCanvas.getContext('2d');
            tempCtx.imageSmoothingEnabled = true;
            tempCtx.imageSmoothingQuality = 'high';
            // 将模型结果等比铺满原始带 padding 的画布，裁掉多余画布，避免
            // 标准输出比例被强行拉伸或因 contain 造成对象整体缩小。
            drawImageCoveringTarget(tempCtx, reconstructedImg, bgCanvas.width, bgCanvas.height);

            const generatedImageDataObj = tempCtx.getImageData(0, 0, tempWorkspaceCanvas.width, tempWorkspaceCanvas.height);
            const sampledBackdrop = sampleGeneratedBackdrop(generatedImageDataObj, bestBackdrop.rgb);
            const matteBackgroundRgb = sampledBackdrop.rgb;
            console.info('[Layer Asset Edit] generated backdrop sampled', {
                requested: bestBackdrop.rgb,
                sampled: sampledBackdrop.sampledRgb,
                used: matteBackgroundRgb,
                source: sampledBackdrop.source,
                stable: sampledBackdrop.stable,
                sampleCount: sampledBackdrop.sampleCount,
                spread: Number(sampledBackdrop.spread.toFixed(2))
            });
            const taskId = `iso_asset_${Date.now()}`;

            // 针对实体服装等需要走严格固体抠图算法的，传指定 bgColor 避免依赖边缘猜色。
            // 针对云雾则留空，让其内部降级为 channel matting
            const extractionConfig = useMaskFormat ? {} : { type: 'solid', bgColor: matteBackgroundRgb };

            let processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                taskId, 
                generatedImageDataObj, 
                null, 
                childItem.layerName || 'Asset', 
                (msg) => { if (tempMsg && tempMsg.update) tempMsg.update(msg); },
                extractionConfig
            );

            const extractedImageData = new ImageData(processedAlphaData, tempWorkspaceCanvas.width, tempWorkspaceCanvas.height);
            tempCtx.putImageData(extractedImageData, 0, 0);

            const matteImageData = tempCtx.getImageData(0, 0, tempWorkspaceCanvas.width, tempWorkspaceCanvas.height);
            const despillStats = despillMatteImageData(matteImageData, matteBackgroundRgb);
            tempCtx.putImageData(matteImageData, 0, 0);
            console.info('[Layer Asset Edit] matte edge cleanup', {
                background: matteBackgroundRgb,
                edgePixels: despillStats.edgePixels,
                correctedPixels: despillStats.correctedPixels,
                averageSpill: Number(despillStats.averageSpill.toFixed(2)),
                unmixedPixels: despillStats.unmixedPixels,
                averageUnmixStrength: Number(despillStats.averageUnmixStrength.toFixed(3))
            });

            // 裁切回原始尺寸 (Crop padding back)
            const finalCanvas = document.createElement('canvas');
            finalCanvas.width = childImg.naturalWidth;
            finalCanvas.height = childImg.naturalHeight;
            const finalCtx = finalCanvas.getContext('2d');
            finalCtx.drawImage(tempWorkspaceCanvas, padX, padY, childImg.naturalWidth, childImg.naturalHeight, 0, 0, childImg.naturalWidth, childImg.naturalHeight);

            const finalAssetDataUrl = finalCanvas.toDataURL('image/png');

            if (options.replaceCurrent) {
                const refinedAssetDataUrl = await tryRefineEditedAssetWithSam(
                    finalAssetDataUrl,
                    childItem,
                    childImg.naturalWidth,
                    childImg.naturalHeight,
                    tempMsg
                );
                const replacementDataUrl = await alignAssetBottomToSource(
                    refinedAssetDataUrl || finalAssetDataUrl,
                    childImg,
                    childImg.naturalWidth,
                    childImg.naturalHeight
                );
                const replacement = await replaceExistingLayerAsset(childId, childItem, childItem.parentId, parentItem, replacementDataUrl, promptToUse, {
                    persist: options.persist !== false,
                    recordHistory: options.recordHistory !== false
                });
                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                if (!options.suppressMessages) addMessage({ sender: 'bot', type: 'text', content: `✅ **图层版本已替换**！“${childItem.layerName || '当前图层'}”已更新，位置、比例和透视保持不变。` });
                return { success: true, childId, versionId: replacement.versionId, parentLayerId: replacement.parentLayer?.id || null };
            }

            const finalAssetFile = await dataURLToFile(finalAssetDataUrl, `edited-asset-${Date.now()}.png`);

            // 6. 将新的透明资产添加到工作台
            const smartPos = calculateSmartPosition(childItem.el, 1); // 生成资产放在当前图层右侧
            const newAssetId = await addImageToWorkbench(finalAssetFile, `独立编辑: ${childItem.layerName || '子图'}`, {
                x: smartPos.x,
                y: smartPos.y,
                initialWidth: sourceDisplayWidth,
                initialHeight: sourceDisplayHeight,
                zIndex: sourceZIndex,
                sourceZIndex,
                sourceChildId: childId,
                sourceLayerId,
                parentId: childItem.parentId, // 保持族谱关联
                layerName: childItem.layerName,
                type: 'isolated-edit',
                originalBbox: childItem.originalBbox,
                generationParams: {
                    prompt: promptToUse,
                    sourceChildId: childId
                }
            });

            // 7. 选中新生成的资产
            setTimeout(() => {
                selectWorkbenchItem(newAssetId);
                const newItem = workbenchItems.get(newAssetId);
                if (newItem && newItem.el) {
                    newItem.el.style.boxShadow = "0 0 20px #2A5C82";
                    setTimeout(() => newItem.el.style.boxShadow = "", 2000);
                }
            }, 100);

            if (tempMsg && tempMsg.parentNode) tempMsg.remove();
            if (window.addWorkbenchActionToChat && options.source !== 'agent_object_editor') {
                await window.addWorkbenchActionToChat(`独立编辑 [${childItem.layerName || '子图'}]`, promptToUse, finalAssetDataUrl, executeIsolatedEdit);
            } else if (options.source !== 'agent_object_editor') {
                addMessage({ sender: 'bot', type: 'text', content: `✅ **独立编辑完成**！已为您生成新的透明资产：${promptToUse}` });
            }
            return { success: true, assetId: newAssetId, childId: newAssetId };
        } catch (e) {
            console.error("Isolated Edit Failed:", e);
            if (tempMsg && tempMsg.parentNode) tempMsg.remove();
            if (!options.suppressMessages) addMessage({ sender: 'bot', type: 'text', content: `❌ 独立编辑失败: ${e.message}` });
            if (options.throwOnError) throw e;
            return { success: false, error: e?.message || String(e), childId };
        }
    };

    return executeIsolatedEdit();
}

/**
 * 执行精准修图
 * 策略：1. 截图框选区域 2. 发给 AI 识别内容 3. 生成精准蒙版 4. 执行修改
 */
export async function performPreciseEdit(itemId, box, promptText) {
    const item = workbenchItems.get(itemId);
    if (!item) return;

    const isRem = isRemovalRequest(promptText);
    const isMat = isMaterialRequest(promptText);
    const intentType = isRem ? 'REMOVE' : (isMat ? 'MATERIAL' : 'SMART_EDIT');
    
    const taskConfig = {
        'REMOVE': { icon: '👁️', label: '语义移除', status: '正在分析背景并补全像素...' },
        'MATERIAL': { icon: '🎯', label: '材质替换', status: '正在执行视觉引导下的纹理重构...' },
        'SMART_EDIT': { icon: '🪄', label: '智能重绘', status: '正在解析指令并提取语义特征...' }
    }[intentType];

    const executePreciseEdit = async (customPrompt) => {
        const promptToUse = customPrompt || promptText;
        const tempMsg = addMessage({ 
            sender: 'bot', 
            type: 'text', 
            visibility: 'progress',
            persist: false,
            content: `${taskConfig.icon} **正在调用：${taskConfig.label}专业技能**\n${taskConfig.status}` 
        });

        try {
            // 1. 准备物理参数
            const dimensions = await getImageDimensions(item.file || item.dataUrl);
            const closestAspectRatio = getClosestSupportedAspectRatio(dimensions.width, dimensions.height);
            const [ymin, xmin, ymax, xmax] = box;

            const parentImg = new Image();
            parentImg.crossOrigin = "anonymous";
            const itemSrc = getProxiedUrl(item.dataUrl);
            if (!itemSrc) {
                throw new Error('无效的原图地址，无法执行精准修图');
            }
            parentImg.src = itemSrc;
            await new Promise(r => parentImg.onload = r);

            const imgW = parentImg.naturalWidth;
            const imgH = parentImg.naturalHeight;

            // 2. 【核心思路】准备原图和遮罩图 (Plan 2: Dual-Image Guidance)
            const originalCanvas = document.createElement('canvas');
            originalCanvas.width = imgW;
            originalCanvas.height = imgH;
            const oCtx = originalCanvas.getContext('2d');
            oCtx.drawImage(parentImg, 0, 0);
            
            const bx = (xmin / 1000) * imgW;
            const by = (ymin / 1000) * imgH;
            const bw = ((xmax - xmin) / 1000) * imgW;
            const bh = ((ymax - ymin) / 1000) * imgH;
            
            // B. 准备用于引导大模型和最终合成的硬蒙版
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = imgW;
            maskCanvas.height = imgH;
            const mCtx = maskCanvas.getContext('2d');
            mCtx.fillStyle = 'black';
            mCtx.fillRect(0, 0, imgW, imgH);
            mCtx.fillStyle = 'white';
            const inset = Math.max(1, Math.round(imgW / 1000));
            mCtx.fillRect(bx + inset, by + inset, bw - inset * 2, bh - inset * 2);

            // Calculate closest supported aspect ratio and pad the image
            const supportedRatios = [
                { str: '1:1', val: 1 },
                { str: '4:3', val: 4/3 },
                { str: '3:4', val: 3/4 },
                { str: '16:9', val: 16/9 },
                { str: '9:16', val: 9/16 }
            ];
            const imgRatio = imgW / imgH;
            let bestRatio = supportedRatios[0];
            let minDiff = Math.abs(imgRatio - bestRatio.val);
            for (const r of supportedRatios) {
                const diff = Math.abs(imgRatio - r.val);
                if (diff < minDiff) {
                    minDiff = diff;
                    bestRatio = r;
                }
            }

            let padW = imgW;
            let padH = imgH;
            if (imgRatio > bestRatio.val) {
                padH = imgW / bestRatio.val;
            } else {
                padW = imgH * bestRatio.val;
            }

            const offsetX = (padW - imgW) / 2;
            const offsetY = (padH - imgH) / 2;

            const paddedOriginalCanvas = document.createElement('canvas');
            paddedOriginalCanvas.width = padW;
            paddedOriginalCanvas.height = padH;
            const poCtx = paddedOriginalCanvas.getContext('2d');
            poCtx.fillStyle = '#FFFFFF';
            poCtx.fillRect(0, 0, padW, padH);
            poCtx.drawImage(originalCanvas, offsetX, offsetY, imgW, imgH);
            const base64Original = paddedOriginalCanvas.toDataURL('image/png').split(',')[1];

            const paddedMaskCanvas = document.createElement('canvas');
            paddedMaskCanvas.width = padW;
            paddedMaskCanvas.height = padH;
            const pmCtx = paddedMaskCanvas.getContext('2d');
            pmCtx.fillStyle = '#000000';
            pmCtx.fillRect(0, 0, padW, padH);
            pmCtx.drawImage(maskCanvas, offsetX, offsetY, imgW, imgH);
            const base64Mask = paddedMaskCanvas.toDataURL('image/png').split(',')[1];

            // 3. 收集参考素材
            const allReferenceFiles = [];
            if (state.mainImageFile && state.mainImageFile !== item.file) allReferenceFiles.push(state.mainImageFile);
            state.referenceImageFiles.forEach(f => { if (f !== item.file) allReferenceFiles.push(f); });

            // 4. 构造指令
            let materialProtocol = "";
            if (intentType === 'MATERIAL') {
                materialProtocol = `
            ## SPECIAL SKILL: STRUCTURAL-PRESERVING MATERIAL REPLACEMENT
            - **3D MESH LOCK (CRITICAL)**: Treat the object in Image 0 as a strictly fixed 3D mesh. You are a rendering engine. Do not alter its geometry, volume, or silhouette.
            - **SURFACE PROJECTION**: Do not "redraw" the object. Instead, "project" and "wrap" the requested material from Image 2+ onto the existing pixels as a UV texture map. The underlying structural skeleton must not shift by even a single pixel.
            - **NEGATIVE CONSTRAINTS**: FORBIDDEN ACTIONS: Do not add new structural elements (e.g., new wrinkles, folds, seams, buttons, or bumps) that do not exist in Image 0. Do not smooth out or delete existing structural indentations. The micro-structure must remain 100% identical.
            - **TEXTURE PURGE & REFERENCE FIDELITY**: Completely discard the original texture/patterns. Strictly use the textures, materials, and surface properties from Image 2+.
            - **LIGHTING ADAPTATION**: Apply the new material while inheriting the exact lighting, shadows, and reflections from the environment in Image 0.
                `;
            }

            let semanticAction = intentType === 'MATERIAL' 
                ? `**MATERIAL PROJECTION**: Project and wrap the new material onto the object to match the instruction: "${promptToUse}", without altering the underlying structure.`
                : `**SEMANTIC REGENERATION**: Redraw the object/area within the box to match the instruction: "${promptToUse}".`;

            let systemProtocol = `
            ## SKILL: PRECISION INPAINTING & MATERIAL TRANSFORMATION
            - Image 0: The clean original source image.
            - Image 1: A spatial mask (Black & White). The WHITE area defines the STRICT and ONLY quarantine zone you are allowed to edit. The BLACK area MUST remain 100% identical to Image 0.
            - Image 2+: Reference textures or styles to be applied.
            
            ${materialProtocol}

            CRITICAL DIRECTIVES & PENALTIES:
            1. **THE QUARANTINE ZONE**: You are ONLY permitted to alter the pixels corresponding to the WHITE area in Image 1. 
            2. **ABSOLUTE FREEZE OUTSIDE**: Every single pixel corresponding to the BLACK area in Image 1 MUST remain 100% identical to Image 0. If you alter the background, lighting, or any object in the black zone, the operation is considered a catastrophic failure.
            3. **SEAMLESS INTEGRATION**: ${semanticAction} The new content inside the white zone must blend flawlessly with the frozen pixels outside the zone. Match the original lighting, shadows, and grain perfectly.
            4. **NO HALLUCINATIONS**: Do not invent new objects or structures that were not requested.
            `;

            let finalCommandAction = intentType === 'MATERIAL'
                ? `Perform a high-precision material projection strictly confined to the boxed area.`
                : `Perform a high-precision redraw strictly confined to the boxed area.`;

            const finalEditPrompt = `
            [SYSTEM INSTRUCTION]
            ${systemProtocol}
            
            FINAL COMMAND: ${finalCommandAction} 
            WARNING: You will be penalized if ANY pixel corresponding to the BLACK area in Image 1 is changed. 
            MANDATORY: Return ONLY the clean, final image.
            `;

            // 5. 组装 Parts
            const contentsParts = [
                { inlineData: { data: base64Original, mimeType: 'image/png' } },
                { inlineData: { data: base64Mask, mimeType: 'image/png' } }
            ];
            for (const refFile of allReferenceFiles) {
                contentsParts.push({ inlineData: { data: await fileToBase64(refFile), mimeType: refFile.type || 'image/png' } }); 
            }
            contentsParts.push({ text: finalEditPrompt });

            // 6. 调用模型
            let resultImageData = await generatePreciseEditImage(contentsParts, bestRatio.str);

            // 7. 最终合成
            if (resultImageData) {
                const aiResultImg = new Image();
                aiResultImg.crossOrigin = "anonymous";
                aiResultImg.src = `data:image/png;base64,${resultImageData}`;
                await new Promise(r => aiResultImg.onload = r);

                const unpaddedCanvas = document.createElement('canvas');
                unpaddedCanvas.width = imgW;
                unpaddedCanvas.height = imgH;
                const uCtx = unpaddedCanvas.getContext('2d');

                const scaleX = aiResultImg.width / padW;
                const scaleY = aiResultImg.height / padH;
                
                const sx = offsetX * scaleX;
                const sy = offsetY * scaleY;
                const sWidth = imgW * scaleX;
                const sHeight = imgH * scaleY;

                uCtx.drawImage(aiResultImg, sx, sy, sWidth, sHeight, 0, 0, imgW, imgH);

                const finalCanvas = document.createElement('canvas');
                finalCanvas.width = imgW;
                finalCanvas.height = imgH;
                const fCtx = finalCanvas.getContext('2d');

                fCtx.drawImage(parentImg, 0, 0);

                const topLayer = document.createElement('canvas');
                topLayer.width = imgW;
                topLayer.height = imgH;
                const tCtx = topLayer.getContext('2d');
                tCtx.drawImage(unpaddedCanvas, 0, 0);
                
                tCtx.globalCompositeOperation = 'destination-in';
                tCtx.drawImage(maskCanvas, 0, 0);

                fCtx.drawImage(topLayer, 0, 0);

                const finalDataUrl = finalCanvas.toDataURL('image/png');
                const newFile = await dataURLToFile(finalDataUrl, `box-edit-${Date.now()}.png`);
                
                const smartPos = calculateSmartPosition(item.el, 1); 
                const newId = await addImageToWorkbench(newFile, `引导：${taskConfig.label}`, {
                    x: smartPos.x,
                    y: smartPos.y,
                    parentId: itemId,
                    type: 'skill-edit'
                });

                // 清理
                state.mainImageFile = null;
                state.referenceImageFiles = [];
                if (typeof window.updateImagePreview === 'function') window.updateImagePreview();

                setTimeout(() => selectWorkbenchItem(newId), 100);
                
                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat(`视觉引导修图 [${taskConfig.label}]`, promptToUse, finalDataUrl, executePreciseEdit);
                } else {
                    addMessage({ sender: 'bot', type: 'text', content: `✅ **视觉引导修图完成**！已通过“单图红框定位”技术精准重构选区内容。` });
                }
            }
        } catch (e) {
            console.error("Skill Execution Failed:", e);
            if (tempMsg && tempMsg.parentNode) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `❌ 技能执行失败: ${e.message}` });
        }
    };

    await executePreciseEdit();
}


export async function extractSingleBoxLayer(itemId, layerName, bbox, dataUrl) {
    const item = state.workbenchItems.get(itemId);
    if (!item) throw new Error("Item not found");

    const layerDefinition = {
        name: layerName,
        bbox: bbox,
        assetStatus: 'pending' // will be updated
    };

    // Initialize layers array if it doesn't exist yet
    if (!item.scene) item.scene = { layers: [] };
    if (!item.layers) item.layers = [];
    
    // Auto-sync between item.scene.layers and item.layers
    const currentLayers = item.scene.layers.length > 0 ? item.scene.layers : item.layers;
    
    layerDefinition.id = `box-layer-${Date.now()}`;
    currentLayers.unshift(layerDefinition); // Insert to front so it shows on top!

    if (item.scene && item.scene.layers) item.scene.layers = currentLayers;
    item.layers = currentLayers;

    state.workbenchItems.set(itemId, item);
    await persistLayerStateToRuntime(itemId, item);
    // Render the layer list if visible
    if(document.getElementById('workbenchLayerListModal') && document.getElementById('workbenchLayerListModal').style.display !== 'none') {
        renderLayerList(currentLayers, itemId);
    }
    renderCanvasLayers(itemId); // show the rect on canvas

    // Now extract it just like executeLayerExplosion does for one layer!
    const parentImg = new Image();
    parentImg.crossOrigin = "anonymous";
    const dataSrc = getProxiedUrl(dataUrl);
    if (!dataSrc) {
        throw new Error('无效的图片地址，无法提取手动画框图层');
    }
    parentImg.src = dataSrc;
    await new Promise(r => parentImg.onload = r);

    try {
        const segmented = await segmentSingleLayer({
            item,
            layer: layerDefinition,
            onProgress: (message) => console.log(`[FastSAM Box ${layerName}] ${message}`),
            qualityProfile: 'completion'
        });

        if (segmented?.dataUrl) {
            layerDefinition.cutoutUrl = segmented.dataUrl;
            layerDefinition.assetStatus = 'ready';
            layerDefinition.activeVersionId = 'base';
            if (!layerDefinition.versions) {
                layerDefinition.versions = [];
            }
            layerDefinition.versions.push({
                id: 'base',
                prompt: 'FastSAM 原始提取',
                cutoutUrl: layerDefinition.cutoutUrl,
                timestamp: Date.now(),
                type: 'segmentation'
            });

            state.workbenchItems.set(itemId, item);
            await persistLayerStateToRuntime(itemId, item);
            addMessage({ sender: 'bot', type: 'text', content: `✅ 图层 **${layerName}** 已通过 FastSAM 提取完成。`});

            if(document.getElementById('workbenchLayerListModal') && document.getElementById('workbenchLayerListModal').style.display !== 'none') {
                renderLayerList(currentLayers, itemId);
            }
            renderCanvasLayers(itemId);
            return;
        }
    } catch (error) {
        if (DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST) {
            throw new Error(`FastSAM 单图层提取失败，已阻止进入 Gemini 兜底: ${error.message || error}`);
        }
        console.warn(`[FastSAM Box ${layerName}] fallback to legacy extraction:`, error);
    }

    const [ymin, xmin, ymax, xmax] = bbox;
    const pxMinX = (xmin / 1000) * parentImg.naturalWidth;
    const pxMinY = (ymin / 1000) * parentImg.naturalHeight;
    const pxMaxX = (xmax / 1000) * parentImg.naturalWidth;
    const pxMaxY = (ymax / 1000) * parentImg.naturalHeight;
    const w = pxMaxX - pxMinX;
    const h = pxMaxY - pxMinY;

    // Use 15% padding uniformly to ensure strategy-dispatcher has enough background pixels
    const padX = Math.floor(w * 0.15);
    const padY = Math.floor(h * 0.15);
    const cropX = Math.max(0, Math.floor(pxMinX - padX));
    const cropY = Math.max(0, Math.floor(pxMinY - padY));
    const rawCropW = Math.min(Math.floor(parentImg.naturalWidth) - cropX, Math.floor(w + padX * 2));
    const rawCropH = Math.min(Math.floor(parentImg.naturalHeight) - cropY, Math.floor(h + padY * 2));

    let cropW = rawCropW;
    let cropH = rawCropH;
    const maxDim = Math.max(rawCropW, rawCropH);
    if (maxDim < 1024 && maxDim > 0) {
        const scaleFactor = 1024 / maxDim;
        cropW = Math.floor(rawCropW * scaleFactor);
        cropH = Math.floor(rawCropH * scaleFactor);
    }

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const ctx = cropCanvas.getContext('2d');
    
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(parentImg, cropX, cropY, rawCropW, rawCropH, 0, 0, cropW, cropH);
    
    // Dynamic Backdrop Selection
    const cropImageData = ctx.getImageData(0, 0, cropW, cropH);
    
    console.log(`[Strategy Dispatcher BoxLayer ${layerName}] Analyzing...`);
    const strategy = StrategyDispatcher.analyze(cropImageData, layerName);
    console.log(`[Strategy Dispatcher ${layerName}] Path: ${strategy.path} | Reason: ${strategy.reason}`);

    let transparentDataUrl;
    const taskId = `matte_task_${Date.now()}`;
    
    if (strategy.path === 'cv') {
        const bgMeanRGB = strategy.metrics?.bgMean ? 
            [strategy.metrics.bgMean.r, strategy.metrics.bgMean.g, strategy.metrics.bgMean.b] : 
            [255, 255, 255];
        
        let processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
            taskId, 
            cropImageData, 
            null, 
            layerName, 
            () => {},
            { type: 'cv_euclidean', bgColor: bgMeanRGB } 
        );
        
        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = cropW;
        finalCanvas.height = cropH;
        const finalCtx = finalCanvas.getContext('2d');
        
        const finalImageData = new ImageData(processedAlphaData, finalCanvas.width, finalCanvas.height);
        finalCtx.putImageData(finalImageData, 0, 0);
        const trimmed = trimTransparentCanvas(finalCanvas);
        cropW = trimmed.width;
        cropH = trimmed.height;
        transparentDataUrl = trimmed.toDataURL('image/png');
        
    } else {
        const isCloudOrSmoke = layerName.includes('云') || layerName.includes('烟') || layerName.includes('雾') || layerName.includes('火') || layerName.includes('光') || layerName.includes('冰') || layerName.includes('水') || layerName.includes('纱') || layerName.includes('玻璃') || layerName.includes('透明');
        const isFineDetail = (layerName.includes('发') && !layerName.includes('沙发')) || layerName.includes('毛') || layerName.includes('羽') || layerName.includes('树') || layerName.includes('草') || layerName.includes('叶') || layerName.includes('线') || layerName.includes('网');
        const useMaskFormat = false; // 框选强制避开 MaskFormat 陷阱，走 solid/cv_euclidean 路线
        
        const avgR = strategy.metrics?.bgMean?.r || 127;
        const avgG = strategy.metrics?.bgMean?.g || 127;
        const avgB = strategy.metrics?.bgMean?.b || 127;
        const fgChroma = strategy.metrics?.fgChroma || { r: 0, g: 0, b: 0, m: 0 };
        
        const backdrops = [
            { name: 'pure, solid green', hex: '#00FF00', rgb: [0, 255, 0], rule: 'Use pure green (#00FF00) ONLY.', conflict: fgChroma.g },
            { name: 'pure, solid magenta', hex: '#FF00FF', rgb: [255, 0, 255], rule: 'Use pure magenta (#FF00FF) ONLY.', conflict: fgChroma.m },
            { name: 'pure, solid blue', hex: '#0000FF', rgb: [0, 0, 255], rule: 'Use pure blue (#0000FF) ONLY.', conflict: fgChroma.b },
            { name: 'pure, solid red', hex: '#FF0000', rgb: [255, 0, 0], rule: 'Use pure red (#FF0000) ONLY.', conflict: fgChroma.r }
        ];
        
        let bestBackdrop = backdrops[0];
        let minConflict = Infinity;
        for (const bd of backdrops) {
            if (bd.conflict < minConflict) {
                minConflict = bd.conflict;
                bestBackdrop = bd;
            }
        }
        
        const bgColorHex = bestBackdrop.hex;
        const bgColorName = bestBackdrop.name;
        const bgColorRule = 'BACKGROUND RULE: ' + bestBackdrop.rule;
        
        let finalPrompt = "";

        if (useMaskFormat) {
            finalPrompt = `Act as a precise computer vision masking tool.
Your task is to generate a semantic segmentation grayscale mask for the object "${layerName}" from the provided cropped image.

CRITICAL REQUIREMENTS:
1. OUTPUT FORMAT: Return ONLY a grayscale mask image.
  - Pure White (#FFFFFF) = opaque parts of the object "${layerName}"
  - Gradients of Gray = semi-transparent edges, fine details (like hair/feathers), or motion blur
  - Pure Black (#000000) = background and everything else
2. NO RENDERING: DO NOT render, composite, or redraw the object. DO NOT cut the object out.
3. PRESERVE DETAILS: Faithfully map every fine tip, hair, and gradient of transparency into grayscale values.
4. STRICT BOUNDARIES: Everything that is not the object MUST be pure black.`;
        } else {
            finalPrompt = `Act as a precise image extraction tool.
Your task is to extract the object "${layerName}" from the provided cropped image and place it on a perfectly solid, uniform ${bgColorName} background (${bgColorHex}).

CRITICAL REQUIREMENTS:
1. ZERO HALLUCINATION: You MUST NOT reconstruct, redraw, or alter the object in any way.
2. PIXEL FIDELITY: Strictly preserve the original pixels, material, texture, color, and lighting of the object itself.
3. BACKGROUND ONLY: Your ONLY task is to change the background surrounding the object to a mathematically solid ${bgColorName} color (${bgColorHex}). Do not touch the object itself.
${bgColorRule}`;
        }

        const croppedDataUrl = cropCanvas.toDataURL('image/png');
        const croppedFile = await dataURLToFile(croppedDataUrl, `cropped-${Date.now()}.png`);
        
        const geminiResult = await editOrQueryImageWithGemini(finalPrompt, croppedFile);
        if (!geminiResult.success || !geminiResult.imageData) {
            throw new Error("Gemini extraction failed");
        }
        const reconstructedDataUrl = `data:${geminiResult.mimeType};base64,${geminiResult.imageData}`;
        
        const reconstructedImg = new Image();
        reconstructedImg.src = reconstructedDataUrl;
        await new Promise(r => {
            reconstructedImg.onload = r;
            reconstructedImg.onerror = () => r();
        });
        
        const finalCanvas = document.createElement('canvas');
        // CRITICAL FIX: Ensure the final canvas strictly matches our physical crop dimensions
        finalCanvas.width = cropW;
        finalCanvas.height = cropH;
        const finalCtx = finalCanvas.getContext('2d');
        finalCtx.imageSmoothingEnabled = true;
        finalCtx.imageSmoothingQuality = 'high';
        
        // Force the AI output to stretch/shrink back to our exact padded local coordinate box
        finalCtx.drawImage(reconstructedImg, 0, 0, cropW, cropH);
        
        const originalImageDataObj = ctx.getImageData(0, 0, cropW, cropH);
        const generatedImageDataObj = finalCtx.getImageData(0, 0, cropW, cropH);
        
        let processedAlphaData;
        if (useMaskFormat) {
            processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                taskId, originalImageDataObj, reconstructedDataUrl, layerName, () => {}
            );
        } else {
            processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                taskId, generatedImageDataObj, null, layerName, () => {},
                { type: 'solid', bgColor: bestBackdrop.rgb }
            );
        }
        
        const finalImageData = new ImageData(processedAlphaData, finalCanvas.width, finalCanvas.height);
        finalCtx.putImageData(finalImageData, 0, 0);
        
        const trimmed = trimTransparentCanvas(finalCanvas);
        cropW = trimmed.width;
        cropH = trimmed.height;
        transparentDataUrl = trimmed.toDataURL('image/png');
    }

    // Success! Update the layer definition
    layerDefinition.cutoutUrl = transparentDataUrl;
    layerDefinition.assetStatus = 'ready';
    layerDefinition.activeVersionId = 'base';
    
    if (!layerDefinition.versions) {
        layerDefinition.versions = [];
    }
    
    // Also trigger clean background removal if it creates transparency issues!
    await cleanBackground(transparentDataUrl, layerDefinition);
    
    layerDefinition.versions.push({
        id: 'base',
        prompt: '原始提取',
        cutoutUrl: layerDefinition.cutoutUrl,
        timestamp: Date.now(),
        type: 'extraction'
    });

    state.workbenchItems.set(itemId, item);
    await persistLayerStateToRuntime(itemId, item);
    addMessage({ sender: 'bot', type: 'text', content: `✅ 图层 **${layerName}** 已成功提取并加入到图层列表。`});

    if(document.getElementById('workbenchLayerListModal') && document.getElementById('workbenchLayerListModal').style.display !== 'none') {
        renderLayerList(currentLayers, itemId);
    }
    renderCanvasLayers(itemId);
}

// Expose to window for legacy support
window.triggerLayerExplosion = triggerLayerExplosion;
window.triggerMagicLayers = triggerMagicLayers;
window.runAutomaticCompletionForItem = runAutomaticCompletionForItem;
window.handleQuickFusionSync = handleQuickFusionSync;
window.handleIsolatedAssetEdit = handleIsolatedAssetEdit;
window.performPreciseEdit = performPreciseEdit;
window.extractSingleBoxLayer = extractSingleBoxLayer;
