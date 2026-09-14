import { analyzeImageLayers, editOrQueryImageWithGemini } from '../ai-services/skills-engine.js';
import { dataURLToFile, getClosestSupportedAspectRatio, getProxiedUrl } from '../core/utils.js';
import { getImageModel } from '../ai-services/gemini-client.js';
import { segmentSingleLayer } from './segmentation-service.js';
import { uploadImageToOSS } from './ossService.js';
import {
    prepareCompletionAssetPreflight
} from './object-completion-planner.js';
import { resolveCompletionTaskEntities } from './completion-contract.js';
import { buildCanonicalEvidence } from './completion-evidence.js';

// Keep completion previews opt-in. Each preview holds a decoded image, a
// base64/data URL, and a browser image element, so leaving them in the chat
// during a large Magic Layers task can materially increase renderer memory.
const COMPLETION_DEBUG_CHAT_PREVIEWS = true;
// Keep the complete completion chain visible during diagnosis: observed SAM,
// GPT scene, raw second SAM, and the reconciled canonical asset.  These are
// intentionally transient chat previews and are not persisted as messages.
const MAX_COMPLETION_DEBUG_PREVIEWS = 10;
const completionDebugPreviewNodes = [];

function asDataUrl(imageData, mimeType = 'image/png') {
    if (!imageData) return '';
    if (typeof imageData === 'string' && imageData.startsWith('data:image/')) return imageData;
    return `data:${mimeType};base64,${imageData}`;
}

function getFirstGeneratedImage(result) {
    const image = Array.isArray(result?.imageData) ? result.imageData[0] : result?.imageData;
    if (image && typeof image === 'object') {
        return asDataUrl(image.imageData, image.mimeType || result?.mimeType);
    }
    return asDataUrl(image, result?.mimeType || 'image/png');
}

function pushCompletionDebugPreview(label, source, metadata = {}) {
    if (!COMPLETION_DEBUG_CHAT_PREVIEWS || !source || typeof window === 'undefined') return;
    const addMessage = window.addMessage;
    if (typeof addMessage !== 'function') return;
    const details = Object.entries(metadata)
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .map(([key, value]) => `${key}: ${Array.isArray(value) ? `[${value.join(', ')}]` : value}`)
        .join('  |  ');
    const messageNode = addMessage({
        sender: 'bot',
        type: 'image',
        visibility: 'debug',
        persist: false,
        content: `补全调试 - ${label}${details ? `\n${details}` : ''}`,
        imageData: { src: source, mimeType: 'image/png' }
    });
    if (messageNode) {
        completionDebugPreviewNodes.push(messageNode);
        while (completionDebugPreviewNodes.length > MAX_COMPLETION_DEBUG_PREVIEWS) {
            const staleNode = completionDebugPreviewNodes.shift();
            staleNode?.remove();
        }
    }
}

function loadImage(source) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error('completion_generated_image_load_failed'));
        image.src = getProxiedUrl(source);
    });
}

function releaseImage(image) {
    if (!image) return;
    image.onload = null;
    image.onerror = null;
    try {
        image.src = '';
    } catch (error) {
        // The image is already eligible for collection; cleanup is best effort.
    }
}

function releaseCanvas(canvas) {
    if (!canvas) return;
    try {
        // Resetting dimensions releases the browser's native backing store
        // immediately instead of waiting for a later renderer GC cycle.
        canvas.width = 1;
        canvas.height = 1;
        canvas.getContext('2d')?.clearRect(0, 0, 1, 1);
    } catch (error) {
        // Temporary diagnostic/compositing canvases have no durable owner.
    }
}

function getLayerMaskSource(layer) {
    if (layer?.mask?.cutoutUrl) return { url: layer.mask.cutoutUrl, mode: 'alpha' };
    if (layer?.mask?.visibleMaskUrl) return { url: layer.mask.visibleMaskUrl, mode: 'luminance' };
    if (layer?.mask?.fullMaskUrl) return { url: layer.mask.fullMaskUrl, mode: 'luminance' };
    if (layer?.cutoutUrl) return { url: layer.cutoutUrl, mode: 'alpha' };
    if (layer?.previewUrl) return { url: layer.previewUrl, mode: 'alpha' };
    return null;
}

function getTaskLockedSceneSource(item, preflight) {
    const sceneLock = item?.__magicLayersSceneLock;
    if (
        sceneLock?.dataUrl &&
        preflight?.source?.taskLockId &&
        sceneLock.id === preflight.source.taskLockId
    ) {
        return sceneLock.dataUrl;
    }
    return preflight?.source?.imageUrl;
}

function resolveCompletionContext(graph, task) {
    return resolveCompletionTaskEntities(task, graph?.layers || []);
}

async function drawLayerMaskInScene(context, layer, sceneWidth, sceneHeight) {
    const maskSource = getLayerMaskSource(layer);
    if (!maskSource || !Array.isArray(layer?.bbox) || layer.bbox.length !== 4) return false;
    const image = await loadImage(maskSource.url);
    const [ymin, xmin, ymax, xmax] = layer.bbox.map(Number);
    const x = (xmin / 1000) * sceneWidth;
    const y = (ymin / 1000) * sceneHeight;
    const width = Math.max(1, ((xmax - xmin) / 1000) * sceneWidth);
    const height = Math.max(1, ((ymax - ymin) / 1000) * sceneHeight);

    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = Math.max(1, image.naturalWidth || image.width);
    sourceCanvas.height = Math.max(1, image.naturalHeight || image.height);
    const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
    sourceContext.drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height);
    const pixels = sourceContext.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const data = pixels.data;
    let hasVisiblePixels = false;
    for (let index = 0; index < data.length; index += 4) {
        const luminance = data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722;
        const alpha = data[index + 3];
        // Transparent cutouts use alpha; grayscale masks use luminance.
        const maskAlpha = maskSource.mode === 'alpha'
            ? alpha
            : Math.round((luminance / 255) * alpha);
        data[index] = 255;
        data[index + 1] = 255;
        data[index + 2] = 255;
        data[index + 3] = maskAlpha;
        hasVisiblePixels ||= maskAlpha >= 32;
    }
    if (!hasVisiblePixels) {
        releaseCanvas(sourceCanvas);
        releaseImage(image);
        return false;
    }
    sourceContext.putImageData(pixels, 0, 0);
    context.drawImage(sourceCanvas, x, y, width, height);
    releaseCanvas(sourceCanvas);
    releaseImage(image);
    return true;
}

async function drawLayerPixelsInScene(context, source, bbox, sceneWidth, sceneHeight) {
    if (!source || !Array.isArray(bbox) || bbox.length !== 4) return false;
    const image = await loadImage(source);
    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    const x = (xmin / 1000) * sceneWidth;
    const y = (ymin / 1000) * sceneHeight;
    const width = Math.max(1, ((xmax - xmin) / 1000) * sceneWidth);
    const height = Math.max(1, ((ymax - ymin) / 1000) * sceneHeight);
    context.drawImage(image, x, y, width, height);
    releaseImage(image);
    return true;
}

function expandSceneMaskWithinBbox(maskCanvas, bbox, radius) {
    const width = maskCanvas.width;
    const height = maskCanvas.height;
    const context = maskCanvas.getContext('2d', { willReadFrequently: true });
    const imageData = context.getImageData(0, 0, width, height);
    const alpha = new Uint8ClampedArray(width * height);
    for (let pixel = 0; pixel < alpha.length; pixel += 1) {
        alpha[pixel] = imageData.data[pixel * 4 + 3];
    }
    const expanded = dilateAlphaChannel(alpha, width, height, radius);
    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    const clipX1 = Math.max(0, Math.floor((xmin / 1000) * width));
    const clipY1 = Math.max(0, Math.floor((ymin / 1000) * height));
    const clipX2 = Math.min(width, Math.ceil((xmax / 1000) * width));
    const clipY2 = Math.min(height, Math.ceil((ymax / 1000) * height));
    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            const pixel = y * width + x;
            const nextAlpha = x >= clipX1 && x < clipX2 && y >= clipY1 && y < clipY2
                ? expanded[pixel]
                : 0;
            const offset = pixel * 4;
            imageData.data[offset] = 255;
            imageData.data[offset + 1] = 255;
            imageData.data[offset + 2] = 255;
            imageData.data[offset + 3] = nextAlpha;
        }
    }
    context.putImageData(imageData, 0, 0);
    return { radius, clipBbox: [ymin, xmin, ymax, xmax] };
}

function dilateAlphaChannel(alpha, width, height, radius) {
    const horizontal = new Uint8ClampedArray(alpha.length);
    const output = new Uint8ClampedArray(alpha.length);
    const span = Math.max(0, Math.round(radius));

    for (let y = 0; y < height; y += 1) {
        let windowMax = 0;
        for (let x = 0; x < width + span; x += 1) {
            const enteringX = x;
            if (enteringX < width) {
                windowMax = Math.max(windowMax, alpha[y * width + enteringX]);
            }
            const leavingX = x - (span * 2 + 1);
            if (leavingX >= 0 && leavingX < width) {
                let replacementMax = 0;
                const start = Math.max(0, x - span * 2);
                const end = Math.min(width - 1, x);
                for (let candidateX = start; candidateX <= end; candidateX += 1) {
                    replacementMax = Math.max(replacementMax, alpha[y * width + candidateX]);
                }
                windowMax = replacementMax;
            }
            const outputX = x - span;
            if (outputX >= 0 && outputX < width) {
                horizontal[y * width + outputX] = windowMax;
            }
        }
    }

    for (let x = 0; x < width; x += 1) {
        let windowMax = 0;
        for (let y = 0; y < height + span; y += 1) {
            const enteringY = y;
            if (enteringY < height) {
                windowMax = Math.max(windowMax, horizontal[enteringY * width + x]);
            }
            const leavingY = y - (span * 2 + 1);
            if (leavingY >= 0 && leavingY < height) {
                let replacementMax = 0;
                const start = Math.max(0, y - span * 2);
                const end = Math.min(height - 1, y);
                for (let candidateY = start; candidateY <= end; candidateY += 1) {
                    replacementMax = Math.max(replacementMax, horizontal[candidateY * width + x]);
                }
                windowMax = replacementMax;
            }
            const outputY = y - span;
            if (outputY >= 0 && outputY < height) {
                output[outputY * width + x] = windowMax;
            }
        }
    }
    return output;
}

function retainObservedConnectedComponents(observedPixels, candidatePixels, allowedGeneratedArea, width, height) {
    const totalPixels = width * height;
    const visited = new Uint8Array(totalPixels);
    const keep = new Uint8Array(totalPixels);
    const queue = new Int32Array(totalPixels);
    const component = [];
    let keptComponentCount = 0;
    let discardedComponentCount = 0;
    let keptGeneratedOnlyComponentCount = 0;
    // A hidden support/leg can be completely disconnected from the visible
    // observation after inpainting. Keep a meaningful component only when it
    // is wholly inside the verified occlusion band; tiny isolated speckles
    // remain rejected.
    const minimumGeneratedOnlyPixels = 16;
    const isObserved = pixel => observedPixels[pixel * 4 + 3] >= 16;
    const isGenerated = pixel => candidatePixels[pixel * 4 + 3] >= 16 && allowedGeneratedArea[pixel] >= 48;

    for (let start = 0; start < totalPixels; start += 1) {
        if (visited[start] || (!isObserved(start) && !isGenerated(start))) continue;
        let head = 0;
        let tail = 0;
        let hasObserved = false;
        let hasOutsideAllowedGenerated = false;
        component.length = 0;
        queue[tail++] = start;
        visited[start] = 1;

        while (head < tail) {
            const pixel = queue[head++];
            component.push(pixel);
            hasObserved ||= isObserved(pixel);
            if (!isObserved(pixel) && !isGenerated(pixel)) hasOutsideAllowedGenerated = true;
            const x = pixel % width;
            const y = Math.floor(pixel / width);
            for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
                for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
                    if (offsetX === 0 && offsetY === 0) continue;
                    const nextX = x + offsetX;
                    const nextY = y + offsetY;
                    if (nextX < 0 || nextY < 0 || nextX >= width || nextY >= height) continue;
                    const next = nextY * width + nextX;
                    if (visited[next] || (!isObserved(next) && !isGenerated(next))) continue;
                    visited[next] = 1;
                    queue[tail++] = next;
                }
            }
        }
        const generatedOnly = !hasObserved && !hasOutsideAllowedGenerated &&
            component.length >= minimumGeneratedOnlyPixels;
        if (hasObserved || generatedOnly) {
            keptComponentCount += 1;
            if (generatedOnly) keptGeneratedOnlyComponentCount += 1;
            component.forEach(pixel => { keep[pixel] = 1; });
        } else {
            discardedComponentCount += 1;
        }
    }
    return {
        keep,
        keptComponentCount,
        discardedComponentCount,
        keptGeneratedOnlyComponentCount,
        minimumGeneratedOnlyPixels
    };
}

function subtractOccluderAlphaFromObserved(observedPixels, occluderPixels) {
    const sanitized = new Uint8ClampedArray(observedPixels);
    let removedPixels = 0;
    let removedAlpha = 0;
    for (let offset = 0; offset < sanitized.length; offset += 4) {
        const observedAlpha = sanitized[offset + 3];
        if (observedAlpha < 16) continue;
        const occluderAlpha = occluderPixels[offset + 3];
        if (occluderAlpha < 16) continue;
        const nextAlpha = Math.round(observedAlpha * (255 - occluderAlpha) / 255);
        if (nextAlpha < observedAlpha) {
            removedPixels += 1;
            removedAlpha += observedAlpha - nextAlpha;
        }
        sanitized[offset + 3] = nextAlpha;
    }
    return { pixels: sanitized, removedPixels, removedAlpha };
}

function trimAlphaCanvas(canvas, sourceWidth, sourceHeight) {
    const context = canvas.getContext('2d', { willReadFrequently: true });
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    let top = canvas.height;
    let left = canvas.width;
    let bottom = 0;
    let right = 0;
    for (let y = 0; y < canvas.height; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
            if (pixels.data[(y * canvas.width + x) * 4 + 3] < 16) continue;
            top = Math.min(top, y);
            left = Math.min(left, x);
            bottom = Math.max(bottom, y + 1);
            right = Math.max(right, x + 1);
        }
    }
    if (right <= left || bottom <= top) return null;
    const trimmed = document.createElement('canvas');
    trimmed.width = right - left;
    trimmed.height = bottom - top;
    trimmed.getContext('2d').putImageData(
        context.getImageData(left, top, trimmed.width, trimmed.height),
        0,
        0
    );
    return {
        canvas: trimmed,
        bbox: [
            Number((top / sourceHeight * 1000).toFixed(3)),
            Number((left / sourceWidth * 1000).toFixed(3)),
            Number((bottom / sourceHeight * 1000).toFixed(3)),
            Number((right / sourceWidth * 1000).toFixed(3))
        ],
        pixelBbox: [top, left, bottom, right]
    };
}

function isNormalizedBbox(bbox) {
    return Array.isArray(bbox) && bbox.length === 4 && bbox.every(value => Number.isFinite(Number(value)));
}

function unionNormalizedBboxes(first, second) {
    const [firstYmin, firstXmin, firstYmax, firstXmax] = first.map(Number);
    const [secondYmin, secondXmin, secondYmax, secondXmax] = second.map(Number);
    return [
        Math.min(firstYmin, secondYmin),
        Math.min(firstXmin, secondXmin),
        Math.max(firstYmax, secondYmax),
        Math.max(firstXmax, secondXmax)
    ];
}

function normalizedBboxArea(bbox) {
    if (!isNormalizedBbox(bbox)) return 0;
    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    return Math.max(0, ymax - ymin) * Math.max(0, xmax - xmin);
}

function expandNormalizedBbox(bbox, paddingY, paddingX) {
    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    return [
        Math.max(0, ymin - paddingY),
        Math.max(0, xmin - paddingX),
        Math.min(1000, ymax + paddingY),
        Math.min(1000, xmax + paddingX)
    ].map(value => Number(value.toFixed(3)));
}

function normalizedBboxesTouch(first, second, padding = 0) {
    const [firstYmin, firstXmin, firstYmax, firstXmax] = first.map(Number);
    const [secondYmin, secondXmin, secondYmax, secondXmax] = second.map(Number);
    return !(
        firstYmax + padding < secondYmin ||
        secondYmax + padding < firstYmin ||
        firstXmax + padding < secondXmin ||
        secondXmax + padding < firstXmin
    );
}

async function deriveCompletionSamPromptBbox({
    observedUrl,
    observedBbox,
    occluderLayers,
    fallbackBbox,
    sourceSize
}) {
    if (!observedUrl || !isNormalizedBbox(observedBbox)) {
        return { bbox: cloneBbox(fallbackBbox), source: 'preflight_fallback' };
    }
    const width = Math.max(1, Number(sourceSize?.width) || 1);
    const height = Math.max(1, Number(sourceSize?.height) || 1);
    const observedCanvas = document.createElement('canvas');
    observedCanvas.width = width;
    observedCanvas.height = height;
    const drawn = await drawLayerPixelsInScene(
        observedCanvas.getContext('2d'),
        observedUrl,
        observedBbox,
        width,
        height
    );
    const visibleBounds = drawn ? trimAlphaCanvas(observedCanvas, width, height) : null;
    if (!visibleBounds?.bbox) {
        return { bbox: cloneBbox(fallbackBbox), source: 'preflight_fallback_no_observed_alpha' };
    }

    let seed = visibleBounds.bbox;
    for (const occluder of occluderLayers || []) {
        if (!isNormalizedBbox(occluder?.bbox)) continue;
        // A foreground object only expands the prompt when it touches the
        // visible target. Distant poster decorations must not turn a person
        // completion request back into a whole-scene SAM prompt.
        if (normalizedBboxesTouch(seed, occluder.bbox, 32)) {
            seed = unionNormalizedBboxes(seed, occluder.bbox);
        }
    }
    const heightSpan = Math.max(1, seed[2] - seed[0]);
    const widthSpan = Math.max(1, seed[3] - seed[1]);
    return {
        bbox: expandNormalizedBbox(
            seed,
            Math.max(20, heightSpan * 0.045),
            Math.max(20, widthSpan * 0.045)
        ),
        source: 'observed_alpha_plus_touching_occluders',
        observedAlphaBbox: visibleBounds.bbox
    };
}

function canUseDirectSecondSamCutout(segmented, foregroundOwnershipEvidenceStatus) {
    const quality = segmented?.quality || {};
    const selected = Array.isArray(quality.debugCandidates)
        ? quality.debugCandidates.find(candidate => candidate?.selected)
        : null;
    const recoveryAudit = quality.completionRecoveryAudit || {};
    const hiddenEvidence = Math.max(
        Number(recoveryAudit.allowedNewPixels) || 0,
        Number(quality.completionHiddenRescuePixels) || 0,
        Number(selected?.completionRecoveryPixels) || 0
    );
    const ownershipAvailable = foregroundOwnershipEvidenceStatus === 'attached' ||
        foregroundOwnershipEvidenceStatus === 'empty_valid';
    const recoveryAccepted = String(recoveryAudit.status || '').startsWith('accepted:');
    return Boolean(
        segmented?.dataUrl &&
        quality.completionIdentityVerified === true &&
        quality.completionStructureVerified === true &&
        quality.runtimeAction !== 'hold' &&
        quality.shouldGenerateRuntimeLayer !== false &&
        selected?.candidate !== false &&
        hiddenEvidence >= 64 &&
        (recoveryAccepted || Number(selected?.completionRecoveryPixels) >= 256) &&
        ownershipAvailable
    );
}

async function reconcileCompletionSegmentation({
    observedUrl,
    observedBbox,
    segmented,
    completionStructureVerified = false,
    completionForegroundContextMask = null,
    foregroundOwnershipEvidenceStatus = 'unavailable',
    graph,
    task,
    sourceSize
}) {
    if (!observedUrl || !Array.isArray(observedBbox) || !segmented?.dataUrl || !Array.isArray(segmented.bbox)) {
        throw new Error('completion_reconciliation_inputs_unavailable');
    }
    const width = Math.max(1, Number(sourceSize?.width) || 1);
    const height = Math.max(1, Number(sourceSize?.height) || 1);
    const observedCanvas = document.createElement('canvas');
    observedCanvas.width = width;
    observedCanvas.height = height;
    const candidateCanvas = document.createElement('canvas');
    candidateCanvas.width = width;
    candidateCanvas.height = height;
    const occluderCanvas = document.createElement('canvas');
    occluderCanvas.width = width;
    occluderCanvas.height = height;

    await drawLayerPixelsInScene(observedCanvas.getContext('2d'), observedUrl, observedBbox, width, height);
    await drawLayerPixelsInScene(candidateCanvas.getContext('2d'), segmented.dataUrl, segmented.bbox, width, height);

    // Ownership context marks non-target foreground layers that SAM must not
    // absorb after inpainting (flowers, stools, ornaments, etc.).
    let foregroundContextPixels = null;
    let foregroundContextCanvas = null;
    if (completionForegroundContextMask) {
        try {
            const contextImage = await loadImage(completionForegroundContextMask);
            foregroundContextCanvas = document.createElement('canvas');
            foregroundContextCanvas.width = width;
            foregroundContextCanvas.height = height;
            const foregroundContext = foregroundContextCanvas.getContext('2d', {
                willReadFrequently: true
            });
            foregroundContext.drawImage(contextImage, 0, 0, width, height);
            foregroundContextPixels = foregroundContext.getImageData(0, 0, width, height).data;
            releaseImage(contextImage);
        } catch (error) {
            console.warn('[Object Completion] foreground context mask unavailable during reconciliation:', error);
        }
    }

    const layersById = new Map((graph?.layers || []).map(layer => [layer.layerId, layer]));
    const resolvedTask = resolveCompletionContext(graph, task);
    const occluderContext = occluderCanvas.getContext('2d');
    const drawnOccluderIds = [];
    for (const layerId of resolvedTask.occluderLayerIds) {
        const occluder = layersById.get(layerId);
        if (!occluder) continue;
        try {
            if (await drawLayerMaskInScene(occluderContext, occluder, width, height)) {
                drawnOccluderIds.push(layerId);
            }
        } catch (error) {
            console.warn('[Object Completion] reconciliation occluder mask unavailable:', layerId, error);
        }
    }
    if (!drawnOccluderIds.length) throw new Error('completion_reconciliation_occluder_masks_unavailable');

    const rawObservedPixels = observedCanvas.getContext('2d', { willReadFrequently: true })
        .getImageData(0, 0, width, height).data;
    const candidatePixels = candidateCanvas.getContext('2d', { willReadFrequently: true })
        .getImageData(0, 0, width, height).data;
    const occluderPixels = occluderContext.getImageData(0, 0, width, height).data;
    // The original visible cutout may already contain foreground pixels if the
    // first SAM pass merged a chair/stool into the target. Remove those pixels
    // before deciding which regions may anchor the generated completion.
    const observedSanitization = subtractOccluderAlphaFromObserved(rawObservedPixels, occluderPixels);
    const observedPixels = observedSanitization.pixels;
    const occluderAlpha = new Uint8ClampedArray(width * height);
    for (let index = 0; index < occluderAlpha.length; index += 1) {
        occluderAlpha[index] = occluderPixels[index * 4 + 3];
    }
    // Allow a narrow edge band so SAM can recover the part directly behind a
    // foreground mask, while preventing model drift on visible scene pixels.
    const dilationRadius = Math.max(2, Math.round(Math.min(width, height) * 0.008));
    const allowedGeneratedArea = dilateAlphaChannel(occluderAlpha, width, height, dilationRadius);
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = width;
    outputCanvas.height = height;
    const outputContext = outputCanvas.getContext('2d', { willReadFrequently: true });
    const output = outputContext.createImageData(width, height);
    const observedConnectedComponents = retainObservedConnectedComponents(
        observedPixels,
        candidatePixels,
        allowedGeneratedArea,
        width,
        height
    );
    // The backend's identity/structure flags are evidence about the candidate;
    // they are never pixel authority.  A candidate can overlap the observed
    // target and still contain generated background, a stool, or an ornament.
    // The only pixel sources allowed into the canonical asset are therefore:
    //   1) the sanitized first-pass observation, and
    //   2) a second-SAM delta inside the verified occlusion transition band.
    // Keep the flag in the audit output so callers can see that the candidate
    // was structurally verified, but do not let it bypass the hidden-region
    // containment rules.
    const verifiedCandidateEvidence = completionStructureVerified === true;
    // Occlusion tells us where the target may be hidden; it does not prove
    // that pixels in that area belong to the target.  A real foreground
    // ownership mask is therefore mandatory before any generated hidden
    // pixel can enter canonical.  With no mask, retain observed pixels for
    // diagnostics but hold the completion instead of publishing contamination.
    const foregroundOwnershipEvidenceAvailable =
        foregroundOwnershipEvidenceStatus === 'attached' ||
        foregroundOwnershipEvidenceStatus === 'empty_valid';
    let observedPixelCount = 0;
    let generatedPixelCount = 0;
    let observedPriorityPixels = 0;
    let hiddenCandidatePixels = 0;
    let allowedHiddenCandidatePixels = 0;
    let hiddenKeptPixels = 0;
    let allowedHiddenRegionPixels = 0;
    let unknownGeneratedPixels = 0;
    let foregroundContextBlockedPixels = 0;
    let ownershipEvidenceMissingPixels = 0;
    let discardedCandidatePixelCount = 0;
    let disconnectedCandidatePixelCount = 0;
    for (let pixel = 0; pixel < width * height; pixel += 1) {
        const offset = pixel * 4;
        const observedAlpha = observedPixels[offset + 3];
        const candidateAlpha = candidatePixels[offset + 3];
        const candidateAllowed = allowedGeneratedArea[pixel] >= 48;
        const candidateContextBlocked = Boolean(
            foregroundContextPixels && foregroundContextPixels[offset + 3] >= 48
        );
        if (candidateAllowed) allowedHiddenRegionPixels += 1;
        if (candidateAlpha >= 16 && observedAlpha >= 16) {
            // A generated candidate may redraw an observed pixel, but it is
            // never allowed to replace the sanitized original observation.
            observedPriorityPixels += 1;
        }
        if (candidateAlpha >= 16 && observedAlpha < 16) {
            hiddenCandidatePixels += 1;
            if (candidateAllowed) allowedHiddenCandidatePixels += 1;
            if (!foregroundOwnershipEvidenceAvailable) {
                ownershipEvidenceMissingPixels += 1;
                unknownGeneratedPixels += 1;
            } else if (!candidateContextBlocked &&
                candidateAllowed &&
                observedConnectedComponents.keep[pixel] === 1) {
                hiddenKeptPixels += 1;
            } else {
                unknownGeneratedPixels += 1;
                if (candidateContextBlocked) foregroundContextBlockedPixels += 1;
            }
        }
        // Pixel authority is explicit: sanitized observed pixels first;
        // generated pixels are limited to the allowed occlusion band and
        // connected-component evidence, regardless of the backend status.
        const candidateUsable = candidateAlpha >= 16 &&
            observedAlpha < 16 &&
            foregroundOwnershipEvidenceAvailable &&
            !candidateContextBlocked &&
            candidateAllowed &&
            observedConnectedComponents.keep[pixel] === 1;
        const useGenerated = candidateUsable;
        const useObserved = !useGenerated && observedAlpha >= 16;
        if (useObserved) observedPixelCount += 1;
        if (useGenerated) generatedPixelCount += 1;
        if (!candidateUsable && candidateAlpha >= 16 && !candidateAllowed) discardedCandidatePixelCount += 1;
        if (!candidateUsable && candidateAlpha >= 16 && candidateAllowed && observedConnectedComponents.keep[pixel] === 0) {
            disconnectedCandidatePixelCount += 1;
        }
        if (!useObserved && !useGenerated) continue;
        const sourcePixels = useGenerated ? candidatePixels : observedPixels;
        output.data[offset] = sourcePixels[offset];
        output.data[offset + 1] = sourcePixels[offset + 1];
        output.data[offset + 2] = sourcePixels[offset + 2];
        output.data[offset + 3] = useObserved
            ? observedAlpha
            : candidateAlpha;
    }
    outputContext.putImageData(output, 0, 0);
    const trimmed = trimAlphaCanvas(outputCanvas, width, height);
    if (!trimmed) throw new Error('completion_reconciliation_empty_result');
    const referenceGeometryArea = Math.max(
        normalizedBboxArea(observedBbox),
        normalizedBboxArea(segmented.bbox)
    );
    const reconciledBboxExpansion = Math.max(
        0,
        normalizedBboxArea(trimmed.bbox) / Math.max(1, referenceGeometryArea) - 1
    );
    console.info('[Object Completion] reconciled canonical mask', {
        targetLayerId: task?.targetLayerId || null,
        sourceSize: `${width}x${height}`,
        observedBbox,
        segmentedBbox: segmented.bbox,
        reconciledBbox: trimmed.bbox,
        dilationRadius,
        drawnOccluderIds,
        observedPixelCount,
        generatedPixelCount,
        discardedCandidatePixelCount,
        disconnectedCandidatePixelCount,
        observedPriorityPixels,
        hiddenCandidatePixels,
        hiddenKeptPixels,
        hiddenCoverage: Number((hiddenKeptPixels / Math.max(
            1,
            allowedHiddenCandidatePixels
        )).toFixed(4)),
        allowedHiddenCandidatePixels,
        allowedHiddenRegionPixels,
        unknownGeneratedPixels,
        foregroundContextBlockedPixels,
        ownershipEvidenceMissingPixels,
        bboxExpansion: Number(reconciledBboxExpansion.toFixed(4)),
        observedOccluderPixelsRemoved: observedSanitization.removedPixels,
        observedOccluderAlphaRemoved: observedSanitization.removedAlpha,
        keptComponentCount: observedConnectedComponents.keptComponentCount,
        discardedComponentCount: observedConnectedComponents.discardedComponentCount,
        keptGeneratedOnlyComponentCount: observedConnectedComponents.keptGeneratedOnlyComponentCount,
        minimumGeneratedOnlyPixels: observedConnectedComponents.minimumGeneratedOnlyPixels,
        policy: 'sanitized_observed_first_hidden_only_allowed_occlusion_band',
        // Compatibility field: this is deliberately always false now.  The
        // old meaning incorrectly granted full candidate-bbox authority.
        verifiedCandidateAuthority: false,
        verifiedCandidateEvidence,
        foregroundOwnershipEvidenceAvailable,
        foregroundContextMaskAttached: Boolean(foregroundContextPixels),
        foregroundOwnershipEvidenceStatus
    });
    const dataUrl = trimmed.canvas.toDataURL('image/png');
    releaseCanvas(observedCanvas);
    releaseCanvas(candidateCanvas);
    releaseCanvas(occluderCanvas);
    releaseCanvas(foregroundContextCanvas);
    releaseCanvas(outputCanvas);
    releaseCanvas(trimmed.canvas);
    return {
        dataUrl,
        bbox: trimmed.bbox,
        pixelBbox: trimmed.pixelBbox,
        observedPixelCount,
        generatedPixelCount,
        discardedCandidatePixelCount,
        disconnectedCandidatePixelCount,
        observedPriorityPixels,
        hiddenCandidatePixels,
        hiddenKeptPixels,
        hiddenCoverage: Number((hiddenKeptPixels / Math.max(
            1,
            allowedHiddenCandidatePixels
        )).toFixed(4)),
        allowedHiddenCandidatePixels,
        allowedHiddenRegionPixels,
        unknownGeneratedPixels,
        foregroundContextBlockedPixels,
        ownershipEvidenceMissingPixels,
        bboxExpansion: Number(reconciledBboxExpansion.toFixed(4)),
        observedOccluderPixelsRemoved: observedSanitization.removedPixels,
        observedOccluderAlphaRemoved: observedSanitization.removedAlpha,
        keptComponentCount: observedConnectedComponents.keptComponentCount,
        discardedComponentCount: observedConnectedComponents.discardedComponentCount,
        dilationRadius,
        drawnOccluderIds,
        // Kept for existing UI consumers; see the policy comment above.
        verifiedCandidateAuthority: false,
        verifiedCandidateEvidence,
        foregroundOwnershipEvidenceAvailable,
        foregroundContextMaskAttached: Boolean(foregroundContextPixels),
        foregroundOwnershipEvidenceStatus
    };
}

async function materializeSceneCompletionInput(item, asset, preflight, targetLayer) {
    const sourceUrl = getTaskLockedSceneSource(item, preflight);
    const graph = item?.semanticViews?.layerGraph;
    const task = graph?.completionTasks?.find(candidate => candidate.id === asset.taskId);
    if (!sourceUrl || !task || !targetLayer) throw new Error('scene_completion_context_unavailable');

    const sceneImage = await loadImage(sourceUrl);
    const width = Math.max(1, sceneImage.naturalWidth || sceneImage.width);
    const height = Math.max(1, sceneImage.naturalHeight || sceneImage.height);
    const sceneCanvas = document.createElement('canvas');
    sceneCanvas.width = width;
    sceneCanvas.height = height;
    sceneCanvas.getContext('2d').drawImage(sceneImage, 0, 0, width, height);

    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = width;
    maskCanvas.height = height;
    const maskContext = maskCanvas.getContext('2d');
    const occluderMaskCanvas = document.createElement('canvas');
    occluderMaskCanvas.width = width;
    occluderMaskCanvas.height = height;
    const occluderMaskContext = occluderMaskCanvas.getContext('2d');
    const layersById = new Map((graph?.layers || []).map(layer => [layer.layerId, layer]));
    const resolvedTask = resolveCompletionContext(graph, task);
    const occluders = resolvedTask.occluderLayerIds
        .map(layerId => layersById.get(layerId))
        .filter(Boolean);
    const baseClipBbox = preflight.geometry.cropBbox || targetLayer.bbox;
    // The planner crop is derived from the target geometry. A foreground
    // stool can extend below/beyond that crop, so clipping the edit mask to
    // the planner box leaves part of the occluder outside the GPT edit zone.
    // Union every verified occluder bbox into the edit region while keeping
    // the rest of the scene protected.
    const clipBbox = occluders.reduce((current, occluder) => {
        return isNormalizedBbox(occluder?.bbox)
            ? unionNormalizedBboxes(current, occluder.bbox)
            : current;
    }, cloneBbox(baseClipBbox) || [0, 0, 1000, 1000]);
    const [clipYmin, clipXmin, clipYmax, clipXmax] = clipBbox.map(Number);
    maskContext.save();
    maskContext.beginPath();
    maskContext.rect(
        (clipXmin / 1000) * width,
        (clipYmin / 1000) * height,
        Math.max(1, ((clipXmax - clipXmin) / 1000) * width),
        Math.max(1, ((clipYmax - clipYmin) / 1000) * height)
    );
    maskContext.clip();

    const drawnOccluderIds = [];
    for (const occluder of occluders) {
        try {
            if (await drawLayerMaskInScene(maskContext, occluder, width, height)) {
                await drawLayerMaskInScene(occluderMaskContext, occluder, width, height);
                drawnOccluderIds.push(occluder.layerId);
            }
        } catch (error) {
            console.warn('[Object Completion] occluder mask unavailable for scene inpaint:', occluder.layerId, error);
        }
    }
    maskContext.restore();
    if (!drawnOccluderIds.length) throw new Error('scene_completion_occluder_masks_unavailable');

    // Give the editor a small transition band around the real foreground
    // silhouettes. The original scene remains protected outside this band;
    // the band only prevents a hard SAM edge from being copied into the GPT
    // inpaint and leaving a visible stool outline.
    // Give the inpaint a little more room around the verified occluder
    // silhouette.  A one-pixel residual edge is enough for GPT-image to
    // reconstruct a recognizable stool, which then contaminates the second
    // SAM pass.  This remains clipped to the union of the target and verified
    // occluder bboxes, so unrelated scene pixels stay protected.
    const maskExpansionRadius = Math.max(3, Math.round(Math.min(width, height) * 0.018));
    expandSceneMaskWithinBbox(maskCanvas, clipBbox, maskExpansionRadius);
    expandSceneMaskWithinBbox(occluderMaskCanvas, clipBbox, maskExpansionRadius);

    console.info('[Object Completion] scene inpaint input', {
        taskLockId: preflight?.source?.taskLockId || null,
        targetLayerId: asset.targetLayerId,
        sceneCanvas: `${width}x${height}`,
        sourceKind: preflight?.source?.kind || 'unknown',
        targetBbox: targetLayer.bbox,
        editBbox: clipBbox,
        editBboxSource: 'planner_crop_union_verified_occluders',
        occluders: occluders.map(occluder => ({
            id: occluder.layerId,
            name: occluder.name,
            zIndex: occluder.zIndex,
            maskDrawn: drawnOccluderIds.includes(occluder.layerId)
        })),
        targetZIndex: targetLayer.zIndex,
        targetEntityLayerIds: resolvedTask.target.layerIds,
        maskExpansionRadius,
        policy: 'original_scene_masked_occluders_only_with_transition_band'
    });
    const sceneDataUrl = sceneCanvas.toDataURL('image/png');
    const editMaskDataUrl = maskCanvas.toDataURL('image/png');
    const occluderMaskDataUrl = occluderMaskCanvas.toDataURL('image/png');
    releaseCanvas(sceneCanvas);
    releaseCanvas(maskCanvas);
    releaseCanvas(occluderMaskCanvas);
    releaseImage(sceneImage);
    return {
        sceneDataUrl,
        editMaskDataUrl,
        occluderMaskDataUrl,
        sourceSize: { width, height },
        drawnOccluderIds,
        maskExpansionRadius,
        targetLayer: { ...targetLayer, id: targetLayer.layerId || targetLayer.id }
    };
}

async function materializeCompletionObservationMask(asset, targetLayer, width, height) {
    const observedUrl = asset?.observedCutout?.cutoutUrl || targetLayer?.mask?.cutoutUrl;
    const observedBbox = asset?.observedCutout?.bbox || targetLayer?.bbox;
    if (!observedUrl || !Array.isArray(observedBbox)) return null;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const drawn = await drawLayerMaskInScene(
        canvas.getContext('2d'),
        { bbox: observedBbox, mask: { cutoutUrl: observedUrl } },
        width,
        height
    );
    if (!drawn) {
        releaseCanvas(canvas);
        return null;
    }
    const dataUrl = canvas.toDataURL('image/png');
    releaseCanvas(canvas);
    return dataUrl;
}

async function materializeCompletionForegroundContextMask(
    contextLayers,
    targetLayerId,
    occluderIds,
    promptBbox,
    width,
    height
) {
    if (!Array.isArray(contextLayers) || !Array.isArray(promptBbox)) return null;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    const excluded = new Set([targetLayerId, ...(occluderIds || [])]);
    let drawn = false;
    for (const layer of contextLayers) {
        const id = layer?.layerId || layer?.id;
        if (!id || excluded.has(id) || !isNormalizedBbox(layer?.bbox)) continue;
        const [y1, x1, y2, x2] = layer.bbox.map(Number);
        const [py1, px1, py2, px2] = promptBbox.map(Number);
        if (Math.min(x2, px2) <= Math.max(x1, px1) || Math.min(y2, py2) <= Math.max(y1, py1)) continue;
        try {
            drawn = (await drawLayerMaskInScene(context, layer, width, height)) || drawn;
        } catch (error) {
            console.warn('[Object Completion] foreground context mask unavailable:', id, error);
        }
    }
    if (!drawn) {
        releaseCanvas(canvas);
        return null;
    }
    const dataUrl = canvas.toDataURL('image/png');
    releaseCanvas(canvas);
    return dataUrl;
}

async function materializeCompletionOcclusionMask(graph, task, width, height) {
    if (!graph || !task) return null;
    const layersById = new Map((graph.layers || []).map(layer => [layer.layerId, layer]));
    const resolvedTask = resolveCompletionContext(graph, task);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    let drawn = false;
    for (const layerId of resolvedTask.occluderLayerIds) {
        const layer = layersById.get(layerId);
        if (!layer) continue;
        try {
            drawn = (await drawLayerMaskInScene(context, layer, width, height)) || drawn;
        } catch (error) {
            console.warn('[Object Completion] occlusion mask unavailable for second SAM:', layerId, error);
        }
    }
    if (!drawn) {
        releaseCanvas(canvas);
        return null;
    }
    const dataUrl = canvas.toDataURL('image/png');
    releaseCanvas(canvas);
    return dataUrl;
}

async function auditCanonicalSegmentationAgainstOccluders({
    segmented,
    targetLayer,
    graph,
    sourceSize
}) {
    const task = (graph?.completionTasks || [])
        .find(candidate => candidate.targetLayerId === targetLayer.layerId);
    const resolvedTask = task ? resolveCompletionContext(graph, task) : null;
    const taskOccluderIds = new Set(resolvedTask?.occluderLayerIds || []);
    const occluders = (graph?.layers || [])
        .filter(layer => taskOccluderIds.has(layer.layerId) && getLayerMaskSource(layer));
    if (!occluders.length || !Array.isArray(segmented?.bbox) || !segmented?.dataUrl) {
        return { status: 'not_applicable', checkedOccluderIds: [] };
    }

    const width = Math.max(1, Number(sourceSize?.width) || 1);
    const height = Math.max(1, Number(sourceSize?.height) || 1);
    const candidateImage = await loadImage(segmented.dataUrl);
    const candidateCanvas = document.createElement('canvas');
    candidateCanvas.width = width;
    candidateCanvas.height = height;
    const candidateContext = candidateCanvas.getContext('2d', { willReadFrequently: true });
    const [ymin, xmin, ymax, xmax] = segmented.bbox.map(Number);
    candidateContext.drawImage(
        candidateImage,
        (xmin / 1000) * width,
        (ymin / 1000) * height,
        Math.max(1, ((xmax - xmin) / 1000) * width),
        Math.max(1, ((ymax - ymin) / 1000) * height)
    );

    const occluderCanvas = document.createElement('canvas');
    occluderCanvas.width = width;
    occluderCanvas.height = height;
    const occluderContext = occluderCanvas.getContext('2d', { willReadFrequently: true });
    const drawnOccluderIds = [];
    for (const occluder of occluders) {
        try {
            if (await drawLayerMaskInScene(occluderContext, occluder, width, height)) {
                drawnOccluderIds.push(occluder.layerId);
            }
        } catch (error) {
            console.warn('[Object Completion] foreground exclusion mask unavailable:', occluder.layerId, error);
        }
    }
    if (!drawnOccluderIds.length) {
        releaseCanvas(candidateCanvas);
        releaseCanvas(occluderCanvas);
        releaseImage(candidateImage);
        return { status: 'unavailable', checkedOccluderIds: [] };
    }

    const candidatePixels = candidateContext.getImageData(0, 0, width, height).data;
    const occluderPixels = occluderContext.getImageData(0, 0, width, height).data;
    let candidatePixelsCount = 0;
    let overlapPixels = 0;
    for (let index = 0; index < candidatePixels.length; index += 4) {
        if (candidatePixels[index + 3] < 48) continue;
        candidatePixelsCount += 1;
        if (occluderPixels[index + 3] >= 96) overlapPixels += 1;
    }
    const overlapRatio = overlapPixels / Math.max(1, candidatePixelsCount);
    const result = {
        // A completed background object is expected to occupy the original
        // foreground occluder's coordinates. This overlap is scene evidence,
        // not leakage evidence: rejecting it would reject every successful
        // table-behind-stools completion.
        status: 'observed',
        checkedOccluderIds: drawnOccluderIds,
        candidatePixels: candidatePixelsCount,
        overlapPixels,
        overlapRatio: Number(overlapRatio.toFixed(5)),
        interpretation: 'expected_completed_target_behind_foreground_occluder'
    };
    releaseCanvas(candidateCanvas);
    releaseCanvas(occluderCanvas);
    releaseImage(candidateImage);
    return result;
}

async function auditCanonicalSegmentationAgainstObservation({
    segmented,
    asset,
    targetLayer,
    graph,
    sourceSize
}) {
    const task = (graph?.completionTasks || []).find(candidate => candidate.id === asset.taskId);
    const observedUrl = asset?.observedCutout?.cutoutUrl || targetLayer?.mask?.cutoutUrl;
    const observedBbox = asset?.observedCutout?.bbox || targetLayer?.bbox;
    if (!task || !observedUrl || !Array.isArray(observedBbox) || !segmented?.dataUrl || !Array.isArray(segmented?.bbox)) {
        return {
            status: 'unavailable',
            reason: 'original_scene_observation_mask_unavailable'
        };
    }

    const width = Math.max(1, Number(sourceSize?.width) || 1);
    const height = Math.max(1, Number(sourceSize?.height) || 1);
    const referenceCanvas = document.createElement('canvas');
    referenceCanvas.width = width;
    referenceCanvas.height = height;
    const candidateCanvas = document.createElement('canvas');
    candidateCanvas.width = width;
    candidateCanvas.height = height;
    const occluderCanvas = document.createElement('canvas');
    occluderCanvas.width = width;
    occluderCanvas.height = height;

    const referenceDrawn = await drawLayerMaskInScene(referenceCanvas.getContext('2d'), {
        bbox: observedBbox,
        mask: { cutoutUrl: observedUrl }
    }, width, height);
    const candidateDrawn = await drawLayerMaskInScene(candidateCanvas.getContext('2d'), {
        bbox: segmented.bbox,
        mask: { cutoutUrl: segmented.dataUrl }
    }, width, height);
    if (!referenceDrawn || !candidateDrawn) {
        releaseCanvas(referenceCanvas);
        releaseCanvas(candidateCanvas);
        releaseCanvas(occluderCanvas);
        return { status: 'unavailable', reason: 'original_or_completed_target_mask_unavailable' };
    }

    const layersById = new Map((graph?.layers || []).map(layer => [layer.layerId, layer]));
    const resolvedTask = resolveCompletionContext(graph, task);
    const checkedOccluderIds = [];
    const occluderContext = occluderCanvas.getContext('2d');
    for (const layerId of resolvedTask?.occluderLayerIds || []) {
        const occluder = layersById.get(layerId);
        if (!occluder) continue;
        try {
            if (await drawLayerMaskInScene(occluderContext, occluder, width, height)) {
                checkedOccluderIds.push(layerId);
            }
        } catch (error) {
            console.warn('[Object Completion] observation audit occluder mask unavailable:', layerId, error);
        }
    }

    const referencePixels = referenceCanvas.getContext('2d', { willReadFrequently: true })
        .getImageData(0, 0, width, height).data;
    const candidatePixels = candidateCanvas.getContext('2d', { willReadFrequently: true })
        .getImageData(0, 0, width, height).data;
    const occluderPixels = occluderContext.getImageData(0, 0, width, height).data;
    let referenceVisiblePixels = 0;
    let candidateVisiblePixels = 0;
    let intersectionPixels = 0;
    let unionPixels = 0;
    let candidateDriftPixels = 0;
    let generatedOccludedPixels = 0;

    for (let index = 0; index < referencePixels.length; index += 4) {
        const inOccluder = occluderPixels[index + 3] >= 96;
        const inReference = referencePixels[index + 3] >= 48;
        const inCandidate = candidatePixels[index + 3] >= 48;
        if (inOccluder) {
            if (inCandidate) generatedOccludedPixels += 1;
            continue;
        }
        if (inReference) referenceVisiblePixels += 1;
        if (inCandidate) candidateVisiblePixels += 1;
        if (inReference && inCandidate) intersectionPixels += 1;
        if (inReference || inCandidate) unionPixels += 1;
        if (inCandidate && !inReference) candidateDriftPixels += 1;
    }

    const observedRecall = intersectionPixels / Math.max(1, referenceVisiblePixels);
    const visibleIoU = intersectionPixels / Math.max(1, unionPixels);
    const unoccludedDriftRatio = candidateDriftPixels / Math.max(1, candidateVisiblePixels);
    const metrics = {
        checkedOccluderIds,
        referenceVisiblePixels,
        candidateVisiblePixels,
        intersectionPixels,
        generatedOccludedPixels,
        observedRecall: Number(observedRecall.toFixed(4)),
        visibleIoU: Number(visibleIoU.toFixed(4)),
        unoccludedDriftRatio: Number(unoccludedDriftRatio.toFixed(4)),
        minimumObservedRecall: 0.78,
        minimumVisibleIoU: 0.62,
        maximumUnoccludedDriftRatio: 0.18
    };
    const accepted = observedRecall >= 0.78 && visibleIoU >= 0.62 && unoccludedDriftRatio <= 0.18;
    const result = {
        status: accepted ? 'passed' : 'rejected',
        reason: accepted
            ? 'original_visible_silhouette_preserved_outside_occluders'
            : 'completed_target_drifted_from_original_visible_silhouette',
        ...metrics
    };
    releaseCanvas(referenceCanvas);
    releaseCanvas(candidateCanvas);
    releaseCanvas(occluderCanvas);
    return result;
}

function parseAspectRatio(value) {
    const [width, height] = String(value || '1:1').split(':').map(Number);
    return width > 0 && height > 0 ? width / height : 1;
}

async function materializeSceneEditCanvas(
    sceneDataUrl,
    maskDataUrl,
    occluderMaskDataUrl,
    sourceWidth,
    sourceHeight,
    imageModel
) {
    const shouldPadForFixedAspect = imageModel !== 'gpt-image-2';
    const aspectRatio = shouldPadForFixedAspect
        ? getClosestSupportedAspectRatio(sourceWidth, sourceHeight)
        : null;
    const targetRatio = parseAspectRatio(aspectRatio);
    const canvasWidth = shouldPadForFixedAspect
        ? Math.max(sourceWidth, Math.ceil(sourceHeight * targetRatio))
        : sourceWidth;
    const canvasHeight = shouldPadForFixedAspect
        ? Math.max(sourceHeight, Math.ceil(sourceWidth / targetRatio))
        : sourceHeight;
    const sourceX = Math.floor((canvasWidth - sourceWidth) / 2);
    const sourceY = Math.floor((canvasHeight - sourceHeight) / 2);
    const [sceneImage, maskImage, occluderMaskImage] = await Promise.all([
        loadImage(sceneDataUrl),
        loadImage(maskDataUrl),
        occluderMaskDataUrl ? loadImage(occluderMaskDataUrl) : Promise.resolve(null)
    ]);

    const sceneCanvas = document.createElement('canvas');
    sceneCanvas.width = canvasWidth;
    sceneCanvas.height = canvasHeight;
    const sceneContext = sceneCanvas.getContext('2d');
    sceneContext.fillStyle = '#ffffff';
    sceneContext.fillRect(0, 0, canvasWidth, canvasHeight);
    sceneContext.drawImage(sceneImage, sourceX, sourceY, sourceWidth, sourceHeight);

    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = canvasWidth;
    maskCanvas.height = canvasHeight;
    maskCanvas.getContext('2d').drawImage(maskImage, sourceX, sourceY, sourceWidth, sourceHeight);

    const occluderMaskCanvas = document.createElement('canvas');
    occluderMaskCanvas.width = canvasWidth;
    occluderMaskCanvas.height = canvasHeight;
    if (occluderMaskImage) {
        occluderMaskCanvas.getContext('2d').drawImage(
            occluderMaskImage,
            sourceX,
            sourceY,
            sourceWidth,
            sourceHeight
        );
    }

    const result = {
        sceneDataUrl: sceneCanvas.toDataURL('image/png'),
        maskDataUrl: maskCanvas.toDataURL('image/png'),
        occluderMaskDataUrl: occluderMaskCanvas.toDataURL('image/png'),
        forcedAspectRatio: aspectRatio,
        sourceRect: { x: sourceX, y: sourceY, width: sourceWidth, height: sourceHeight },
        canvasSize: { width: canvasWidth, height: canvasHeight },
        padded: shouldPadForFixedAspect && (canvasWidth !== sourceWidth || canvasHeight !== sourceHeight)
    };
    releaseCanvas(occluderMaskCanvas);
    releaseImage(occluderMaskImage);
    releaseImage(maskImage);
    releaseImage(sceneImage);
    releaseCanvas(sceneCanvas);
    releaseCanvas(maskCanvas);
    return result;
}

async function restoreMaskedSceneToSource(generatedDataUrl, editCanvas) {
    const [generatedImage, sourceImage, maskImage] = await Promise.all([
        loadImage(generatedDataUrl),
        loadImage(editCanvas.sceneDataUrl),
        loadImage(editCanvas.maskDataUrl)
    ]);
    const canvas = document.createElement('canvas');
    canvas.width = editCanvas.canvasSize.width;
    canvas.height = editCanvas.canvasSize.height;
    const context = canvas.getContext('2d');
    context.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

    const editedLayer = document.createElement('canvas');
    editedLayer.width = canvas.width;
    editedLayer.height = canvas.height;
    const editedContext = editedLayer.getContext('2d');
    editedContext.imageSmoothingEnabled = true;
    editedContext.imageSmoothingQuality = 'high';
    editedContext.drawImage(generatedImage, 0, 0, canvas.width, canvas.height);
    editedContext.globalCompositeOperation = 'destination-in';
    editedContext.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
    context.drawImage(editedLayer, 0, 0);

    const { x, y, width, height } = editCanvas.sourceRect;
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = width;
    sourceCanvas.height = height;
    sourceCanvas.getContext('2d').drawImage(canvas, x, y, width, height, 0, 0, width, height);
    const dataUrl = sourceCanvas.toDataURL('image/png');
    const result = {
        dataUrl,
        generatedWidth: Math.max(1, generatedImage.naturalWidth || generatedImage.width),
        generatedHeight: Math.max(1, generatedImage.naturalHeight || generatedImage.height),
        padded: editCanvas.padded,
        editCanvasSize: `${canvas.width}x${canvas.height}`
    };
    releaseImage(generatedImage);
    releaseImage(sourceImage);
    releaseImage(maskImage);
    releaseCanvas(canvas);
    releaseCanvas(editedLayer);
    releaseCanvas(sourceCanvas);
    return result;
}

async function auditSceneInpaintEdit(generatedDataUrl, editCanvas) {
    if (!generatedDataUrl || !editCanvas?.sceneDataUrl || !editCanvas?.maskDataUrl) {
        return { status: 'unavailable', reason: 'scene_inpaint_audit_inputs_missing' };
    }
    const [generatedImage, sourceImage, maskImage, occluderMaskImage] = await Promise.all([
        loadImage(generatedDataUrl),
        loadImage(editCanvas.sceneDataUrl),
        loadImage(editCanvas.maskDataUrl),
        editCanvas.occluderMaskDataUrl ? loadImage(editCanvas.occluderMaskDataUrl) : Promise.resolve(null)
    ]);
    const width = Math.max(1, Number(editCanvas.canvasSize?.width) || sourceImage.naturalWidth || sourceImage.width);
    const height = Math.max(1, Number(editCanvas.canvasSize?.height) || sourceImage.naturalHeight || sourceImage.height);
    const sourceCanvas = document.createElement('canvas');
    const generatedCanvas = document.createElement('canvas');
    const maskCanvas = document.createElement('canvas');
    const occluderMaskCanvas = document.createElement('canvas');
    sourceCanvas.width = generatedCanvas.width = maskCanvas.width = occluderMaskCanvas.width = width;
    sourceCanvas.height = generatedCanvas.height = maskCanvas.height = occluderMaskCanvas.height = height;
    const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
    const generatedContext = generatedCanvas.getContext('2d', { willReadFrequently: true });
    const maskContext = maskCanvas.getContext('2d', { willReadFrequently: true });
    const occluderMaskContext = occluderMaskCanvas.getContext('2d', { willReadFrequently: true });
    sourceContext.drawImage(sourceImage, 0, 0, width, height);
    generatedContext.drawImage(generatedImage, 0, 0, width, height);
    maskContext.drawImage(maskImage, 0, 0, width, height);
    if (occluderMaskImage) occluderMaskContext.drawImage(occluderMaskImage, 0, 0, width, height);
    const sourcePixels = sourceContext.getImageData(0, 0, width, height).data;
    const generatedPixels = generatedContext.getImageData(0, 0, width, height).data;
    const maskPixels = maskContext.getImageData(0, 0, width, height).data;
    const occluderMaskPixels = occluderMaskContext.getImageData(0, 0, width, height).data;
    let maskedPixels = 0;
    let changedPixels = 0;
    let totalDifference = 0;
    let occluderPixels = 0;
    let occluderChangedPixels = 0;
    let occluderDifference = 0;
    for (let offset = 0; offset < maskPixels.length; offset += 4) {
        const difference =
            Math.abs(sourcePixels[offset] - generatedPixels[offset]) +
            Math.abs(sourcePixels[offset + 1] - generatedPixels[offset + 1]) +
            Math.abs(sourcePixels[offset + 2] - generatedPixels[offset + 2]);
        if (occluderMaskPixels[offset + 3] >= 32) {
            occluderPixels += 1;
            occluderDifference += difference;
            if (difference >= 36) occluderChangedPixels += 1;
        }
        if (maskPixels[offset + 3] < 32) continue;
        maskedPixels += 1;
        totalDifference += difference;
        if (difference >= 36) changedPixels += 1;
    }
    const changedRatio = changedPixels / Math.max(1, maskedPixels);
    const unchangedRatio = 1 - changedRatio;
    const meanDifference = totalDifference / Math.max(1, maskedPixels * 3 * 255);
    const occluderChangedRatio = occluderChangedPixels / Math.max(1, occluderPixels);
    const occluderMeanDifference = occluderDifference / Math.max(1, occluderPixels * 3 * 255);
    const noOp = maskedPixels === 0 || changedRatio < 0.15 || meanDifference < 0.04;
    // A response can change enough of the mask to pass the old global no-op
    // test while leaving a large, recognizable stool region untouched. Treat
    // that as a separate residual state so the completion retry is used.
    const occluderResidual = Boolean(
        !noOp && (
            (unchangedRatio >= 0.45 && changedRatio < 0.55) ||
            (occluderPixels >= 64 && (
                occluderChangedRatio < 0.28 ||
                occluderMeanDifference < 0.08
            ))
        )
    );
    const failureReason = noOp
        ? 'empty_or_missing_cutout'
        : occluderResidual
        ? 'occluder_residual'
        : null;
    const retryStrategy = failureReason === 'occluder_residual'
        ? 'expand_verified_occluder_edit_instruction'
        : failureReason === 'empty_or_missing_cutout'
        ? 'repeat_same_bounded_edit_once'
        : null;
    const result = {
        status: noOp ? 'unchanged' : (occluderResidual ? 'occluder_residual' : 'changed'),
        failureReason,
        retryStrategy,
        maskedPixels,
        changedPixels,
        changedRatio: Number(changedRatio.toFixed(5)),
        unchangedRatio: Number(unchangedRatio.toFixed(5)),
        meanDifference: Number(meanDifference.toFixed(5)),
        occluderPixels,
        occluderChangedPixels,
        occluderChangedRatio: Number(occluderChangedRatio.toFixed(5)),
        occluderMeanDifference: Number(occluderMeanDifference.toFixed(5)),
        minimumChangedRatio: 0.15,
        minimumMeanDifference: 0.04,
        residualUnchangedRatioThreshold: 0.45
    };
    releaseCanvas(sourceCanvas);
    releaseCanvas(generatedCanvas);
    releaseCanvas(maskCanvas);
    releaseCanvas(occluderMaskCanvas);
    releaseImage(generatedImage);
    releaseImage(sourceImage);
    releaseImage(maskImage);
    releaseImage(occluderMaskImage);
    return result;
}

function buildSceneInpaintPrompt(targetLayer, drawnOccluderIds) {
    const name = targetLayer?.name || targetLayer?.semanticType || '目标物体';
    return [
        '执行严格的原场景局部修补，不要生成透明背景或独立商品图。',
        `目标物体是“${name}”。白色 mask 对应 API 的透明可编辑区域，仅表示前景遮挡区域。`,
        '只在白色 mask 内延续目标物体被遮挡的结构、材质、纹理、透视、光影和边缘，使它在遮挡物后方成为完整物体。',
        '白色 mask 内的凳子、椅子及其座面、凳腿、阴影和残片必须全部删除，不得把前景遮挡物原样保留或重新绘制；输出中不能再出现白色圆柱凳、条纹凳或其轮廓。',
        '目标物体必须覆盖原遮挡物所在位置，并与 mask 外可见目标边缘连续接续；这张结果会直接用于后续完整物体分割。',
        '保留目标物体所有未遮挡的可见像素。白色 mask 外的原场景必须保持不变。',
        '不要移动、缩放、重绘或删除任何未遮挡物体；不要新增第二个主体；不要改变整体构图。',
        `本次遮挡层标识：${drawnOccluderIds.join(', ') || '已提供的前景遮挡区域'}。`
    ].join('\n');
}

function flattenPostCompletionSemanticCandidates(layers = [], parent = null, result = []) {
    if (!Array.isArray(layers)) return result;
    layers.forEach(layer => {
        if (!layer || typeof layer !== 'object') return;
        result.push({ ...layer, parentLayerId: layer.parentLayerId || parent?.id || null });
        if (Array.isArray(layer.children)) {
            flattenPostCompletionSemanticCandidates(layer.children, layer, result);
        }
    });
    return result;
}

function bboxIntersectionArea(first, second) {
    if (!isNormalizedBbox(first) || !isNormalizedBbox(second)) return 0;
    const [ay1, ax1, ay2, ax2] = first.map(Number);
    const [by1, bx1, by2, bx2] = second.map(Number);
    return Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1)) *
        Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
}

function normalizeSceneSemanticBbox(bbox) {
    if (!isNormalizedBbox(bbox)) return null;
    const values = bbox.map(value => Math.max(0, Math.min(1000, Number(value))));
    const [ymin, xmin, ymax, xmax] = values;
    if (ymax <= ymin || xmax <= xmin) return null;
    return values.map(value => Number(value.toFixed(3)));
}

function isPostCompletionSemanticCandidateSelectable(candidate) {
    if (!candidate || typeof candidate !== 'object') return false;

    const groupMarkers = [
        candidate.compositeRole,
        candidate.semanticType,
        candidate.runtimeType,
        candidate.renderMode,
        candidate.semanticGroup
    ].map(value => String(value || '').toLowerCase());
    const isComposite = groupMarkers.includes('composite_group') ||
        groupMarkers.includes('semantic_group') ||
        groupMarkers.includes('group') ||
        (Array.isArray(candidate.children) && candidate.children.length > 0) ||
        (Array.isArray(candidate.childLayerIds) && candidate.childLayerIds.length > 0);
    if (isComposite) return false;

    const editable = String(candidate.editable).toLowerCase();
    const locked = String(candidate.locked).toLowerCase();
    if (candidate.editable === false || editable === 'false') return false;
    if (candidate.locked === true || locked === 'true') return false;
    if (String(candidate.layerType || '').toLowerCase() === 'background_plate') return false;
    if (String(candidate.category || '').toLowerCase() === 'background') return false;

    const renderMode = String(candidate.renderMode || '').toLowerCase();
    const semanticType = String(candidate.semanticType || '').toLowerCase();
    const designRole = String(candidate.designRole || '').toLowerCase();
    if (['background_plate', 'deferred', 'semantic_group', 'text_css', 'vector_shape'].includes(renderMode)) {
        return false;
    }
    if (['ad_background', 'shape_panel', 'price_badge', 'cta_button', 'logo_mark', 'flat_ad_layout', 'element_text'].includes(semanticType)) {
        return false;
    }
    if (['base_background', 'local_panel', 'price_badge', 'headline_text', 'body_text', 'label_text', 'price_text', 'logo_text', 'url_text'].includes(designRole)) {
        return false;
    }

    return true;
}

function normalizedBboxGapDistance(first, second) {
    if (!isNormalizedBbox(first) || !isNormalizedBbox(second)) return Infinity;
    const [ay1, ax1, ay2, ax2] = first.map(Number);
    const [by1, bx1, by2, bx2] = second.map(Number);
    const horizontalGap = Math.max(0, Math.max(ax1, bx1) - Math.min(ax2, bx2));
    const verticalGap = Math.max(0, Math.max(ay1, by1) - Math.min(ay2, by2));
    return Math.sqrt(horizontalGap ** 2 + verticalGap ** 2);
}

function postCompletionSemanticCompatibility(targetLayer, candidate) {
    const targetText = [
        targetLayer?.semanticType,
        targetLayer?.name,
        targetLayer?.extractionProfile,
        targetLayer?.designRole
    ].join(' ').toLowerCase();
    const candidateText = [
        candidate?.semanticType,
        candidate?.name,
        candidate?.promptHint,
        candidate?.reasoning
    ].join(' ').toLowerCase();
    const targetFurniture = /table|desk|console|sideboard|cabinet|counter|桌|案|几|台|柜|家具/.test(targetText);
    const candidateFurniture = /table|desk|console|sideboard|cabinet|counter|桌|案|几|台|柜|家具/.test(candidateText);
    const obviousOther = /stool|chair|seat|椅|凳|座椅|vase|flower|plant|花瓶|花|person|人|wall|floor|墙|地面/.test(candidateText);
    if (targetFurniture && obviousOther && !candidateFurniture) return false;
    if (targetFurniture && candidateFurniture) return true;
    return !['element_text', 'ad_background', 'surface_wall', 'surface_floor', 'surface_ceiling'].includes(
        String(candidate?.semanticType || '').toLowerCase()
    );
}

async function runPostCompletionSemanticAnalysis(completedDataUrl, targetLayer, observedBbox) {
    if (!completedDataUrl) return null;
    const startedAt = Date.now();
    console.info('[Object Completion] post-completion semantic start', {
        targetLayerId: targetLayer?.layerId || targetLayer?.id || null,
        source: 'completed_original_scene'
    });
    try {
        const file = await dataURLToFile(
            completedDataUrl,
            `completion-semantic-${targetLayer?.layerId || 'target'}-${Date.now()}.png`
        );
        const analysis = await analyzeImageLayers(file);
        const candidates = flattenPostCompletionSemanticCandidates(analysis?.rawLayers || []);
        const referenceBbox = normalizeSceneSemanticBbox(targetLayer?.bbox) || normalizeSceneSemanticBbox(observedBbox);
        const observedArea = normalizedBboxArea(referenceBbox);
        const scored = candidates
            .filter(isPostCompletionSemanticCandidateSelectable)
            .filter(candidate => normalizeSceneSemanticBbox(candidate?.bbox))
            .map(candidate => {
                const bbox = normalizeSceneSemanticBbox(candidate.bbox);
                const area = normalizedBboxArea(bbox);
                const overlap = bboxIntersectionArea(bbox, referenceBbox);
                const observedRecall = overlap / Math.max(1, observedArea);
                const bboxDistance = normalizedBboxGapDistance(bbox, referenceBbox);
                const canvasCoverage = area / 1000000;
                const compatible = postCompletionSemanticCompatibility(targetLayer, candidate);
                const reasonableExtent = canvasCoverage > 0.0005 && canvasCoverage < 0.92;
                const score = (compatible ? 0.55 : 0) +
                    Math.min(0.3, observedRecall * 0.3) +
                    (reasonableExtent ? 0.15 : -0.4);
                return {
                    candidate,
                    bbox,
                    area,
                    overlap,
                    observedRecall: Number(observedRecall.toFixed(4)),
                    bboxDistance: Number(bboxDistance.toFixed(4)),
                    canvasCoverage: Number(canvasCoverage.toFixed(4)),
                    compatible,
                    reasonableExtent,
                    score: Number(score.toFixed(4))
                };
            })
            .filter(entry => entry.compatible && entry.reasonableExtent)
            .sort((a, b) => {
                const aOverlaps = a.overlap > 0;
                const bOverlaps = b.overlap > 0;
                if (aOverlaps !== bOverlaps) return aOverlaps ? -1 : 1;
                if (aOverlaps && a.observedRecall !== b.observedRecall) {
                    return b.observedRecall - a.observedRecall;
                }
                if (a.bboxDistance !== b.bboxDistance) return a.bboxDistance - b.bboxDistance;
                return b.score - a.score;
            });
        const selected = scored[0] || null;
        const snapshot = selected ? {
            source: 'post_completion_semantic_original_scene',
            coordinateSpace: 'normalized_scene_1000',
            candidateCount: candidates.length,
            selectedCandidateCount: scored.length,
            semanticType: selected.candidate.semanticType || 'other',
            name: selected.candidate.name || targetLayer?.name || 'completed target',
            bbox: selected.bbox,
            confidence: selected.score,
            observedRecall: selected.observedRecall,
            canvasCoverage: selected.canvasCoverage,
            matchedOriginal: selected.compatible,
            analyzedAt: Date.now(),
            durationMs: Date.now() - startedAt
        } : null;
        console.info('[Object Completion] post-completion semantic done', {
            targetLayerId: targetLayer?.layerId || targetLayer?.id || null,
            candidateCount: candidates.length,
            selectedCandidateCount: scored.length,
            selected: snapshot
        });
        return snapshot;
    } catch (error) {
        console.warn('[Object Completion] post-completion semantic unavailable', {
            targetLayerId: targetLayer?.layerId || targetLayer?.id || null,
            error: error?.message || String(error),
            durationMs: Date.now() - startedAt
        });
        return null;
    }
}

function buildFailureAwareRetryInstruction(targetLayer, drawnOccluderIds, audit) {
    const targetName = targetLayer?.name || targetLayer?.semanticType || '目标物体';
    const occluderNames = (drawnOccluderIds || []).join(', ') || '已验证的前景遮挡区域';
    if (audit?.failureReason === 'occluder_residual') {
        return [
            '这是同一张原场景的受限重试，只允许修改已验证的前景遮挡区域。',
            `目标“${targetName}”必须延续到遮挡物后方，并完整覆盖该区域中原目标的隐藏结构。`,
            `必须移除这些已验证遮挡物的全部像素、轮廓、阴影和残片：${occluderNames}。`,
            '禁止把遮挡物重新绘制，也禁止改变白色 mask 外的任何像素。'
        ].join('\n');
    }
    return [
        '上一次编辑没有对受限编辑区域产生有效变化。',
        `请在同一张原场景中补全“${targetName}”的隐藏部分。`,
        '仅修改透明可编辑区域，保留所有未遮挡目标像素和其余场景像素。',
        '不要生成独立商品图、透明背景或新的场景。'
    ].join('\n');
}

function recordCanonicalCompletionSegmentationQuality(segmented, targetLayer) {
    const quality = segmented?.quality || {};
    const descriptor = [
        targetLayer?.name,
        targetLayer?.semanticType,
        targetLayer?.extractionProfile
    ].join(' ').toLowerCase();
    const isTableLike = /table|desk|console|sideboard|桌|案|几|台/.test(descriptor);
    const targetFillRatio = Number(quality.targetFillRatio);
    if (isTableLike && Number.isFinite(targetFillRatio) && targetFillRatio < 0.18) {
        // A table's semantic box includes the open area between its supports,
        // so its alpha fill is not a valid standalone failure signal. The
        // completion SAM service has already applied its L-model candidate,
        // containment, observation, and quality gates before this point.
        // Keep this diagnostic without converting an accepted second-SAM
        // result into an automatic-completion failure.
        console.info('[Object Completion] low-fill table accepted by second SAM-L quality gate', {
            targetLayerId: targetLayer?.layerId || targetLayer?.id || null,
            targetFillRatio,
            samModelVariant: quality.samModelVariant || null,
            selectionRoute: quality.samSelectionRoute || null
        });
    }
}

function findAsset(item, assetId) {
    return item?.semanticViews?.completionAssets?.find(candidate => candidate.id === assetId) || null;
}

function appendAttempt(asset, attempt) {
    const attempts = Array.isArray(asset.generation?.attempts) ? asset.generation.attempts : [];
    asset.generation = {
        ...(asset.generation || {}),
        attempts: [...attempts.slice(-4), attempt]
    };
}

// Magic Layers calls this automatically for verified candidates. It only
// produces canonicalAsset pixels; scene replacement is performed by the
// layer-manager after this function returns successfully.
export async function executeObjectCompletion(item, assetId, options = {}) {
    if (typeof document === 'undefined' || typeof Image === 'undefined') {
        throw new Error('object_completion_requires_browser');
    }
    const asset = findAsset(item, assetId);
    if (!asset) throw new Error('completion_asset_not_found');
    if (asset.status !== 'ready_for_completion') {
        throw new Error(`completion_asset_not_ready:${asset.status || 'unknown'}`);
    }

    const graph = item.semanticViews?.layerGraph;
    const targetLayer = graph?.layers?.find(layer => layer.layerId === asset.targetLayerId) || null;
    const onProgress = typeof options.onProgress === 'function' ? options.onProgress : () => {};
    const attemptId = `completion_attempt_${Date.now()}`;
    const attempt = { id: attemptId, startedAt: Date.now(), status: 'started' };
    appendAttempt(asset, attempt);
    asset.status = 'completing';
    asset.generation = { ...(asset.generation || {}), state: 'materializing', lastError: null };

    try {
        const preflight = asset.preflight || prepareCompletionAssetPreflight(item, assetId);
        if (!preflight) throw new Error('completion_preflight_unavailable');

        if (!targetLayer) throw new Error('completion_target_layer_unavailable');

        console.info('[Object Completion] execution started', {
            targetLayerId: asset.targetLayerId,
            taskId: asset.taskId,
            completionInputMode: 'original_scene'
        });
        onProgress('正在准备原场景遮挡补全输入...');
        const materialized = await materializeSceneCompletionInput(item, asset, preflight, targetLayer);
        const sourceWidth = materialized.sourceSize.width;
        const sourceHeight = materialized.sourceSize.height;
        const imageModel = getImageModel();
        const completionInput = materialized;
        const editCanvas = await materializeSceneEditCanvas(
            completionInput.sceneDataUrl,
            completionInput.editMaskDataUrl,
            completionInput.occluderMaskDataUrl,
            sourceWidth,
            sourceHeight,
            imageModel
        );
        // The edit canvas now owns its encoded copies; release the original
        // scene/mask strings before the model request starts.
        materialized.sceneDataUrl = null;
        materialized.editMaskDataUrl = null;
        materialized.occluderMaskDataUrl = null;
        completionInput.sceneDataUrl = null;
        completionInput.editMaskDataUrl = null;
        completionInput.occluderMaskDataUrl = null;
        let sceneFile = await dataURLToFile(
            editCanvas.sceneDataUrl,
            `completion-scene-${asset.targetLayerId}-${Date.now()}.png`
        );
        asset.generation = { ...(asset.generation || {}), state: 'generating' };
        onProgress('正在在原场景中补全被遮挡区域...');
        const sceneInpaintPrompt = buildSceneInpaintPrompt(
            targetLayer,
            materialized.drawnOccluderIds
        );
        const completionPrompt = sceneInpaintPrompt;
        let generation = await editOrQueryImageWithGemini(
            completionPrompt,
            sceneFile,
            [],
            editCanvas.maskDataUrl,
            editCanvas.forcedAspectRatio,
            false,
            {
                imageCount: 1,
                // Keep this option scoped to scene completion. Normal GPT image
                // edits continue using their provider-safe size normalization.
                completionSceneInpaint: true
            }
        );
        if (!generation?.success) throw new Error('completion_scene_inpaint_failed');
        let generatedDataUrl = getFirstGeneratedImage(generation);
        if (!generatedDataUrl) throw new Error('completion_scene_inpaint_returned_no_image');

        // A successful HTTP response can still be a no-op when the provider
        // receives an opaque/misaligned mask. Detect that only for this scene
        // completion request and spend one retry on a hard removal instruction
        // before the result reaches the second SAM pass.
        let inpaintAudit = null;
        try {
            inpaintAudit = await auditSceneInpaintEdit(generatedDataUrl, editCanvas);
        } catch (error) {
            console.warn('[Object Completion] scene inpaint audit unavailable:', error);
        }
        if (inpaintAudit?.status === 'unchanged' || inpaintAudit?.status === 'occluder_residual') {
            console.warn('[Object Completion] scene inpaint retained an occluder region; retrying once', {
                targetLayerId: asset.targetLayerId,
                inpaintAudit
            });
            const retryGeneration = await editOrQueryImageWithGemini(
                [
                    completionPrompt,
                    buildFailureAwareRetryInstruction(
                        targetLayer,
                        materialized.drawnOccluderIds,
                        inpaintAudit
                    )
                ].join('\n'),
                sceneFile,
                [],
                editCanvas.maskDataUrl,
                editCanvas.forcedAspectRatio,
                false,
                { imageCount: 1, completionSceneInpaint: true }
            );
            const retryDataUrl = getFirstGeneratedImage(retryGeneration);
            if (retryGeneration?.success && retryDataUrl) {
                const retryAudit = await auditSceneInpaintEdit(retryDataUrl, editCanvas).catch(() => null);
                console.info('[Object Completion] scene inpaint retry audit', {
                    targetLayerId: asset.targetLayerId,
                    firstAudit: inpaintAudit,
                    retryAudit
                });
                generation = retryGeneration;
                generatedDataUrl = retryDataUrl;
                inpaintAudit = retryAudit || inpaintAudit;
            }
        }
        console.info('[Object Completion] scene inpaint edit audit', {
            targetLayerId: asset.targetLayerId,
            completionInputMode: 'original_scene',
            audit: inpaintAudit
        });
        if (inpaintAudit?.status === 'occluder_residual') {
            throw new Error('completion_scene_inpaint_occluder_residual');
        }
        // Keep only the normalized generated URL. The response object can also
        // contain the same image as raw base64 or a raw provider payload.
        if (generation && typeof generation === 'object') {
            generation.imageData = null;
            generation.raw = null;
        }

        const completedScene = await restoreMaskedSceneToSource(generatedDataUrl, editCanvas);
        generatedDataUrl = null;
        sceneFile = null;
        editCanvas.sceneDataUrl = null;
        editCanvas.maskDataUrl = null;
        const segmentationSceneDataUrl = completedScene.dataUrl;
        // Re-describe the completed original scene. This semantic pass is
        // temporary evidence for the second SAM prompt; it never replaces
        // the original layer identity or graph entry.
        const completionSemanticSnapshot = await runPostCompletionSemanticAnalysis(
            segmentationSceneDataUrl,
            targetLayer,
            asset?.observedCutout?.bbox || targetLayer?.bbox
        );
        asset.generation = {
            ...(asset.generation || {}),
            completionSemanticSnapshot: completionSemanticSnapshot || null
        };
        // Use the observed alpha's actual scene bounds for the second SAM
        // prompt. The semantic layer bbox can be a full poster when upstream
        // labels are coarse; passing that bbox makes SAM choose the poster,
        // not the completed object. Include only touching foreground
        // occluders and a small transition margin.
        const completionTask = (graph?.completionTasks || [])
            .find(candidate => candidate.id === asset.taskId);
        const resolvedCompletionTask = resolveCompletionContext(graph, completionTask);
        const completionOccluderLayers = resolvedCompletionTask.occluderLayerIds
            .map(layerId => graph?.layers?.find(candidate => candidate.layerId === layerId))
            .filter(Boolean);
        const observedUrlForPrompt = asset?.observedCutout?.cutoutUrl || targetLayer?.mask?.cutoutUrl;
        const observedBboxForPrompt = asset?.observedCutout?.bbox || targetLayer?.bbox;
        pushCompletionDebugPreview(
            `首次SAM observed cutout - ${asset.targetLayerId}`,
            observedUrlForPrompt,
            {
                targetLayerId: asset.targetLayerId,
                bbox: observedBboxForPrompt,
                source: 'initial_sam_observed_cutout'
            }
        );
        const derivedPrompt = await deriveCompletionSamPromptBbox({
            observedUrl: observedUrlForPrompt,
            observedBbox: observedBboxForPrompt,
            occluderLayers: completionOccluderLayers,
            fallbackBbox: preflight.geometry.cropBbox || targetLayer.bbox,
            sourceSize: materialized.sourceSize
        });
        const samPromptBbox = cloneBbox(completionSemanticSnapshot?.bbox || derivedPrompt.bbox);
        if (!samPromptBbox) throw new Error('completion_scene_segmentation_prompt_bbox_missing');
        console.info('[Object Completion] scene inpaint generated', {
            targetLayerId: asset.targetLayerId,
            imageModel,
            sourceSize: `${sourceWidth}x${sourceHeight}`,
            generatedSize: `${completedScene.generatedWidth}x${completedScene.generatedHeight}`,
            editCanvasSize: completedScene.editCanvasSize,
            paddedForModel: completedScene.padded,
            completionInputMode: 'original_scene',
            unmaskedPixelsRestored: true,
            samPromptBbox,
            samPromptSource: derivedPrompt.source,
            observedAlphaBbox: derivedPrompt.observedAlphaBbox || null,
            completionSemanticBbox: completionSemanticSnapshot?.bbox || null,
            completionSemanticType: completionSemanticSnapshot?.semanticType || null,
            completionSemanticMatchedOriginal: completionSemanticSnapshot?.matchedOriginal ?? null
        });
        pushCompletionDebugPreview(
            `GPT补全场景 - ${asset.targetLayerId}`,
            completedScene.dataUrl,
            {
            targetLayerId: asset.targetLayerId,
            sourceSize: `${sourceWidth}x${sourceHeight}`,
                generatedSize: `${completedScene.generatedWidth}x${completedScene.generatedHeight}`,
            restoredSize: `${sourceWidth}x${sourceHeight}`,
            samPromptBbox,
            completionInputMode: 'original_scene',
            completionSemanticBbox: completionSemanticSnapshot?.bbox || null,
            completionSemanticType: completionSemanticSnapshot?.semanticType || null
        }
        );

        asset.generation = { ...(asset.generation || {}), state: 'resegmenting' };
        onProgress('正在从补全场景重新提取完整物体...');
        // The inpaint only removes the verified occluders.  Other extracted
        // foreground layers (vases, paintings, ornaments, etc.) remain in
        // the completed scene and must be supplied to the second SAM pass as
        // ownership context.  Previously this request contained just the
        // table and its stools, so SAM-L could legitimately absorb those
        // unrelated objects into the table mask.
        const completionContextIds = new Set([
            materialized.targetLayer.layerId || materialized.targetLayer.id,
            ...resolvedCompletionTask.occluderLayerIds
        ]);
        const isOverlappingCompletionPrompt = layer => {
            const bbox = layer?.bbox;
            if (!isNormalizedBbox(bbox)) return false;
            const [y1, x1, y2, x2] = bbox.map(Number);
            const [py1, px1, py2, px2] = samPromptBbox.map(Number);
            return Math.min(x2, px2) > Math.max(x1, px1) &&
                Math.min(y2, py2) > Math.max(y1, py1);
        };
        const completionSceneLayers = [
            ...(graph?.layers || []),
            ...(item?.semanticViews?.editableSceneLayers || [])
        ].reduce((layers, layer) => {
            const id = layer?.layerId || layer?.id;
            if (!id || layers.some(existing => (existing?.layerId || existing?.id) === id)) return layers;
            layers.push(layer);
            return layers;
        }, []);
        const overlappingForegroundContexts = completionSceneLayers
            .filter(layer => {
                const layerId = layer?.layerId || layer?.id;
                const contextDescriptor = [
                    layer?.name,
                    layer?.semanticType,
                    layer?.category,
                    layer?.runtimeType,
                    layer?.renderMode
                ].join(' ').toLowerCase();
                const backgroundLike = /background|wall|floor|ceiling|背景|墙面|地面|天花/.test(contextDescriptor);
                return layerId &&
                    !completionContextIds.has(layerId) &&
                    // The graph may carry a semantic layer before its cutout
                    // URL is copied into `mask`.  Its bbox/semantic group is
                    // still valid negative ownership evidence for SAM; do
                    // not silently drop it and fall back to the old 3-layer
                    // context.
                    // Composite/group layers can still carry the only usable
                    // alpha ownership mask for flowers, ornaments, or a
                    // combined foreground object.  Do not discard them just
                    // because their semantic group is not atomic.  The
                    // materializer below will count only layers with a real
                    // drawable mask as ownership evidence.
                    !['background', 'text'].includes(String(layer?.semanticGroup || '').toLowerCase()) &&
                    !backgroundLike &&
                    isOverlappingCompletionPrompt(layer);
            })
            .map(layer => ({ ...layer, completionForegroundContext: true }));
        const completionContextLayers = [
            materialized.targetLayer,
            ...resolvedCompletionTask.occluderLayerIds
                .filter(layerId => materialized.drawnOccluderIds.includes(layerId))
                .map(layerId => {
                    const layer = graph?.layers?.find(candidate => candidate.layerId === layerId);
                    return layer ? { ...layer, completionOccluder: true } : null;
                })
                .filter(Boolean),
            ...overlappingForegroundContexts
        ];
        const editableLayersById = new Map(
            (item?.semanticViews?.editableSceneLayers || [])
                .map(layer => [layer?.layerId || layer?.id, layer])
        );
        const foregroundMaskLayers = overlappingForegroundContexts.map(layer => {
            const id = layer?.layerId || layer?.id;
            return getLayerMaskSource(layer) ? layer : (editableLayersById.get(id) || layer);
        });
        const completionForegroundContextMask = await materializeCompletionForegroundContextMask(
            foregroundMaskLayers,
            materialized.targetLayer.layerId || materialized.targetLayer.id,
            resolvedCompletionTask.occluderLayerIds,
            samPromptBbox,
            sourceWidth,
            sourceHeight
        );
        // An empty ownership mask is valid only after we have inspected all
        // overlapping foreground candidates and confirmed that there are no
        // such layers.  If candidates exist but none yielded a real alpha
        // source, the status is unavailable and completion must be held.
        const foregroundOwnershipEvidenceStatus = completionForegroundContextMask
            ? 'attached'
            : (
                overlappingForegroundContexts.length === 0 &&
                // The verified occluder alpha is sufficient ownership
                // evidence for the only region where generated pixels may be
                // admitted. It is intentionally distinct from a foreground
                // context mask: pixels outside this occlusion zone remain
                // blocked by reconciliation.
                materialized.drawnOccluderIds.length > 0
                    ? 'empty_valid'
                    : 'unavailable'
            );
        let completionObservationMask = await materializeCompletionObservationMask(
            asset,
            targetLayer,
            sourceWidth,
            sourceHeight
        );
        let completionOcclusionMask = await materializeCompletionOcclusionMask(
            graph,
            completionTask,
            sourceWidth,
            sourceHeight
        );
        console.info('[Object Completion] second SAM context', {
            targetLayerId: asset.targetLayerId,
            completionInputMode: 'original_scene',
            excludedForegroundLayerIds: materialized.drawnOccluderIds,
            foregroundContextLayerIds: overlappingForegroundContexts.map(layer => layer.layerId || layer.id),
            contextLayerIds: completionContextLayers.map(layer => layer.layerId || layer.id),
            sourceSize: `${sourceWidth}x${sourceHeight}`,
            observationMaskAttached: Boolean(completionObservationMask),
            occlusionMaskAttached: Boolean(completionOcclusionMask),
            foregroundContextMaskAttached: Boolean(completionForegroundContextMask),
            foregroundOwnershipEvidenceStatus,
            ownershipMaskRequired: resolvedCompletionTask.occluderLayerIds.length > 0 || overlappingForegroundContexts.length > 0,
            foregroundContextCandidateCount: overlappingForegroundContexts.length,
            foregroundContextMaskSourceCount: foregroundMaskLayers.length,
            foregroundOwnershipEvidenceSource: completionForegroundContextMask
                ? 'foreground_context_alpha'
                : (materialized.drawnOccluderIds.length > 0
                    ? 'verified_occluder_alpha_only'
                    : 'none'),
            maskExpansionRadius: materialized.maskExpansionRadius
        });
        const segmented = await segmentSingleLayer({
            // Do not provide cleanPlateDataUrl here. The completed scene is the
            // only valid source for this segmentation pass.
            item: {
                dataUrl: segmentationSceneDataUrl,
                originalDataUrl: segmentationSceneDataUrl,
                segmentationSourceUrl: segmentationSceneDataUrl,
                // The segmentation service serializes these layers as backend
                // context. Only masks successfully drawn for this task are
                // sent as exclusion priors, avoiding guessed sibling boxes.
                scene: { layers: completionContextLayers }
            },
            layer: {
                ...materialized.targetLayer,
                id: `completion-scene-sam-${asset.targetLayerId}-${Date.now()}`,
                // Post-completion semantic is prompt evidence only. Preserve
                // the original layer identity while allowing SAM-L to use the
                // newly observed type and tight completed bbox.
                name: completionSemanticSnapshot?.name || materialized.targetLayer.name,
                semanticType: completionSemanticSnapshot?.semanticType || materialized.targetLayer.semanticType,
                completionSegmentation: true,
                completionInputMode: 'original_scene',
                // This marker is intentionally scoped to the post-inpaint
                // extraction pass. Initial extraction and every other
                // category retain their established B/L routing policy.
                samModelVariant: 'l',
                completionObservationMask,
                completionOcclusionMask,
                completionForegroundContextMask,
                foregroundOwnershipEvidenceStatus,
                bbox: samPromptBbox,
                completionSemanticSnapshot
            },
            onProgress,
            qualityProfile: 'canonical_completion'
        });
        completionObservationMask = null;
        completionOcclusionMask = null;
        completedScene.dataUrl = null;
        pushCompletionDebugPreview(
            `二次SAM原始cutout - ${asset.targetLayerId}`,
            segmented?.dataUrl,
            {
                targetLayerId: asset.targetLayerId,
                bbox: segmented?.bbox,
                size: segmented?.width && segmented?.height
                    ? `${segmented.width}x${segmented.height}`
                    : null,
                score: segmented?.quality?.score,
                fill: segmented?.quality?.targetFillRatio,
                selectedIndexes: segmented?.quality?.selectedIndexes,
                observationRecall: segmented?.quality?.debugCandidates?.find(candidate => candidate.selected)?.observationRecall,
                observationIoU: segmented?.quality?.debugCandidates?.find(candidate => candidate.selected)?.observationIoU,
                completionAnchor: segmented?.quality?.debugCandidates?.find(candidate => candidate.selected)?.completionAnchor,
                completionRecovery: segmented?.quality?.completionRecoveryAudit?.status || null,
                recoveredPixels: segmented?.quality?.completionRecoveryAudit?.allowedNewPixels || 0,
                hiddenRescuePixels: segmented?.quality?.completionHiddenRescuePixels || 0,
                hiddenCoverage: segmented?.quality?.completionRecoveryAudit?.hiddenCoverage || null,
                identityVerified: segmented?.quality?.completionIdentityVerified,
                structureVerified: segmented?.quality?.completionStructureVerified,
                candidatePreviewCount: segmented?.quality?.completionCandidatePreviews?.length || 0,
                completionInputMode: 'original_scene',
                qualityReason: segmented?.quality?.reason
            }
        );
        // Expose every raw SAM-L multimask candidate (including rejected
        // candidates) as an alpha-only crop.  This is diagnostic evidence;
        // it is never used as a replacement asset and lets us distinguish a
        // model miss from candidate arbitration/reconciliation loss.
        for (const preview of (segmented?.quality?.completionCandidatePreviews || [])) {
            pushCompletionDebugPreview(
                `二次SAM候选 #${preview?.index ?? '?' } - ${asset.targetLayerId}`,
                preview?.dataUrl,
                {
                    targetLayerId: asset.targetLayerId,
                    candidate: preview?.index,
                    selected: preview?.selected,
                    candidateAccepted: preview?.candidate,
                    rejectReason: preview?.rejectReason || 'none',
                    source: 'raw_second_sam_candidate_alpha'
                }
            );
        }
        const quality = segmented?.quality || {
            status: 'failed',
            runtimeAction: 'hold',
            shouldGenerateRuntimeLayer: false,
            issues: ['completion_scene_segmentation_missing_quality']
        };
        if (!segmented?.dataUrl || quality.runtimeAction === 'hold' || quality.shouldGenerateRuntimeLayer === false) {
            const reason = Array.isArray(quality.issues) ? quality.issues.join(',') : 'completion_scene_segmentation_quality_held';
            throw new Error(`completion_scene_segmentation_held:${reason}`);
        }
        recordCanonicalCompletionSegmentationQuality(segmented, targetLayer);
        const task = completionTask;
        const observedUrl = asset?.observedCutout?.cutoutUrl || targetLayer?.mask?.cutoutUrl;
        const observedBbox = asset?.observedCutout?.bbox || targetLayer?.bbox;
        const canUseDirectSecondSam = canUseDirectSecondSamCutout(
            segmented,
            foregroundOwnershipEvidenceStatus
        );
        let directSecondSamUsed = canUseDirectSecondSam;
        let reconciliation = null;
        let canonicalSegmentation = segmented;

        if (!canUseDirectSecondSam) {
            reconciliation = await reconcileCompletionSegmentation({
                observedUrl,
                observedBbox,
                segmented,
                completionStructureVerified: quality.completionStructureVerified === true,
                completionForegroundContextMask,
                foregroundOwnershipEvidenceStatus,
                graph,
                task,
                sourceSize: materialized.sourceSize
            });
            canonicalSegmentation = {
                ...segmented,
                dataUrl: reconciliation.dataUrl,
                bbox: reconciliation.bbox
            };
            pushCompletionDebugPreview(
                `reconciliation后canonical候选 - ${asset.targetLayerId}`,
                canonicalSegmentation.dataUrl,
                {
                    targetLayerId: asset.targetLayerId,
                    bbox: canonicalSegmentation.bbox,
                    segmentedBbox: segmented?.bbox,
                    hiddenCoverage: reconciliation.hiddenCoverage,
                    hiddenKeptPixels: reconciliation.hiddenKeptPixels,
                    hiddenCandidatePixels: reconciliation.hiddenCandidatePixels,
                    allowedHiddenCandidatePixels: reconciliation.allowedHiddenCandidatePixels,
                    unknownGeneratedPixels: reconciliation.unknownGeneratedPixels,
                    ownershipEvidenceMissingPixels: reconciliation.ownershipEvidenceMissingPixels,
                    foregroundContextBlockedPixels: reconciliation.foregroundContextBlockedPixels,
                    verifiedCandidateAuthority: reconciliation.verifiedCandidateAuthority,
                    verifiedCandidateEvidence: reconciliation.verifiedCandidateEvidence,
                    foregroundOwnershipEvidenceAvailable: reconciliation.foregroundOwnershipEvidenceAvailable,
                    foregroundOwnershipEvidenceStatus: reconciliation.foregroundOwnershipEvidenceStatus,
                    foregroundContextMaskAttached: reconciliation.foregroundContextMaskAttached,
                    source: 'reconciled_canonical_asset'
                }
            );
            console.info('[Object Completion] canonical reconciliation applied', {
                targetLayerId: asset.targetLayerId,
                segmentedBbox: segmented.bbox,
                completionOutputMode: quality.completionOutputMode,
                hiddenCoverage: reconciliation.hiddenCoverage,
                hiddenCandidatePixels: reconciliation.hiddenCandidatePixels,
                allowedHiddenCandidatePixels: reconciliation.allowedHiddenCandidatePixels,
                unknownGeneratedPixels: reconciliation.unknownGeneratedPixels,
                ownershipEvidenceMissingPixels: reconciliation.ownershipEvidenceMissingPixels,
                foregroundContextBlockedPixels: reconciliation.foregroundContextBlockedPixels,
                bboxExpansion: reconciliation.bboxExpansion,
                verifiedCandidateAuthority: reconciliation.verifiedCandidateAuthority,
                verifiedCandidateEvidence: reconciliation.verifiedCandidateEvidence,
                foregroundOwnershipEvidenceAvailable: reconciliation.foregroundOwnershipEvidenceAvailable,
                foregroundOwnershipEvidenceStatus: reconciliation.foregroundOwnershipEvidenceStatus,
                foregroundContextMaskAttached: reconciliation.foregroundContextMaskAttached
            });
        } else {
            console.info('[Object Completion] direct second SAM canonical candidate selected', {
                targetLayerId: asset.targetLayerId,
                segmentedBbox: segmented.bbox,
                completionOutputMode: quality.completionOutputMode,
                selectedIndexes: quality.selectedIndexes,
                completionIdentityVerified: quality.completionIdentityVerified,
                completionStructureVerified: quality.completionStructureVerified,
                completionRecoveryAudit: quality.completionRecoveryAudit?.status || null,
                completionHiddenRescuePixels: quality.completionHiddenRescuePixels || 0,
                foregroundOwnershipEvidenceStatus
            });
        }

        let occluderAudit = await auditCanonicalSegmentationAgainstOccluders({
            segmented: canonicalSegmentation,
            targetLayer,
            graph,
            sourceSize: materialized.sourceSize
        });
        console.info('[Object Completion] final SAM occlusion overlap diagnostic', {
            taskLockId: preflight?.source?.taskLockId || null,
            targetLayerId: asset.targetLayerId,
            targetZIndex: targetLayer.zIndex,
            ...occluderAudit
        });
        let observationAudit = await auditCanonicalSegmentationAgainstObservation({
            segmented: canonicalSegmentation,
            asset,
            targetLayer,
            graph,
            sourceSize: materialized.sourceSize
        });
        console.info('[Object Completion] final SAM original-observation audit', {
            taskLockId: preflight?.source?.taskLockId || null,
            targetLayerId: asset.targetLayerId,
            targetZIndex: targetLayer.zIndex,
            ...observationAudit
        });
        // Direct second-SAM mode is optimistic but reversible. If its complete
        // cutout does not preserve the original visible silhouette, retain the
        // observed-first reconciliation as a per-task fallback.
        if (directSecondSamUsed && observationAudit.status !== 'passed') {
            console.warn('[Object Completion] direct second SAM audit rejected; falling back to reconciled mask', {
                targetLayerId: asset.targetLayerId,
                reason: observationAudit.reason || observationAudit.status
            });
            reconciliation = await reconcileCompletionSegmentation({
                observedUrl,
                observedBbox,
                segmented,
                completionStructureVerified: quality.completionStructureVerified === true,
                completionForegroundContextMask,
                foregroundOwnershipEvidenceStatus,
                graph,
                task,
                sourceSize: materialized.sourceSize
            });
            directSecondSamUsed = false;
            canonicalSegmentation = {
                ...segmented,
                dataUrl: reconciliation.dataUrl,
                bbox: reconciliation.bbox
            };
            occluderAudit = await auditCanonicalSegmentationAgainstOccluders({
                segmented: canonicalSegmentation,
                targetLayer,
                graph,
                sourceSize: materialized.sourceSize
            });
            observationAudit = await auditCanonicalSegmentationAgainstObservation({
                segmented: canonicalSegmentation,
                asset,
                targetLayer,
                graph,
                sourceSize: materialized.sourceSize
            });
            console.info('[Object Completion] reconciled fallback audits', {
                targetLayerId: asset.targetLayerId,
                occluderAudit,
                observationAudit
            });
        }
        if (observationAudit.status !== 'passed') {
            throw new Error(
                `completion_scene_segmentation_failed_original_observation_audit:` +
                `${observationAudit.reason || observationAudit.status}`
            );
        }

        // The canonical decision is made from the final evidence, not from a
        // stale generic SAM fill/status calculated before reconciliation. A
        // direct second-SAM result uses its own verified hidden-structure
        // evidence; the fallback path uses the reconciler's retained pixels.
        const canonicalEvidence = buildCanonicalEvidence({
            observationAudit,
            reconciliation,
            occluderAudit,
            directSegmentation: directSecondSamUsed ? canonicalSegmentation : null,
            directForegroundOwnershipEvidenceStatus: directSecondSamUsed
                ? foregroundOwnershipEvidenceStatus
                : null,
            directForegroundContextMaskAttached: directSecondSamUsed &&
                Boolean(completionForegroundContextMask)
        });
        const foregroundOwnershipEvidenceAvailable = reconciliation
            ? reconciliation.foregroundOwnershipEvidenceAvailable === true
            : foregroundOwnershipEvidenceStatus === 'attached' ||
                foregroundOwnershipEvidenceStatus === 'empty_valid';
        if (!foregroundOwnershipEvidenceAvailable) {
            console.warn('[Object Completion] canonical held: foreground ownership evidence unavailable', {
                targetLayerId: asset.targetLayerId,
                foregroundContextMaskAttached: reconciliation?.foregroundContextMaskAttached ||
                    quality.foregroundContextMaskAttached || false,
                foregroundOwnershipEvidenceStatus: reconciliation?.foregroundOwnershipEvidenceStatus ||
                    foregroundOwnershipEvidenceStatus || 'unavailable',
                ownershipEvidenceMissingPixels: reconciliation?.ownershipEvidenceMissingPixels || 0,
                hiddenCandidatePixels: reconciliation?.hiddenCandidatePixels ||
                    quality.completionRecoveryAudit?.allowedNewPixels || 0,
                allowedHiddenCandidatePixels: reconciliation?.allowedHiddenCandidatePixels || 0,
                directSecondSam: directSecondSamUsed
            });
            throw new Error('completion_canonical_foreground_ownership_evidence_missing');
        }
        if (canonicalEvidence.status !== 'accepted') {
            throw new Error('completion_canonical_hidden_evidence_missing');
        }
        quality.status = 'ok';
        quality.runtimeAction = 'accept';
        quality.shouldGenerateRuntimeLayer = true;
        quality.needsHigherPrecision = false;
        quality.issues = (quality.issues || []).filter(issue => ![
            'completion_recovery_not_verified',
            'low_quality_status',
            'low_target_fill',
            'low_quality_score'
        ].includes(issue));
        quality.canonicalEvidence = canonicalEvidence;
        quality.directSecondSamUsed = directSecondSamUsed;
        quality.canonicalSource = directSecondSamUsed
            ? 'direct_second_sam'
            : 'reconciled_canonical_asset';

        const canonicalDataUrl = canonicalSegmentation.dataUrl;
        const canonicalImage = await loadImage(canonicalDataUrl);
        const intrinsicWidth = Math.max(1, canonicalImage.naturalWidth || canonicalImage.width);
        const intrinsicHeight = Math.max(1, canonicalImage.naturalHeight || canonicalImage.height);
        releaseImage(canonicalImage);
        const placementBbox = cloneBbox(canonicalSegmentation.bbox) || cloneBbox(samPromptBbox);
        if (!placementBbox) throw new Error('completion_scene_segmentation_bbox_missing');
        console.info('[Object Completion] scene resegmented target', {
            targetLayerId: asset.targetLayerId,
            samPromptBbox,
            segmentedBbox: segmented.bbox || null,
            reconciledBbox: canonicalSegmentation.bbox || null,
            finalPlacementBbox: placementBbox,
            quality
        });

        onProgress('正在保存完整资产版本...');
        console.info('[Object Completion] canonical ready', {
            targetLayerId: asset.targetLayerId,
            source: 'scene_inpaint_sam',
            intrinsicSize: `${intrinsicWidth}x${intrinsicHeight}`,
            placementBbox
        });
        // Keep the just-created File in memory for the immediate workbench
        // replacement. Re-fetching its OSS URL in the browser can fail on CORS
        // even after the upload itself succeeded.
        const canonicalFile = await dataURLToFile(canonicalDataUrl, `canonical-${asset.targetLayerId}-${Date.now()}.png`);
        let storedUrl = canonicalDataUrl;
        try {
            storedUrl = await uploadImageToOSS(canonicalFile);
        } catch (error) {
            console.warn('[Object Completion] OSS upload failed; retaining local canonical asset:', error);
        }

        const now = Date.now();
        asset.canonicalAsset = {
            ...(asset.canonicalAsset || {}),
            status: 'ready',
            cutoutUrl: storedUrl,
            maskUrl: null,
            previewUrl: storedUrl,
            bbox: cloneBbox(canonicalSegmentation.bbox),
            sourceCropBbox: cloneBbox(preflight.geometry.cropBbox),
            observedPlacementBbox: cloneBbox(asset.observedCutout?.bbox || preflight.geometry.targetVisibleBbox),
            placementBbox,
            targetCompletionBbox: cloneBbox(preflight.geometry.targetCompletionBbox || preflight.geometry.targetVisibleBbox),
            intrinsicWidth,
            intrinsicHeight,
            intrinsicAspectRatio: Number((intrinsicWidth / Math.max(1, intrinsicHeight)).toFixed(5)),
            geometryPolicy: directSecondSamUsed
                ? 'scene_inpaint_direct_second_sam'
                : reconciliation
                ? 'scene_inpaint_observed_mask_reconciled'
                : 'scene_inpaint_full_scene_sam',
            generatedAt: now,
            source: 'scene_inpaint_sam',
            generationId: attemptId,
            quality: {
                ...quality,
                completionDifficulty: preflight.difficulty || null,
                completionInputMode: 'original_scene',
                sceneInpaintAudit: inpaintAudit,
                generatedSceneSize: `${completedScene.generatedWidth}x${completedScene.generatedHeight}`,
                editCanvasSize: completedScene.editCanvasSize,
                paddedForModel: completedScene.padded,
                unmaskedPixelsRestored: true,
                samPromptBbox,
                completionSemanticSnapshot,
                segmentedBbox: cloneBbox(segmented.bbox),
                reconciledBbox: cloneBbox(canonicalSegmentation.bbox),
                maskReconciliation: reconciliation
                    ? {
                        observedPixelCount: reconciliation.observedPixelCount,
                        generatedPixelCount: reconciliation.generatedPixelCount,
                        observedPriorityPixels: reconciliation.observedPriorityPixels,
                        hiddenCandidatePixels: reconciliation.hiddenCandidatePixels,
                        allowedHiddenCandidatePixels: reconciliation.allowedHiddenCandidatePixels,
                        hiddenKeptPixels: reconciliation.hiddenKeptPixels,
                        hiddenCoverage: reconciliation.hiddenCoverage,
                        allowedHiddenRegionPixels: reconciliation.allowedHiddenRegionPixels,
                        unknownGeneratedPixels: reconciliation.unknownGeneratedPixels,
                        ownershipEvidenceMissingPixels: reconciliation.ownershipEvidenceMissingPixels,
                        foregroundOwnershipEvidenceAvailable: reconciliation.foregroundOwnershipEvidenceAvailable,
                        foregroundOwnershipEvidenceStatus: reconciliation.foregroundOwnershipEvidenceStatus,
                        bboxExpansion: reconciliation.bboxExpansion,
                        discardedCandidatePixelCount: reconciliation.discardedCandidatePixelCount,
                        disconnectedCandidatePixelCount: reconciliation.disconnectedCandidatePixelCount,
                        keptComponentCount: reconciliation.keptComponentCount,
                        discardedComponentCount: reconciliation.discardedComponentCount,
                        keptGeneratedOnlyComponentCount: reconciliation.keptGeneratedOnlyComponentCount,
                        minimumGeneratedOnlyPixels: reconciliation.minimumGeneratedOnlyPixels,
                        dilationRadius: reconciliation.dilationRadius,
                        drawnOccluderIds: reconciliation.drawnOccluderIds,
                        policy: 'observed_first_hidden_only',
                        verifiedCandidateEvidence: reconciliation.verifiedCandidateEvidence,
                        foregroundContextMaskAttached: reconciliation.foregroundContextMaskAttached
                    }
                    : {
                        status: 'skipped_direct_second_sam',
                        directSecondSam: true,
                        hiddenCandidatePixels: canonicalEvidence.hiddenCandidatePixels,
                        hiddenKeptPixels: canonicalEvidence.hiddenKeptPixels,
                        hiddenCoverage: canonicalEvidence.hiddenCoverage,
                        foregroundOwnershipEvidenceAvailable: canonicalEvidence.foregroundOwnershipEvidenceAvailable,
                        foregroundOwnershipEvidenceStatus: canonicalEvidence.foregroundOwnershipEvidenceStatus,
                        foregroundContextMaskAttached: canonicalEvidence.foregroundContextMaskAttached,
                        policy: 'direct_second_sam_verified_complete_subject'
                    },
                foregroundExclusionAudit: occluderAudit,
                originalObservationAudit: observationAudit
            }
        };
        asset.status = 'canonical_ready';
        if (asset.preflight?.contract) asset.preflight.contract.state = 'canonical_ready';
        asset.generation = { ...(asset.generation || {}), state: 'completed', lastError: null };
        asset.updatedAt = now;
        appendAttempt(asset, {
            ...attempt,
            endedAt: now,
            status: 'completed',
            refinement: 'scene_inpaint_sam',
            completionInputMode: 'original_scene',
            difficulty: preflight.difficulty || null,
            sceneInpaintAudit: inpaintAudit,
            quality
        });
        return { success: true, status: 'canonical_ready', asset, quality, canonicalFile };
    } catch (error) {
        console.error('[Object Completion] execution failed', {
            targetLayerId: asset.targetLayerId,
            taskId: asset.taskId,
            assetStatusBeforeFailure: asset.status,
            generationStateBeforeFailure: asset.generation?.state,
            error: error?.message || String(error)
        });
        asset.status = 'manual_review';
        if (asset.preflight?.contract) asset.preflight.contract.state = 'manual_review';
        asset.generation = {
            ...(asset.generation || {}),
            state: 'failed',
            lastError: error?.message || 'completion_execution_failed'
        };
        asset.updatedAt = Date.now();
        appendAttempt(asset, {
            ...attempt,
            endedAt: Date.now(),
            status: 'failed',
            error: error?.message || 'completion_execution_failed'
        });
        throw error;
    }
}

function cloneBbox(bbox) {
    return Array.isArray(bbox) ? bbox.map(value => Number(value)) : null;
}
