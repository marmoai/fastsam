import { getProxiedUrl } from '../core/utils.js';
import { state } from '../core/state.js';

function bboxArea(bbox) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return 0;
    return Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function clampBbox(bbox) {
    return [
        clamp(Number(bbox[0]), 0, 1000),
        clamp(Number(bbox[1]), 0, 1000),
        clamp(Number(bbox[2]), 0, 1000),
        clamp(Number(bbox[3]), 0, 1000)
    ];
}

function bboxIntersection(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return null;
    const ymin = Math.max(Number(a[0]), Number(b[0]));
    const xmin = Math.max(Number(a[1]), Number(b[1]));
    const ymax = Math.min(Number(a[2]), Number(b[2]));
    const xmax = Math.min(Number(a[3]), Number(b[3]));
    if (ymax <= ymin || xmax <= xmin) return null;
    return [ymin, xmin, ymax, xmax];
}

function isContainerLayer(layer) {
    const semantic = String(layer?.semanticType || '').toLowerCase();
    const role = String(layer?.designRole || '').toLowerCase();
    const mode = String(layer?.renderMode || '').toLowerCase();
    return semantic === 'shape_panel' ||
        semantic === 'price_badge' ||
        semantic === 'cta_button' ||
        role === 'local_panel' ||
        role === 'price_badge' ||
        mode === 'vector_shape';
}

function bboxToWorkbenchRect(bbox, baseX, baseY, parentWidth, parentHeight, minWidth = 6, minHeight = 6) {
    const safeBbox = Array.isArray(bbox) && bbox.length === 4 ? bbox : [0, 0, 1000, 1000];
    const [ymin, xmin, ymax, xmax] = safeBbox.map(value => Number(value));
    return {
        left: baseX + (xmin / 1000) * parentWidth,
        top: baseY + (ymin / 1000) * parentHeight,
        width: Math.max(((xmax - xmin) / 1000) * parentWidth, minWidth),
        height: Math.max(((ymax - ymin) / 1000) * parentHeight, minHeight)
    };
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        if (!src) {
            reject(new Error('Missing image source'));
            return;
        }
        const img = new Image();
        let objectUrl = '';
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            if (objectUrl) URL.revokeObjectURL(objectUrl);
            resolve(img);
        };
        img.onerror = (error) => {
            if (objectUrl) URL.revokeObjectURL(objectUrl);
            reject(error);
        };
        if (src instanceof Blob) {
            objectUrl = URL.createObjectURL(src);
            img.src = objectUrl;
        } else {
            img.src = getProxiedUrl(src);
        }
    });
}

function rgbToHex(r, g, b) {
    return `#${[r, g, b].map(value => Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0')).join('')}`;
}

function averagePixels(samples) {
    if (!samples.length) return null;
    const sum = samples.reduce((acc, pixel) => {
        acc[0] += pixel[0];
        acc[1] += pixel[1];
        acc[2] += pixel[2];
        return acc;
    }, [0, 0, 0]);
    return [
        sum[0] / samples.length,
        sum[1] / samples.length,
        sum[2] / samples.length
    ];
}

async function createPanelDebugImageContext(sourceImage) {
    const img = await loadImage(sourceImage);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, img.naturalWidth || img.width || 1);
    canvas.height = Math.max(1, img.naturalHeight || img.height || 1);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return {
        canvas,
        ctx,
        width: canvas.width,
        height: canvas.height
    };
}

async function sampleBboxFillColor(sourceImage, bbox, imageContext = null) {
    try {
        const context = imageContext || await createPanelDebugImageContext(sourceImage);
        const [ymin, xmin, ymax, xmax] = bbox.map(value => Number(value));
        const sx = Math.max(0, (xmin / 1000) * context.width);
        const sy = Math.max(0, (ymin / 1000) * context.height);
        const sw = Math.max(1, ((xmax - xmin) / 1000) * context.width);
        const sh = Math.max(1, ((ymax - ymin) / 1000) * context.height);
        const sampleW = 48;
        const sampleH = 48;
        const canvas = document.createElement('canvas');
        canvas.width = sampleW;
        canvas.height = sampleH;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(context.canvas, sx, sy, sw, sh, 0, 0, sampleW, sampleH);
        const data = ctx.getImageData(0, 0, sampleW, sampleH).data;
        const buckets = new Map();

        for (let i = 0; i < data.length; i += 4) {
            if (data[i + 3] < 24) continue;
            const key = [data[i], data[i + 1], data[i + 2]]
                .map(value => Math.round(value / 18) * 18)
                .join(',');
            const bucket = buckets.get(key) || { count: 0, sum: [0, 0, 0] };
            bucket.count += 1;
            bucket.sum[0] += data[i];
            bucket.sum[1] += data[i + 1];
            bucket.sum[2] += data[i + 2];
            buckets.set(key, bucket);
        }

        const dominant = Array.from(buckets.values()).sort((a, b) => b.count - a.count)[0];
        if (!dominant) return '#ffffff';
        return rgbToHex(
            dominant.sum[0] / dominant.count,
            dominant.sum[1] / dominant.count,
            dominant.sum[2] / dominant.count
        );
    } catch (error) {
        console.warn('[TextContainerRestore] Failed to sample fill color:', error.message || error);
        return '#ffffff';
    }
}

function parseHexColor(color) {
    if (!color) return null;
    const value = String(color).trim();
    if (/^#[0-9a-f]{3}$/i.test(value)) {
        return {
            r: parseInt(value[1] + value[1], 16),
            g: parseInt(value[2] + value[2], 16),
            b: parseInt(value[3] + value[3], 16)
        };
    }
    if (!/^#[0-9a-f]{6}$/i.test(value)) return null;
    return {
        r: parseInt(value.slice(1, 3), 16),
        g: parseInt(value.slice(3, 5), 16),
        b: parseInt(value.slice(5, 7), 16)
    };
}

function colorDistance(color, r, g, b) {
    const dr = color.r - r;
    const dg = color.g - g;
    const db = color.b - b;
    return Math.sqrt(dr * dr + dg * dg + db * db);
}

function fitEdgeLine(points) {
    if (!Array.isArray(points) || points.length < 8) return null;
    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumXY = 0;
    for (const [x, y] of points) {
        sumX += x;
        sumY += y;
        sumXX += x * x;
        sumXY += x * y;
    }
    const n = points.length;
    const denom = n * sumXX - sumX * sumX;
    if (Math.abs(denom) < 0.0001) return null;
    const slope = (n * sumXY - sumX * sumY) / denom;
    const intercept = (sumY - slope * sumX) / n;
    return { slope, intercept };
}

function evaluateEdgeLine(line, x, fallback) {
    return line ? line.slope * x + line.intercept : fallback;
}

function readPanelDebugCrop(imageContext, bbox, sampleW, sampleH) {
    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    const sx = Math.max(0, (xmin / 1000) * imageContext.width);
    const sy = Math.max(0, (ymin / 1000) * imageContext.height);
    const sw = Math.max(1, ((xmax - xmin) / 1000) * imageContext.width);
    const sh = Math.max(1, ((ymax - ymin) / 1000) * imageContext.height);
    const canvas = document.createElement('canvas');
    canvas.width = sampleW;
    canvas.height = sampleH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imageContext.canvas, sx, sy, sw, sh, 0, 0, sampleW, sampleH);
    return ctx.getImageData(0, 0, sampleW, sampleH).data;
}

function detectSyntheticPriceBadgeVisual(imageContext, bbox) {
    if (!imageContext || !Array.isArray(bbox) || bbox.length !== 4) {
        return { valid: false, reason: 'missing_image_or_bbox' };
    }

    try {
        const sampleW = 36;
        const sampleH = 36;
        const data = readPanelDebugCrop(imageContext, clampBbox(bbox), sampleW, sampleH);
        const centerPixels = [];
        const outerPixels = [];

        for (let y = 0; y < sampleH; y += 1) {
            for (let x = 0; x < sampleW; x += 1) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                const nx = (x + 0.5) / sampleW;
                const ny = (y + 0.5) / sampleH;
                const dx = nx - 0.5;
                const dy = ny - 0.5;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist <= 0.22) {
                    centerPixels.push(pixel);
                } else if (dist >= 0.42) {
                    outerPixels.push(pixel);
                }
            }
        }

        const center = averagePixels(centerPixels);
        const outer = averagePixels(outerPixels);
        if (!center || !outer) {
            return { valid: false, reason: 'insufficient_pixels' };
        }

        const centerOuterDistance = colorDistance(
            { r: center[0], g: center[1], b: center[2] },
            outer[0],
            outer[1],
            outer[2]
        );
        const centerUniformity = centerPixels.length
            ? centerPixels.reduce((sum, pixel) => (
                sum + colorDistance(
                    { r: center[0], g: center[1], b: center[2] },
                    pixel[0],
                    pixel[1],
                    pixel[2]
                )
            ), 0) / centerPixels.length
            : Infinity;

        return {
            valid: centerOuterDistance >= 42 && centerUniformity <= 42,
            reason: centerOuterDistance >= 42 && centerUniformity <= 42 ? 'distinct_badge_blob' : 'flat_background_like',
            centerOuterDistance,
            centerUniformity
        };
    } catch (error) {
        return { valid: false, reason: error.message || String(error) };
    }
}

function detectPanelShapeDebugCandidate(imageContext, bbox, fillColor) {
    try {
        const fill = parseHexColor(fillColor);
        if (!imageContext || !fill || !Array.isArray(bbox) || bbox.length !== 4) {
            return { status: 'skipped', reason: 'missing_image_or_color' };
        }

        const [ymin, xmin, ymax, xmax] = bbox.map(Number);
        const boxW = xmax - xmin;
        const boxH = ymax - ymin;
        const cropBbox = clampBbox(bbox);
        const aspect = Math.max(0.35, Math.min(3.2, boxW / Math.max(1, boxH)));
        const sampleW = Math.round(clamp(150 * Math.sqrt(aspect), 118, 220));
        const sampleH = Math.round(clamp(sampleW / aspect, 64, 170));
        const data = readPanelDebugCrop(imageContext, cropBbox, sampleW, sampleH);
        const luma = 0.2126 * fill.r + 0.7152 * fill.g + 0.0722 * fill.b;
        const threshold = luma > 180 ? 58 : 48;
        const mask = Array(sampleW * sampleH).fill(false);
        let maskCount = 0;
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        for (let y = 0; y < sampleH; y += 1) {
            for (let x = 0; x < sampleW; x += 1) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                if (colorDistance(fill, data[offset], data[offset + 1], data[offset + 2]) > threshold) continue;
                mask[y * sampleW + x] = true;
                maskCount += 1;
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
        }

        if (!maskCount || !Number.isFinite(minX) || !Number.isFinite(minY)) {
            return { status: 'skipped', reason: 'no_matching_panel_pixels' };
        }

        const maskW = maxX - minX + 1;
        const maskH = maxY - minY + 1;
        const coverage = maskCount / Math.max(1, maskW * maskH);
        if (maskW < sampleW * 0.32 || maskH < sampleH * 0.24 || coverage < 0.18) {
            return { status: 'skipped', reason: 'weak_panel_mask', coverage };
        }

        const minRun = Math.max(2, Math.round(sampleH * 0.025));
        const topLimit = Math.round(sampleH * 0.38);
        const bottomLimit = Math.round(sampleH * 0.62);
        const edgePadX = Math.round(sampleW * 0.04);
        const topPoints = [];
        const bottomPoints = [];

        for (let x = edgePadX; x < sampleW - edgePadX; x += 1) {
            let topY = null;
            let topRun = 0;
            for (let y = 0; y <= topLimit; y += 1) {
                if (mask[y * sampleW + x]) {
                    topRun += 1;
                    if (topRun >= minRun) {
                        topY = y - topRun + 1;
                        break;
                    }
                } else {
                    topRun = 0;
                }
            }
            if (topY !== null) topPoints.push([x, topY]);

            let bottomY = null;
            let bottomRun = 0;
            for (let y = sampleH - 1; y >= bottomLimit; y -= 1) {
                if (mask[y * sampleW + x]) {
                    bottomRun += 1;
                    if (bottomRun >= minRun) {
                        bottomY = y + bottomRun - 1;
                        break;
                    }
                } else {
                    bottomRun = 0;
                }
            }
            if (bottomY !== null) bottomPoints.push([x, bottomY]);
        }

        const minEdgePoints = Math.max(12, Math.round(sampleW * 0.28));
        const topLine = topPoints.length >= minEdgePoints ? fitEdgeLine(topPoints) : null;
        const bottomLine = bottomPoints.length >= minEdgePoints ? fitEdgeLine(bottomPoints) : null;
        if (!topLine && !bottomLine) {
            return {
                status: 'skipped',
                reason: 'edge_band_fit_failed',
                coverage,
                topEdgePoints: topPoints.length,
                bottomEdgePoints: bottomPoints.length
            };
        }

        const topSlant = topLine
            ? Math.abs(evaluateEdgeLine(topLine, sampleW - 1, 0) - evaluateEdgeLine(topLine, 0, 0)) / Math.max(1, sampleH)
            : 0;
        const bottomSlant = bottomLine
            ? Math.abs(evaluateEdgeLine(bottomLine, sampleW - 1, sampleH - 1) - evaluateEdgeLine(bottomLine, 0, sampleH - 1)) / Math.max(1, sampleH)
            : 0;
        const strongestSlant = Math.max(topSlant, bottomSlant);
        const toPctY = (sampleY) => clamp((sampleY / Math.max(1, sampleH - 1)) * 100, 0, 100);
        const polygon = [
            [0, toPctY(evaluateEdgeLine(topLine, 0, 0))],
            [100, toPctY(evaluateEdgeLine(topLine, sampleW - 1, 0))],
            [100, toPctY(evaluateEdgeLine(bottomLine, sampleW - 1, sampleH - 1))],
            [0, toPctY(evaluateEdgeLine(bottomLine, 0, sampleH - 1))]
        ];

        return {
            status: strongestSlant >= 0.12 ? 'candidate' : 'regular',
            coverage,
            topSlant,
            bottomSlant,
            strongestSlant,
            polygon,
            topEdgePoints: topPoints.length,
            bottomEdgePoints: bottomPoints.length
        };
    } catch (error) {
        return { status: 'error', reason: error.message || String(error) };
    }
}

function horizontalOverlapRatio(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b)) return 0;
    const overlap = Math.max(0, Math.min(Number(a[3]), Number(b[3])) - Math.max(Number(a[1]), Number(b[1])));
    const minWidth = Math.max(1, Math.min(Number(a[3]) - Number(a[1]), Number(b[3]) - Number(b[1])));
    return overlap / minWidth;
}

function isLargeRectPanelCandidate(layer) {
    if (!Array.isArray(layer?.bbox) || layer.bbox.length !== 4) return false;
    const bbox = layer.bbox.map(Number);
    const width = bbox[3] - bbox[1];
    const height = bbox[2] - bbox[0];
    const semantic = String(layer.semanticType || layer.designRole || '').toLowerCase();
    return !semantic.includes('price') && width >= 130 && height >= 55 && !(height < 75 && width < 260);
}

function getDominantPanelStats(siblingPanels = []) {
    const panels = siblingPanels
        .filter(isLargeRectPanelCandidate)
        .map(layer => {
            const bbox = layer.bbox.map(Number);
            const width = bbox[3] - bbox[1];
            const height = bbox[2] - bbox[0];
            return { layer, bbox, width, height, area: width * height };
        })
        .filter(panel => panel.area > 0);

    if (!panels.length) {
        return { panels: [], maxArea: 0, maxWidth: 0, maxHeight: 0 };
    }

    return {
        panels,
        maxArea: Math.max(...panels.map(panel => panel.area)),
        maxWidth: Math.max(...panels.map(panel => panel.width)),
        maxHeight: Math.max(...panels.map(panel => panel.height))
    };
}

function isDominantContentPanel(layer, siblingPanels = []) {
    if (!isLargeRectPanelCandidate(layer)) return false;
    const stats = getDominantPanelStats(siblingPanels);
    const panel = stats.panels.find(candidate => candidate.layer === layer);
    if (!panel || stats.panels.length < 2) return false;

    return panel.area >= stats.maxArea * 0.42 &&
        panel.height >= stats.maxHeight * 0.58 &&
        panel.width >= stats.maxWidth * 0.52;
}

function pctYInBbox(globalY, bbox) {
    const height = Math.max(1, Number(bbox[2]) - Number(bbox[0]));
    return clamp(((globalY - Number(bbox[0])) / height) * 100, 0, 100);
}

function findLayoutGuidedPanelNeighbor(layer, siblingPanels = []) {
    if (!isDominantContentPanel(layer, siblingPanels)) return null;
    const bbox = layer.bbox.map(Number);
    const centerY = (bbox[0] + bbox[2]) / 2;
    const centerX = (bbox[1] + bbox[3]) / 2;
    const height = bbox[2] - bbox[0];
    const width = bbox[3] - bbox[1];
    const siblings = siblingPanels
        .filter(candidate => candidate !== layer && isDominantContentPanel(candidate, siblingPanels))
        .map(candidate => {
            const otherBbox = candidate.bbox.map(Number);
            const otherCenterY = (otherBbox[0] + otherBbox[2]) / 2;
            const otherCenterX = (otherBbox[1] + otherBbox[3]) / 2;
            const otherHeight = otherBbox[2] - otherBbox[0];
            const otherWidth = otherBbox[3] - otherBbox[1];
            const verticalDistance = Math.abs(otherCenterY - centerY);
            const horizontalDistance = Math.abs(otherCenterX - centerX);
            const overlapX = horizontalOverlapRatio(bbox, otherBbox);
            const closeStack = overlapX >= 0.34 && verticalDistance <= Math.max(height, otherHeight) * 2.15;
            const closeDiagonal = overlapX >= 0.18 &&
                verticalDistance <= Math.max(height, otherHeight) * 2.35 &&
                horizontalDistance <= Math.max(width, otherWidth) * 0.82;
            return {
                layer: candidate,
                bbox: otherBbox,
                centerY: otherCenterY,
                centerX: otherCenterX,
                verticalDistance,
                horizontalDistance,
                overlapX,
                related: closeStack || closeDiagonal
            };
        })
        .filter(candidate => candidate.related)
        .sort((a, b) => (b.overlapX - a.overlapX) || (a.verticalDistance - b.verticalDistance) || (a.horizontalDistance - b.horizontalDistance));

    return siblings[0] || null;
}

function lineYAtX(line, x) {
    if (!line || Math.abs(line.x2 - line.x1) < 0.0001) {
        return line ? (line.y1 + line.y2) / 2 : 0;
    }
    const t = (x - line.x1) / (line.x2 - line.x1);
    return line.y1 + (line.y2 - line.y1) * t;
}

function lineYAtClampedX(line, x) {
    if (!line) return 0;
    const minX = Math.min(line.x1, line.x2);
    const maxX = Math.max(line.x1, line.x2);
    return lineYAtX(line, clamp(x, minX, maxX));
}

function sampleNormalizedPixel(imageContext, x, y) {
    if (!imageContext?.ctx) return null;
    const px = clamp(Math.round((x / 1000) * (imageContext.width - 1)), 0, imageContext.width - 1);
    const py = clamp(Math.round((y / 1000) * (imageContext.height - 1)), 0, imageContext.height - 1);
    const data = imageContext.ctx.getImageData(px, py, 1, 1).data;
    if (data[3] < 24) return null;
    return [data[0], data[1], data[2]];
}

function estimateFillThreshold(fill) {
    const luma = 0.2126 * fill.r + 0.7152 * fill.g + 0.0722 * fill.b;
    return luma > 180 ? 54 : 46;
}

function fillRatioNear(imageContext, fill, x, y, offsets, threshold) {
    let hits = 0;
    let total = 0;
    for (const offset of offsets) {
        const pixel = sampleNormalizedPixel(imageContext, x, y + offset);
        if (!pixel) continue;
        total += 1;
        if (colorDistance(fill, pixel[0], pixel[1], pixel[2]) <= threshold) {
            hits += 1;
        }
    }
    return total ? hits / total : 0;
}

function snapPanelBoundaryLineToPixels({ imageContext, fillColor, bbox, line, isUpper, gapOffset }) {
    const fill = parseHexColor(fillColor);
    if (!imageContext || !fill || !line || !Array.isArray(bbox) || bbox.length !== 4) return null;

    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    const width = xmax - xmin;
    const height = ymax - ymin;
    const threshold = estimateFillThreshold(fill);
    const searchRadius = clamp(height * 0.18, 18, 56);
    const band = clamp(height * 0.035, 5, 13);
    const stepCount = Math.round(clamp(width / 14, 22, 48));
    const edgePad = width * 0.04;
    const points = [];

    for (let i = 0; i < stepCount; i += 1) {
        const t = stepCount === 1 ? 0.5 : i / (stepCount - 1);
        const x = xmin + edgePad + t * Math.max(1, width - edgePad * 2);
        const predictedY = lineYAtClampedX(line, x) + gapOffset;
        let best = null;

        for (let y = predictedY - searchRadius; y <= predictedY + searchRadius; y += 2) {
            if (y <= ymin || y >= ymax) continue;
            const above = fillRatioNear(imageContext, fill, x, y, [-band * 1.7, -band, -band * 0.45], threshold);
            const below = fillRatioNear(imageContext, fill, x, y, [band * 0.45, band, band * 1.7], threshold);
            const transitionScore = isUpper ? above - below : below - above;
            const confidence = Math.abs(above - below);
            const distancePenalty = Math.abs(y - predictedY) / Math.max(1, searchRadius);
            const score = transitionScore + confidence * 0.35 - distancePenalty * 0.32;
            if (transitionScore < 0.34 || confidence < 0.34) continue;
            if (!best || score > best.score) {
                best = { x, y, score, above, below };
            }
        }

        if (best) points.push([best.x, best.y]);
    }

    const minPoints = Math.max(9, Math.round(stepCount * 0.32));
    if (points.length < minPoints) {
        return { snapped: false, points: points.length };
    }

    const fitted = fitEdgeLine(points);
    if (!fitted) {
        return { snapped: false, points: points.length };
    }

    const fallbackLeftY = lineYAtClampedX(line, xmin) + gapOffset;
    const fallbackRightY = lineYAtClampedX(line, xmax) + gapOffset;
    const rawLeftY = evaluateEdgeLine(fitted, xmin, fallbackLeftY);
    const rawRightY = evaluateEdgeLine(fitted, xmax, fallbackRightY);
    const maxUpShift = height * 0.18;
    const maxDownShift = isUpper ? height * 0.04 : height * 0.16;
    const leftY = clamp(rawLeftY, fallbackLeftY - maxUpShift, fallbackLeftY + maxDownShift);
    const rightY = clamp(rawRightY, fallbackRightY - maxUpShift, fallbackRightY + maxDownShift);
    const leftClamped = Math.abs(leftY - rawLeftY) > 0.001;
    const rightClamped = Math.abs(rightY - rawRightY) > 0.001;

    return {
        snapped: true,
        points: points.length,
        clamped: leftClamped || rightClamped,
        leftY,
        rightY,
        slope: fitted.slope,
        intercept: fitted.intercept,
        rawLeftY,
        rawRightY,
        fallbackLeftY,
        fallbackRightY
    };
}

function evaluateSnappedBoundary(snap, x, fallback) {
    if (!snap?.snapped || typeof snap.slope !== 'number' || typeof snap.intercept !== 'number') {
        return fallback;
    }
    return snap.slope * x + snap.intercept;
}

function scoreBoundarySnap(snap, bbox) {
    if (!snap?.snapped || !Array.isArray(bbox) || bbox.length !== 4) return 0;
    const height = Math.max(1, Number(bbox[2]) - Number(bbox[0]));
    const pointScore = clamp((Number(snap.points) || 0) / 45, 0, 1);
    const leftDrift = Math.abs(Number(snap.rawLeftY) - Number(snap.fallbackLeftY)) / height;
    const rightDrift = Math.abs(Number(snap.rawRightY) - Number(snap.fallbackRightY)) / height;
    const driftPenalty = clamp((leftDrift + rightDrift) / 0.42, 0, 1);
    const clampPenalty = snap.clamped ? 0.22 : 0;
    return clamp(pointScore * 0.78 + (1 - driftPenalty) * 0.22 - clampPenalty, 0, 1);
}

function detectLayoutGuidedPanelCandidate(layer, siblingPanels = [], options = {}) {
    const neighbor = findLayoutGuidedPanelNeighbor(layer, siblingPanels);
    if (!neighbor) return null;
    const bbox = layer.bbox.map(Number);
    const centerY = (bbox[0] + bbox[2]) / 2;
    const height = bbox[2] - bbox[0];
    const otherHeight = neighbor.bbox[2] - neighbor.bbox[0];
    const isUpper = centerY < neighbor.centerY;
    const upperBbox = isUpper ? bbox : neighbor.bbox;
    const lowerBbox = isUpper ? neighbor.bbox : bbox;
    const upperHeight = upperBbox[2] - upperBbox[0];
    const lowerHeight = lowerBbox[2] - lowerBbox[0];
    const overlapLeft = Math.max(upperBbox[1], lowerBbox[1]);
    const overlapRight = Math.min(upperBbox[3], lowerBbox[3]);
    const hasOverlap = overlapRight > overlapLeft;
    const line = hasOverlap
        ? {
            x1: overlapLeft,
            y1: upperBbox[2] - upperHeight * 0.035,
            x2: overlapRight,
            y2: lowerBbox[0] + lowerHeight * 0.045
        }
        : {
            x1: Math.min(upperBbox[1], lowerBbox[1]),
            y1: upperBbox[2] - upperHeight * 0.035,
            x2: Math.max(upperBbox[3], lowerBbox[3]),
            y2: lowerBbox[0] + lowerHeight * 0.045
    };
    const gap = clamp(Math.min(upperHeight, lowerHeight) * 0.085, 12, 30);
    const gapOffset = isUpper ? -gap / 2 : gap / 2;
    const ownSnap = snapPanelBoundaryLineToPixels({
        imageContext: options.imageContext,
        fillColor: options.fillColor,
        bbox,
        line,
        isUpper,
        gapOffset
    });
    const neighborSnap = snapPanelBoundaryLineToPixels({
        imageContext: options.imageContext,
        fillColor: options.fillColor,
        bbox: isUpper ? lowerBbox : upperBbox,
        line,
        isUpper: !isUpper,
        gapOffset: isUpper ? gap / 2 : -gap / 2
    });
    const ownScore = scoreBoundarySnap(ownSnap, bbox);
    const neighborScore = scoreBoundarySnap(neighborSnap, isUpper ? lowerBbox : upperBbox);
    const useNeighborAnchor = neighborSnap?.snapped && (!ownSnap?.snapped || neighborScore >= ownScore + 0.08);
    const snappedBoundary = useNeighborAnchor
        ? {
            ...neighborSnap,
            anchoredFromNeighbor: true,
            anchorRole: isUpper ? 'lower_panel_top_boundary' : 'upper_panel_bottom_boundary',
            ownScore,
            neighborScore
        }
        : ownSnap?.snapped
            ? {
                ...ownSnap,
                anchorRole: isUpper ? 'own_bottom_boundary' : 'own_top_boundary',
                ownScore,
                neighborScore
            }
            : null;
    const leftBoundaryY = snappedBoundary?.snapped
        ? (useNeighborAnchor
            ? evaluateSnappedBoundary(neighborSnap, bbox[1], lineYAtClampedX(line, bbox[1]) - gapOffset) + gapOffset * 2
            : snappedBoundary.leftY)
        : lineYAtClampedX(line, bbox[1]) + gapOffset;
    const rightBoundaryY = snappedBoundary?.snapped
        ? (useNeighborAnchor
            ? evaluateSnappedBoundary(neighborSnap, bbox[3], lineYAtClampedX(line, bbox[3]) - gapOffset) + gapOffset * 2
            : snappedBoundary.rightY)
        : lineYAtClampedX(line, bbox[3]) + gapOffset;
    const slant = Math.abs(rightBoundaryY - leftBoundaryY) / Math.max(1, height);
    if (slant < 0.10) return null;

    const polygon = isUpper
        ? [
            [0, 0],
            [100, 0],
            [100, pctYInBbox(rightBoundaryY, bbox)],
            [0, pctYInBbox(leftBoundaryY, bbox)]
        ]
        : [
            [0, pctYInBbox(leftBoundaryY, bbox)],
            [100, pctYInBbox(rightBoundaryY, bbox)],
            [100, 100],
            [0, 100]
        ];

    return {
        status: 'candidate',
        reason: 'layout_guided_stacked_panel',
        strongestSlant: slant,
        topSlant: isUpper ? 0 : slant,
        bottomSlant: isUpper ? slant : 0,
        coverage: null,
        polygon,
        neighborName: neighbor.layer.name || 'neighbor-panel',
        pixelSnap: snappedBoundary || null,
        relation: {
            overlapX: neighbor.overlapX,
            verticalDistance: neighbor.verticalDistance,
            horizontalDistance: neighbor.horizontalDistance
        }
    };
}

function diagnosePanelRefineCandidate({ layer, shapeType, textLines, fillColor, imageContext, siblingPanels = [] }) {
    const bbox = Array.isArray(layer?.bbox) ? layer.bbox.map(Number) : null;
    const reasons = [];
    if (!bbox || bbox.length !== 4) {
        return { status: 'excluded', reasons: ['invalid_bbox'] };
    }

    const width = bbox[3] - bbox[1];
    const height = bbox[2] - bbox[0];
    const semantic = String(layer.semanticType || layer.designRole || '').toLowerCase();
    const overlappingTextCount = textLines.filter(line => {
        const intersection = bboxIntersection(bbox, line?.bbox);
        return intersection && bboxArea(intersection) / Math.max(1, bboxArea(line.bbox)) >= 0.35;
    }).length;

    const layoutGuidedShape = shapeType === 'rect'
        ? detectLayoutGuidedPanelCandidate(layer, siblingPanels, { imageContext, fillColor })
        : null;

    if (shapeType !== 'rect') reasons.push('not_rect');
    if (semantic.includes('price')) reasons.push('price_or_badge');
    if (width < 130 || height < 55) reasons.push('too_small');
    if (height < 75 && width < 260) reasons.push('small_label_like_panel');
    if (overlappingTextCount < 2 && !layer.generatedBy && !layoutGuidedShape) reasons.push('weak_text_group');

    const debugShape = reasons.length === 0
        ? (layoutGuidedShape || detectPanelShapeDebugCandidate(imageContext, bbox, fillColor))
        : { status: 'skipped', reason: 'pre_filter_excluded' };
    const shouldRefine = reasons.length === 0 && debugShape.status === 'candidate';

    return {
        layerName: layer.name || 'text-container',
        bbox,
        shapeType,
        fillColor,
        status: shouldRefine ? 'candidate' : 'excluded',
        reasons: shouldRefine ? [] : reasons.concat(debugShape.reason || debugShape.status || 'regular_shape'),
        metrics: {
            width,
            height,
            overlappingTextCount,
            coverage: debugShape.coverage ?? null,
            topSlant: debugShape.topSlant ?? null,
            bottomSlant: debugShape.bottomSlant ?? null,
            strongestSlant: debugShape.strongestSlant ?? null,
            neighborName: debugShape.neighborName || null,
            pixelSnap: debugShape.pixelSnap || null
        },
        candidatePolygon: shouldRefine ? debugShape.polygon : null
    };
}

function publishPanelRefineDiagnostics(item, itemId, diagnostics) {
    const payload = {
        itemId,
        generatedAt: new Date().toISOString(),
        diagnostics
    };
    item.textContainerShapeRefineDiagnostics = payload;
    if (typeof window !== 'undefined') {
        window.__marmoTextPanelRefineDiagnostics = window.__marmoTextPanelRefineDiagnostics || {};
        window.__marmoTextPanelRefineDiagnostics[itemId] = payload;
        if (diagnostics.length > 0) {
            console.info('[TextContainerRestore] panel refine diagnostics', payload);
            console.table(diagnostics.map(entry => ({
                layer: entry.layerName,
                status: entry.status,
                reasons: entry.reasons.join(','),
                width: Math.round(entry.metrics.width),
                height: Math.round(entry.metrics.height),
                slant: entry.metrics.strongestSlant
            })));
        }
    }
    return payload;
}

function clearPanelRefinePreviewOverlays(itemId) {
    if (typeof document === 'undefined') return;
    document
        .querySelectorAll(`.text-panel-refine-preview[data-parent-id="${itemId}"]`)
        .forEach(el => el.remove());
}

function renderPanelRefinePreviewOverlays({
    itemId,
    diagnostics,
    baseX,
    baseY,
    parentWidth,
    parentHeight,
    zIndexBase
}) {
    if (typeof document === 'undefined' || !window.workbenchGrid) return;
    clearPanelRefinePreviewOverlays(itemId);

    diagnostics
        .filter(entry => entry.status === 'candidate' && Array.isArray(entry.bbox) && Array.isArray(entry.candidatePolygon))
        .forEach((entry, index) => {
            const rect = bboxToWorkbenchRect(entry.bbox, baseX, baseY, parentWidth, parentHeight);
            const overlay = document.createElement('div');
            overlay.className = 'text-panel-refine-preview';
            overlay.dataset.parentId = itemId;
            overlay.dataset.layerName = entry.layerName || '';
            overlay.style.position = 'absolute';
            overlay.style.left = `${rect.left}px`;
            overlay.style.top = `${rect.top}px`;
            overlay.style.width = `${rect.width}px`;
            overlay.style.height = `${rect.height}px`;
            overlay.style.zIndex = String(Math.max(0, zIndexBase + 80 + index));
            overlay.style.pointerEvents = 'none';

            const points = entry.candidatePolygon
                .map(([x, y]) => `${(Number(x) / 100) * rect.width},${(Number(y) / 100) * rect.height}`)
                .join(' ');
            overlay.innerHTML = `
                <svg width="100%" height="100%" viewBox="0 0 ${rect.width} ${rect.height}" preserveAspectRatio="none">
                    <polygon points="${points}" fill="rgba(255, 59, 48, 0.08)" stroke="#ff3b30" stroke-width="3" stroke-dasharray="8 5" vector-effect="non-scaling-stroke"></polygon>
                </svg>
            `;
            window.workbenchGrid.appendChild(overlay);
        });
}

function polygonToClipPath(polygon) {
    if (!Array.isArray(polygon) || polygon.length < 3) return null;
    const points = polygon
        .map(point => {
            if (!Array.isArray(point) || point.length !== 2) return null;
            const x = clamp(Number(point[0]), 0, 100);
            const y = clamp(Number(point[1]), 0, 100);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            return `${x.toFixed(3)}% ${y.toFixed(3)}%`;
        })
        .filter(Boolean);
    return points.length >= 3 ? `polygon(${points.join(', ')})` : null;
}

function applyShapeClipPath(shapeItem, clipPath) {
    const el = shapeItem?.el;
    if (!el) return;
    if (clipPath) {
        el.style.clipPath = clipPath;
        el.style.webkitClipPath = clipPath;
        shapeItem.clipPath = clipPath;
    } else {
        el.style.clipPath = '';
        el.style.webkitClipPath = '';
        shapeItem.clipPath = null;
    }
}

function safeDiagnosePanelRefineCandidate(params) {
    try {
        return diagnosePanelRefineCandidate(params);
    } catch (error) {
        return {
            layerName: params?.layer?.name || 'text-container',
            bbox: Array.isArray(params?.layer?.bbox) ? params.layer.bbox.map(Number) : null,
            shapeType: params?.shapeType || 'unknown',
            fillColor: params?.fillColor || null,
            status: 'error',
            reasons: [error.message || String(error)],
            metrics: {
                width: null,
                height: null,
                overlappingTextCount: null,
                coverage: null,
                topSlant: null,
                bottomSlant: null,
                strongestSlant: null
            },
            candidatePolygon: null
        };
    }
}

export function prepareTextContainerCandidates(item, textLines = [], sourceImage = null) {
    const layers = [
        ...(Array.isArray(item?.semanticViews?.editableSceneLayers) ? item.semanticViews.editableSceneLayers : []),
        ...(Array.isArray(item?.scene?.layers) ? item.scene.layers : []),
        ...(Array.isArray(item?.layers) ? item.layers : [])
    ];
    const textBboxes = textLines.map(line => line?.bbox).filter(bbox => Array.isArray(bbox) && bbox.length === 4);
    const seen = new Set();
    let candidates = layers.filter(layer => {
        if (!isContainerLayer(layer) || !Array.isArray(layer.bbox) || layer.bbox.length !== 4) return false;
        const key = layer.bbox.map(value => Math.round(Number(value))).join(',');
        if (seen.has(key)) return false;
        seen.add(key);
        return textBboxes.some(textBbox => {
            const intersection = bboxIntersection(layer.bbox, textBbox);
            if (!intersection) return false;
            return bboxArea(intersection) / Math.max(1, bboxArea(textBbox)) >= 0.45;
        });
    });

    const fallbackCandidates = buildFallbackContainers(textLines);
    if (fallbackCandidates.length) {
        const merged = [...candidates];
        fallbackCandidates.forEach(candidate => {
            const duplicated = merged.some(existing =>
                bboxContains(existing.bbox, candidate.bbox, 8) ||
                bboxContains(candidate.bbox, existing.bbox, 8)
            );
            if (!duplicated) {
                merged.push(candidate);
            }
        });
        candidates = merged;
    }

    if (sourceImage) {
        return createPanelDebugImageContext(sourceImage)
            .then(imageContext => candidates.filter(candidate => {
                if (candidate?.generatedBy !== 'text_container_restore_fallback' ||
                    String(candidate?.semanticType || '').toLowerCase() !== 'price_badge') {
                    return true;
                }
                const visual = detectSyntheticPriceBadgeVisual(imageContext, candidate.bbox);
                return visual.valid;
            }))
            .catch(() => candidates);
    }

    return candidates;
}

function findCandidateContainers(item, textLines = []) {
    return prepareTextContainerCandidates(item, textLines);
}

function isLargeDisplayLine(line) {
    const text = String(line?.textContent || '').trim();
    if (!text || !Array.isArray(line?.bbox) || line.bbox.length !== 4) return false;
    const [ymin, xmin, ymax, xmax] = line.bbox.map(Number);
    const width = xmax - xmin;
    const height = ymax - ymin;
    return height >= 34 ||
        width >= 115 ||
        /^[A-Z0-9\s&]+$/.test(text) && /[A-Z0-9]/.test(text) && text.length <= 24;
}

function isPriceLine(line) {
    return /^\$?\d+(?:\.\d{1,2})?$/.test(String(line?.textContent || '').trim());
}

function bboxContains(outer, inner, tolerance = 0) {
    if (!Array.isArray(outer) || !Array.isArray(inner)) return false;
    return Number(outer[0]) <= Number(inner[0]) + tolerance &&
        Number(outer[1]) <= Number(inner[1]) + tolerance &&
        Number(outer[2]) >= Number(inner[2]) - tolerance &&
        Number(outer[3]) >= Number(inner[3]) - tolerance;
}

function makeSyntheticPanel(textLines, index, allLines = []) {
    const boxes = textLines.map(line => line.bbox).filter(bbox => Array.isArray(bbox) && bbox.length === 4);
    if (!boxes.length) return null;
    const ymin = Math.min(...boxes.map(bbox => Number(bbox[0])));
    const xmin = Math.min(...boxes.map(bbox => Number(bbox[1])));
    const ymax = Math.max(...boxes.map(bbox => Number(bbox[2])));
    const xmax = Math.max(...boxes.map(bbox => Number(bbox[3])));
    const width = xmax - xmin;
    const height = ymax - ymin;
    const centerY = (ymin + ymax) / 2;
    const nearbySmallText = allLines.filter(line => {
        if (!Array.isArray(line?.bbox) || line.bbox.length !== 4 || textLines.includes(line) || isPriceLine(line)) return false;
        const [ly1, lx1, ly2, lx2] = line.bbox.map(Number);
        const lineCenterY = (ly1 + ly2) / 2;
        return Math.abs(lineCenterY - centerY) <= 95 &&
            lx2 >= xmin - 45 &&
            lx1 <= xmax + 240;
    });
    const nearbyPrice = allLines.some(line => {
        if (!isPriceLine(line) || !Array.isArray(line?.bbox) || line.bbox.length !== 4) return false;
        const [ly1, lx1, ly2, lx2] = line.bbox.map(Number);
        const lineCenterY = (ly1 + ly2) / 2;
        return Math.abs(lineCenterY - centerY) <= 100 &&
            lx2 >= xmin - 50 &&
            lx1 <= xmax + 260;
    });
    if (!nearbySmallText.length && !nearbyPrice) return null;
    const allPanelBoxes = boxes.concat(nearbySmallText.map(line => line.bbox));
    const panelYmin = Math.min(...allPanelBoxes.map(bbox => Number(bbox[0])));
    const panelXmin = Math.min(...allPanelBoxes.map(bbox => Number(bbox[1])));
    const panelYmax = Math.max(...allPanelBoxes.map(bbox => Number(bbox[2])));
    const panelXmax = Math.max(...allPanelBoxes.map(bbox => Number(bbox[3])));
    const panelWidth = panelXmax - panelXmin;
    const panelHeight = panelYmax - panelYmin;
    const padX = clamp(panelWidth * 0.36, 48, 160);
    const padY = clamp(panelHeight * 0.54, 28, 72);
    const bbox = clampBbox([
        panelYmin - padY,
        panelXmin - padX,
        panelYmax + padY,
        panelXmax + padX
    ]);
    const finalPanelWidth = bbox[3] - bbox[1];
    const finalPanelHeight = bbox[2] - bbox[0];
    if (finalPanelWidth < 120 || finalPanelHeight < 58 || finalPanelWidth > 860 || finalPanelHeight > 280) return null;

    return {
        id: `synthetic_text_panel_${index}`,
        name: '文本底板',
        semanticType: 'shape_panel',
        designRole: 'local_panel',
        renderMode: 'vector_shape',
        generatedBy: 'text_container_restore_fallback',
        bbox
    };
}

function hasNearbyMenuDisplayText(line, allLines = []) {
    if (!Array.isArray(line?.bbox) || line.bbox.length !== 4) return false;
    const [ymin, xmin, ymax, xmax] = line.bbox.map(Number);
    const centerY = (ymin + ymax) / 2;
    const centerX = (xmin + xmax) / 2;
    return allLines.some(candidate => {
        if (candidate === line || isPriceLine(candidate) || !isLargeDisplayLine(candidate)) return false;
        const text = String(candidate?.textContent || '').trim();
        if (!/[A-Z]/.test(text) || text.length <= 2) return false;
        if (!Array.isArray(candidate?.bbox) || candidate.bbox.length !== 4) return false;
        const [cy1, cx1, cy2, cx2] = candidate.bbox.map(Number);
        const candidateCenterY = (cy1 + cy2) / 2;
        const candidateCenterX = (cx1 + cx2) / 2;
        return Math.abs(candidateCenterY - centerY) <= 100 &&
            Math.abs(candidateCenterX - centerX) <= 260 &&
            cx2 >= xmin - 260 &&
            cx1 <= xmax + 260;
    });
}

function makeSyntheticPriceBadge(line, index, allLines = []) {
    if (!Array.isArray(line?.bbox) || line.bbox.length !== 4) return null;
    if (!hasNearbyMenuDisplayText(line, allLines)) return null;
    const [ymin, xmin, ymax, xmax] = line.bbox.map(Number);
    const width = xmax - xmin;
    const height = ymax - ymin;
    const size = clamp(Math.max(width, height) * 2.15, 42, 74);
    const cy = (ymin + ymax) / 2;
    const cx = (xmin + xmax) / 2;
    return {
        id: `synthetic_price_badge_${index}`,
        name: '价格圆底板',
        semanticType: 'price_badge',
        designRole: 'price_badge',
        renderMode: 'vector_shape',
        generatedBy: 'text_container_restore_fallback',
        bbox: clampBbox([cy - size / 2, cx - size / 2, cy + size / 2, cx + size / 2])
    };
}

function buildFallbackContainers(textLines = []) {
    const normalized = textLines.filter(line => Array.isArray(line?.bbox) && line.bbox.length === 4);
    if (!normalized.length) return [];

    const additions = [];
    normalized
        .filter(isPriceLine)
        .forEach((line, index) => {
            const badge = makeSyntheticPriceBadge(line, index, normalized);
            if (badge) additions.push(badge);
        });

    const displayLines = normalized
        .filter(line => isLargeDisplayLine(line) && !isPriceLine(line))
        .sort((a, b) => Number(a.bbox[0]) - Number(b.bbox[0]));
    const groups = [];
    for (const line of displayLines) {
        const [ymin, xmin, ymax, xmax] = line.bbox.map(Number);
        const centerY = (ymin + ymax) / 2;
        const centerX = (xmin + xmax) / 2;
        let group = groups.find(candidate => {
            const gapY = Math.abs(candidate.centerY - centerY);
            const overlapsX = xmax >= candidate.xmin - 170 && xmin <= candidate.xmax + 170;
            return gapY <= 105 && overlapsX;
        });
        if (!group) {
            group = { lines: [], centerY, centerX, xmin, xmax };
            groups.push(group);
        }
        group.lines.push(line);
        group.centerY = group.lines.reduce((sum, item) => sum + (Number(item.bbox[0]) + Number(item.bbox[2])) / 2, 0) / group.lines.length;
        group.centerX = group.lines.reduce((sum, item) => sum + (Number(item.bbox[1]) + Number(item.bbox[3])) / 2, 0) / group.lines.length;
        group.xmin = Math.min(group.xmin, xmin);
        group.xmax = Math.max(group.xmax, xmax);
    }

    groups
        .filter(group => group.lines.length >= 2)
        .forEach((group, index) => {
            const panel = makeSyntheticPanel(group.lines, index, normalized);
            if (panel && !additions.some(existing => bboxContains(existing.bbox, panel.bbox, 8) || bboxContains(panel.bbox, existing.bbox, 8))) {
                additions.push(panel);
            }
        });

    return additions.slice(0, 8);
}

function findRestoredContainerShape(parentId, bbox) {
    const key = Array.isArray(bbox) ? bbox.map(value => Math.round(Number(value))).join(',') : '';
    if (!key) return null;
    for (const [itemId, item] of state.workbenchItems.entries()) {
        if (item?.type !== 'shape' || item.parentId !== parentId) continue;
        const el = item.el;
        const isConnected = !!el?.isConnected;
        const width = el ? parseFloat(el.style.width || '0') : 0;
        const height = el ? parseFloat(el.style.height || '0') : 0;
        const visible = isConnected && el.style.display !== 'none' && width > 0 && height > 0;
        if (!visible) {
            state.workbenchItems.delete(itemId);
            continue;
        }
        const existingKey = Array.isArray(item.originalBbox)
            ? item.originalBbox.map(value => Math.round(Number(value))).join(',')
            : '';
        if (existingKey === key) return item;
    }
    return null;
}

function hasRestoredContainerShape(parentId, bbox) {
    return !!findRestoredContainerShape(parentId, bbox);
}

export async function restoreTextContainerShapes({
    item,
    itemId,
    textLines = [],
    baseX = 0,
    baseY = 0,
    parentWidth = 300,
    parentHeight = 300,
    sourceImage = null,
    zIndexBase = 0
}) {
    let candidates = await prepareTextContainerCandidates(item, textLines, sourceImage);
    if (!candidates.length) {
        publishPanelRefineDiagnostics(item, itemId, [{
            layerName: 'none',
            bbox: null,
            shapeType: 'none',
            fillColor: null,
            status: 'no_candidates',
            reasons: ['no_semantic_container_or_text_fallback'],
            metrics: {
                width: null,
                height: null,
                overlappingTextCount: Array.isArray(textLines) ? textLines.length : 0,
                coverage: null,
                topSlant: null,
                bottomSlant: null,
                strongestSlant: null
            },
            candidatePolygon: null
        }]);
        return 0;
    }

    const { restoreShapeToWorkbench } = await import('./workbench/shapes.js');
    let imageContext = null;
    if (sourceImage) {
        try {
            imageContext = await createPanelDebugImageContext(sourceImage);
        } catch (error) {
            console.warn('[TextContainerRestore] Failed to prepare panel diagnostics:', error.message || error);
        }
    }
    const diagnostics = [];
    let restored = 0;
    for (const [index, layer] of candidates.entries()) {
        const width = layer.bbox[3] - layer.bbox[1];
        const height = layer.bbox[2] - layer.bbox[0];
        const semantic = String(layer.semanticType || layer.designRole || '').toLowerCase();
        const shapeType = semantic.includes('price') || Math.abs(width - height) / Math.max(width, height) < 0.2
            ? 'ellipse'
            : 'rect';
        const rect = bboxToWorkbenchRect(layer.bbox, baseX, baseY, parentWidth, parentHeight);
        const fillColor = await sampleBboxFillColor(sourceImage, layer.bbox, imageContext);
        const diagnostic = safeDiagnosePanelRefineCandidate({
            layer,
            shapeType,
            textLines,
            fillColor,
            imageContext,
            siblingPanels: candidates
        });
        const clipPath = diagnostic.status === 'candidate'
            ? polygonToClipPath(diagnostic.candidatePolygon)
            : null;
        const existingShape = findRestoredContainerShape(itemId, layer.bbox);
        if (existingShape) {
            applyShapeClipPath(existingShape, clipPath);
            diagnostics.push({
                ...diagnostic,
                status: diagnostic.status === 'candidate' ? 'applied_existing' : 'skipped_existing',
                appliedClipPath: clipPath
            });
            continue;
        }
        restoreShapeToWorkbench({
            id: `shape-${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${index}`,
            shapeType,
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height,
            zIndex: Math.max(0, zIndexBase - 1),
            fillColor,
            borderWidth: 0,
            borderColor: 'transparent',
            borderRadius: shapeType === 'rect' ? Math.min(18, Math.max(6, rect.height * 0.12)) : undefined,
            clipPath,
            parentId: itemId,
            layerName: layer.name || 'text-container',
            originalBbox: layer.bbox
        });
        restored += 1;
        diagnostics.push({
            ...diagnostic,
            appliedClipPath: clipPath
        });
    }
    publishPanelRefineDiagnostics(item, itemId, diagnostics);
    clearPanelRefinePreviewOverlays(itemId);
    return restored;
}
