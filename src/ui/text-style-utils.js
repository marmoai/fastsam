import { getProxiedUrl } from '../core/utils.js';

const WEB_FONT_ALLOWLIST = [
    'Noto Sans SC',
    'Noto Serif SC',
    'Ma Shan Zheng',
    'Zhi Mang Xing',
    'ZCOOL KuaiLe',
    'ZCOOL QingKe HuangYou',
    'Long Cang',
    'Archivo Black',
    'Anton',
    'Bowlby One SC',
    'Ultra',
    'Alfa Slab One',
    'Black Ops One',
    'Paytone One',
    'Russo One',
    'Bungee',
    'Oswald',
    'Playfair Display',
    'Bebas Neue',
    'League Spartan',
    'Teko',
    'Fjalla One',
    'Kanit',
    'Changa One'
];

const DISPLAY_FONT_CANDIDATES = [
    { family: 'Arial Black', widthBias: 1.20, scoreBias: 0.16 },
    { family: 'Archivo Black', widthBias: 1.16, scoreBias: 0.13 },
    { family: 'Bowlby One SC', widthBias: 1.14, scoreBias: 0.10 },
    { family: 'Ultra', widthBias: 1.12, scoreBias: 0.09 },
    { family: 'Alfa Slab One', widthBias: 1.10, scoreBias: 0.08 },
    { family: 'Anton', widthBias: 1.04, scoreBias: 0.06 },
    { family: 'Bungee', widthBias: 1.01, scoreBias: 0.04 },
    { family: 'Paytone One', widthBias: 1.00, scoreBias: 0.04 },
    { family: 'Black Ops One', widthBias: 0.98, scoreBias: 0.03 },
    { family: 'Russo One', widthBias: 0.96, scoreBias: 0.02 },
    { family: 'Oswald', widthBias: 0.90, scoreBias: 0.01 }
];

const SUBGROUP_DISPLAY_FONT_CANDIDATES = [
    { family: 'League Spartan', widthBias: 1.18, scoreBias: 0.18 },
    { family: 'Arial Black', widthBias: 1.16, scoreBias: 0.17 },
    { family: 'Archivo Black', widthBias: 1.14, scoreBias: 0.15 },
    { family: 'Bowlby One SC', widthBias: 1.12, scoreBias: 0.12 },
    { family: 'Anton', widthBias: 1.08, scoreBias: 0.1 },
    { family: 'Fjalla One', widthBias: 1.06, scoreBias: 0.09 },
    { family: 'Teko', widthBias: 1.03, scoreBias: 0.08 },
    { family: 'Bebas Neue', widthBias: 1.0, scoreBias: 0.07 },
    { family: 'Kanit', widthBias: 0.98, scoreBias: 0.05 },
    { family: 'Changa One', widthBias: 0.97, scoreBias: 0.04 },
    { family: 'Bungee', widthBias: 0.96, scoreBias: 0.03 },
    { family: 'Oswald', widthBias: 0.92, scoreBias: 0.02 }
];

function cssValue(css, ...keys) {
    if (!css || typeof css !== 'object') return null;
    for (const key of keys) {
        if (css[key] !== undefined && css[key] !== null && css[key] !== '') return css[key];
    }
    return null;
}

function normalizeFontFamily(value) {
    if (!value || typeof value !== 'string') return '';
    return value.replace(/['"]/g, '').split(',')[0].trim();
}

function isVisibleColor(value) {
    if (!value || typeof value !== 'string') return false;
    const normalized = value.trim().toLowerCase();
    return normalized !== 'transparent' && normalized !== 'rgba(0, 0, 0, 0)' && normalized !== 'rgba(0,0,0,0)';
}

function parseFontStyle(styleStr = '') {
    if (!styleStr || typeof styleStr !== 'string') {
        return {};
    }
    const parts = styleStr.split(',').map(part => part.trim()).filter(Boolean);
    const parsed = {};
    if (parts[0]) parsed.fontFamily = normalizeFontFamily(parts[0]);
    const colorPart = parts.find(part => /^#|^rgb|^hsl/i.test(part));
    if (colorPart) parsed.fontColor = colorPart;
    return parsed;
}

function parsePixelValue(value) {
    if (value === undefined || value === null || value === '') return null;
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value !== 'string') return null;
    const match = value.match(/-?\d+(\.\d+)?/);
    return match ? Number(match[0]) : null;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function sameBbox(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === 4 &&
        b.length === 4 &&
        a.every((value, index) => Math.abs(Number(value) - Number(b[index])) < 0.01);
}

function bboxCenterPoint(bbox) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return null;
    return {
        y: (Number(bbox[0]) + Number(bbox[2])) / 2,
        x: (Number(bbox[1]) + Number(bbox[3])) / 2
    };
}

function bboxContainsPoint(bbox, point) {
    if (!Array.isArray(bbox) || bbox.length !== 4 || !point) return false;
    return point.y >= Number(bbox[0]) &&
        point.y <= Number(bbox[2]) &&
        point.x >= Number(bbox[1]) &&
        point.x <= Number(bbox[3]);
}

function bboxIntersectionArea(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return 0;
    const ymin = Math.max(Number(a[0]), Number(b[0]));
    const xmin = Math.max(Number(a[1]), Number(b[1]));
    const ymax = Math.min(Number(a[2]), Number(b[2]));
    const xmax = Math.min(Number(a[3]), Number(b[3]));
    return ymax > ymin && xmax > xmin ? (ymax - ymin) * (xmax - xmin) : 0;
}

function bboxArea(bbox) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return 0;
    return Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
}

function resolveMatchedPriceBadge(lineBbox, containerCandidates = []) {
    if (!Array.isArray(lineBbox) || lineBbox.length !== 4 || !Array.isArray(containerCandidates)) return null;
    const priceBadges = containerCandidates.filter(candidate => {
        const semantic = String(candidate?.semanticType || candidate?.designRole || '').toLowerCase();
        return semantic.includes('price') && Array.isArray(candidate?.bbox) && candidate.bbox.length === 4;
    });
    if (!priceBadges.length) return null;

    const lineCenter = bboxCenterPoint(lineBbox);
    const lineArea = Math.max(1, bboxArea(lineBbox));
    let best = null;

    for (const badge of priceBadges) {
        const badgeBbox = badge.bbox.map(Number);
        const badgeCenter = bboxCenterPoint(badgeBbox);
        if (!badgeCenter || !lineCenter) continue;
        const badgeWidth = Math.max(1, badgeBbox[3] - badgeBbox[1]);
        const badgeHeight = Math.max(1, badgeBbox[2] - badgeBbox[0]);
        const badgeSize = Math.max(badgeWidth, badgeHeight);
        const overlapArea = bboxIntersectionArea(lineBbox, badgeBbox);
        const overlapRatio = overlapArea / lineArea;
        const containsCenter = bboxContainsPoint(badgeBbox, lineCenter);
        const distance = Math.hypot(lineCenter.x - badgeCenter.x, lineCenter.y - badgeCenter.y);
        const normalizedDistance = distance / badgeSize;
        const score = overlapRatio * 4 + (containsCenter ? 2.5 : 0) - normalizedDistance * 0.85;

        if (
            overlapRatio > 0.08 ||
            containsCenter ||
            normalizedDistance <= 0.95
        ) {
            if (!best || score > best.score) {
                best = {
                    bbox: badgeBbox,
                    score,
                    overlapRatio,
                    containsCenter,
                    normalizedDistance
                };
            }
        }
    }

    return best;
}

function bboxToRect(bbox, baseX, baseY, parentWidth, parentHeight, minWidth = 20, minHeight = 10) {
    const safeBbox = Array.isArray(bbox) && bbox.length === 4 ? bbox : [0, 0, 1000, 1000];
    const [ymin, xmin, ymax, xmax] = safeBbox.map(value => Number(value));
    return {
        left: baseX + (xmin / 1000) * parentWidth,
        top: baseY + (ymin / 1000) * parentHeight,
        width: Math.max(((xmax - xmin) / 1000) * parentWidth, minWidth),
        height: Math.max(((ymax - ymin) / 1000) * parentHeight, minHeight)
    };
}

function expandNormalizedBbox(bbox, ratioX = 0.12, ratioY = 0.16) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return bbox;
    const [ymin, xmin, ymax, xmax] = bbox.map(Number);
    const width = Math.max(1, xmax - xmin);
    const height = Math.max(1, ymax - ymin);
    return [
        clamp(ymin - height * ratioY, 0, 1000),
        clamp(xmin - width * ratioX, 0, 1000),
        clamp(ymax + height * ratioY, 0, 1000),
        clamp(xmax + width * ratioX, 0, 1000)
    ];
}

function rgbToHex(r, g, b) {
    return `#${[r, g, b].map(value => clamp(Math.round(value), 0, 255).toString(16).padStart(2, '0')).join('')}`;
}

function parseCssColorToRgb(value) {
    if (!value || typeof value !== 'string') return null;
    const color = value.trim();
    const hex3 = color.match(/^#([0-9a-f]{3})$/i);
    if (hex3) {
        return hex3[1].split('').map(char => parseInt(char + char, 16));
    }
    const hex6 = color.match(/^#([0-9a-f]{6})$/i);
    if (hex6) {
        return [
            parseInt(hex6[1].slice(0, 2), 16),
            parseInt(hex6[1].slice(2, 4), 16),
            parseInt(hex6[1].slice(4, 6), 16)
        ];
    }
    const rgb = color.match(/^rgba?\(([^)]+)\)$/i);
    if (rgb) {
        const parts = rgb[1].split(',').map(part => Number(part.trim()));
        if (parts.length >= 3 && parts.slice(0, 3).every(Number.isFinite)) {
            return parts.slice(0, 3);
        }
    }
    return null;
}

function colorDistance(a, b) {
    return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]);
}

function luminance(pixel) {
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
}

function averageClusterColor(cluster) {
    if (!cluster || !cluster.weight) return null;
    return [
        cluster.sum[0] / cluster.weight,
        cluster.sum[1] / cluster.weight,
        cluster.sum[2] / cluster.weight
    ];
}

function chooseBackgroundFromRegionClusters(clusters, edgeBackground = null) {
    if (!clusters?.size) return edgeBackground;
    const sorted = Array.from(clusters.values())
        .map(cluster => ({
            cluster,
            color: averageClusterColor(cluster),
            edgeWeight: cluster.edgeWeight || 0
        }))
        .filter(entry => entry.color)
        .sort((a, b) => b.cluster.weight - a.cluster.weight);
    if (!sorted.length) return edgeBackground;

    const dominant = sorted[0].color;
    if (!edgeBackground) return dominant;

    const dominantWeight = sorted[0].cluster.weight;
    const totalWeight = sorted.reduce((sum, entry) => sum + entry.cluster.weight, 0);
    const dominantRatio = totalWeight ? dominantWeight / totalWeight : 0;
    const edgeDistance = colorDistance(dominant, edgeBackground);
    const edgeLike = sorted.find(entry => colorDistance(entry.color, edgeBackground) <= 42);
    const edgeSupportRatio = edgeLike ? edgeLike.edgeWeight / Math.max(1, dominantWeight) : 0;

    if (edgeDistance <= 58) {
        return dominant;
    }

    // If a tight display-text bbox is mostly glyph color, the dominant region
    // cluster can be foreground. Require edge support before trusting it.
    if (dominantRatio >= 0.28 && sorted[0].edgeWeight >= Math.max(4, (edgeLike?.edgeWeight || 0) * 0.75)) {
        return dominant;
    }

    if (edgeLike && edgeSupportRatio >= 0.08) {
        return edgeLike.color;
    }

    return edgeBackground;
}

function chooseTextColorFromClusters(clusters, background, fallback = null) {
    if (!clusters?.size || !background) return fallback;
    const totalWeight = Array.from(clusters.values()).reduce((sum, cluster) => sum + cluster.weight, 0);
    if (fallback && colorDistance(fallback, background) >= 58) {
        const fallbackLum = luminance(fallback);
        const backgroundLum = luminance(background);
        const fallbackWeightRatio = Array.from(clusters.values())
            .filter(cluster => {
                const color = averageClusterColor(cluster);
                return color && colorDistance(color, fallback) <= 42;
            })
            .reduce((sum, cluster) => sum + cluster.weight, 0) / Math.max(1, totalWeight);
        if (fallbackWeightRatio <= 0.58 || Math.abs(fallbackLum - backgroundLum) >= 70) {
            return fallback;
        }
    }

    const candidates = Array.from(clusters.values())
        .map(cluster => {
            const color = averageClusterColor(cluster);
            if (!color) return null;
            const contrast = colorDistance(color, background);
            if (contrast < 58) return null;
            const weightRatio = cluster.weight / Math.max(1, totalWeight);
            if (weightRatio > 0.72) return null;
            const avgContrast = cluster.contrast ? cluster.contrast / Math.max(1, cluster.weight) : contrast;
            const bgLum = luminance(background);
            const fgLum = luminance(color);
            const darkOnLightBonus = bgLum >= 165 && fgLum <= bgLum - 48 ? 80 : 0;
            const lightOnDarkBonus = bgLum <= 145 && fgLum >= bgLum + 48 ? 80 : 0;
            const score = Math.sqrt(cluster.weight) +
                avgContrast * 0.7 +
                contrast * 0.35 +
                darkOnLightBonus +
                lightOnDarkBonus -
                Math.max(0, weightRatio - 0.46) * 90;
            return { color, score, contrast };
        })
        .filter(Boolean)
        .sort((a, b) => b.score - a.score || b.contrast - a.contrast);

    return candidates[0]?.color || fallback;
}

function quantizeColor(pixel, step = 24) {
    return pixel.map(value => Math.round(value / step) * step).join(',');
}

function centerWeight(x, y, width, height) {
    const dx = Math.abs((x + 0.5) / width - 0.5);
    const dy = Math.abs((y + 0.5) / height - 0.5);
    return 1 + Math.max(0, 0.5 - Math.max(dx, dy));
}

function averageColor(samples) {
    if (!samples.length) return null;
    const total = samples.reduce((acc, sample) => {
        acc[0] += sample[0];
        acc[1] += sample[1];
        acc[2] += sample[2];
        return acc;
    }, [0, 0, 0]);
    return total.map(value => value / samples.length);
}

function median(values) {
    const nums = values.filter(value => Number.isFinite(value)).sort((a, b) => a - b);
    if (!nums.length) return null;
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
}

function percentile(values, ratio) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = clamp(Math.floor((sorted.length - 1) * ratio), 0, sorted.length - 1);
    return sorted[index];
}

function estimateFontWeightFromInk({ inkAreaRatio, textContent, glyphHeightPx }) {
    const text = String(textContent || '').trim();
    const isUpperDisplay = /^[A-Z0-9\s.,:;!?%+-]+$/.test(text) && /[A-Z0-9]/.test(text);
    const displayBonus = isUpperDisplay && glyphHeightPx >= 18 ? 0.035 : 0;
    const score = inkAreaRatio + displayBonus;

    if (score >= 0.42) return '900';
    if (score >= 0.34) return '800';
    if (score >= 0.27) return '700';
    if (score >= 0.21) return '600';
    if (score >= 0.15) return '500';
    return '400';
}

function estimateLetterSpacingFromInk({ inkWidthPx, glyphHeightPx, textContent }) {
    const text = String(textContent || '').replace(/\s+/g, '');
    if (text.length < 3 || !glyphHeightPx) return null;

    const widthPerChar = inkWidthPx / text.length;
    const isUpperDisplay = /^[A-Z0-9]+$/.test(text);
    if (isUpperDisplay && widthPerChar > glyphHeightPx * 0.68) return '0.04em';
    if (isUpperDisplay && widthPerChar > glyphHeightPx * 0.56) return '0.02em';
    if (widthPerChar < glyphHeightPx * 0.34) return '-0.02em';
    return null;
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

async function measureTextVisualStyle(sourceImage, bbox, textContent = '') {
    if (!sourceImage || !Array.isArray(bbox) || bbox.length !== 4) return null;

    try {
        const img = await loadImage(sourceImage);
        const [ymin, xmin, ymax, xmax] = bbox.map(value => Number(value));
        const sx = clamp((xmin / 1000) * img.naturalWidth, 0, img.naturalWidth - 1);
        const sy = clamp((ymin / 1000) * img.naturalHeight, 0, img.naturalHeight - 1);
        const sw = clamp(((xmax - xmin) / 1000) * img.naturalWidth, 1, img.naturalWidth - sx);
        const sh = clamp(((ymax - ymin) / 1000) * img.naturalHeight, 1, img.naturalHeight - sy);
        const scale = Math.min(1, 240 / sw, 180 / sh);
        const sampleW = Math.max(12, Math.round(sw * scale));
        const sampleH = Math.max(10, Math.round(sh * scale));

        const canvas = document.createElement('canvas');
        canvas.width = sampleW;
        canvas.height = sampleH;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sampleW, sampleH);
        const data = ctx.getImageData(0, 0, sampleW, sampleH).data;

        const border = [];
        const allPixels = [];
        const borderSize = Math.max(1, Math.floor(Math.min(sampleW, sampleH) * 0.1));
        for (let y = 0; y < sampleH; y++) {
            for (let x = 0; x < sampleW; x++) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                allPixels.push(pixel);
                if (x < borderSize || y < borderSize || x >= sampleW - borderSize || y >= sampleH - borderSize) {
                    border.push(pixel);
                }
            }
        }

        const background = averageColor(border.length >= 8 ? border : allPixels);
        if (!background) return null;

        const contrasts = [];
        for (const pixel of allPixels) {
            contrasts.push(colorDistance(pixel, background));
        }
        const adaptiveThreshold = clamp(percentile(contrasts, 0.86) * 0.58, 30, 86);

        const clusters = new Map();
        const maskPixels = [];
        let minX = sampleW;
        let minY = sampleH;
        let maxX = -1;
        let maxY = -1;
        let maskCount = 0;

        for (let y = 0; y < sampleH; y++) {
            for (let x = 0; x < sampleW; x++) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                const contrast = colorDistance(pixel, background);
                if (contrast < adaptiveThreshold) continue;

                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
                maskCount += 1;
                maskPixels.push({ pixel, contrast, x, y });

                const key = quantizeColor(pixel, 20);
                const weight = centerWeight(x, y, sampleW, sampleH) * Math.max(1, contrast / 64);
                const cluster = clusters.get(key) || { weight: 0, sum: [0, 0, 0] };
                cluster.weight += weight;
                cluster.sum[0] += pixel[0] * weight;
                cluster.sum[1] += pixel[1] * weight;
                cluster.sum[2] += pixel[2] * weight;
                clusters.set(key, cluster);
            }
        }

        const minMaskPixels = Math.max(4, sampleW * sampleH * 0.003);
        if (maskCount < minMaskPixels || maxX < minX || maxY < minY) return null;

        const inkWidth = Math.max(1, maxX - minX + 1);
        const inkHeight = Math.max(1, maxY - minY + 1);
        const inkAreaRatio = maskCount / Math.max(1, inkWidth * inkHeight);
        const refinedClusters = new Map();
        for (const { pixel, contrast, x, y } of maskPixels) {
            const key = quantizeColor(pixel, 14);
            const weight = centerWeight(x, y, sampleW, sampleH) * Math.max(1, contrast / 64);
            const cluster = refinedClusters.get(key) || { weight: 0, contrast: 0, sum: [0, 0, 0] };
            cluster.weight += weight;
            cluster.contrast += contrast * weight;
            cluster.sum[0] += pixel[0] * weight;
            cluster.sum[1] += pixel[1] * weight;
            cluster.sum[2] += pixel[2] * weight;
            refinedClusters.set(key, cluster);
        }
        const dominant = Array.from(refinedClusters.values()).sort((a, b) => {
            const scoreA = a.weight * Math.max(1, a.contrast / Math.max(1, a.weight));
            const scoreB = b.weight * Math.max(1, b.contrast / Math.max(1, b.weight));
            return scoreB - scoreA;
        })[0] ||
            Array.from(clusters.values()).sort((a, b) => b.weight - a.weight)[0];
        const fontColor = dominant ? rgbToHex(
            dominant.sum[0] / dominant.weight,
            dominant.sum[1] / dominant.weight,
            dominant.sum[2] / dominant.weight
        ) : null;

        return {
            fontColor,
            colorDiagnostics: {
                background: rgbToHex(background[0], background[1], background[2]),
                sampledFontColor: fontColor,
                maskCount,
                maskCoverage: maskCount / Math.max(1, sampleW * sampleH),
                sampleSize: [sampleW, sampleH]
            },
            fontWeight: estimateFontWeightFromInk({
                inkAreaRatio,
                textContent,
                glyphHeightPx: inkHeight
            }),
            inkBounds: {
                leftRatio: minX / sampleW,
                topRatio: minY / sampleH,
                widthRatio: inkWidth / sampleW,
                heightRatio: inkHeight / sampleH
            },
            inkAreaRatio,
            maskCoverage: maskCount / Math.max(1, sampleW * sampleH),
            letterSpacingHint: estimateLetterSpacingFromInk({
                inkWidthPx: inkWidth,
                glyphHeightPx: inkHeight,
                textContent
            })
        };
    } catch (error) {
        console.warn('[TextStyle] Visual text measurement failed:', error.message || error);
        return null;
    }
}

async function sampleForegroundColor(sourceImage, bbox, textContent = '') {
    if (!sourceImage || !Array.isArray(bbox) || bbox.length !== 4) return null;

    try {
        const img = await loadImage(sourceImage);
        const [ymin, xmin, ymax, xmax] = bbox.map(value => Number(value));
        const sx = clamp((xmin / 1000) * img.naturalWidth, 0, img.naturalWidth - 1);
        const sy = clamp((ymin / 1000) * img.naturalHeight, 0, img.naturalHeight - 1);
        const sw = clamp(((xmax - xmin) / 1000) * img.naturalWidth, 1, img.naturalWidth - sx);
        const sh = clamp(((ymax - ymin) / 1000) * img.naturalHeight, 1, img.naturalHeight - sy);
        const sampleW = Math.min(120, Math.max(12, Math.round(sw)));
        const sampleH = Math.min(120, Math.max(12, Math.round(sh)));

        const canvas = document.createElement('canvas');
        canvas.width = sampleW;
        canvas.height = sampleH;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sampleW, sampleH);
        const data = ctx.getImageData(0, 0, sampleW, sampleH).data;

        const border = [];
        const all = [];
        for (let y = 0; y < sampleH; y++) {
            for (let x = 0; x < sampleW; x++) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                all.push(pixel);
                const borderSize = Math.max(1, Math.floor(Math.min(sampleW, sampleH) * 0.12));
                if (x < borderSize || y < borderSize || x >= sampleW - borderSize || y >= sampleH - borderSize) {
                    border.push(pixel);
                }
            }
        }

        const background = averageColor(border.length ? border : all);
        if (!background) return null;

        const clusters = new Map();
        for (let y = 0; y < sampleH; y++) {
            for (let x = 0; x < sampleW; x++) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                const contrast = colorDistance(pixel, background);
                if (contrast < 48) continue;
                const key = quantizeColor(pixel);
                const weight = centerWeight(x, y, sampleW, sampleH);
                const cluster = clusters.get(key) || { weight: 0, contrast: 0, sum: [0, 0, 0] };
                cluster.weight += weight;
                cluster.contrast += contrast * weight;
                cluster.sum[0] += pixel[0] * weight;
                cluster.sum[1] += pixel[1] * weight;
                cluster.sum[2] += pixel[2] * weight;
                clusters.set(key, cluster);
            }
        }

        if (clusters.size === 0) return null;

        const dominant = Array.from(clusters.values())
            .sort((a, b) => b.weight - a.weight || (b.contrast / b.weight) - (a.contrast / a.weight))[0];

        return rgbToHex(
            dominant.sum[0] / dominant.weight,
            dominant.sum[1] / dominant.weight,
            dominant.sum[2] / dominant.weight
        );
    } catch (error) {
        console.warn('[TextStyle] Foreground color sampling failed:', error.message || error);
        return null;
    }
}

async function extractSourceGlyphProfile(sourceImage, bbox, group = null) {
    if (!sourceImage || !Array.isArray(bbox) || bbox.length !== 4) return null;

    try {
        const img = await loadImage(sourceImage);
        const [ymin, xmin, ymax, xmax] = bbox.map(value => Number(value));
        const sx = clamp((xmin / 1000) * img.naturalWidth, 0, img.naturalWidth - 1);
        const sy = clamp((ymin / 1000) * img.naturalHeight, 0, img.naturalHeight - 1);
        const sw = clamp(((xmax - xmin) / 1000) * img.naturalWidth, 1, img.naturalWidth - sx);
        const sh = clamp(((ymax - ymin) / 1000) * img.naturalHeight, 1, img.naturalHeight - sy);
        const scale = Math.min(1, 180 / sw, 120 / sh);
        const sampleW = Math.max(18, Math.round(sw * scale));
        const sampleH = Math.max(18, Math.round(sh * scale));

        const canvas = document.createElement('canvas');
        canvas.width = sampleW;
        canvas.height = sampleH;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return null;
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sampleW, sampleH);
        const data = ctx.getImageData(0, 0, sampleW, sampleH).data;

        let cropMinX = 0;
        let cropMinY = 0;
        let cropMaxX = sampleW - 1;
        let cropMaxY = sampleH - 1;
        if (group) {
            cropMinX = clamp(Math.floor(group.leftRatio * sampleW), 0, sampleW - 1);
            cropMinY = clamp(Math.floor(group.topRatio * sampleH), 0, sampleH - 1);
            cropMaxX = clamp(Math.ceil((group.leftRatio + group.widthRatio) * sampleW) - 1, cropMinX, sampleW - 1);
            cropMaxY = clamp(Math.ceil((group.topRatio + group.heightRatio) * sampleH) - 1, cropMinY, sampleH - 1);
        }

        const border = [];
        const pixels = [];
        const borderPad = Math.max(1, Math.floor(Math.min(cropMaxX - cropMinX + 1, cropMaxY - cropMinY + 1) * 0.12));
        for (let y = cropMinY; y <= cropMaxY; y += 1) {
            for (let x = cropMinX; x <= cropMaxX; x += 1) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                pixels.push({ x: x - cropMinX, y: y - cropMinY, pixel });
                if (
                    x - cropMinX < borderPad ||
                    y - cropMinY < borderPad ||
                    cropMaxX - x < borderPad ||
                    cropMaxY - y < borderPad
                ) {
                    border.push(pixel);
                }
            }
        }
        if (!pixels.length) return null;

        const background = averageColor(border.length >= 8 ? border : pixels.map(entry => entry.pixel));
        if (!background) return null;

        const contrasts = pixels.map(entry => colorDistance(entry.pixel, background));
        const adaptiveThreshold = clamp(percentile(contrasts, 0.84) * 0.56, 26, 84);
        const cropW = cropMaxX - cropMinX + 1;
        const cropH = cropMaxY - cropMinY + 1;
        const mask = Array(cropW * cropH).fill(0);
        let maskCount = 0;
        let minX = cropW;
        let minY = cropH;
        let maxX = -1;
        let maxY = -1;

        for (const entry of pixels) {
            const contrast = colorDistance(entry.pixel, background);
            if (contrast < adaptiveThreshold) continue;
            const idx = entry.y * cropW + entry.x;
            mask[idx] = 1;
            maskCount += 1;
            minX = Math.min(minX, entry.x);
            minY = Math.min(minY, entry.y);
            maxX = Math.max(maxX, entry.x);
            maxY = Math.max(maxY, entry.y);
        }

        if (!maskCount || maxX < minX || maxY < minY) return null;

        const inkWidth = Math.max(1, maxX - minX + 1);
        const inkHeight = Math.max(1, maxY - minY + 1);
        const normalizedMask = normalizeBinaryMask(mask, cropW, cropH, 24, 24);
        return {
            widthUtil: inkWidth / Math.max(1, cropW),
            heightUtil: inkHeight / Math.max(1, cropH),
            inkAreaRatio: maskCount / Math.max(1, inkWidth * inkHeight),
            bottomUtil: maxY / Math.max(1, cropH),
            centerYUtil: ((minY + maxY) / 2) / Math.max(1, cropH),
            mask: normalizedMask
        };
    } catch (error) {
        console.warn('[TextStyle] Source glyph profile extraction failed:', error.message || error);
        return null;
    }
}

function normalizeBinaryMask(mask, width, height, outW = 24, outH = 24) {
    const normalized = Array(outW * outH).fill(0);
    for (let oy = 0; oy < outH; oy += 1) {
        for (let ox = 0; ox < outW; ox += 1) {
            const sx0 = Math.floor(ox / outW * width);
            const sx1 = Math.max(sx0 + 1, Math.ceil((ox + 1) / outW * width));
            const sy0 = Math.floor(oy / outH * height);
            const sy1 = Math.max(sy0 + 1, Math.ceil((oy + 1) / outH * height));
            let active = 0;
            let total = 0;
            for (let sy = sy0; sy < sy1; sy += 1) {
                for (let sx = sx0; sx < sx1; sx += 1) {
                    total += 1;
                    active += mask[sy * width + sx] ? 1 : 0;
                }
            }
            normalized[oy * outW + ox] = total > 0 && active / total >= 0.28 ? 1 : 0;
        }
    }
    return normalized;
}

function compareBinaryMasks(targetMask, candidateMask) {
    if (!Array.isArray(targetMask) || !Array.isArray(candidateMask) || targetMask.length !== candidateMask.length) {
        return null;
    }
    let intersection = 0;
    let union = 0;
    let mismatch = 0;
    for (let i = 0; i < targetMask.length; i += 1) {
        const a = targetMask[i] ? 1 : 0;
        const b = candidateMask[i] ? 1 : 0;
        if (a || b) union += 1;
        if (a && b) intersection += 1;
        if (a !== b) mismatch += 1;
    }
    return {
        iou: union > 0 ? intersection / union : 0,
        mismatchRatio: mismatch / Math.max(1, targetMask.length)
    };
}

async function collectNonBadgePriceCohortProfile({
    allTextLines = [],
    containerCandidates = [],
    sourceImage = null
}) {
    const lines = Array.isArray(allTextLines) ? allTextLines : [];
    if (!sourceImage || !lines.length) return null;

    const fallbackDigitGroup = {
        leftRatio: 0.16,
        topRatio: 0.02,
        widthRatio: 0.8,
        heightRatio: 0.96
    };

    const profiles = [];
    for (const line of lines) {
        if (!Array.isArray(line?.bbox) || line.bbox.length !== 4) continue;
        const matchedBadge = resolveMatchedPriceBadge(line.bbox, containerCandidates);
        if (matchedBadge?.bbox) continue;
        const split = getSplitPriceParts(line.textContent || '', line.bbox, matchedBadge);
        if (!split) continue;
        const profile = await extractSourceGlyphProfile(sourceImage, line.bbox, fallbackDigitGroup);
        if (profile) profiles.push(profile);
    }

    if (!profiles.length) return null;
    return {
        widthUtil: median(profiles.map(profile => profile.widthUtil)),
        heightUtil: median(profiles.map(profile => profile.heightUtil)),
        inkAreaRatio: median(profiles.map(profile => profile.inkAreaRatio)),
        bottomUtil: median(profiles.map(profile => profile.bottomUtil)),
        centerYUtil: median(profiles.map(profile => profile.centerYUtil))
    };
}

function boostNonBadgePriceDigitsTypography({
    typography,
    width,
    height,
    text,
    fontStyle,
    letterSpacingValue = '0',
    targetProfile = null
}) {
    if (!typography || !text || !width || !height) return typography;

    const targetHeightUtil = Number(targetProfile?.heightUtil);
    const targetWidthUtil = Number(targetProfile?.widthUtil);
    const renderedHeightUtil = Number(typography?.renderedProfile?.heightUtil);
    const renderedWidthUtil = Number(typography?.renderedProfile?.widthUtil);

    if (!Number.isFinite(targetHeightUtil) || !Number.isFinite(renderedHeightUtil)) return typography;
    if (renderedHeightUtil >= targetHeightUtil * 0.95) return typography;

    const heightScale = targetHeightUtil / Math.max(0.01, renderedHeightUtil);
    const upscale = clamp(heightScale, 1, 1.28);
    if (upscale <= 1.02) return typography;

    const boostedFontSize = typography.fontSize * upscale;
    const measured = measureTextBlock(
        text,
        boostedFontSize,
        typography.family,
        typography.weight,
        fontStyle,
        letterSpacingValue
    );

    const boostedWidthUtil = measured.width / Math.max(1, width);
    const boostedHeightUtil = measured.visualLineHeight / Math.max(1, height);
    const widthOverflowLimit = Number.isFinite(targetWidthUtil)
        ? Math.max(1.08, targetWidthUtil + 0.18)
        : 1.12;
    if (boostedWidthUtil > widthOverflowLimit || boostedHeightUtil > 1.12) return typography;

    const renderedProfile = measureRenderedGlyphProfile({
        text,
        width,
        height,
        fontSize: boostedFontSize,
        fontFamily: typography.family,
        fontWeight: typography.weight,
        fontStyle,
        letterSpacingValue,
        textAlign: typography.textAlign || 'left'
    });
    if (!renderedProfile) return typography;
    if (renderedProfile.widthUtil > widthOverflowLimit || renderedProfile.heightUtil > 1.12) return typography;

    return {
        ...typography,
        fontSize: boostedFontSize,
        lineHeight: Math.max(1, boostedFontSize * 0.88),
        widthUtil: boostedWidthUtil,
        heightUtil: boostedHeightUtil,
        renderedProfile
    };
}

async function detectInlineVisualHierarchySplit(sourceImage, bbox, textContent, matchedPriceBadge = null) {
    if (!shouldTryVisualHierarchySplit(textContent, bbox, matchedPriceBadge) || !sourceImage) return null;

    try {
        const img = await loadImage(sourceImage);
        const [ymin, xmin, ymax, xmax] = bbox.map(value => Number(value));
        const sx = clamp((xmin / 1000) * img.naturalWidth, 0, img.naturalWidth - 1);
        const sy = clamp((ymin / 1000) * img.naturalHeight, 0, img.naturalHeight - 1);
        const sw = clamp(((xmax - xmin) / 1000) * img.naturalWidth, 1, img.naturalWidth - sx);
        const sh = clamp(((ymax - ymin) / 1000) * img.naturalHeight, 1, img.naturalHeight - sy);
        const scale = Math.min(1, 200 / sw, 120 / sh);
        const sampleW = Math.max(24, Math.round(sw * scale));
        const sampleH = Math.max(18, Math.round(sh * scale));

        const canvas = document.createElement('canvas');
        canvas.width = sampleW;
        canvas.height = sampleH;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sampleW, sampleH);
        const data = ctx.getImageData(0, 0, sampleW, sampleH).data;

        const border = [];
        const allPixels = [];
        const borderSize = Math.max(1, Math.floor(Math.min(sampleW, sampleH) * 0.12));
        for (let y = 0; y < sampleH; y += 1) {
            for (let x = 0; x < sampleW; x += 1) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                allPixels.push(pixel);
                if (x < borderSize || y < borderSize || x >= sampleW - borderSize || y >= sampleH - borderSize) {
                    border.push(pixel);
                }
            }
        }

        const background = averageColor(border.length >= 8 ? border : allPixels);
        if (!background) return null;

        const contrasts = [];
        for (const pixel of allPixels) {
            contrasts.push(colorDistance(pixel, background));
        }
        const adaptiveThreshold = clamp(percentile(contrasts, 0.84) * 0.56, 28, 84);

        const columnStats = Array.from({ length: sampleW }, () => ({
            count: 0,
            minY: sampleH,
            maxY: -1
        }));

        for (let y = 0; y < sampleH; y += 1) {
            for (let x = 0; x < sampleW; x += 1) {
                const offset = (y * sampleW + x) * 4;
                if (data[offset + 3] < 24) continue;
                const pixel = [data[offset], data[offset + 1], data[offset + 2]];
                const contrast = colorDistance(pixel, background);
                if (contrast < adaptiveThreshold) continue;
                columnStats[x].count += 1;
                columnStats[x].minY = Math.min(columnStats[x].minY, y);
                columnStats[x].maxY = Math.max(columnStats[x].maxY, y);
            }
        }

        const activeThreshold = Math.max(1, Math.floor(sampleH * 0.08));
        const gapTolerance = Math.max(1, Math.floor(sampleW * 0.03));
        const groups = [];
        let current = null;
        let gap = 0;

        for (let x = 0; x < sampleW; x += 1) {
            const stat = columnStats[x];
            const active = stat.count >= activeThreshold;
            if (active) {
                if (!current) {
                    current = { minX: x, maxX: x, minY: stat.minY, maxY: stat.maxY, count: stat.count };
                } else {
                    current.maxX = x;
                    current.minY = Math.min(current.minY, stat.minY);
                    current.maxY = Math.max(current.maxY, stat.maxY);
                    current.count += stat.count;
                }
                gap = 0;
            } else if (current) {
                gap += 1;
                if (gap <= gapTolerance) continue;
                groups.push(current);
                current = null;
                gap = 0;
            }
        }
        if (current) groups.push(current);

        const filteredGroups = groups
            .map(group => ({
                ...group,
                width: Math.max(1, group.maxX - group.minX + 1),
                height: Math.max(1, group.maxY - group.minY + 1)
            }))
            .filter(group => group.count >= Math.max(8, sampleW * sampleH * 0.004) && group.width >= 2)
            .sort((a, b) => a.minX - b.minX);

        if (filteredGroups.length !== 2) return null;

        const [leftGroup, rightGroup] = filteredGroups;
        const heightRatio = Math.max(filteredGroups[0].height, filteredGroups[1].height) /
            Math.max(1, Math.min(filteredGroups[0].height, filteredGroups[1].height));
        if (heightRatio < 1.2) return null;
        const baselineRatioDiff = Math.abs(leftGroup.maxY - rightGroup.maxY) / Math.max(1, sampleH);
        const centerYDiff = Math.abs(
            ((leftGroup.minY + leftGroup.maxY) / 2) - ((rightGroup.minY + rightGroup.maxY) / 2)
        ) / Math.max(1, sampleH);
        const gapPx = Math.max(0, rightGroup.minX - leftGroup.maxX - 1);
        const gapRatio = gapPx / Math.max(1, sampleW);
        const areaRatio = Math.max(leftGroup.count, rightGroup.count) / Math.max(1, Math.min(leftGroup.count, rightGroup.count));
        if (baselineRatioDiff > 0.26 || centerYDiff > 0.24) return null;
        if (gapRatio > 0.24) return null;

        const candidates = inferTextSplitCandidates(textContent);
        if (!candidates.length) return null;

        const groupWidthSum = filteredGroups[0].width + filteredGroups[1].width;
        let bestCandidate = null;

        for (const candidate of candidates) {
            const leftShare = candidate.left.length / Math.max(1, textContent.length);
            const widthShare = filteredGroups[0].width / Math.max(1, groupWidthSum);
            const leftClass = getCharClass(candidate.left[candidate.left.length - 1]);
            const rightClass = getCharClass(candidate.right[0]);
            const sizeDirectionBonus =
                (filteredGroups[0].height < filteredGroups[1].height && candidate.left.length <= candidate.right.length)
                    ? 0.7
                    : 0;
            const baselineBonus = baselineRatioDiff <= 0.12 ? 0.8 : baselineRatioDiff <= 0.18 ? 0.35 : -0.9;
            const centerYBonus = centerYDiff <= 0.1 ? 0.45 : centerYDiff <= 0.16 ? 0.15 : -0.55;
            const gapBonus = gapRatio >= 0.01 && gapRatio <= 0.1 ? 0.55 : gapRatio <= 0.16 ? 0.15 : -0.4;
            const contrastBonus = heightRatio >= 1.4 ? 0.8 : 0.3;
            const areaBonus = areaRatio >= 1.25 ? 0.35 : 0;
            const classBonus = leftClass !== rightClass ? 0.8 : 0;
            const score = candidate.score
                - Math.abs(leftShare - widthShare) * 1.4
                + sizeDirectionBonus
                + baselineBonus
                + centerYBonus
                + gapBonus
                + contrastBonus
                + areaBonus
                + classBonus;
            if (!bestCandidate || score > bestCandidate.score) {
                bestCandidate = { ...candidate, score };
            }
        }

        if (!bestCandidate || bestCandidate.score < 3.1) return null;

        return {
            parts: [bestCandidate.left, bestCandidate.right],
            groups: filteredGroups.map(group => ({
                leftRatio: group.minX / sampleW,
                topRatio: group.minY / sampleH,
                widthRatio: group.width / sampleW,
                heightRatio: group.height / sampleH
            })),
            metrics: {
                sampleSize: [sampleW, sampleH],
                heightRatio,
                baselineRatioDiff,
                centerYDiff,
                gapRatio,
                areaRatio,
                splitIndex: bestCandidate.index
            }
        };
    } catch (error) {
        console.warn('[TextStyle] Inline visual hierarchy split failed:', error.message || error);
        return null;
    }
}

function hasTextGradient(css) {
    const backgroundImage = String(cssValue(css, 'background-image', 'backgroundImage', 'background') || '').toLowerCase();
    const clip = String(cssValue(css, '-webkit-background-clip', 'background-clip', 'WebkitBackgroundClip', 'backgroundClip') || '').toLowerCase();
    const fill = String(cssValue(css, '-webkit-text-fill-color', 'WebkitTextFillColor') || '').toLowerCase();
    return backgroundImage.includes('gradient') && (clip === 'text' || fill === 'transparent');
}

function parseLetterSpacingPx(value, fontSize) {
    if (value === undefined || value === null || value === '' || value === 'normal') return 0;
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    const raw = String(value).trim();
    const numeric = parsePixelValue(raw);
    if (numeric === null) return 0;
    if (raw.endsWith('em')) return numeric * fontSize;
    if (raw.endsWith('%')) return numeric / 100 * fontSize;
    return numeric;
}

function getLineHeightRatio(value, fallback = 1.08) {
    if (value === undefined || value === null || value === '' || value === 'normal') return fallback;
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value > 4 ? fallback : value;
    }
    const raw = String(value).trim();
    const numeric = parsePixelValue(raw);
    if (numeric === null) return fallback;
    if (raw.endsWith('px')) return null;
    if (raw.endsWith('%')) return numeric / 100;
    return numeric > 4 ? fallback : numeric;
}

function isDisplayTitleText(textContent, bbox) {
    const text = String(textContent || '').trim();
    if (!text || !Array.isArray(bbox) || bbox.length !== 4) return false;
    const compact = text.replace(/\s+/g, '');
    const isUpper = /^[A-Z0-9&$.,:;!?%+-]+$/.test(compact) && /[A-Z0-9]/.test(compact);
    const height = Number(bbox[2]) - Number(bbox[0]);
    const width = Number(bbox[3]) - Number(bbox[1]);
    return isUpper && height >= 80 && width >= 170 && compact.length >= 3;
}

function isSingleLineDisplayTitleText(textContent, bbox) {
    return isDisplayTitleText(textContent, bbox) && !String(textContent || '').includes('\n');
}

function isDoubleLineDisplayTitleText(textContent, bbox) {
    const text = String(textContent || '').trim();
    if (!isDisplayTitleText(text, bbox) || !text.includes('\n')) return false;
    const lines = text.split('\n').map(line => line.trim()).filter(Boolean);
    if (lines.length !== 2) return false;
    return lines.every(line =>
        /^[A-Z0-9$.,:;!?%+-]{2,}$/.test(line) &&
        /[A-Z0-9]/.test(line)
    );
}

function isMultiLineDisplayTitleText(textContent, bbox) {
    const text = String(textContent || '').trim();
    if (!isDisplayTitleText(text, bbox) || !text.includes('\n')) return false;
    const lines = text.split('\n').map(line => line.trim()).filter(Boolean);
    if (lines.length < 2 || lines.length > 4) return false;
    return lines.every(line => /^[A-Z0-9&$.,:;!?%+-]+$/.test(line));
}

function getStackedSymbolDisplayLines(textContent, bbox) {
    const text = String(textContent || '').trim();
    if (!isDisplayTitleText(text, bbox) || !text.includes('\n')) return null;
    const lines = text.split('\n').map(line => line.trim()).filter(Boolean);
    if (lines.length !== 3) return null;
    const [top, middle, bottom] = lines;
    const isWordLine = line => /^[A-Z0-9$.,:;!?%+-]{2,}$/.test(line) && /[A-Z0-9]/.test(line);
    const isSymbolLine = line => /^[&+×x•·-]$/.test(line);
    return isWordLine(top) && isSymbolLine(middle) && isWordLine(bottom) ? lines : null;
}

function shouldPreserveExtractedLineBreaks(textContent) {
    const text = String(textContent || '').trim();
    if (!text.includes('\n')) return false;
    const compact = text.replace(/\s+/g, '');
    const isMostlyUpper = /^[A-Z0-9&$.,:;!?%+-]+$/.test(compact) && /[A-Z0-9]/.test(compact);
    return isMostlyUpper || compact.length <= 32;
}

function isDisplayTextContent(textContent) {
    const compact = String(textContent || '').replace(/\s+/g, '');
    return compact.length > 0 &&
        compact.length <= 32 &&
        /^[A-Z0-9&$.,:;!?%+-]+$/.test(compact);
}

function isPriceTextContent(textContent) {
    return /^\$?\d+(?:\.\d{1,2})?$/.test(String(textContent || '').trim());
}

function getCharClass(char) {
    if (!char) return 'other';
    if (/[0-9]/.test(char)) return 'digit';
    if (/[A-Za-z\u00C0-\u024F\u4E00-\u9FFF]/.test(char)) return 'letter';
    return 'symbol';
}

function inferTextSplitCandidates(textContent) {
    const text = String(textContent || '').trim();
    const candidates = [];
    if (text.length < 2) return candidates;

    for (let index = 1; index < text.length; index += 1) {
        const left = text.slice(0, index);
        const right = text.slice(index);
        const leftClass = getCharClass(text[index - 1]);
        const rightClass = getCharClass(text[index]);
        let score = 0;

        if (leftClass !== rightClass) score += 2.4;
        if (leftClass === 'symbol' || rightClass === 'symbol') score += 1.8;
        if (index === 1 && leftClass === 'symbol') score += 1.4;
        if (index === text.length - 1 && rightClass === 'symbol') score += 1.4;
        if (leftClass === 'digit' && rightClass === 'letter') score += 0.6;
        if (leftClass === 'letter' && rightClass === 'digit') score += 0.6;

        candidates.push({ index, left, right, score });
    }

    return candidates.sort((a, b) => b.score - a.score);
}

function shouldTryVisualHierarchySplit(textContent, bbox, matchedPriceBadge = null) {
    const text = String(textContent || '').trim();
    if (!text || text.includes('\n') || /\s/.test(text)) return false;
    if (text.length < 2 || text.length > 6) return false;
    if (matchedPriceBadge?.bbox) return false;
    if (!Array.isArray(bbox) || bbox.length !== 4) return false;
    const width = Math.max(1, Number(bbox[3]) - Number(bbox[1]));
    const height = Math.max(1, Number(bbox[2]) - Number(bbox[0]));
    return width >= 30 && height >= 20;
}

function getSplitPriceParts(textContent, bbox, matchedPriceBadge = null) {
    const text = String(textContent || '').trim();
    if (matchedPriceBadge?.bbox) return null;
    const match = text.match(/^\$(\d{1,3})$/);
    if (!match || !Array.isArray(bbox) || bbox.length !== 4) return null;
    const width = Math.max(1, Number(bbox[3]) - Number(bbox[1]));
    const height = Math.max(1, Number(bbox[2]) - Number(bbox[0]));
    if (width < 42 || height < 28) return null;
    if (width / height < 1.15) return null;
    return {
        symbol: '$',
        digits: match[1]
    };
}

function scoreDisplayFontCandidate(candidate, textContent, width, height, lineCount, letterSpacingValue) {
    const compact = String(textContent || '').replace(/\s+/g, '');
    const safeWidth = Math.max(1, Number(width) || 1);
    const safeHeight = Math.max(1, Number(height) || 1);
    const safeLineCount = Math.max(1, Number(lineCount) || compact.split('\n').length || 1);
    const probeFontSize = Math.max(12, safeHeight / (safeLineCount * 0.92));
    const measuredWidth = measureTextBlock(
        compact,
        probeFontSize,
        'sans-serif',
        '400',
        'normal',
        letterSpacingValue
    ).width * (candidate.widthBias || 1);
    const utilization = measuredWidth / safeWidth;

    let score = candidate.scoreBias || 0;
    score += 1.3 - Math.abs(utilization - 0.96) * 2.1;
    if (compact.length <= 4) score += 0.12;
    else if (compact.length <= 8) score += 0.07;
    else if (compact.length <= 16) score += 0.03;
    if (utilization < 0.68) score -= 0.14;
    if (utilization > 1.15) score -= (utilization - 1.15) * 2.3;
    return score;
}

export function resolveDisplayFontFamily(fontFamily, textContent, metrics = {}) {
    const normalized = normalizeFontFamily(fontFamily);
    if (!isDisplayTextContent(textContent)) return normalized || 'sans-serif';
    const lower = normalized.toLowerCase();
    const isNarrowFamily = !normalized || lower === 'impact' || lower.includes('condensed') || lower.includes('narrow');

    const candidates = [];
    if (normalized && !isNarrowFamily) {
        candidates.push({ family: normalized, widthBias: 1, scoreBias: 0.08 });
    }
    candidates.push(...DISPLAY_FONT_CANDIDATES);

    let bestFamily = normalized || 'sans-serif';
    let bestScore = -Infinity;
    for (const candidate of candidates) {
        const score = scoreDisplayFontCandidate(
            candidate,
            textContent,
            metrics.width,
            metrics.height,
            metrics.lineCount,
            metrics.letterSpacingValue
        );
        if (score > bestScore) {
            bestScore = score;
            bestFamily = candidate.family;
        }
    }
    return bestFamily;
}

function resolveSubgroupTypography({
    text,
    width,
    height,
    baseFontFamily,
    baseFontWeight,
    fontStyle,
    letterSpacingValue = '0',
    targetProfile = null
}) {
    const partText = String(text || '').trim();
    const partClass = getCharClass(partText[0] || '');
    const displayLike = isDisplayTextContent(partText) || /^[0-9$%#+-]+$/.test(partText);
    const normalizedBaseFamily = normalizeFontFamily(baseFontFamily);
    const resolvedDisplayFamily = resolveDisplayFontFamily(baseFontFamily, partText, {
        width,
        height,
        lineCount: 1,
        letterSpacingValue
    });

    const familyCandidates = [];
    const addFamily = family => {
        const normalized = normalizeFontFamily(family);
        if (!normalized) return;
        if (familyCandidates.includes(normalized)) return;
        familyCandidates.push(normalized);
    };

    addFamily(normalizedBaseFamily);
    addFamily(resolvedDisplayFamily);
    if (displayLike || partClass === 'digit' || partClass === 'symbol') {
        SUBGROUP_DISPLAY_FONT_CANDIDATES.forEach(candidate => addFamily(candidate.family));
        DISPLAY_FONT_CANDIDATES.forEach(candidate => addFamily(candidate.family));
        addFamily('Arial Black');
    }
    if (!familyCandidates.length) addFamily('sans-serif');

    const weightCandidates = [];
    const addWeight = weight => {
        const normalized = String(weight || '').trim();
        if (!normalized) return;
        if (weightCandidates.includes(normalized)) return;
        weightCandidates.push(normalized);
    };
    addWeight(baseFontWeight);

    const preferModerateWeight = targetProfile?.preferModerateWeight === true;
    const preferHeightDominantFit = targetProfile?.preferHeightDominantFit === true;
    if (partClass === 'symbol') {
        addWeight('700');
        addWeight('600');
    } else if (preferModerateWeight) {
        addWeight('700');
        addWeight('600');
        addWeight('800');
        addWeight('500');
    } else {
        addWeight('900');
        addWeight('800');
        addWeight('700');
        addWeight('600');
    }
    if (!weightCandidates.length) addWeight('700');

    const targetWidthUtil = targetProfile?.widthUtil ?? (partClass === 'symbol' ? 0.58 : partClass === 'digit' ? 0.84 : 0.88);
    const targetHeightUtil = targetProfile?.heightUtil ?? (partClass === 'symbol' ? 0.66 : partClass === 'digit' ? 0.9 : 0.86);
    const targetInkAreaRatio = targetProfile?.inkAreaRatio ?? null;
    const targetBottomUtil = targetProfile?.bottomUtil ?? null;
    const targetCenterYUtil = targetProfile?.centerYUtil ?? null;
    const targetMask = targetProfile?.mask ?? null;
    let best = null;

    for (const candidateFamily of familyCandidates) {
        loadFontIfNeeded(candidateFamily);
        for (const candidateWeight of weightCandidates) {
            const fittedFontSize = fitDisplayTitleFontSizeToRect({
                text: partText,
                width,
                height,
                fontFamily: candidateFamily,
                fontWeight: candidateWeight,
                fontStyle,
                lineHeightValue: null,
                letterSpacingValue
            });
            const measured = measureTextBlock(
                partText,
                fittedFontSize,
                candidateFamily,
                candidateWeight,
                fontStyle,
                letterSpacingValue
            );
            const renderedProfile = measureRenderedGlyphProfile({
                text: partText,
                width,
                height,
                fontSize: fittedFontSize,
                fontFamily: candidateFamily,
                fontWeight: candidateWeight,
                fontStyle,
                letterSpacingValue,
                textAlign: partClass === 'symbol' ? 'center' : 'left'
            });
            const widthUtil = measured.width / Math.max(1, width);
            const heightUtil = measured.visualLineHeight / Math.max(1, height);
            const familyBonus = normalizeFontFamily(candidateFamily) === normalizeFontFamily(resolvedDisplayFamily) ? 0.18 : 0;
            const baseFamilyBonus = normalizeFontFamily(candidateFamily) === normalizedBaseFamily ? 0.08 : 0;
            const weightNum = Number(candidateWeight);
            const weightBonus = Number.isFinite(weightNum)
                ? (partClass === 'symbol'
                    ? (weightNum >= 600 && weightNum <= 800 ? 0.06 : -0.04)
                    : preferModerateWeight
                        ? (weightNum >= 600 && weightNum <= 700 ? 0.08 : weightNum >= 800 ? -0.06 : -0.02)
                        : (weightNum >= 700 ? 0.08 : -0.05))
                : 0;
            const contourPenalty = renderedProfile && targetInkAreaRatio !== null
                ? Math.abs(renderedProfile.inkAreaRatio - targetInkAreaRatio) * 1.1
                : 0;
            const bottomPenalty = renderedProfile && targetBottomUtil !== null
                ? Math.abs(renderedProfile.bottomUtil - targetBottomUtil) * 1.05
                : 0;
            const centerYPenalty = renderedProfile && targetCenterYUtil !== null
                ? Math.abs(renderedProfile.centerYUtil - targetCenterYUtil) * 0.9
                : 0;
            const maskCompare = renderedProfile && targetMask ? compareBinaryMasks(targetMask, renderedProfile.mask) : null;
            const maskPenalty = maskCompare
                ? (1 - maskCompare.iou) * 2.2 + maskCompare.mismatchRatio * 0.9
                : 0;
            const widthPenaltyWeight = preferHeightDominantFit ? 0.8 : 1.4;
            const heightPenaltyWeight = preferHeightDominantFit ? 1.8 : 1.5;
            const score = 1.8
                - Math.abs(widthUtil - targetWidthUtil) * widthPenaltyWeight
                - Math.abs(heightUtil - targetHeightUtil) * heightPenaltyWeight
                - contourPenalty
                - bottomPenalty
                - centerYPenalty
                - maskPenalty
                + familyBonus
                + baseFamilyBonus
                + weightBonus;
            if (!best || score > best.score) {
                best = {
                    family: candidateFamily,
                    weight: candidateWeight,
                    fontSize: fittedFontSize,
                    lineHeight: Math.max(1, fittedFontSize * (partClass === 'symbol' ? 0.82 : 0.88)),
                    textAlign: partClass === 'symbol' ? 'center' : 'left',
                    score,
                    widthUtil,
                    heightUtil,
                    renderedProfile,
                    maskCompare
                };
            }
        }
    }

    return best || {
        family: normalizedBaseFamily || 'sans-serif',
        weight: String(baseFontWeight || '700'),
        fontSize: Math.max(1, Math.min(width, height)),
        lineHeight: Math.max(1, Math.min(width, height) * 0.88),
        textAlign: partClass === 'symbol' ? 'center' : 'left',
        score: null,
        widthUtil: null,
        heightUtil: null
    };
}

function measureRenderedGlyphProfile({
    text,
    width,
    height,
    fontSize,
    fontFamily,
    fontWeight,
    fontStyle,
    letterSpacingValue = '0',
    textAlign = 'left'
}) {
    const sampleW = Math.max(18, Math.round(width));
    const sampleH = Math.max(18, Math.round(height));
    const canvas = document.createElement('canvas');
    canvas.width = sampleW;
    canvas.height = sampleH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, sampleW, sampleH);
    ctx.fillStyle = '#000000';
    ctx.font = `${fontStyle || 'normal'} ${fontWeight || '400'} ${fontSize}px ${fontFamily || 'sans-serif'}`;
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = textAlign === 'center' ? 'center' : 'left';

    const letterSpacingPx = parseLetterSpacingPx(letterSpacingValue, fontSize);
    const lines = String(text || '').split('\n');
    const lineHeightPx = Math.max(1, fontSize * 0.88);
    const totalHeight = Math.max(1, lineHeightPx * lines.length);
    const ascentEstimate = fontSize * 0.8;
    const startY = Math.max(ascentEstimate, (sampleH - totalHeight) / 2 + ascentEstimate);

    lines.forEach((line, lineIndex) => {
        const drawY = startY + lineIndex * lineHeightPx;
        if (textAlign === 'center') {
            ctx.fillText(line, sampleW / 2, drawY);
        } else if (letterSpacingPx === 0) {
            ctx.fillText(line, 0, drawY);
        } else {
            let cursorX = 0;
            for (const char of line) {
                ctx.fillText(char, cursorX, drawY);
                cursorX += ctx.measureText(char).width + letterSpacingPx;
            }
        }
    });

    const data = ctx.getImageData(0, 0, sampleW, sampleH).data;
    let minX = sampleW;
    let maxX = -1;
    let minY = sampleH;
    let maxY = -1;
    let inkCount = 0;

    for (let y = 0; y < sampleH; y += 1) {
        for (let x = 0; x < sampleW; x += 1) {
            const offset = (y * sampleW + x) * 4;
            const luminanceValue = data[offset];
            if (luminanceValue > 220) continue;
            inkCount += 1;
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
    }

    if (!inkCount || maxX < minX || maxY < minY) return null;

    const inkWidth = Math.max(1, maxX - minX + 1);
    const inkHeight = Math.max(1, maxY - minY + 1);
    const mask = Array(sampleW * sampleH).fill(0);
    for (let y = 0; y < sampleH; y += 1) {
        for (let x = 0; x < sampleW; x += 1) {
            const offset = (y * sampleW + x) * 4;
            if (data[offset] <= 220) {
                mask[y * sampleW + x] = 1;
            }
        }
    }
    return {
        widthUtil: inkWidth / Math.max(1, sampleW),
        heightUtil: inkHeight / Math.max(1, sampleH),
        inkAreaRatio: inkCount / Math.max(1, inkWidth * inkHeight),
        bottomUtil: maxY / Math.max(1, sampleH),
        centerYUtil: ((minY + maxY) / 2) / Math.max(1, sampleH),
        mask: normalizeBinaryMask(mask, sampleW, sampleH, 24, 24)
    };
}

function measureTextBlock(text, fontSize, fontFamily, fontWeight, fontStyle, letterSpacingValue) {
    const canvas = measureTextBlock.canvas || (measureTextBlock.canvas = document.createElement('canvas'));
    const ctx = canvas.getContext('2d');
    const safeFamily = fontFamily || 'sans-serif';
    const safeWeight = fontWeight || '400';
    const safeStyle = fontStyle || 'normal';
    ctx.font = `${safeStyle} ${safeWeight} ${fontSize}px ${safeFamily}`;

    const letterSpacingPx = parseLetterSpacingPx(letterSpacingValue, fontSize);
    let maxWidth = 0;
    let maxLineHeight = 0;
    const lines = String(text || '').split('\n');
    for (const line of lines) {
        const measuredText = ctx.measureText(line || ' ');
        const measured = measuredText.width;
        const spacingWidth = Math.max(0, line.length - 1) * letterSpacingPx;
        maxWidth = Math.max(maxWidth, measured + spacingWidth);
        const ascent = Number.isFinite(measuredText.actualBoundingBoxAscent)
            ? measuredText.actualBoundingBoxAscent
            : fontSize * 0.78;
        const descent = Number.isFinite(measuredText.actualBoundingBoxDescent)
            ? measuredText.actualBoundingBoxDescent
            : fontSize * 0.24;
        maxLineHeight = Math.max(maxLineHeight, ascent + descent);
    }
    return {
        width: maxWidth,
        lineCount: Math.max(1, lines.length),
        visualLineHeight: Math.max(1, maxLineHeight || fontSize * 0.96)
    };
}

function fitFontSizeToRect({
    text,
    width,
    height,
    fontFamily,
    fontWeight,
    fontStyle,
    lineHeightValue,
    letterSpacingValue
}) {
    const lines = Math.max(1, String(text || '').split('\n').length);
    const explicitLineHeightPx = String(lineHeightValue || '').trim().endsWith('px') ? parsePixelValue(lineHeightValue) : null;
    const lineHeightRatio = getLineHeightRatio(lineHeightValue) || 1.08;
    const maxByHeight = explicitLineHeightPx
        ? Math.max(1, explicitLineHeightPx / lineHeightRatio)
        : Math.max(1, height / (lines * lineHeightRatio));
    let low = 1;
    let high = Math.max(2, Math.min(maxByHeight * 1.4, height * 1.45, 420));

    for (let i = 0; i < 18; i++) {
        const mid = (low + high) / 2;
        const measured = measureTextBlock(text, mid, fontFamily, fontWeight, fontStyle, letterSpacingValue);
        const lineStep = explicitLineHeightPx || mid * lineHeightRatio;
        const blockHeight = measured.lineCount > 1
            ? measured.visualLineHeight + (measured.lineCount - 1) * lineStep
            : measured.visualLineHeight;
        const fitsWidth = measured.width <= width * 1.02;
        const fitsHeight = blockHeight <= height * 1.03;
        if (fitsWidth && fitsHeight) low = mid;
        else high = mid;
    }

    return Math.max(1, low);
}

function fitDisplayTitleFontSizeToRect({
    text,
    width,
    height,
    fontFamily,
    fontWeight,
    fontStyle,
    lineHeightValue,
    letterSpacingValue
}) {
    const explicitLineHeightPx = String(lineHeightValue || '').trim().endsWith('px') ? parsePixelValue(lineHeightValue) : null;
    const lineHeightRatio = getLineHeightRatio(lineHeightValue) || 0.94;
    const targetWidth = Math.max(1, width * 0.92);
    let low = 1;
    let high = Math.max(2, Math.min(height * 2.2, width * 0.95, 420));

    for (let i = 0; i < 20; i++) {
        const mid = (low + high) / 2;
        const measured = measureTextBlock(text, mid, fontFamily, fontWeight, fontStyle, letterSpacingValue);
        const lineStep = explicitLineHeightPx || mid * lineHeightRatio;
        const blockHeight = measured.lineCount > 1
            ? measured.visualLineHeight + (measured.lineCount - 1) * lineStep
            : measured.visualLineHeight;
        const fitsWidth = measured.width <= targetWidth;
        const fitsHeight = blockHeight <= height * 1.34;
        if (fitsWidth && fitsHeight) low = mid;
        else high = mid;
    }

    return Math.max(1, low);
}

function fitMultiLineDisplayTitleFontSizeToRect({
    text,
    width,
    height,
    fontFamily,
    fontWeight,
    fontStyle,
    letterSpacingValue
}) {
    const lines = String(text || '').split('\n').filter(line => line.trim());
    const lineHeightRatio = 0.76;
    const targetWidth = Math.max(1, width * 0.98);
    let low = 1;
    let high = Math.max(2, Math.min(height * 0.9, width * 0.95, 420));

    for (let i = 0; i < 20; i++) {
        const mid = (low + high) / 2;
        const measured = measureTextBlock(text, mid, fontFamily, fontWeight, fontStyle, letterSpacingValue);
        const blockHeight = measured.visualLineHeight + (Math.max(1, lines.length) - 1) * mid * lineHeightRatio;
        const fitsWidth = measured.width <= targetWidth;
        const fitsHeight = blockHeight <= height * 1.04;
        if (fitsWidth && fitsHeight) low = mid;
        else high = mid;
    }

    return Math.max(1, low);
}

function fitDoubleLineDisplayTitleFontSizeToRect({
    text,
    width,
    height,
    fontFamily,
    fontWeight,
    fontStyle,
    letterSpacingValue
}) {
    const lineHeightRatio = 0.82;
    const targetWidth = Math.max(1, width * 0.97);
    let low = 1;
    let high = Math.max(2, Math.min(height * 1.18, width * 0.95, 420));

    for (let i = 0; i < 20; i++) {
        const mid = (low + high) / 2;
        const measured = measureTextBlock(text, mid, fontFamily, fontWeight, fontStyle, letterSpacingValue);
        const blockHeight = measured.visualLineHeight + mid * lineHeightRatio;
        const fitsWidth = measured.width <= targetWidth;
        const fitsHeight = blockHeight <= height * 1.08;
        if (fitsWidth && fitsHeight) low = mid;
        else high = mid;
    }

    return Math.max(1, low);
}

function fitPriceBadgeFontSizeToRect({
    text,
    badgeSize,
    fontFamily,
    fontWeight,
    fontStyle,
    letterSpacingValue
}) {
    const targetWidth = Math.max(1, badgeSize * 0.76);
    const targetHeight = Math.max(1, badgeSize * 0.44);
    let low = 1;
    let high = Math.max(2, Math.min(badgeSize * 1.2, 160));

    for (let i = 0; i < 18; i++) {
        const mid = (low + high) / 2;
        const measured = measureTextBlock(text, mid, fontFamily, fontWeight, fontStyle, letterSpacingValue);
        const fitsWidth = measured.width <= targetWidth;
        const fitsHeight = measured.visualLineHeight <= targetHeight;
        if (fitsWidth && fitsHeight) low = mid;
        else high = mid;
    }

    return Math.max(1, low);
}

export function loadFontIfNeeded(fontFamily) {
    const cleanFamily = normalizeFontFamily(fontFamily);
    if (!cleanFamily || !WEB_FONT_ALLOWLIST.includes(cleanFamily)) return;

    const fontUrl = `https://fonts.googleapis.com/css2?family=${cleanFamily.replace(/ /g, '+')}&display=swap`;
    if (document.querySelector(`link[href="${fontUrl}"]`)) return;

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = fontUrl;
    document.head.appendChild(link);
}

export async function buildExtractedTextState({
    lineObj,
    layerObj,
    fallbackBbox,
    containerCandidates = [],
    allTextLines = [],
    index = 0,
    baseX = 0,
    baseY = 0,
    parentWidth = 300,
    parentHeight = 300,
    zIndex = 1,
    parentId = null,
    sourceImage = null,
    minWidth = 20,
    minHeight = 10
}) {
    const css = lineObj?.css && typeof lineObj.css === 'object' ? { ...lineObj.css } : {};
    const textContent = lineObj?.textContent || '';
    const parsedStyle = parseFontStyle(lineObj?.fontStyle || layerObj?.fontStyle || '');

    const cssFamily = cssValue(css, 'font-family', 'fontFamily');
    const cssColor = cssValue(css, 'color');
    const cssFontSize = cssValue(css, 'font-size', 'fontSize');
    const cssFontWeight = cssValue(css, 'font-weight', 'fontWeight');
    const cssFontStyle = cssValue(css, 'font-style', 'fontStyle');
    const cssLineHeight = cssValue(css, 'line-height', 'lineHeight');
    const cssLetterSpacing = cssValue(css, 'letter-spacing', 'letterSpacing');
    const cssTextAlign = cssValue(css, 'text-align', 'textAlign');
    const cssTextShadow = cssValue(css, 'text-shadow', 'textShadow');
    const cssTextStroke = cssValue(css, '-webkit-text-stroke', 'WebkitTextStroke', 'text-stroke', 'textStroke');
    const isPriceText = isPriceTextContent(textContent);

    const explicitLineBbox = Array.isArray(lineObj?.bbox) && lineObj.bbox.length === 4;
    const lineBbox = explicitLineBbox ? lineObj.bbox : fallbackBbox;
    const rect = bboxToRect(lineBbox, baseX, baseY, parentWidth, parentHeight, minWidth, minHeight);
    const visualStyle = lineObj?.visualStyle || await measureTextVisualStyle(sourceImage, lineBbox, textContent);

    let lY = rect.top;
    let lH = rect.height;
    let lX = rect.left;
    let lW = rect.width;
    let numLines = Math.max(1, String(textContent).split('\n').length);
    let matchedPriceBadge = null;
    if (!explicitLineBbox && sameBbox(lineBbox, fallbackBbox) && allTextLines.length > 1) {
        lH = lH / allTextLines.length;
        lY += index * lH;
        numLines = 1;
    }

    if (visualStyle?.inkBounds && explicitLineBbox) {
        const bounds = visualStyle.inkBounds;
        const hasUsefulBounds = bounds.widthRatio > 0.08 &&
            bounds.heightRatio > 0.15 &&
            visualStyle.maskCoverage > 0.004;
        if (hasUsefulBounds) {
            const padX = Math.min(rect.width * 0.035, Math.max(2, rect.width * 0.015));
            const padY = Math.min(rect.height * 0.08, Math.max(1, rect.height * 0.025));
            lX = rect.left + bounds.leftRatio * rect.width - padX;
            lY = rect.top + bounds.topRatio * rect.height - padY;
            lW = Math.max(minWidth, bounds.widthRatio * rect.width + padX * 2);
            lH = Math.max(minHeight, bounds.heightRatio * rect.height + padY * 2);
        }
    }

    if (isPriceText) {
        matchedPriceBadge = resolveMatchedPriceBadge(lineBbox, containerCandidates);
        if (matchedPriceBadge?.bbox) {
            const badgeRect = bboxToRect(matchedPriceBadge.bbox, baseX, baseY, parentWidth, parentHeight, minWidth, minHeight);
            const targetWidth = Math.max(minWidth, badgeRect.width * 0.76);
            const targetHeight = Math.max(minHeight, badgeRect.height * 0.44);
            lW = targetWidth;
            lH = targetHeight;
            lX = badgeRect.left + (badgeRect.width - targetWidth) / 2;
            lY = badgeRect.top + (badgeRect.height - targetHeight) / 2;
            numLines = 1;
        }
    }

    const compactUpper = /^[A-Z0-9\s.,:;!?%+-]+$/.test(textContent.trim()) && /[A-Z0-9]/.test(textContent);
    const rawFontFamily = normalizeFontFamily(cssFamily) || parsedStyle.fontFamily || '';
    const fontFamily = rawFontFamily || 'sans-serif';
    loadFontIfNeeded(fontFamily);
    const cssFontSizePx = parsePixelValue(cssFontSize);
    const fontWeight = cssFontWeight || visualStyle?.fontWeight || (compactUpper ? '400' : (cssFontSizePx && cssFontSizePx >= 26 ? '700' : '400'));
    const fontStyle = cssFontStyle || '';
    const letterSpacing = cssLetterSpacing || visualStyle?.letterSpacingHint || (isPriceText ? '0' : (compactUpper && textContent.trim().length <= 24 ? '0.02em' : '0'));
    const preserveLineBreaks = shouldPreserveExtractedLineBreaks(textContent);
    const isSingleLineDisplayTitle = isSingleLineDisplayTitleText(textContent, lineBbox);
    const isDoubleLineDisplayTitle = isDoubleLineDisplayTitleText(textContent, lineBbox);
    const stackedSymbolLines = getStackedSymbolDisplayLines(textContent, lineBbox);
    const isMultiLineDisplayTitle = Boolean(stackedSymbolLines);
    const rawExplicitLineHeightPx = parsePixelValue(cssLineHeight);
    const lineHeightIsClearlyInvalid = rawExplicitLineHeightPx !== null &&
        lH >= 12 &&
        rawExplicitLineHeightPx < Math.max(2, (lH / numLines) * 0.25);
    const usableLineHeightValue = lineHeightIsClearlyInvalid ? null : cssLineHeight;
    const fittedFontSize = isSingleLineDisplayTitle
        ? fitDisplayTitleFontSizeToRect({
            text: textContent,
            width: lW,
            height: lH,
            fontFamily,
            fontWeight,
            fontStyle,
            lineHeightValue: usableLineHeightValue,
            letterSpacingValue: letterSpacing
        })
        : isDoubleLineDisplayTitle
            ? fitDoubleLineDisplayTitleFontSizeToRect({
                text: textContent,
                width: lW,
                height: lH,
                fontFamily,
                fontWeight,
                fontStyle,
                letterSpacingValue: letterSpacing
            })
        : isMultiLineDisplayTitle
            ? fitMultiLineDisplayTitleFontSizeToRect({
                text: textContent,
                width: lW,
                height: lH,
                fontFamily,
                fontWeight,
                fontStyle,
                letterSpacingValue: letterSpacing
            })
        : fitFontSizeToRect({
            text: textContent,
            width: lW,
            height: lH,
            fontFamily,
            fontWeight,
            fontStyle,
            lineHeightValue: usableLineHeightValue,
            letterSpacingValue: letterSpacing
        });
    const explicitLineHeightPx = lineHeightIsClearlyInvalid ? null : rawExplicitLineHeightPx;
    const lineHeightRatio = isDoubleLineDisplayTitle
        ? 0.82
        : isMultiLineDisplayTitle
        ? 0.76
        : (getLineHeightRatio(usableLineHeightValue) || (compactUpper ? 0.96 : 1.08));
    const maxFontSizeByHeight = explicitLineHeightPx
        ? Math.max(1, Math.min(lH / numLines, explicitLineHeightPx) / lineHeightRatio)
        : Math.max(1, lH / (numLines * lineHeightRatio));
    const effectiveMaxFontSizeByHeight = isSingleLineDisplayTitle
        ? maxFontSizeByHeight * 1.34
        : isDoubleLineDisplayTitle
            ? maxFontSizeByHeight * 1.18
        : isMultiLineDisplayTitle
            ? maxFontSizeByHeight * 1.16
        : maxFontSizeByHeight;
    const displayTitleBoost = compactUpper && (!preserveLineBreaks || isDoubleLineDisplayTitle || isMultiLineDisplayTitle) ? 1.12 : 1;
    const cssFontSizeLooksInvalid = cssFontSizePx !== null &&
        maxFontSizeByHeight >= 6 &&
        cssFontSizePx < maxFontSizeByHeight * 0.35;
    const usableCssFontSizePx = cssFontSizeLooksInvalid ? null : cssFontSizePx;
    const measuredFontSize = visualStyle?.inkBounds?.heightRatio
        ? Math.max(1, Math.min(lH * (numLines > 1 ? 0.92 / numLines : 1.04), effectiveMaxFontSizeByHeight))
        : null;
    const preferredFontSize = (isDoubleLineDisplayTitle || isMultiLineDisplayTitle
        ? fittedFontSize
        : (usableCssFontSizePx || measuredFontSize || fittedFontSize)) * displayTitleBoost;
    const fitCollapsed = preserveLineBreaks && fittedFontSize < maxFontSizeByHeight * 0.35;
    const protectedPreferredFontSize = fitCollapsed
        ? ((measuredFontSize || maxFontSizeByHeight * 0.86) * displayTitleBoost)
        : preferredFontSize;
    let fontSizeNum = preserveLineBreaks && !fitCollapsed
        ? Math.max(1, Math.min(preferredFontSize, fittedFontSize * 1.03, effectiveMaxFontSizeByHeight))
        : Math.max(1, Math.min(protectedPreferredFontSize, effectiveMaxFontSizeByHeight));
    let priceBadgeFontLimit = null;

    if (isPriceText) {
        const estimatedBadgeSize = matchedPriceBadge?.bbox
            ? Math.max(
                ((matchedPriceBadge.bbox[3] - matchedPriceBadge.bbox[1]) / 1000) * parentWidth,
                ((matchedPriceBadge.bbox[2] - matchedPriceBadge.bbox[0]) / 1000) * parentHeight
            )
            : clamp(Math.max(lW, lH) * 2.15, 42, 74);
        priceBadgeFontLimit = fitPriceBadgeFontSizeToRect({
            text: textContent,
            badgeSize: estimatedBadgeSize,
            fontFamily,
            fontWeight,
            fontStyle,
            letterSpacingValue: letterSpacing
        });
        fontSizeNum = Math.min(fontSizeNum, priceBadgeFontLimit);
    }
    let displayScaleX = 1;
    let displayWidthUtilization = null;

    if (isSingleLineDisplayTitle) {
        const fittedMetrics = measureTextBlock(textContent, fontSizeNum, fontFamily, fontWeight, fontStyle, letterSpacing);
        displayWidthUtilization = fittedMetrics.width / Math.max(1, lW);
        if (displayWidthUtilization > 0 && displayWidthUtilization < 0.86) {
            displayScaleX = clamp((lW * 0.9) / Math.max(1, fittedMetrics.width), 1, 1.65);
        }
    }

    if (!usableCssFontSizePx && measuredFontSize && preserveLineBreaks) {
        const measuredText = measureTextBlock(textContent, fontSizeNum, fontFamily, fontWeight, fontStyle, letterSpacing);
        if (measuredText.width > lW * 0.98) {
            lW = Math.min(parentWidth - (lX - baseX), measuredText.width * 1.04);
        }
    }

    let fontColor = isVisibleColor(cssColor) ? cssColor : parsedStyle.fontColor;
    let effectiveVisualStyle = visualStyle;
    let sampledColor = effectiveVisualStyle?.fontColor || await sampleForegroundColor(sourceImage, lineBbox, textContent);
    const hasDeclaredColor = isVisibleColor(fontColor);
    let visualCoverage = effectiveVisualStyle?.maskCoverage ?? effectiveVisualStyle?.colorDiagnostics?.maskCoverage ?? 0;
    let sampledRgb = parseCssColorToRgb(sampledColor);
    const declaredRgb = parseCssColorToRgb(fontColor);
    let sampledOverridesDeclared = hasDeclaredColor &&
        isVisibleColor(sampledColor) &&
        sampledRgb &&
        declaredRgb &&
        colorDistance(sampledRgb, declaredRgb) >= 90;
    let backgroundRgb = effectiveVisualStyle?.colorDiagnostics?.background
        ? parseCssColorToRgb(effectiveVisualStyle.colorDiagnostics.background)
        : null;
    let sampledLooksLikeBackground = sampledRgb &&
        declaredRgb &&
        backgroundRgb &&
        colorDistance(sampledRgb, backgroundRgb) <
            colorDistance(declaredRgb, backgroundRgb) * 0.72;
    let samplingLooksFlooded = visualCoverage >= 0.72;
    let expandedColorDiagnostics = null;

    if (isDisplayTitleText(textContent, lineBbox) &&
        sampledOverridesDeclared &&
        (samplingLooksFlooded || sampledLooksLikeBackground)) {
        const expandedBbox = expandNormalizedBbox(lineBbox);
        const expandedVisualStyle = await measureTextVisualStyle(sourceImage, expandedBbox, textContent);
        if (expandedVisualStyle?.fontColor) {
            effectiveVisualStyle = expandedVisualStyle;
            sampledColor = expandedVisualStyle.fontColor;
            visualCoverage = expandedVisualStyle.maskCoverage ?? expandedVisualStyle.colorDiagnostics?.maskCoverage ?? 0;
            sampledRgb = parseCssColorToRgb(sampledColor);
            backgroundRgb = expandedVisualStyle?.colorDiagnostics?.background
                ? parseCssColorToRgb(expandedVisualStyle.colorDiagnostics.background)
                : null;
            sampledOverridesDeclared = hasDeclaredColor &&
                isVisibleColor(sampledColor) &&
                sampledRgb &&
                declaredRgb &&
                colorDistance(sampledRgb, declaredRgb) >= 90;
            sampledLooksLikeBackground = sampledRgb &&
                declaredRgb &&
                backgroundRgb &&
                colorDistance(sampledRgb, backgroundRgb) <
                    colorDistance(declaredRgb, backgroundRgb) * 0.72;
            samplingLooksFlooded = visualCoverage >= 0.72;
            expandedColorDiagnostics = {
                bbox: expandedBbox,
                visual: expandedVisualStyle.colorDiagnostics || null
            };
        }
    }

    if (!hasTextGradient(css) &&
        isVisibleColor(sampledColor) &&
        !(sampledOverridesDeclared && (samplingLooksFlooded || sampledLooksLikeBackground))) {
        fontColor = sampledColor;
    } else if (!isVisibleColor(fontColor)) {
        fontColor = sampledColor;
    }
    if (!isVisibleColor(fontColor)) fontColor = '#000000';
    const visualHierarchySplit = await detectInlineVisualHierarchySplit(sourceImage, lineBbox, textContent, matchedPriceBadge);

    if (typeof window !== 'undefined') {
        window.__marmoTextColorDiagnostics = window.__marmoTextColorDiagnostics || [];
        window.__marmoTextColorDiagnostics.push({
            text: textContent,
            bbox: Array.isArray(lineBbox) ? lineBbox.map(Number) : null,
            finalColor: fontColor,
            visual: effectiveVisualStyle?.colorDiagnostics || null,
            expandedColorDiagnostics,
            cssColor: cssColor || null,
            parsedColor: parsedStyle.fontColor || null
        });
        if (window.__marmoTextColorDiagnostics.length > 200) {
            window.__marmoTextColorDiagnostics.splice(0, window.__marmoTextColorDiagnostics.length - 200);
        }
    }

    const maxLineHeightPx = Math.max(1, lH / numLines);
    const requestedLineHeightPx = explicitLineHeightPx || fontSizeNum * (isPriceText ? 0.94 : lineHeightRatio);
    const lineHeight = `${Math.max(1, Math.min(requestedLineHeightPx, maxLineHeightPx))}px`;
    const textAlign = isPriceText && matchedPriceBadge?.bbox ? 'center' : (cssTextAlign || 'left');
    const outputCss = { ...css };
    if (displayScaleX > 1.04) {
        const existingTransform = cssValue(outputCss, 'transform');
        const scaleTransform = `scaleX(${Number(displayScaleX.toFixed(3))})`;
        outputCss.transform = existingTransform ? `${existingTransform} ${scaleTransform}` : scaleTransform;
    }
    if (cssFontSizeLooksInvalid || lineHeightIsClearlyInvalid) {
        delete outputCss['font-size'];
        delete outputCss.fontSize;
        delete outputCss['line-height'];
        delete outputCss.lineHeight;
    }

    if (stackedSymbolLines) {
        const splitCssBase = { ...outputCss };
        delete splitCssBase.transform;
        delete splitCssBase['transform'];
        delete splitCssBase.fontSize;
        delete splitCssBase['font-size'];
        delete splitCssBase.lineHeight;
        delete splitCssBase['line-height'];

        const [topLine, symbolLine, bottomLine] = stackedSymbolLines;
        const insetX = Math.max(2, lW * 0.05);
        const availableWidth = Math.max(minWidth, lW - insetX * 2);
        const topHeight = lH * 0.34;
        const symbolHeight = lH * 0.16;
        const bottomHeight = lH * 0.34;
        const gapHeight = lH * 0.05;
        const topTop = lY + lH * 0.04;
        const symbolTop = topTop + topHeight + gapHeight;
        const bottomTop = symbolTop + symbolHeight + gapHeight;
        const lineLayouts = [
            { text: topLine, top: topTop, height: topHeight, role: 'word' },
            { text: symbolLine, top: symbolTop, height: symbolHeight, role: 'symbol' },
            { text: bottomLine, top: bottomTop, height: bottomHeight, role: 'word' }
        ];

        const splitStates = lineLayouts.map((layout, lineIndex) => {
            const fittedSize = fitDisplayTitleFontSizeToRect({
                text: layout.text,
                width: availableWidth,
                height: layout.height,
                fontFamily,
                fontWeight,
                fontStyle,
                lineHeightValue: null,
                letterSpacingValue: letterSpacing
            });
            const size = layout.role === 'symbol'
                ? Math.max(8, Math.min(fittedSize, fontSizeNum * 0.6))
                : Math.max(8, fittedSize * 0.97);
            const measured = measureTextBlock(layout.text, size, fontFamily, fontWeight, fontStyle, letterSpacing);
            const lineWidth = Math.min(availableWidth, Math.max(minWidth, measured.width * (layout.role === 'symbol' ? 1.18 : 1.02)));
            const left = lX + Math.max(0, (lW - lineWidth) / 2);
            return {
                id: `text-${Date.now()}-${Math.random().toString(36).substr(2, 6)}-${index}-${lineIndex}`,
                content: layout.text,
                left,
                top: layout.top,
                width: lineWidth,
                height: Math.max(minHeight, layout.height),
                fontColor,
                fontFamily,
                fontSize: `${Math.max(1, size)}px`,
                fontWeight: String(fontWeight),
                fontStyle,
                lineHeight: `${Math.max(1, size * (layout.role === 'symbol' ? 0.9 : 0.92))}px`,
                letterSpacing,
                textAlign: 'center',
                textShadow: cssTextShadow || '',
                WebkitTextStroke: cssTextStroke || '',
                zIndex: zIndex + lineIndex * 0.01,
                parentId,
                css: {
                    ...splitCssBase,
                    wordBreak: 'normal',
                    overflowWrap: 'normal',
                    whiteSpace: 'nowrap'
                }
            };
        });

        if (typeof window !== 'undefined') {
            window.__marmoTextLayoutDiagnostics = window.__marmoTextLayoutDiagnostics || [];
            window.__marmoTextLayoutDiagnostics.push({
                text: textContent,
                bbox: Array.isArray(lineBbox) ? lineBbox.map(Number) : null,
                rect: { left: lX, top: lY, width: lW, height: lH },
                splitDisplayTitle: true,
                splitLines: splitStates.map(state => ({
                    text: state.content,
                    rect: { left: state.left, top: state.top, width: state.width, height: state.height },
                    fontSize: state.fontSize,
                    lineHeight: state.lineHeight
                }))
            });
            if (window.__marmoTextLayoutDiagnostics.length > 200) {
                window.__marmoTextLayoutDiagnostics.splice(0, window.__marmoTextLayoutDiagnostics.length - 200);
            }
        }

        return splitStates;
    }

    if (visualHierarchySplit?.parts?.length === 2 && Array.isArray(visualHierarchySplit.groups) && visualHierarchySplit.groups.length === 2) {
        const splitCssBase = { ...outputCss };
        delete splitCssBase.transform;
        delete splitCssBase['transform'];
        delete splitCssBase.fontSize;
        delete splitCssBase['font-size'];
        delete splitCssBase.lineHeight;
        delete splitCssBase['line-height'];
        const sourceProfiles = await Promise.all(
            visualHierarchySplit.groups.map(group => extractSourceGlyphProfile(sourceImage, lineBbox, group))
        );

        const splitStates = visualHierarchySplit.parts.map((partText, partIndex) => {
            const group = visualHierarchySplit.groups[Math.min(partIndex, visualHierarchySplit.groups.length - 1)];
            const sourceProfile = sourceProfiles[Math.min(partIndex, sourceProfiles.length - 1)] || null;
            const padX = Math.max(1, lW * 0.015);
            const padY = Math.max(1, lH * 0.02);
            const partLeft = lX + group.leftRatio * lW - padX;
            const partTop = lY + group.topRatio * lH - padY;
            const partWidth = Math.max(minWidth, group.widthRatio * lW + padX * 2);
            const partHeight = Math.max(minHeight, group.heightRatio * lH + padY * 2);
            const targetProfile = {
                widthUtil: sourceProfile?.widthUtil ?? clamp(group.widthRatio, 0.2, 0.98),
                heightUtil: sourceProfile?.heightUtil ?? clamp(group.heightRatio, 0.2, 0.98),
                inkAreaRatio: sourceProfile?.inkAreaRatio ?? visualStyle?.inkAreaRatio ?? null,
                bottomUtil: sourceProfile?.bottomUtil ?? clamp(group.topRatio + group.heightRatio, 0, 1),
                centerYUtil: sourceProfile?.centerYUtil ?? clamp(group.topRatio + group.heightRatio / 2, 0, 1),
                mask: sourceProfile?.mask ?? null
            };
            const typography = resolveSubgroupTypography({
                text: partText,
                width: partWidth,
                height: partHeight,
                baseFontFamily: fontFamily,
                baseFontWeight: fontWeight,
                fontStyle,
                letterSpacingValue: '0',
                targetProfile
            });
            return {
                id: `text-${Date.now()}-${Math.random().toString(36).substr(2, 6)}-${index}-visual-${partIndex}`,
                content: partText,
                left: partLeft,
                top: partTop,
                width: partWidth,
                height: partHeight,
                fontColor,
                fontFamily: typography.family,
                fontSize: `${Math.max(1, typography.fontSize)}px`,
                fontWeight: String(typography.weight),
                fontStyle,
                lineHeight: `${Math.max(1, typography.lineHeight)}px`,
                letterSpacing: '0',
                textAlign: typography.textAlign,
                textShadow: cssTextShadow || '',
                WebkitTextStroke: cssTextStroke || '',
                zIndex: zIndex + partIndex * 0.01,
                parentId,
                css: {
                    ...splitCssBase,
                    wordBreak: 'normal',
                    overflowWrap: 'normal',
                    whiteSpace: 'nowrap'
                }
            };
        });

        if (typeof window !== 'undefined') {
            window.__marmoTextLayoutDiagnostics = window.__marmoTextLayoutDiagnostics || [];
            window.__marmoTextLayoutDiagnostics.push({
                text: textContent,
                bbox: Array.isArray(lineBbox) ? lineBbox.map(Number) : null,
                rect: { left: lX, top: lY, width: lW, height: lH },
                splitVisualHierarchy: true,
                visualHierarchyMetrics: visualHierarchySplit.metrics || null,
                splitLines: splitStates.map(state => ({
                    text: state.content,
                    rect: { left: state.left, top: state.top, width: state.width, height: state.height },
                    fontSize: state.fontSize,
                    lineHeight: state.lineHeight
                }))
            });
            if (window.__marmoTextLayoutDiagnostics.length > 200) {
                window.__marmoTextLayoutDiagnostics.splice(0, window.__marmoTextLayoutDiagnostics.length - 200);
            }
        }

        return splitStates;
    }

    const splitPriceParts = getSplitPriceParts(textContent, lineBbox, matchedPriceBadge);
    if (splitPriceParts) {
        const splitCssBase = { ...outputCss };
        delete splitCssBase.transform;
        delete splitCssBase['transform'];
        delete splitCssBase.fontSize;
        delete splitCssBase['font-size'];
        delete splitCssBase.lineHeight;
        delete splitCssBase['line-height'];

        const priceFamily = fontFamily;
        const priceWeight = String(fontWeight);
        const symbolWidth = lW * 0.14;
        const digitsWidth = lW * 0.8;
        const horizontalGap = lW * 0.045;
        const symbolLeft = lX;
        const digitsLeft = lX + symbolWidth + horizontalGap;
        const symbolHeight = lH * 0.62;
        const digitsHeight = lH;

        const fallbackGroups = [
            {
                leftRatio: 0,
                topRatio: 0.22,
                widthRatio: 0.14,
                heightRatio: 0.62
            },
            {
                leftRatio: 0.16,
                topRatio: 0.02,
                widthRatio: 0.8,
                heightRatio: 0.96
            }
        ];
        const sourceProfiles = await Promise.all(
            fallbackGroups.map(group => extractSourceGlyphProfile(sourceImage, lineBbox, group))
        );
        const cohortProfile = await collectNonBadgePriceCohortProfile({
            allTextLines,
            containerCandidates,
            sourceImage
        });

        const symbolTargetProfile = {
            widthUtil: sourceProfiles[0]?.widthUtil ?? 0.42,
            heightUtil: sourceProfiles[0]?.heightUtil ?? 0.62,
            inkAreaRatio: sourceProfiles[0]?.inkAreaRatio ?? visualStyle?.inkAreaRatio ?? null,
            bottomUtil: sourceProfiles[0]?.bottomUtil ?? 0.82,
            centerYUtil: sourceProfiles[0]?.centerYUtil ?? 0.54,
            mask: sourceProfiles[0]?.mask ?? null
        };
        const digitsTargetProfile = {
            widthUtil: Math.max(
                Math.min(cohortProfile?.widthUtil ?? 1, 0.82),
                Math.min(sourceProfiles[1]?.widthUtil ?? 1, 0.84),
                0.72
            ),
            heightUtil: Math.max(
                cohortProfile?.heightUtil ?? 0,
                sourceProfiles[1]?.heightUtil ?? 0,
                0.94
            ),
            inkAreaRatio: cohortProfile?.inkAreaRatio ?? sourceProfiles[1]?.inkAreaRatio ?? visualStyle?.inkAreaRatio ?? null,
            bottomUtil: cohortProfile?.bottomUtil ?? sourceProfiles[1]?.bottomUtil ?? 0.9,
            centerYUtil: cohortProfile?.centerYUtil ?? sourceProfiles[1]?.centerYUtil ?? 0.52,
            mask: sourceProfiles[1]?.mask ?? null,
            preferModerateWeight: true,
            preferHeightDominantFit: true
        };

        let splitStates = [
            {
                id: `text-${Date.now()}-${Math.random().toString(36).substr(2, 6)}-${index}-symbol`,
                content: splitPriceParts.symbol,
                left: symbolLeft,
                top: lY + lH * 0.22,
                width: Math.max(minWidth, symbolWidth),
                height: Math.max(minHeight, symbolHeight),
                typography: resolveSubgroupTypography({
                    text: splitPriceParts.symbol,
                    width: Math.max(minWidth, symbolWidth),
                    height: Math.max(minHeight, symbolHeight),
                    baseFontFamily: priceFamily,
                    baseFontWeight: priceWeight,
                    fontStyle,
                    letterSpacingValue: '0',
                    targetProfile: symbolTargetProfile
                }),
                zOffset: 0
            },
            {
                id: `text-${Date.now()}-${Math.random().toString(36).substr(2, 6)}-${index}-digits`,
                content: splitPriceParts.digits,
                left: digitsLeft,
                top: lY + lH * 0.02,
                width: Math.max(minWidth, digitsWidth),
                height: Math.max(minHeight, digitsHeight),
                typography: resolveSubgroupTypography({
                    text: splitPriceParts.digits,
                    width: Math.max(minWidth, digitsWidth),
                    height: Math.max(minHeight, digitsHeight),
                    baseFontFamily: priceFamily,
                    baseFontWeight: priceWeight,
                    fontStyle,
                    letterSpacingValue: '0',
                    targetProfile: digitsTargetProfile
                }),
                zOffset: 0.01
            }
        ];

        splitStates = splitStates.map(part => {
            const typography = part.content === splitPriceParts.digits
                ? boostNonBadgePriceDigitsTypography({
                    typography: part.typography,
                    width: part.width,
                    height: part.height,
                    text: part.content,
                    fontStyle,
                    letterSpacingValue: '0',
                    targetProfile: digitsTargetProfile
                })
                : part.typography;
            return {
                ...part,
                typography
            };
        }).map(part => ({
            id: part.id,
            content: part.content,
            left: part.left,
            top: part.top,
            width: part.width,
            height: part.height,
            fontColor,
            fontFamily: part.typography.family,
            fontSize: `${Math.max(1, part.typography.fontSize)}px`,
            fontWeight: String(part.typography.weight),
            fontStyle,
            lineHeight: `${Math.max(1, part.typography.lineHeight)}px`,
            letterSpacing: '0',
            textAlign: part.typography.textAlign,
            textShadow: cssTextShadow || '',
            WebkitTextStroke: cssTextStroke || '',
            zIndex: zIndex + part.zOffset,
            parentId,
            css: {
                ...splitCssBase,
                wordBreak: 'normal',
                overflowWrap: 'normal',
                whiteSpace: 'nowrap'
            }
        }));

        if (typeof window !== 'undefined') {
            window.__marmoTextLayoutDiagnostics = window.__marmoTextLayoutDiagnostics || [];
            window.__marmoTextLayoutDiagnostics.push({
                text: textContent,
                bbox: Array.isArray(lineBbox) ? lineBbox.map(Number) : null,
                rect: { left: lX, top: lY, width: lW, height: lH },
                splitPriceText: true,
                splitLines: splitStates.map(state => ({
                    text: state.content,
                    rect: { left: state.left, top: state.top, width: state.width, height: state.height },
                    fontSize: state.fontSize,
                    lineHeight: state.lineHeight
                }))
            });
            if (window.__marmoTextLayoutDiagnostics.length > 200) {
                window.__marmoTextLayoutDiagnostics.splice(0, window.__marmoTextLayoutDiagnostics.length - 200);
            }
        }

        return splitStates;
    }

    if (typeof window !== 'undefined') {
        window.__marmoTextLayoutDiagnostics = window.__marmoTextLayoutDiagnostics || [];
        window.__marmoTextLayoutDiagnostics.push({
            text: textContent,
            bbox: Array.isArray(lineBbox) ? lineBbox.map(Number) : null,
            rect: { left: lX, top: lY, width: lW, height: lH },
            lines: numLines,
            preserveLineBreaks,
            isDoubleLineDisplayTitle,
            isMultiLineDisplayTitle,
            rawFontFamily,
            fontFamily,
            cssFontSize,
            cssFontSizePx,
            cssFontSizeLooksInvalid,
            lineHeightIsClearlyInvalid,
            usableCssFontSizePx,
            fittedFontSize,
            measuredFontSize,
            preferredFontSize,
            fitCollapsed,
            isPriceText,
            priceBadgeFontLimit,
            matchedPriceBadgeBbox: matchedPriceBadge?.bbox || null,
            matchedPriceBadgeScore: matchedPriceBadge?.score || null,
            maxFontSizeByHeight,
            displayScaleX,
            displayWidthUtilization,
            finalFontSize: fontSizeNum,
            lineHeight,
            whiteSpace: preserveLineBreaks ? 'pre' : (textContent.includes('\n') ? 'pre-wrap' : 'nowrap')
        });
        if (window.__marmoTextLayoutDiagnostics.length > 200) {
            window.__marmoTextLayoutDiagnostics.splice(0, window.__marmoTextLayoutDiagnostics.length - 200);
        }
    }

    return {
        id: `text-${Date.now()}-${Math.random().toString(36).substr(2, 6)}-${index}`,
        content: textContent,
        left: lX,
        top: lY,
        width: lW,
        height: lH,
        fontColor,
        fontFamily,
        fontSize: `${Math.max(1, fontSizeNum)}px`,
        fontWeight: String(fontWeight),
        fontStyle,
        lineHeight,
        letterSpacing,
        textAlign,
        textShadow: cssTextShadow || '',
        WebkitTextStroke: cssTextStroke || '',
        zIndex,
        parentId,
        css: {
            ...outputCss,
            wordBreak: 'normal',
            overflowWrap: 'normal',
            whiteSpace: preserveLineBreaks ? 'pre' : (textContent.includes('\n') ? 'pre-wrap' : 'nowrap')
        }
    };
}
