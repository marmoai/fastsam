function cloneSerializable(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function normalizeTextLine(line, fallbackLayer = null) {
    if (!line) return null;
    const textContent = String(line.textContent || '').trim();
    if (!textContent) return null;
    return {
        textContent,
        fontStyle: line.fontStyle || fallbackLayer?.fontStyle || '',
        bbox: Array.isArray(line.bbox) && line.bbox.length === 4 ? line.bbox : fallbackLayer?.bbox,
        css: cloneSerializable(line.css || fallbackLayer?.css || null)
    };
}

export function normalizeTextLines(lines = [], fallbackLayer = null) {
    if (!Array.isArray(lines)) return [];
    return repairBrokenUppercaseWordLines(lines
        .map(line => normalizeTextLine(line, fallbackLayer))
        .filter(Boolean));
}

function mergeBboxes(a, b) {
    if (!Array.isArray(a) || a.length !== 4) return cloneSerializable(b);
    if (!Array.isArray(b) || b.length !== 4) return cloneSerializable(a);
    return [
        Math.min(Number(a[0]), Number(b[0])),
        Math.min(Number(a[1]), Number(b[1])),
        Math.max(Number(a[2]), Number(b[2])),
        Math.max(Number(a[3]), Number(b[3]))
    ];
}

function repairBrokenUppercaseWordLines(lines = []) {
    const repaired = [];
    for (const line of lines) {
        const text = String(line.textContent || '').trim();
        const previous = repaired[repaired.length - 1];
        const prevText = String(previous?.textContent || '').trim();
        const isSingleUppercaseTail = /^[A-Z]$/.test(text);
        const canAppendToPrevious = previous &&
            /^[A-Z]{3,}$/.test(prevText) &&
            Array.isArray(previous.bbox) &&
            Array.isArray(line.bbox) &&
            Math.abs(Number(line.bbox[1]) - Number(previous.bbox[1])) < 70 &&
            Number(line.bbox[0]) >= Number(previous.bbox[0]);

        if (isSingleUppercaseTail && canAppendToPrevious) {
            previous.textContent = `${prevText}${text}`;
            previous.bbox = mergeBboxes(previous.bbox, line.bbox);
            previous.css = previous.css || line.css || null;
            continue;
        }
        repaired.push(line);
    }
    return repaired;
}

function isReliableExtractionSource(source) {
    return !source || !['semantic_fallback', 'semantic_line'].includes(String(source));
}

export function getCachedTextExtraction(layer, options = {}) {
    const {
        allowSemanticFallback = true,
        allowTextContentFallback = true
    } = options;
    if (!layer) return [];

    if (allowSemanticFallback || isReliableExtractionSource(layer.textExtraction?.source)) {
        const extractionLines = normalizeTextLines(layer.textExtraction?.lines, layer);
        if (extractionLines.length > 0) return extractionLines;
    }

    const directLines = normalizeTextLines(layer.textLines, layer);
    if (directLines.length > 0) return directLines;

    if (!allowTextContentFallback) return [];

    const textContent = String(layer.textContent || '').trim();
    if (!textContent) return [];

    return [{
        textContent,
        fontStyle: layer.fontStyle || '',
        bbox: Array.isArray(layer.bbox) && layer.bbox.length === 4 ? layer.bbox : undefined,
        css: cloneSerializable(layer.css || null)
    }];
}

export function buildTextExtractionCache(lines, fallbackLayer = null, source = 'semantic_ocr') {
    const normalizedLines = normalizeTextLines(lines, fallbackLayer);
    return {
        source,
        bbox: Array.isArray(fallbackLayer?.bbox) ? cloneSerializable(fallbackLayer.bbox) : null,
        lines: normalizedLines,
        generatedAt: Date.now()
    };
}

function getIntersectionArea(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return 0;
    const top = Math.max(Number(a[0]), Number(b[0]));
    const left = Math.max(Number(a[1]), Number(b[1]));
    const bottom = Math.min(Number(a[2]), Number(b[2]));
    const right = Math.min(Number(a[3]), Number(b[3]));
    return Math.max(0, bottom - top) * Math.max(0, right - left);
}

function getBboxArea(bbox) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return 0;
    return Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
}

export function filterTextLinesToBbox(lines = [], bbox, minCoverage = 0.55) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return lines;
    return normalizeTextLines(lines)
        .filter(line => {
            if (!Array.isArray(line.bbox) || line.bbox.length !== 4) return true;
            const lineArea = getBboxArea(line.bbox);
            if (lineArea <= 0) return true;
            return getIntersectionArea(line.bbox, bbox) / lineArea >= minCoverage;
        });
}

export function normalizeOcrTextLines(ocrResult, fallbackLayer = null) {
    if (Array.isArray(ocrResult?.lines) && ocrResult.lines.length > 0) {
        return normalizeTextLines(ocrResult.lines, fallbackLayer);
    }
    if (ocrResult?.textContent) {
        return normalizeTextLines([{
            textContent: ocrResult.textContent,
            fontStyle: ocrResult.fontStyle || fallbackLayer?.fontStyle || '',
            bbox: ocrResult.bbox || fallbackLayer?.bbox,
            css: ocrResult.css || fallbackLayer?.css || null
        }], fallbackLayer);
    }
    return [];
}
