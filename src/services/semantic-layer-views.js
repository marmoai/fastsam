import { extractTextFromImage } from '../ai-services/skills-engine.js';
import { buildTextExtractionCache } from './text-extraction-cache.js';
import { getProxiedUrl } from '../core/utils.js';
import { prepareCompletionAssetPreflight } from './object-completion-planner.js';
import {
    createCompletionLayerIndex,
    resolveEntityParts
} from './completion-contract.js';

const ENABLE_FLAT_AD_TEXT_OCR_SUPPLEMENT = true;
const ENABLE_PER_LAYER_TEXT_OCR_DURING_SEMANTIC_BUILD = false;
const ENABLE_REGION_TEXT_OCR_SUPPLEMENT = false;

// Semantic boxes describe only the visible part of an occluded object and are
// therefore not pixel-perfect. Keep this low enough to catch a table/stool
// contact, while still requiring an actual intersection before completion.
const COMPLETION_MIN_BBOX_OVERLAP = 0.015;
const COMPLETION_MAX_NEAR_CONTACT_GAP = 18;

function cloneSerializable(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function getRuntimeType(layer) {
    if (layer.renderMode === 'text_css') {
        return 'text_node';
    }
    if (layer.renderMode === 'semantic_group') {
        return 'semantic_group';
    }
    if (layer.category === 'background' || layer.layerType === 'background_plate') {
        return 'background_master';
    }
    if (layer.semanticType === 'element_text') {
        return 'text_node';
    }
    if (layer.compositeRole === 'composite_group' || layer.semanticType === 'composite_group') {
        return 'semantic_group';
    }
    return 'semantic_object';
}

function inferExtractionProfile(layer = {}) {
    const semantic = String(layer.semanticType || '').toLowerCase();
    const designRole = String(layer.designRole || '').toLowerCase();
    const renderMode = String(layer.renderMode || '').toLowerCase();
    const runtimeType = String(layer.runtimeType || '').toLowerCase();
    const name = String(layer.name || '').toLowerCase();
    const text = `${semantic} ${designRole} ${renderMode} ${runtimeType} ${name}`;

    if (
        renderMode === 'text_css' ||
        runtimeType === 'text_node' ||
        semantic === 'element_text'
    ) {
        return 'text_layer';
    }

    if (
        renderMode === 'background_plate' ||
        semantic === 'ad_background' ||
        designRole === 'base_background'
    ) {
        return 'background_plate';
    }

    if (
        ['product_food', 'product_drink'].includes(semantic) ||
        /(food|dish|meal|plate|rice|pork|roasted|burger|pizza|noodle|salad|drink|beverage|cola|tea|coffee|choco|食物|食品|菜品|餐盘|炒饭|米饭|猪肉|烤肉|饮料|可乐|茶|咖啡|热巧|杯)/.test(text)
    ) {
        return 'layout_embedded_product';
    }

    // A food/product asset may be used as a poster decoration, but it is
    // still a raster object rather than a vector layout primitive. Preserve
    // that distinction before considering its design role.
    if (
        renderMode === 'vector_shape' ||
        ['shape_panel', 'price_badge', 'cta_button', 'logo_mark', 'decor_graphic', 'flat_ad_layout'].includes(semantic) ||
        ['local_panel', 'price_badge', 'decor_shape'].includes(designRole)
    ) {
        return 'vector_layout_element';
    }

    if (
        semantic === 'product_packaging' ||
        designRole === 'product_image' ||
        /(product|packaging|earphone|earphones|earbud|earbuds|headphone|headphones|case|device|electronics|gadget|商品图|产品图|商品|产品|耳机|蓝牙耳机|充电盒|电子产品|包装)/.test(text)
    ) {
        return 'multi_part_hard_product';
    }

    if (
        semantic === 'decor_vase' ||
        semantic === 'decor_flower' ||
        /(vase flowers|vase and flowers|flowers in vase|flower arrangement|bouquet|plant arrangement|potted plant|花瓶花艺|花艺|插花|花束|盆栽|植物组合)/.test(text)
    ) {
        return 'compound_object';
    }

    if (
        ['furniture_sofa', 'furniture_table', 'furniture_chair', 'furniture_stool', 'furniture_bed', 'furniture_cabinet'].includes(semantic)
    ) {
        return 'multi_part_hard_object';
    }

    return 'standard_object';
}

function normalizeLayer(layer, overrides = {}) {
    const next = {
        ...cloneSerializable(layer),
        ...overrides
    };
    next.runtimeType = overrides.runtimeType || next.runtimeType || getRuntimeType(next);
    next.viewRole = overrides.viewRole || next.viewRole || 'editable-scene';
    next.extractionProfile = overrides.extractionProfile || next.extractionProfile || inferExtractionProfile(next);
    return next;
}

function hasFlatDesignSignal(layer) {
    if (!layer) return false;
    const text = [
        layer.semanticType,
        layer.designRole,
        layer.renderMode,
        layer.name
    ].map(value => String(value || '').toLowerCase()).join(' ');

    return /(ad_background|shape_panel|price_badge|product_food|product_drink|local_panel|product_image|flat_ad|menu|poster|海报|广告|菜单|面板|底板|价格|价签|徽章|饮料|菜品|产品图|商品图)/.test(text);
}

function shouldHideFlatDesignGroup(layer, children) {
    if (!Array.isArray(children) || children.length === 0) return false;
    const isGroup = layer?.runtimeType === 'semantic_group' ||
        layer?.compositeRole === 'composite_group' ||
        layer?.semanticType === 'composite_group' ||
        layer?.semanticType === 'flat_ad_layout';
    if (!isGroup) return false;

    return hasFlatDesignSignal(layer) || children.some(hasFlatDesignSignal);
}

function flattenSemanticLayers(layers = [], parentLayer = null, depth = 0) {
    const result = [];
    if (!Array.isArray(layers)) return result;

    layers.forEach((layer, index) => {
        const children = Array.isArray(layer?.children) ? layer.children : [];
        const compositeRole = layer.compositeRole || (children.length > 0 ? 'composite_group' : 'atomic_object');
        const normalized = normalizeLayer(layer, {
            id: layer.id || `${parentLayer?.id || 'layer'}_${index + 1}`,
            parentLayerId: layer.parentLayerId || parentLayer?.id || null,
            compositeRole,
            runtimeType: compositeRole === 'composite_group' ? 'semantic_group' : undefined,
            viewRole: layer.viewRole || 'editable-scene',
            editable: layer.editable !== undefined ? layer.editable : compositeRole !== 'composite_group',
            hierarchyDepth: depth
        });

        const hideGroup = shouldHideFlatDesignGroup(normalized, children);
        if (!hideGroup) {
            result.push(normalized);
        }

        if (children.length > 0) {
            result.push(...flattenSemanticLayers(children, hideGroup ? parentLayer : normalized, hideGroup ? depth : depth + 1));
        }
    });

    return result;
}

function buildEditableTextLayer(baseLayer, line, index) {
    const text = (line?.textContent || '').trim();
    const suffix = text ? text.slice(0, 16) : `line_${index + 1}`;
    const textExtraction = buildTextExtractionCache([line], {
        ...baseLayer,
        bbox: Array.isArray(line?.bbox) ? line.bbox : baseLayer.bbox
    }, line?.source || baseLayer.textExtraction?.source || 'semantic_line');

    return normalizeLayer(baseLayer, {
        id: `${baseLayer.id || 'text'}_line_${index + 1}`,
        name: text ? `文字: ${suffix}` : `${baseLayer.name || '文字'} ${index + 1}`,
        textContent: text,
        fontStyle: line?.fontStyle || baseLayer.fontStyle || '',
        css: cloneSerializable(line?.css || baseLayer.css || null),
        textExtraction: cloneSerializable(textExtraction),
        bbox: Array.isArray(line?.bbox) ? line.bbox : baseLayer.bbox,
        cleanPlateLayerId: baseLayer.id || null,
        sourceTextLayerId: baseLayer.id || null,
        semanticType: 'element_text',
        runtimeType: 'text_node',
        viewRole: 'editable-scene'
    });
}

function buildTextLinesFromSemanticLayer(layer) {
    const text = (layer?.textContent || '').trim();
    if (!text) return [];

    const lines = text.split(/\n+/).map(value => value.trim()).filter(Boolean);
    if (!lines.length) return [];

    const baseBbox = Array.isArray(layer.bbox) && layer.bbox.length === 4
        ? layer.bbox
        : [0, 0, 1000, 1000];
    const [ymin, xmin, ymax, xmax] = baseBbox;
    const lineHeight = (ymax - ymin) / lines.length;

    return lines.map((lineText, index) => ({
        textContent: lineText,
        fontStyle: layer.fontStyle || '',
        css: cloneSerializable(layer.css || null),
        bbox: [
            ymin + lineHeight * index,
            xmin,
            ymin + lineHeight * (index + 1),
            xmax
        ]
    }));
}

function splitCompositeAdTextLine(line) {
    const text = String(line?.textContent || '').trim();
    if (!Array.isArray(line?.bbox) || line.bbox.length !== 4) {
        return [line];
    }

    const parts = text.split(/\n+/).map(value => value.trim()).filter(Boolean);
    if (parts.length <= 1) return [line];

    const [ymin, xmin, ymax, xmax] = line.bbox;
    const lineHeight = (ymax - ymin) / parts.length;
    return parts.map((part, index) => ({
        ...line,
        textContent: part,
        bbox: [
            ymin + lineHeight * index,
            xmin,
            ymin + lineHeight * (index + 1),
            xmax
        ]
    }));
}

function splitCompositeAdTextLines(lines = []) {
    return lines.flatMap(splitCompositeAdTextLine);
}

function getLineHeight(line) {
    if (!Array.isArray(line?.bbox) || line.bbox.length !== 4) return 0;
    return Math.max(0, line.bbox[2] - line.bbox[0]);
}

function getLineWidth(line) {
    if (!Array.isArray(line?.bbox) || line.bbox.length !== 4) return 0;
    return Math.max(0, line.bbox[3] - line.bbox[1]);
}

function mergeBboxes(a, b) {
    if (!Array.isArray(a) || a.length !== 4) return cloneSerializable(b);
    if (!Array.isArray(b) || b.length !== 4) return cloneSerializable(a);
    return [
        Math.min(a[0], b[0]),
        Math.min(a[1], b[1]),
        Math.max(a[2], b[2]),
        Math.max(a[3], b[3])
    ];
}

function shouldKeepTextLineAtomic(line) {
    const text = String(line?.textContent || '').trim();
    if (!text) return true;
    if (isPriceTextLine(line)) return true;
    if (/^www\.|\.com$|https?:\/\//i.test(text)) return true;
    if (text.length <= 12 && !/\s/.test(text) && getLineHeight(line) >= 24) return true;
    return false;
}

function canMergeTextLines(previous, current) {
    if (!previous || !current) return false;
    if (shouldKeepTextLineAtomic(previous) || shouldKeepTextLineAtomic(current)) return false;
    if (!Array.isArray(previous.bbox) || !Array.isArray(current.bbox)) return false;

    const [py1, px1, py2, px2] = previous.bbox;
    const [cy1, cx1, cy2, cx2] = current.bbox;
    if (cy1 < py1) return false;

    const previousHeight = Math.max(1, getLineHeight(previous));
    const currentHeight = Math.max(1, getLineHeight(current));
    const averageHeight = (previousHeight + currentHeight) / 2;
    const verticalGap = cy1 - py2;
    if (verticalGap < -averageHeight * 0.25 || verticalGap > averageHeight * 1.25) return false;

    const leftDelta = Math.abs(cx1 - px1);
    const rightDelta = Math.abs(cx2 - px2);
    const widthRatio = getLineWidth(current) / Math.max(1, getLineWidth(previous));
    const alignedLeft = leftDelta <= Math.max(28, averageHeight * 1.8);
    const alignedRight = rightDelta <= Math.max(42, averageHeight * 2.6);
    const similarWidth = widthRatio >= 0.38 && widthRatio <= 1.85;
    const similarHeight = currentHeight / previousHeight >= 0.55 && currentHeight / previousHeight <= 1.75;
    const horizontalOverlap = Math.max(0, Math.min(px2, cx2) - Math.max(px1, cx1));
    const overlapRatio = horizontalOverlap / Math.max(1, Math.min(getLineWidth(previous), getLineWidth(current)));

    return similarHeight && similarWidth && (alignedLeft || alignedRight || overlapRatio >= 0.55);
}

function mergeFlatAdTextBlocks(lines = []) {
    const normalized = lines
        .filter(line => String(line?.textContent || '').trim() && Array.isArray(line?.bbox) && line.bbox.length === 4)
        .map(line => ({
            ...line,
            textContent: String(line.textContent || '').trim()
        }))
        .sort((a, b) => {
            const ay = a.bbox[0];
            const by = b.bbox[0];
            if (Math.abs(ay - by) > 12) return ay - by;
            return a.bbox[1] - b.bbox[1];
        });

    const groups = [];
    for (const line of normalized) {
        const lastGroup = groups[groups.length - 1];
        const lastLine = lastGroup?.lines?.[lastGroup.lines.length - 1];
        if (lastGroup && canMergeTextLines(lastLine, line)) {
            lastGroup.lines.push(line);
            lastGroup.bbox = mergeBboxes(lastGroup.bbox, line.bbox);
            continue;
        }
        groups.push({
            lines: [line],
            bbox: cloneSerializable(line.bbox)
        });
    }

    return groups.map(group => {
        const first = group.lines[0] || {};
        return {
            ...first,
            textContent: group.lines.map(line => line.textContent).join('\n'),
            bbox: group.bbox,
            css: cloneSerializable(first.css || null),
            fontStyle: first.fontStyle || ''
        };
    });
}

function normalizeTextContent(line) {
    return String(line?.textContent || '')
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();
}

function getTextLineConflict(line, seenLines = []) {
    const text = normalizeTextContent(line);
    if (!text) return { action: 'skip' };
    const lineArea = getBboxArea(line.bbox);

    for (let index = 0; index < seenLines.length; index += 1) {
        const seen = seenLines[index];
        const intersection = getBboxIntersection(line.bbox, seen.bbox);
        if (!intersection) continue;
        const overlap = getBboxArea(intersection) / Math.max(1, Math.min(getBboxArea(line.bbox), getBboxArea(seen.bbox)));
        if (seen.text === text && overlap >= 0.45) return { action: 'skip' };
        if (
            overlap >= 0.72 &&
            text.length > seen.text.length &&
            text.includes(seen.text) &&
            lineArea > getBboxArea(seen.bbox) * 1.35
        ) {
            return { action: 'skip' };
        }
        if (
            overlap >= 0.72 &&
            seen.text.length > text.length &&
            seen.text.includes(text) &&
            getBboxArea(seen.bbox) > lineArea * 1.35
        ) {
            return { action: 'keep' };
        }
    }

    return { action: 'keep' };
}

function dedupeTextLines(lines = [], seenLines = []) {
    const result = [];
    for (const line of lines) {
        const conflict = getTextLineConflict(line, seenLines);
        if (conflict.action === 'skip') continue;
        seenLines.push({
            text: normalizeTextContent(line),
            bbox: cloneSerializable(line.bbox || null)
        });
        result.push(line);
    }
    return result;
}

function findDocumentOcrLinesForLayer(layer, ocrLines = []) {
    if (!Array.isArray(ocrLines) || ocrLines.length === 0) return [];
    const layerText = normalizeTextContent(layer);
    const layerBbox = layer?.bbox;
    return ocrLines.filter(line => {
        const lineText = normalizeTextContent(line);
        if (!lineText) return false;

        const textMatches = layerText &&
            (lineText === layerText ||
                layerText.includes(lineText) ||
                lineText.includes(layerText));

        const intersection = getBboxIntersection(line.bbox, layerBbox);
        const overlap = intersection
            ? getBboxArea(intersection) / Math.max(1, Math.min(getBboxArea(line.bbox), getBboxArea(layerBbox)))
            : 0;

        return (textMatches && overlap >= 0.18) || overlap >= 0.55;
    });
}

function pruneCompositeDuplicateTextLayers(layers = []) {
    const textLayers = layers
        .map((layer, index) => ({ layer, index }))
        .filter(({ layer }) => layer?.semanticType === 'element_text' || layer?.runtimeType === 'text_node');
    const removeIndexes = new Set();

    for (const current of textLayers) {
        const currentText = normalizeTextContent(current.layer);
        const currentArea = getBboxArea(current.layer.bbox);
        if (!currentText || currentArea <= 0) continue;

        for (const other of textLayers) {
            if (current.index === other.index || removeIndexes.has(current.index)) continue;
            const otherText = normalizeTextContent(other.layer);
            const otherArea = getBboxArea(other.layer.bbox);
            if (!otherText || otherArea <= 0) continue;

            const intersection = getBboxIntersection(current.layer.bbox, other.layer.bbox);
            if (!intersection) continue;
            const overlap = getBboxArea(intersection) / Math.max(1, Math.min(currentArea, otherArea));
            if (
                overlap >= 0.72 &&
                currentText.length > otherText.length &&
                currentText.includes(otherText) &&
                currentArea > otherArea * 1.35
            ) {
                removeIndexes.add(current.index);
            }
        }
    }

    return layers.filter((_, index) => !removeIndexes.has(index));
}

function isLikelyFlatAdLayerSet(layers = []) {
    if (!Array.isArray(layers)) return false;
    let flatSignalCount = 0;
    for (const layer of layers) {
        const text = [
            layer?.semanticType,
            layer?.designRole,
            layer?.renderMode,
            layer?.name
        ].map(value => String(value || '').toLowerCase()).join(' ');

        if (/(ad_background|shape_panel|price_badge|product_food|product_drink|local_panel|price_badge|product_image|海报|广告|面板|底板|价格|价签|徽章)/.test(text)) {
            flatSignalCount += 1;
        }
    }
    return flatSignalCount >= 4;
}

function normalizeOcrLines(ocrResult, fallbackBbox = [0, 0, 1000, 1000]) {
    if (Array.isArray(ocrResult?.lines) && ocrResult.lines.length > 0) {
        return ocrResult.lines;
    }
    if (ocrResult?.textContent) {
        return [{
            textContent: ocrResult.textContent,
            fontStyle: ocrResult.fontStyle || '',
            bbox: ocrResult.bbox || fallbackBbox,
            css: ocrResult.css || null
        }];
    }
    return [];
}

function getEditableTextLayerCount(layers = []) {
    return layers.filter(layer => layer?.semanticType === 'element_text' || layer?.runtimeType === 'text_node').length;
}

function isProductImageLayer(layer) {
    const semantic = String(layer?.semanticType || '').toLowerCase();
    const role = String(layer?.designRole || '').toLowerCase();
    return ['product_food', 'product_drink', 'product_packaging'].includes(semantic) || role === 'product_image';
}

function isShapeCarrierLayer(layer) {
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

function isPriceTextLine(line) {
    const text = String(line?.textContent || '').trim();
    return /^\$?\s*\d+([.,]\d+)?$/.test(text) || /^\$\s*\d+/.test(text);
}

function isLikelyLabelTextLine(line) {
    const bbox = line?.bbox;
    if (!Array.isArray(bbox) || bbox.length !== 4) return false;
    if (isPriceTextLine(line)) return false;
    const text = String(line?.textContent || '').trim();
    if (!text || text.length > 32) return false;
    const [ymin, xmin, ymax, xmax] = bbox;
    const width = xmax - xmin;
    const height = ymax - ymin;
    if (width < 24 || width > 220 || height < 12 || height > 85) return false;
    return true;
}

function hasSimilarLayer(layers, bbox, predicate, overlapThreshold = 0.55) {
    const bboxArea = Math.max(1, getBboxArea(bbox));
    return layers.some(layer => {
        if (predicate && !predicate(layer)) return false;
        const layerArea = getBboxArea(layer?.bbox);
        if (layerArea <= 0) return false;
        if (layerArea > bboxArea * 2.2) return false;
        const intersection = getBboxIntersection(layer.bbox, bbox);
        if (!intersection) return false;
        const overlap = getBboxArea(intersection) / Math.max(1, Math.min(layerArea, bboxArea));
        return overlap >= overlapThreshold;
    });
}

function clampBbox(bbox) {
    return bbox.map(value => Math.round(Math.max(0, Math.min(1000, value))));
}

function findNearbyProductForTextLine(line, productLayers = []) {
    const textBbox = line?.bbox;
    if (!Array.isArray(textBbox) || textBbox.length !== 4) return null;
    const [ty1, tx1, ty2, tx2] = textBbox;
    const textCenter = getBboxCenter(textBbox);
    if (!textCenter) return null;
    const textHeight = ty2 - ty1;
    const textWidth = tx2 - tx1;

    let best = null;
    let bestScore = Infinity;
    for (const product of productLayers) {
        if (!Array.isArray(product?.bbox) || product.bbox.length !== 4) continue;
        const [py1, px1, py2, px2] = product.bbox;
        const productCenter = getBboxCenter(product.bbox);
        if (!productCenter) continue;

        const verticalDistance = Math.abs(productCenter.y - textCenter.y);
        const horizontalGap = Math.max(0, Math.max(px1 - tx2, tx1 - px2));
        const verticalNear = verticalDistance <= Math.max(115, textHeight * 4.2);
        const horizontalNear = horizontalGap <= Math.max(150, textWidth * 2.8);
        const overlapsOrAdjacent = px2 >= tx1 - 30 && px1 <= tx2 + 240;
        if (!verticalNear || !horizontalNear || !overlapsOrAdjacent) continue;

        const score = verticalDistance + horizontalGap * 0.75;
        if (score < bestScore) {
            best = product;
            bestScore = score;
        }
    }

    return best;
}

function buildFlatAdAtomicPanelSupplements(layers = [], textLines = []) {
    if (!Array.isArray(layers) || !Array.isArray(textLines) || textLines.length === 0) return [];
    const productLayers = layers.filter(isProductImageLayer);
    const additions = [];

    textLines.forEach((line, index) => {
        if (!isLikelyLabelTextLine(line)) return;
        const nearbyProduct = findNearbyProductForTextLine(line, productLayers);
        if (!nearbyProduct) return;

        const [ty1, tx1, ty2, tx2] = line.bbox;
        const [py1, px1, py2, px2] = nearbyProduct.bbox;
        const productHeight = py2 - py1;
        const productLowerHalfTop = py1 + productHeight * 0.38;
        const panelBbox = clampBbox([
            Math.min(ty1, productLowerHalfTop) - 10,
            Math.min(tx1, px1) - 18,
            Math.max(ty2, py2) + 8,
            Math.max(tx2, px2) + 14
        ]);
        const panelWidth = panelBbox[3] - panelBbox[1];
        const panelHeight = panelBbox[2] - panelBbox[0];
        const panelArea = getBboxArea(panelBbox);
        if (panelWidth < 70 || panelHeight < 24 || panelWidth > 340 || panelHeight > 150 || panelArea > 42000) return;
        if (hasSimilarLayer(layers.concat(additions), panelBbox, isShapeCarrierLayer, 0.68)) return;

        additions.push({
            id: `flat_ad_atomic_label_panel_${index + 1}`,
            name: `${line.textContent || '标签'}底板`,
            category: 'object',
            layerType: 'foreground_asset',
            reasoning: '平面广告解析层根据 OCR 文本与邻近产品图生成的独立标签底板。',
            semanticType: 'shape_panel',
            designRole: 'local_panel',
            renderMode: 'vector_shape',
            compositeRole: 'atomic_object',
            parentLayerId: null,
            children: [],
            textContent: '',
            fontStyle: '',
            bbox: panelBbox,
            zIndex: Math.max(1, Number(nearbyProduct.zIndex || 1) - 1),
            editable: true,
            promptHint: '产品标签底板',
            assetStatus: 'idle',
            extractionProfile: 'vector_layout_element',
            generatedBy: 'flat_ad_layout_analyzer'
        });
    });

    return additions;
}

function removeBroadPanelsCoveredByAtomicPanels(layers = []) {
    const generatedPanels = layers.filter(layer => layer?.generatedBy === 'flat_ad_layout_analyzer' && isShapeCarrierLayer(layer));
    if (generatedPanels.length < 2) return layers;

    return layers.filter(layer => {
        if (!isShapeCarrierLayer(layer) || layer.generatedBy === 'flat_ad_layout_analyzer') return true;
        const bbox = layer.bbox;
        if (!Array.isArray(bbox) || bbox.length !== 4) return true;
        const [ymin, xmin, ymax, xmax] = bbox;
        const width = xmax - xmin;
        const height = ymax - ymin;
        const area = getBboxArea(bbox);
        if (area < 50000 || width / Math.max(1, height) < 2.6) return true;

        const coveredAtomicPanels = generatedPanels.filter(panel => {
            const intersection = getBboxIntersection(bbox, panel.bbox);
            if (!intersection) return false;
            return getBboxArea(intersection) / Math.max(1, getBboxArea(panel.bbox)) >= 0.72;
        });
        return coveredAtomicPanels.length < 2;
    });
}

function getBboxArea(bbox) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return 0;
    const [ymin, xmin, ymax, xmax] = bbox;
    return Math.max(0, xmax - xmin) * Math.max(0, ymax - ymin);
}

function getBboxIntersection(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return null;
    const [ay1, ax1, ay2, ax2] = a;
    const [by1, bx1, by2, bx2] = b;
    const ymin = Math.max(ay1, by1);
    const xmin = Math.max(ax1, bx1);
    const ymax = Math.min(ay2, by2);
    const xmax = Math.min(ax2, bx2);
    if (ymax <= ymin || xmax <= xmin) return null;
    return [ymin, xmin, ymax, xmax];
}

function getBboxNearContact(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return null;
    const [ay1, ax1, ay2, ax2] = a;
    const [by1, bx1, by2, bx2] = b;
    const horizontalGap = Math.max(0, Math.max(ax1 - bx2, bx1 - ax2));
    const verticalGap = Math.max(0, Math.max(ay1 - by2, by1 - ay2));
    const overlapWidth = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
    const overlapHeight = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
    const smallestWidth = Math.max(1, Math.min(ax2 - ax1, bx2 - bx1));
    const smallestHeight = Math.max(1, Math.min(ay2 - ay1, by2 - by1));
    const horizontalContact = horizontalGap > 0 && horizontalGap <= COMPLETION_MAX_NEAR_CONTACT_GAP &&
        overlapHeight / smallestHeight >= 0.12;
    const verticalContact = verticalGap > 0 && verticalGap <= COMPLETION_MAX_NEAR_CONTACT_GAP &&
        overlapWidth / smallestWidth >= 0.12;
    if (!horizontalContact && !verticalContact) return null;

    return {
        horizontalGap,
        verticalGap,
        distance: Math.max(horizontalGap, verticalGap),
        axis: horizontalContact ? 'horizontal' : 'vertical'
    };
}

function getBboxCenter(bbox) {
    if (!Array.isArray(bbox) || bbox.length !== 4) return null;
    const [ymin, xmin, ymax, xmax] = bbox;
    return {
        y: (ymin + ymax) / 2,
        x: (xmin + xmax) / 2
    };
}

function isPointInsideBbox(point, bbox, padding = 0) {
    if (!point || !Array.isArray(bbox) || bbox.length !== 4) return false;
    const [ymin, xmin, ymax, xmax] = bbox;
    return point.y >= ymin - padding &&
        point.y <= ymax + padding &&
        point.x >= xmin - padding &&
        point.x <= xmax + padding;
}

function getTextOverlapCount(regionBbox, textLayers = []) {
    if (!Array.isArray(regionBbox) || regionBbox.length !== 4) return 0;
    return textLayers.reduce((count, textLayer) => {
        const textBbox = textLayer?.bbox;
        if (!Array.isArray(textBbox) || textBbox.length !== 4) return count;
        const textCenter = getBboxCenter(textBbox);
        if (isPointInsideBbox(textCenter, regionBbox, 12)) return count + 1;
        const intersection = getBboxIntersection(regionBbox, textBbox);
        if (!intersection) return count;
        const overlap = getBboxArea(intersection) / Math.max(1, getBboxArea(textBbox));
        return overlap >= 0.45 ? count + 1 : count;
    }, 0);
}

function filterOcrLinesAgainstExistingText(lines = [], existingTextLayers = []) {
    return lines.filter(line => {
        const lineText = normalizeTextContent(line);
        if (!lineText || !Array.isArray(line?.bbox) || line.bbox.length !== 4) return false;

        return !existingTextLayers.some(existing => {
            const existingText = normalizeTextContent(existing);
            if (!existingText || !Array.isArray(existing?.bbox) || existing.bbox.length !== 4) return false;
            const intersection = getBboxIntersection(line.bbox, existing.bbox);
            if (!intersection) return false;
            const overlap = getBboxArea(intersection) / Math.max(1, Math.min(getBboxArea(line.bbox), getBboxArea(existing.bbox)));
            if (overlap < 0.45) return false;
            return existingText.includes(lineText) || lineText.includes(existingText);
        });
    });
}

function isUsefulFlatAdOcrSupplement(line) {
    const text = String(line?.textContent || '').trim();
    if (!text || !Array.isArray(line?.bbox) || line.bbox.length !== 4) return false;
    const width = getLineWidth(line);
    const height = getLineHeight(line);
    if (width < 18 || height < 8) return false;
    if (text.length > 120) return false;
    if (isPriceTextLine(line)) return true;
    if (/^www\.|\.com$|https?:\/\//i.test(text)) return true;
    if (height <= 34 && text.length >= 8) return true;
    if (text.length <= 28 && width <= 230) return true;
    return false;
}

function getFlatAdTextOcrRegions(layers = [], editableTextLayers = [], maxRegions = 10, options = {}) {
    const {
        minArea = 900,
        maxArea = 120000,
        minWidth = 45,
        minHeight = 18,
        excludeGenerated = false,
        excludePriceBadge = false
    } = options;
    const candidates = [];
    const seenKeys = new Set();

    for (const layer of layers) {
        const semantic = String(layer?.semanticType || '').toLowerCase();
        const role = String(layer?.designRole || '').toLowerCase();
        const mode = String(layer?.renderMode || '').toLowerCase();
        const bbox = layer?.bbox;
        if (!Array.isArray(bbox) || bbox.length !== 4) continue;
        if (excludeGenerated && layer?.generatedBy === 'flat_ad_layout_analyzer') continue;
        if (excludePriceBadge && (semantic === 'price_badge' || role === 'price_badge')) continue;

        const area = getBboxArea(bbox);
        if (area < minArea || area > maxArea) continue;
        if (semantic === 'ad_background' || role === 'base_background' || mode === 'background_plate') continue;

        const isTextCarrier = semantic === 'shape_panel' ||
            semantic === 'price_badge' ||
            semantic === 'cta_button' ||
            role === 'local_panel' ||
            role === 'price_badge' ||
            mode === 'vector_shape';
        if (!isTextCarrier) continue;

        const key = bbox.map(value => Math.round(value / 8) * 8).join(',');
        if (seenKeys.has(key)) continue;
        seenKeys.add(key);

        const textCount = getTextOverlapCount(bbox, editableTextLayers);
        if (textCount >= 2) continue;
        const [ymin, xmin, ymax, xmax] = bbox;
        const height = ymax - ymin;
        const width = xmax - xmin;
        if (height < minHeight || width < minWidth) continue;
        candidates.push({
            layer,
            bbox,
            area,
            textCount,
            height,
            width
        });
    }

    return candidates
        .sort((a, b) => {
            if (a.textCount !== b.textCount) return a.textCount - b.textCount;
            const aSmall = a.area < 55000 ? 0 : 1;
            const bSmall = b.area < 55000 ? 0 : 1;
            if (aSmall !== bSmall) return aSmall - bSmall;
            return a.area - b.area;
        })
        .slice(0, maxRegions);
}

function getSemanticGroup(layer) {
    if (!layer) return 'unknown';
    if (layer.renderMode === 'vector_shape') return 'vector_shape';
    if (layer.runtimeType === 'background_master' || layer.category === 'background' || layer.layerType === 'background_plate') {
        return 'background';
    }
    if (layer.runtimeType === 'text_node' || layer.semanticType === 'element_text') {
        return 'text';
    }
    if (layer.runtimeType === 'semantic_group' || layer.compositeRole === 'composite_group') {
        return 'group';
    }

    const semantic = String(layer.semanticType || layer.category || layer.name || '').toLowerCase();
    const englishTokens = semantic.split(/[^a-z]+/);
    const isSoftEnglishEntity = ['hair', 'feather', 'fur', 'smoke', 'cloud', 'fog', 'glass', 'curtain', 'veil', 'plant', 'tree', 'grass']
        .some(token => englishTokens.includes(token));
    if (isSoftEnglishEntity || /羽毛|毛发|烟|云|雾|玻璃|纱|植物|树|草/.test(semantic)) {
        return 'soft_edge';
    }
    return 'hard_object';
}

function getCompletionEligibility(layer) {
    if (!layer) {
        return { eligibility: 'excluded', reason: 'missing_layer' };
    }

    const semantic = String(layer.semanticType || '').toLowerCase();
    const designRole = String(layer.designRole || '').toLowerCase();
    const profile = String(layer.extractionProfile || '').toLowerCase();
    const runtimeType = String(layer.runtimeType || '').toLowerCase();
    const renderMode = String(layer.renderMode || '').toLowerCase();
    const text = [layer.name, semantic, designRole, profile, runtimeType, renderMode]
        .map(value => String(value || '').toLowerCase())
        .join(' ');

    if (
        layer.semanticGroup === 'background' ||
        layer.semanticGroup === 'text' ||
        layer.semanticGroup === 'soft_edge' ||
        layer.semanticGroup === 'group'
    ) {
        return { eligibility: 'excluded', reason: `semantic_group_${layer.semanticGroup}` };
    }

    // Product food/drink/packaging can legitimately be assigned a decorative
    // role in a flat ad. Their raster masks are still physical occluders, so
    // do not discard a real foreground lemon/candy before mask validation.
    const isRasterProductAsset =
        renderMode === 'raster_cutout' &&
        ['product_food', 'product_drink', 'product_packaging'].includes(semantic);
    if (isRasterProductAsset) {
        return { eligibility: 'auto', reason: 'eligible_raster_product_asset' };
    }

    if (
        renderMode === 'vector_shape' ||
        renderMode === 'text_css' ||
        runtimeType === 'semantic_group' ||
        /shape_panel|price_badge|cta_button|logo_mark|decor_graphic|flat_ad_layout|element_text/.test(semantic) ||
        /local_panel|price_badge|price_text|label_text|base_background|decor_shape/.test(designRole) ||
        /vector_layout_element|text_layer|background_plate/.test(profile) ||
        /panel|card|badge|button|label|tag|frame|logo|poster|advert|menu|面板|底板|卡片|徽章|价签|按钮|标签|边框|广告|海报|菜单/.test(text)
    ) {
        return { eligibility: 'excluded', reason: 'layout_or_graphic_element' };
    }

    const isExplicitHardEntity =
        /^furniture_/.test(semantic) ||
        /^lighting_/.test(semantic) ||
        semantic === 'product_packaging' ||
        semantic === 'product_food' ||
        semantic === 'product_drink' ||
        ['decor_vase', 'decor_sculpture'].includes(semantic) ||
        /furniture|sofa|chair|stool|table|cabinet|bed|lamp|lighting|chandelier|vase|sculpture|package|furniture|沙发|椅|凳|桌|柜|床|灯|吊灯|花瓶|雕塑|包装/.test(text);

    if (isExplicitHardEntity) {
        return { eligibility: 'auto', reason: 'eligible_hard_entity' };
    }

    // Anything that survived the background/text/layout/soft-edge exclusions
    // is an opaque hard object. Magic Layers runs completion automatically for
    // these objects; requiring a name dictionary here would silently skip
    // valid objects such as console tables, sideboards, benches, and platforms.
    return { eligibility: 'auto', reason: 'eligible_opaque_hard_entity' };
}

function getRecommendedEngine(layer) {
    if (layer.renderMode === 'vector_shape') return 'runtime_vector_or_css';
    if (layer.renderMode === 'text_css') return 'ocr_css';
    if (layer.renderMode === 'background_plate') return 'clean_plate';
    const group = getSemanticGroup(layer);
    if (group === 'background') return 'clean_plate';
    if (group === 'text') return 'ocr_css';
    if (group === 'soft_edge') return 'matting_pending';
    return 'segmentation';
}

function normalizeGraphLayer(layer, index) {
    const id = layer.id || `semantic-layer-${index + 1}`;
    const semanticGroup = getSemanticGroup(layer);
    return {
        layerId: id,
        sourceLayerId: layer.cleanPlateLayerId || layer.sourceTextLayerId || id,
        name: layer.name || `图层 ${index + 1}`,
        bbox: cloneSerializable(layer.extractedBbox || layer.bbox || null),
        zIndex: Number.isFinite(Number(layer.zIndex)) ? Number(layer.zIndex) : index,
        entityId: layer.entityId || layer.semanticEntityId || layer.entity?.id || null,
        parentLayerId: layer.parentLayerId || null,
        compositeRole: layer.compositeRole || 'atomic_object',
        childLayerIds: Array.isArray(layer.childLayerIds) ? cloneSerializable(layer.childLayerIds) : [],
        runtimeType: layer.runtimeType || getRuntimeType(layer),
        semanticType: layer.semanticType || layer.category || 'unknown',
        designRole: layer.designRole || 'unknown',
        renderMode: layer.renderMode || null,
        extractionProfile: layer.extractionProfile || inferExtractionProfile(layer),
        semanticGroup,
        viewRole: layer.viewRole || 'editable-scene',
        extractEngine: layer.extractEngine || getRecommendedEngine(layer),
        quality: {
            status: layer.quality?.status || 'unknown',
            score: layer.quality?.score ?? null,
            reason: layer.quality?.reason || '',
            maskCount: layer.quality?.maskCount ?? null,
            runtimeAction: layer.quality?.runtimeAction || 'unknown',
            shouldGenerateRuntimeLayer: layer.quality?.shouldGenerateRuntimeLayer ?? null,
            needsHigherPrecision: layer.quality?.needsHigherPrecision ?? false,
            issues: Array.isArray(layer.quality?.issues) ? cloneSerializable(layer.quality.issues) : [],
            completionOccluderAudit: cloneSerializable(layer.quality?.completionOccluderAudit) || null,
            recommendedEngine: layer.quality?.recommendedEngine || layer.extractEngine || getRecommendedEngine(layer)
        },
        mask: {
            visibleMaskUrl: layer.maskUrl || null,
            fullMaskUrl: layer.fullMaskUrl || null,
            cutoutUrl: layer.cutoutUrl || null,
            missingMaskUrl: layer.missingMaskUrl || null
        },
        occluders: [],
        occludes: [],
        occludedBy: [],
        completionTaskIds: []
    };
}

function mergeBboxList(bboxes = []) {
    const valid = bboxes.filter(bbox => Array.isArray(bbox) && bbox.length === 4);
    if (!valid.length) return null;
    return [
        Math.min(...valid.map(bbox => bbox[0])),
        Math.min(...valid.map(bbox => bbox[1])),
        Math.max(...valid.map(bbox => bbox[2])),
        Math.max(...valid.map(bbox => bbox[3]))
    ];
}

function getCompletionTaskSignature(task = {}) {
    const target = String(task.targetLayerId || 'unknown');
    const occluders = Array.isArray(task.occluderLayerIds)
        ? [...new Set(task.occluderLayerIds.map(String))].sort().join(',')
        : '';
    const bbox = Array.isArray(task.missingRegionBbox)
        ? task.missingRegionBbox.map(value => Math.round(Number(value) || 0)).join(',')
        : '';
    return `${target}|${occluders}|${bbox}`;
}

function makeCompletionAssetContract(task, layer) {
    const now = Date.now();
    const observedCutoutUrl = layer?.mask?.cutoutUrl || null;
    return {
        id: `completion_asset_${task.targetLayerId}`,
        schemaVersion: 'completion-asset-v1',
        targetLayerId: task.targetLayerId,
        taskId: task.id,
        taskSignature: getCompletionTaskSignature(task),
        status: 'ready_for_completion',
        observedCutout: {
            status: observedCutoutUrl ? 'ready' : 'deferred',
            cutoutUrl: observedCutoutUrl,
            maskUrl: layer?.mask?.visibleMaskUrl || null,
            bbox: cloneSerializable(layer?.bbox || null),
            capturedAt: now,
            source: observedCutoutUrl ? 'magic_layers_extraction' : 'deferred_scene_completion',
            immutable: true
        },
        canonicalAsset: {
            status: 'not_requested',
            cutoutUrl: null,
            maskUrl: null,
            previewUrl: null,
            bbox: null,
            generatedAt: null,
            source: null,
            generationId: null
        },
        composition: {
            activeRepresentation: 'observed_cutout',
            preserveOriginalOcclusion: true,
            sourceLayerVisibility: 'visible',
            originalZIndex: layer?.zIndex ?? null,
            occluderLayerIds: cloneSerializable(task.occluderLayerIds) || [],
            targetEntityLayerIds: cloneSerializable(task.targetEntityLayerIds) || [task.targetLayerId],
            occluderEntityLayerIds: cloneSerializable(task.occluderEntityLayerIds) || cloneSerializable(task.occluderLayerIds) || [],
            placementPolicy: 'canonical_asset_not_auto_composited'
        },
        generation: {
            state: 'not_started',
            attempts: [],
            lastError: null
        },
        createdAt: now,
        updatedAt: now
    };
}

// Keep completed canonical assets only when the underlying occlusion relation is
// unchanged. The observed cutout is deliberately frozen at first capture.
export function reconcileCompletionAssetContracts(item) {
    const graph = item?.semanticViews?.layerGraph;
    if (!graph || !Array.isArray(graph.layers) || !Array.isArray(graph.completionTasks)) return [];

    const priorAssets = Array.isArray(item.semanticViews?.completionAssets)
        ? item.semanticViews.completionAssets
        : [];
    const priorBySignature = new Map(priorAssets.map(asset => [asset.taskSignature, asset]));
    const layersById = new Map(graph.layers.map(layer => [layer.layerId, layer]));
    const nextAssets = [];

    graph.completionTasks.forEach(task => {
        const signature = getCompletionTaskSignature(task);
        const previous = priorBySignature.get(signature);
        const isVerifiedAutoCandidate = task.eligibility === 'auto' && task.maskValidation?.status === 'verified';

        if (!previous && !isVerifiedAutoCandidate) return;
        const asset = previous
            ? cloneSerializable(previous)
            : makeCompletionAssetContract(task, layersById.get(task.targetLayerId));

        asset.taskId = task.id;
        asset.taskSignature = signature;
        asset.updatedAt = Date.now();
        asset.composition = {
            ...(asset.composition || {}),
            activeRepresentation: 'observed_cutout',
            preserveOriginalOcclusion: true,
            sourceLayerVisibility: 'visible',
            originalZIndex: layersById.get(task.targetLayerId)?.zIndex ?? asset.composition?.originalZIndex ?? null,
            occluderLayerIds: cloneSerializable(task.occluderLayerIds) || [],
            placementPolicy: 'canonical_asset_not_auto_composited'
        };

        // The target may have been extracted as an observation-only mask after
        // this contract was first created. Hydrate that missing reference once
        // without replacing an already frozen observation or canonical asset.
        const currentTargetLayer = layersById.get(task.targetLayerId);
        const currentObservedCutoutUrl = currentTargetLayer?.mask?.cutoutUrl || null;
        if (
            currentObservedCutoutUrl &&
            !asset.observedCutout?.cutoutUrl
        ) {
            asset.observedCutout = {
                ...(asset.observedCutout || {}),
                status: 'ready',
                cutoutUrl: currentObservedCutoutUrl,
                maskUrl: currentTargetLayer?.mask?.visibleMaskUrl || null,
                bbox: cloneSerializable(currentTargetLayer?.bbox || null),
                capturedAt: asset.observedCutout?.capturedAt || Date.now(),
                source: 'magic_layers_observation_reference',
                immutable: true
            };
        }

        // A later implementation can advance generation/canonicalAsset states.
        // Revalidation must never make canonical pixels the visible scene source.
        if (asset.canonicalAsset?.status === 'ready') {
            asset.status = 'canonical_ready';
        } else if (isVerifiedAutoCandidate) {
            asset.status = 'ready_for_completion';
        }

        task.assetContractId = asset.id;
        nextAssets.push(asset);
    });

    item.semanticViews.completionAssets = nextAssets;
    nextAssets.forEach(asset => prepareCompletionAssetPreflight(item, asset.id));
    graph.stats = {
        ...(graph.stats || {}),
        completionAssetCount: nextAssets.length,
        readyCompletionAssetCount: nextAssets.filter(asset => asset.status === 'ready_for_completion').length,
        canonicalAssetReadyCount: nextAssets.filter(asset => asset.canonicalAsset?.status === 'ready').length
    };
    return nextAssets;
}

function inferLayerOrder(front, back) {
    if (front.semanticGroup === 'text' && back.semanticGroup !== 'text') return true;
    if (front.semanticGroup !== 'background' && back.semanticGroup === 'background') return true;
    if (front.zIndex !== back.zIndex) return front.zIndex > back.zIndex;
    const frontArea = getBboxArea(front.bbox);
    const backArea = getBboxArea(back.bbox);
    return frontArea <= backArea;
}

export function buildLayerGraph(editableLayers = [], cleanPlateLayers = []) {
    const graphLayers = (Array.isArray(editableLayers) ? editableLayers : [])
        .map((layer, index) => normalizeGraphLayer(layer, index));
    const relations = [];
    const completionCandidates = new Map();
    const byId = new Map(graphLayers.map(layer => [layer.layerId, layer]));
    const completionLayerIndex = createCompletionLayerIndex(graphLayers);

    for (let i = 0; i < graphLayers.length; i += 1) {
        for (let j = i + 1; j < graphLayers.length; j += 1) {
            const a = graphLayers[i];
            const b = graphLayers[j];
            if (a.semanticGroup === 'background' && b.semanticGroup === 'background') continue;

            const intersection = getBboxIntersection(a.bbox, b.bbox);
            const nearContact = intersection ? null : getBboxNearContact(a.bbox, b.bbox);
            if (!intersection && !nearContact) continue;

            const intersectionArea = intersection ? getBboxArea(intersection) : 0;
            const overlapSmallest = intersectionArea / Math.max(1, Math.min(getBboxArea(a.bbox), getBboxArea(b.bbox)));
            if (intersection && overlapSmallest < COMPLETION_MIN_BBOX_OVERLAP) continue;

            const aInFront = inferLayerOrder(a, b);
            const front = aInFront ? a : b;
            const back = aInFront ? b : a;
            const relationId = `occ_${front.layerId}_${back.layerId}`;

            front.occludes.push(back.layerId);
            back.occludedBy.push(front.layerId);

            relations.push({
                id: relationId,
                type: 'occludes',
                sourceLayerId: front.layerId,
                targetLayerId: back.layerId,
                confidence: Number(Math.min(0.95, overlapSmallest).toFixed(3)),
                overlapBbox: intersection,
                nearContact,
                method: intersection ? 'bbox_overlap_prescreen_v2' : 'bbox_near_contact_prescreen_v1'
            });

            const targetCompletion = getCompletionEligibility(back);
            if (targetCompletion.eligibility === 'excluded') continue;

            // A composite occluder is a semantic container, not a reason to
            // discard the relation. Resolve its atomic children once here so
            // every downstream stage receives the same complete foreground
            // entity (all stools, not whichever child happened to overlap).
            const frontEntity = resolveEntityParts(completionLayerIndex, front.layerId);
            const targetEntity = resolveEntityParts(completionLayerIndex, back.layerId);
            const occluderLayers = frontEntity.layers.filter(layer => {
                if (targetEntity.layerIds.includes(layer.layerId)) return false;
                const eligibility = getCompletionEligibility(layer).eligibility;
                return eligibility !== 'excluded';
            });
            if (!occluderLayers.length) continue;
            const occluderCompletions = occluderLayers.map(layer => getCompletionEligibility(layer));
            const hasManualOccluder = occluderCompletions.some(result => result.eligibility === 'manual_review');

            const requiresManualReview =
                targetCompletion.eligibility === 'manual_review' ||
                hasManualOccluder;

            const existing = completionCandidates.get(back.layerId) || {
                targetLayerId: back.layerId,
                occluderLayerIds: [],
                overlapBboxes: [],
                nearContactRelations: [],
                maxOverlapRatio: 0,
                eligibility: requiresManualReview ? 'manual_review' : 'auto',
                reason: requiresManualReview
                    ? `target_${targetCompletion.reason};occluder_composite_review`
                    : targetCompletion.reason
            };
            existing.occluderLayerIds.push(...occluderLayers.map(layer => layer.layerId));
            if (intersection) existing.overlapBboxes.push(intersection);
            if (nearContact) {
                existing.nearContactRelations.push({ occluderLayerId: front.layerId, ...nearContact });
            }
            existing.maxOverlapRatio = Math.max(existing.maxOverlapRatio, overlapSmallest);
            if (requiresManualReview) {
                existing.eligibility = 'manual_review';
                existing.reason = `target_${targetCompletion.reason};occluder_composite_review`;
            }
            completionCandidates.set(back.layerId, existing);
        }
    }

    const completionTasks = [...completionCandidates.values()].map(candidate => {
        const taskId = `complete_${candidate.targetLayerId}`;
        const target = byId.get(candidate.targetLayerId);
        target?.completionTaskIds.push(taskId);
        return {
            id: taskId,
            targetLayerId: candidate.targetLayerId,
            occluderLayerIds: [...new Set(candidate.occluderLayerIds)],
            targetEntityLayerIds: resolveEntityParts(completionLayerIndex, candidate.targetLayerId).layerIds,
            occluderEntityLayerIds: [...new Set(candidate.occluderLayerIds)],
            missingRegionBbox: mergeBboxList(candidate.overlapBboxes),
            nearContactRelations: candidate.nearContactRelations,
            status: 'pending',
            eligibility: candidate.eligibility,
            engine: 'inpainting_pending',
            priority: candidate.maxOverlapRatio >= 0.25 ? 'high' : 'normal',
            bboxOverlapRatio: Number(candidate.maxOverlapRatio.toFixed(3)),
            reason: candidate.reason,
            maskValidation: {
                status: 'pending',
                reason: 'awaiting_extracted_masks'
            }
        };
    });

    return {
        version: 'layer-graph-v2',
        generatedAt: Date.now(),
        layers: graphLayers.map(layer => {
            const cleanLayer = (Array.isArray(cleanPlateLayers) ? cleanPlateLayers : []).find(candidate =>
                candidate.id === layer.sourceLayerId || candidate.id === layer.layerId
            );
            return {
                ...layer,
                cleanPlateLayerId: cleanLayer?.id || layer.sourceLayerId || null
            };
        }),
        relations,
        completionTasks,
        stats: {
            layerCount: graphLayers.length,
            relationCount: relations.length,
            completionTaskCount: completionTasks.length,
            autoCompletionCandidateCount: completionTasks.filter(task => task.eligibility === 'auto').length,
            manualCompletionReviewCount: completionTasks.filter(task => task.eligibility === 'manual_review').length
        },
        notes: [
            'v2 uses bbox overlap as a prescreen only. Completion candidates require post-extraction mask validation before automatic processing.'
        ]
    };
}

function getMaskSource(layer) {
    if (layer?.mask?.cutoutUrl) return { url: layer.mask.cutoutUrl, mode: 'alpha' };
    if (layer?.mask?.visibleMaskUrl) return { url: layer.mask.visibleMaskUrl, mode: 'luminance' };
    if (layer?.mask?.fullMaskUrl) return { url: layer.mask.fullMaskUrl, mode: 'luminance' };
    return null;
}

function loadImageElement(source) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error('mask_image_load_failed'));
        image.src = getProxiedUrl(source);
    });
}

async function loadLayerMask(layer) {
    const source = getMaskSource(layer);
    if (!source || !Array.isArray(layer?.bbox) || layer.bbox.length !== 4) {
        throw new Error('mask_unavailable');
    }
    if (typeof document === 'undefined' || typeof Image === 'undefined') {
        throw new Error('mask_validation_requires_browser');
    }

    const image = await loadImageElement(source.url);
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth || image.width;
    canvas.height = image.naturalHeight || image.height;
    if (!canvas.width || !canvas.height) throw new Error('mask_invalid_dimensions');
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return {
        bbox: layer.bbox,
        width: canvas.width,
        height: canvas.height,
        pixels: context.getImageData(0, 0, canvas.width, canvas.height).data,
        mode: source.mode
    };
}

function sampleMaskAlpha(mask, normalizedY, normalizedX) {
    const [ymin, xmin, ymax, xmax] = mask.bbox;
    if (normalizedY < ymin || normalizedY > ymax || normalizedX < xmin || normalizedX > xmax) return 0;
    const x = Math.min(mask.width - 1, Math.max(0, Math.floor(((normalizedX - xmin) / Math.max(1e-6, xmax - xmin)) * mask.width)));
    const y = Math.min(mask.height - 1, Math.max(0, Math.floor(((normalizedY - ymin) / Math.max(1e-6, ymax - ymin)) * mask.height)));
    const offset = (y * mask.width + x) * 4;
    const alpha = mask.pixels[offset + 3] / 255;
    if (mask.mode === 'alpha') return alpha;
    const luminance = (mask.pixels[offset] * 0.2126 + mask.pixels[offset + 1] * 0.7152 + mask.pixels[offset + 2] * 0.0722) / 255;
    return alpha * luminance;
}

function countAdjacentPixels(targetPixels, occluderPixels, gridSize, radius = 2) {
    let adjacent = 0;
    for (let y = 0; y < gridSize; y += 1) {
        for (let x = 0; x < gridSize; x += 1) {
            const index = y * gridSize + x;
            if (!occluderPixels[index] || targetPixels[index]) continue;
            let touchesTarget = false;
            for (let offsetY = -radius; offsetY <= radius && !touchesTarget; offsetY += 1) {
                for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
                    if (offsetX === 0 && offsetY === 0) continue;
                    const nextX = x + offsetX;
                    const nextY = y + offsetY;
                    if (nextX < 0 || nextY < 0 || nextX >= gridSize || nextY >= gridSize) continue;
                    if (targetPixels[nextY * gridSize + nextX]) {
                        touchesTarget = true;
                        break;
                    }
                }
            }
            if (touchesTarget) adjacent += 1;
        }
    }
    return adjacent;
}

async function validateCompletionTaskMasks(task, layersById, options = {}) {
    const target = layersById.get(task.targetLayerId);
    // The graph builder normally stores expanded IDs. This defensive expansion
    // also upgrades older persisted graphs before mask validation and asset
    // contract creation, so GPT/SAM never see a stale single-child list.
    const completionIndex = createCompletionLayerIndex([...layersById.values()]);
    const targetParts = resolveEntityParts(completionIndex, task.targetLayerId);
    const configuredOccluderIds = [...new Set((task.occluderLayerIds || []).flatMap(id =>
        resolveEntityParts(completionIndex, id).layerIds
    ))].filter(id => !targetParts.layerIds.includes(id));
    task.occluderLayerIds = configuredOccluderIds;
    const configuredOccluders = configuredOccluderIds.map(id => layersById.get(id)).filter(Boolean);
    if (!target || !configuredOccluders.length || !Array.isArray(target.bbox)) {
        return { status: 'unavailable', reason: 'missing_target_or_occluder' };
    }

    // A batch may intentionally extract only part of a semantic scene. Keep
    // unextracted relations as completion context, but do not let an unrelated
    // missing cutout block verification of the selected table/stool pair.
    const unavailableOccluderLayerIds = configuredOccluders
        .filter(layer => !getMaskSource(layer))
        .map(layer => layer.layerId);
    const occluders = configuredOccluders.filter(layer => getMaskSource(layer));
    const unverifiedOccluderLayerIds = configuredOccluders
        .filter(layer => layer.quality?.completionOccluderAudit?.status === 'unverified')
        .map(layer => layer.layerId);
    if (unverifiedOccluderLayerIds.length > 0) {
        return {
            status: 'unavailable',
            reason: 'foreground_occluder_l_review_unverified',
            unverifiedOccluderLayerIds,
            unavailableOccluderLayerIds
        };
    }
    if (!getMaskSource(target) && options.allowDeferredTarget === true) {
        if (!occluders.length) {
            return {
                status: 'unavailable',
                reason: 'no_extracted_occluder_masks',
                unavailableOccluderLayerIds
            };
        }
        try {
            const masks = await Promise.all(occluders.map(loadLayerMask));
            const validationBbox = mergeBboxList([target.bbox, ...occluders.map(layer => layer.bbox)]);
            if (!validationBbox) return { status: 'unavailable', reason: 'validation_bbox_unavailable' };
            const [ymin, xmin, ymax, xmax] = validationBbox;
            const gridSize = Math.max(48, Math.min(128, Number(options.gridSize) || 96));
            let occluderCount = 0;
            for (let y = 0; y < gridSize; y += 1) {
                const normalizedY = ymin + ((y + 0.5) / gridSize) * (ymax - ymin);
                for (let x = 0; x < gridSize; x += 1) {
                    const normalizedX = xmin + ((x + 0.5) / gridSize) * (xmax - xmin);
                    if (masks.some(mask => sampleMaskAlpha(mask, normalizedY, normalizedX) >= 0.35)) {
                        occluderCount += 1;
                    }
                }
            }
            if (!occluderCount) {
                return {
                    status: 'rejected',
                    reason: 'occluder_mask_has_no_coverage_in_validation_region',
                    validatedOccluderLayerIds: occluders.map(layer => layer.layerId),
                    unavailableOccluderLayerIds
                };
            }
            return {
                status: 'verified',
                recommendation: 'auto',
                reason: 'deferred_target_foreground_masks_ready',
                deferredTarget: true,
                coverageRatio: Number((occluderCount / (gridSize * gridSize)).toFixed(3)),
                gridSize,
                validationBbox,
                validatedOccluderLayerIds: occluders.map(layer => layer.layerId),
                unavailableOccluderLayerIds
            };
        } catch (error) {
            return {
                status: 'unavailable',
                reason: error?.message || 'mask_load_failed',
                validatedOccluderLayerIds: occluders.map(layer => layer.layerId),
                unavailableOccluderLayerIds
            };
        }
    }
    if (!getMaskSource(target)) {
        return {
            status: 'unavailable',
            reason: 'target_mask_unavailable',
            unavailableOccluderLayerIds
        };
    }
    if (!occluders.length) {
        return {
            status: 'unavailable',
            reason: 'no_extracted_occluder_masks',
            unavailableOccluderLayerIds
        };
    }

    let masks;
    try {
        masks = await Promise.all([loadLayerMask(target), ...occluders.map(loadLayerMask)]);
    } catch (error) {
        return {
            status: 'unavailable',
            reason: error?.message || 'mask_load_failed',
            validatedOccluderLayerIds: occluders.map(layer => layer.layerId),
            unavailableOccluderLayerIds
        };
    }

    const [targetMask, ...occluderMasks] = masks;
    // Validate in the common scene region rather than only in the target's
    // bbox. Semantic boxes for two touching objects can be adjacent by a few
    // pixels even when the actual silhouettes visibly meet.
    const validationBbox = mergeBboxList([target.bbox, ...occluders.map(layer => layer.bbox)]);
    if (!validationBbox) return { status: 'unavailable', reason: 'validation_bbox_unavailable' };
    const [ymin, xmin, ymax, xmax] = validationBbox;
    const gridSize = Math.max(48, Math.min(128, Number(options.gridSize) || 96));
    const targetPixels = new Uint8Array(gridSize * gridSize);
    const occluderPixels = new Uint8Array(gridSize * gridSize);
    let targetCount = 0;
    let occluderCount = 0;
    let overlapCount = 0;

    for (let y = 0; y < gridSize; y += 1) {
        const normalizedY = ymin + ((y + 0.5) / gridSize) * (ymax - ymin);
        for (let x = 0; x < gridSize; x += 1) {
            const normalizedX = xmin + ((x + 0.5) / gridSize) * (xmax - xmin);
            const index = y * gridSize + x;
            const targetVisible = sampleMaskAlpha(targetMask, normalizedY, normalizedX) >= 0.35;
            const occluderVisible = occluderMasks.some(mask => sampleMaskAlpha(mask, normalizedY, normalizedX) >= 0.35);
            if (targetVisible) {
                targetPixels[index] = 1;
                targetCount += 1;
            }
            if (occluderVisible) {
                occluderPixels[index] = 1;
                occluderCount += 1;
                if (targetVisible) overlapCount += 1;
            }
        }
    }

    if (!targetCount) return { status: 'rejected', reason: 'target_mask_has_no_observed_pixels' };
    if (!occluderCount) return { status: 'rejected', reason: 'occluder_mask_has_no_coverage_in_validation_region' };

    // Segmentation masks of two visibly touching objects often have a narrow
    // transparent seam between them. Three samples at this grid scale keeps
    // that seam from incorrectly invalidating a real occlusion relation.
    const adjacentCount = countAdjacentPixels(targetPixels, occluderPixels, gridSize, 3);
    const coverageRatio = occluderCount / (gridSize * gridSize);
    const contactRatio = (adjacentCount + overlapCount) / occluderCount;
    const metrics = {
        coverageRatio: Number(coverageRatio.toFixed(3)),
        targetVisibleRatio: Number((targetCount / (gridSize * gridSize)).toFixed(3)),
        contactRatio: Number(contactRatio.toFixed(3)),
        gridSize,
        validationBbox,
        validatedOccluderLayerIds: occluders.map(layer => layer.layerId),
        unavailableOccluderLayerIds
    };

    const hasObservedMaskContact = contactRatio > 0;
    const hasVerifiedNearContactPair = hasObservedMaskContact &&
        Array.isArray(task.nearContactRelations) &&
        task.nearContactRelations.some(relation => metrics.validatedOccluderLayerIds.includes(relation.occluderLayerId));
    const hasBboxOcclusionEvidence = Number(task.bboxOverlapRatio) > 0;
    if (contactRatio < 0.03) {
        // A thin transparent seam can reduce the sampled contact ratio below
        // 3% even when the semantic boxes overlap and the two extracted masks
        // share a real boundary. A bbox-near-contact relation alone is not
        // enough: adjacent stools frequently have a small gap and must not be
        // treated as occluders. Require positive mask contact before allowing
        // the near-contact exception.
        if (hasVerifiedNearContactPair || (hasObservedMaskContact && hasBboxOcclusionEvidence)) {
            return {
                status: 'verified',
                recommendation: 'auto',
                reason: hasVerifiedNearContactPair
                    ? 'near_contact_extracted_hard_objects'
                    : 'low_contact_bbox_overlap_confirmed',
                ...metrics
            };
        }
        return { status: 'rejected', reason: 'mask_regions_not_adjacent', ...metrics };
    }
    // Once both masks are present and spatially adjacent, the relation is
    // trustworthy enough for automatic completion. Small overlaps are common
    // for a stool/table edge; large overlaps are exactly the cases completion
    // is meant to recover. Keep the measurements as diagnostics instead of
    // turning them into a hidden manual-confirmation gate.
    if (coverageRatio < 0.08) return { status: 'verified', recommendation: 'auto', reason: 'mask_confirmed_small_occlusion', ...metrics };
    if (coverageRatio > 0.35) return { status: 'verified', recommendation: 'auto', reason: 'mask_confirmed_large_occlusion', ...metrics };
    return { status: 'verified', recommendation: 'auto', reason: 'mask_confirmed_occlusion', ...metrics };
}

export async function validateCompletionCandidatesWithMasks(item, options = {}) {
    const graph = item?.semanticViews?.layerGraph;
    if (!graph || !Array.isArray(graph.completionTasks) || !Array.isArray(graph.layers)) return graph || null;

    const layersById = new Map(graph.layers.map(layer => [layer.layerId, layer]));
    for (const task of graph.completionTasks) {
        if (task.eligibility === 'excluded') continue;
        const validation = await validateCompletionTaskMasks(task, layersById, options);
        task.maskValidation = validation;

        if (validation.status === 'verified') {
            if (task.eligibility === 'auto' && validation.recommendation === 'auto') {
                task.status = 'pending';
                task.reason = validation.reason;
            } else {
                task.eligibility = 'manual_review';
                task.status = 'manual_review';
                task.reason = validation.reason;
            }
        } else if (validation.status === 'unavailable') {
            task.eligibility = 'manual_review';
            task.status = 'manual_review';
            task.reason = validation.reason;
        } else {
            task.eligibility = 'excluded';
            task.status = 'rejected';
            task.reason = validation.reason;
        }
    }

    graph.validatedAt = Date.now();
    graph.stats = {
        ...(graph.stats || {}),
        completionTaskCount: graph.completionTasks.length,
        autoCompletionCandidateCount: graph.completionTasks.filter(task => task.eligibility === 'auto' && task.maskValidation?.status === 'verified').length,
        manualCompletionReviewCount: graph.completionTasks.filter(task => task.eligibility === 'manual_review').length,
        rejectedCompletionCandidateCount: graph.completionTasks.filter(task => task.status === 'rejected').length
    };
    reconcileCompletionAssetContracts(item);
    return graph;
}

function getMotionReadyTextContent(layer) {
    const explicit = String(layer?.textContent || '').trim();
    if (explicit) return explicit;
    return String(layer?.name || '')
        .replace(/^文字[:：]\s*/i, '')
        .replace(/^Text:\s*/i, '')
        .trim();
}

function hasMotionCutoutAsset(layer) {
    if (!layer) return false;
    if (layer.cutoutUrl || layer.previewUrl) return true;
    const activeVersion = Array.isArray(layer.versions)
        ? layer.versions.find(version => version?.id === layer.activeVersionId)
        : null;
    return !!(activeVersion?.cutoutUrl || activeVersion?.previewUrl);
}

function getMotionReadyRole(layer) {
    if (!layer) return 'unknown';
    if (
        layer.category === 'background' ||
        layer.layerType === 'background_plate' ||
        layer.runtimeType === 'background_master'
    ) {
        return 'background';
    }

    const text = [
        layer?.name,
        layer?.semanticType,
        layer?.designRole,
        layer?.renderMode,
        layer?.runtimeType,
        layer?.extractionProfile,
        layer?.textContent
    ].map(value => String(value || '').toLowerCase()).join(' ');
    const semanticType = String(layer?.semanticType || '').toLowerCase();
    const designRole = String(layer?.designRole || '').toLowerCase();
    const runtimeType = String(layer?.runtimeType || '').toLowerCase();
    const renderMode = String(layer?.renderMode || '').toLowerCase();
    const hasTextContent = !!getMotionReadyTextContent(layer);

    if (
        semanticType === 'price_badge' ||
        designRole === 'price_badge'
    ) {
        return hasTextContent ? 'price' : 'panel';
    }

    if (
        semanticType === 'shape_panel' ||
        semanticType === 'cta_button' ||
        designRole === 'local_panel' ||
        renderMode === 'vector_shape'
    ) {
        return 'panel';
    }

    if (
        runtimeType === 'text_node' ||
        semanticType === 'element_text' ||
        renderMode === 'text_css' ||
        /text|文字|文案|标题|logo|品牌|price|价格|价签|¥|\$|￥/.test(text)
    ) {
        if (/logo|brand|品牌/.test(text)) return 'logo';
        if (/price|价格|价签|badge|¥|\$|￥/.test(text)) return 'price';
        return 'text';
    }

    if (
        ['product_food', 'product_drink', 'product_packaging'].includes(String(layer?.semanticType || '').toLowerCase()) ||
        /(product|food|drink|dish|meal|rice|pork|cola|tea|coffee|choco|商品|产品|菜品|食物|饮料|烤肉|炒饭|可乐|热茶|咖啡|杯)/.test(text)
    ) {
        return 'product';
    }

    if (
        /(shape_panel|price_badge|cta_button|card|panel|label|tag|frame|local_panel|色块|标签框|卡片|底板|面板)/.test(text)
    ) {
        return 'panel';
    }

    if (
        /(decor|shape|graphic|sticker|particle|spark|light|flare|ornament|装饰|光效|粒子|贴纸|图形)/.test(text)
    ) {
        return 'decoration';
    }

    return hasMotionCutoutAsset(layer) ? 'decoration' : 'unknown';
}

function isEligibleMotionReadyLayer(layer, role) {
    if (!layer || role === 'background') return false;
    if (layer.runtimeType === 'semantic_group' || layer.compositeRole === 'composite_group') return false;

    const bbox = Array.isArray(layer.bbox) && layer.bbox.length === 4 ? layer.bbox : null;
    const area = getBboxArea(bbox);
    const width = bbox ? Math.max(0, bbox[3] - bbox[1]) : 0;
    const height = bbox ? Math.max(0, bbox[2] - bbox[0]) : 0;
    const hasChildren = Array.isArray(layer.childLayerIds) && layer.childLayerIds.length > 0;
    const semanticType = String(layer?.semanticType || '').toLowerCase();
    const designRole = String(layer?.designRole || '').toLowerCase();
    const renderMode = String(layer?.renderMode || '').toLowerCase();
    const allowsCarrierChildren =
        role === 'panel' ||
        role === 'product' ||
        semanticType === 'shape_panel' ||
        semanticType === 'price_badge' ||
        semanticType === 'product_food' ||
        semanticType === 'product_drink' ||
        designRole === 'local_panel' ||
        designRole === 'price_badge' ||
        designRole === 'product_image' ||
        renderMode === 'vector_shape' ||
        renderMode === 'raster_cutout';

    if (hasChildren && !allowsCarrierChildren) return false;

    if (role === 'text' || role === 'price' || role === 'logo') {
        return !!getMotionReadyTextContent(layer);
    }

    if (!bbox || area <= 0 || width < 20 || height < 20) return false;
    if (role === 'panel') {
        if (area > 520000) return false;
        if (width > 920 || height > 620) return false;
    } else if (role === 'product') {
        if (area > 520000) return false;
    } else if (area > 360000) {
        return false;
    }
    return ['product', 'panel', 'decoration', 'unknown'].includes(role);
}

function getMotionReadyPriority(layer, role) {
    let score = 0;
    if (layer?.viewRole === 'editable-scene') score += 5;
    if (layer?.generatedBy === 'motion_ready_ocr') score += 6;
    if (layer?.runtimeType === 'text_node') score += 5;
    if (hasMotionCutoutAsset(layer)) score += 9;
    if (role === 'product') score += 4;
    if (role === 'price') score += 3;
    if (role === 'logo') score += 2;
    if (role === 'unknown') score -= 4;
    const areaPenalty = getBboxArea(layer?.bbox) / 100000;
    return score - areaPenalty;
}

function buildMotionReadyOcrTextLayers(ocrLines = [], existingLayers = []) {
    const baseLayers = Array.isArray(existingLayers) ? existingLayers : [];
    return mergeFlatAdTextBlocks(ocrLines)
        .filter(line => String(line?.textContent || '').trim())
        .map((line, index) => normalizeLayer({
            id: `motion_text_${index + 1}`,
            name: `文字: ${String(line.textContent || '').trim().slice(0, 16) || index + 1}`,
            semanticType: 'element_text',
            designRole: isPriceTextLine(line) ? 'price_text' : 'label_text',
            renderMode: 'text_css',
            runtimeType: 'text_node',
            layerType: 'foreground_asset',
            category: 'object',
            bbox: cloneSerializable(line.bbox),
            zIndex: baseLayers.length + index + 1,
            editable: false,
            textContent: String(line.textContent || '').trim(),
            fontStyle: line.fontStyle || '',
            css: cloneSerializable(line.css || null),
            generatedBy: 'motion_ready_ocr'
        }, {
            runtimeType: 'text_node',
            viewRole: 'motion-ready'
        }));
}

function dedupeMotionReadyLayers(layers = []) {
    const sorted = [...layers].sort((a, b) => getMotionReadyPriority(b, b.motionRole) - getMotionReadyPriority(a, a.motionRole));
    const kept = [];

    for (const candidate of sorted) {
        const candidateRole = candidate.motionRole || getMotionReadyRole(candidate);
        const candidateText = normalizeTextContent({ textContent: getMotionReadyTextContent(candidate) });
        const duplicate = kept.find(existing => {
            const existingRole = existing.motionRole || getMotionReadyRole(existing);
            if (existingRole !== candidateRole) return false;

            const overlap = getBboxOverlapRatio(existing.bbox, candidate.bbox);
            if (candidateRole === 'text' || candidateRole === 'price' || candidateRole === 'logo') {
                const existingText = normalizeTextContent({ textContent: getMotionReadyTextContent(existing) });
                return overlap >= 0.58 && existingText && candidateText && existingText === candidateText;
            }
            return overlap >= 0.82;
        });

        if (!duplicate) kept.push(candidate);
    }

    return kept.sort((a, b) => (Number(a.zIndex || 0) - Number(b.zIndex || 0)));
}

function buildMotionReadyLayers(editableLayers = [], cleanPlateLayers = [], options = {}) {
    const editable = Array.isArray(editableLayers) ? editableLayers : [];
    const clean = Array.isArray(cleanPlateLayers) ? cleanPlateLayers : [];
    const ocrLines = Array.isArray(options.ocrLines) ? options.ocrLines : [];

    const extraOcrLayers = buildMotionReadyOcrTextLayers(ocrLines, editable);

    const candidatePool = [
        ...editable.map(layer => normalizeLayer({ ...layer }, { viewRole: layer.viewRole || 'editable-scene' })),
        ...clean.map(layer => normalizeLayer({ ...layer }, { viewRole: 'motion-ready-source' })),
        ...extraOcrLayers
    ];

    const motionCandidates = candidatePool
        .map(layer => {
            const role = getMotionReadyRole(layer);
            return normalizeLayer({
                ...layer,
                motionRole: role,
                textContent: getMotionReadyTextContent(layer) || layer.textContent || '',
                extractionProfile: layer.extractionProfile || inferExtractionProfile(layer)
            }, {
                viewRole: layer.viewRole || 'motion-ready'
            });
        })
        .filter(layer => isEligibleMotionReadyLayer(layer, layer.motionRole));

    return dedupeMotionReadyLayers(motionCandidates);
}

export async function buildSemanticLayerViews(baseImage, layers = [], options = {}) {
    const { expandText = false } = options;
    const cleanPlateLayers = cloneSerializable(Array.isArray(layers) ? layers : []);
    const flattenedLayers = flattenSemanticLayers(cleanPlateLayers);
    const groupChildMap = new Map();
    flattenedLayers.forEach(layer => {
        if (!layer.parentLayerId) return;
        const children = groupChildMap.get(layer.parentLayerId) || [];
        children.push(layer.id);
        groupChildMap.set(layer.parentLayerId, children);
    });
    const editableSceneLayers = [];
    let expandedTextCount = 0;
    const seenTextLines = [];
    const isLikelyFlatAd = isLikelyFlatAdLayerSet(flattenedLayers);
    let flatAdOcrTextLines = [];

    if (expandText && isLikelyFlatAd) {
        try {
            const ocrResult = await extractTextFromImage(baseImage, [0, 0, 1000, 1000], { mode: 'document' });
            flatAdOcrTextLines = mergeFlatAdTextBlocks(normalizeOcrLines(ocrResult, [0, 0, 1000, 1000])).slice(0, 32);
            if (flatAdOcrTextLines.length > 0) {
                console.info(`[semantic-layer-views] Flat ad OCR authority returned ${flatAdOcrTextLines.length} text lines.`);
            }
        } catch (error) {
            console.warn('[semantic-layer-views] Failed to run flat ad OCR authority:', error);
        }
    }

    const flatAdPanelSupplements = isLikelyFlatAd
        ? buildFlatAdAtomicPanelSupplements(flattenedLayers, flatAdOcrTextLines)
        : [];
    const flattenedLayoutLayers = removeBroadPanelsCoveredByAtomicPanels(flattenedLayers.concat(flatAdPanelSupplements));

    for (const layer of flattenedLayoutLayers) {
        if (layer.semanticType !== 'element_text') {
            editableSceneLayers.push(normalizeLayer(layer, {
                cleanPlateLayerId: layer.cleanPlateLayerId || layer.id || null,
                childLayerIds: groupChildMap.get(layer.id) || [],
                viewRole: 'editable-scene'
            }));
            continue;
        }

        let textLines = [];
        let textExtractionCache = null;
        const documentOcrLines = findDocumentOcrLinesForLayer(layer, flatAdOcrTextLines);
        if (documentOcrLines.length > 0) {
            textLines = documentOcrLines;
            textExtractionCache = buildTextExtractionCache(textLines, layer, 'semantic_document_ocr');
        }
        if (ENABLE_PER_LAYER_TEXT_OCR_DURING_SEMANTIC_BUILD && expandText && (!isLikelyFlatAd || ENABLE_FLAT_AD_TEXT_OCR_SUPPLEMENT)) {
            try {
                const ocrResult = await extractTextFromImage(baseImage, layer.bbox || [0, 0, 1000, 1000], { mode: 'document' });
                textLines = normalizeOcrLines(ocrResult, layer.bbox || [0, 0, 1000, 1000]);
                if (textLines.length > 0) {
                    textExtractionCache = buildTextExtractionCache(textLines, layer, 'semantic_ocr');
                }
            } catch (error) {
                console.warn('[semantic-layer-views] Failed to expand text layer:', layer.name, error);
            }
        }
        if (!textLines.length) {
            textLines = isLikelyFlatAd && (layer?.textContent || '').trim()
                ? [{
                    textContent: String(layer.textContent || '').trim(),
                    fontStyle: layer.fontStyle || '',
                    css: cloneSerializable(layer.css || null),
                    bbox: layer.bbox
                }]
                : buildTextLinesFromSemanticLayer(layer);
        }

        textLines = dedupeTextLines(isLikelyFlatAd ? textLines : splitCompositeAdTextLines(textLines), seenTextLines);
        if (!textExtractionCache && textLines.length > 0) {
            textExtractionCache = buildTextExtractionCache(textLines, layer, 'semantic_fallback');
        }

        if (!textLines.length) {
            editableSceneLayers.push(normalizeLayer(layer, {
                cleanPlateLayerId: layer.id || null,
                textExtraction: textExtractionCache,
                runtimeType: 'text_node',
                viewRole: 'editable-scene'
            }));
            continue;
        }

        textLines.forEach((line, index) => {
            editableSceneLayers.push(buildEditableTextLayer({
                ...layer,
                textExtraction: textExtractionCache
            }, line, index));
            expandedTextCount += 1;
        });
    }

    const addOcrTextLines = (lines, sourceLayer, sourceLabel, maxLines = 24) => {
        const dedupedLines = dedupeTextLines(splitCompositeAdTextLines(lines), seenTextLines).slice(0, maxLines);
        dedupedLines.forEach((line, index) => {
            editableSceneLayers.push(buildEditableTextLayer(sourceLayer, line, index));
            expandedTextCount += 1;
        });
        if (dedupedLines.length > 0) {
            console.info(`[semantic-layer-views] Added ${dedupedLines.length} flat ad OCR text lines from ${sourceLabel}.`);
        }
        return dedupedLines.length;
    };

    if (ENABLE_REGION_TEXT_OCR_SUPPLEMENT && expandText && isLikelyFlatAd && ENABLE_FLAT_AD_TEXT_OCR_SUPPLEMENT) {
        const ocrRegions = getFlatAdTextOcrRegions(flattenedLayoutLayers, editableSceneLayers, 4, {
            minArea: 32000,
            maxArea: 180000,
            minWidth: 260,
            minHeight: 70,
            excludeGenerated: true,
            excludePriceBadge: true
        });
        for (const region of ocrRegions) {
            try {
                const ocrResult = await extractTextFromImage(baseImage, region.bbox, { mode: 'document' });
                const regionTextLines = filterOcrLinesAgainstExistingText(
                    mergeFlatAdTextBlocks(normalizeOcrLines(ocrResult, region.bbox)),
                    editableSceneLayers
                ).slice(0, 3);
                if (!regionTextLines.length) continue;
                const regionTextLayer = normalizeLayer({
                    id: `flat_ad_region_text_${region.layer.id || editableSceneLayers.length}`,
                    name: `${region.layer.name || '底板'}文字补全`,
                    semanticType: 'element_text',
                    designRole: region.layer.semanticType === 'price_badge' ? 'price_text' : 'label_text',
                    renderMode: 'text_css',
                    layerType: 'foreground_asset',
                    category: 'object',
                    bbox: region.bbox,
                    zIndex: Number(region.layer.zIndex || 1) + 1,
                    editable: true,
                    textContent: ''
                }, {
                    runtimeType: 'text_node',
                    viewRole: 'editable-scene'
                });
                addOcrTextLines(regionTextLines, regionTextLayer, region.layer.name || 'region', 3);
            } catch (error) {
                console.warn('[semantic-layer-views] Failed to supplement flat ad text via region OCR:', region.layer?.name, error);
            }
        }
    }

    const finalEditableSceneLayers = pruneCompositeDuplicateTextLayers(editableSceneLayers)
        .sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));

    const cleanPlateSourceLayers = isLikelyFlatAd ? flattenedLayoutLayers : cleanPlateLayers;
    const normalizedCleanPlateLayers = cleanPlateSourceLayers.map(layer => normalizeLayer(layer, { viewRole: 'clean-plate' }));
    const motionReadyLayers = buildMotionReadyLayers(finalEditableSceneLayers, normalizedCleanPlateLayers, {
        ocrLines: flatAdOcrTextLines
    });
    const layerGraph = buildLayerGraph(finalEditableSceneLayers, normalizedCleanPlateLayers);

    return {
        cleanPlateLayers: normalizedCleanPlateLayers,
        editableSceneLayers: finalEditableSceneLayers,
        motionReadyLayers,
        layerGraph,
        stats: {
            expandedTextCount,
            cleanPlateLayerCount: cleanPlateLayers.length,
            editableLayerCount: finalEditableSceneLayers.length,
            motionReadyLayerCount: motionReadyLayers.length,
            layerGraphRelationCount: layerGraph.stats.relationCount,
            completionTaskCount: layerGraph.stats.completionTaskCount
        }
    };
}

export function applySemanticLayerViewsToItem(item, views) {
    if (!item || !views) return item;

    item.semanticViews = {
        cleanPlateLayers: cloneSerializable(views.cleanPlateLayers) || [],
        editableSceneLayers: cloneSerializable(views.editableSceneLayers) || [],
        motionReadyLayers: cloneSerializable(views.motionReadyLayers) || [],
        layerGraph: cloneSerializable(views.layerGraph) || null,
        completionAssets: [],
        stats: cloneSerializable(views.stats) || {},
        generatedAt: Date.now()
    };

    item.motionReadyLayers = cloneSerializable(views.motionReadyLayers) || [];
    item.layers = cloneSerializable(views.editableSceneLayers) || [];
    item.scene = item.scene || {};
    item.scene.layers = cloneSerializable(views.editableSceneLayers) || [];
    return item;
}

export function getCleanPlateLayers(item) {
    if (Array.isArray(item?.semanticViews?.cleanPlateLayers) && item.semanticViews.cleanPlateLayers.length > 0) {
        return item.semanticViews.cleanPlateLayers;
    }
    return item?.scene?.layers || item?.layers || [];
}

function getBboxOverlapRatio(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) return 0;
    const [ay1, ax1, ay2, ax2] = a;
    const [by1, bx1, by2, bx2] = b;
    const overlapX = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
    const overlapY = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
    const overlap = overlapX * overlapY;
    const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
    const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
    const smallest = Math.min(areaA, areaB);
    return smallest > 0 ? overlap / smallest : 0;
}

export function findCleanPlateLayerForEditableLayer(item, editableLayer) {
    if (!item || !editableLayer || Array.isArray(editableLayer)) return null;

    const cleanLayers = getCleanPlateLayers(item);
    if (!Array.isArray(cleanLayers) || cleanLayers.length === 0) return null;

    const candidateIds = [
        editableLayer.cleanPlateLayerId,
        editableLayer.sourceTextLayerId,
        editableLayer.id
    ].filter(Boolean);

    const byId = cleanLayers.find(layer => candidateIds.includes(layer.id));
    if (byId) return byId;

    const byName = cleanLayers.find(layer => layer.name && layer.name === editableLayer.name);
    if (byName) return byName;

    let best = null;
    let bestScore = 0;
    for (const layer of cleanLayers) {
        if (editableLayer.semanticType && layer.semanticType && editableLayer.semanticType !== layer.semanticType) {
            continue;
        }

        const score = getBboxOverlapRatio(editableLayer.bbox, layer.bbox);
        if (score > bestScore) {
            best = layer;
            bestScore = score;
        }
    }

    return bestScore >= 0.5 ? best : null;
}

export function getCleanupLayerForEditableLayer(item, editableLayer, options = {}) {
    if (!editableLayer || Array.isArray(editableLayer)) return editableLayer;

    const { preferEditableTextBbox = true } = options;
    const isTextLayer = editableLayer.semanticType === 'element_text' || editableLayer.runtimeType === 'text_node';

    if (isTextLayer && preferEditableTextBbox) {
        return editableLayer;
    }

    return findCleanPlateLayerForEditableLayer(item, editableLayer) || editableLayer;
}

function isBackgroundCandidateLayer(layer) {
    if (!layer) return false;
    const semanticType = String(layer.semanticType || '').toLowerCase();
    const designRole = String(layer.designRole || '').toLowerCase();
    const renderMode = String(layer.renderMode || '').toLowerCase();
    const runtimeType = String(layer.runtimeType || '').toLowerCase();
    const layerType = String(layer.layerType || '').toLowerCase();
    const category = String(layer.category || '').toLowerCase();
    const name = String(layer.name || '').toLowerCase();
    return (
        renderMode === 'background_plate' ||
        semanticType === 'ad_background' ||
        designRole === 'base_background' ||
        runtimeType === 'background_master' ||
        layerType === 'background_plate' ||
        category === 'background' ||
        name.includes('background') ||
        name.includes('背景') ||
        name.includes('底板') ||
        name.includes('底色')
    );
}

function scoreBackgroundCandidate(layer) {
    if (!layer) return -1;
    let score = 0;
    const semanticType = String(layer.semanticType || '').toLowerCase();
    const designRole = String(layer.designRole || '').toLowerCase();
    const renderMode = String(layer.renderMode || '').toLowerCase();
    const runtimeType = String(layer.runtimeType || '').toLowerCase();
    const bbox = Array.isArray(layer.bbox) ? layer.bbox : null;

    if (renderMode === 'background_plate') score += 5;
    if (semanticType === 'ad_background') score += 4;
    if (designRole === 'base_background') score += 3;
    if (runtimeType === 'background_master') score += 2;
    if (bbox && bbox.length === 4) {
        const area = Math.max(0, Number(bbox[2]) - Number(bbox[0])) * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
        score += area / 1000000;
    }
    return score;
}

export function getBackgroundSemanticHint(item) {
    const candidatePools = [
        item?.semanticViews?.cleanPlateLayers,
        item?.semanticViews?.editableSceneLayers,
        item?.scene?.layers,
        item?.layers
    ];

    const seen = new Set();
    const candidates = [];
    candidatePools.forEach(pool => {
        if (!Array.isArray(pool)) return;
        pool.forEach(layer => {
            if (!isBackgroundCandidateLayer(layer)) return;
            const key = layer.id || `${layer.name || 'background'}:${Array.isArray(layer.bbox) ? layer.bbox.join(',') : 'no-bbox'}`;
            if (seen.has(key)) return;
            seen.add(key);
            candidates.push(layer);
        });
    });

    if (!candidates.length) return '';

    const best = candidates.sort((a, b) => scoreBackgroundCandidate(b) - scoreBackgroundCandidate(a))[0];
    if (!best) return '';

    const parts = [
        best.name ? `background layer name: ${best.name}` : '',
        best.semanticType ? `semantic type: ${best.semanticType}` : '',
        best.designRole ? `design role: ${best.designRole}` : '',
        best.promptHint ? `visual hint: ${best.promptHint}` : '',
        best.reasoning ? `semantic note: ${best.reasoning}` : ''
    ].filter(Boolean);

    return parts.join('; ');
}

function isSameLayerByIdentity(layer, identity = {}) {
    if (!layer) return false;
    const ids = [
        identity.layerId,
        identity.id,
        identity.cleanPlateLayerId,
        identity.sourceTextLayerId
    ].filter(Boolean);

    if (ids.some(id => [layer.id, layer.cleanPlateLayerId, layer.sourceTextLayerId].includes(id))) {
        return true;
    }

    if (identity.name && layer.name === identity.name) return true;
    return false;
}

function applyExtractionMetadataToLayer(layer, metadata = {}) {
    if (!layer) return layer;
    if (metadata.extractEngine) layer.extractEngine = metadata.extractEngine;
    if (metadata.quality) layer.quality = cloneSerializable(metadata.quality);
    if (metadata.bbox) layer.extractedBbox = cloneSerializable(metadata.bbox);
    if (metadata.cutoutUrl) layer.cutoutUrl = metadata.cutoutUrl;
    if (metadata.previewUrl) layer.previewUrl = metadata.previewUrl;
    if (metadata.shapeStyleText !== undefined) layer.shapeStyleText = metadata.shapeStyleText;
    if (metadata.clipPath !== undefined) layer.clipPath = metadata.clipPath;
    if (metadata.shapeType !== undefined) layer.shapeType = metadata.shapeType;
    if (metadata.fillColor !== undefined) layer.fillColor = metadata.fillColor;
    return layer;
}

function updateLayerArrayMetadata(layers, identity, metadata) {
    if (!Array.isArray(layers)) return false;
    let updated = false;
    layers.forEach(layer => {
        if (isSameLayerByIdentity(layer, identity)) {
            applyExtractionMetadataToLayer(layer, metadata);
            updated = true;
        }
    });
    return updated;
}

export function updateLayerExtractionMetadata(item, identity = {}, metadata = {}) {
    if (!item) return item;

    updateLayerArrayMetadata(item.layers, identity, metadata);
    updateLayerArrayMetadata(item.scene?.layers, identity, metadata);
    updateLayerArrayMetadata(item.semanticViews?.editableSceneLayers, identity, metadata);
    updateLayerArrayMetadata(item.semanticViews?.cleanPlateLayers, identity, metadata);
    updateLayerArrayMetadata(item.semanticViews?.motionReadyLayers, identity, metadata);
    updateLayerArrayMetadata(item.motionReadyLayers, identity, metadata);

    if (item.semanticViews) {
        const cleanLayers = item.semanticViews.cleanPlateLayers || [];
        const editableLayers = item.semanticViews.editableSceneLayers || item.scene?.layers || item.layers || [];
        item.semanticViews.motionReadyLayers = buildMotionReadyLayers(editableLayers, cleanLayers);
        item.motionReadyLayers = cloneSerializable(item.semanticViews.motionReadyLayers) || [];
        item.semanticViews.layerGraph = buildLayerGraph(editableLayers, cleanLayers);
        reconcileCompletionAssetContracts(item);
        item.semanticViews.stats = {
            ...(item.semanticViews.stats || {}),
            motionReadyLayerCount: item.semanticViews.motionReadyLayers.length,
            layerGraphRelationCount: item.semanticViews.layerGraph.stats.relationCount,
            completionTaskCount: item.semanticViews.layerGraph.stats.completionTaskCount
        };
    }

    return item;
}

export function refreshMotionReadyViewsForItem(item) {
    if (!item?.semanticViews) return item;

    const cleanLayers = item.semanticViews.cleanPlateLayers || [];
    const editableLayers = item.semanticViews.editableSceneLayers || item.scene?.layers || item.layers || [];
    item.semanticViews.motionReadyLayers = buildMotionReadyLayers(editableLayers, cleanLayers);
    item.motionReadyLayers = cloneSerializable(item.semanticViews.motionReadyLayers) || [];
    item.semanticViews.layerGraph = buildLayerGraph(editableLayers, cleanLayers);
    reconcileCompletionAssetContracts(item);
    item.semanticViews.stats = {
        ...(item.semanticViews.stats || {}),
        motionReadyLayerCount: item.semanticViews.motionReadyLayers.length,
        layerGraphRelationCount: item.semanticViews.layerGraph.stats.relationCount,
        completionTaskCount: item.semanticViews.layerGraph.stats.completionTaskCount
    };

    return item;
}
