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
    calculateSmartPosition 
} from './workbench-core.js';
import { renderLayerList, getLayerState, updateLayerState, showLayerManagerModal, updateFusionUI } from './modals.js';

import { renderCanvasLayers } from './workbench/layers.js';
import { renderSceneToCanvas } from './workbench/renderer.js';
import { dilateAlphaChannel } from '../graphics/mask-utils.js';
import { cleanBackground, cleanMultipleBackgrounds } from './workbench/layer-assets.js';
import { globalMatteTaskSystem } from '../graphics/matte-task-system.js';
import { StrategyDispatcher } from '../ai-services/strategy-dispatcher.js';
import { buildSemanticLayerViews, applySemanticLayerViewsToItem, getCleanupLayerForEditableLayer, updateLayerExtractionMetadata } from '../services/semantic-layer-views.js';
import { segmentLayers, segmentSingleLayer } from '../services/segmentation-service.js';
import { buildExtractedTextState } from './text-style-utils.js';
import { filterTextLinesToBbox, getCachedTextExtraction, normalizeOcrTextLines } from '../services/text-extraction-cache.js';
import { prepareTextContainerCandidates, restoreTextContainerShapes } from './text-container-restore.js';
import { ensureLayerPanelFooter } from './modals.js';

const DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST = true;

function cloneSerializable(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
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
    return (
        ['vector_shape', 'background_plate', 'deferred'].includes(renderMode) ||
        ['base_background', 'local_panel', 'price_badge', 'decor_shape'].includes(designRole) ||
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

async function persistLayerStateToRuntime(itemId, item) {
    const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
    const asset = workspace ? workspace.currentState.assetRegistry.get(itemId) : null;

    if (workspace && asset) {
        workspace.dispatcher.dispatch({
            type: 'UPDATE_ASSET_METADATA',
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

    if (state.currentSessionId && window.dbHelper?.saveSession) {
        const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
        if (currentSession) {
            await window.dbHelper.saveSession(currentSession);
        }
    }
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
export async function triggerLayerExplosion(itemId) {
    console.log(`[Explosion] Triggered for item: ${itemId}`);
    const item = workbenchItems.get(itemId);
    if (!item) {
        alert("找不到图片对象");
        return;
    }

    // 1. 如果还没有图层数据，先自动分析
    const layersToUse = item.scene && item.scene.layers ? item.scene.layers : item.layers;

    if (!layersToUse || layersToUse.length === 0) {
        addMessage({ sender: 'bot', type: 'text', content: '🔍 正在分析图层结构以进行拆解...' });
        try {
            const analysisResult = await analyzeImageLayers(item.file || item.dataUrl);
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
            addMessage({ sender: 'bot', type: 'text', content: '❌ 无法识别图层，拆解中止。' });
            return;
        }
    }

    // 2. 准备拆解任务
    const executeLayerExplosion = async (customPrompt) => {
        const allLayers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
        const currentLayers = [];
        
        allLayers.forEach((layer, index) => {
            const state = getLayerState(itemId, index);
            if (state.selected && !state.locked) {
                currentLayers.push(layer);
            }
        });

        if (currentLayers.length === 0) {
            addMessage({ sender: 'bot', type: 'text', content: '❌ 拆解中止：未能找到被勾选且未锁定的图层。' });
            return;
        }

        const layerNames = currentLayers.map(l => l.name);
        const batchAutoOpenToken = `explode-batch-${itemId}-${Date.now()}`;
        const tempMsg = addMessage({ 
            sender: 'bot', 
            type: 'text', 
            content: `💣 **启动语义拆解模式**\n正在批量提取 **${layerNames.length}** 个独立图层：[${layerNames.join(', ')}]...` 
        });

        try {
            const parentImg = new Image();
            parentImg.crossOrigin = "anonymous";
            const itemSrc = getProxiedUrl(item.dataUrl);
            if (!itemSrc) {
                throw new Error('无效的原图地址，无法执行图层拆解');
            }
            parentImg.src = itemSrc;
            await new Promise(r => parentImg.onload = r);

            let fastSamResults = new Map();
            const objectLayers = currentLayers.filter(layer => isRasterSegmentationRuntimeLayer(layer));
            if (objectLayers.length > 0) {
                try {
                    fastSamResults = await segmentLayers({
                        item,
                        layers: objectLayers,
                        onProgress: (message) => tempMsg.update(message)
                    });
                } catch (error) {
                    if (DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST) {
                        throw new Error(`FastSAM 批量分割失败，已阻止进入 Gemini 兜底: ${error.message || error}`);
                    }
                    console.warn('[FastSAM] Batch segmentation unavailable, fallback to legacy extraction:', error);
                }
            }

            // 3. 批量生成 Promise (并行提取每个物体)
            const originalTextSource = item.originalDataUrl || item.file || item.dataUrl;
            const promises = currentLayers.map(async (layer, index) => {
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

                const segmented = fastSamResults.get(layer);
                if (segmented?.dataUrl) {
                    return {
                        status: 'fulfilled',
                        value: {
                            success: true,
                            dataUrl: segmented.dataUrl,
                            width: segmented.width,
                            height: segmented.height,
                            segmentedBbox: segmented.bbox || layer.bbox,
                            extractEngine: segmented.extractEngine || 'fastsam',
                            quality: segmented.quality || null,
                            shouldGenerateRuntimeLayer: segmented.shouldGenerateRuntimeLayer !== false,
                            runtimeAction: segmented.runtimeAction || segmented.quality?.runtimeAction || 'accept'
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
            });

            // 4. 等待所有提取完成
            const results = await Promise.all(promises);
            
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
                    } else if (res.value.isFlatDesignLayer || res.value.shouldGenerateRuntimeLayer === false || res.value.runtimeAction === 'hold') {
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
            if (cleanupLayers.length > 0) {
                addMessage({ sender: 'bot', type: 'text', content: `🧹 正在净化底板，联合抹除已提取的 ${cleanupLayers.length} 个元素...` });
                const baseBg = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
                
                const customPrompt = hasTextLayers ? "Remove all the specified objects, including any text, letters, logomarks, or typography enclosed within the masked white areas. Inpaint and fill the background seamlessly matching the surrounding texture." : undefined;
                const cleanedBg = await cleanMultipleBackgrounds(baseBg, cleanupLayers, customPrompt, {
                    preserveBackgroundOnly: true
                });
                
                if (cleanedBg) {
                    item.cleanPlateDataUrl = cleanedBg;
                    item.dataUrl = cleanedBg;
                    item.cleanPlateStatus = 'ready';
                    
                    // Hide extracted layers (both image and text) in the list since they are now separate objects
                    currentLayers.forEach(layerObj => {
                        const layerIndex = allLayers.findIndex(l => (l.name || l) === layerObj.name);
                        if (layerIndex >= 0) {
                            updateLayerState(itemId, layerIndex, { visible: false, selected: false });
                        }
                    });
                    renderLayerList(allLayers, itemId);
                    renderCanvasLayers(itemId);
                    await persistLayerStateToRuntime(itemId, item);
                    if (window.historyManager) window.historyManager.pushState();

                    // Update the base image element in place
                    if (item.el) {
                        const imgEl = item.el.querySelector('.crop-container > img');
                        if (imgEl) {
                            imgEl.src = cleanedBg;
                        }
                    }
                } else {
                    console.warn("[Background Purification] Could not clean the backgrounds. Using original plate.");
                }
            } else {
                console.warn("[Background Purification] No successfully extracted layers to clean up.");
            }
            
            // 6. 将结果添加到工作台
            const baseX = parseFloat(item.el.style.left) || 0;
            const baseY = parseFloat(item.el.style.top) || 0;
            const itemWidth = parseFloat(item.el.style.width) || 300;
            const itemHeight = parseFloat(item.el.style.height) || 300;
            
            let successCount = 0;
            let heldCount = 0;

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
                    if (res.value.shouldGenerateRuntimeLayer === false || res.value.runtimeAction === 'hold') {
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
                            bbox: res.value.segmentedBbox || layerObj.bbox
                        });
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
                        originalBbox,
                        layerName: res.layerName,
                        type: 'layer-explode',
                        zIndex: getExtractedLayerZIndex(item, layerObj, i),
                        extractEngine: res.value.extractEngine || 'fastsam',
                        quality: res.value.quality || null,
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
                        bbox: res.value.segmentedBbox || layerObj.bbox
                    });
                    
                    successCount++;
                }
            } else {
                console.warn(`Layer extraction failed for ${res.layerName}:`, res.reason);
            }
            }

            if (tempMsg) tempMsg.remove();
            if (successCount > 0) {
                const heldText = heldCount > 0 ? `，${heldCount} 个低质量候选已保留为待高精度处理` : '';
                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat('语义拆解', `拆解了 ${successCount}/${layerNames.length} 个图层${heldText}: ${layerNames.join(', ')}`, item.dataUrl || item.file, executeLayerExplosion);
                } else {
                    addMessage({ sender: 'bot', type: 'text', content: `✅ **拆解完成**！成功提取并还原了 ${successCount}/${layerNames.length} 个完整图层${heldText}。` });
                }
            } else if (heldCount > 0) {
                addMessage({ sender: 'bot', type: 'text', content: `⚠️ **拆解已暂停生成图层**：${heldCount} 个候选质量不足，已标记为需要高精度模型处理。` });
            } else {
                addMessage({ sender: 'bot', type: 'text', content: `❌ 拆解失败，未能提取任何图层。` });
            }

        } catch (e) {
            console.error("[Explosion] Fatal error:", e);
            if (tempMsg) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `❌ 拆解过程出错: ${e.message}` });
        }
    };

    await executeLayerExplosion();
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
        addMessage({ sender: 'bot', type: 'text', content: '🔍 Magic Layers 正在先执行全图语义分析...' });
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
        content: `✨ **Magic Layers 启动**\n已自动选中 ${selectedIndices.length} 个非背景图层，开始语义提取与一次性背景净化。`
    });

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
export async function handleQuickFusionSync(childId, promptText) {
    const childItem = workbenchItems.get(childId);
    const parentItem = workbenchItems.get(childItem.parentId);
    if (!childItem || !parentItem) return;

    const executeQuickFusionSync = async (customPrompt) => {
        const promptToUse = customPrompt || promptText;
        const tempMsg = addMessage({ sender: 'bot', type: 'text', content: `🔄 **正在同步至原图**\n正在将针对“${childItem.layerName || '子图'}”的修改（${promptToUse}）同步回主场景中...` });

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

            ctx.save();
            ctx.translate(drawX + drawW / 2, drawY + drawH / 2);
            if (rotation !== 0) {
                ctx.rotate(rotation);
            }
            ctx.fillRect(-drawW / 2, -drawH / 2, drawW, drawH);
            ctx.restore();
            
            const preciseMask = maskCanvas.toDataURL('image/png');

            // 1.5. 准备底图：使用渲染器获取当前工作台的视觉状态
            const baseCanvas = await renderSceneToCanvas(parentItem.id);
            if (!baseCanvas) throw new Error("无法渲染场景");

            const baseDataUrl = baseCanvas.toDataURL('image/png');
            const baseFile = await dataURLToFile(baseDataUrl, `base-${Date.now()}.png`);

            // 2. 调用 Gemini 进行融合重绘
            const syncPrompt = `${promptToUse}. Change only the visual look of the object within the masked area. Keep the background and perspective of the rest of the scene perfect.`;
            
            let result = await editOrQueryImageWithGemini(syncPrompt, baseFile, [], preciseMask);

            if (result && result.success && result.imageData) {
                const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
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
export async function handleIsolatedAssetEdit(childId, promptText) {
    const childItem = workbenchItems.get(childId);
    if (!childItem) return;

    const executeIsolatedEdit = async (customPrompt) => {
        const promptToUse = customPrompt || promptText;
        const tempMsg = addMessage({ sender: 'bot', type: 'text', content: `🎨 **独立资产编辑中**\n正在对“${childItem.layerName || '子图'}”进行独立修改（${promptToUse}）...` });

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
                addMessage({ sender: 'bot', type: 'text', content: `⚡ **前端增强秒改触发**\n检测到纯颜色编辑指令，启动针对半透明材质(如头纱)的高光保护与渐变映射渲染算法...` });
                
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
                const finalAssetFile = await dataURLToFile(finalAssetDataUrl, `colorized-asset-${Date.now()}.png`);

                const smartPos = calculateSmartPosition(childItem.el, 2); 
                const newAssetId = await addImageToWorkbench(finalAssetFile, `颜色秒改: ${childItem.layerName || '子图'}`, {
                    x: smartPos.x,
                    y: smartPos.y,
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

                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat(`独立编辑 [${childItem.layerName || '子图'}]`, promptToUse, finalAssetDataUrl, executeIsolatedEdit);
                } else {
                    addMessage({ sender: 'bot', type: 'text', content: `✅ **秒改完成**！已为您生成新的调整目标图。`});
                }
                return; // 直接拦截后续 AI 生成流程
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

            const avgR = strategy.metrics?.bgMean?.r || 127;
            const avgG = strategy.metrics?.bgMean?.g || 127;
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

            // Hardcode logic for specific name hints just in case
            if ((childItem.layerName || '').includes('云') || (childItem.layerName || '').includes('雾') || (childItem.layerName || '').includes('光')) {
                bestBackdrop = backdrops[1]; // Black / Magenta in original V5 fallback
            } else if ((childItem.layerName || '').includes('皮') || (childItem.layerName || '').includes('木') || (childItem.layerName || '').includes('人')) {
                bestBackdrop = backdrops[2]; // Green
            } else if ((childItem.layerName || '').includes('树') || (childItem.layerName || '').includes('草') || (childItem.layerName || '').includes('叶')) {
                bestBackdrop = backdrops[3]; // Magenta
            }

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

            // 3. 调用 Gemini 进行独立资产重绘材质与造型
            // FORCE standard generation in edit mode. Overrides useMaskFormat.
            let isolatedPrompt = `Act as a precise image asset editor.
Your task is to modify the object "${childItem.layerName || 'object'}" in the image according to this instruction: "${promptToUse}".

CRITICAL REQUIREMENTS:
1. ZERO HALLUCINATION: You MUST NOT redraw or alter the bounding box shape of the object unless instructed.
2. BACKGROUND RULE: Your ONLY task regarding the background is to keep it a perfectly solid, uniform ${bgColorName} color (${bgColorHex}). ${bgColorRule} Do not add shadows, gradients, or environment.
3. OBJECT ISOLATION: Only modify the object itself based on the instruction. Maintain its original scale and general position.
4. HIGH QUALITY: Ensure the modified object has realistic textures and lighting.`;

            let result = await editOrQueryImageWithGemini(isolatedPrompt, paddedFile);

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
            // 强制将 Gemini 吐回来的任意尺寸图片，重新拉伸/缩放贴合我们的物理画布
            tempCtx.drawImage(reconstructedImg, 0, 0, bgCanvas.width, bgCanvas.height);

            const generatedImageDataObj = tempCtx.getImageData(0, 0, tempWorkspaceCanvas.width, tempWorkspaceCanvas.height);
            const taskId = `iso_asset_${Date.now()}`;

            // 针对实体服装等需要走严格固体抠图算法的，传指定 bgColor 避免依赖边缘猜色。
            // 针对云雾则留空，让其内部降级为 channel matting
            const extractionConfig = useMaskFormat ? {} : { type: 'solid', bgColor: bestBackdrop.rgb };

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

            // 裁切回原始尺寸 (Crop padding back)
            const finalCanvas = document.createElement('canvas');
            finalCanvas.width = childImg.naturalWidth;
            finalCanvas.height = childImg.naturalHeight;
            const finalCtx = finalCanvas.getContext('2d');
            finalCtx.drawImage(tempWorkspaceCanvas, padX, padY, childImg.naturalWidth, childImg.naturalHeight, 0, 0, childImg.naturalWidth, childImg.naturalHeight);

            const finalAssetDataUrl = finalCanvas.toDataURL('image/png');
            const finalAssetFile = await dataURLToFile(finalAssetDataUrl, `edited-asset-${Date.now()}.png`);

            // 6. 将新的透明资产添加到工作台
            const smartPos = calculateSmartPosition(childItem.el, 2); // 2=下方
            const newAssetId = await addImageToWorkbench(finalAssetFile, `独立编辑: ${childItem.layerName || '子图'}`, {
                x: smartPos.x,
                y: smartPos.y,
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
            if (window.addWorkbenchActionToChat) {
                await window.addWorkbenchActionToChat(`独立编辑 [${childItem.layerName || '子图'}]`, promptToUse, finalAssetDataUrl, executeIsolatedEdit);
            } else {
                addMessage({ sender: 'bot', type: 'text', content: `✅ **独立编辑完成**！已为您生成新的透明资产：${promptToUse}` });
            }
        } catch (e) {
            console.error("Isolated Edit Failed:", e);
            if (tempMsg && tempMsg.parentNode) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `❌ 独立编辑失败: ${e.message}` });
        }
    };

    await executeIsolatedEdit();
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
            onProgress: (message) => console.log(`[FastSAM Box ${layerName}] ${message}`)
        });

        if (segmented?.dataUrl) {
            if (segmented.shouldGenerateRuntimeLayer === false || segmented.runtimeAction === 'hold') {
                layerDefinition.assetStatus = 'quality_hold';
                layerDefinition.extractEngine = segmented.extractEngine || 'fastsam';
                layerDefinition.quality = segmented.quality || {
                    status: 'low_quality',
                    runtimeAction: 'hold',
                    reason: 'quality_gate_hold'
                };
                updateLayerExtractionMetadata(item, {
                    id: layerDefinition.id,
                    name: layerDefinition.name
                }, {
                    extractEngine: layerDefinition.extractEngine,
                    quality: layerDefinition.quality,
                    bbox: segmented.bbox || bbox
                });
                state.workbenchItems.set(itemId, item);
                await persistLayerStateToRuntime(itemId, item);
                addMessage({ sender: 'bot', type: 'text', content: `⚠️ 图层 **${layerName}** 的 FastSAM 结果质量不足，已标记为需要高精度模型处理。`});
                if(document.getElementById('workbenchLayerListModal') && document.getElementById('workbenchLayerListModal').style.display !== 'none') {
                    renderLayerList(currentLayers, itemId);
                }
                renderCanvasLayers(itemId);
                return;
            }

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
window.handleQuickFusionSync = handleQuickFusionSync;
window.handleIsolatedAssetEdit = handleIsolatedAssetEdit;
window.performPreciseEdit = performPreciseEdit;
window.extractSingleBoxLayer = extractSingleBoxLayer;
