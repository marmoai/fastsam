import { DESIGN_PATTERNS } from '../runtime/DesignPatternSystem';

// Add the original state and other imports
import { state } from '../core/state.js';
import { dbHelper } from '../core/session.js';
import { addMessage } from './chat-panel.js';
import { editOrQueryImageWithGemini, extractTextFromImage } from '../ai-services/skills-engine.js';
import { dataURLToFile, isRemovalRequest, isMaterialRequest, getProxiedUrl } from '../core/utils.js';
import { addImageToWorkbench } from './workbench-core.js';
import { analyzeImageLayers } from '../ai-services/skills-engine.js';
import { renderCanvasLayers } from './workbench/layers.js';
import { renderSceneToCanvas } from './workbench/renderer.js';
import { openMaskEditor } from '../graphics/mask-drawer.js';
import { performDeepFusion } from './fusion.js';

import { extractLayerAsset, editLayerAsset, undoLayerVersion, resetLayerVersion, exportCurrentSceneImage, cleanBackground, cleanMultipleBackgrounds } from './workbench/layer-assets.js';
import { PRESETS } from '../config/presets.js';
import { globalMatteTaskSystem } from '../graphics/matte-task-system.js';
import { buildSemanticLayerViews, applySemanticLayerViewsToItem, getCleanupLayerForEditableLayer, updateLayerExtractionMetadata } from '../services/semantic-layer-views.js';
import { segmentSingleLayer } from '../services/segmentation-service.js';
import { buildExtractedTextState } from './text-style-utils.js';
import { filterTextLinesToBbox, getCachedTextExtraction, normalizeOcrTextLines } from '../services/text-extraction-cache.js';
import { prepareTextContainerCandidates, restoreTextContainerShapes } from './text-container-restore.js';

const DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST = true;

function getDataUrlImageDimensions(dataUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve({
            width: img.naturalWidth || img.width,
            height: img.naturalHeight || img.height
        });
        img.onerror = () => reject(new Error('无法读取透明图层尺寸'));
        img.src = dataUrl;
    });
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

function cloneSerializable(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function isTextRuntimeLayer(layer, fallbackName = '') {
    if (!layer) return false;
    const name = String(layer.name || fallbackName || '').toLowerCase();
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

function isFlatDesignRuntimeLayer(layer, fallbackName = '') {
    if (!layer) return false;
    const semanticType = String(layer.semanticType || '').toLowerCase();
    const renderMode = String(layer.renderMode || '').toLowerCase();
    const designRole = String(layer.designRole || '').toLowerCase();
    const name = String(layer.name || fallbackName || '').toLowerCase();
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

async function persistWorkbenchItemSemanticState(itemId, item) {
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

    if (state.currentSessionId) {
        const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
        if (currentSession) {
            await dbHelper.saveSession(currentSession);
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
import { StrategyDispatcher } from '../ai-services/strategy-dispatcher.js';

const { workbenchItems } = state;

// 获取DOM元素 (懒加载或在函数内获取)
const getEl = (id) => document.getElementById(id);



/**
 * 优化后的通用确认弹窗函数
 */
export function showCustomConfirm(message) {
    return new Promise((resolve) => {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background: rgba(0,0,0,0.5); z-index: 10005; display: flex; 
            align-items: center; justify-content: center;
        `;
        
        modal.innerHTML = `
            <div style="background: white; padding: 20px; border-radius: 12px; max-width: 320px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); animation: fadeInScale 0.2s ease-out;">
                <p style="margin: 0; color: #333; font-size: 15px; line-height: 1.5;">${message}</p>
                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button id="customConfirmOk" style="flex: 1; padding: 10px; background: #2A5C82; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; outline: none;">确定</button>
                    <button id="customConfirmCancel" style="flex: 1; padding: 10px; background: #f0f0f0; color: #666; border: none; border-radius: 8px; cursor: pointer; font-weight: 500;">取消</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);

        const okBtn = document.getElementById('customConfirmOk');
        const cancelBtn = document.getElementById('customConfirmCancel');

        okBtn.focus();

        const handleKeyboard = (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                okBtn.click();
            } else if (e.key === 'Escape') {
                cancelBtn.click();
            }
        };

        document.addEventListener('keydown', handleKeyboard, true);

        const close = (result) => {
            document.removeEventListener('keydown', handleKeyboard, true);
            if (document.body.contains(modal)) {
                document.body.removeChild(modal);
            }
            resolve(result);
        };

        okBtn.onclick = () => close(true);
        cancelBtn.onclick = () => close(false);
    });
}

/**
 * 局部图层编辑 Prompt 弹窗
 */
export function showLayerEditPrompt(layerName) {
    return new Promise((resolve) => {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background: rgba(0,0,0,0.5); z-index: 10006; display: flex; 
            align-items: center; justify-content: center;
        `;
        
        modal.innerHTML = `
            <div style="background: white; padding: 24px; border-radius: 16px; width: 400px; box-shadow: 0 20px 40px rgba(0,0,0,0.3); animation: fadeInScale 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);">
                <h3 style="margin: 0 0 12px 0; color: #1f2937; font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                    <i class="fas fa-magic" style="color: #6366f1;"></i> 局部智能重绘: ${layerName}
                </h3>
                <p style="margin: 0 0 16px 0; color: #6b7280; font-size: 14px; line-height: 1.5;">
                    请输入您想对该图层进行的修改指令（例如：“换成红色丝绒材质”、“改成现代简约风格”）。
                </p>
                <textarea id="layerEditInput" placeholder="输入修改指令..." style="width: 100%; height: 100px; padding: 12px; border: 1.5px solid #e5e7eb; border-radius: 10px; font-size: 14px; resize: none; outline: none; transition: border-color 0.2s; margin-bottom: 20px; box-sizing: border-box;"></textarea>
                <div style="display: flex; gap: 12px;">
                    <button id="layerEditCancel" style="flex: 1; padding: 12px; background: #f3f4f6; color: #4b5563; border: none; border-radius: 10px; cursor: pointer; font-weight: 600; transition: background 0.2s;">取消</button>
                    <button id="layerEditConfirm" style="flex: 1.5; padding: 12px; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: 600; transition: transform 0.1s, box-shadow 0.2s; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);">开始重绘</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        const input = document.getElementById('layerEditInput');
        const confirmBtn = document.getElementById('layerEditConfirm');
        const cancelBtn = document.getElementById('layerEditCancel');
        
        input.focus();
        
        input.onfocus = () => input.style.borderColor = '#6366f1';
        input.onblur = () => input.style.borderColor = '#e5e7eb';
        
        const close = (result) => {
            if (document.body.contains(modal)) {
                document.body.removeChild(modal);
            }
            resolve(result);
        };
        
        confirmBtn.onclick = () => {
            const val = input.value.trim();
            if (val) close(val);
        };
        
        cancelBtn.onclick = () => close(null);
        
        modal.onclick = (e) => {
            if (e.target === modal) close(null);
        };
        
        input.onkeydown = (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                confirmBtn.click();
            } else if (e.key === 'Escape') {
                cancelBtn.click();
            }
        };
    });
}

// --- Magic Wand Modal ---
export function openMagicWandModal(file) {
    const magicWandModal = getEl('magicWandModal');
    if (magicWandModal) {
        magicWandModal.style.display = 'block';
        magicWandModal.dataset.targetFileId = file.name + file.lastModified;
    }
}

export function closeMagicWandModal() {
    const magicWandModal = getEl('magicWandModal');
    if (magicWandModal) {
        magicWandModal.style.display = 'none';
        delete magicWandModal.dataset.targetFileId;
    }
}

export function handleMagicWandAction(action, { mainImageFile, lastGeneratedImageForEditing, userInput, updateSendBtnState }) {
    const effectiveBaseImage = mainImageFile || lastGeneratedImageForEditing;
    switch (action) {
        case 'style-transfer': userInput.value = "请将这张图转换成赛博朋克风格"; break;
        case 'inpaint': 
            userInput.value = "请在蒙版区域绘制...";
            if (effectiveBaseImage) openMaskEditor(effectiveBaseImage);
            break;
        case 'outpaint': userInput.value = "请将这张图的边界向外扩展，保持原有风格"; break;
        case 'query': userInput.value = "请详细描述这张图片的内容"; break;
    }
    userInput.focus();
    closeMagicWandModal();
    if (typeof updateSendBtnState === 'function') updateSendBtnState();
}

// --- Camera Angle Modal ---
let currentCameraAngleItem = null;

export function openCameraAngleModal(item) {
    currentCameraAngleItem = item;
    const cameraPreviewImg = getEl('cameraPreviewImg');
    const cameraRotate = getEl('cameraRotate');
    const cameraVertical = getEl('cameraVertical');
    const cameraZoom = getEl('cameraZoom');
    const cameraAngleModal = getEl('cameraAngleModal');

    if (cameraPreviewImg) cameraPreviewImg.src = item.dataUrl || URL.createObjectURL(item.file);
    
    if (cameraRotate) cameraRotate.value = 0;
    if (cameraVertical) cameraVertical.value = 0;
    if (cameraZoom) cameraZoom.value = 0;
    
    updateCameraAnglePreview();
    
    if (cameraAngleModal) cameraAngleModal.style.display = 'flex';
}

export function closeCameraAngleModal() {
    const cameraAngleModal = getEl('cameraAngleModal');
    if (cameraAngleModal) cameraAngleModal.style.display = 'none';
    currentCameraAngleItem = null;
}

export function updateCameraAnglePreview() {
    const cameraRotate = getEl('cameraRotate');
    const cameraVertical = getEl('cameraVertical');
    const cameraZoom = getEl('cameraZoom');
    const cameraRotateVal = getEl('cameraRotateVal');
    const cameraVerticalVal = getEl('cameraVerticalVal');
    const cameraZoomVal = getEl('cameraZoomVal');

    if (!cameraRotate || !cameraVertical || !cameraZoom) return;

    const rot = parseInt(cameraRotate.value);
    const vert = parseInt(cameraVertical.value);
    const zoom = parseInt(cameraZoom.value);
    
    if (cameraRotateVal) cameraRotateVal.textContent = `${rot}°`;
    if (cameraVerticalVal) cameraVerticalVal.textContent = `${vert}°`;
    if (cameraZoomVal) cameraZoomVal.textContent = `${zoom}%`;

    const orbitCameraWrapper = document.getElementById('orbitCameraWrapper');
    const orbitCamera = document.getElementById('orbitCamera');
    const orbitCameraLine = document.getElementById('orbitCameraLine');
    
    const distance = 80 - zoom;
    
    if (orbitCameraWrapper && orbitCamera && orbitCameraLine) {
        orbitCameraWrapper.style.transform = `rotateY(${rot}deg) rotateX(${vert}deg)`;
        orbitCamera.style.transform = `translateZ(${distance}px) rotateY(180deg)`;
        orbitCameraLine.style.height = `${distance}px`;
    }
}

export function initCameraAngleModal(handleSend) {
    const cameraRotate = getEl('cameraRotate');
    const cameraVertical = getEl('cameraVertical');
    const cameraZoom = getEl('cameraZoom');
    const closeCameraAngleBtn = getEl('closeCameraAngleBtn');
    const resetCameraBtn = getEl('resetCameraBtn');
    const applyCameraAngleBtn = getEl('applyCameraAngleBtn');
    const userInput = getEl('userInput');

    if (cameraRotate) cameraRotate.addEventListener('input', updateCameraAnglePreview);
    if (cameraVertical) cameraVertical.addEventListener('input', updateCameraAnglePreview);
    if (cameraZoom) cameraZoom.addEventListener('input', updateCameraAnglePreview);

    if (closeCameraAngleBtn) closeCameraAngleBtn.addEventListener('click', closeCameraAngleModal);
    
    if (resetCameraBtn) resetCameraBtn.addEventListener('click', () => {
        if (cameraRotate) cameraRotate.value = 0;
        if (cameraVertical) cameraVertical.value = 0;
        if (cameraZoom) cameraZoom.value = 0;
        updateCameraAnglePreview();
    });

    if (applyCameraAngleBtn) applyCameraAngleBtn.addEventListener('click', async () => {
        if (!currentCameraAngleItem) return;
        
        const rot = parseInt(cameraRotate.value);
        const vert = parseInt(cameraVertical.value);
        const zoom = parseInt(cameraZoom.value);
        
        if (rot === 0 && vert === 0 && zoom === 0) {
            closeCameraAngleModal();
            return;
        }

        // Construct the prompt based on sliders
        let anglePrompt = "【强烈指令：改变相机透视角度】请以全新的相机视角重新绘制这张图片。";
        let terms = [];
        
        if (Math.abs(rot) >= 60) {
            terms.push(rot > 0 ? "完全的右侧面视角 (Right profile view)" : "完全的左侧面视角 (Left profile view)");
        } else if (Math.abs(rot) >= 20) {
            terms.push(rot > 0 ? "右侧 3/4 视角 (Three-quarter view from right)" : "左侧 3/4 视角 (Three-quarter view from left)");
        } else if (rot !== 0) {
            terms.push(`向${rot > 0 ? '右' : '左'}微转`);
        }

        if (vert >= 50) {
            terms.push("正上方俯视/上帝视角 (Top-down drone view)");
        } else if (vert >= 20) {
            terms.push("高角度俯拍 (High angle shot)");
        } else if (vert <= -50) {
            terms.push("极低角度仰拍 (Extreme low angle / Worm's-eye view)");
        } else if (vert <= -20) {
            terms.push("低角度仰拍 (Low angle shot)");
        }

        if (zoom >= 30) {
            terms.push("极度特写/微距镜头 (Extreme close-up shot)");
        } else if (zoom >= 10) {
            terms.push("拉近特写 (Close-up shot)");
        } else if (zoom <= -30) {
            terms.push("极远大远景 (Extreme wide establishing shot)");
        } else if (zoom <= -10) {
            terms.push("拉远广角 (Wide angle shot)");
        }

        if (terms.length > 0) {
            anglePrompt += "新的相机镜头要求：" + terms.join("，") + "。";
        }
        
        anglePrompt += "必须严格执行新的视角！在改变透视和构图的同时，尽可能保留原主体的人物/物体特征和环境氛围，但绝对不要直接复制原图的角度。";

        // Store the file before closing the modal (which clears currentCameraAngleItem)
        const targetFile = currentCameraAngleItem.file;
        
        closeCameraAngleModal();
        
        // Use the existing image editing flow
        state.mainImageFile = targetFile;
        if (userInput) userInput.value = anglePrompt;
        
        // Trigger the send logic
        if (typeof handleSend === 'function') handleSend();
    });
}

// --- Layer Manager Modal ---
export function getLayerState(itemId, layerIndex) {
    const item = workbenchItems.get(itemId);
    if (!item) return { selected: false, visible: true, locked: false };
    
    if (!item.layerStates) item.layerStates = new Map();
    if (!item.layerStates.has(layerIndex)) {
        item.layerStates.set(layerIndex, { selected: false });
    }
    const localState = item.layerStates.get(layerIndex);

    if (item.scene && item.scene.layers && item.scene.layers[layerIndex]) {
        const layer = item.scene.layers[layerIndex];
        const isSemanticGroup = layer.runtimeType === 'semantic_group' || layer.compositeRole === 'composite_group';
        // Sync visible/locked from scene layer to localState for easy reference
        localState.visible = layer.visible;
        localState.locked = isSemanticGroup ? true : layer.locked;
        if (isSemanticGroup) localState.selected = false;
    } else {
        if (localState.visible === undefined) localState.visible = true;
        if (localState.locked === undefined) localState.locked = false;
    }

    return localState;
}

export function updateLayerState(itemId, layerIndex, updates) {
    const item = workbenchItems.get(itemId);
    if (!item) return;
    
    const localState = getLayerState(itemId, layerIndex);
    if (updates.selected !== undefined) localState.selected = updates.selected;
    
    if (item.scene && item.scene.layers && item.scene.layers[layerIndex]) {
        const layer = item.scene.layers[layerIndex];
        const isSemanticGroup = layer.runtimeType === 'semantic_group' || layer.compositeRole === 'composite_group';
        if (updates.visible !== undefined) layer.visible = updates.visible;
        if (updates.locked !== undefined && !isSemanticGroup) layer.locked = updates.locked;
        if (isSemanticGroup) localState.selected = false;
    } else {
        if (updates.visible !== undefined) localState.visible = updates.visible;
        if (updates.locked !== undefined) localState.locked = updates.locked;
    }
    
    // Update canvas selection directly in DOM to avoid breaking dblclick events
    if (updates.selected !== undefined && item.el) {
        const layerEls = item.el.querySelectorAll('.canvas-layer');
        if (layerEls && layerEls[layerIndex]) {
            if (updates.selected) {
                layerEls[layerIndex].classList.add('selected');
                layerEls[layerIndex].style.boxShadow = '0 0 0 2px #4f46e5 inset, 0 0 15px rgba(79, 70, 229, 0.6)';
                layerEls[layerIndex].style.backgroundColor = 'rgba(79, 70, 229, 0.1)';
            } else {
                layerEls[layerIndex].classList.remove('selected');
                layerEls[layerIndex].style.boxShadow = 'none';
                layerEls[layerIndex].style.backgroundColor = 'transparent';
            }
        }
    }
    
    // Only re-render canvas if visibility or lock state changed.
    if (updates.visible !== undefined || updates.locked !== undefined) {
        renderCanvasLayers(itemId);
    }
}

export function ensureSchemeSwitcher(itemId) {
    const layerManagerModal = getEl('layerManagerPanel');
    if (!layerManagerModal) return;
    
    let switcher = layerManagerModal.querySelector('.scheme-switcher');
    if (!switcher) {
        switcher = document.createElement('div');
        switcher.className = 'scheme-switcher';
        
        const header = layerManagerModal.querySelector('.layer-panel-header');
        if (header) {
            header.insertAdjacentElement('afterend', switcher);
        } else {
            layerManagerModal.insertBefore(switcher, layerManagerModal.firstChild);
        }
    }
    
    const item = workbenchItems.get(itemId);
    if (!item || !item.schemes) return;

    switcher.innerHTML = '';
    
    item.schemes.forEach(scheme => {
        const btn = document.createElement('button');
        btn.innerText = scheme.name;
        const isActive = scheme.id === item.activeSchemeId;
        btn.className = `scheme-btn ${isActive ? 'active' : ''}`;
        btn.onclick = () => {
            switchScheme(itemId, scheme.id);
        };
        switcher.appendChild(btn);
    });

    const addBtn = document.createElement('button');
    addBtn.innerHTML = '<i class="fas fa-plus"></i>';
    addBtn.className = 'scheme-add-btn';
    addBtn.title = "新建方案";
    addBtn.onclick = () => {
        createNewScheme(itemId);
    };
    switcher.appendChild(addBtn);

    const exportBtn = document.createElement('button');
    exportBtn.innerHTML = '<i class="fas fa-download"></i> 导出方案';
    exportBtn.className = 'scheme-export-btn';
    exportBtn.onclick = async () => {
        const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
        if (layers && layers.some(l => l.assetStatus === 'editing' || l.assetStatus === 'processing')) {
            alert("有图层正在处理中，请稍后再导出。");
            return;
        }
        const dataUrl = await exportCurrentSceneImage(itemId);
        if (dataUrl) {
            const a = document.createElement('a');
            a.href = dataUrl;
            a.download = `scheme_${item.activeSchemeId}.jpg`;
            a.click();
        }
    };
    switcher.appendChild(exportBtn);
}

export function switchScheme(itemId, schemeId) {
    const item = workbenchItems.get(itemId);
    if (!item || !item.schemes) return;
    
    // Save current layer versions to current scheme
    const currentScheme = item.schemes.find(s => s.id === item.activeSchemeId);
    if (currentScheme) {
        const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
        if (layers) {
            layers.forEach((layer, idx) => {
                currentScheme.layerVersions[layer.id] = layer.activeVersionId;
            });
        }
    }

    // Switch to new scheme
    item.activeSchemeId = schemeId;
    const newScheme = item.schemes.find(s => s.id === schemeId);
    if (newScheme) {
        const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
        if (layers) {
            layers.forEach((layer, idx) => {
                if (newScheme.layerVersions[layer.id]) {
                    layer.activeVersionId = newScheme.layerVersions[layer.id];
                    // Update layer assets based on activeVersionId
                    const version = layer.versions?.find(v => v.id === layer.activeVersionId);
                    if (version) {
                        layer.cutoutUrl = version.cutoutUrl;
                        layer.maskUrl = version.maskUrl;
                        layer.previewUrl = version.previewUrl;
                    }
                }
            });
        }
    }

    workbenchItems.set(itemId, item);
    ensureSchemeSwitcher(itemId);
    const layersToRender = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    renderLayerList(layersToRender, itemId);
    renderCanvasLayers(itemId);
}

export function createNewScheme(itemId) {
    const item = workbenchItems.get(itemId);
    if (!item || !item.schemes) return;
    
    const newId = `scheme-${item.schemes.length + 1}`;
    const newName = `方案 ${item.schemes.length + 1}`;
    
    // Copy current layer versions
    const layerVersions = {};
    const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
    if (layers) {
        layers.forEach((layer) => {
            layerVersions[layer.id] = layer.activeVersionId;
        });
    }

    item.schemes.push({ id: newId, name: newName, layerVersions });
    workbenchItems.set(itemId, item);
    switchScheme(itemId, newId);
}
export function ensureLayerPanelFooter(itemId) {
    const layerManagerModal = getEl('layerManagerPanel');
    if (!layerManagerModal) return;
    let footer = layerManagerModal.querySelector('.layer-panel-footer');
    if (!footer) {
        footer = document.createElement('div');
        footer.className = 'layer-panel-footer';
        layerManagerModal.appendChild(footer);
    }
    
    footer.innerHTML = `
        <div class="layer-footer-row">
            <input type="text" id="fusionEditInput" class="fusion-edit-input" placeholder="对选中层进行融合修改...">
            <button id="fusionEditBtn" class="fusion-edit-btn">
                <i class="fas fa-wand-magic-sparkles"></i> 融合修改
            </button>
        </div>
        <div class="layer-footer-row">
             <button id="explodeBtn" class="explode-btn" title="批量物理拆解选定的非锁定图层">
                <i class="fas fa-layer-group"></i> 拆解
            </button>
             <button id="recomposeBtn" class="recompose-btn" title="回贴验证">
                <i class="fas fa-object-group"></i> 回贴
            </button>
        </div>
    `;

    const fusionBtn = footer.querySelector('#fusionEditBtn');
    const explodeBtn = footer.querySelector('#explodeBtn');
    const recomposeBtn = footer.querySelector('#recomposeBtn');
    
    if (fusionBtn) fusionBtn.onclick = () => handleFusionEdit(itemId);
    if (explodeBtn) explodeBtn.onclick = () => window.triggerLayerExplosion(itemId);
    if (recomposeBtn) recomposeBtn.onclick = () => handleRecomposeVerification(itemId);
}

export function updateFusionUI(itemId) {
    const item = workbenchItems.get(itemId);
    if (!item || !item.layers) return;
    
    const selectedCount = item.layers.filter((_, idx) => getLayerState(itemId, idx).selected).length;
    const fusionBtn = document.getElementById('fusionEditBtn');
    const explodeBtn = document.getElementById('explodeBtn');
    
    if (fusionBtn) {
        fusionBtn.disabled = selectedCount === 0;
        fusionBtn.style.opacity = selectedCount === 0 ? '0.5' : '1';
        fusionBtn.style.cursor = selectedCount === 0 ? 'not-allowed' : 'pointer';
    }
    if (explodeBtn) {
        explodeBtn.style.display = 'flex';
    }
}

export async function showLayerManagerModal(itemId, triggerAnalysis = false) {
    const item = workbenchItems.get(itemId);
    if (!item) return;
    const layerManagerModal = getEl('layerManagerPanel');
    const layerList = getEl('layerList');
    if (!layerManagerModal) return;

    layerManagerModal.style.display = 'flex';
    
    // 是否需要进行全图分析？
    // 只有在明确触发分析，且还未进行过全图分析的情况下，才执行分析。
    const hasAutoDetectedLayers = (item.scene && item.scene.layers ? item.scene.layers : (item.layers || [])).some(l => l.layerType || (l.id && !l.id.startsWith('box-layer-')));
    const needsAnalysis = triggerAnalysis && (!item.hasFullSemanticAnalysis && !hasAutoDetectedLayers);

    if (!needsAnalysis) {
        // 如果不需要进行新的全图分析，就直接展示当前已有的图层（框选的或者之前分析过了的）
        if (item.scene && item.scene.layers && item.scene.layers.length > 0) {
             ensureSchemeSwitcher(itemId);
             ensureLayerPanelFooter(itemId);
             renderLayerList(item.scene.layers, itemId);
             renderCanvasLayers(itemId);
             return;
        } else if (item.layers && Array.isArray(item.layers) && item.layers.length > 0) {
             ensureSchemeSwitcher(itemId);
             ensureLayerPanelFooter(itemId);
             renderLayerList(item.layers, itemId);
             renderCanvasLayers(itemId);
             return;
        }
        
        // 如果目前完全没有图层，显示一个空状态
        if (layerList) {
             layerList.innerHTML = '<div class="empty-state" style="padding: 20px; text-align: center;"><p style="color:#999;font-size:14px;">暂无图层数据。请点击右侧"语义图层"按钮进行智能提取，或进行框选提取。</p></div>';
        }
        return;
    }

    // 获取可能已经手动框选的图层
    const existingManualLayers = (item.scene && item.scene.layers ? item.scene.layers : (item.layers || [])).filter(l => l.id && l.id.startsWith('box-layer-'));

    if (layerList) layerList.innerHTML = '<div class="layer-loading"><div class="spinner"></div><p>正在扫描视觉元素...</p></div>';

    try {
        const analysisResult = await analyzeImageLayers(item.file || item.dataUrl);
        let newLayers = analysisResult.rawLayers;
        let scene = analysisResult.scene;
        
        // 合并逻辑：保留手动框选图层，跳过与手动图层重叠过大的自动识别图层
        if (existingManualLayers.length > 0) {
            const calculateOverlapPercent = (box1, box2) => {
                // box: [ymin, xmin, ymax, xmax] 0-1000
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
                if (newLayer.layerType === 'background_plate') return true; // 通常不跳过背景层
                const newBox = newLayer.bbox;
                for (let manualLayer of existingManualLayers) {
                    const overlap = calculateOverlapPercent(newBox, manualLayer.bbox);
                    if (overlap > 0.6) { // 重叠率超过 60% 视为重复
                        console.log(`Skipping auto-detected layer "${newLayer.name}" due to ${Math.round(overlap*100)}% overlap with manual layer "${manualLayer.name}"`);
                        return false; 
                    }
                }
                return true;
            });
            
            // 手动图层置顶
            newLayers = [...existingManualLayers, ...newLayers];
            
            // 修复 zIndex 保证合并后的排序逻辑
            newLayers.forEach((l, idx) => {
                if(l.id && l.id.startsWith('box-layer-')) {
                    l.zIndex = 100 - idx;
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
        item.hasFullSemanticAnalysis = true; // 记录已完成全图分析
        workbenchItems.set(itemId, item);
        await persistWorkbenchItemSemanticState(itemId, item);
        
        ensureLayerPanelFooter(itemId);
        renderLayerList(item.scene.layers, itemId);
        renderCanvasLayers(itemId);
        
        if (window.addWorkbenchActionToChat) {
            const expandedTextCount = semanticViews?.stats?.expandedTextCount || 0;
            const message = expandedTextCount > 0
                ? `分析出 ${newLayers.length} 个语义元素，并展开 ${expandedTextCount} 个可编辑文本块`
                : `分析出 ${newLayers.length} 个视觉元素`;
            await window.addWorkbenchActionToChat('语义图层分析', message, null, null);
        }
    } catch (e) {
        console.error("Layer analysis failed:", e);
        if (layerList) layerList.innerHTML = `<div style="padding:20px;color:red;text-align:center;font-size:12px;">分析失败: ${e.message}</div>`;
    }
}

export function closeLayerManagerModal() {
    const layerManagerModal = getEl('layerManagerPanel');
    if (layerManagerModal) {
        layerManagerModal.style.display = 'none';
    }
}

function isEditableTextLayer(layer) {
    return !!layer && (
        layer.runtimeType === 'text_node' ||
        layer.semanticType === 'element_text' ||
        layer.sourceTextLayerId ||
        layer.cleanPlateLayerId && String(layer.name || '').startsWith('文字:')
    );
}

function getTextGroupExpanded(itemId) {
    window.__textLayerGroupExpanded = window.__textLayerGroupExpanded || new Map();
    return window.__textLayerGroupExpanded.get(itemId) === true;
}

function setTextGroupExpanded(itemId, expanded) {
    window.__textLayerGroupExpanded = window.__textLayerGroupExpanded || new Map();
    window.__textLayerGroupExpanded.set(itemId, expanded);
}

function buildLayerDisplayEntries(layers, itemId) {
    const textEntries = [];
    layers.forEach((layer, index) => {
        if (isEditableTextLayer(layer)) {
            textEntries.push({ layer, index });
        }
    });

    if (textEntries.length <= 1) {
        return layers.map((layer, index) => ({ type: 'layer', layer, index }));
    }

    const expanded = getTextGroupExpanded(itemId);
    const firstTextIndex = textEntries[0].index;
    const textIndexSet = new Set(textEntries.map(entry => entry.index));
    const entries = [];
    let textGroupInserted = false;

    layers.forEach((layer, index) => {
        if (!textIndexSet.has(index)) {
            entries.push({ type: 'layer', layer, index });
            return;
        }

        if (!textGroupInserted && index === firstTextIndex) {
            entries.push({ type: 'text-group', entries: textEntries, expanded });
            if (expanded) {
                textEntries.forEach(entry => {
                    entries.push({ type: 'layer', layer: entry.layer, index: entry.index, grouped: true });
                });
            }
            textGroupInserted = true;
        }
    });

    return entries;
}

export function renderLayerList(layers, itemId) {
    const layerList = getEl('layerList');
    if (!layerList) return;
    layerList.innerHTML = '';
    
    const displayEntries = buildLayerDisplayEntries(layers, itemId);
    displayEntries.forEach((entry) => {
        if (entry.type === 'text-group') {
            const textStates = entry.entries.map(({ index }) => getLayerState(itemId, index));
            const selectedCount = textStates.filter(st => st.selected).length;
            const visibleCount = textStates.filter(st => st.visible).length;
            const lockedCount = textStates.filter(st => st.locked).length;
            const allSelected = selectedCount === entry.entries.length;
            const someSelected = selectedCount > 0 && !allSelected;
            const expanded = entry.expanded;

            const wrapper = document.createElement('div');
            wrapper.className = 'layer-list-wrapper text-layer-group-wrapper';
            wrapper.style.display = 'flex';
            wrapper.style.flexDirection = 'column';

            const el = document.createElement('div');
            el.className = `layer-list-item text-layer-group ${allSelected || someSelected ? 'selected' : ''}`;
            el.style.cssText = `
                display: flex; align-items: center; padding: 6px 8px;
                transition: all 0.2s; cursor: pointer;
                opacity: ${visibleCount > 0 ? '1' : '0.5'};
                background: ${allSelected || someSelected ? '#e3f2fd' : 'rgba(15, 23, 42, 0.03)'};
            `;

            const expandBtn = document.createElement('button');
            expandBtn.className = 'layer-btn';
            expandBtn.innerHTML = `<i class="fas fa-chevron-${expanded ? 'down' : 'right'}"></i>`;
            expandBtn.title = expanded ? '收起文字图层' : '展开文字图层';
            expandBtn.onclick = (e) => {
                e.stopPropagation();
                setTextGroupExpanded(itemId, !expanded);
                renderLayerList(layers, itemId);
            };

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = allSelected;
            checkbox.indeterminate = someSelected;
            checkbox.disabled = lockedCount === entry.entries.length;
            checkbox.style.marginRight = '8px';
            checkbox.onchange = (e) => {
                const isChecked = e.target.checked;
                entry.entries.forEach(({ index }) => {
                    const stateForLayer = getLayerState(itemId, index);
                    if (!stateForLayer.locked) {
                        updateLayerState(itemId, index, { selected: isChecked });
                    }
                });
                renderLayerList(layers, itemId);
                updateFusionUI(itemId);
            };

            const nameDiv = document.createElement('div');
            nameDiv.style.flex = '1';
            nameDiv.style.fontSize = '13px';
            nameDiv.style.cursor = 'pointer';
            nameDiv.style.display = 'flex';
            nameDiv.style.alignItems = 'center';
            nameDiv.style.minWidth = '0';
            nameDiv.innerHTML = `
                <div style="display:flex;align-items:center;gap:6px;min-width:0;white-space:nowrap;overflow:hidden;">
                    <i class="fas fa-font" style="font-size:11px;color:#475569;"></i>
                    <span style="font-weight:600;overflow:hidden;text-overflow:ellipsis;">文字</span>
                    <span style="font-size:11px;color:#94a3b8;">${entry.entries.length}</span>
                </div>
            `;
            nameDiv.onclick = (e) => {
                e.stopPropagation();
                setTextGroupExpanded(itemId, !expanded);
                renderLayerList(layers, itemId);
            };

            const actionsDiv = document.createElement('div');
            actionsDiv.style.display = 'flex';
            actionsDiv.style.gap = '8px';

            const eyeBtn = document.createElement('button');
            eyeBtn.className = 'layer-btn';
            eyeBtn.innerHTML = `<i class="${visibleCount > 0 ? 'fas fa-eye' : 'fas fa-eye-slash'}"></i>`;
            eyeBtn.title = visibleCount > 0 ? '隐藏全部文字层' : '显示全部文字层';
            eyeBtn.onclick = (e) => {
                e.stopPropagation();
                const nextVisible = visibleCount === 0;
                entry.entries.forEach(({ index }) => updateLayerState(itemId, index, { visible: nextVisible }));
                renderLayerList(layers, itemId);
                renderCanvasLayers(itemId);
            };

            const lockBtn = document.createElement('button');
            lockBtn.className = 'layer-btn';
            lockBtn.innerHTML = `<i class="${lockedCount === entry.entries.length ? 'fas fa-lock' : 'fas fa-lock-open'}"></i>`;
            lockBtn.title = lockedCount === entry.entries.length ? '解锁全部文字层' : '锁定全部文字层';
            lockBtn.onclick = (e) => {
                e.stopPropagation();
                const nextLocked = lockedCount !== entry.entries.length;
                entry.entries.forEach(({ index }) => updateLayerState(itemId, index, { locked: nextLocked, selected: nextLocked ? false : getLayerState(itemId, index).selected }));
                renderLayerList(layers, itemId);
                updateFusionUI(itemId);
            };

            actionsDiv.appendChild(eyeBtn);
            actionsDiv.appendChild(lockBtn);
            el.appendChild(expandBtn);
            el.appendChild(checkbox);
            el.appendChild(nameDiv);
            el.appendChild(actionsDiv);
            wrapper.appendChild(el);
            layerList.appendChild(wrapper);
            return;
        }

        const { layer, index } = entry;
        const layerState = getLayerState(itemId, index);
        const layerName = layer.name || layer;
        
        const wrapper = document.createElement('div');
        wrapper.className = 'layer-list-wrapper';
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        if (entry.grouped) {
            wrapper.style.marginLeft = '10px';
        }

        const el = document.createElement('div');
        el.className = `layer-list-item ${layerState.selected ? 'selected' : ''}`;
        el.style.cssText = `
            display: flex; align-items: center; padding: 8px 10px; 
            transition: all 0.2s; cursor: pointer;
            opacity: ${layerState.visible ? '1' : '0.5'};
        `;
        if (entry.grouped) {
            el.style.padding = '6px 8px';
        }
        
        // Click to toggle selection (V2.5)
        el.onclick = (e) => {
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'I' || e.target.tagName === 'INPUT') return;
            const newSelected = !layerState.selected;
            updateLayerState(itemId, index, { selected: newSelected });
            renderLayerList(layers, itemId); 
            updateFusionUI(itemId);
            
            state.currentActiveWorkbenchItemId = itemId;
            if (newSelected && typeof window.triggerCapsuleAlert === 'function') {
                window.triggerCapsuleAlert(itemId);
            }
        };
        
        // Double click to edit (V2.3)
        el.addEventListener('dblclick', async (e) => {
            e.stopPropagation();
            if (layer.category === 'background') {
                console.log("Cannot edit background layer");
                return;
            }
            if (layer.assetStatus === 'editing' || layer.assetStatus === 'processing') {
                console.log("Layer is currently processing or editing");
                return; 
            }
            
            console.log("Opening layer edit prompt for:", layerName);
            const prompt = await showLayerEditPrompt(layerName);
            if (prompt) {
                editLayerAsset(itemId, index, prompt);
            }
        });
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = layerState.selected;
        checkbox.disabled = layerState.locked; 
        checkbox.style.marginRight = '10px';
        checkbox.onchange = (e) => {
            const isChecked = e.target.checked;
            updateLayerState(itemId, index, { selected: isChecked });
            renderLayerList(layers, itemId); 
            updateFusionUI(itemId);
            
            state.currentActiveWorkbenchItemId = itemId;
            if (isChecked && typeof window.triggerCapsuleAlert === 'function') {
                window.triggerCapsuleAlert(itemId);
            }
        };

        const nameDiv = document.createElement('div');
        nameDiv.style.flex = '1';
        nameDiv.style.fontSize = '13px';
        nameDiv.style.cursor = 'pointer';
        nameDiv.style.display = 'flex';
        nameDiv.style.flexDirection = 'column';
        nameDiv.style.gap = '2px';
        nameDiv.style.minWidth = '0';
        
        const titleRow = document.createElement('div');
        titleRow.style.display = 'flex';
        titleRow.style.alignItems = 'center';
        titleRow.style.gap = '5px';
        titleRow.style.minWidth = '0';
        
        let statusIcon = '';
        if (layer.category === 'background') {
            const currentItem = state.workbenchItems.get(itemId);
            if (currentItem && currentItem.cleanPlateStatus === 'cleaning') {
                statusIcon = `<i class="fas fa-broom fa-bounce" style="color: #3b82f6; font-size: 10px;" title="正在清理背景残影..."></i>`;
            }
        } else if (layer.interactionLock?.reason === 'rendering' || layer.assetStatus === 'editing') {
            statusIcon = `<i class="fas fa-spinner fa-spin" style="color: #8b5cf6; font-size: 10px;" title="正在重绘图层..."></i>`;
        } else if (layer.assetStatus === 'pending' || layer.assetStatus === 'processing') {
            statusIcon = `<i class="fas fa-spinner fa-spin" style="color: #8b5cf6; font-size: 10px;" title="正在生成图层资产..."></i>`;
        } else if (layer.assetStatus === 'error') {
            statusIcon = '<i class="fas fa-exclamation-circle" style="color: #ef4444; font-size: 10px;" title="生成失败"></i>';
        } else if (layer.assetStatus === 'ready') {
            statusIcon = '<i class="fas fa-check-circle" style="color: #10b981; font-size: 10px;" title="图层已就绪"></i>';
        }

        titleRow.innerHTML = `<span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;">${layerName}</span> ${statusIcon}`;
        nameDiv.appendChild(titleRow);

        nameDiv.onclick = (e) => {
            e.stopPropagation();
            highlightLayerInImage(itemId, layerName, layer.bbox || [0,0,1000,1000], el);
        };

        const actionsDiv = document.createElement('div');
        actionsDiv.style.display = 'flex';
        actionsDiv.style.gap = '8px';

        // Add drag and drop functionality
        el.draggable = true;
        el.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', index);
            e.dataTransfer.effectAllowed = 'move';
            el.style.opacity = '0.5';
        });
        
        el.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            el.style.borderTop = '2px solid #4f46e5';
        });
        
        el.addEventListener('dragleave', (e) => {
            el.style.borderTop = '';
        });
        
        el.addEventListener('drop', (e) => {
            e.preventDefault();
            el.style.borderTop = '';
            el.style.opacity = '1';
            
            const fromIndex = parseInt(e.dataTransfer.getData('text/plain'), 10);
            const toIndex = index;
            
            if (fromIndex !== toIndex && !isNaN(fromIndex)) {
                // Reorder the layers array
                const movedLayer = layers.splice(fromIndex, 1)[0];
                layers.splice(toIndex, 0, movedLayer);
                
                // Reorder layerStates if they exist
                const item = workbenchItems.get(itemId);
                if (item && item.layerStates) {
                    const oldStates = Array.from(item.layerStates.entries())
                        .sort((a, b) => a[0] - b[0])
                        .map(entry => entry[1]);
                        
                    const movedState = oldStates.splice(fromIndex, 1)[0];
                    oldStates.splice(toIndex, 0, movedState);
                    
                    const newStates = new Map();
                    oldStates.forEach((st, i) => newStates.set(i, st));
                    item.layerStates = newStates;
                }
                
                renderLayerList(layers, itemId);
                renderCanvasLayers(itemId);
            }
        });
        
        el.addEventListener('dragend', (e) => {
            el.style.opacity = '1';
            el.style.borderTop = '';
        });

        // 还原按钮 (V2.4 Versioning)
        if (layer.versions && layer.versions.length > 1) {
            const currentIndex = layer.versions.findIndex(v => v.id === layer.activeVersionId);
            
            if (currentIndex > 0) {
                const undoBtn = document.createElement('button');
                undoBtn.className = 'layer-btn';
                undoBtn.innerHTML = '<i class="fas fa-undo" style="font-size: 11px;"></i>';
                undoBtn.title = "撤销上一步";
                undoBtn.style.color = '#6366f1';
                undoBtn.onclick = (e) => {
                    e.stopPropagation();
                    undoLayerVersion(itemId, index);
                    renderCanvasLayers(itemId);
                };
                actionsDiv.appendChild(undoBtn);
            }
            
            if (currentIndex > 0) {
                const resetBtn = document.createElement('button');
                resetBtn.className = 'layer-btn';
                resetBtn.innerHTML = '<i class="fas fa-history" style="font-size: 11px;"></i>';
                resetBtn.title = "恢复原始版本";
                resetBtn.style.color = '#ef4444';
                resetBtn.onclick = (e) => {
                    e.stopPropagation();
                    resetLayerVersion(itemId, index);
                    renderCanvasLayers(itemId);
                };
                actionsDiv.appendChild(resetBtn);
            }
        }

        const eyeBtn = document.createElement('button');
        eyeBtn.className = 'layer-btn';
        eyeBtn.innerHTML = `<i class="${layerState.visible ? 'fas fa-eye' : 'fas fa-eye-slash'}"></i>`;
        eyeBtn.title = layerState.visible ? "点击隐藏 (移除)" : "点击显示";
        eyeBtn.onclick = async (e) => {
            e.stopPropagation();
            if (layerState.visible) {
                updateLayerState(itemId, index, { visible: false });
                renderLayerList(layers, itemId);
            } else {
                updateLayerState(itemId, index, { visible: true });
                renderLayerList(layers, itemId);
            }
        };

        const lockBtn = document.createElement('button');
        lockBtn.className = 'layer-btn';
        lockBtn.innerHTML = `<i class="${layerState.locked ? 'fas fa-lock' : 'fas fa-lock-open'}"></i>`;
        lockBtn.title = layerState.locked ? "已锁定 (保护)" : "点击锁定";
        lockBtn.style.color = layerState.locked ? '#e53e3e' : '';
        lockBtn.onclick = (e) => {
            e.stopPropagation();
            const newLocked = !layerState.locked;
            updateLayerState(itemId, index, { locked: newLocked, selected: newLocked ? false : layerState.selected });
            renderLayerList(layers, itemId);
            updateFusionUI(itemId);
        };

        const extractBtn = document.createElement('button');
        extractBtn.className = 'layer-btn';
        extractBtn.innerHTML = '<i class="fas fa-external-link-alt"></i>';
        extractBtn.title = "提取为完整独立图层";
        extractBtn.onclick = (e) => {
            e.stopPropagation();
            performLayerAction('extract', layerName, itemId, layer);
        };

        const removeBtn = document.createElement('button');
        removeBtn.className = 'layer-btn';
        removeBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
        removeBtn.title = "移除图层并净化底板";
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            performLayerAction('remove', layerName, itemId, layer);
        };

        actionsDiv.appendChild(eyeBtn);
        actionsDiv.appendChild(lockBtn);
        actionsDiv.appendChild(extractBtn);
        actionsDiv.appendChild(removeBtn);

        el.appendChild(checkbox);
        el.appendChild(nameDiv);
        el.appendChild(actionsDiv);
        
        wrapper.appendChild(el);

        // Add Quick Toolbox if selected and has presets
        const semanticType = layer.semanticType || (layer.category === 'background' ? 'surface_wall' : 'other');
        if (layerState.selected && PRESETS[semanticType]) {
            const presetsDiv = document.createElement('div');
            presetsDiv.style.cssText = `
                display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 10px 8px 30px;
                background: ${layerState.selected ? '#e3f2fd' : 'white'};
            `;
            PRESETS[semanticType].forEach(preset => {
                const btn = document.createElement('button');
                btn.innerText = preset;
                btn.style.cssText = `
                    font-size: 10px; padding: 2px 6px; border-radius: 4px;
                    border: 1px solid #cbd5e1; background: white; color: #475569;
                    cursor: pointer; transition: all 0.2s;
                `;
                btn.onmouseover = () => btn.style.borderColor = '#6366f1';
                btn.onmouseout = () => btn.style.borderColor = '#cbd5e1';
                btn.onclick = (e) => {
                    e.stopPropagation();
                    editLayerAsset(itemId, index, `将它替换为${preset}风格`);
                };
                presetsDiv.appendChild(btn);
            });
            wrapper.appendChild(presetsDiv);
        }

        layerList.appendChild(wrapper);
    });
}

export async function handleFusionEdit(itemId) {
    const item = workbenchItems.get(itemId);
    if (!item || !item.layers) return;
    
    const promptInput = document.getElementById('fusionEditInput');
    const promptText = promptInput.value.trim();
    if (!promptText) { alert("请输入编辑指令"); return; }

    const selectedIndices = [];
    item.layers.forEach((_, idx) => {
        if (getLayerState(itemId, idx).selected) selectedIndices.push(idx);
    });

    if (selectedIndices.length === 0) return;

    const maskCanvas = document.createElement('canvas');
    const tempImg = new Image();
    tempImg.crossOrigin = "anonymous";
    const itemSrc = getProxiedUrl(item.dataUrl);
    if (!itemSrc) {
        throw new Error('无效的图片地址，无法生成蒙版');
    }
    tempImg.src = itemSrc;
    await new Promise(r => tempImg.onload = r);
    
    maskCanvas.width = tempImg.naturalWidth;
    maskCanvas.height = tempImg.naturalHeight;
    const ctx = maskCanvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    ctx.fillStyle = 'white';
    const layerNames = [];

    selectedIndices.forEach(idx => {
        const layer = item.layers[idx];
        layerNames.push(layer.name);
        if (layer.bbox) {
            const [ymin, xmin, ymax, xmax] = layer.bbox;
            const bx = (xmin / 1000) * maskCanvas.width;
            const by = (ymin / 1000) * maskCanvas.height;
            const bw = ((xmax - xmin) / 1000) * maskCanvas.width;
            const bh = ((ymax - ymin) / 1000) * maskCanvas.height;
            ctx.fillRect(bx, by, bw, bh);
        }
    });
    
    const maskDataUrl = maskCanvas.toDataURL('image/png');
    const layerManagerModal = getEl('layerManagerPanel');
    if (layerManagerModal) layerManagerModal.style.display = 'none';
    
    const executeMultiLayerEdit = async (customPrompt) => {
        const promptToUse = customPrompt || promptText;
        const tempMsg = addMessage({ sender: 'bot', type: 'text', content: `🎨 **正在融合编辑**: [${layerNames.join(', ')}]\n指令: "${promptToUse}"` });

        try {
            const fullPrompt = `${promptToUse}. Change ONLY the visual appearance (color, material, style) of the selected objects. Keep the background and other objects unchanged.`;
            const sceneImage = await exportCurrentSceneImage(itemId);
            const result = await editOrQueryImageWithGemini(fullPrompt, sceneImage, [], maskDataUrl);

            if (result.success && result.imageData) {
                 const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
                 const newFile = await dataURLToFile(imgSrc, `fusion-edit-${Date.now()}.png`);
                 
                 addImageToWorkbench(newFile, `编辑: ${layerNames.join('+')}`, {
                     x: parseFloat(item.el.style.left),
                     y: parseFloat(item.el.style.top),
                     parentId: itemId
                 });
                 
                 if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                 if (window.addWorkbenchActionToChat) {
                     await window.addWorkbenchActionToChat(`融合编辑 [${layerNames.join(', ')}]`, promptToUse, imgSrc, executeMultiLayerEdit);
                 } else {
                     addMessage({ sender: 'bot', type: 'text', content: '✅ 编辑完成！' });
                 }
            }
        } catch (e) {
            console.error(e);
            if (tempMsg && tempMsg.parentNode) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `❌ 编辑失败: ${e.message}` });
        }
    };

    await executeMultiLayerEdit();
}

let currentHighlightOverlay = null;

export function highlightLayerInImage(itemId, layerName, bbox, listItemEl) {
    const item = workbenchItems.get(itemId);
    if (!item || !item.el) return;
    
    const itemEl = item.el;
    const itemWidth = itemEl.offsetWidth;
    const itemHeight = itemEl.offsetHeight;
    
    if (currentHighlightOverlay) currentHighlightOverlay.remove();
    
    const [ymin, xmin, ymax, xmax] = bbox;
    const x = (xmin / 1000) * itemWidth;
    const y = (ymin / 1000) * itemHeight;
    const width = ((xmax - xmin) / 1000) * itemWidth;
    const height = ((ymax - ymin) / 1000) * itemHeight;
    
    const overlay = document.createElement('div');
    overlay.className = 'layer-highlight-overlay';

    const visualBorderWidth = 2;
    const adjustedBorderWidth = visualBorderWidth / state.workbenchZoom;

    overlay.style.position = 'absolute';
    overlay.style.left = `${x}px`;
    overlay.style.top = `${y}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;
    overlay.style.background = 'rgba(33, 150, 243, 0.1)';
    overlay.style.pointerEvents = 'none';
    overlay.style.zIndex = '1001';
    overlay.style.borderRadius = '4px';
    overlay.style.border = `${adjustedBorderWidth}px dashed #2196F3`;

    itemEl.appendChild(overlay);
    currentHighlightOverlay = overlay;
    
    setTimeout(() => { if (overlay) overlay.remove(); }, 3000);
}

export async function performLayerAction(action, layerName, itemId, layerInfo) {
    console.log(`[LayerAction] Triggered: ${action} on ${layerName}`);
    const item = workbenchItems.get(itemId);
    if (!item) return;  
    
    // layerInfo could be a bbox array (legacy/fallback) or an object
    const bbox = Array.isArray(layerInfo) ? layerInfo : (layerInfo.bbox || [250,250,750,750]);
    const cleanupLayerInfo = Array.isArray(layerInfo)
        ? { bbox, name: layerName }
        : getCleanupLayerForEditableLayer(item, layerInfo, { preferEditableTextBbox: true });
    const cleanupBbox = cleanupLayerInfo?.bbox || bbox;
    const isTextLayer = !Array.isArray(layerInfo) && isTextRuntimeLayer(layerInfo, layerName);

    const layerManagerModal = getEl('layerManagerPanel');
    if(layerManagerModal) layerManagerModal.style.display = 'none';

    let defaultPrompt = "";
    if (action === 'remove') {
        defaultPrompt = `Remove the object "${layerName}" from this image. Use context-aware inpainting to fill the gap seamlessly.`;
    } else {
        defaultPrompt = `Act as a precise image extraction tool for the object: "${layerName}". (The optimal background color will be calculated automatically based on the object's color to maximize contrast).`;
    }

    const executeLayerAction = async (customPrompt) => {
        const promptToUse = customPrompt || defaultPrompt;
        const tempMsg = addMessage({ 
            sender: 'bot', 
            type: 'text', 
            content: action === 'remove' 
                ? `👁️ **正在移除**: "${layerName}"... AI 正在计算背景补全。\n指令: "${promptToUse}"` 
                : `✨ **正在重建**: "${layerName}"... AI 正在进行独立资产重建。\n指令: "${promptToUse}"` 
        });
        
        try {
            let finalPrompt = promptToUse;
            const applyCleanPlateForExtraction = async (cleanupPromptHint = promptToUse, cleanupLayerOverride = null) => {
                const baseBg = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
                let cleanedBg = null;
                try {
                    const cleanupLayerPayload = cleanupLayerOverride || cleanupLayerInfo;
                    cleanedBg = await cleanBackground(baseBg, {
                        ...cleanupLayerPayload,
                        name: cleanupLayerPayload?.name || layerName,
                        promptHint: cleanupLayerPayload?.promptHint || cleanupPromptHint
                    });
                } catch (e) {
                    console.warn(`[Modals] Failed to clean background for ${layerName}`, e);
                }

                if (cleanedBg) {
                    item.cleanPlateDataUrl = cleanedBg;
                    item.dataUrl = cleanedBg;
                    item.cleanPlateStatus = 'ready';
                    workbenchItems.set(itemId, item);
                    await persistWorkbenchItemSemanticState(itemId, item);
                    if (window.historyManager) window.historyManager.pushState();

                    if (item.el) {
                        const imgEl = item.el.querySelector('.crop-container > img');
                        if (imgEl) {
                            imgEl.src = cleanedBg;
                        }
                    }
                } else {
                    console.warn(`[Modals] Clean plate failed or skipped, preserving original background context.`);
                }

                return cleanedBg;
            };

            if (isTextLayer && action === 'extract') {
                // TEXT EXTRACTION PIPELINE (LOCAL LAYER)
                tempMsg.update(`🔍 **文字提取**: 正在解析 "${layerName}"...`);
                
                const originalTextSource = item.originalDataUrl || item.file || item.dataUrl;
                const baseBg = originalTextSource;
                
                let textLines = !Array.isArray(layerInfo)
                    ? getCachedTextExtraction(layerInfo, {
                        allowSemanticFallback: false,
                        allowTextContentFallback: false
                    })
                    : [];

                if (textLines.length === 0) {
                    const ocrResult = await extractTextFromImage(baseBg, bbox, { mode: 'strict' });
                    textLines = normalizeOcrTextLines(ocrResult, Array.isArray(layerInfo) ? { bbox } : layerInfo);
                }

                textLines = filterTextLinesToBbox(textLines, bbox, 0.45);

                if (textLines.length === 0) {
                    tempMsg.update(`⚠️ 文本内容提取失败或未在其区域检测到有效文字，但将继续为您净化底板以消除原有痕迹。`);
                } else {
                    tempMsg.update(`🧹 **文字背景净化**: 根据识别到的 ${textLines.length} 处文字位置精确抹除并修补背景...`);
                }
                
                let layersForCleanup = [];
                if (textLines && textLines.length > 0) {
                    layersForCleanup = textLines.map(line => ({
                        bbox: line.bbox || bbox,
                        name: `Text: "${line.textContent || layerName}"`
                    }));
                } else {
                    layersForCleanup = [{ bbox: cleanupBbox, name: `Text Container: "${layerName}"` }];
                }

                const cleanedBg = await cleanMultipleBackgrounds(
                    baseBg, 
                    layersForCleanup,
                    "Remove all text, letters, logomarks, icon elements, typography, and any text-container artifacts from the image. Preserve only the background and inpaint the cleared regions seamlessly.",
                    { preserveBackgroundOnly: true }
                );
                
                if (cleanedBg) {
                    item.cleanPlateDataUrl = cleanedBg;
                    item.dataUrl = cleanedBg;
                    item.cleanPlateStatus = 'ready';
                    workbenchItems.set(itemId, item);
                    await persistWorkbenchItemSemanticState(itemId, item);
                    if (window.historyManager) window.historyManager.pushState();
                    if (item.el) {
                        const imgEl = item.el.querySelector('.crop-container > img');
                        if (imgEl) imgEl.src = cleanedBg;
                    }
                } else {
                    console.warn(`[Modals] Background clean failed for text layer "${layerName}". Continuing to extract text.`);
                }

                const parentWidth = parseFloat(item.el.style.width) || 300;
                const parentHeight = parseFloat(item.el.style.height) || 300;
                const baseX = parseFloat(item.el.style.left) || 0;
                const baseY = parseFloat(item.el.style.top) || 0;

                const { addTextNoteToWorkbench } = await import('./workbench/notes.js');
                const containerCandidates = await prepareTextContainerCandidates(item, textLines, originalTextSource);
                await restoreTextContainerShapes({
                    item,
                    itemId,
                    textLines,
                    baseX,
                    baseY,
                    parentWidth,
                    parentHeight,
                    sourceImage: originalTextSource,
                    zIndexBase: parseInt(item.el.style.zIndex || 0) + 1
                });

                for (const [index, lineObj] of textLines.entries()) {
                    const textState = await buildExtractedTextState({
                        lineObj,
                        layerObj: layerInfo,
                        fallbackBbox: bbox,
                        containerCandidates,
                        allTextLines: textLines,
                        index,
                        baseX,
                        baseY,
                        parentWidth,
                        parentHeight,
                        zIndex: parseInt(item.el.style.zIndex || 0) + 1,
                        parentId: itemId,
                        sourceImage: originalTextSource,
                        minWidth: 20,
                        minHeight: 10
                    });
                    const textStates = Array.isArray(textState) ? textState : [textState];
                    textStates.forEach(state => addTextNoteToWorkbench(0, 0, state));
                }
                    
                    if (textLines.length > 0) {
                        tempMsg.update(`✅ **文本提取完成**: "${layerName}" 已转换为可双击编辑的文本层。`);
                    } else {
                        tempMsg.update(`✅ **处理完成**: "${layerName}" 区域已抹除，但未检测到文本可提取。`);
                    }
                    
                    if (window.historyManager) {
                        window.historyManager.pushState();
                    }
                    
                    if (item.el && cleanedBg) {
                        const imgEl = item.el.querySelector('.crop-container > img');
                        if (imgEl) imgEl.src = cleanedBg;
                    }
                    
                    const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
                    if (layers) {
                        const layerIndex = layers.findIndex(l => (l.name || l) === layerName);
                        if (layerIndex >= 0) {
                            updateLayerState(itemId, layerIndex, { visible: false });
                            renderLayerList(layers, itemId);
                            renderCanvasLayers(itemId);
                        }
                    }
                    return; // Skip standard extraction
                }

            let extractedFile = null;
            let extractedParams = null;
            let extractedDataUrl = null;
            let finalPromptFromB = null;

            if (action !== 'remove') {
                try {
                    const segmented = await segmentSingleLayer({
                        item,
                        layer: {
                            ...(!Array.isArray(layerInfo) ? layerInfo : {}),
                            id: layerInfo?.id || `layer-${Date.now()}`,
                            name: layerName,
                            bbox
                        },
                        onProgress: (message) => tempMsg.update(message)
                    });

                    if (segmented?.dataUrl) {
                        if (segmented.shouldGenerateRuntimeLayer === false || segmented.runtimeAction === 'hold') {
                            const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
                            if (layers) {
                                const layerIndex = layers.findIndex(l => (l.name || l) === layerName);
                                if (layerIndex >= 0) {
                                    updateLayerExtractionMetadata(item, {
                                        id: layers[layerIndex].id,
                                        name: layerName,
                                        cleanPlateLayerId: layers[layerIndex].cleanPlateLayerId,
                                        sourceTextLayerId: layers[layerIndex].sourceTextLayerId
                                    }, {
                                        extractEngine: segmented.extractEngine || 'fastsam',
                                        quality: segmented.quality || {
                                            status: 'low_quality',
                                            runtimeAction: 'hold',
                                            reason: 'quality_gate_hold'
                                        },
                                        bbox: segmented.bbox || bbox
                                    });
                                    renderLayerList(layers, itemId);
                                    renderCanvasLayers(itemId);
                                    await persistWorkbenchItemSemanticState(itemId, item);
                                }
                            }
                            tempMsg.update(`⚠️ **FastSAM 质量不足**: "${layerName}" 已标记为需要高精度模型处理，未生成可编辑图层。`);
                            return;
                        }

	                        const newFile = await dataURLToFile(segmented.dataUrl, `extracted-${Date.now()}.png`);
	                        const parentWidth = parseFloat(item.el.style.width) || 300;
	                        const parentHeight = parseFloat(item.el.style.height) || 300;
	                        const baseX = parseFloat(item.el.style.left) || 0;
	                        const baseY = parseFloat(item.el.style.top) || 0;
	                        const originalBbox = segmented.bbox || bbox;
	                        const layerRect = bboxToWorkbenchRect(originalBbox, baseX, baseY, parentWidth, parentHeight, 1, 1);

	                        addImageToWorkbench(newFile, `拆解-${layerName}`, {
	                            x: layerRect.left,
	                            y: layerRect.top,
	                            initialWidth: layerRect.width,
	                            initialHeight: layerRect.height,
	                            parentId: itemId,
	                            originalBbox,
	                            layerName,
	                            type: 'layer-explode',
	                            zIndex: getExtractedLayerZIndex(item, layerInfo, 0),
	                            extractEngine: segmented.extractEngine || 'fastsam',
	                            quality: segmented.quality || null
	                        });

                        const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
                        if (layers) {
                            const layerIndex = layers.findIndex(l => (l.name || l) === layerName);
                            if (layerIndex >= 0) {
                                updateLayerExtractionMetadata(item, {
                                    id: layers[layerIndex].id,
                                    name: layerName,
                                    cleanPlateLayerId: layers[layerIndex].cleanPlateLayerId,
                                    sourceTextLayerId: layers[layerIndex].sourceTextLayerId
                                }, {
                                    extractEngine: segmented.extractEngine || 'fastsam',
                                    quality: segmented.quality || {
                                        status: 'ok',
                                        score: null,
                                        reason: 'fastsam_extracted'
                                    },
                                    bbox: segmented.bbox || bbox
                                });
                                updateLayerState(itemId, layerIndex, { visible: false });
                                renderLayerList(layers, itemId);
                                renderCanvasLayers(itemId);
                                await persistWorkbenchItemSemanticState(itemId, item);
                            }
                        }

                        tempMsg.update(`🧹 **底板净化**: 正在从原图中擦除 "${layerName}" 并补全背景...`);
                        await applyCleanPlateForExtraction(promptToUse, {
                            ...cleanupLayerInfo,
                            bbox: segmented.bbox || cleanupLayerInfo?.bbox || bbox,
                            name: cleanupLayerInfo?.name || layerName,
                            promptHint: cleanupLayerInfo?.promptHint || promptToUse
                        });

                        tempMsg.update(`✅ **FastSAM 提取完成**: "${layerName}" 已生成透明图层。`);
                        return;
                    }
                } catch (error) {
                    if (DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST) {
                        throw new Error(`FastSAM 提取失败，已阻止进入 Gemini 兜底: ${error.message || error}`);
                    }
                    console.warn(`[FastSAM ${layerName}] fallback to legacy extraction:`, error);
                }

                // Pipeline B: Independent Asset Reconstruction
                const img = new Image();
                img.crossOrigin = "anonymous";
                const itemSrc = getProxiedUrl(item.dataUrl);
                if (!itemSrc) {
                    throw new Error('无效的图片地址，无法执行提取');
                }
                img.src = itemSrc;
                await new Promise(r => img.onload = r);

                const [ymin, xmin, ymax, xmax] = bbox;
                const pxMinX = (xmin / 1000) * img.naturalWidth;
                const pxMinY = (ymin / 1000) * img.naturalHeight;
                const pxMaxX = (xmax / 1000) * img.naturalWidth;
                const pxMaxY = (ymax / 1000) * img.naturalHeight;
                const w = pxMaxX - pxMinX;
                const h = pxMaxY - pxMinY;

                // Reduce padding if it's a box layer created manually.
                const isManualBox = (item.layers && item.layers.find(l => l.name === layerName && l.id && l.id.startsWith('box-layer'))) || 
                                    (item.scene && item.scene.layers && item.scene.layers.find(l => l.name === layerName && l.id && l.id.startsWith('box-layer')));
                
                const paddingFactor = 0.15; // Uniformly use 15% padding to ensure strategy-dispatcher has enough background pixels
                const padX = Math.floor(w * paddingFactor);
                const padY = Math.floor(h * paddingFactor);
                const cropX = Math.max(0, Math.floor(pxMinX - padX));
                const cropY = Math.max(0, Math.floor(pxMinY - padY));
                const rawCropW = Math.min(Math.floor(img.naturalWidth) - cropX, Math.floor(w + padX * 2));
                const rawCropH = Math.min(Math.floor(img.naturalHeight) - cropY, Math.floor(h + padY * 2));

                // ==========================================
                // AI SUPER-RESOLUTION UPSCALE (Local Preparation)
                // ==========================================
                // If the bounding box is small, we aggressively upscale the crop to 1024px before giving it to the AI.
                // The generative model will automatically "Super-Resolve" and sharpen edges during the reconstruction/recoloring step.
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
                
                // Use highest quality inherent interpolation for the base up-res
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                ctx.drawImage(img, cropX, cropY, rawCropW, rawCropH, 0, 0, cropW, cropH);
                
                // --- 动态背景色逻辑 (Dynamic Backdrop Selection) ---
                const cropImageData = ctx.getImageData(0, 0, cropW, cropH);
                
                tempMsg.update(`🧠 **正在决策**: "${layerName}"... 执行场景分析调度...`);
                // IMPORT StrategyDispatcher at top (we'll add import right away)
                const strategy = StrategyDispatcher.analyze(cropImageData, layerName);
                console.log(`[Strategy Dispatcher ${layerName}] Path: ${strategy.path} | Reason: ${strategy.reason}`);
                
                let processedAlphaData;
                let transparentDataUrl;
                const taskId = `matte_task_${Date.now()}`;
                
                if (strategy.path === 'cv') {
                    // FAST PATH: Pure CV
                    tempMsg.update(`⚡ **纯前台秒抠**: "${layerName}"... (原因: ${strategy.reason})`);
                    const bgMeanRGB = strategy.metrics?.bgMean ? 
                        [strategy.metrics.bgMean.r, strategy.metrics.bgMean.g, strategy.metrics.bgMean.b] : 
                        [255, 255, 255];
                    
                    processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                        taskId, 
                        cropImageData, // ORIGINAL PIXELS
                        null, 
                        layerName, 
                        (msg) => tempMsg.update(msg),
                        { type: 'cv_euclidean', bgColor: bgMeanRGB } // CV Math knows the exact BG
                    );
                    
                    const finalCanvas = document.createElement('canvas');
                    finalCanvas.width = cropW;
                    finalCanvas.height = cropH;
                    const finalCtx = finalCanvas.getContext('2d');
                    
                    // Construct final image
                    const finalImageData = new ImageData(processedAlphaData, finalCanvas.width, finalCanvas.height);
                    finalCtx.putImageData(finalImageData, 0, 0);
                    
                    const trimmed = trimTransparentCanvas(finalCanvas);
                    cropW = trimmed.width;
                    cropH = trimmed.height;
                    transparentDataUrl = trimmed.toDataURL('image/png');
                } else {
                    // SLOW PATH: Gemini Hybrid
                    
                    const isCloudOrSmoke = layerName.includes('云') || layerName.includes('烟') || layerName.includes('雾') || layerName.includes('火') || layerName.includes('光') || layerName.includes('冰') || layerName.includes('水') || layerName.includes('纱') || layerName.includes('玻璃') || layerName.includes('透明');
                    const isFineDetail = (layerName.includes('发') && !layerName.includes('沙发')) || layerName.includes('毛') || layerName.includes('羽') || layerName.includes('树') || layerName.includes('草') || layerName.includes('叶') || layerName.includes('线') || layerName.includes('网');

                    if (!transparentDataUrl) {
                    const useMaskFormat = isCloudOrSmoke || isFineDetail;

                    // We need the backdrop selection logic for Hybrid
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
                    if (layerName.includes('云') || layerName.includes('雾') || layerName.includes('光')) {
                        // For bright soft things, Black backdrop creates highest contrast and works flawlessly with Luminance opacity
                        bestBackdrop = backdrops[1]; // Black / Magenta in original V5 fallback
                    } else if (layerName.includes('皮') || layerName.includes('木') || layerName.includes('人')) {
                        bestBackdrop = backdrops[2]; // Green
                    } else if (layerName.includes('树') || layerName.includes('草') || layerName.includes('叶')) {
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
                            finalPrompt = `${customPrompt}
                            
CRITICAL REQUIREMENTS:
1. OUTPUT FORMAT: Return ONLY a grayscale mask image.
  - Pure White (#FFFFFF) = opaque parts of the object "${layerName}"
  - Gradients of Gray = semi-transparent edges, fine details, or motion blur
  - Pure Black (#000000) = background
2. NO RENDERING: DO NOT render, composite, or redraw the object. DO NOT cut the object out.
3. PRESERVE DETAILS: You must output smooth grayscale gradients for fuzzy/semi-transparent edges.`;
                        }
                    } else {
                        // Generative Green Screen for hard edge objects
                        finalPrompt = customPrompt || `Act as a precise image extraction tool.
Your task is to extract the object "${layerName}" from the provided cropped image and place it on a perfectly solid, uniform ${bgColorName} background (${bgColorHex}).

CRITICAL REQUIREMENTS:
1. ZERO HALLUCINATION: You MUST NOT reconstruct, redraw, or alter the object in any way.
2. PIXEL FIDELITY: Strictly preserve the original pixels, material, texture, color, and lighting of the object itself.
3. BACKGROUND ONLY: Your ONLY task is to change the background surrounding the object to a mathematically solid ${bgColorName} color (${bgColorHex}). Do not touch the object itself.
${bgColorRule}`;
                    }

                    const croppedDataUrl = cropCanvas.toDataURL('image/png');
                    const croppedFile = await dataURLToFile(croppedDataUrl, `cropped-${Date.now()}.png`);
                    
                    tempMsg.update(`✨ **正在重建**: "${layerName}"... AI 正在根据实体状态补全遮挡并净化背景。`);
                    
                    const geminiResult = await editOrQueryImageWithGemini(finalPrompt, croppedFile);
                    if (!geminiResult.success || !geminiResult.imageData) {
                        throw new Error("Gemini 资产重建失败");
                    }
                    const reconstructedDataUrl = `data:${geminiResult.mimeType};base64,${geminiResult.imageData}`;
                    
                    tempMsg.update(`✨ **正在提取**: "${layerName}"... 执行纯色背景抹除计算。`);
                    
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
                    finalCtx.drawImage(reconstructedImg, 0, 0, cropW, cropH);
                    
                    const originalImageDataObj = ctx.getImageData(0, 0, cropW, cropH);
                    const generatedImageDataObj = finalCtx.getImageData(0, 0, cropW, cropH);
                    
                    if (useMaskFormat) {
                        processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                            taskId, 
                            originalImageDataObj, // ORIGINAL PIXELS
                            reconstructedDataUrl, // Gemini's grayscale mask
                            layerName, 
                            (msg) => tempMsg.update(msg)
                        );
                    } else {
                        processedAlphaData = await globalMatteTaskSystem.enqueueProcess(
                            taskId, 
                            generatedImageDataObj, // Use Gemini's clean output for keys
                            null, 
                            layerName, 
                            (msg) => tempMsg.update(msg),
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
                }

                const newFile = await dataURLToFile(transparentDataUrl, `extracted-${Date.now()}.png`);
                
                const parentWidth = parseFloat(item.el.style.width) || 300;
                const parentHeight = parseFloat(item.el.style.height) || 300;
                
                // Uniform size: roughly 1/8 of the parent's area (1/2.8 of dimensions)
                const baseSize = Math.max(parentWidth, parentHeight) / 2.8;
                // Calculate TRUE pixel ratio to precisely match the cropped image
                const layerRatio = cropW / cropH;
                
                let initialWidth, initialHeight;
                if (layerRatio > 1) {
                    initialWidth = baseSize;
                    initialHeight = baseSize / layerRatio;
                } else {
                    initialHeight = baseSize;
                    initialWidth = baseSize * layerRatio;
                }

	                const baseX = parseFloat(item.el.style.left) || 0;
	                const baseY = parseFloat(item.el.style.top) || 0;
	                const layerRect = bboxToWorkbenchRect(bbox, baseX, baseY, parentWidth, parentHeight, 1, 1);

	                // Generate workbench extraction params temporarily
	                extractedDataUrl = transparentDataUrl;
	                finalPromptFromB = finalPrompt;

	                if (newFile) {
	                    extractedFile = newFile;
	                    extractedParams = {
	                        x: layerRect.left,
	                        y: layerRect.top,
	                        initialWidth: layerRect.width,
	                        initialHeight: layerRect.height,
	                        parentId: itemId,
	                        originalBbox: bbox,
	                        layerName: layerName,
	                        type: 'layer-explode',
	                        zIndex: getExtractedLayerZIndex(item, layerInfo, 0)
	                    };
	                }
            }

            // Pipeline C: Clean Plate Reconstruction (For all layer extractions)
            tempMsg.update(`🧹 **底板净化**: 正在从原图中擦除 "${layerName}" 并补全背景...`);
            const cleanedBg = await applyCleanPlateForExtraction(promptToUse);

            // Hide the layer since it's processed (always hide even if cleanup failed visually)
            const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
            if (layers) {
                const layerIndex = layers.findIndex(l => (l.name || l) === layerName);
                if (layerIndex >= 0) {
                    updateLayerState(itemId, layerIndex, { visible: false });
                    renderLayerList(layers, itemId);
                    renderCanvasLayers(itemId);
                }
            }
            
            // NOW push the extracted image to the workbench visually, AFTER the background cleans
            if (action !== 'remove' && extractedFile && extractedParams) {
                addImageToWorkbench(extractedFile, `拆解-${layerName}`, extractedParams);
                tempMsg.update(`✅ **操作完成**: "${layerName}" 已提取为独立图层。`);
                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat(`提取图层 [${layerName}]`, finalPromptFromB || promptToUse, extractedDataUrl, executeLayerAction);
                }
            } else if (action === 'remove') {
                tempMsg.update(`✅ **操作完成**: "${layerName}" 已处理，底板已净化。`);
            }
                
                setTimeout(() => { if (tempMsg) tempMsg.remove(); }, 3000);
            
        } catch (e) {
            console.error("[LayerAction] Failed:", e);
            if (tempMsg) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `❌ 操作失败: ${e.message}` });
        }
    };

    await executeLayerAction();
}


export function showVideoPromptModal(onConfirm) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    modal.innerHTML = `
        <div style="background: white; padding: 20px; border-radius: 12px; width: 300px;">
            <h3 style="margin-top: 0; color: #2A5C82;">🎥 视频生成描述</h3>
            <textarea id="videoPromptInput" placeholder="请输入视频生成描述（例如：镜头推进，赛博朋克风格...）" 
                style="width: 100%; height: 100px; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 8px; outline: none; font-family: inherit;"></textarea>
            <div style="display: flex; gap: 10px;">
                <button id="cancelVideoPrompt" style="padding: 8px 16px; background: #eee; border: none; border-radius: 6px; cursor: pointer;">取消</button>
                <button id="confirmVideoPrompt" style="padding: 8px 16px; background: #2A5C82; color: white; border: none; border-radius: 6px; cursor: pointer;">生成</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    const confirmBtn = document.getElementById('confirmVideoPrompt');
    const cancelBtn = document.getElementById('cancelVideoPrompt');
    const promptInput = document.getElementById('videoPromptInput');

    confirmBtn.onclick = async () => {
        const videoPrompt = promptInput.value.trim();
        if (!videoPrompt) {
            alert('请输入视频生成描述');
            return;
        }
        document.body.removeChild(modal);
        if (onConfirm) onConfirm(videoPrompt);
    };

    cancelBtn.onclick = () => {
        document.body.removeChild(modal);
    };

    setTimeout(() => promptInput.focus(), 100);
}

export async function handleRecomposeVerification(itemId) {
    const parentItem = workbenchItems.get(itemId);
    if (!parentItem) return;

    const tempMsg = addMessage({ sender: 'bot', type: 'text', content: `🧩 **回贴验证 (Recomposition)**\n正在将提取的资产重新贴回净化后的底板，验证重建完整性...` });

    try {
        // 1. Get current scene composite using the renderer (with high resolution)
        const img = new Image();
        img.crossOrigin = "anonymous";
        const parentSrc = getProxiedUrl(parentItem.cleanPlateDataUrl || parentItem.originalDataUrl);
        if (!parentSrc) {
            throw new Error('无效的背景图片地址，无法执行回贴验证');
        }
        img.src = parentSrc;
        await new Promise(r => img.onload = r);
        
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        
        // Draw background
        ctx.drawImage(img, 0, 0);
        
        // Enable high quality rendering for child items
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // 2. Find and render child items
        const childItems = Array.from(workbenchItems.values()).filter(item => item.parentId === itemId);
        
        // --- MASK GENERATION ---
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = canvas.width;
        maskCanvas.height = canvas.height;
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.fillStyle = 'black';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        maskCtx.fillStyle = 'white';
        const buffer = 30;

        for (const childItem of childItems) {
            if (!childItem.el) continue;

            const img = new Image();
            img.crossOrigin = "anonymous";
            const childSrc = getProxiedUrl(childItem.dataUrl || childItem.originalDataUrl);
            if (!childSrc) {
                continue;
            }
            img.src = childSrc;
            await new Promise(r => img.onload = r);

            // Calculate position relative to parent using logical coordinates
            const parentEl = parentItem.el;
            const childEl = childItem.el;

            const pL = parseFloat(parentEl.style.left) || 0;
            const pT = parseFloat(parentEl.style.top) || 0;
            const pW = parseFloat(parentEl.style.width) || parentEl.offsetWidth;
            const pH = parseFloat(parentEl.style.height) || parentEl.offsetHeight;

            const cL = parseFloat(childEl.style.left) || 0;
            const cT = parseFloat(childEl.style.top) || 0;
            const cW = parseFloat(childEl.style.width) || childEl.offsetWidth;
            const cH = parseFloat(childEl.style.height) || childEl.offsetHeight;
            
            // Container dimensions on canvas (scaled)
            const containerWidthOnCanvas = cW / pW * canvas.width;
            const containerHeightOnCanvas = cH / pH * canvas.height;
            const containerLeftOnCanvas = (cL - pL) / pW * canvas.width;
            const containerTopOnCanvas = (cT - pT) / pH * canvas.height;

            maskCtx.fillRect(
                containerLeftOnCanvas,
                containerTopOnCanvas,
                containerWidthOnCanvas,
                containerHeightOnCanvas
            );

            // Render to composite canvas
            const imgW = img.naturalWidth || img.width;
            const imgH = img.naturalHeight || img.height;
            const imgRatio = imgW / imgH;
            const containerRatio = containerWidthOnCanvas / containerHeightOnCanvas;

            let drawW, drawH;
            if (imgRatio > containerRatio) {
                drawW = containerWidthOnCanvas;
                drawH = containerWidthOnCanvas / imgRatio;
            } else {
                drawH = containerHeightOnCanvas;
                drawW = containerHeightOnCanvas * imgRatio;
            }

            const transform = childItem.el.style.transform || '';
            const rotateMatch = transform.match(/rotate\(([-\d.]+)deg\)/);
            const rotation = rotateMatch ? parseFloat(rotateMatch[1]) * Math.PI / 180 : 0;

            ctx.save();
            ctx.translate(containerLeftOnCanvas + containerWidthOnCanvas / 2, containerTopOnCanvas + containerHeightOnCanvas / 2);
            if (rotation !== 0) ctx.rotate(rotation);
            ctx.drawImage(img, -drawW / 2, -drawH / 2, drawW, drawH);
            ctx.restore();
        }
        
        const maskDataUrl = maskCanvas.toDataURL('image/png');
        // --- END MASK GENERATION ---

        // 3. Generate raw composite result
        const recomposedDataUrl = canvas.toDataURL('image/png');
        
        // --- AI FUSION STEP ---
        tempMsg.update(`🧩 **深度融合 (AI Fusion)**\n已生成基础合成图，正在调用 AI 进行深度光影与透视融合，请稍候...`);

        const fusionPrompt = `Please seamlessly blend the foreground objects into the background. Generate realistic shadows, match the ambient lighting, and ensure correct perspective based on their current positions. Do not change the objects' core identities.`;
        
        // Prepare files
        const baseImageFile = await dataURLToFile(parentItem.cleanPlateDataUrl || parentItem.originalDataUrl, 'base.png');
        const refImageFile = await dataURLToFile(childItems[0].dataUrl || childItems[0].originalDataUrl, 'ref.png'); // Assuming single child for now
        
        let finalDataUrl;
        try {
            finalDataUrl = await performDeepFusion(baseImageFile, refImageFile, maskDataUrl, fusionPrompt);
        } catch (error) {
            console.warn("AI Fusion failed, falling back to raw composite.", error);
            finalDataUrl = recomposedDataUrl;
        }
        
        let finalFile = await dataURLToFile(finalDataUrl, `ai-fused-${Date.now()}.png`);

        // 5. Add to workbench
        const smartPos = window.calculateSmartPosition ? window.calculateSmartPosition(parentItem.el, 1) : { x: 0, y: 0 };
        await addImageToWorkbench(finalFile, `AI融合: ${parentItem.name || '原图'}`, {
            x: smartPos.x,
            y: smartPos.y,
            parentId: itemId,
            type: 'recomposition-result'
        });

        // 6. Show comparison in chat
        const originalUrl = parentItem.originalDataUrl || parentItem.dataUrl;
        
        tempMsg.update(`✅ **AI 深度融合完成**: 已生成光影匹配的融合图像，请在工作台查看。
        
        **对比验证**:
        左侧为原始图，右侧为 AI 深度融合图。`);
        
        // Use the imageData array feature of addMessage
        const comparisonMsg = addMessage({
            sender: 'bot',
            type: 'text',
            content: `🔍 **视觉对比 (左: 原图 | 右: AI 融合)**`,
            imageData: [
                { src: originalUrl, label: '原图' },
                { src: finalDataUrl, label: 'AI 融合' }
            ]
        });

        setTimeout(() => { if (tempMsg) tempMsg.remove(); }, 5000);

    } catch (e) {
        console.error("[Recomposition] Failed:", e);
        tempMsg.update(`❌ **回贴验证失败**: ${e.message}`);
    }
}


// Expose to window for legacy support or inline handlers
window.showCustomConfirm = showCustomConfirm;
window.showVideoPromptModal = showVideoPromptModal;

window.showLayerManagerModal = showLayerManagerModal;
window.closeLayerManagerModal = closeLayerManagerModal;
window.openMagicWandModal = openMagicWandModal;
window.closeMagicWandModal = closeMagicWandModal;
window.openCameraAngleModal = openCameraAngleModal;
window.closeCameraAngleModal = closeCameraAngleModal;
window.updateLayerState = updateLayerState;
window.renderLayerList = renderLayerList;
window.updateCameraAnglePreview = updateCameraAnglePreview;
