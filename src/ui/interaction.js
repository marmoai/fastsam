import { state } from '../core/state.js';
import { runtime } from '../runtime/CoreRuntime.js';
import { SNAP_THRESHOLD, ZOOM_MIN, ZOOM_MAX, ZOOM_STEP, RESIZE_RADIUS, OUTPAINT_RADIUS, CANVAS_CENTER, DEFAULT_ZOOM } from '../core/config.js';
import { deleteSelectedItems, addImageToWorkbench, addTextNoteToWorkbench, addAtmosphereNode } from './workbench-core.js';
import { restoreShapeToWorkbench } from './workbench/shapes.js';
import { showCustomConfirm } from './modals.js';

const { workbenchItems, selectedWorkbenchItems } = state;

let draggingStateTimeout = null;

function startCanvasDragging() {
    const wb = document.getElementById('workbench');
    if (wb) {
        wb.classList.add('is-canvas-dragging');
    }
    if (draggingStateTimeout) {
        clearTimeout(draggingStateTimeout);
        draggingStateTimeout = null;
    }
}

function stopCanvasDraggingDelayed() {
    if (draggingStateTimeout) {
        clearTimeout(draggingStateTimeout);
    }
    draggingStateTimeout = setTimeout(() => {
        const wb = document.getElementById('workbench');
        if (wb) {
            wb.classList.remove('is-canvas-dragging');
        }
        draggingStateTimeout = null;
    }, 200);
}

window.startCanvasDragging = startCanvasDragging;
window.stopCanvasDraggingDelayed = stopCanvasDraggingDelayed;

const workbenchZoomContainer = document.getElementById('workbenchZoomContainer');
const workbenchGrid = document.getElementById('workbenchGrid');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');

function updateWorkbenchItemsScale() {
    const inverseScale = 1 / state.workbenchZoom;
    
    // --- 1. 四角手柄配置 ---
    const resizePos = -RESIZE_RADIUS + 'px';

    // --- 2. 扩图箭头配置 ---
    const outpaintPos = -OUTPAINT_RADIUS + 'px';
    
    // --- 3. [新增] 便利贴视觉配置 ---
    const desiredVisualOffsetY = 40;

    const items = document.querySelectorAll('.workbench-item');
    items.forEach(item => {
        // 如果是环境/灯光节点...
        if (item.classList.contains('atmosphere-node')) {
            item.style.transform = `scale(${inverseScale})`;
            return;
        }

        // --- A. 四角手柄 ---
        const nw = item.querySelector('.resize-nw'); if (nw) { nw.style.transform = `scale(${inverseScale})`; nw.style.top = resizePos; nw.style.left = resizePos; }
        const ne = item.querySelector('.resize-ne'); if (ne) { ne.style.transform = `scale(${inverseScale})`; ne.style.top = resizePos; ne.style.right = resizePos; }
        const sw = item.querySelector('.resize-sw'); if (sw) { sw.style.transform = `scale(${inverseScale})`; sw.style.bottom = resizePos; sw.style.left = resizePos; }
        const se = item.querySelector('.resize-se'); if (se) { se.style.transform = `scale(${inverseScale})`; se.style.bottom = resizePos; se.style.right = resizePos; }

        // --- B. 扩图箭头 ---
        const topHandle = item.querySelector('.handle-top'); if (topHandle) { topHandle.style.transform = `translateX(-50%) scale(${inverseScale})`; topHandle.style.top = outpaintPos; }
        const bottomHandle = item.querySelector('.handle-bottom'); if (bottomHandle) { bottomHandle.style.transform = `translateX(-50%) scale(${inverseScale})`; bottomHandle.style.bottom = outpaintPos; }
        const leftHandle = item.querySelector('.handle-left'); if (leftHandle) { leftHandle.style.transform = `translateY(-50%) scale(${inverseScale})`; leftHandle.style.left = outpaintPos; }
        const rightHandle = item.querySelector('.handle-right'); if (rightHandle) { rightHandle.style.transform = `translateY(-50%) scale(${inverseScale})`; rightHandle.style.right = outpaintPos; }

        // --- C. [核心修复] 便利贴的动态更新逻辑 ---
        const notesOnThisItem = item.querySelectorAll('.sticky-note');
        if (notesOnThisItem.length > 0) {
            notesOnThisItem.forEach((note, noteIndex) => {
                // 1. 修复间距：每次缩放都重新计算并应用 top 值
                const verticalOffset = noteIndex * (desiredVisualOffsetY / state.workbenchZoom);
                note.style.top = `${verticalOffset}px`;

                // 2. 修复悬浮：用 `:hover` 伪类实时判断悬浮状态，并更新 transform
                const isHovered = note.matches(':hover');
                const scaleMultiplier = isHovered ? 1.05 : 1;
                const rotation = isHovered ? 'rotate(0deg)' : 'rotate(1deg)';
                note.style.transform = `scale(${inverseScale * scaleMultiplier}) ${rotation}`;
            });
        }

        // --- D. [新增] 编组标签的缩放处理 ---
        if (item.classList.contains('workbench-group-label')) {
            const isHovered = item.matches(':hover');
            const scaleMultiplier = isHovered ? 1.1 : 1;
            item.style.transform = `scale(${inverseScale * scaleMultiplier})`;
        }

        // --- E. [新增] 裁切手柄 ---
        const cropHandles = item.querySelectorAll('.crop-handle');
        cropHandles.forEach(handle => {
            handle.style.transform = `scale(${inverseScale})`;
        });

        // --- F. [新增] 旋转手柄 ---
        const rotateHandle = item.querySelector('.rotate-handle');
        if (rotateHandle) {
            rotateHandle.style.transform = `translateX(-50%) scale(${inverseScale})`;
        }

        // --- G. [新增] AI 创意建议 Spark 徽标/触发按钮的缩放 ---
        const spark = item.querySelector('.ai-spark-trigger');
        if (spark) {
            const isHovered = spark.matches(':hover');
            const scaleMultiplier = isHovered ? 1.15 : 1;
            spark.style.transform = `scale(${inverseScale * scaleMultiplier})`;
        }
    });

    // --- D. 其他元素更新 ---
    const connectors = document.querySelectorAll('.world-connector');
    connectors.forEach(connector => { connector.style.transform = `translate(-50%, -50%) scale(${inverseScale})`; });
    
    // --- E. [新增] 世界占位符的缩放处理 ---
    const placeholders = document.querySelectorAll('.world-placeholder');
    placeholders.forEach(placeholder => {
        const visualBorderWidth = 2;
        const visualFontSize = 12;
        placeholder.style.borderWidth = `${visualBorderWidth * inverseScale}px`;
        const span = placeholder.querySelector('span');
        if (span) {
            span.style.fontSize = `${visualFontSize * inverseScale}px`;
        }
    });
    
    const vLine = document.getElementById('guide-v');
    const hLine = document.getElementById('guide-h');
    const visualThickness = 1; 
    const adjustedThickness = visualThickness / state.workbenchZoom;
    if (vLine) { vLine.style.borderLeftWidth = `${adjustedThickness}px`; }
    if (hLine) { hLine.style.borderTopWidth = `${adjustedThickness}px`; }

    const genealogyLayer = document.getElementById('genealogyLayer');
    if (genealogyLayer) {
        const visualStrokeWidth = 1.5;
        const visualHoverStrokeWidth = 2.5;
        const adjustedStrokeWidth = visualStrokeWidth / state.workbenchZoom;
        const adjustedHoverStrokeWidth = visualHoverStrokeWidth / state.workbenchZoom;
        genealogyLayer.style.setProperty('--genealogy-stroke-width', `${adjustedStrokeWidth}px`);
        genealogyLayer.style.setProperty('--genealogy-hover-stroke-width', `${adjustedHoverStrokeWidth}px`);
    }
}

// 重置缩放
function resetWorkbenchZoom() {
    applyWorkbenchZoom(1);
}

// 修改 updateZoomIndicator 函数
function updateZoomIndicator() {
    const zoomLevelElement = document.getElementById('zoomLevel');
    if (zoomLevelElement) {
        zoomLevelElement.textContent = `${Math.round(state.workbenchZoom * 100)}%`;
    }
}

// 新增键盘快捷键监听
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + 加号放大
        if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '=')) {
            e.preventDefault();
            zoomIn();
        }
        // Ctrl/Cmd + 减号缩小
        if ((e.ctrlKey || e.metaKey) && e.key === '-') {
            e.preventDefault();
            zoomOut();
        }
        // Ctrl/Cmd + 0 重置
        if ((e.ctrlKey || e.metaKey) && e.key === '0') {
            e.preventDefault();
            resetZoom();
        }
        
        // Undo/Redo
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
            e.preventDefault();
            if (e.shiftKey) {
                // Redo (Cmd+Shift+Z or Ctrl+Shift+Z)
                if (window.historyManager) window.historyManager.redo();
            } else {
                // Undo (Cmd+Z or Ctrl+Z)
                if (window.historyManager) window.historyManager.undo();
            }
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
            // Redo (Ctrl+Y)
            e.preventDefault();
            if (window.historyManager) window.historyManager.redo();
        }
    });
}

// 修改现有的 zoomIn 函数
function zoomIn() {
    const newZoom = state.workbenchZoom * 1.2;
    applyWorkbenchZoom(Math.min(ZOOM_MAX, newZoom));
}

// 修改现有的 zoomOut 函数  
function zoomOut() {
    const newZoom = state.workbenchZoom * 0.8;
    applyWorkbenchZoom(Math.max(ZOOM_MIN, newZoom));
}

// 修改现有的 resetZoom 函数
function resetZoom() {
    resetPan(); 
}

// 初始化画布拖拽功能
function initWorkbenchPan() {
    workbenchGrid.addEventListener('mousedown', handlePanStart);
    document.addEventListener('mousemove', handlePanMove);
    document.addEventListener('mouseup', handlePanEnd);
    workbenchGrid.addEventListener('auxclick', (e) => {
        if (e.button === 1) { // 中键
            e.preventDefault();
        }
    });
}

// 开始拖拽画布
function handlePanStart(e) {
    if (state.isCropping) return;
    if (typeof window.currentDrawingShape !== 'undefined' && window.currentDrawingShape) return;

    if (typeof window.hideWorkbenchToolbox === 'function') {
        window.hideWorkbenchToolbox();
    }

    if (e.button === 1 || (e.button === 0 && (e.shiftKey || state.isSpacePressed))) {
        e.preventDefault();
        state.isPanning = true;
        state.panStartX = e.clientX - state.panOffsetX * state.workbenchZoom;
        state.panStartY = e.clientY - state.panOffsetY * state.workbenchZoom;
        workbenchGrid.classList.add('panning');
        document.body.style.userSelect = 'none';
        startCanvasDragging();
    }
}

let panAnimationFrameId = null;
let latestPanEvent = null;

function processPanMove() {
    if (!latestPanEvent || !state.isPanning) return;
    const e = latestPanEvent;
    const deltaX = (e.clientX - state.panStartX) / state.workbenchZoom;
    const deltaY = (e.clientY - state.panStartY) / state.workbenchZoom;
    state.panOffsetX = deltaX;
    state.panOffsetY = deltaY;
    applyPanTransform();
    panAnimationFrameId = null;
}

function handlePanMove(e) {
    if (!state.isPanning) return;
    e.preventDefault();
    latestPanEvent = e;
    startCanvasDragging();
    if (!panAnimationFrameId) {
        panAnimationFrameId = requestAnimationFrame(processPanMove);
    }
}

// 结束拖拽画布
function handlePanEnd(e) {
    if (panAnimationFrameId) {
        cancelAnimationFrame(panAnimationFrameId);
        panAnimationFrameId = null;
    }
    if (!state.isPanning) return;
    state.isPanning = false;
    workbenchGrid.classList.remove('panning');
    document.body.style.userSelect = '';
    stopCanvasDraggingDelayed();
}

function applyPanTransform() {
    workbenchZoomContainer.style.transform = `
        translate(${state.panOffsetX * state.workbenchZoom}px, ${state.panOffsetY * state.workbenchZoom}px)
        scale(${state.workbenchZoom})
    `;
    window.updateBackgroundGrid();
}

function resetPan() {
    const viewport = document.getElementById('workbench');
    state.workbenchZoom = DEFAULT_ZOOM;
    updateZoomIndicator();

    if (viewport) {
        const viewportW = viewport.clientWidth || window.innerWidth;
        const viewportH = viewport.clientHeight || window.innerHeight;
        state.panOffsetX = (viewportW / 2) / state.workbenchZoom - CANVAS_CENTER;
        state.panOffsetY = (viewportH / 2) / state.workbenchZoom - CANVAS_CENTER;
    } else {
        state.panOffsetX = -CANVAS_CENTER;
        state.panOffsetY = -CANVAS_CENTER;
    }
    
    applyPanTransform();
    updateWorkbenchItemsScale();
}

function applyWorkbenchZoom(zoomLevel, event = null) {
    const oldZoom = state.workbenchZoom;
    state.workbenchZoom = zoomLevel;
    workbenchZoomContainer.style.transformOrigin = '0 0';
    workbenchZoomContainer.style.transform = `
        translate(${state.panOffsetX * state.workbenchZoom}px, ${state.panOffsetY * state.workbenchZoom}px)
        scale(${state.workbenchZoom})
    `;
    
    let centerScreenX, centerScreenY;
    const viewport = document.getElementById('workbench');
    const rect = viewport.getBoundingClientRect();

    if (event) {
        centerScreenX = event.clientX - rect.left;
        centerScreenY = event.clientY - rect.top;
    } else {
        centerScreenX = rect.width / 2;
        centerScreenY = rect.height / 2;
    }

    state.panOffsetX = state.panOffsetX + (centerScreenX / state.workbenchZoom - centerScreenX / oldZoom);
    state.panOffsetY = state.panOffsetY + (centerScreenY / state.workbenchZoom - centerScreenY / oldZoom);
    document.querySelectorAll('.workbench-item.selected').forEach(item => {
        const inverseScale = 1 / state.workbenchZoom;
        item.style.borderWidth = `${2 * inverseScale}px`;
    });

    applyPanTransform();
    updateZoomIndicator();
    updateWorkbenchItemsScale();
}

function viewAllWorkbenchItems() {
    if (workbenchItems.size === 0) {
        resetPan();
        return;
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    workbenchItems.forEach(item => {
        const el = item.el;
        const left = parseFloat(el.style.left);
        const top = parseFloat(el.style.top);
        const width = parseFloat(el.style.width) || el.offsetWidth;
        const height = parseFloat(el.style.height) || el.offsetHeight;
        if (!isNaN(left) && !isNaN(top) && !isNaN(width) && !isNaN(height)) {
            minX = Math.min(minX, left);
            minY = Math.min(minY, top);
            maxX = Math.max(maxX, left + width);
            maxY = Math.max(maxY, top + height);
        }
    });

    if (minX === Infinity) {
        resetPan();
        return;
    }

    const padding = 100;
    minX -= padding;
    minY -= padding;
    maxX += padding;
    maxY += padding;

    const viewport = document.getElementById('workbench');
    const viewportW = viewport.clientWidth || window.innerWidth;
    const viewportH = viewport.clientHeight || window.innerHeight;

    const contentW = maxX - minX;
    const contentH = maxY - minY;

    let newZoom = Math.min(viewportW / contentW, viewportH / contentH);
    newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newZoom));

    const contentCenterX = (minX + maxX) / 2;
    const contentCenterY = (minY + maxY) / 2;

    state.workbenchZoom = newZoom;
    state.panOffsetX = (viewportW / 2) / state.workbenchZoom - contentCenterX;
    state.panOffsetY = (viewportH / 2) / state.workbenchZoom - contentCenterY;

    applyPanTransform();
    updateZoomIndicator();
    updateWorkbenchItemsScale();
}

function screenToWorkbenchCoord(screenX, screenY) {
    const containerRect = workbenchZoomContainer.getBoundingClientRect();
    const physicalX = screenX - containerRect.left;
    const physicalY = screenY - containerRect.top;
    const logicalX = physicalX / state.workbenchZoom;
    const logicalY = physicalY / state.workbenchZoom;
    return { x: logicalX, y: logicalY };
}

function updateSelectedItems(left, top, width, height, isCrossingSelection = false) {
    if (left === undefined) return;
    const selectionRect = { 
        left: left, 
        top: top, 
        right: left + width, 
        bottom: top + height 
    };
    
    document.querySelectorAll('.workbench-item').forEach(item => {
        if (item.classList.contains('atmosphere-node')) return;

        const itemRect = item.getBoundingClientRect();
        const containerRect = workbenchZoomContainer.getBoundingClientRect();
        
        const itemLeft = (itemRect.left - containerRect.left) / state.workbenchZoom;
        const itemTop = (itemRect.top - containerRect.top) / state.workbenchZoom;
        const itemRight = itemLeft + (itemRect.width / state.workbenchZoom);
        const itemBottom = itemTop + (itemRect.height / state.workbenchZoom);
        
        let isSelected = false;
        
        if (isCrossingSelection) {
            // Crossing selection (right to left): Just needs to intersect
            isSelected = !(itemRight < selectionRect.left || 
                           itemLeft > selectionRect.right || 
                           itemBottom < selectionRect.top || 
                           itemTop > selectionRect.bottom);
        } else {
            // Window selection (left to right): Must be fully contained
            isSelected = (itemLeft >= selectionRect.left && 
                          itemRight <= selectionRect.right && 
                          itemTop >= selectionRect.top && 
                          itemBottom <= selectionRect.bottom);
        }
        
        if (isSelected) {
            item.classList.add('selected');
            selectedWorkbenchItems.add(item.dataset.itemId);
        } else if (!state.isCtrlPressed) {
            item.classList.remove('selected');
            selectedWorkbenchItems.delete(item.dataset.itemId);
        }
    });
}

function handleWorkbenchDragStart(event) {
    if (state.isSpacePressed || state.isCropping) return;
    
    if (typeof window.hideWorkbenchToolbox === 'function') {
        window.hideWorkbenchToolbox();
    }
    
    const itemEl = event.currentTarget;
    const itemId = itemEl.dataset.itemId;
    
    const wasSelected = selectedWorkbenchItems.has(itemId);
    let hasMoved = false;
    
    if (!wasSelected) {
        if (!state.isCtrlPressed) {
            document.querySelectorAll('.workbench-item').forEach(el => el.classList.remove('selected'));
            selectedWorkbenchItems.clear();
        }
        itemEl.classList.add('selected');
        selectedWorkbenchItems.add(itemId);
    }

    if (state.isAltPressed) {
        selectedWorkbenchItems.forEach(id => {
            const original = workbenchItems.get(id);
            if (!original) return;
            
            const newId = `item-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            const rect = {
                left: original.el.style.left,
                top: original.el.style.top,
                width: original.el.style.width,
                height: original.el.style.height,
                zIndex: original.el.style.zIndex
            };

            if (original.el.classList.contains('workbench-text-note')) {
                const content = original.el.querySelector('.note-content').textContent;
                const style = original.el.querySelector('.note-content').style;
                addTextNoteToWorkbench(0, 0, {
                    id: newId,
                    content: content,
                    fontSize: style.fontSize,
                    fontColor: style.color,
                    fontFamily: style.fontFamily,
                    fontWeight: style.fontWeight,
                    fontStyle: style.fontStyle,
                    lineHeight: style.lineHeight,
                    letterSpacing: style.letterSpacing,
                    textAlign: style.textAlign,
                    textShadow: style.textShadow,
                    WebkitTextStroke: style.webkitTextStroke,
                    css: original.css || null,
                    rect: rect
                });
            } else if (original.el.classList.contains('atmosphere-node')) {
                const optKey = original.el.dataset.atmType;
                addAtmosphereNode(optKey, parseFloat(rect.left) + 30, parseFloat(rect.top) + 30);
            } else if (original.el.classList.contains('workbench-shape')) {
                const shapeType = original.el.dataset.shapeType;
                restoreShapeToWorkbench({
                    id: newId,
                    shapeType: shapeType,
                    rect: rect,
                    content: original.el.innerHTML,
                    fillColor: original.el.style.backgroundColor || null,
                    borderColor: original.el.style.borderColor || null,
                    borderWidth: original.el.style.borderWidth || null,
                    borderRadius: original.el.style.borderRadius || null,
                    clipPath: original.el.style.clipPath || original.el.style.webkitClipPath || null,
                    parentId: original.parentId || null,
                    layerName: original.layerName || null,
                    originalBbox: original.originalBbox || null
                });
            } else {
                const metadata = {
                    ...original,
                    id: newId,
                    left: rect.left,
                    top: rect.top,
                    initialWidth: parseFloat(rect.width),
                    initialHeight: parseFloat(rect.height),
                    zIndex: rect.zIndex
                };
                delete metadata.el;
                addImageToWorkbench(original.file, '', metadata);
            }
        });
    }

    const snapPointsX = [];
    const snapPointsY = [];
    
    workbenchItems.forEach((item, id) => {
        if (selectedWorkbenchItems.has(id)) return;
        
        const rect = {
            left: parseFloat(item.el.style.left),
            top: parseFloat(item.el.style.top),
            width: parseFloat(item.el.style.width),
            height: parseFloat(item.el.style.height)
        };
        
        snapPointsX.push(rect.left, rect.left + rect.width / 2, rect.left + rect.width);
        snapPointsY.push(rect.top, rect.top + rect.height / 2, rect.top + rect.height);
    });

    const startX = event.clientX;
    const startY = event.clientY;
    
    const workspace = runtime.getCurrentWorkspace();

    const initialPositions = new Map();
    const addInitialPosition = (id) => {
        if (initialPositions.has(id)) return;
        const item = workbenchItems.get(id);
        if (item && item.el) {
            const asset = workspace ? workspace.currentState.assetRegistry.get(id) : null;
            const startLeft = asset?.transform?.x ?? parseFloat(item.el.style.left);
            const startTop = asset?.transform?.y ?? parseFloat(item.el.style.top);

            initialPositions.set(id, {
                left: startLeft,
                top: startTop,
                originalZIndex: item.el.style.zIndex,
                el: item.el
            });
            item.el.style.zIndex = '1000';
            
            // Add children too
            workbenchItems.forEach((childItem, childId) => {
                if (childItem.parentId === id) addInitialPosition(childId);
            });
        }
    };
    
    selectedWorkbenchItems.forEach(id => addInitialPosition(id));

    const primaryAsset = workspace ? workspace.currentState.assetRegistry.get(itemId) : null;
    const primaryStartLeft = primaryAsset?.transform?.x ?? parseFloat(itemEl.style.left);
    const primaryStartTop = primaryAsset?.transform?.y ?? parseFloat(itemEl.style.top);
    const itemWidth = parseFloat(itemEl.style.width);
    const itemHeight = parseFloat(itemEl.style.height);

    const vLine = document.getElementById('guide-v');
    const hLine = document.getElementById('guide-h');

    let animationFrameId = null;
    let latestMoveEvent = null;

    function processMove() {
        if (!latestMoveEvent) return;
        const moveEvent = latestMoveEvent;
        
        const dx = (moveEvent.clientX - startX) / Math.max(state.workbenchZoom || 1, 0.1);
        const dy = (moveEvent.clientY - startY) / Math.max(state.workbenchZoom || 1, 0.1);
        
        let targetLeft = primaryStartLeft + dx;
        let targetTop = primaryStartTop + dy;

        const currentPointsX = [targetLeft, targetLeft + itemWidth / 2, targetLeft + itemWidth];
        const currentPointsY = [targetTop, targetTop + itemHeight / 2, targetTop + itemHeight];

        let bestSnapX = { diff: Infinity, pos: null, linePos: null };
        let bestSnapY = { diff: Infinity, pos: null, linePos: null };

        snapPointsX.forEach(refX => {
            currentPointsX.forEach((curX, idx) => {
                const diff = Math.abs(curX - refX);
                if (diff < SNAP_THRESHOLD && diff < bestSnapX.diff) {
                    const offset = [0, itemWidth / 2, itemWidth][idx];
                    bestSnapX = { diff, pos: refX - offset, linePos: refX };
                }
            });
        });

        snapPointsY.forEach(refY => {
            currentPointsY.forEach((curY, idx) => {
                const diff = Math.abs(curY - refY);
                if (diff < SNAP_THRESHOLD && diff < bestSnapY.diff) {
                    const offset = [0, itemHeight / 2, itemHeight][idx];
                    bestSnapY = { diff, pos: refY - offset, linePos: refY };
                }
            });
        });

        if (bestSnapX.pos !== null) {
            targetLeft = bestSnapX.pos;
            if(vLine) {
                vLine.style.left = `${bestSnapX.linePos}px`;
                vLine.style.display = 'block';
            }
        } else if(vLine) {
            vLine.style.display = 'none';
        }

        if (bestSnapY.pos !== null) {
            targetTop = bestSnapY.pos;
            if(hLine) {
                hLine.style.top = `${bestSnapY.linePos}px`;
                hLine.style.display = 'block';
            }
        } else if(hLine) {
            hLine.style.display = 'none';
        }

        const effectiveDx = targetLeft - primaryStartLeft;
        const effectiveDy = targetTop - primaryStartTop;

        const workspace = runtime.getCurrentWorkspace();

        initialPositions.forEach((startPos, id) => {
            const finalLeft = startPos.left + effectiveDx;
            const finalTop = startPos.top + effectiveDy;
            
            // Immediate UI responsivness
            startPos.el.style.left = `${finalLeft}px`;
            startPos.el.style.top = `${finalTop}px`;
            if (window.updateParentBBox) {
                window.updateParentBBox(id);
            }
        });

        if (typeof window.drawGenealogyConnections === 'function') {
            window.drawGenealogyConnections();
        }

        if (itemEl.classList.contains('atmosphere-node')) {
            const lightNode = workbenchItems.get(itemId);
            if (lightNode && typeof window.updateRelightingPreview === 'function') {
                window.updateRelightingPreview(lightNode);
            }
        }

        const draggedRect = itemEl.getBoundingClientRect();
        let maxIntersection = 0;
        let potentialTarget = null;
        
        document.querySelectorAll('.workbench-item').forEach(item => {
            if (selectedWorkbenchItems.has(item.dataset.itemId)) return;
            
            const itemRect = item.getBoundingClientRect();
            const intersection = typeof window.getIntersectionArea === 'function' ? window.getIntersectionArea(draggedRect, itemRect) : 0;
            const itemArea = itemRect.width * itemRect.height;
            
            if (intersection > itemArea * 0.5) {
                if (intersection > maxIntersection) {
                    maxIntersection = intersection;
                    potentialTarget = item;
                }
            }
            item.classList.remove('collision-active');
        });
        
        if (potentialTarget) {
            potentialTarget.classList.add('collision-active');
            state.collisionTargetId = potentialTarget.dataset.itemId;
            state.injectionSourceId = itemId;
        } else {
            state.collisionTargetId = null;
            state.injectionSourceId = null;
        }
        
        animationFrameId = null;
    }

    function onMouseMove(moveEvent) {
        hasMoved = true;
        latestMoveEvent = moveEvent;
        startCanvasDragging();
        if (!animationFrameId) {
            animationFrameId = requestAnimationFrame(processMove);
        }
    }

    function onMouseUp() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        
        stopCanvasDraggingDelayed();
        
        const workspace = runtime.getCurrentWorkspace();

        initialPositions.forEach((startPos, id) => {
            startPos.el.style.zIndex = startPos.originalZIndex || '1';
        });

        if (workspace && hasMoved) {
            const transformsToBatch = [];
            selectedWorkbenchItems.forEach((id) => {
                const item = workbenchItems.get(id);
                if (item && item.el) {
                    const isNormalAsset = workspace.currentState.assetRegistry.get(id) != null;
                    if (!isNormalAsset) return;

                    // Skip if parent is also selected; parent's dispatch will cascade movement to children automatically!
                    if (item.parentId && selectedWorkbenchItems.has(item.parentId)) {
                        return;
                    }

                    const finalLeft = parseFloat(item.el.style.left);
                    const finalTop = parseFloat(item.el.style.top);
                    transformsToBatch.push({
                        uid: id,
                        transform: { x: finalLeft, y: finalTop }
                    });
                }
            });

            if (transformsToBatch.length > 0) {
                workspace.dispatcher.dispatch({
                    type: 'BATCH_UPDATE_TRANSFORMS',
                    payload: { transforms: transformsToBatch }
                });
            }
        }

        if(vLine) vLine.style.display = 'none';
        if(hLine) hLine.style.display = 'none';
        
        if (state.collisionTargetId && state.injectionSourceId) {
            const targetItem = document.querySelector(`.workbench-item[data-item-id="${state.collisionTargetId}"]`);
            if (targetItem) {
                const rect = targetItem.getBoundingClientRect();
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                
                if (window.injectionMenu) {
                    window.injectionMenu.style.left = `${centerX}px`;
                    window.injectionMenu.style.top = `${centerY}px`;
                    window.injectionMenu.classList.add('active');
                }
            }
        }
        
        if (!hasMoved) {
            if (!state.isCtrlPressed) {
                document.querySelectorAll('.workbench-item').forEach(el => el.classList.remove('selected'));
                selectedWorkbenchItems.clear();
                itemEl.classList.add('selected');
                selectedWorkbenchItems.add(itemId);
            } else {
                if (wasSelected) {
                    itemEl.classList.remove('selected');
                    selectedWorkbenchItems.delete(itemId);
                }
            }
        }
        
        if (typeof window.checkProximity === 'function') {
            window.checkProximity();
        }
        
        if (itemEl.classList.contains('atmosphere-node')) {
            const nodeRect = itemEl.getBoundingClientRect();
            let nearestImg = null;
            let minDistance = Infinity;
            
            workbenchItems.forEach((imgItem, imgId) => {
                if (imgId === itemId || imgItem.el.classList.contains('atmosphere-node')) return;
                
                const isImage = (item) => item.type !== 'text-note' && item.type !== 'shape';
                if (!isImage(imgItem)) return;
                
                const imgRect = imgItem.el.getBoundingClientRect();
                const dx = (nodeRect.left + nodeRect.width/2) - (imgRect.left + imgRect.width/2);
                const dy = (nodeRect.top + nodeRect.height/2) - (imgRect.top + imgRect.height/2);
                const dist = Math.hypot(dx, dy);
                
                if (dist < minDistance && dist < 300) {
                    minDistance = dist;
                    nearestImg = { id: imgId, el: imgItem.el, dx, dy };
                }
            });
            
            const btn = itemEl.querySelector('.apply-light-btn');
            if (nearestImg) {
                const angle = Math.atan2(nearestImg.dy, nearestImg.dx) * 180 / Math.PI;
                let dir = "";
                if (angle > -45 && angle <= 45) dir = "左侧";
                else if (angle > 45 && angle <= 135) dir = "上方";
                else if (angle > -135 && angle <= -45) dir = "下方";
                else dir = "右侧";
                
                if (btn) {
                    btn.style.display = 'block';
                    btn.textContent = `应用${dir}光照`;
                    btn.onclick = () => {
                        if (typeof window.applyAtmosphereToImage === 'function') {
                            window.applyAtmosphereToImage(itemId, nearestImg.id);
                        }
                    };
                }
            } else {
                if (btn) btn.style.display = 'none';
            }
        }
        
        if (window.historyManager) window.historyManager.pushState();
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
}

function showWorkbenchToolbox(id) {
    state.currentActiveWorkbenchItemId = id;
    const item = workbenchItems.get(id);
    if (!item) return;

    if (typeof window.triggerCapsuleAlert === 'function') {
        window.triggerCapsuleAlert(id);
    }

    window.workbenchToolbox.style.display = 'flex';
    
    const toolboxWidth = window.workbenchToolbox.offsetWidth || 350; 
    const toolboxHeight = window.workbenchToolbox.offsetHeight || 60;
    
    const rect = item.el.getBoundingClientRect();
    
    let posX = rect.left + rect.width / 2 - toolboxWidth / 2;
    let posY = rect.top - toolboxHeight - 20;
    
    if (posX < 10) posX = 10;
    if (posX + toolboxWidth > window.innerWidth - 10) posX = window.innerWidth - toolboxWidth - 10;
    
    if (posY < 10) {
        posY = rect.bottom + 20;
    }
    
    window.workbenchToolbox.style.left = `${posX}px`;
    window.workbenchToolbox.style.top = `${posY}px`;

  //  console.log('showWorkbenchToolbox called for item:', id, item);
    
    // Smooth selection transition: Do not pre-hide the editor so it updates smoothly in-place without flashing
    if (item.type === 'text-note') {
        window.workbenchToolbox.style.display = 'none';
        if (typeof window.hideFloatingFusionEditor === 'function') {
            window.hideFloatingFusionEditor();
        }
        import('./workbench/notes.js').then(({ showTextAdjuster }) => {
            showTextAdjuster(item.el, posX + toolboxWidth/2, posY);
        });
        return;
    } else {
        const textAdjuster = document.getElementById('textAdjuster');
        if (textAdjuster) textAdjuster.style.display = 'none';
    }

    // Check if the AI Layer/Fusion panel is already open (and not collapsing)
    const floatingEditor = document.getElementById('floatingFusionEditor');
    const isFusionEditorOpen = floatingEditor !== null && !floatingEditor.hasAttribute('data-collapsing');

    if (isFusionEditorOpen) {
        // If it's already open, sync it so user can still edit colors
        if (typeof window.showFloatingFusionEditor === 'function') {
            window.showFloatingFusionEditor(id);
        }
        // Hide toolbox to prevent interference
        window.workbenchToolbox.style.display = 'none';
    } else {
        // AI suggestions / editor is CLOSED: Show the standard toolbox for both ordinary assets and layers.
        window.workbenchToolbox.style.display = 'flex';
    }
}

let isResizing = false;
let resizeState = null;

function handleResizeStart(e, direction, itemEl) {
    if (state.isCropping) return;
    e.stopPropagation();
    e.preventDefault();

    isResizing = true;
    
    const itemId = itemEl.dataset.itemId;
    const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
    const asset = workspace ? workspace.currentState.assetRegistry.get(itemId) : null;
    
    const currentWidth = asset?.transform?.width ?? (parseFloat(itemEl.style.width) || itemEl.offsetWidth);
    const currentHeight = asset?.transform?.height ?? (parseFloat(itemEl.style.height) || itemEl.offsetHeight);
    const startLeft = asset?.transform?.x ?? (parseFloat(itemEl.style.left) || 0);
    const startTop = asset?.transform?.y ?? (parseFloat(itemEl.style.top) || 0);
    
    const isNormalAsset = !itemEl.classList.contains('workbench-text-note') && 
                          !itemEl.classList.contains('atmosphere-node') && 
                          !itemEl.classList.contains('workbench-shape');
    
    let baseTransform = null;
    if (isNormalAsset && asset) {
        baseTransform = {
            x: asset.transform?.x ?? startLeft,
            y: asset.transform?.y ?? startTop,
            width: asset.transform?.width ?? currentWidth,
            height: asset.transform?.height ?? currentHeight,
            rotation: asset.transform?.rotation ?? 0,
            zIndex: asset.transform?.zIndex ?? (parseFloat(itemEl.style.zIndex) || 1)
        };
    }
    
    let startFontSize = 24;
    if (itemEl.classList.contains('workbench-text-note')) {
        const contentEl = itemEl.querySelector('.note-content');
        if (contentEl) {
            startFontSize = parseFloat(window.getComputedStyle(contentEl).fontSize) || 24;
        }
    }
    
    resizeState = {
        direction: direction,
        itemEl: itemEl,
        startX: e.clientX,
        startY: e.clientY,
        startWidth: currentWidth,
        startHeight: currentHeight,
        startLeft: startLeft,
        startTop: startTop,
        aspectRatio: currentWidth / currentHeight,
        startFontSize: startFontSize,
        isNormalAsset: isNormalAsset,
        baseTransform: baseTransform,
        pendingTransform: null
    };

    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);
}

let resizeAnimationFrameId = null;
let latestResizeEvent = null;

function processResize() {
    if (!latestResizeEvent || !isResizing || !resizeState) return;
    const e = latestResizeEvent;

    const { direction, itemEl, startX, startWidth, startHeight, startLeft, startTop, aspectRatio, isNormalAsset } = resizeState;
    
    const zoom = (typeof state.workbenchZoom !== 'undefined') ? state.workbenchZoom : 1;
    const deltaX = (e.clientX - startX) / zoom;
    
    const MIN_WIDTH = 30;
    
    let newWidth = startWidth;
    let newHeight = 0;
    let newLeft = startLeft;
    let newTop = startTop;

    if (direction === 'se') {
        newWidth = Math.max(MIN_WIDTH, startWidth + deltaX);
    } else if (direction === 'sw') {
        newWidth = Math.max(MIN_WIDTH, startWidth - deltaX);
        newLeft = startLeft + (startWidth - newWidth);
    } else if (direction === 'ne') {
        newWidth = Math.max(MIN_WIDTH, startWidth + deltaX);
    } else if (direction === 'nw') {
        newWidth = Math.max(MIN_WIDTH, startWidth - deltaX);
        newLeft = startLeft + (startWidth - newWidth);
    }

    newHeight = newWidth / aspectRatio;

    if (direction === 'ne' || direction === 'nw') {
        newTop = startTop + (startHeight - newHeight);
    }

    if (isNormalAsset) {
        itemEl.style.width = `${newWidth}px`;
        itemEl.style.height = `${newHeight}px`;
        itemEl.style.left = `${newLeft}px`;
        itemEl.style.top = `${newTop}px`;
        
        resizeState.pendingTransform = {
            x: newLeft,
            y: newTop,
            width: newWidth,
            height: newHeight
        };
    } else {
        if (!itemEl.classList.contains('workbench-text-note')) {
            itemEl.style.width = `${newWidth}px`;
            itemEl.style.height = `${newHeight}px`;
            itemEl.style.left = `${newLeft}px`;
            itemEl.style.top = `${newTop}px`;
        } else {
            const contentEl = itemEl.querySelector('.note-content');
            if (contentEl) {
                const scaleRatio = newHeight / resizeState.startHeight;
                const newFontSize = Math.max(1, resizeState.startFontSize * scaleRatio);
                contentEl.style.fontSize = `${newFontSize}px`;
                
                itemEl.style.width = 'auto';
                itemEl.style.height = 'auto';
                
                const actualWidth = itemEl.offsetWidth;
                const actualHeight = itemEl.offsetHeight;
                
                if (direction === 'se') {
                    itemEl.style.left = `${startLeft}px`;
                    itemEl.style.top = `${startTop}px`;
                } else if (direction === 'sw') {
                    itemEl.style.left = `${startLeft + startWidth - actualWidth}px`;
                    itemEl.style.top = `${startTop}px`;
                } else if (direction === 'ne') {
                    itemEl.style.left = `${startLeft}px`;
                    itemEl.style.top = `${startTop + startHeight - actualHeight}px`;
                } else if (direction === 'nw') {
                    itemEl.style.left = `${startLeft + startWidth - actualWidth}px`;
                    itemEl.style.top = `${startTop + startHeight - actualHeight}px`;
                }
            }
        }
    }

    if (typeof window.drawGenealogyConnections === 'function') {
        window.drawGenealogyConnections();
    }
    
    if (window.updateParentBBox) {
        window.updateParentBBox(itemEl.dataset.itemId);
    }
    
    resizeAnimationFrameId = null;
}

function handleResizeMove(e) {
    if (!isResizing || !resizeState) return;
    e.preventDefault();
    latestResizeEvent = e;
    startCanvasDragging();
    if (!resizeAnimationFrameId) {
        resizeAnimationFrameId = requestAnimationFrame(processResize);
    }
}

function handleResizeEnd() {
    if (resizeAnimationFrameId) {
        cancelAnimationFrame(resizeAnimationFrameId);
        resizeAnimationFrameId = null;
    }
    isResizing = false;
    
    let resizedItemId = null;
    let savedItemEl = null;
    let isNormalAsset = false;
    let pendingTransform = null;

    if (resizeState) {
        if (resizeState.itemEl) {
            resizedItemId = resizeState.itemEl.dataset.itemId;
            savedItemEl = resizeState.itemEl;
        }
        isNormalAsset = resizeState.isNormalAsset;
        pendingTransform = resizeState.pendingTransform;
    }
    
    resizeState = null;
    document.removeEventListener('mousemove', handleResizeMove);
    document.removeEventListener('mouseup', handleResizeEnd);
    
    stopCanvasDraggingDelayed();
    
    if (resizedItemId && savedItemEl) {
        if (typeof window.checkProximity === 'function') {
            window.checkProximity();
        }
        
        // --- Sync Resize to MVR Runtime ---
        if (window.mvrRuntime) {
            const workspace = window.mvrRuntime.getCurrentWorkspace();
            if (workspace) {
                if (isNormalAsset && pendingTransform) {
                    workspace.dispatcher.dispatch({
                        type: 'UPDATE_TRANSFORM',
                        payload: {
                            uid: resizedItemId,
                            transform: pendingTransform
                        }
                    });
                }
            }
        }
    }
    
    if (window.historyManager) window.historyManager.pushState();
}

window.updateWorkbenchItemsScale = updateWorkbenchItemsScale;
window.resetWorkbenchZoom = resetWorkbenchZoom;
window.updateZoomIndicator = updateZoomIndicator;
window.setupKeyboardShortcuts = setupKeyboardShortcuts;
window.zoomIn = zoomIn;
window.zoomOut = zoomOut;
window.resetZoom = resetZoom;
window.initWorkbenchPan = initWorkbenchPan;
window.handlePanStart = handlePanStart;
window.processPanMove = processPanMove;
window.handlePanMove = handlePanMove;
window.handlePanEnd = handlePanEnd;
window.applyPanTransform = applyPanTransform;
window.resetPan = resetPan;
window.applyWorkbenchZoom = applyWorkbenchZoom;
window.viewAllWorkbenchItems = viewAllWorkbenchItems;
window.screenToWorkbenchCoord = screenToWorkbenchCoord;
window.updateSelectedItems = updateSelectedItems;
window.handleWorkbenchDragStart = handleWorkbenchDragStart;
window.handleResizeStart = handleResizeStart;
window.showWorkbenchToolbox = showWorkbenchToolbox;

export {
    updateWorkbenchItemsScale,
    resetWorkbenchZoom,
    updateZoomIndicator,
    setupKeyboardShortcuts,
    zoomIn,
    zoomOut,
    resetZoom,
    initWorkbenchPan,
    handlePanStart,
    processPanMove,
    handlePanMove,
    handlePanEnd,
    applyPanTransform,
    resetPan,
    applyWorkbenchZoom,
    viewAllWorkbenchItems,
    screenToWorkbenchCoord,
    updateSelectedItems,
    handleWorkbenchDragStart,
    handleResizeStart,
    showWorkbenchToolbox,
    initGuideLines,
    initWorkbenchZoom,
    handleWorkbenchWheel,
    updateBackgroundGrid,
    setupSelectionBox
};

// Helper to create guide lines
function initGuideLines() {
    const workbenchGrid = document.getElementById('workbenchGrid');
    if (!workbenchGrid || document.getElementById('guide-h')) return;
    
    const hLine = document.createElement('div');
    hLine.id = 'guide-h';
    hLine.className = 'guide-line guide-line-h';
    
    const vLine = document.createElement('div');
    vLine.id = 'guide-v';
    vLine.className = 'guide-line guide-line-v';
    
    workbenchGrid.appendChild(hLine);
    workbenchGrid.appendChild(vLine);
}

// 在initWorkbenchZoom函数中添加背景层初始化
function initWorkbenchZoom() {
    const workbenchGrid = document.getElementById('workbenchGrid');
    if (!workbenchGrid) return;

    // 初始化背景层
    const background = document.getElementById('workbenchBackground');
    
    // 初始化对齐参考线
    initGuideLines();

    // 监听鼠标滚轮事件（带Ctrl键）
    workbenchGrid.addEventListener('wheel', handleWorkbenchWheel, { passive: false });
    
    // 初始化画布拖拽功能
    initWorkbenchPan();
    setupKeyboardShortcuts();
    
    // 初始化视图位置到画布中心 (Infinite Canvas Effect)
    if (!window.__hasRestoredViewport) {
        resetPan();
    } else {
        applyPanTransform();
        updateZoomIndicator();
        updateWorkbenchItemsScale();
    }
    
    // 添加空格键监听（空格+拖拽）
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && !e.target.matches('textarea, input')) {
            e.preventDefault();
            state.isSpacePressed = true;
            workbenchGrid.classList.add('space-pressed');
        }
    });
    
    document.addEventListener('keyup', (e) => {
        if (e.code === 'Space') {
            state.isSpacePressed = false;
            workbenchGrid.classList.remove('space-pressed');
        }
    });
    
    updateZoomIndicator();
}

// 处理滚轮缩放 (Fixed: Math based on 0,0 origin to support pan+zoom)
function handleWorkbenchWheel(e) {
    startCanvasDragging();
    stopCanvasDraggingDelayed();

    // 检查是否按下Ctrl或Cmd键
    if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        
        // 隐藏工具箱
        if (typeof window.hideWorkbenchToolbox === 'function') {
            window.hideWorkbenchToolbox();
        }
        
        // 确定缩放方向
        const delta = e.deltaY > 0 ? -1 : 1;
        
        // 动态计算缩放步长
        const currentZoom = Math.round(state.workbenchZoom * 100) / 100;
        let currentStep = ZOOM_STEP; // 默认 0.05
        
        if (currentZoom < 0.10 || (currentZoom === 0.10 && delta < 0)) {
            currentStep = 0.01; // 10% 以下使用 1% 的步长
        } else if (currentZoom > 1.20 || (currentZoom === 1.20 && delta > 0)) {
            currentStep = 0.2; // 120% 以上使用 20% 的步长
        }
        
        // 计算新的缩放比例
        let newZoom = state.workbenchZoom + (delta * currentStep);
        
        // 限制缩放范围
        newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newZoom));
        
        // 调用缩放应用函数，传入事件对象以计算鼠标中心
        applyWorkbenchZoom(newZoom, e);
    } else {
        // 如果鼠标在 textarea 上，不阻止默认滚动
        if (e.target.closest('textarea') || e.target.closest('.scrollable-panel')) {
            return;
        }

        // 隐藏工具箱
        if (typeof window.hideWorkbenchToolbox === 'function') {
            window.hideWorkbenchToolbox();
        }

        // 普通滚轮滚动，实现画布平移
        e.preventDefault();
        
        // 考虑缩放比例，使得滚动速度在不同缩放层级下保持一致的视觉效果
        const scrollSpeed = 1 / state.workbenchZoom; 
        
        if (e.shiftKey) {
            // 按住Shift键时，水平滚动
            state.panOffsetX -= e.deltaY * scrollSpeed;
        } else {
            // 默认垂直滚动，同时支持触控板的水平滚动(e.deltaX)
            state.panOffsetX -= e.deltaX * scrollSpeed;
            state.panOffsetY -= e.deltaY * scrollSpeed;
        }
        
        applyPanTransform();
    }
}

// 新增函数：更新背景网格
function updateBackgroundGrid() {
    const background = document.getElementById('workbenchBackground');
    if (!background) return;
    const gridSize = 20; // 固定的视觉网格大小（像素）
    
    // 计算背景位置（基于平移偏移量）
    const bgPosX = (state.panOffsetX * state.workbenchZoom) % gridSize;
    const bgPosY = (state.panOffsetY * state.workbenchZoom) % gridSize;
    
    // 应用背景位置
    background.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
}

window.updateBackgroundGrid = updateBackgroundGrid;

function setupSelectionBox() {
    const workbenchGrid = document.getElementById('workbenchGrid');
    if (!workbenchGrid) return;

    workbenchGrid.addEventListener('mousedown', (e) => {
        if (e.button === 1 || e.shiftKey || state.isSpacePressed) return;
        if (e.target !== workbenchGrid || e.button !== 0) return;
        if (e.target.classList.contains('world-connector') || e.target.classList.contains('outpaint-handle') || e.target.closest('.workbench-item.atmosphere-node')) return; // 防止与世界构建工具冲突
        
        workbenchGrid.classList.add('selecting');
        state.isSelecting = true;
        
        state.selectionStart = screenToWorkbenchCoord(e.clientX, e.clientY);
        
        if (!state.isCtrlPressed) {
            document.querySelectorAll('.workbench-item').forEach(el => el.classList.remove('selected'));
            state.selectedWorkbenchItems.clear();
        }
        
        state.selectionBox = document.createElement('div');
        state.selectionBox.className = 'selection-box';

        // 根据缩放比例调整边框宽度
        const baseBorderWidth = 2; // 基础边框宽度（100%缩放时）
        const adjustedBorderWidth = baseBorderWidth / state.workbenchZoom;
        state.selectionBox.style.borderWidth = `${adjustedBorderWidth}px`;
        workbenchGrid.appendChild(state.selectionBox);
    });

    document.addEventListener('mousemove', (e) => {
        if (state.isPanning) {
            handlePanMove(e);
            return;
        }
        
        if (!state.isSelecting) return;
        
        const start = state.selectionStart; // 已经是逻辑坐标
        const current = screenToWorkbenchCoord(e.clientX, e.clientY);

        const left = Math.min(start.x, current.x);
        const top = Math.min(start.y, current.y);
        const width = Math.abs(current.x - start.x);
        const height = Math.abs(current.y - start.y);
        
        const isCrossingSelection = current.x < start.x;
            
        if (state.selectionBox) {
            const baseBorderWidth = 2;
            const adjustedBorderWidth = baseBorderWidth / state.workbenchZoom;
            state.selectionBox.style.borderWidth = `${adjustedBorderWidth}px`;
            
            state.selectionBox.style.left = `${left}px`;
            state.selectionBox.style.top = `${top}px`;
            state.selectionBox.style.width = `${width}px`;
            state.selectionBox.style.height = `${height}px`;
            
            // Visual feedback: crossing selection uses green tint, window selection uses blue tint
            if (isCrossingSelection) {
                state.selectionBox.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
                state.selectionBox.style.borderColor = 'rgba(76, 175, 80, 0.8)';
                state.selectionBox.style.borderStyle = 'dashed';
            } else {
                state.selectionBox.style.backgroundColor = 'rgba(42, 92, 130, 0.1)';
                state.selectionBox.style.borderColor = 'var(--primary-color)';
                state.selectionBox.style.borderStyle = 'solid';
            }
        }
            
        updateSelectedItems(left, top, width, height, isCrossingSelection);
    });

    document.addEventListener('mouseup', (e) => {
        if (state.isSelecting) {
            state.isSelecting = false;
            workbenchGrid.classList.remove('selecting');
            if (state.selectionBox) {
                if (state.selectionBox.parentNode) {
                    state.selectionBox.parentNode.removeChild(state.selectionBox);
                }
                state.selectionBox = null;
            }
            if (typeof window.checkProximity === 'function') {
                window.checkProximity();
            }
        }
        
        if (state.isPanning) {
            handlePanEnd(e);
        }
    });
}

// --- Cropping Mode Logic ---

window.enterCroppingMode = function(id) {
    if (state.isCropping) return;
    
    const item = workbenchItems.get(id);
    if (!item || item.el.classList.contains('atmosphere-node') || item.el.classList.contains('workbench-text-note') || item.el.classList.contains('workbench-shape')) return;

    state.isCropping = true;
    state.croppingItemId = id;
    
    const el = item.el;
    el.classList.add('cropping');
    
    // Hide toolbox
    if (window.hideWorkbenchToolbox) window.hideWorkbenchToolbox();
    
    // Store original dimensions and current crop
    if (!item.originalWidth) {
        item.originalWidth = parseFloat(el.style.width);
        item.originalHeight = parseFloat(el.style.height);
        item.crop = { top: 0, left: 0, right: 0, bottom: 0 };
        item.originalLeft = parseFloat(el.style.left);
        item.originalTop = parseFloat(el.style.top);
    }
    
    // Ensure image is wrapped in crop-container
    let container = el.querySelector('.crop-container');
    if (!container) {
        const img = el.querySelector('img');
        if (img) {
            container = document.createElement('div');
            container.className = 'crop-container';
            container.style.width = '100%';
            container.style.height = '100%';
            container.style.overflow = 'hidden';
            container.style.position = 'relative';
            img.style.position = 'absolute';
            img.parentNode.insertBefore(container, img);
            container.appendChild(img);
        }
    }
    
    // Create handles
    ['nw', 'ne', 'sw', 'se'].forEach(dir => {
        const handle = document.createElement('div');
        handle.className = `crop-handle crop-handle-${dir}`;
        handle.dataset.dir = dir;
        handle.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            startCropResize(e, dir);
        });
        el.appendChild(handle);
    });
    
    // Create dimmers
    ['top', 'bottom', 'left', 'right'].forEach(dir => {
        const dimmer = document.createElement('div');
        dimmer.className = `crop-dimmer crop-dimmer-${dir}`;
        el.appendChild(dimmer);
    });
    
    // Create border
    const border = document.createElement('div');
    border.className = 'crop-border';
    el.appendChild(border);
    
    updateCroppingUI();
};

function updateCroppingUI() {
    const id = state.croppingItemId;
    const item = workbenchItems.get(id);
    if (!item) return;
    
    const el = item.el;
    const crop = item.crop;
    const originalWidth = item.originalWidth;
    const originalHeight = item.originalHeight;
    
    // Show full image
    el.style.width = `${originalWidth}px`;
    el.style.height = `${originalHeight}px`;
    el.style.left = `${item.originalLeft}px`;
    el.style.top = `${item.originalTop}px`;
    el.style.overflow = 'visible';
    el.style.border = 'none'; // Remove normal border
    
    const container = el.querySelector('.crop-container');
    if (container) {
        container.style.overflow = 'visible';
        container.style.width = '100%';
        container.style.height = '100%';
    }
    
    const img = el.querySelector('img');
    if (img) {
        img.style.left = '0px';
        img.style.top = '0px';
        img.style.width = '100%';
        img.style.height = '100%';
    }
    
    // Update dimmers
    const dimmerTop = el.querySelector('.crop-dimmer-top');
    const dimmerBottom = el.querySelector('.crop-dimmer-bottom');
    const dimmerLeft = el.querySelector('.crop-dimmer-left');
    const dimmerRight = el.querySelector('.crop-dimmer-right');
    
    dimmerTop.style.height = `${crop.top}px`;
    dimmerBottom.style.height = `${crop.bottom}px`;
    
    dimmerLeft.style.top = `${crop.top}px`;
    dimmerLeft.style.height = `${originalHeight - crop.top - crop.bottom}px`;
    dimmerLeft.style.width = `${crop.left}px`;
    
    dimmerRight.style.top = `${crop.top}px`;
    dimmerRight.style.height = `${originalHeight - crop.top - crop.bottom}px`;
    dimmerRight.style.width = `${crop.right}px`;
    
    const border = el.querySelector('.crop-border');
    border.style.top = `${crop.top}px`;
    border.style.left = `${crop.left}px`;
    border.style.width = `${originalWidth - crop.left - crop.right}px`;
    border.style.height = `${originalHeight - crop.top - crop.bottom}px`;
    
    // Update handles position
    const nw = el.querySelector('.crop-handle-nw');
    const ne = el.querySelector('.crop-handle-ne');
    const sw = el.querySelector('.crop-handle-sw');
    const se = el.querySelector('.crop-handle-se');
    
    nw.style.top = `${crop.top - 4}px`; nw.style.left = `${crop.left - 4}px`;
    ne.style.top = `${crop.top - 4}px`; ne.style.right = `${crop.right - 4}px`;
    sw.style.bottom = `${crop.bottom - 4}px`; sw.style.left = `${crop.left - 4}px`;
    se.style.bottom = `${crop.bottom - 4}px`; se.style.right = `${crop.right - 4}px`;
}

window.exitCroppingMode = async function(commit = true) {
    if (!state.isCropping) return;
    
    const id = state.croppingItemId;
    const item = workbenchItems.get(id);
    if (!item) {
        state.isCropping = false;
        state.croppingItemId = null;
        return;
    }
    
    const el = item.el;
    el.classList.remove('cropping');
    el.style.border = ''; // Restore normal border
    
    // Remove crop UI elements
    el.querySelectorAll('.crop-handle, .crop-dimmer, .crop-border').forEach(child => child.remove());
    
    if (commit) {
        const { top, left, right, bottom } = item.crop;
        
        // If no actual crop was made, just exit
        if (top === 0 && left === 0 && right === 0 && bottom === 0) {
            state.isCropping = false;
            state.croppingItemId = null;
            return;
        }

        const img = el.querySelector('img');
        if (img) {
            try {
                // 1. Calculate ratios
                const ratioX = img.naturalWidth / item.originalWidth;
                const ratioY = img.naturalHeight / item.originalHeight;
                
                const naturalLeft = left * ratioX;
                const naturalTop = top * ratioY;
                const naturalWidth = (item.originalWidth - left - right) * ratioX;
                const naturalHeight = (item.originalHeight - top - bottom) * ratioY;
                
                // 2. Create canvas for physical crop
                const canvas = document.createElement('canvas');
                canvas.width = naturalWidth;
                canvas.height = naturalHeight;
                const ctx = canvas.getContext('2d');
                
                // 3. Draw the cropped portion
                ctx.drawImage(img, naturalLeft, naturalTop, naturalWidth, naturalHeight, 0, 0, naturalWidth, naturalHeight);
                
                // 4. Generate new data URL
                const newDataUrl = canvas.toDataURL('image/png');
                
                // 5. Genealogy Evolution Logic
                const visibleWidth = item.originalWidth - left - right;
                const visibleHeight = item.originalHeight - top - bottom;
                const newLeft = item.originalLeft + left;
                const newTop = item.originalTop + top;

                // Hide the original item
                el.style.display = 'none';
                item.hidden = true; // Mark as hidden in state

                // Create new item as a "child" (evolution)
                const newMetadata = {
                    left: `${newLeft}px`,
                    top: `${newTop}px`,
                    initialWidth: visibleWidth,
                    initialHeight: visibleHeight,
                    parentId: id, // Link to original
                    type: 'crop',
                    genealogy: {
                        parents: [id],
                        action: 'crop',
                        prompt: item.genealogy?.prompt || '裁切自原图'
                    }
                };

                // Add the new cropped image to workbench
                // We pass null as file because we have the dataUrl in metadata
                // addImageToWorkbench will handle the dataUrl from metadata
                await window.addImageToWorkbench(newDataUrl, '裁切图', newMetadata);
                
                // 6. Cleanup original item's crop state (it's hidden now)
                delete item.originalWidth;
                delete item.originalHeight;
                delete item.crop;
                delete item.originalLeft;
                delete item.originalTop;
                
                // Update genealogy lines
                if (typeof window.drawGenealogyConnections === 'function') {
                    window.drawGenealogyConnections();
                }
                
            } catch (e) {
                console.error('Physical crop failed, falling back to visual crop:', e);
                // Fallback to visual crop if canvas fails (e.g. CORS)
                applyVisualCrop(item, el);
            }
        }
    } else {
        // Revert to visual state before entering crop mode
        applyVisualCrop(item, el);
    }
    
    state.isCropping = false;
    state.croppingItemId = null;
    
    if (window.historyManager) window.historyManager.pushState();
};

function applyVisualCrop(item, el) {
    const { top, left, right, bottom } = item.crop || { top: 0, left: 0, right: 0, bottom: 0 };
    const visibleWidth = (item.originalWidth || parseFloat(el.style.width)) - left - right;
    const visibleHeight = (item.originalHeight || parseFloat(el.style.height)) - top - bottom;
    
    el.style.width = `${visibleWidth}px`;
    el.style.height = `${visibleHeight}px`;
    el.style.left = `${(item.originalLeft || parseFloat(el.style.left)) + left}px`;
    el.style.top = `${(item.originalTop || parseFloat(el.style.top)) + top}px`;
    el.style.overflow = 'visible';
    
    const container = el.querySelector('.crop-container');
    if (container) {
        container.style.overflow = 'hidden';
    }
    
    const img = el.querySelector('img');
    if (img) {
        img.style.width = `${item.originalWidth || visibleWidth}px`;
        img.style.height = `${item.originalHeight || visibleHeight}px`;
        img.style.left = `${-left}px`;
        img.style.top = `${-top}px`;
    }
}

function startCropResize(e, dir) {
    const id = state.croppingItemId;
    const item = workbenchItems.get(id);
    if (!item) return;
    
    const startX = e.clientX;
    const startY = e.clientY;
    const startCrop = { ...item.crop };
    const originalWidth = item.originalWidth;
    const originalHeight = item.originalHeight;
    const SNAP_DIST = 15; // Snap to original edges
    
    function onMouseMove(moveEvent) {
        const dx = (moveEvent.clientX - startX) / state.workbenchZoom;
        const dy = (moveEvent.clientY - startY) / state.workbenchZoom;
        
        const newCrop = { ...startCrop };
        
        if (dir.includes('n')) newCrop.top = Math.max(0, Math.min(originalHeight - newCrop.bottom - 20, startCrop.top + dy));
        if (dir.includes('s')) newCrop.bottom = Math.max(0, Math.min(originalHeight - newCrop.top - 20, startCrop.bottom - dy));
        if (dir.includes('w')) newCrop.left = Math.max(0, Math.min(originalWidth - newCrop.right - 20, startCrop.left + dx));
        if (dir.includes('e')) newCrop.right = Math.max(0, Math.min(originalWidth - newCrop.left - 20, startCrop.right - dx));
        
        // Snap to original edges
        if (newCrop.top < SNAP_DIST) newCrop.top = 0;
        if (newCrop.bottom < SNAP_DIST) newCrop.bottom = 0;
        if (newCrop.left < SNAP_DIST) newCrop.left = 0;
        if (newCrop.right < SNAP_DIST) newCrop.right = 0;
        
        item.crop = newCrop;
        updateCroppingUI();
    }
    
    function onMouseUp() {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    }
    
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
}

function handleRotateStart(e, wrapper) {
    if (state.isSpacePressed) return;
    
    const id = wrapper.dataset.itemId;
    const item = workbenchItems.get(id);
    if (!item) return;

    const rect = wrapper.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    const startAngle = Math.atan2(e.clientY - centerY, e.clientX - centerX) * (180 / Math.PI);
    
    // Get current rotation
    const transform = wrapper.style.transform;
    let currentRotation = 0;
    const rotateMatch = transform.match(/rotate\(([-\d.]+)deg\)/);
    if (rotateMatch) {
        currentRotation = parseFloat(rotateMatch[1]);
    }

    let rotation = currentRotation;

    function onMouseMove(moveEvent) {
        const angle = Math.atan2(moveEvent.clientY - centerY, moveEvent.clientX - centerX) * (180 / Math.PI);
        rotation = currentRotation + (angle - startAngle);
        
        // Snap to 45 degrees
        if (moveEvent.shiftKey) {
            rotation = Math.round(rotation / 45) * 45;
        }
        
        wrapper.style.transform = `rotate(${rotation}deg)`;
        
        if (item.scene && item.scene.transform) {
            item.scene.transform.rotation = rotation;
        }
        
        if (window.updateParentBBox) {
            window.updateParentBBox(id);
        }
    }

    function onMouseUp() {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);

        // --- Sync Rotate to MVR Runtime ---
        if (window.mvrRuntime) {
            const workspace = window.mvrRuntime.getCurrentWorkspace();
            if (workspace && workspace.currentState.assetRegistry.get(id)) {
                workspace.dispatcher.dispatch({
                    type: 'UPDATE_TRANSFORM',
                    payload: {
                        uid: id,
                        transform: {
                            rotation: rotation
                        }
                    }
                });
            }
        }

        if (window.historyManager) window.historyManager.pushState();
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
}

window.handleRotateStart = handleRotateStart;

function updateParentBBox(childId) {
    // V3: Disabled reverse mapping BBOX preview
}

window.updateParentBBox = updateParentBBox;
