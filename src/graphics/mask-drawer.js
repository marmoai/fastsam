import { state } from "../core/state.js";
import { getProxiedUrl } from "../core/utils.js";

// DOM Elements
let maskEditorModal, maskCanvasContainer, closeMaskEditorBtn, cancelMaskBtn, applyMaskBtn;
let updateImagePreviewCallback, updateSendBtnStateCallback;

export function initMaskDrawer(elements, callbacks) {
    maskEditorModal = elements.maskEditorModal;
    maskCanvasContainer = elements.maskCanvasContainer;
    closeMaskEditorBtn = elements.closeMaskEditorBtn;
    cancelMaskBtn = elements.cancelMaskBtn;
    applyMaskBtn = elements.applyMaskBtn;
    
    updateImagePreviewCallback = callbacks.updateImagePreview;
    updateSendBtnStateCallback = callbacks.updateSendBtnState;

    // Event Listeners
    if (maskCanvasContainer) {
        maskCanvasContainer.addEventListener('mousedown', handleLassoStart);
        maskCanvasContainer.addEventListener('touchstart', handleLassoStart, { passive: false });
    }
    if (closeMaskEditorBtn) closeMaskEditorBtn.addEventListener('click', closeMaskEditor);
    if (cancelMaskBtn) cancelMaskBtn.addEventListener('click', closeMaskEditor);
    if (applyMaskBtn) applyMaskBtn.addEventListener('click', applyMask);
    
    // Tool selection
    const toolRadios = document.querySelectorAll('input[name="maskTool"]');
    toolRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.maskEditor.toolMode = e.target.value;
            const hint = document.getElementById('maskToolHint');
            if (hint) {
                hint.textContent = e.target.value === 'lasso' ? '按住鼠标左键沿物体边缘拖拽完成套索' : '按住鼠标左键拖拽绘制矩形选区';
            }
        });
    });
}

export function openMaskEditor(file) {
    state.maskEditor.activeFile = file;
    state.maskEditor.toolMode = 'lasso';
    const lassoRadio = document.querySelector('input[name="maskTool"][value="lasso"]');
    if (lassoRadio) lassoRadio.checked = true;
    const hint = document.getElementById('maskToolHint');
    if (hint) hint.textContent = '按住鼠标左键沿物体边缘拖拽完成套索';
    
    const img = new Image();
    img.crossOrigin = "anonymous";
    let objectURL = null;
    
    if (file instanceof File || file instanceof Blob) {
        objectURL = URL.createObjectURL(file);
        img.src = objectURL;
    } else {
        img.src = getProxiedUrl(file);
    }

    img.onload = () => {
        if (objectURL) URL.revokeObjectURL(objectURL);
        requestAnimationFrame(() => requestAnimationFrame(() => {
            const editorBody = maskEditorModal.querySelector('.mask-editor-body');
            const containerWidth = editorBody.clientWidth - 40, containerHeight = editorBody.clientHeight - 40;
            const scale = Math.min(1, containerWidth / img.naturalWidth, containerHeight / img.naturalHeight);
            const width = img.naturalWidth * scale, height = img.naturalHeight * scale;
            state.maskEditor.scale = scale; 
            state.maskEditor.points = []; 
            state.maskEditor.hasSelection = false;
            maskCanvasContainer.innerHTML = ''; 
            maskCanvasContainer.style.width = `${width}px`; 
            maskCanvasContainer.style.height = `${height}px`;
            state.maskEditor.imageCanvas = document.createElement('canvas');
            state.maskEditor.drawingCanvas = document.createElement('canvas');
            [state.maskEditor.imageCanvas, state.maskEditor.drawingCanvas].forEach((canvas, index) => {
                canvas.width = width; canvas.height = height;
                canvas.style.cssText = `position: absolute; top: 0; left: 0; z-index: ${index + 1};`;
                maskCanvasContainer.appendChild(canvas);
            });
            state.maskEditor.imageCanvas.getContext('2d').drawImage(img, 0, 0, width, height);
            state.maskEditor.ctx = state.maskEditor.drawingCanvas.getContext('2d');
            if (state.maskDataUrl) {
                const maskImg = new Image();
                maskImg.onload = () => { 
                    state.maskEditor.ctx.drawImage(maskImg, 0, 0, width, height); 
                    state.maskEditor.hasSelection = true; 
                };
                maskImg.src = state.maskDataUrl;
            }
        }));
    };
    img.onerror = () => { if (objectURL) URL.revokeObjectURL(objectURL); alert("无法加载图片进行编辑。"); closeMaskEditor(); }
    maskEditorModal.style.display = 'flex';
}

export function closeMaskEditor(clearIntent = true) {
    maskEditorModal.style.display = 'none'; 
    maskCanvasContainer.innerHTML = ''; 
    detachGlobalLassoListeners();
    if (clearIntent !== false) {
        state.currentIntentLock = null; 
    }
    state.maskEditor = { 
        isDrawing: false, imageCanvas: null, drawingCanvas: null, ctx: null, 
        scale: 1, activeFile: null, points: [], hasSelection: false, 
        globalMoveListener: null, globalUpListener: null, 
        toolMode: state.maskEditor.toolMode, boxStart: null 
    };
}

function getMaskCanvasCoords(e) {
    if (!state.maskEditor.drawingCanvas) return null;
    const rect = state.maskEditor.drawingCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
    const y = Math.min(Math.max(clientY - rect.top, 0), rect.height);
    return { x, y };
}

function handleLassoStart(e) {
    if (e.touches) e.preventDefault();
    if ((e.button !== 0 && !e.touches) || !state.maskEditor.drawingCanvas || !state.maskEditor.ctx) return;
    const point = getMaskCanvasCoords(e);
    if (!point) return;
    state.maskEditor.isDrawing = true; 
    state.maskEditor.points = [point]; 
    state.maskEditor.hasSelection = false;
    if (state.maskEditor.toolMode === 'box') {
        state.maskEditor.boxStart = point;
    }
    detachGlobalLassoListeners(); 
    attachGlobalLassoListeners();
    const ctx = state.maskEditor.ctx;
    ctx.clearRect(0, 0, state.maskEditor.drawingCanvas.width, state.maskEditor.drawingCanvas.height);
    if (state.maskEditor.toolMode === 'lasso') {
        ctx.beginPath(); ctx.moveTo(point.x, point.y);
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)'; ctx.lineWidth = 2;
    }
}

function handleLassoMove(e) {
    if (e.touches) e.preventDefault();
    if (!state.maskEditor.isDrawing || !state.maskEditor.ctx) return;
    const point = getMaskCanvasCoords(e);
    if (!point) return;
    state.maskEditor.points.push(point);
    const ctx = state.maskEditor.ctx;
    if (state.maskEditor.toolMode === 'lasso') {
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
    } else if (state.maskEditor.toolMode === 'box') {
        ctx.clearRect(0, 0, state.maskEditor.drawingCanvas.width, state.maskEditor.drawingCanvas.height);
        const startX = state.maskEditor.boxStart.x;
        const startY = state.maskEditor.boxStart.y;
        const width = point.x - startX;
        const height = point.y - startY;
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, width, height);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.35)';
        ctx.fillRect(startX, startY, width, height);
    }
}

function handleLassoEnd() {
    if (!state.maskEditor.isDrawing) return;
    state.maskEditor.isDrawing = false; 
    detachGlobalLassoListeners(); 
    finalizeLassoPath();
}

function finalizeLassoPath() {
    const { drawingCanvas: canvas, ctx, points, toolMode, boxStart } = state.maskEditor;
    if (!canvas || !ctx) return;
    if (toolMode === 'lasso') {
        if (!points || points.length < 3) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            state.maskEditor.points = []; 
            state.maskEditor.hasSelection = false;
            alert('请沿着物体边缘勾勒一个闭合的选区。');
            return;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 0, 0, 0.35)';
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.fill(); ctx.stroke();
        state.maskEditor.hasSelection = true;
    } else if (toolMode === 'box') {
        if (!points || points.length < 2) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            state.maskEditor.points = []; 
            state.maskEditor.hasSelection = false;
            alert('请拖拽绘制一个矩形选区。');
            return;
        }
        const endPoint = points[points.length - 1];
        const startX = boxStart.x;
        const startY = boxStart.y;
        const width = endPoint.x - startX;
        const height = endPoint.y - startY;
        
        if (Math.abs(width) < 5 || Math.abs(height) < 5) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            state.maskEditor.points = []; 
            state.maskEditor.hasSelection = false;
            alert('选区太小，请重新框选。');
            return;
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.35)';
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.fillRect(startX, startY, width, height);
        ctx.strokeRect(startX, startY, width, height);
        state.maskEditor.hasSelection = true;
    }
}

function attachGlobalLassoListeners() {
    if (!state.maskEditor.globalMoveListener) {
        state.maskEditor.globalMoveListener = (event) => handleLassoMove(event);
        window.addEventListener('mousemove', state.maskEditor.globalMoveListener);
        window.addEventListener('touchmove', state.maskEditor.globalMoveListener, { passive: false });
    }
    if (!state.maskEditor.globalUpListener) {
        state.maskEditor.globalUpListener = () => handleLassoEnd();
        window.addEventListener('mouseup', state.maskEditor.globalUpListener);
        window.addEventListener('touchend', state.maskEditor.globalUpListener);
    }
}

function detachGlobalLassoListeners() {
    if (state.maskEditor.globalMoveListener) {
        window.removeEventListener('mousemove', state.maskEditor.globalMoveListener);
        window.removeEventListener('touchmove', state.maskEditor.globalMoveListener);
        state.maskEditor.globalMoveListener = null;
    }
    if (state.maskEditor.globalUpListener) {
        window.removeEventListener('mouseup', state.maskEditor.globalUpListener);
        window.removeEventListener('touchend', state.maskEditor.globalUpListener);
        state.maskEditor.globalUpListener = null;
    }
}

export function applyMask() {
    if (!state.maskEditor.hasSelection || !state.maskEditor.activeFile) {
        alert('请先使用套索工具完成一个选区。'); return;
    }
    const originalImage = new Image();
    const objectURL = URL.createObjectURL(state.maskEditor.activeFile);
    originalImage.src = objectURL;
    originalImage.onload = () => {
        URL.revokeObjectURL(objectURL);
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = originalImage.naturalWidth; 
        tempCanvas.height = originalImage.naturalHeight;
        const tempCtx = tempCanvas.getContext('2d'), scale = state.maskEditor.scale;
        tempCtx.beginPath();
        if (state.maskEditor.points.length > 0) {
            tempCtx.moveTo(state.maskEditor.points[0].x / scale, state.maskEditor.points[0].y / scale);
            for (let i = 1; i < state.maskEditor.points.length; i++) {
                tempCtx.lineTo(state.maskEditor.points[i].x / scale, state.maskEditor.points[i].y / scale);
            }
        }
        tempCtx.closePath();
        tempCtx.fillStyle = 'rgba(255, 255, 255, 1)';
        tempCtx.fill();
        
        // --- 核心修复点 ---
        state.maskDataUrl = tempCanvas.toDataURL('image/png'); // 保存生成的蒙版数据
        state.mainImageFile = state.maskEditor.activeFile;         // 【关键】确保该图被设为对话的主图
        state.pendingBaseImageShare = true;                  // 标记需要在下一条消息中发送该图

        if (updateImagePreviewCallback) updateImagePreviewCallback(); // 立即更新输入框上方的图片预览
        if (updateSendBtnStateCallback) updateSendBtnStateCallback(); // 更新发送按钮的可用状态
        closeMaskEditor(false);
    };
    originalImage.onerror = () => { 
        URL.revokeObjectURL(objectURL); 
        alert('无法加载图片以应用蒙版，请重试。'); 
        closeMaskEditor(); 
    };
}
