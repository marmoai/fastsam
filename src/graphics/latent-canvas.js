import { state } from "../core/state.js";
import { generateLatentImage } from "../ai-services/gemini-client.js";
import { dataURLToFile } from "../core/utils.js";

// Callbacks
let addImageToWorkbenchCallback, addMessageCallback;

export function initLatentCanvas(callbacks) {
    addImageToWorkbenchCallback = callbacks.addImageToWorkbench;
    addMessageCallback = callbacks.addMessage;
}

export function setupLatentCanvas() {
    const latentSketchCanvas = document.getElementById('latentSketchCanvas');
    const openLatentCanvasBtn = document.getElementById('openLatentCanvasBtn');
    const latentCanvasModal = document.getElementById('latentCanvasModal');
    const latentResultImg = document.getElementById('latentResultImg');
    
    if (!latentSketchCanvas) return;
    
    state.latentState.ctx = latentSketchCanvas.getContext('2d', { willReadFrequently: true });
    
    // Init white bg
    state.latentState.ctx.fillStyle = state.latentState.backgroundColor;
    state.latentState.ctx.fillRect(0, 0, latentSketchCanvas.width, latentSketchCanvas.height);
    saveLatentHistory(latentSketchCanvas);

    // Event Listeners
    latentSketchCanvas.addEventListener('mousedown', (e) => startLatentDraw(e, latentSketchCanvas));
    latentSketchCanvas.addEventListener('mousemove', (e) => drawLatent(e, latentSketchCanvas));
    window.addEventListener('mouseup', endLatentDraw);
    
    latentSketchCanvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        startLatentDraw({ clientX: touch.clientX, clientY: touch.clientY }, latentSketchCanvas);
    }, { passive: false });
    latentSketchCanvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        drawLatent({ clientX: touch.clientX, clientY: touch.clientY }, latentSketchCanvas);
    }, { passive: false });
    latentSketchCanvas.addEventListener('touchend', endLatentDraw);

    // Tools
    const toolPencil = document.getElementById('toolPencil');
    const toolEraser = document.getElementById('toolEraser');
    const toolClear = document.getElementById('toolClear');
    const latentUndo = document.getElementById('latentUndo');
    const latentRandomizeSeed = document.getElementById('latentRandomizeSeed'); 
    const latentCloseBtn = document.getElementById('latentCloseBtn');
    const latentGenerateBtn = document.getElementById('latentGenerateBtn');
    const latentSendToWorkbench = document.getElementById('latentSendToWorkbench');

    if (toolPencil) toolPencil.onclick = (e) => setLatentTool('pencil', e.currentTarget);
    if (toolEraser) toolEraser.onclick = (e) => setLatentTool('eraser', e.currentTarget);
    if (toolClear) toolClear.onclick = () => clearLatentCanvas(latentSketchCanvas);
    if (latentUndo) latentUndo.onclick = undoLatentCanvas;
    if (latentRandomizeSeed) {
        latentRandomizeSeed.onclick = () => {
            // 1. 生成新的随机数
            state.latentState.seed = Math.floor(Math.random() * 1000000000);
            
            // 2. 给按钮一点视觉反馈（旋转一下）
            const icon = latentRandomizeSeed.querySelector('i');
            if(icon) {
                icon.style.transition = 'transform 0.5s';
                icon.style.transform = `rotate(${Math.random() * 360}deg)`;
            }

            // 3. 强制触发生成 (force=true)
            triggerLatentGeneration(true, latentSketchCanvas);
        };
    }
    
    if (openLatentCanvasBtn) openLatentCanvasBtn.onclick = () => {
        if (latentCanvasModal) latentCanvasModal.style.display = 'flex';
    };
    if (latentCloseBtn) latentCloseBtn.onclick = () => {
        if (latentCanvasModal) latentCanvasModal.style.display = 'none';
    };
    if (latentGenerateBtn) latentGenerateBtn.onclick = () => triggerLatentGeneration(true, latentSketchCanvas);
    
    if (latentSendToWorkbench) latentSendToWorkbench.onclick = async () => {
        if (latentResultImg && latentResultImg.src && !latentResultImg.src.includes('placeholder')) {
            const executeLatentSend = async (customPrompt) => {
                if (customPrompt) {
                    const latentPromptInput = document.getElementById('latentPromptInput');
                    if (latentPromptInput) latentPromptInput.value = customPrompt;
                    await triggerLatentGeneration(true, latentSketchCanvas);
                }
                
                if (latentResultImg && latentResultImg.src && !latentResultImg.src.includes('placeholder')) {
                    const file = await dataURLToFile(latentResultImg.src, `latent-art-${Date.now()}.png`);
                    if (addImageToWorkbenchCallback) addImageToWorkbenchCallback(file, '灵感绘图');
                    if (latentCanvasModal) latentCanvasModal.style.display = 'none';
                    
                    if (window.addWorkbenchActionToChat) {
                        const promptToUse = customPrompt || state.latentState.prompt || '手绘草图';
                        await window.addWorkbenchActionToChat('灵感绘图', promptToUse, latentResultImg.src, executeLatentSend);
                    } else if (addMessageCallback) {
                        addMessageCallback({ sender: 'bot', type: 'text', content: '✅ 灵感绘图已发送到工作台。' });
                    }
                }
            };
            
            await executeLatentSend();
        }
    };
}

function setLatentTool(tool, btn) {
    state.latentState.tool = tool;
    document.querySelectorAll('.brush-btn').forEach(b => b.classList.remove('active'));
    if(btn) btn.classList.add('active');
}

function getLatentCoords(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height)
    };
}

function startLatentDraw(e, canvas) {
    state.latentState.isDrawing = true;
    const coords = getLatentCoords(e, canvas);
    state.latentState.lastX = coords.x;
    state.latentState.lastY = coords.y;
    saveLatentHistory(canvas);
}

function drawLatent(e, canvas) {
    if (!state.latentState.isDrawing) return;
    const coords = getLatentCoords(e, canvas);
    const ctx = state.latentState.ctx;
    
    ctx.beginPath();
    ctx.moveTo(state.latentState.lastX, state.latentState.lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (state.latentState.tool === 'pencil') {
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
    } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 20;
    }

    ctx.stroke();
    state.latentState.lastX = coords.x;
    state.latentState.lastY = coords.y;
}

function endLatentDraw() {
    if (state.latentState.isDrawing) {
        state.latentState.isDrawing = false;
        const realtimeToggle = document.getElementById('realtimeToggle');
        const latentSketchCanvas = document.getElementById('latentSketchCanvas');
        if (realtimeToggle && realtimeToggle.checked) {
            clearTimeout(state.latentState.debounceTimer);
            state.latentState.debounceTimer = setTimeout(() => triggerLatentGeneration(false, latentSketchCanvas), 600);
        }
    }
}

function clearLatentCanvas(canvas) {
    saveLatentHistory(canvas);
    state.latentState.ctx.fillStyle = '#ffffff';
    state.latentState.ctx.fillRect(0, 0, canvas.width, canvas.height);
    state.latentState.seed = Math.floor(Math.random() * 1000000000);
    const realtimeToggle = document.getElementById('realtimeToggle');
    if (realtimeToggle && realtimeToggle.checked) triggerLatentGeneration(false, canvas);
}

function saveLatentHistory(canvas) {
    if (state.latentState.history.length > 10) state.latentState.history.shift();
    state.latentState.history.push(canvas.toDataURL());
}

function undoLatentCanvas() {
    if (state.latentState.history.length > 0) {
        const prev = state.latentState.history.pop();
        const img = new Image();
        img.src = prev;
        img.onload = () => {
            state.latentState.ctx.drawImage(img, 0, 0);
            const realtimeToggle = document.getElementById('realtimeToggle');
            const latentSketchCanvas = document.getElementById('latentSketchCanvas');
            if (realtimeToggle && realtimeToggle.checked) triggerLatentGeneration(false, latentSketchCanvas);
        };
    }
}

async function triggerLatentGeneration(force = false, canvas) {
    if (state.latentState.isGenerating && !force) return;
    
    const latentPromptInput = document.getElementById('latentPromptInput');
    const latentLoader = document.getElementById('latentLoader');
    const latentResultImg = document.getElementById('latentResultImg');

    const prompt = latentPromptInput ? latentPromptInput.value.trim() : '';
    if (!prompt && !force) return; // Wait for prompt if auto
    
    // Default prompt if empty
    const effectivePrompt = prompt || "A high quality realistic rendering of this sketch";

    state.latentState.isGenerating = true;
    if (latentLoader) latentLoader.style.display = 'flex';

    try {
        const base64Sketch = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        const imageData = await generateLatentImage(effectivePrompt, base64Sketch, state.latentState.seed);

        if (imageData && latentResultImg) {
            latentResultImg.src = `data:image/png;base64,${imageData}`;
        } else {
             console.log("No image data in response");
        }

    } catch (e) {
        console.error("Real-time gen error:", e);
    } finally {
        state.latentState.isGenerating = false;
        if (latentLoader) latentLoader.style.display = 'none';
    }
}