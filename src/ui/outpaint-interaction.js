import { state } from '../core/state.js';
import { RATIO_MAP } from '../core/config.js';
import { fileToDataURL, dataURLToFile } from '../core/utils.js';
import { addMessage } from './chat-panel.js';
import { editOrQueryImageWithGemini } from '../ai-services/skills-engine.js';
import { addImageToWorkbench } from './workbench-core.js';
import { selectWorkbenchItem } from './workbench/items.js';

export let outpaintState = {
    isDragging: false,
    itemId: null,
    direction: null,
    startX: 0,
    startY: 0,
    itemRect: null,
    ghostEl: null,
    delta: 0
};

let outpaintAnimationFrameId = null;
let latestOutpaintEvent = null;

function processOutpaintMove() {
    if (!latestOutpaintEvent || !outpaintState.isDragging) return;
    const e = latestOutpaintEvent;

    const zoom = state.workbenchZoom;
    const dx = (e.clientX - outpaintState.startX) / zoom;
    const dy = (e.clientY - outpaintState.startY) / zoom;
    const { direction, itemRect, ghostEl, itemId } = outpaintState;

    const item = state.workbenchItems.get(itemId);
    const originalImg = item.el.querySelector('img');
    const naturalW = originalImg.naturalWidth;
    const naturalH = originalImg.naturalHeight;
    
    const uiScale = naturalW / itemRect.width;

    let newLeft = itemRect.left;
    let newTop = itemRect.top;
    let newWidth = itemRect.width;
    let newHeight = itemRect.height;
    let rawUiDelta = 0;

    if (direction === 'right') rawUiDelta = Math.max(0, dx);
    else if (direction === 'bottom') rawUiDelta = Math.max(0, dy);
    else if (direction === 'left') rawUiDelta = Math.max(0, -dx);
    else if (direction === 'top') rawUiDelta = Math.max(0, -dy);

    const SUPPORTED_RATIOS = [
        { name: '1:1', val: 1 }, { name: '4:3', val: 4/3 }, { name: '3:4', val: 3/4 },
        { name: '16:9', val: 16/9 }, { name: '9:16', val: 9/16 }
    ];

    let snappedUiDelta = rawUiDelta;
    let activeRatioName = null;
    const SNAP_THRESHOLD_PX = 30;

    if (direction === 'left' || direction === 'right') {
        const currentTotalWidth = naturalW + (rawUiDelta * uiScale);
        const currentRatio = currentTotalWidth / naturalH;
        let bestMatch = null;
        for (const ratio of SUPPORTED_RATIOS) {
            if (ratio.val < (naturalW / naturalH) - 0.05) continue; 
            const targetNaturalW = naturalH * ratio.val;
            const targetUiDelta = (targetNaturalW - naturalW) / uiScale;
            if (Math.abs(targetUiDelta - rawUiDelta) < SNAP_THRESHOLD_PX) {
                bestMatch = { ratio, targetUiDelta }; break;
            }
        }
        if (bestMatch) {
            snappedUiDelta = bestMatch.targetUiDelta; activeRatioName = bestMatch.ratio.name;
        }
    } else { 
        const currentTotalHeight = naturalH + (rawUiDelta * uiScale);
        const currentRatio = naturalW / currentTotalHeight;
        let bestMatch = null;
        for (const ratio of SUPPORTED_RATIOS) {
            if (ratio.val > (naturalW / naturalH) + 0.05) continue;
            const targetNaturalH = naturalW / ratio.val;
            const targetUiDelta = (targetNaturalH - naturalH) / uiScale;
            if (Math.abs(targetUiDelta - rawUiDelta) < SNAP_THRESHOLD_PX) {
                bestMatch = { ratio, targetUiDelta }; break;
            }
        }
        if (bestMatch) {
            snappedUiDelta = bestMatch.targetUiDelta; activeRatioName = bestMatch.ratio.name;
        }
    }

    outpaintState.delta = snappedUiDelta;
    outpaintState.snappedRatio = activeRatioName;

    if (direction === 'right') newWidth = itemRect.width + snappedUiDelta;
    else if (direction === 'bottom') newHeight = itemRect.height + snappedUiDelta;
    else if (direction === 'left') {
        newLeft = itemRect.left - snappedUiDelta; newWidth = itemRect.width + snappedUiDelta;
    } else if (direction === 'top') {
        newTop = itemRect.top - snappedUiDelta; newHeight = itemRect.height + snappedUiDelta;
    }

    const inverseScale = 1 / state.workbenchZoom;
    const visualBorderWidth = 2; // CSS中边框视觉宽度
    const visualFontSize = 12;   // CSS中标签视觉字号
    const scaledBorderWidth = `${visualBorderWidth * inverseScale}px`;

    // 1. 更新虚影 UI 的尺寸
    ghostEl.style.left = `${newLeft}px`;
    ghostEl.style.top = `${newTop}px`;
    ghostEl.style.width = `${newWidth}px`;
    ghostEl.style.height = `${newHeight}px`;
    
    // 2. 更新虚影 UI 的样式（包括缩放补偿）
    if (activeRatioName) {
        ghostEl.style.borderColor = '#4CAF50';
        ghostEl.style.background = 'rgba(76, 175, 80, 0.1)';
        if (navigator.vibrate && outpaintState.lastSnapped !== activeRatioName) {
            navigator.vibrate(10);
            outpaintState.lastSnapped = activeRatioName;
        }
    } else {
        ghostEl.style.borderColor = '#a78bfa';
        ghostEl.style.background = 'rgba(139, 92, 246, 0.15)';
        outpaintState.lastSnapped = null;
    }
    // 统一应用缩放后的边框宽度
    ghostEl.style.borderWidth = scaledBorderWidth;
    
    // 3. 动态创建/更新 Label (替代CSS伪元素，从而可以控制字体大小)
    const labelHTML = activeRatioName 
        ? `<span style="font-size: ${visualFontSize * inverseScale}px;"><i class="fas fa-magnet"></i> 吸附至 ${activeRatioName}</span>` 
        : '';
    ghostEl.innerHTML = labelHTML;
    
    outpaintAnimationFrameId = null;
}

export function handleOutpaintDragMove(e) {
    if (!outpaintState.isDragging) return;
    e.preventDefault();
    
    latestOutpaintEvent = e;
    if (!outpaintAnimationFrameId) {
        outpaintAnimationFrameId = requestAnimationFrame(processOutpaintMove);
    }
}

export async function handleOutpaintDragEnd(e) {
    if (outpaintAnimationFrameId) {
        cancelAnimationFrame(outpaintAnimationFrameId);
        outpaintAnimationFrameId = null;
    }
    // 1. 标准的拖拽状态清理
    if (!outpaintState.isDragging) return;
    const { itemId, direction, delta, ghostEl, itemRect, snappedRatio } = outpaintState;
    document.removeEventListener('mousemove', handleOutpaintDragMove);
    document.removeEventListener('mouseup', handleOutpaintDragEnd);
    outpaintState.isDragging = false;
    if (ghostEl) ghostEl.remove();
    if (delta < 10) return;
    const item = state.workbenchItems.get(itemId);
    if (!item) return;

    // --- 2. 精确预计算占位符的尺寸和位置 ---
    const originalImg = new Image();
    originalImg.src = await fileToDataURL(item.file || item.dataUrl);
    await new Promise(r => originalImg.onload = r);
    const naturalW = originalImg.naturalWidth;
    const naturalH = originalImg.naturalHeight;
    const uiScale = naturalW / itemRect.width;
    const rawPixelDelta = Math.round(delta * uiScale);
    let rawTargetW = naturalW;
    let rawTargetH = naturalH;
    if (direction === 'left' || direction === 'right') rawTargetW += rawPixelDelta;
    else rawTargetH += rawPixelDelta;
    
    let targetRatioName = snappedRatio;
    let targetRatioVal = 1;


    if (targetRatioName && RATIO_MAP[targetRatioName]) {
        targetRatioVal = RATIO_MAP[targetRatioName];
    } else {
        const currentRatio = rawTargetW / rawTargetH;
        let minDiff = Infinity;
        for (const [name, val] of Object.entries(RATIO_MAP)) {
            const diff = Math.abs(currentRatio - val);
            if (diff < minDiff) {
                minDiff = diff;
                targetRatioName = name;
                targetRatioVal = val;
            }
        }
    }
    
    let actualExpansionDelta = 0;
    if (direction === 'left' || direction === 'right') {
        const finalNaturalW = Math.round(naturalH * targetRatioVal);
        actualExpansionDelta = Math.max(0, finalNaturalW - naturalW);
    } else {
        const finalNaturalH = Math.round(naturalW / targetRatioVal);
        actualExpansionDelta = Math.max(0, finalNaturalH - naturalH);
    }

    const correctedUiDelta = actualExpansionDelta / uiScale;
    let newWorkbenchX = itemRect.left;
    let newWorkbenchY = itemRect.top;
    let newWorkbenchW = itemRect.width;
    let newWorkbenchH = itemRect.height;

    if (direction === 'left') { newWorkbenchX -= correctedUiDelta; newWorkbenchW += correctedUiDelta; }
    else if (direction === 'right') { newWorkbenchW += correctedUiDelta; }
    else if (direction === 'top') { newWorkbenchY -= correctedUiDelta; newWorkbenchH += correctedUiDelta; }
    else if (direction === 'bottom') { newWorkbenchH += correctedUiDelta; }

    // --- 4. 创建增强型Prompt ---
    let basePrompt = "";
    const negativePrompt = "Do not add black bars or letterboxing. The original image content must be perfectly preserved and visible on one side of the final image. Seamlessly blend the new content with the old.";

    switch(direction) {
        case 'top': basePrompt = `Extend this image upwards, filling the new space while keeping the original image at the bottom. The final image must have a ${targetRatioName} aspect ratio.`; break;
        case 'bottom': basePrompt = `Extend this image downwards, filling the new space while keeping the original image at the top. The final image must have a ${targetRatioName} aspect ratio.`; break;
        case 'left': basePrompt = `Extend this image to the left, filling the new space while keeping the original image on the right. The final image must have a ${targetRatioName} aspect ratio.`; break;
        case 'right': basePrompt = `Extend this image to the right, filling the new space while keeping the original image on the left. The final image must have a ${targetRatioName} aspect ratio.`; break;
    }
    const fullPrompt = `${basePrompt} ${negativePrompt}`;

    const executeOutpaint = async (customPrompt) => {
        // --- 3. 创建并显示占位符 ---
        const placeholder = document.createElement('div');
        placeholder.className = 'world-placeholder';
        placeholder.style.left = `${newWorkbenchX}px`;
        placeholder.style.top = `${newWorkbenchY}px`;
        placeholder.style.width = `${newWorkbenchW}px`;
        placeholder.style.height = `${newWorkbenchH}px`;
        
        const inverseScale = 1 / state.workbenchZoom;
        const visualBorderWidth = 2;
        const visualFontSize = 12;
        placeholder.style.border = `${visualBorderWidth * inverseScale}px dashed #ccc`;
        placeholder.innerHTML = '<span><i class="fas fa-magic"></i> AI 扩图中...</span>';
        const span = placeholder.querySelector('span');
        if (span) {
            span.style.fontSize = `${visualFontSize * inverseScale}px`;
        }
        const workbenchGrid = document.getElementById('workbenchGrid');
        workbenchGrid.appendChild(placeholder);

        // --- 添加聊天提示信息 ---
        const tempMsg = addMessage({ sender: 'bot', type: 'text', content: `🎨 正在向 **${direction}** 方向扩展至 **${targetRatioName}** 比例...` });

        const promptToUse = customPrompt || fullPrompt;

        try {
            // --- 5. 执行API调用 ---
            const result = await editOrQueryImageWithGemini(promptToUse, item.file || item.dataUrl, [], null, targetRatioName);

            // --- 6. 处理成功结果 (包含校准逻辑) ---
            if (result.success && result.imageData) {
                const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
                const newFile = await dataURLToFile(imgSrc, `outpaint-${direction}-${Date.now()}.png`);
                
                // *** 核心校准步骤 ***
                const newImg = new Image();
                newImg.src = imgSrc;
                await new Promise(r => newImg.onload = r);
                
                // 根据返回图片的真实尺寸和原图的UI缩放比例，计算出最终完美的UI尺寸
                const finalUiWidth = newImg.naturalWidth / uiScale;
                const finalUiHeight = newImg.naturalHeight / uiScale;
                // *** 校准结束 ***

                if (placeholder && placeholder.parentNode) placeholder.remove();
                
                // 使用“预测的位置”和“校准后的尺寸”添加新图，杜绝白边
                const newId = addImageToWorkbench(newFile, 'AI扩图', { 
                    x: newWorkbenchX, 
                    y: newWorkbenchY,
                    initialWidth: finalUiWidth,
                    initialHeight: finalUiHeight,
                    parentId: itemId, 
                    generationParams: { prompt: promptToUse, ratio: targetRatioName }
                });
                
                setTimeout(() => {
                    selectWorkbenchItem(newId);
                }, 100);

                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                if (window.addWorkbenchActionToChat) {
                    await window.addWorkbenchActionToChat(`向${direction}扩图至${targetRatioName}`, promptToUse, imgSrc, executeOutpaint);
                } else {
                    addMessage({ sender: 'bot', type: 'text', content: `✅ **扩图完成**！` });
                }

            } else {
                 throw new Error(result.text || "AI did not return a valid image.");
            }

        } catch (e) {
            // --- 7. 处理失败结果 ---
            console.error(e);
            const currentInverseScale = 1 / state.workbenchZoom;
            const errorFontSize = `${visualFontSize * currentInverseScale}px`;
            if (placeholder && placeholder.parentNode) {
                placeholder.innerHTML = `<span style="color:red; font-size: ${errorFontSize};">扩展失败</span>`;
                setTimeout(() => { if (placeholder.parentNode) placeholder.remove(); }, 2000);
            }
            if (tempMsg && tempMsg.parentNode) tempMsg.remove();
            addMessage({ sender: 'bot', type: 'text', content: `扩图失败: ${e.message}` });
        }
    };

    await executeOutpaint();
}

export function startOutpaintDrag(e, itemId, direction, itemEl) {
    outpaintState.isDragging = true;
    outpaintState.itemId = itemId;
    outpaintState.direction = direction;
    outpaintState.startX = e.clientX;
    outpaintState.startY = e.clientY;
    
    const rect = {
        left: parseFloat(itemEl.style.left),
        top: parseFloat(itemEl.style.top),
        width: parseFloat(itemEl.style.width),
        height: parseFloat(itemEl.style.height)
    };
    outpaintState.itemRect = rect;

    const ghost = document.createElement('div');
    ghost.className = 'outpaint-ghost';
    ghost.style.left = `${rect.left}px`;
    ghost.style.top = `${rect.top}px`;
    ghost.style.width = `${rect.width}px`;
    ghost.style.height = `${rect.height}px`;
    ghost.style.display = 'block';

    const inverseScale = 1 / state.workbenchZoom;
    const visualBorderWidth = 2;
    ghost.style.borderWidth = `${visualBorderWidth * inverseScale}px`;
    
    const workbenchGrid = document.getElementById('workbenchGrid');
    workbenchGrid.appendChild(ghost);
    outpaintState.ghostEl = ghost;

    document.addEventListener('mousemove', handleOutpaintDragMove);
    document.addEventListener('mouseup', handleOutpaintDragEnd);
}
