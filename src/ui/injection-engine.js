import { state } from '../core/state.js';
import { addMessage } from './chat-panel.js';
import { addImageToWorkbench } from './workbench/items.js';
import { editOrQueryImageWithGemini_Multiple } from '../ai-services/skills-engine.js';
import { dataURLToFile, fileToDataURL } from '../core/utils.js';
import { handleSend } from './chat-actions.js';

const { workbenchItems, selectedWorkbenchItems } = state;

let activeConnectors = [];

export function initInjectionEngine() {
    const injectionMenu = document.getElementById('injectionMenu');
    if (injectionMenu) {
        injectionMenu.addEventListener('click', (e) => {
            const option = e.target.closest('.injection-option');
            if (!option) return;
            
            const action = option.dataset.action;
            injectionMenu.classList.remove('active');
            
            if (action === 'cancel') return;
            
            handleInjection(action, state.injectionSourceId, state.collisionTargetId);
            
            // Reset state
            state.injectionSourceId = null;
            state.collisionTargetId = null;
        });
    }

    document.body.addEventListener('click', (e) => {
        // Click outside injection menu to close it
        const injectionMenu = document.getElementById('injectionMenu');
        if (injectionMenu && injectionMenu.classList.contains('active') && !injectionMenu.contains(e.target)) {
             injectionMenu.classList.remove('active');
             state.collisionTargetId = null;
             state.injectionSourceId = null;
             document.querySelectorAll('.collision-active').forEach(el => el.classList.remove('collision-active'));
        }
    });
}

export function getIntersectionArea(rect1, rect2) {
    const x_overlap = Math.max(0, Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left));
    const y_overlap = Math.max(0, Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top));
    return x_overlap * y_overlap;
}

export function checkProximity() {
    // 清除旧的连接器
    activeConnectors.forEach(el => el.remove());
    activeConnectors = [];

    // --- 方案A：仅在选中 1 或 2 张图片时显示融合图标 ---
    if (selectedWorkbenchItems.size === 0 || selectedWorkbenchItems.size > 2) return;

    const isImage = (item) => item.type !== 'text-note' && item.type !== 'shape' && item.type !== 'group-label';
    const processedPairs = new Set();

    selectedWorkbenchItems.forEach(selectedId => {
        const movedItem = workbenchItems.get(selectedId);
        if (!movedItem || movedItem.el.classList.contains('atmosphere-node')) return;
        if (!isImage(movedItem)) return;

        const r1 = {
            x: parseFloat(movedItem.el.style.left),
            y: parseFloat(movedItem.el.style.top),
            w: parseFloat(movedItem.el.style.width) || movedItem.el.offsetWidth,
            h: parseFloat(movedItem.el.style.height) || movedItem.el.offsetHeight
        };

        workbenchItems.forEach((otherItem, otherId) => {
            if (otherId === selectedId || otherItem.el.classList.contains('atmosphere-node')) return;

            // 仅允许图片之间触发融合
            if (!isImage(otherItem)) return;

            // 避免重复创建 A-B 和 B-A
            const pairKey = [selectedId, otherId].sort().join('-');
            if (processedPairs.has(pairKey)) return;

            const r2 = {
                x: parseFloat(otherItem.el.style.left),
                y: parseFloat(otherItem.el.style.top),
                w: parseFloat(otherItem.el.style.width) || otherItem.el.offsetWidth,
                h: parseFloat(otherItem.el.style.height) || otherItem.el.offsetHeight
            };

            // 1. 使用中心点距离来判断是否应该显示连接器
            const c1 = { x: r1.x + r1.w / 2, y: r1.y + r1.h / 2 };
            const c2 = { x: r2.x + r2.w / 2, y: r2.y + r2.h / 2 };
            const dist = Math.hypot(c2.x - c1.x, c2.y - c1.y);
            const threshold = (r1.w + r2.w) / 2 * 1.5;

            if (dist < threshold) {
                processedPairs.add(pairKey);

                // --- 2. 核心修改：计算“间隙”中点，而不是“中心”中点 ---
                
                // a. 确定哪张图在左/右，哪张在上/下
                const leftImg = (c1.x < c2.x) ? r1 : r2;
                const rightImg = (leftImg === r1) ? r2 : r1;
                const topImg = (c1.y < c2.y) ? r1 : r2;
                const bottomImg = (topImg === r1) ? r2 : r1;

                // b. 计算间隙的四个边界
                const gapLeftEdge = leftImg.x + leftImg.w;
                const gapRightEdge = rightImg.x;
                const gapTopEdge = topImg.y + topImg.h;
                const gapBottomEdge = bottomImg.y;
                
                // c. 计算间隙的中心点坐标
                const connectorX = (gapLeftEdge + gapRightEdge) / 2;
                const connectorY = (gapTopEdge + gapBottomEdge) / 2;

                // d. 使用新的、精确的坐标创建连接器
                createConnector(selectedId, otherId, {
                    x: connectorX,
                    y: connectorY
                });
            }
        });
    });
}

export function createConnector(id1, id2, pos) {
    const btn = document.createElement('div');
    btn.className = 'world-connector';
    btn.innerHTML = '<i class="fas fa-link"></i>';
    btn.title = '智能融合：生成中间的过渡场景';
    btn.style.left = `${pos.x}px`;
    btn.style.top = `${pos.y}px`;

    // 1. 禁用原有的、会引起冲突的CSS动画
    btn.style.animation = 'none';

    // 2. 设置初始状态：完全透明，缩放为0
    btn.style.opacity = '0';
    btn.style.transform = `translate(-50%, -50%) scale(0)`;
    
    // 3. 添加一个平滑过渡的效果
    btn.style.transition = 'transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.2s';

    btn.onclick = () => handleBridgeGeneration(id1, id2);

    document.getElementById('workbenchGrid').appendChild(btn);
    activeConnectors.push(btn);

    // 4. 使用 requestAnimationFrame 确保在下一帧应用最终样式，从而触发过渡动画
    requestAnimationFrame(() => {
        btn.style.opacity = '1';
        // 计算并应用最终的、正确的缩放比例
        const inverseScale = 1 / state.workbenchZoom;
        btn.style.transform = `translate(-50%, -50%) scale(${inverseScale})`;
    });
}

async function handleInjection(action, sourceId, targetId) {
    const sourceItem = workbenchItems.get(sourceId);
    const targetItem = workbenchItems.get(targetId);
    
    if (!sourceItem || !targetItem) return;

    // Visual feedback: remove highlighting
    document.querySelectorAll('.collision-active').forEach(el => el.classList.remove('collision-active'));

    let prompt = "";
    
    switch (action) {
        case 'style':
            prompt = `Apply the artistic style, color palette, and mood from the reference image (source) to the main image (target). The core subject matter and composition of the main image should remain unchanged, only its visual style should be transformed.`;
            break;
        case 'composition':
            prompt = `Analyze the main subject of the reference image (source) and seamlessly integrate it into the scene of the main image (target). The new subject should adopt the lighting, shadows, and perspective of the target scene. The original subject of the target image should be replaced.`;
            break;
        case 'composite': // 主体融合
            prompt = `Analyze the main subject from the reference image (source). Place this subject into the main image (target) in a natural and believable way. The subject should be realistically blended into the target's environment, matching its lighting, shadows, and perspective. Do not alter the background of the target image.`;
            break;
    }

    const userInput = document.getElementById('userInput');
    if (userInput) userInput.value = prompt; // Show prompt to user
    
    // 标记这些图片需要在聊天记录中显示
    window.mainImageFile = targetItem.file || targetItem.dataUrl;
    window.referenceImageFiles = [sourceItem.file || sourceItem.dataUrl];
    if (window.pendingReferenceImageShares) {
        window.pendingReferenceImageShares.add(sourceItem.file || sourceItem.dataUrl);
    }
    window.pendingBaseImageShare = true;
    
    // 为新生成的图片记录正确的“父母”ID，用于基因谱系
    window.nextGenerationParents = {
        parentId: targetId,
        styleParentId: sourceId,
        action: action // 记录注入类型
    };
    
    if (window.updateImagePreview) await window.updateImagePreview();
    await handleSend(); 
}

async function handleBridgeGeneration(id1, id2, customPrompt) {
    const item1 = workbenchItems.get(id1);
    const item2 = workbenchItems.get(id2);
    if (!item1 || !item2) return;

    // 隐藏连接器
    activeConnectors.forEach(el => el.remove());
    activeConnectors = [];

    // 计算两张图的整体边界
    const w1 = parseFloat(item1.el.style.width) || item1.el.offsetWidth;
    const h1 = parseFloat(item1.el.style.height) || item1.el.offsetHeight;
    const w2 = parseFloat(item2.el.style.width) || item2.el.offsetWidth;
    const h2 = parseFloat(item2.el.style.height) || item2.el.offsetHeight;
    
    const r1 = { x: parseFloat(item1.el.style.left), y: parseFloat(item1.el.style.top), w: w1, h: h1 };
    const r2 = { x: parseFloat(item2.el.style.left), y: parseFloat(item2.el.style.top), w: w2, h: h2 };
    
    const minX = Math.min(r1.x, r2.x);
    const maxX = Math.max(r1.x + r1.w, r2.x + r2.w);
    const minY = Math.min(r1.y, r2.y);
    const maxY = Math.max(r1.y + r1.h, r2.y + r2.h);

    const totalWidth = maxX - minX;
    const totalHeight = maxY - minY;

    // 判断是水平排列还是垂直排列
    const isHorizontal = totalWidth > totalHeight;
    
    let prompt = "";
    let placeholderX = 0;
    let placeholderY = 0;
    let placeholderW = 0;
    let placeholderH = 0;

    if (isHorizontal) {
        prompt = customPrompt || "Synthesize a completely new, single image that creatively combines the core subjects, characteristic elements, and artistic styles of the two scenes shown side-by-side in this image. This is NOT a stitch, a collage, or a side-by-side comparison. Instead, reimagine the content of both sides into one unified, harmonious composition. The new image must preserve the recognizable 'feature points' and original elements from both sources, blending them into a single cohesive scene with consistent lighting, perspective, and style. The goal is a creative synthesis where the two worlds become one, maintaining 50% of the essence from each source. The final result should be a professional 16:9 horizontal masterpiece.";
        placeholderX = minX;
        placeholderY = maxY + 20; // 放在下方
        placeholderW = totalWidth;
        placeholderH = totalWidth / (16/9);
    } else {
        prompt = customPrompt || "Synthesize a completely new, single image that creatively combines the core subjects, characteristic elements, and artistic styles of the two scenes shown top-and-bottom in this image. This is NOT a stitch, a collage, or a side-by-side comparison. Instead, reimagine the content of both sides into one unified, harmonious composition. The new image must preserve the recognizable 'feature points' and original elements from both sources, blending them into a single cohesive scene with consistent lighting, perspective, and style. The goal is a creative synthesis where the two worlds become one, maintaining 50% of the essence from each source. The final result should be a professional 9:16 vertical masterpiece.";
        placeholderX = maxX + 20; // 放在右侧
        placeholderY = minY;
        placeholderW = totalHeight * (9/16);
        placeholderH = totalHeight;
    }

    // 创建占位符
    const placeholder = document.createElement('div');
    placeholder.className = 'world-placeholder';
    placeholder.style.left = `${placeholderX}px`;
    placeholder.style.top = `${placeholderY}px`;
    placeholder.style.width = `${placeholderW}px`;
    placeholder.style.height = `${placeholderH}px`;
    
    const inverseScale = 1 / state.workbenchZoom;
    const visualBorderWidth = 1; 
    const visualFontSize = 12;   
    
    placeholder.style.border = `${visualBorderWidth * inverseScale}px dashed #ccc`;
    placeholder.innerHTML = '<span><i class="fas fa-magic"></i> 智能融合中...</span>';
    
    const span = placeholder.querySelector('span');
    if (span) {
        span.style.fontSize = `${visualFontSize * inverseScale}px`;
    }

    document.getElementById('workbenchGrid').appendChild(placeholder);

    try {
        const url1 = await fileToDataURL(item1.file || item1.dataUrl);
        const url2 = await fileToDataURL(item2.file || item2.dataUrl);
        
        const img1 = new Image();
        img1.crossOrigin = "anonymous";
        img1.src = url1;
        await new Promise(r => img1.onload = r);

        const img2 = new Image();
        img2.crossOrigin = "anonymous";
        img2.src = url2;
        await new Promise(r => img2.onload = r);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        if (isHorizontal) {
            const targetHeight = Math.max(img1.height, img2.height);
            const w1 = img1.width * (targetHeight / img1.height);
            const w2 = img2.width * (targetHeight / img2.height);
            canvas.width = w1 + w2;
            canvas.height = targetHeight;
            ctx.drawImage(img1, 0, 0, w1, targetHeight);
            ctx.drawImage(img2, w1, 0, w2, targetHeight);
        } else {
            const targetWidth = Math.max(img1.width, img2.width);
            const h1 = img1.height * (targetWidth / img1.width);
            const h2 = img2.height * (targetWidth / img2.width);
            canvas.width = targetWidth;
            canvas.height = h1 + h2;
            ctx.drawImage(img1, 0, 0, targetWidth, h1);
            ctx.drawImage(img2, 0, h1, targetWidth, h2);
        }

        const combinedDataUrl = canvas.toDataURL('image/png');
        const file = await dataURLToFile(combinedDataUrl, `fused-${Date.now()}.png`);
        
        // 设置全局状态
        state.mainImageFile = file;
        state.referenceImageFiles = [];
        state.pendingBaseImageShare = true;
        
        // 设置提示词
        const userInput = document.getElementById('userInput');
        if (userInput) userInput.value = prompt;
        
        // 触发任务
        await handleSend();
        
        placeholder.remove();
        
    } catch (e) {
        console.error(e);
        placeholder.remove();
        addMessage({ sender: 'bot', type: 'text', content: `❌ 融合准备失败: ${e.message}` });
    }
}
