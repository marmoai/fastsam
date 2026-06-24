import { state } from '../core/state.js';
import { addMessage } from './chat-panel.js';
import { editOrQueryImageWithGemini, classifyImageCategory } from '../ai-services/skills-engine.js';
import { dataURLToFile, fileToBase64 } from '../core/utils.js';
import { addImageToWorkbench, deleteWorkbenchItem } from './workbench-core.js';
import { openMaskEditor } from '../graphics/mask-drawer.js';
import { showLayerManagerModal, openCameraAngleModal, showVideoPromptModal } from './modals.js';
import { startPreciseEditMode } from './fusion-editor.js';
import { generateVeoVideo } from '../ai-services/gemini-client.js';
import { triggerMagicLayers } from './layer-manager.js';

export function setupToolboxEvents(context) {
    const {
        workbenchToolbox,
        workbenchGrid,
        workbenchZoomContainer,
        pushSelectedToChat
    } = context;

    const bindToolboxBtn = (selector, handler) => {
        const btn = workbenchToolbox.querySelector(selector);
        if (btn) {
            btn.addEventListener('click', handler);
        }
    };

    bindToolboxBtn('.select', () => {
        pushSelectedToChat();
        window.hideWorkbenchToolbox();
    });
    

    bindToolboxBtn('.ai-spark', () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        
        window.hideWorkbenchToolbox();
        if (typeof window.showFloatingFusionEditor === 'function') {
            window.showFloatingFusionEditor(itemId);
        }
    });

    bindToolboxBtn('.camera-angle', () => {
        if (!state.currentActiveWorkbenchItemId) return;
        const item = state.workbenchItems.get(state.currentActiveWorkbenchItemId);
        if (!item) return;
        
        window.hideWorkbenchToolbox();
        openCameraAngleModal(item);
    });

    bindToolboxBtn('.upscale', async () => {
        if (!state.currentActiveWorkbenchItemId) return;
        const itemId = state.currentActiveWorkbenchItemId;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();

        const executeUpscale = async (customPrompt) => {
            const promptToUse = customPrompt || "Image Restoration & Reconstruction: Redraw this low-quality image into a pristine, ultra-high-resolution (4K) masterpiece. Aggressively remove all blur, noise, and compression artifacts. CRITICAL: Do not just sharpen the existing pixels. Instead, synthesise and hallucinate missing high-frequency details (such as skin texture, hair strands, fabric patterns, and sharp edges) that are lost in the original. Re-imagine the subject with perfect focus and clarity while keeping the original subject identity, pose, and overall composition intact. The output must look like a sharp, professional commercial photograph taken with a modern high-end DSLR.";
            
            const placeholder = document.createElement('div');
            placeholder.className = 'world-placeholder';
            placeholder.style.left = `${parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20}px`;
            placeholder.style.top = `${parseFloat(item.el.style.top)}px`;
            placeholder.style.width = `${parseFloat(item.el.style.width)}px`;
            placeholder.style.height = `${parseFloat(item.el.style.height)}px`;
            
            const inverseScale = 1 / state.workbenchZoom;
            const visualBorderWidth = 2;
            const visualFontSize = 12;
            placeholder.style.border = `${visualBorderWidth * inverseScale}px dashed #ccc`;
            
            placeholder.innerHTML = '<span><i class="fas fa-spinner fa-spin"></i> 高清增强中...</span>';
            
            const span = placeholder.querySelector('span');
            if (span) {
                span.style.fontSize = `${visualFontSize * inverseScale}px`;
            }

            workbenchGrid.appendChild(placeholder);

            const tempMsg = addMessage({ 
                sender: 'bot', 
                type: 'text', 
                content: '✨ 正在执行智能高清增强 (Super Upscale)...\nAI 正在重绘细节、锐化边缘并提升整体质感，请稍候...' 
            });

            try {
                const result = await editOrQueryImageWithGemini(promptToUse, item.file || item.dataUrl);
                
                if (result.success && result.imageData) {
                    const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
                    const file = await dataURLToFile(imgSrc, `hd-upscaled-${Date.now()}.png`);
                    
                    const itemRect = item.el.getBoundingClientRect();
                    const containerRect = workbenchZoomContainer.getBoundingClientRect();
                    
                    if (placeholder && placeholder.parentNode) placeholder.remove();
                    
                    const newId = await addImageToWorkbench(file, '高清增强', {
                        x: parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20,
                        y: parseFloat(item.el.style.top),
                        parentId: itemId
                    });
                    
                    if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat('高清增强', promptToUse, imgSrc, executeUpscale);
                    } else {
                        addMessage({ sender: 'bot', type: 'text', content: '✅ **高清增强完成！**' });
                    }
                    window.selectWorkbenchItem(newId);
                } else {
                    throw new Error("生成结果为空");
                }
            } catch (e) {
                console.error("Upscale failed:", e);
                if (placeholder && placeholder.parentNode) placeholder.remove();
                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                addMessage({ sender: 'bot', type: 'text', content: `❌ 高清增强失败: ${e.message}` });
            }
        };

        await executeUpscale();
    });

    bindToolboxBtn('.box-extract', () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item || !item.el) return;

        window.hideWorkbenchToolbox();

        // Remove any existing overlay
        const existingOverlay = item.el.querySelector('.box-extract-overlay');
        if (existingOverlay) existingOverlay.remove();

        // Create overlay covering the image
        const overlay = document.createElement('div');
        overlay.className = 'box-extract-overlay';
        overlay.style.cssText = `
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            z-index: 1000; cursor: crosshair; background: rgba(0,0,0,0.1);
        `;

        let isDrawing = false;
        let startX, startY;
        let rectEl = null;

        const onMouseDown = (e) => {
            e.stopPropagation();
            e.preventDefault();
            isDrawing = true;
            const zoom = state.workbenchZoom || 1;
            const rect = overlay.getBoundingClientRect();
            startX = (e.clientX - rect.left) / zoom;
            startY = (e.clientY - rect.top) / zoom;

            if (rectEl) {
                rectEl.remove();
                rectEl = null;
            }

            rectEl = document.createElement('div');
            rectEl.style.cssText = `
                position: absolute; border: 2px dashed #00ffcc;
                background: rgba(0,255,204,0.2); pointer-events: none;
                left: ${startX}px; top: ${startY}px; width: 0; height: 0;
            `;
            overlay.appendChild(rectEl);
        };

        const onMouseMove = (e) => {
            if (!isDrawing || !rectEl) return;
            e.stopPropagation();
            e.preventDefault();
            const zoom = state.workbenchZoom || 1;
            const rect = overlay.getBoundingClientRect();
            const currentX = (e.clientX - rect.left) / zoom;
            const currentY = (e.clientY - rect.top) / zoom;

            const width = Math.abs(currentX - startX);
            const height = Math.abs(currentY - startY);
            const left = Math.min(startX, currentX);
            const top = Math.min(startY, currentY);

            rectEl.style.left = `${left}px`;
            rectEl.style.top = `${top}px`;
            rectEl.style.width = `${width}px`;
            rectEl.style.height = `${height}px`;
        };

        const onMouseUp = async (e) => {
            if (!isDrawing) return;
            e.stopPropagation();
            e.preventDefault();
            isDrawing = false;
            
            // Allow user to click again to redraw if the box is too small Let's not remove listeners, just wait for popup input.
            if (!rectEl || parseInt(rectEl.style.width) < 10 || parseInt(rectEl.style.height) < 10) {
                if (rectEl) rectEl.remove();
                return;
            }

            // Remove mouse events so we don't redraw while typing
            overlay.removeEventListener('mousedown', onMouseDown);
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);

            // Pop up a loading indicator
            const popup = document.createElement('div');
            popup.style.cssText = `
                position: absolute;
                top: ${parseInt(rectEl.style.top) + parseInt(rectEl.style.height) + 5}px;
                left: ${rectEl.style.left};
                background: rgba(0,0,0,0.8); color: #fff; border-radius: 8px; padding: 15px 25px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); font-size: 16px; pointer-events: auto; z-index: 1001;
            `;
            popup.innerHTML = `<i class="fas fa-spinner fa-spin"></i> 正在智能识别主体...`;
            overlay.appendChild(popup);

            try {
                const ow = overlay.clientWidth;
                const oh = overlay.clientHeight;
                const rL = parseInt(rectEl.style.left);
                const rT = parseInt(rectEl.style.top);
                const rW = parseInt(rectEl.style.width);
                const rH = parseInt(rectEl.style.height);

                // Load the image to crop
                const parentImg = new Image();
                parentImg.crossOrigin = "anonymous";
                const { getProxiedUrl } = await import('../core/utils.js');
                
                let srcToCrop = item.dataUrl || item.originalDataUrl;
                if (!srcToCrop && item.file) {
                    const { fileToDataURL } = await import('../core/utils.js');
                    srcToCrop = await fileToDataURL(item.file);
                }
                
                parentImg.src = getProxiedUrl(srcToCrop);
                await new Promise((resolve, reject) => {
                    parentImg.onload = resolve;
                    parentImg.onerror = reject;
                });

                // Calculate crop relative to natural image size
                const pxMinX = (rL / ow) * parentImg.naturalWidth;
                const pxMinY = (rT / oh) * parentImg.naturalHeight;
                const pxMaxX = ((rL + rW) / ow) * parentImg.naturalWidth;
                const pxMaxY = ((rT + rH) / oh) * parentImg.naturalHeight;

                const cropCanvas = document.createElement('canvas');
                cropCanvas.width = pxMaxX - pxMinX;
                cropCanvas.height = pxMaxY - pxMinY;
                const ctx = cropCanvas.getContext('2d');
                ctx.drawImage(parentImg, pxMinX, pxMinY, cropCanvas.width, cropCanvas.height, 0, 0, cropCanvas.width, cropCanvas.height);
                const croppedDataUrl = cropCanvas.toDataURL('image/png');

                // Identify the object
                const { identifyObjectInCrop } = await import('../ai-services/skills-engine.js');
                const objectName = await identifyObjectInCrop(croppedDataUrl);

                // Map bbox to 0-1000
                const ymin = Math.floor((rT / oh) * 1000);
                const xmin = Math.floor((rL / ow) * 1000);
                const ymax = Math.floor(((rT + rH) / oh) * 1000);
                const xmax = Math.floor(((rL + rW) / ow) * 1000);
                const bbox = [ymin, xmin, ymax, xmax];

                // Append layer
                if (!item.scene) item.scene = { layers: [] };
                if (!item.layers) item.layers = [];
                const currentLayers = item.scene.layers.length > 0 ? item.scene.layers : item.layers;

                currentLayers.unshift({
                    id: `box-layer-${Date.now()}`,
                    name: objectName || '未命名物体',
                    bbox: bbox,
                    assetStatus: 'idle'
                });

                if (item.scene && item.scene.layers) item.scene.layers = currentLayers;
                item.layers = currentLayers;
                state.workbenchItems.set(itemId, item);

                overlay.remove();

                // Open Layer Manager
                const { showLayerManagerModal, renderLayerList } = await import('./modals.js');
                showLayerManagerModal(itemId);
                
                // Re-render layers over the picture
                const { renderCanvasLayers } = await import('./workbench/layers.js');
                renderCanvasLayers(itemId);

            } catch (e) {
                console.error("Auto identify failed:", e);
                alert("识别失败：" + e.message);
                overlay.remove();
            }
        };

        overlay.addEventListener('mousedown', onMouseDown);
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);

        item.el.appendChild(overlay);
    });

    bindToolboxBtn('.erase', () => {
        if (state.currentActiveWorkbenchItemId) {
            const item = state.workbenchItems.get(state.currentActiveWorkbenchItemId);
            if (item) {
                state.mainImageFile = item.file || item.dataUrl;
                state.maskDataUrl = null;
                state.pendingBaseImageShare = true;
                state.isContextPreviewHidden = false;
                
                state.currentIntentLock = 'erase';
                
                window.updateImagePreview();
                window.updateSendBtnState();
                
                openMaskEditor(item.file || item.dataUrl);
                window.selectWorkbenchItem(state.currentActiveWorkbenchItemId);
            }
        }
        window.hideWorkbenchToolbox();
    });

    bindToolboxBtn('.precise-edit', () => {
        if (state.currentActiveWorkbenchItemId) {
            const itemId = state.currentActiveWorkbenchItemId;
            const item = state.workbenchItems.get(itemId);
            if (item) {
                state.mainImageFile = item.file || item.dataUrl;
                state.maskDataUrl = null;
                state.pendingBaseImageShare = true;
                
                window.updateImagePreview();
                window.updateSendBtnState();
                
                window.hideWorkbenchToolbox();
                startPreciseEditMode(itemId);
            }
        }
    });

    bindToolboxBtn('.material', () => {
        if (state.currentActiveWorkbenchItemId) {
            const item = state.workbenchItems.get(state.currentActiveWorkbenchItemId);
            if (item) {
                state.mainImageFile = item.file || item.dataUrl;
                state.maskDataUrl = null;
                state.pendingBaseImageShare = true;
                state.isContextPreviewHidden = false;
                
                state.currentIntentLock = 'material';
                
                window.updateImagePreview();
                window.updateSendBtnState();
                
                openMaskEditor(item.file || item.dataUrl);
                window.selectWorkbenchItem(state.currentActiveWorkbenchItemId);
            }
        }
        window.hideWorkbenchToolbox();
    });

    bindToolboxBtn('.delete', () => {
        if (state.currentActiveWorkbenchItemId) deleteWorkbenchItem(state.currentActiveWorkbenchItemId);
        window.hideWorkbenchToolbox();
    });

    bindToolboxBtn('.download', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();

        let dataUrl = item.dataUrl;
        if (!dataUrl && item.file) {
            dataUrl = await window.fileToDataURL(item.file);
        } else if (!dataUrl && item.el && item.el.querySelector('img')) {
            dataUrl = item.el.querySelector('img').src;
        }

        if (dataUrl) {
            try {
                // Use getProxiedUrl to avoid CORS issues when fetching the image
                const { getProxiedUrl } = await import('../core/utils.js');
                const fetchUrl = getProxiedUrl(dataUrl);
                
                // Fetch the image to get a Blob
                const response = await fetch(fetchUrl);
                const blob = await response.blob();
                
                // Determine correct extension from blob type
                let extension = 'png';
                if (blob.type) {
                    extension = blob.type.split('/')[1] || 'png';
                    if (extension === 'jpeg') extension = 'jpg';
                }
                
                // Get base filename
                let baseName = `download-${Date.now()}`;
                if (item.file && item.file.name) {
                    const lastDot = item.file.name.lastIndexOf('.');
                    baseName = lastDot !== -1 ? item.file.name.substring(0, lastDot) : item.file.name;
                }
                const suggestedName = `${baseName}.${extension}`;

                // Standard download (might auto-download depending on browser settings)
                const blobUrl = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = blobUrl;
                a.download = suggestedName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                // Clean up the blob URL after a short delay
                setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
            } catch (e) {
                console.error('下载图片失败:', e);
                // Fallback to the old method if fetch fails (e.g., CORS issues without proxy)
                const a = document.createElement('a');
                a.href = dataUrl;
                a.download = item.file ? item.file.name : `download-${Date.now()}.png`;
                a.target = '_blank';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        } else {
            console.error('无法获取图片数据进行下载');
        }
    });

    bindToolboxBtn('.critique', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;
        
        window.hideWorkbenchToolbox();

        if (item.critiquesData && item.critiquesData.length > 0) {
            const existingPanel = document.getElementById(`critique-panel-${itemId}`);
            if (existingPanel) {
                if (existingPanel.style.display === 'none') {
                    existingPanel.style.display = 'block';
                } else {
                    existingPanel.style.display = 'none';
                }
                return;
            }
        }

        const btn = document.querySelector('.critique');
        if(btn) btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 辩论中...';

        try {
            const { triggerAgentDebate } = await import('./debate.js');
            await triggerAgentDebate(itemId);
            
        } catch (e) {
            console.error("Critique failed:", e);
            addMessage({ sender: 'bot', type: 'text', content: `❌ 辩论生成失败: ${e.message}` });
        } finally {
            if(btn) btn.innerHTML = '<i class="fas fa-comments"></i> 专家辩论';
        }
    });

    bindToolboxBtn('.layers', async () => {
        const itemId = state.currentActiveWorkbenchItemId; 
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;
        
        window.hideWorkbenchToolbox();
        showLayerManagerModal(itemId, true);
    });

    bindToolboxBtn('.magic-layers', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();
        console.log('[Toolbox] Magic Layers clicked:', itemId);
        try {
            await triggerMagicLayers(itemId);
        } catch (error) {
            console.error('[Toolbox] Magic Layers failed:', error);
        }
    });

    bindToolboxBtn('.multiview', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();

        const executeMultiview = async (customPrompt) => {
            const placeholder = document.createElement('div');
            placeholder.className = 'world-placeholder';
            const inverseScale = 1 / state.workbenchZoom;
            placeholder.style.cssText = `
                left: ${parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20}px;
                top: ${parseFloat(item.el.style.top)}px;
                width: ${parseFloat(item.el.style.width)}px;
                height: ${parseFloat(item.el.style.height)}px;
                border: ${2 * inverseScale}px dashed #ccc;
            `;
            placeholder.innerHTML = `<span style="font-size: ${14 * inverseScale}px;"><i class="fas fa-eye fa-spin"></i> 正在智能识别场景类型...</span>`;
            workbenchGrid.appendChild(placeholder);

            let tempMsg = null;

            try {
                const base64 = await fileToBase64(item.file || item.dataUrl);
                
                const category = await classifyImageCategory(base64, item.file?.type || 'image/png');
                
                let finalPrompt = "";
                let uiMessage = "";
                let placeholderTitle = "";

                if (category.includes('CHARACTER')) {
                    uiMessage = "👤 **正在执行【角色三视图】推演**...\nAI 正在精准还原角色的正面、侧面及背面细节。";
                    placeholderTitle = "正在绘制角色三视图...";
                    finalPrompt = `Create a professional character sheet (3-view) based on this character. 
                    Must include: Front view, Side view, and Back view.
                    Style: Character design orthographic projection, neutral background, consistent clothing and features.`;
                } else if (category.includes('PRODUCT')) {
                    uiMessage = "🗿 **正在执行【物品多维视图】推演**...\nAI 正在对该物体的结构、材质进行多角度拆解。";
                    placeholderTitle = "正在生成物品结构图...";
                    finalPrompt = `Create a professional product design multi-view (3-view) based on this object.
                    Must include: Front, Side, and Top views.
                    Style: Product design orthographic sheet, industrial design presentation, identical material and lighting.`;
                } else {
                    uiMessage = "📐 **正在执行【空间/建筑多维视图】推演**...\nAI 正在解析大空间的透视逻辑，生成俯视图、左视图及右视图。";
                    placeholderTitle = "正在推演空间三向视图...";
                    finalPrompt = `Create a professional architectural orthographic multi-view presentation (3-view sheet) based on this image. 
                            Maintain 100% consistency in materials, lighting, textures, and spatial modeling from the original image. 
                            This is NOT a drawing, blueprint, or sketch; it is a photorealistic multi-angle photorealistic visualization. 
                    The final image MUST contain exactly:
                    1. A Top view (俯视图) from a bird's eye perspective.
                    2. A Left Elevation (左视图) showing the left side of the structure/space.
                    3. A Right Elevation (右视图) showing the right side of the structure/space.
                    Style: Professional architectural rendering, realistic textures, consistent spatial logic, high-fidelity visualization.`;
                }

                const promptToUse = customPrompt || finalPrompt;

                tempMsg = addMessage({ sender: 'bot', type: 'text', content: uiMessage });
                if (placeholder && placeholder.parentNode) {
                    placeholder.querySelector('span').innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${placeholderTitle}`;
                }

                const result = await editOrQueryImageWithGemini(promptToUse, item.file || item.dataUrl);

                if (result.success && result.imageData) {
                    const imgSrc = `data:${result.mimeType};base64,${result.imageData}`;
                    const file = await dataURLToFile(imgSrc, `multiview-${Date.now()}.png`);
                    
                    if (placeholder && placeholder.parentNode) placeholder.remove();
                    addImageToWorkbench(file, '多维视图', {
                        x: parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 20,
                        y: parseFloat(item.el.style.top),
                        parentId: itemId
                    });
                    
                    if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat('多维视图', promptToUse, imgSrc, executeMultiview);
                    } else {
                        addMessage({ sender: 'bot', type: 'text', content: '✅ **多维视图生成完成！**' });
                    }
                }
            } catch (e) {
                console.error("Multiview adaptation failed:", e);
                if (placeholder && placeholder.parentNode) placeholder.remove();
                if (tempMsg && tempMsg.parentNode) tempMsg.remove();
                addMessage({ sender: 'bot', type: 'text', content: `❌ 处理失败: ${e.message}` });
            }
        };

        await executeMultiview();
    });

    bindToolboxBtn('.veo-video', async () => {
        const itemId = state.currentActiveWorkbenchItemId;
        if (!itemId) return;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        window.hideWorkbenchToolbox();

        showVideoPromptModal(async (videoPrompt) => {
            addMessage({ sender: 'user', type: 'text', content: `🎥 视频生成指令: ${videoPrompt}` });
            addMessage({ sender: 'bot', type: 'text', content: '🎬 **Veo 视频生成中**...\n正在根据您的图片和指令生成视频，请耐心等待 (约 1-2 分钟)。' });

            try {
                const b64 = await fileToBase64(item.file || item.dataUrl);
                
                const rect = item.el.getBoundingClientRect();
                const ratio = rect.width / rect.height;
                const veoRatio = ratio > 1 ? '16:9' : '9:16';

                const vidBlob = await generateVeoVideo(videoPrompt, b64, item.file?.type || 'image/png', veoRatio);

                if (vidBlob) {
                    const vidUrl = URL.createObjectURL(vidBlob);

                    const videoHtml = `
                        <div style="margin-top: 10px; width: 100%;">
                            <video src="${vidUrl}" controls autoplay loop muted playsinline 
                                style="width: 100%; max-width: 300px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); background: #000;">
                            </video>
                            <div style="font-size: 12px; color: #666; margin-top: 8px; display:flex; justify-content:space-between; align-items:center;">
                                <span>✅ 视频生成成功</span>
                                <a href="${vidUrl}" download="veo-${Date.now()}.mp4" style="color:var(--primary-color); text-decoration:none; font-weight:500;">
                                    <i class="fas fa-download"></i> 下载
                                </a>
                            </div>
                        </div>
                    `;
                    
                    if (window.addWorkbenchActionToChat) {
                        await window.addWorkbenchActionToChat('Veo 视频生成', videoPrompt, null, null, videoHtml);
                    } else {
                        addMessage({ sender: 'bot', type: 'html', content: videoHtml });
                    }
                }
            } catch (e) {
                console.error(e);
                addMessage({ sender: 'bot', type: 'text', content: `❌ 视频生成失败: ${e.message}` });
            }
        });
    });
}
