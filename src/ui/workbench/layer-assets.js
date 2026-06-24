import { state } from '../../core/state.js';
import { editOrQueryImageWithGemini } from '../../ai-services/skills-engine.js';
import { dataURLToFile, fileToBase64, getProxiedUrl, RECOGNIZE_BACKEND_URL } from '../../core/utils.js';
import { getCleanupLayerForEditableLayer } from '../../services/semantic-layer-views.js';

// We'll need functions to generate mask, cutout, etc.
// For now, we'll simulate the generation or use basic Gemini calls.

const DISABLE_CLEAN_PLATE_FOR_FASTSAM_TEST = false;
const USE_EXPERIMENTAL_CLEAN_PLATE_TEST_ROUTER = true;

function cloneSerializable(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function stripDataUrlPrefix(dataUrl) {
    if (typeof dataUrl !== 'string') return '';
    const commaIndex = dataUrl.indexOf(',');
    return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
}

async function callExperimentalCleanPlateRouter({ mode, prompt, image, image2 = null, image3 = null, num_inference_steps = 40, cfg = 4, size = null, seed = null, negative_prompt = null }) {
    const payload = {
        mode,
        prompt,
        image: stripDataUrlPrefix(image),
        num_inference_steps,
        cfg
    };
    if (image2) payload.image2 = stripDataUrlPrefix(image2);
    if (image3) payload.image3 = stripDataUrlPrefix(image3);
    if (size) payload.size = size;
    if (seed !== null && seed !== undefined) payload.seed = seed;
    if (negative_prompt) payload.negative_prompt = negative_prompt;

    const response = await fetch(RECOGNIZE_BACKEND_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    const text = await response.text();
    let data = null;
    try {
        data = JSON.parse(text);
    } catch (e) {
        data = { raw: text };
    }

    if (!response.ok) {
        throw new Error(data?.error || data?.message || `clean plate router failed with ${response.status}`);
    }

    const imageData = data?.data?.[0]?.b64_json || data?.data?.[0]?.base64 || data?.imageData || data?.output?.[0]?.b64_json || data?.output?.[0]?.base64 || null;
    if (!imageData) {
        throw new Error('clean plate router returned no image data');
    }

    const mimeType = data?.mimeType || 'image/png';
    const normalizedImageData = typeof imageData === 'string' && imageData.startsWith('data:image/')
        ? imageData
        : `data:${mimeType};base64,${imageData}`;
    return {
        success: true,
        mimeType,
        imageData: normalizedImageData.split(',')[1],
        raw: data
    };
}

function getSelectedCleanPlateModelMode() {
    if (state.selectedModel === 'qwen-cleanplate') return 'qwen-cleanplate';
    if (state.selectedModel === 'gpt-image-2') return 'gpt-image-2';
    return 'gemini';
}

function isTextCleanupLayer(layer) {
    if (!layer) return false;
    return !!(
        layer.isText ||
        layer.semanticType === 'element_text' ||
        (typeof layer.name === 'string' && layer.name.toLowerCase().includes('text'))
    );
}

function getCleanupTargetName(layer) {
    const rawName = typeof layer?.name === 'string' ? layer.name : 'target element';
    return rawName.replace(/^Text:\s*/i, '').trim() || 'target element';
}

function buildQwenSemanticCleanPlatePrompt(layers, customPromptHint = '', options = {}) {
    const cleanupLayers = Array.isArray(layers) ? layers.filter(Boolean) : [];
    const targetNames = cleanupLayers.map(getCleanupTargetName).filter(Boolean);
    const allTextLike = areAllTextCleanupLayers(cleanupLayers);
    const hint = customPromptHint ? `Additional hint: ${customPromptHint}` : '';
    const preserveBackgroundOnly = options.preserveBackgroundOnly === true;

    if (preserveBackgroundOnly) {
        const removalSummary = targetNames.length > 0
            ? `Remove the following foreground elements from the image: ${targetNames.join(', ')}.`
            : 'Remove all visible foreground elements from the image.';

        return `
            Remove all text, logos, product items, food items, drink items, price badges, white cards, decorative shapes, and other foreground elements from this image.
            Preserve only the original background surface and its texture, color, lighting, and pattern.
            Reconstruct the cleared areas so the final result reads as a clean background plate, with no ghosting, no leftover fragments, and no newly invented objects.
            Do not regenerate any of the removed foreground content.
            ${removalSummary}
            ${hint}
        `.trim();
    }

    if (allTextLike) {
        return `
            [TASK: SEMANTIC BACKGROUND CLEANUP]
            Remove all visible text, typography, letters, numbers, captions, labels, prices, logos, badges, watermarks, and website text from the image.
            Preserve all non-text visual content, products, food, decorations, panels, shapes, lighting, perspective, and overall layout.
            Seamlessly reconstruct the background under every removed text region so there are no text fragments, halos, strokes, shadows, or replacement artifacts left behind.
            Do not add any new text or objects.
            ${hint}
        `;
    }

    const listedTargets = targetNames.length
        ? `Remove these visible target elements if they appear in the image: ${targetNames.join(', ')}.`
        : 'Remove the visible target elements that should be cleaned from the image.';

    return `
        [TASK: SEMANTIC BACKGROUND CLEANUP]
        ${listedTargets}
        Preserve the surrounding background structure, products, decorations, lighting, and layout.
        Reconstruct the cleaned regions seamlessly with no ghosting, outlines, leftover fragments, or newly invented objects.
        If any target includes text or logos, remove those traces completely as well.
        ${hint}
    `;
}

function buildQwenObjectRemovalPrompt(layers, customPromptHint = '') {
    const cleanupLayers = Array.isArray(layers) ? layers.filter(Boolean) : [];
    const targetNames = cleanupLayers.map(getCleanupTargetName).filter(Boolean);
    const targetLabel = targetNames.length > 1
        ? targetNames.join('、')
        : (targetNames[0] || '目标素材');
    const hint = customPromptHint ? `参考信息：${customPromptHint}` : '';
    return `把图中的${targetLabel}素材去除掉，只移除这些目标本身，保持其他排版、文字、装饰、背景纹理、光影和剩余内容不变。${hint}`.trim();
}

function areAllTextCleanupLayers(layers) {
    const cleanupLayers = Array.isArray(layers) ? layers.filter(Boolean) : [];
    return cleanupLayers.length > 0 && cleanupLayers.every(isTextCleanupLayer);
}

function shouldUseQwenSemanticCleanPlate(layers) {
    if (getSelectedCleanPlateModelMode() !== 'qwen-cleanplate' || !USE_EXPERIMENTAL_CLEAN_PLATE_TEST_ROUTER) {
        return false;
    }
    const cleanupLayers = Array.isArray(layers) ? layers.filter(Boolean) : [];
    return cleanupLayers.length > 0;
}

async function persistCleanPlateState(itemId, item) {
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
                cleanPlateDataUrl: item.cleanPlateDataUrl || null,
                cleanPlateStatus: item.cleanPlateStatus || 'idle'
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

export async function extractLayerAsset(itemId, layerIndex) {
    const item = state.workbenchItems.get(itemId);
    if (!item || !item.scene || !item.scene.layers) return;

    const layer = item.scene.layers[layerIndex];
    if (!layer) return;
    
        
    // Skip background or uneditable layers for now, or process them differently
    if (!layer.editable || layer.category === 'background') {
        layer.assetStatus = "ready"; // Mark as ready so we don't block
        updateLayerUI(itemId, layerIndex);
        return;
    }

    if (layer.assetStatus === "processing" || layer.assetStatus === "editing") {
        return;
    }

    layer.assetStatus = "processing";
    updateLayerUI(itemId, layerIndex);

    try {
        // Only generate mask and cutout if not already done
        if (!layer.cutoutUrl) {
            // 1. Generate Mask
            const localMaskUrl = await generateMask(item.dataUrl, layer);
            layer.maskUrl = await uploadImageToOSS(localMaskUrl);
            
            // 2. Generate Cutout
            const localCutoutUrl = await generateCutout(item.dataUrl, localMaskUrl, layer.bbox);
            layer.cutoutUrl = await uploadImageToOSS(localCutoutUrl);
            
            // 3. Generate Preview
            layer.previewUrl = layer.cutoutUrl;

            // 4. Initialize Versioning
            layer.versions = [{
                id: 'v1',
                cutoutUrl: layer.cutoutUrl,
                maskUrl: layer.maskUrl,
                previewUrl: layer.previewUrl,
                prompt: 'original',
                createdAt: Date.now(),
                source: 'original'
            }];
            layer.activeVersionId = 'v1';
            layer.interactionLock = null;
        }

        layer.assetStatus = "ready";
        updateLayerUI(itemId, layerIndex);

        // --- V2.5 CLEAN PLATE TRIGGER ---
        item.cleanPlateStatus = 'cleaning';
        updateLayerUI(itemId, layerIndex); // Update UI to show cleaning status
        
        const baseBg = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
        const cleanupLayer = getCleanupLayerForEditableLayer(item, layer, { preferEditableTextBbox: true });
        const cleanedBg = await cleanBackground(baseBg, cleanupLayer);
        
        if (cleanedBg) {
            item.cleanPlateDataUrl = cleanedBg;
            item.dataUrl = cleanedBg;
        }
        item.cleanPlateStatus = 'ready';
        await persistCleanPlateState(itemId, item);
        if (window.historyManager) window.historyManager.pushState();
        updateLayerUI(itemId, layerIndex);
        
    } catch (error) {
        console.error(`Failed to generate assets for layer ${layer.id}:`, error);
        layer.assetStatus = "error";
        item.cleanPlateStatus = 'error';
    }

    updateLayerUI(itemId, layerIndex);
}

/**
 * 局部图层编辑逻辑 (V2.3)
 */
export async function editLayerAsset(itemId, layerIndex, prompt) {
    const item = state.workbenchItems.get(itemId);
    if (!item || !item.scene || !item.scene.layers) return;
    
    const layer = item.scene.layers[layerIndex];
    if (!layer || layer.interactionLock?.editingDisabled || layer.assetStatus === 'processing') return;
    
    console.log(`EDITING LAYER: ${layer.name}, status: ${layer.assetStatus}`);

    // Ensure versions array exists
    if (!layer.versions) {
        layer.versions = [{
            id: 'v1',
            cutoutUrl: layer.cutoutUrl,
            maskUrl: layer.maskUrl,
            previewUrl: layer.previewUrl,
            prompt: 'original',
            createdAt: Date.now(),
            source: 'original'
        }];
        layer.activeVersionId = 'v1';
    }
    
    // Set interaction lock
    layer.interactionLock = {
        draggingDisabled: true,
        editingDisabled: true,
        reason: "rendering"
    };
    
    layer.assetStatus = "editing";
    updateLayerUI(itemId, layerIndex);
    
    try {
        // 构造带约束的 Prompt
        const constrainedPrompt = `
            [TASK: PER-LAYER OBJECT REPAINTING]
            Target Object: "${layer.name}" (Hint: ${layer.promptHint || 'None'})
            User Instruction: "${prompt}"
            
            CRITICAL CONSTRAINTS:
            1. PERSPECTIVE LOCK: Maintain the exact 3D perspective, orientation, and volume of the original object.
            2. COMPOSITION LOCK: The object must stay within its original bounding box and maintain its relative scale.
            3. STYLE HARMONY: The new version must match the lighting, shadows, and overall aesthetic of the surrounding environment.
            4. ISOLATION: Only modify the object itself. Do not add background elements or change the area outside the object's silhouette.
            
            Return the full image with the object modified.
        `;
        
        // 调用 Gemini 进行局部重绘
        // V2.4: 使用当前场景合成图作为底图，而不是原始图
        console.log("--- STARTING LAYER EDIT (V2.4) ---");
        const currentSceneImage = await exportCurrentSceneImage(itemId);
        if (!currentSceneImage) {
            throw new Error("Failed to export current scene image.");
        }
        console.log("SCENE EXPORTED SUCCESSFULLY");

        // V2.4: 重要！如果图层移动过，必须重新生成 Mask，否则 Mask 位置与 BBox 不符
        // 之前的 maskUrl 是在初始分析时生成的，如果用户拖动了图层，旧的 Mask 就会失效
        layer.maskUrl = await generateMask(currentSceneImage, layer);
        console.log("MASK REGENERATED FOR CURRENT BBOX");
        
        const result = await editOrQueryImageWithGemini(
            constrainedPrompt, 
            currentSceneImage, 
            [], 
            layer.maskUrl,
            null, // forcedAspectRatio
            true  // forceMaterialTask
        );
        
        if (result && result.success && result.imageData) {
            const newFullImageData = `data:${result.mimeType};base64,${result.imageData}`;
            
            // Debug logs (V2.4)
            console.log("GEMINI RETURNED IMAGE DATA, size:", result.imageData.length);
            
            // 从生成的全图中重新提取 Cutout
            // V2.5: 使用融合提取，保留阴影和光影
            const backgroundRef = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
            const newCutoutUrlLocal = await extractFusionCutout(newFullImageData, backgroundRef, layer.bbox);
            if (!newCutoutUrlLocal) {
                console.error("CUTOUT GENERATION FAILED for layer:", layer.name);
                throw new Error("Failed to generate cutout from AI result.");
            }
            const newCutoutUrl = await uploadImageToOSS(newCutoutUrlLocal);
            console.log("NEW CUTOUT GENERATED SUCCESSFULLY, length:", newCutoutUrl.length);
            
            // Append new version
            const newVersionId = 'v' + (layer.versions.length + 1);
            layer.versions.push({
                id: newVersionId,
                cutoutUrl: newCutoutUrl,
                maskUrl: layer.maskUrl,
                previewUrl: newCutoutUrl,
                prompt: prompt,
                createdAt: Date.now(),
                source: 'edited'
            });
            
            layer.activeVersionId = newVersionId;
            layer.cutoutUrl = newCutoutUrl;
            layer.previewUrl = newCutoutUrl;
            
            console.log("VERSION SYSTEM UPDATED:", newVersionId);
            
            layer.assetStatus = "ready";
            
            // 自动触发 AI 深度融合
            await fuseSceneWithAI(itemId, layerIndex, prompt);
        } else {
            console.error("Gemini Edit Failed:", result);
            throw new Error("AI generation failed or returned no image.");
        }
    } catch (error) {
        console.error(`Failed to edit layer ${layer.id}:`, error);
        layer.assetStatus = "error";
    } finally {
        // Clear interaction lock
        layer.interactionLock = null;
        updateLayerUI(itemId, layerIndex);
    }
}


export function undoLayerVersion(itemId, layerIndex) {
    const item = state.workbenchItems.get(itemId);
    if (!item || !item.scene || !item.scene.layers) return;
    
    const layer = item.scene.layers[layerIndex];
    if (!layer || !layer.versions || layer.versions.length <= 1) return;
    
    const currentIndex = layer.versions.findIndex(v => v.id === layer.activeVersionId);
    if (currentIndex > 0) {
        const prevVersion = layer.versions[currentIndex - 1];
        layer.activeVersionId = prevVersion.id;
        layer.cutoutUrl = prevVersion.cutoutUrl;
        layer.maskUrl = prevVersion.maskUrl;
        layer.previewUrl = prevVersion.previewUrl;
        updateLayerUI(itemId, layerIndex);
    }
}

export function resetLayerVersion(itemId, layerIndex) {
    const item = state.workbenchItems.get(itemId);
    if (!item || !item.scene || !item.scene.layers) return;
    
    const layer = item.scene.layers[layerIndex];
    if (!layer || !layer.versions || layer.versions.length === 0) return;
    
    const originalVersion = layer.versions[0];
    layer.activeVersionId = originalVersion.id;
    layer.cutoutUrl = originalVersion.cutoutUrl;
    layer.maskUrl = originalVersion.maskUrl;
    layer.previewUrl = originalVersion.previewUrl;
    updateLayerUI(itemId, layerIndex);
}

function updateLayerUI(itemId, layerIndex) {
    const item = state.workbenchItems.get(itemId);
    if (!item) return;

    // Update the base image if cleanPlateDataUrl exists
    if (item.el) {
        const imgEl = item.el.querySelector('.crop-container > img');
        if (imgEl) {
            const bgSrc = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
            if (imgEl.getAttribute('src') !== bgSrc) {
                imgEl.src = bgSrc;
            }
        }
    }

    // Re-render the layer list to show status
    if (window.renderLayerList) {
        const layersToRender = item.scene && item.scene.layers ? item.scene.layers : item.layers;
        if (layersToRender) {
            window.renderLayerList(layersToRender, itemId);
        }
    }
    // Re-render the canvas layers to switch to cutoutUrl if ready
    if (window.renderCanvasLayers) {
        window.renderCanvasLayers(itemId);
    }
}

// --- Helper functions for asset generation ---

/**
 * Helper to load an image with timeout and error handling
 */
async function loadImage(src) {
    if (!src) return null;
    return new Promise((resolve, reject) => {
        const img = new Image();
        // Only use crossOrigin if it's not a data URL
        if (!src.startsWith('data:')) {
            img.crossOrigin = "anonymous";
        }
        
        const timeout = setTimeout(() => {
            img.onload = null;
            img.onerror = null;
            reject(new Error(`Image load timeout for: ${src.substring(0, 50)}...`));
        }, 15000);

        img.onload = () => {
            clearTimeout(timeout);
            resolve(img);
        };
        img.onerror = (e) => {
            clearTimeout(timeout);
            reject(new Error(`Failed to load image: ${src.substring(0, 50)}...`));
        };
        const resolvedSrc = getProxiedUrl(src);
        if (!resolvedSrc) {
            reject(new Error('Failed to load image: invalid image source'));
            return;
        }
        img.src = resolvedSrc;
    });
}

/**
 * 导出当前场景合成图 (V2.4)
 * 将背景和所有已就绪的图层合成一张图，作为 AI 编辑的底图
 */
export async function exportCurrentSceneImage(itemId) {
    const item = state.workbenchItems.get(itemId);
    if (!item) return null;

    try {
        const bgSrc = item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl;
        const bgImg = await loadImage(bgSrc);
        if (!bgImg) return null;

        const canvas = document.createElement('canvas');
        canvas.width = bgImg.width;
        canvas.height = bgImg.height;
        const ctx = canvas.getContext('2d');

        // 1. 绘制背景
        ctx.drawImage(bgImg, 0, 0);

        // 2. 绘制所有已就绪且可见的图层
        const layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
        if (layers) {
            for (const layer of layers) {
                // 跳过背景层（已经画过了）
                if (layer.category === 'background') continue;
                
                // 检查可见性
                const layerState = item.layerStates?.get(layers.indexOf(layer));
                const isVisible = layerState ? layerState.visible !== false : layer.visible !== false;
                
                if (isVisible && layer.assetStatus === 'ready') {
                    const activeVersion = layer.versions?.find(v => v.id === layer.activeVersionId);
                    const cutoutUrl = activeVersion?.cutoutUrl || layer.cutoutUrl;
                    
                    if (cutoutUrl) {
                        try {
                            const layerImg = await loadImage(cutoutUrl);
                            if (layerImg) {
                                const [ymin, xmin, ymax, xmax] = layer.bbox;
                                const x = (xmin / 1000) * canvas.width;
                                const y = (ymin / 1000) * canvas.height;
                                const w = ((xmax - xmin) / 1000) * canvas.width;
                                const h = ((ymax - ymin) / 1000) * canvas.height;
                                ctx.drawImage(layerImg, x, y, w, h);
                            }
                        } catch (e) {
                            console.warn("Skipping layer in export due to load error:", e);
                        }
                    }
                }
            }
        }

        return canvas.toDataURL('image/jpeg', 0.9);
    } catch (error) {
        console.error("exportCurrentSceneImage failed:", error);
        return item.dataUrl; // Fallback to original
    }
}

export async function cleanMultipleBackgrounds(baseBackgroundUrl, layers, customPromptHint, options = {}) {
    if (!layers || layers.length === 0) return baseBackgroundUrl;
    if (DISABLE_CLEAN_PLATE_FOR_FASTSAM_TEST) {
        const names = layers.map(l => l.name).join('、');
        console.log(`[cleanMultipleBackgrounds] Disabled for FastSAM test: ${names}`);
        return null;
    }
    const names = layers.map(l => l.name).join('、');
    console.log(`[cleanMultipleBackgrounds] Starting for layers: ${names}`);
    try {
        const img = await loadImage(baseBackgroundUrl);
        if (!img) {
            console.error("[cleanMultipleBackgrounds] Failed to load base background image");
            return null;
        }

        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        let clusters = layers.map(layer => {
            const isText = layer.isText || (layer.semanticType === 'element_text') || (layer.name && layer.name.includes('Text'));
            const [ymin, xmin, ymax, xmax] = layer.bbox;
            const x = (xmin / 1000) * canvas.width;
            const y = (ymin / 1000) * canvas.height;
            const w = ((xmax - xmin) / 1000) * canvas.width;
            const h = ((ymax - ymin) / 1000) * canvas.height;
            return { x, y, w, h, isText, layers: [layer] };
        });

        // Sort by Y coordinate for vertical grouping
        clusters.sort((a, b) => a.y - b.y);

        let mergedGroups = [];
        if (clusters.length > 0) {
            let currentGroup = Object.assign({}, clusters[0]);
            currentGroup.layers = [...clusters[0].layers];
            for (let i = 1; i < clusters.length; i++) {
                const next = clusters[i];
                // Use a smaller gap (5% of height) to strictly isolate main blocks without over-merging
                const gapY = next.y - (currentGroup.y + currentGroup.h);
                if (gapY < canvas.height * 0.05) {
                    const newX = Math.min(currentGroup.x, next.x);
                    const newRight = Math.max(currentGroup.x + currentGroup.w, next.x + next.w);
                    currentGroup.h = Math.max(currentGroup.y + currentGroup.h, next.y + next.h) - currentGroup.y;
                    currentGroup.x = newX;
                    currentGroup.w = newRight - newX;
                    currentGroup.layers = currentGroup.layers.concat(next.layers);
                    currentGroup.isText = currentGroup.isText || next.isText;
                } else {
                    mergedGroups.push(currentGroup);
                    currentGroup = Object.assign({}, next);
                    currentGroup.layers = [...next.layers];
                }
            }
            mergedGroups.push(currentGroup);
        }
        
        console.log(`[cleanMultipleBackgrounds] Grouped ${layers.length} layers into ${mergedGroups.length} spatial clusters.`);

        ctx.fillStyle = 'white';
        // Draw the mask based on the combined bounding box of the entire group.
        // This simplifies complex text arrangements into solid blocks for Gemini.
        for (const group of mergedGroups) {
            const expandX = group.isText
                ? Math.min(Math.max(group.w * 0.12, 28), 90)
                : Math.min(Math.max(group.w * 0.05, 20), 50);
            const expandTop = group.isText
                ? Math.min(Math.max(group.h * 0.28, 28), 90)
                : Math.min(Math.max(group.h * 0.10, 20), 50);
            const expandBottom = group.isText
                ? Math.min(Math.max(group.h * 0.22, 28), 90)
                : Math.min(Math.max(group.h * 0.10, 20), 50);
            
            const outerX = Math.max(0, group.x - expandX);
            const outerY = Math.max(0, group.y - expandTop);
            const rightEdge = Math.min(canvas.width, group.x + group.w + expandX);
            const bottomEdge = Math.min(canvas.height, group.y + group.h + expandBottom);
            const outerW = rightEdge - outerX;
            const outerH = bottomEdge - outerY;
            
            if (group.isText) {
                ctx.filter = 'none';
            } else {
                ctx.filter = 'blur(10px)';
            }
            
            ctx.fillRect(outerX, outerY, outerW, outerH);
        }
        
        const featheredMaskUrl = canvas.toDataURL('image/png');

        let groupDescriptions = mergedGroups.map((g, index) => {
            const groupNames = g.layers.map(l => l.name.replace(/^Text:\s*/, '')).join(', ');
            return `Area ${index + 1}: [${groupNames}]`;
        }).join('\\n');

        const promptHintText = customPromptHint ? `(Hint: ${customPromptHint})` : '';
        const preserveBackgroundOnly = options.preserveBackgroundOnly === true;
        const prompt = preserveBackgroundOnly
            ? buildQwenSemanticCleanPlatePrompt(layers, customPromptHint, { preserveBackgroundOnly: true })
            : `
                [TASK: BULK CLEAN PLATE / TEXT AND OBJECT REMOVAL]
                The provided mask contains ${mergedGroups.length} large solid white blocks covering specific target areas.
                ${promptHintText}
                
                We have simplified the locations. Please meticulously ERASE everything inside ALL ${mergedGroups.length} masked areas:
                ${groupDescriptions}
                
                CRITICAL CONSTRAINTS:
                1. COMPLETE ANNIHILATION: You MUST process EVERY white block in the mask, not only the largest or most central one. Completely erase EVERY SINGLE TRACE of text, letters, logo marks, button labels, strokes, borders, shadows, and outlines inside each masked block.
                2. SEAMLESS INPAINTING: Fill each erased white block by intelligently and flawlessly extending the surrounding background textures, lighting, patterns, and gradients across the gaps.
                3. NO NEW OBJECTS: Do NOT introduce any new objects, text, subjects, or focal points. The erased region must become a perfectly clean, continuous background.
                
                Return the full image with the contents of the masked areas utterly removed and the background perfectly healed (彻底抹去图层内容，无痕修复背景).
            `;

        const shouldUseQwen = shouldUseQwenSemanticCleanPlate(layers);
        const result = shouldUseQwen
            ? await callExperimentalCleanPlateRouter(
                {
                    mode: 'clean_plate_test_edit',
                    prompt: `[CLEAN_MULTIPLE_BACKGROUNDS] ${buildQwenSemanticCleanPlatePrompt(layers, customPromptHint, options)}`,
                    image: baseBackgroundUrl,
                    num_inference_steps: 40,
                    cfg: 4
                }
            )
            : await editOrQueryImageWithGemini(
                `[CLEAN_MULTIPLE_BACKGROUNDS] ${prompt}`,
                baseBackgroundUrl,
                [],
                featheredMaskUrl,
                null,
                false // forceMaterialTask = false for removal
            );

        if (result && result.success && result.imageData) {
            console.log(`[cleanMultipleBackgrounds] Successfully cleaned background for ${names}`);
            return `data:${result.mimeType};base64,${result.imageData}`;
        } else {
            console.error("[cleanMultipleBackgrounds] Gemini Clean Plate Failed:", result);
            return null;
        }
    } catch (error) {
        console.error("[cleanMultipleBackgrounds] failed:", error);
        return null;
    }
}

export async function cleanBackground(baseBackgroundUrl, layer) {
    if (DISABLE_CLEAN_PLATE_FOR_FASTSAM_TEST) {
        console.log(`[cleanBackground] Disabled for FastSAM test: ${layer?.name || 'unknown layer'}`);
        return null;
    }
    console.log(`[cleanBackground] Starting for layer ${layer.name}`);
    try {
        const img = await loadImage(baseBackgroundUrl);
        if (!img) {
            console.error("[cleanBackground] Failed to load base background image");
            return null;
        }

        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        // Clear canvas (transparent)
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate expand & feather (we want to cover the whole object, especially text)
        const [ymin, xmin, ymax, xmax] = layer.bbox;
        const x = (xmin / 1000) * canvas.width;
        const y = (ymin / 1000) * canvas.height;
        const w = ((xmax - xmin) / 1000) * canvas.width;
        const h = ((ymax - ymin) / 1000) * canvas.height;
        
        const isTextLayer = layer.isText || (layer.semanticType === 'element_text') || (layer.name && layer.name.includes('Text'));
        
        let outerX, outerY, outerW, outerH;
        
        if (isTextLayer) {
            // Expanded mask generation to ensure large text/top text is not missed
            // Use precise and minimal expansion (5%-10%) to prevent masks from merging into a giant blob,
            // but add a tiny bit more to the top to prevent ascenders from being missed.
            const expandX = Math.min(Math.max(w * 0.05, 5), 15);
            const expandTop = Math.min(Math.max(h * 0.12, 6), 18); // Slight top bias for ascenders/caps
            const expandBottom = Math.min(Math.max(h * 0.05, 4), 12);
            
            const rightEdge = Math.min(canvas.width, x + w + expandX);
            const bottomEdge = Math.min(canvas.height, y + h + expandBottom);
            
            outerX = Math.max(0, x - expandX);
            outerY = Math.max(0, y - expandTop);
            outerW = rightEdge - outerX;
            outerH = bottomEdge - outerY;
            ctx.filter = 'none'; // No feathering
        } else {
            const expandX = Math.max(w * 0.10, 20); 
            const expandY = Math.max(h * 0.30, 20);
            outerX = Math.max(0, x - expandX);
            outerY = Math.max(0, y - expandY);
            outerW = Math.min(canvas.width - outerX, w + expandX * 2);
            outerH = Math.min(canvas.height - outerY, h + expandY * 2);
            ctx.filter = 'blur(10px)';
        }
        
        // Solid white mask to ensure model replaces the entire area completely
        ctx.fillStyle = 'white';
        ctx.fillRect(outerX, outerY, outerW, outerH);
        
        const featheredMaskUrl = canvas.toDataURL('image/png');
        console.log(`[cleanBackground] Mask generated for ${layer.name}`);

        const prompt = `
            [TASK: CLEAN PLATE / OBJECT REMOVAL]
            Target Object to Erase: "${layer.name}" (Hint: ${layer.promptHint || 'None'})
            
            CRITICAL CONSTRAINTS:
            1. COMPLETE ANNIHILATION: You MUST completely erase EVERY SINGLE TRACE of the object within the masked area. Leave NO ghosting, reflections, or shadows behind.
            2. SEAMLESS INPAINTING: Fill the erased areas by intelligently extending the surrounding textures, patterns, and lighting flawlessly.
            3. NO NEW OBJECTS: Do NOT introduce any new objects, subjects, or focal points. The erased region must look like a perfectly clean, empty background.
            4. PRESERVE STRUCTURE: Maintain the integrity of any continuous background structures (e.g., walls, floors, horizons) passing behind the object.
            
            Return the full image with the object utterly removed and background perfectly healed.
        `;

        const shouldUseQwen = shouldUseQwenSemanticCleanPlate([layer]);
        const result = shouldUseQwen
            ? await callExperimentalCleanPlateRouter(
                areAllTextCleanupLayers([layer])
                    ? {
                        mode: 'clean_plate_test_edit',
                        prompt: `[CLEAN_BACKGROUND] ${buildQwenSemanticCleanPlatePrompt([layer], layer.promptHint || '')}`,
                        image: baseBackgroundUrl,
                        num_inference_steps: 40,
                        cfg: 4
                    }
                    : {
                        mode: 'clean_plate_test_edit',
                        prompt: `[CLEAN_BACKGROUND] ${buildQwenObjectRemovalPrompt([layer], layer.promptHint || '')}`,
                        image: baseBackgroundUrl,
                        num_inference_steps: 40,
                        cfg: 4
                    }
            )
            : await editOrQueryImageWithGemini(
                `[CLEAN_BACKGROUND] ${prompt}`,
                baseBackgroundUrl,
                [],
                featheredMaskUrl,
                null,
                false // forceMaterialTask = false for removal
            );

        if (result && result.success && result.imageData) {
            console.log(`[cleanBackground] Successfully cleaned background for ${layer.name}`);
            return `data:${result.mimeType};base64,${result.imageData}`;
        } else {
            console.error("[cleanBackground] Gemini Clean Plate Failed:", result);
            return null;
        }
    } catch (error) {
        console.error("[cleanBackground] failed:", error);
        return null;
    }
}

async function generateMask(baseImageUrl, layer) {
    try {
        const img = await loadImage(baseImageUrl);
        if (!img) return null;

        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        // Clear canvas (transparent)
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw white rectangle for the bbox (opaque in mask)
        const [ymin, xmin, ymax, xmax] = layer.bbox;
        const x = (xmin / 1000) * canvas.width;
        const y = (ymin / 1000) * canvas.height;
        const w = ((xmax - xmin) / 1000) * canvas.width;
        const h = ((ymax - ymin) / 1000) * canvas.height;
        
        ctx.fillStyle = 'white';
        ctx.fillRect(x, y, w, h);
        
        return canvas.toDataURL('image/png');
    } catch (error) {
        console.error("generateMask failed:", error);
        return null;
    }
}

async function generateCutout(baseImageUrl, maskUrl, bbox) {
    // Apply the mask to the base image to create a transparent cutout
    try {
        const [img, mask] = await Promise.all([
            loadImage(baseImageUrl),
            loadImage(maskUrl)
        ]);
        
        if (!img || !mask) return null;

        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        // Draw original image
        ctx.drawImage(img, 0, 0);
        
        // Apply mask using globalCompositeOperation
        ctx.globalCompositeOperation = 'destination-in';
        // V2.4: 确保 Mask 缩放到与图片一致的尺寸，防止尺寸不匹配导致的透明裁剪
        ctx.drawImage(mask, 0, 0, canvas.width, canvas.height);
        
        // Crop to bbox to save memory and make it a true "asset"
        const [ymin, xmin, ymax, xmax] = bbox;
        
        // Use precise pixel coordinates
        const x = (xmin / 1000) * img.width;
        const y = (ymin / 1000) * img.height;
        const w = ((xmax - xmin) / 1000) * img.width;
        const h = ((ymax - ymin) / 1000) * img.height;
        
        // Create a canvas that matches the BBox dimensions exactly
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = Math.max(1, Math.round(w));
        cropCanvas.height = Math.max(1, Math.round(h));
        const cropCtx = cropCanvas.getContext('2d');
        
        // Draw the masked portion into the crop canvas, stretching/shrinking slightly to fit if needed
        // but since we calculated w/h from the same img.width/height, it should be a perfect crop.
        cropCtx.drawImage(canvas, x, y, w, h, 0, 0, cropCanvas.width, cropCanvas.height);
        
        return cropCanvas.toDataURL('image/png');
    } catch (error) {
        console.error("generateCutout failed:", error);
        return null;
    }
}

async function extractFusionCutout(fullImageDataUrl, backgroundDataUrl, bbox) {
    try {
        const [fullImg, bgImg] = await Promise.all([
            loadImage(fullImageDataUrl),
            loadImage(backgroundDataUrl)
        ]);
        
        if (!fullImg || !bgImg) return null;

        const [ymin, xmin, ymax, xmax] = bbox;
        
        // Calculate crop for fullImg
        const fx = (xmin / 1000) * fullImg.width;
        const fy = (ymin / 1000) * fullImg.height;
        const fw = ((xmax - xmin) / 1000) * fullImg.width;
        const fh = ((ymax - ymin) / 1000) * fullImg.height;

        // Calculate crop for bgImg
        const bx = (xmin / 1000) * bgImg.width;
        const by = (ymin / 1000) * bgImg.height;
        const bw = ((xmax - xmin) / 1000) * bgImg.width;
        const bh = ((ymax - ymin) / 1000) * bgImg.height;

        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(fw));
        canvas.height = Math.max(1, Math.round(fh));
        const ctx = canvas.getContext('2d');
        
        // Draw AI generated image
        ctx.drawImage(fullImg, fx, fy, fw, fh, 0, 0, canvas.width, canvas.height);
        const fullData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Draw background image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(bgImg, bx, by, bw, bh, 0, 0, canvas.width, canvas.height);
        const bgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < fullData.data.length; i += 4) {
            const rDiff = Math.abs(fullData.data[i] - bgData.data[i]);
            const gDiff = Math.abs(fullData.data[i + 1] - bgData.data[i + 1]);
            const bDiff = Math.abs(fullData.data[i + 2] - bgData.data[i + 2]);
            const diff = rDiff + gDiff + bDiff;

            const threshold = 30; 
            if (diff < threshold) {
                fullData.data[i + 3] = 0; 
            } else {
                const alpha = Math.min(255, ((diff - threshold) / 50) * 255);
                fullData.data[i + 3] = Math.floor(alpha);
            }
        }

        ctx.putImageData(fullData, 0, 0);
        return canvas.toDataURL('image/png');
    } catch (error) {
        console.error("extractFusionCutout failed:", error);
        return null;
    }
}

/**
 * 场景合成：将背景和图层按照当前位置、缩放、旋转绘制到 Canvas
 */
async function composeScene(item, layerIndex) {
    const layer = item.scene.layers[layerIndex];
    const bgImg = await loadImage(item.cleanPlateDataUrl || item.originalDataUrl || item.dataUrl);
    const layerImg = await loadImage(layer.cutoutUrl);
    
    if (!bgImg || !layerImg) return null;

    const canvas = document.createElement('canvas');
    canvas.width = bgImg.width;
    canvas.height = bgImg.height;
    const ctx = canvas.getContext('2d');
    
    // 1. 绘制背景
    ctx.drawImage(bgImg, 0, 0);
    
    // 2. 绘制图层 (需要根据 layer.bbox 或当前 UI 状态计算位置/缩放)
    // 假设 layer 包含当前在画布上的变换信息
    const [ymin, xmin, ymax, xmax] = layer.bbox;
    const x = (xmin / 1000) * bgImg.width;
    const y = (ymin / 1000) * bgImg.height;
    const w = ((xmax - xmin) / 1000) * bgImg.width;
    const h = ((ymax - ymin) / 1000) * bgImg.height;
    
    ctx.drawImage(layerImg, x, y, w, h);
    
    return canvas.toDataURL('image/png');
}

/**
 * AI 深度融合：调用 Gemini 进行场景渲染
 */
export async function fuseSceneWithAI(itemId, layerIndex, prompt) {
    const item = state.workbenchItems.get(itemId);
    if (!item) return;
    
    const sceneDataUrl = await composeScene(item, layerIndex);
    if (!sceneDataUrl) return;

    const sceneFile = await dataURLToFile(sceneDataUrl, `scene-fusion-${Date.now()}.png`);
    
    const fusionPrompt = `这是一张用户自由排版的构图草图（Composition Sketch）。请将其转化为一张高度逼真、光影和谐的最终图像。必须严格遵循草图中物体的空间位置、大小比例、层级遮挡关系。你需要统一全局的光照、阴影、透视关系，并完美融合物体边缘。 ${prompt}`;
    
    const result = await editOrQueryImageWithGemini(fusionPrompt, sceneFile);
    
    if (result && result.success && result.imageData) {
        const fusedDataUrl = `data:${result.mimeType};base64,${result.imageData}`;
        
        // 放置新资产
        // 查找空白位置 (简单实现：在当前 item 右侧)
        const newX = parseFloat(item.el.style.left) + parseFloat(item.el.style.width) + 50;
        const newY = parseFloat(item.el.style.top);
        
        // 假设有个全局函数 addImageToWorkbench
        window.addImageToWorkbench(await dataURLToFile(fusedDataUrl, 'fused-result.png'), '融合结果', {
            x: newX,
            y: newY,
            type: 'fused-asset'
        });
    }
}
