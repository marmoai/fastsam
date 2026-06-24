import { state } from '../../core/state.js';
import { dbHelper } from '../../core/session.js';
import { checkProximity } from '../injection-engine.js';
import { startOutpaintDrag } from '../outpaint-interaction.js';
import { uploadImageToOSS } from '../../services/ossService.js';
import { fileToDataURL, dataURLToFile, getProxiedUrl } from '../../core/utils.js';

const { workbenchItems, selectedWorkbenchItems, fileToWorkbenchIdMap, isCtrlPressed } = state;
const pendingWorkbenchUploads = new Map();

function beginPendingWorkbenchUpload(itemId, source = 'unknown') {
    const token = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    pendingWorkbenchUploads.set(itemId, {
        token,
        source,
        startedAt: Date.now()
    });
    return token;
}

function isPendingWorkbenchUploadActive(itemId, token) {
    const task = pendingWorkbenchUploads.get(itemId);
    return !!task && task.token === token;
}

function finishPendingWorkbenchUpload(itemId, token) {
    const task = pendingWorkbenchUploads.get(itemId);
    if (task && task.token === token) {
        pendingWorkbenchUploads.delete(itemId);
    }
}

export function invalidatePendingWorkbenchUpload(itemId, reason = 'invalidated') {
    const task = pendingWorkbenchUploads.get(itemId);
    if (!task) return;
    pendingWorkbenchUploads.delete(itemId);
    console.log(`[WorkbenchUpload] Invalidated pending upload for ${itemId} (${reason})`);
}

export function invalidateAllPendingWorkbenchUploads(reason = 'reset') {
    if (pendingWorkbenchUploads.size === 0) return;
    pendingWorkbenchUploads.forEach((_, itemId) => {
        invalidatePendingWorkbenchUpload(itemId, reason);
    });
}

window.invalidatePendingWorkbenchUpload = invalidatePendingWorkbenchUpload;
window.invalidateAllPendingWorkbenchUploads = invalidateAllPendingWorkbenchUploads;

// --- Global Sync Function to ensure DOM matches State ---
window.syncDOMToScene = (itemId) => {
    if (!window.mvrRuntime) return;
    const workspace = window.mvrRuntime.getCurrentWorkspace();
    if (!workspace) return;
    
    const item = workbenchItems.get(itemId);
    const asset = workspace.currentState.assetRegistry.get(itemId);
    if (item && item.el && asset && asset.transform) {
        const t = asset.transform;
        // The single source of truth for DOM position/size is the MVR Runtime Asset
        item.el.style.left = `${t.x}px`;
        item.el.style.top = `${t.y}px`;
        item.el.style.width = `${t.width}px`;
        item.el.style.height = `${t.height}px`;
        item.el.style.zIndex = t.zIndex;
        item.el.style.transform = `rotate(${t.rotation}deg)`;
    }
};

export async function deleteSelectedItems(skipConfirm = false) {
    if (selectedWorkbenchItems.size === 0) return;
    
    let confirmed = skipConfirm;
    if (!confirmed) {
        confirmed = await window.showCustomConfirm(`确定要删除选中的 ${selectedWorkbenchItems.size} 个项目吗？`);
    }
    
    if (confirmed) {
        const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
        const uidsToRemove = [];
        const itemsToRemove = new Set();

        const collectItemAndChildren = (id) => {
            if (itemsToRemove.has(id)) return;
            itemsToRemove.add(id);
            workbenchItems.forEach((childItem, childId) => {
                if (childItem.parentId === id) {
                    collectItemAndChildren(childId);
                }
            });
        };

        selectedWorkbenchItems.forEach(id => collectItemAndChildren(id));

        itemsToRemove.forEach(id => {
            const item = workbenchItems.get(id);
            if (item && item.el) {
                item.el.remove();
            }
            invalidatePendingWorkbenchUpload(id, 'delete_selected_items');
            if (workspace) {
                const hasAsset = workspace.currentState.assetRegistry.get(id);
                if (hasAsset) {
                    uidsToRemove.push(id);
                }
            }
            // Clear from fileToWorkbenchIdMap
            fileToWorkbenchIdMap.forEach((val, key) => {
                if (val === id) {
                    fileToWorkbenchIdMap.delete(key);
                }
            });
            workbenchItems.delete(id);
            selectedWorkbenchItems.delete(id);
        });

        if (workspace && uidsToRemove.length > 0) {
            workspace.dispatcher.dispatch({
                type: 'REMOVE_ASSETS',
                payload: { uids: uidsToRemove }
            });
        }

        itemsToRemove.forEach(id => {
            if (window.currentAdjustingShape && window.currentAdjustingShape.dataset.itemId === id) {
                const shapeAdjuster = document.getElementById('shapeAdjuster');
                if (shapeAdjuster) shapeAdjuster.style.display = 'none';
                window.currentAdjustingShape = null;
            }
            if (state.currentActiveWorkbenchItemId === id) {
                if (typeof window.closeLayerManagerModal === 'function') {
                    window.closeLayerManagerModal();
                }
            }
        });

        if (window.checkProximity) window.checkProximity();
        if (window.drawGenealogyConnections) window.drawGenealogyConnections();

        if (workbenchItems.size === 0) {
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-workbench-state';
            emptyState.innerHTML = `
                <i class="fas fa-image"></i>
                <p>上传或生成的图片将出现在这里</p>
                <p style="font-size: 12px; margin-top: 10px;">拖拽图片重叠可触发融合反应</p>
            `;
            document.getElementById('workbench').appendChild(emptyState);
        }

        if (window.updateSelectedItems) window.updateSelectedItems();
        if (window.historyManager) window.historyManager.pushState();
    }
}


export function selectWorkbenchItem(id) {
    const item = workbenchItems.get(id);
    if (!item) return;
    
    if (!isCtrlPressed) {
        // 如果不是多选模式，清除所有选中状态
        document.querySelectorAll('.workbench-item').forEach(el => el.classList.remove('selected'));
        selectedWorkbenchItems.clear();
    }
    
    // 切换当前项的选中状态
    const itemEl = document.querySelector(`.workbench-item[data-item-id="${id}"]`);
    if (itemEl.classList.contains('selected')) {
        itemEl.classList.remove('selected');
        selectedWorkbenchItems.delete(id);
    } else {
        itemEl.classList.add('selected');
        selectedWorkbenchItems.add(id);
    }
    
    // 处理选中项
    checkProximity();
}

export async function addImageToWorkbench(file, label = '', metadata = {}) {
    const id = metadata.id || `item-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    let finalDataUrl = metadata.dataUrl;
    const shouldDeferBackgroundUpload = !!(
        metadata.skipBackgroundUpload ||
        window.isRestoringSession ||
        window.isInitializingAppRestore ||
        (window.historyManager && window.historyManager.isRestoring)
    );
    
    // 如果有本地文件，先使用本地 URL，后台异步上传到 OSS
    if (file) {
        finalDataUrl = await fileToDataURL(file);
        fileToWorkbenchIdMap.set(file, id);
        fileToWorkbenchIdMap.set(finalDataUrl, id);
        
        if (!shouldDeferBackgroundUpload) {
            const uploadToken = beginPendingWorkbenchUpload(id, 'file');
            uploadImageToOSS(file).then(url => {
                if (!isPendingWorkbenchUploadActive(id, uploadToken)) {
                    return;
                }
                const item = workbenchItems.get(id);
                if (item) {
                    item.dataUrl = url;
                    item.originalDataUrl = url; // V2.5
                    const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
                    const asset = workspace ? workspace.currentState.assetRegistry.get(id) : null;
                    if (workspace && asset) {
                        workspace.dispatcher.dispatch({
                            type: 'UPDATE_ASSET_METADATA',
                            payload: {
                                uid: id,
                                sourceImage: url,
                                originalDataUrl: url,
                                cleanPlateDataUrl: asset.cleanPlateDataUrl ?? item.cleanPlateDataUrl ?? null,
                                cleanPlateStatus: asset.cleanPlateStatus ?? item.cleanPlateStatus ?? 'idle'
                            }
                        });
                    }
                    fileToWorkbenchIdMap.set(url, id);
                    if (dbHelper) dbHelper.saveImageCache(url, file);
                    if (window.historyManager) window.historyManager.pushState();
                }
            }).catch(e => {
                if (isPendingWorkbenchUploadActive(id, uploadToken)) {
                    console.error('Background upload failed:', e);
                }
            }).finally(() => {
                finishPendingWorkbenchUpload(id, uploadToken);
            });
        }
        
    } else if (metadata.dataUrl && metadata.dataUrl.startsWith('data:')) {
        // 如果是 base64 数据（如 AI 生成），先使用本地 URL，后台异步上传到 OSS
        finalDataUrl = metadata.dataUrl;
        fileToWorkbenchIdMap.set(finalDataUrl, id);
        
        if (!shouldDeferBackgroundUpload) {
            const uploadToken = beginPendingWorkbenchUpload(id, 'data_url');
            dataURLToFile(metadata.dataUrl, `gen_${Date.now()}.png`).then(blobFile => {
                if (!blobFile || !isPendingWorkbenchUploadActive(id, uploadToken)) {
                    return null;
                }
                return uploadImageToOSS(blobFile).then(url => {
                    if (!isPendingWorkbenchUploadActive(id, uploadToken)) {
                        return;
                    }
                    const item = workbenchItems.get(id);
                    if (item) {
                        item.dataUrl = url;
                        item.originalDataUrl = url; // V2.5
                        const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
                        const asset = workspace ? workspace.currentState.assetRegistry.get(id) : null;
                        if (workspace && asset) {
                            workspace.dispatcher.dispatch({
                                type: 'UPDATE_ASSET_METADATA',
                                payload: {
                                    uid: id,
                                    sourceImage: url,
                                    originalDataUrl: url,
                                    cleanPlateDataUrl: asset.cleanPlateDataUrl ?? item.cleanPlateDataUrl ?? null,
                                    cleanPlateStatus: asset.cleanPlateStatus ?? item.cleanPlateStatus ?? 'idle'
                                }
                            });
                        }
                        fileToWorkbenchIdMap.set(url, id);
                        if (dbHelper) dbHelper.saveImageCache(url, blobFile);
                        if (window.historyManager) window.historyManager.pushState();
                    }
                });
            }).catch(e => {
                if (isPendingWorkbenchUploadActive(id, uploadToken)) {
                    console.error('Background upload failed:', e);
                }
            }).finally(() => {
                finishPendingWorkbenchUpload(id, uploadToken);
            });
        }
        
    } else if (metadata.dataUrl) {
        // Already a URL
        fileToWorkbenchIdMap.set(metadata.dataUrl, id);
    }

    // --- NEW: Cache Handling ---
    let displayUrl = finalDataUrl;
    
    // If we have a local file, use it directly for display (using base64 for better stability)
    if (file) {
        try {
            displayUrl = await fileToDataURL(file);
        } catch (e) {
            console.warn('Failed to convert file to data URL, using blob URL as fallback', e);
            displayUrl = URL.createObjectURL(file);
        }
    } else if (finalDataUrl && finalDataUrl.startsWith('http') && dbHelper) {
        // If we have a cloud URL but no file, try to get from cache
        const cachedBlob = await dbHelper.getImageCache(finalDataUrl);
        if (cachedBlob) {
            try {
                displayUrl = await fileToDataURL(cachedBlob);
            } catch (e) {
                console.warn('Failed to convert cached blob to data URL, using blob URL as fallback', e);
                displayUrl = URL.createObjectURL(cachedBlob);
            }
        } else {
            // Not in cache, download and save to cache
            try {
                const response = await fetch(getProxiedUrl(finalDataUrl));
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.startsWith('image/')) {
                    throw new Error(`Invalid content type: ${contentType}`);
                }
                const blob = await response.blob();
                await dbHelper.saveImageCache(finalDataUrl, blob);
                try {
                    displayUrl = await fileToDataURL(blob);
                } catch (e) {
                    console.warn('Failed to convert fetched blob to data URL, using blob URL as fallback', e);
                    displayUrl = URL.createObjectURL(blob);
                }
            } catch (e) {
                console.warn('Failed to cache image:', finalDataUrl, e);
            }
        }
    }
    
    // If it's a new upload (file + cloud URL), ensure it's in cache
    if (file && finalDataUrl && finalDataUrl.startsWith('http') && dbHelper) {
        dbHelper.saveImageCache(finalDataUrl, file);
    }

    const dataUrlPromise = displayUrl ? Promise.resolve(displayUrl) : (file ? fileToDataURL(file) : Promise.resolve(''));

    return dataUrlPromise.then(dataUrl => {
        return new Promise((resolve) => {
            const img = new Image();
            
            const tryLoad = (useProxy, useCrossOrigin) => {
                if (useCrossOrigin && !dataUrl.startsWith('blob:')) {
                    img.crossOrigin = "anonymous";
                } else {
                    img.removeAttribute('crossOrigin');
                }
                
                const src = (useProxy && dataUrl && dataUrl.startsWith('http') && !dataUrl.startsWith('blob:')) 
                    ? getProxiedUrl(dataUrl) 
                    : dataUrl;

                const safeSourceImage = src && src !== 'undefined' ? src : (finalDataUrl && finalDataUrl !== 'undefined' ? finalDataUrl : dataUrl);
                
                img.onload = () => {
                    let finalWidth, finalHeight;
                    if (metadata.initialWidth && metadata.initialHeight) {
                        finalWidth = metadata.initialWidth;
                        finalHeight = metadata.initialHeight;
                    } else {
                        const maxSize = 600;
                        let width = img.naturalWidth;
                        let height = img.naturalHeight;
                        const ratio = Math.min(maxSize / width, maxSize / height);
                        finalWidth = width * ratio;
                        finalHeight = height * ratio;
                    }

                    let posX, posY;
                    if (metadata.left !== undefined && metadata.top !== undefined) {
                        posX = parseFloat(metadata.left);
                        posY = parseFloat(metadata.top);
                    } else if (metadata.x !== undefined && metadata.y !== undefined) {
                        posX = metadata.x;
                        posY = metadata.y;
                    } else {
                        const pos = typeof window.findNextPosition === 'function'
                            ? window.findNextPosition(finalWidth, finalHeight)
                            : null;
                        posX = pos?.x ?? 100;
                        posY = pos?.y ?? 100;
                    }

                    const rawGenealogy = metadata.genealogy && typeof metadata.genealogy === 'object' ? metadata.genealogy : {};
                    const genealogy = {
                        parents: Array.isArray(rawGenealogy.parents) ? [...rawGenealogy.parents] : [],
                        prompt: rawGenealogy.prompt || metadata.generationParams?.prompt || '',
                        action: rawGenealogy.action || metadata.type || 'upload'
                    };
                    if (metadata.parentId && !genealogy.parents.includes(metadata.parentId)) genealogy.parents.push(metadata.parentId);
                    if (metadata.styleParentId && !genealogy.parents.includes(metadata.styleParentId)) genealogy.parents.push(metadata.styleParentId);

                    // Sync to MVR AssetRuntime FIRST
                    let asset = null;
                    let createdNewAsset = false;
                    const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
                    if (workspace) {
                        const existingAsset = workspace.currentState.assetRegistry.get(id);
                        if (existingAsset) {
                            asset = existingAsset;
                        } else {
                            createdNewAsset = true;
                            asset = {
                                uid: id,
                                type: metadata.type || 'unknown',
                                sourceImage: safeSourceImage,
                                masks: [],
                                variants: [],
                                metadata: {
                                    createdAt: Date.now(),
                                    updatedAt: Date.now(),
                                    creatorId: 'local_user',
                                    usageCount: 1,
                                    tags: [label || 'imported']
                                },
                                fusionProperties: metadata.fusionProperties || {
                                    brightness: 100,
                                    contrast: 100,
                                    saturation: 100,
                                    blur: 0
                                },
                                transform: {
                                    x: posX,
                                    y: posY,
                                    width: finalWidth,
                                    height: finalHeight,
                                    rotation: 0,
                                    zIndex: parseInt(metadata.zIndex || 0)
                                },
                                genealogy,
                                parentId: metadata.parentId || null,
                                layerName: metadata.layerName,
                                originalBbox: metadata.originalBbox,
                                layers: metadata.layers,
                                scene: metadata.scene,
                                semanticViews: metadata.semanticViews,
                                hasFullSemanticAnalysis: metadata.hasFullSemanticAnalysis,
                                originalDataUrl: metadata.originalDataUrl || finalDataUrl,
                                cleanPlateDataUrl: metadata.cleanPlateDataUrl || null,
                                cleanPlateStatus: metadata.cleanPlateStatus || 'idle'
                            };

                            const relations = genealogy.parents.map(parentId => ({
                                id: `edge_${parentId}_${id}`,
                                sourceId: parentId,
                                targetId: id,
                                relationType: 'parent_of',
                                properties: { action: genealogy.action, prompt: genealogy.prompt }
                            }));

                            workspace.dispatcher.dispatch({
                                type: relations.length > 0 ? 'ADD_ASSET_WITH_RELATIONS' : 'ADD_ASSET',
                                payload: relations.length > 0 ? { asset, relations } : { asset }
                            });
                        }
                    }

                    const assetTransform = asset?.transform || {};
                    const actualX = assetTransform.x ?? posX;
                    const actualY = assetTransform.y ?? posY;
                    const actualWidth = assetTransform.width ?? finalWidth;
                    const actualHeight = assetTransform.height ?? finalHeight;
                    const actualZIndex = assetTransform.zIndex ?? parseInt(metadata.zIndex || 0);

                    const wrapper = document.createElement('div');
                    wrapper.className = 'workbench-item';
                    if (metadata.className) {
                        wrapper.className += ` ${metadata.className}`;
                    }
                    wrapper.dataset.itemId = id;
                    wrapper.dataset.aspectRatio = actualWidth / actualHeight;

                    wrapper.style.left = `${actualX}px`;
                    wrapper.style.top = `${actualY}px`;
                    wrapper.style.width = `${actualWidth}px`;
                    wrapper.style.height = `${actualHeight}px`;
                    wrapper.style.zIndex = actualZIndex;

                    workbenchItems.set(id, {
                        file, 
                        el: wrapper, 
                        ...metadata, 
                        dataUrl: finalDataUrl,
                        originalDataUrl: metadata.originalDataUrl || finalDataUrl, // V2.5
                        cleanPlateDataUrl: metadata.cleanPlateDataUrl || null,       // V2.5
                        cleanPlateStatus: metadata.cleanPlateStatus || 'idle',      // V2.5
                        semanticViews: metadata.semanticViews || null,
                        schemes: [{ id: 'scheme-1', name: '方案 1', layerVersions: {} }],
                        activeSchemeId: 'scheme-1',
                        spawnWidth: actualWidth,
                        spawnHeight: actualHeight,
                        spawnLeft: actualX,
                        spawnTop: actualY
                    });

                    const addedItem = workbenchItems.get(id);
                    const shouldAutoOpenDecisionPanel =
                        metadata.autoOpenDecisionPanel === true ||
                        (typeof window.isExtractedLayerItem === 'function' && window.isExtractedLayerItem(addedItem));

                    const batchAutoOpenToken = metadata.autoOpenDecisionPanelBatchToken;
                    const shouldOpenForThisBatch =
                        !batchAutoOpenToken ||
                        metadata.autoOpenDecisionPanelBatchFinal === true ||
                        window.__lastDecisionPanelBatchToken !== batchAutoOpenToken;

                    if (shouldAutoOpenDecisionPanel && shouldOpenForThisBatch && typeof window.showFloatingFusionEditor === 'function') {
                        if (batchAutoOpenToken) {
                            if (metadata.autoOpenDecisionPanelBatchFinal !== true) {
                                window.__lastDecisionPanelBatchToken = batchAutoOpenToken;
                            } else {
                                window.__lastDecisionPanelBatchToken = null;
                            }
                        }
                        requestAnimationFrame(() => {
                            state.currentActiveWorkbenchItemId = id;
                            window.showFloatingFusionEditor(id);
                        });
                    }

                    if (asset && workspace && (
                        metadata.left !== undefined ||
                        metadata.top !== undefined ||
                        metadata.x !== undefined ||
                        metadata.y !== undefined ||
                        metadata.initialWidth !== undefined ||
                        metadata.initialHeight !== undefined ||
                        metadata.zIndex !== undefined
                    )) {
                        workspace.currentState.assetRegistry.updateAssetTransform(id, {
                            x: actualX,
                            y: actualY,
                            width: actualWidth,
                            height: actualHeight,
                            zIndex: actualZIndex
                        });
                    }

                    window.workbenchGrid.appendChild(wrapper);
                    document.querySelector('#workbench > .empty-workbench-state')?.remove();
                    window.workbenchItemCount = (window.workbenchItemCount || 0) + 1;

                    if (window.implicitMemoryEngine) {
                        try {
                            const promptText = metadata?.generationParams?.prompt || label || '';
                            window.implicitMemoryEngine.recordAssetAdded(id, promptText, metadata?.fusionProperties || (asset ? asset.fusionProperties : null));
                        } catch (err) {
                            console.error('[Telemetry] Error recording added asset:', err);
                        }
                    }

                    resolve({ dataUrl: safeSourceImage, width: actualWidth, height: actualHeight, wrapper, genealogy, createdNewAsset });
                };

                img.onerror = (e) => {
                    if (useProxy || useCrossOrigin) {
                        console.warn('Failed to load image with proxy/CORS, retrying without...', src);
                        tryLoad(false, false);
                    } else {
                        console.error('Failed to load image for workbench even without proxy:', dataUrl, 'Error:', e);
                        resolve({ dataUrl, width: 200, height: 200, wrapper: null, genealogy: null, createdNewAsset: false });
                    }
                };

                img.src = src;
            };

            tryLoad(true, true);
        });
    }).then(({ dataUrl, width, height, wrapper, genealogy, createdNewAsset }) => {
        if (!wrapper) return id;

        // --- Sync function removed to global scope ---
        if (window.mvrRuntime && window.syncDOMToScene) {
            window.syncDOMToScene(id);
        }
        // ------------------------------------
        
        // --- Sync to MVR SceneGraph Edges ---
        if (window.mvrRuntime && !createdNewAsset) {
            const workspace = window.mvrRuntime.getCurrentWorkspace();
                    if (workspace && workspace.currentState && genealogy) {
                const asset = workspace.currentState.assetRegistry.get(id);
                if (asset && Array.isArray(genealogy.parents) && genealogy.parents.length > 0) {
                    const existingEdges = genealogy.parents.every(parentId =>
                        workspace.currentState.sceneGraph.getEdgesForNode(id).some(edge =>
                            edge.sourceId === parentId &&
                            edge.targetId === id &&
                            edge.relationType === 'parent_of'
                        )
                    );

                    if (!existingEdges) {
                        const edges = genealogy.parents
                            .filter(parentId => workspace.currentState.assetRegistry.get(parentId))
                            .map(parentId => ({
                            id: `edge_${parentId}_${id}`,
                            sourceId: parentId,
                            targetId: id,
                            relationType: 'parent_of',
                            properties: { action: genealogy.action, prompt: genealogy.prompt }
                        }));

                        if (edges.length > 0) {
                            workspace.dispatcher.dispatch({
                                type: 'ADD_RELATIONS',
                                payload: { edges }
                            });
                        }
                    }
                }
            }
        }
        // ------------------------------------

        const currentItem = workbenchItems.get(id);
        workbenchItems.set(id, { 
            ...currentItem, 
            dataUrl, 
            originalDataUrl: currentItem.originalDataUrl || dataUrl,
            cleanPlateDataUrl: currentItem.cleanPlateDataUrl,
            cleanPlateStatus: currentItem.cleanPlateStatus || 'idle',
            genealogy: genealogy 
        });
        
        const renderDataUrl = currentItem.cleanPlateDataUrl || dataUrl;
        wrapper.innerHTML = `<div class="crop-container" style="width: 100%; height: 100%; overflow: hidden; position: relative;"><img src="${renderDataUrl}" alt="${label}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></div>`;
        
        ['nw', 'ne', 'sw', 'se'].forEach(dir => {
            const handle = document.createElement('div');
            handle.className = `resize-handle resize-${dir}`;
            handle.addEventListener('mousedown', (e) => {
                if (e.button !== 0 || state.isSpacePressed) return; 
                window.handleResizeStart(e, dir, wrapper);
            });
            wrapper.appendChild(handle);
        });

        // Only add outpaint handles for regular images (not exploded layers or other types)
        if (metadata.type !== 'layer-explode' && metadata.type !== 'isolated-edit' && !['text-note', 'group-label', 'shape', 'atmosphere'].includes(metadata.type)) {
            const directions = [
                { dir: 'top', icon: 'fas fa-arrow-up' },
                { dir: 'bottom', icon: 'fas fa-arrow-down' },
                { dir: 'left', icon: 'fas fa-arrow-left' },
                { dir: 'right', icon: 'fas fa-arrow-right' }
            ];
            directions.forEach(d => {
                const handle = document.createElement('div');
                handle.className = `outpaint-handle handle-${d.dir}`;
                handle.innerHTML = `<i class="${d.icon}"></i>`;
                handle.title = '按住拖拽以延展画面';
                handle.onmousedown = (e) => {
                    if (e.button !== 0 || state.isSpacePressed) return; 
                    e.stopPropagation();
                    startOutpaintDrag(e, id, d.dir, wrapper);
                };
                wrapper.appendChild(handle);
            });
        }

        // --- H. [新增] AI 创意建议与图层编辑魔力 Spark 触发器 ---
        const spark = document.createElement('div');
        spark.className = 'ai-spark-trigger';
        spark.style.display = 'none'; // 移除残留的闪电图标，完全转移到灵感舱和时光机控制中
        spark.title = 'AI 创意智能建议与编辑';
        spark.innerHTML = '<i class="fas fa-bolt"></i>';
        
        spark.addEventListener('mousedown', (e) => {
            e.stopPropagation();
        });
        
        spark.addEventListener('click', (e) => {
            e.stopPropagation();
            // 隐藏工具箱避免重叠遮挡
            const tb = document.getElementById('workbenchToolbox');
            if (tb) tb.style.display = 'none';
            
            if (typeof window.showFloatingFusionEditor === 'function') {
                window.showFloatingFusionEditor(id);
            }
        });
        wrapper.appendChild(spark);

        const inverseScale = 1 / state.workbenchZoom;
        wrapper.querySelectorAll('.resize-handle').forEach(h => h.style.transform = `scale(${inverseScale})`);
        wrapper.querySelectorAll('.outpaint-handle').forEach(h => {
            if (h.classList.contains('handle-top') || h.classList.contains('handle-bottom')) h.style.transform = `translateX(-50%) scale(${inverseScale})`;
            else h.style.transform = `translateY(-50%) scale(${inverseScale})`;
        });
        if (spark) {
            spark.style.transform = `scale(${inverseScale})`;
        }
        
        wrapper.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('resize-handle')) return;
            if (e.button !== 0 || e.target.classList.contains('outpaint-handle')) return;
            window.handleWorkbenchDragStart(e);
        });

        wrapper.addEventListener('click', (event) => {
            if (event.target.closest('.sticky-note')) return;
            event.stopPropagation();
            if (event.target.classList.contains('outpaint-handle')) return;
            if (event.target.classList.contains('resize-handle')) return;
            
            // 检测是否是提取出来的独立图层项目 (含有 parentId 且为特定提取类型)
            const itemObj = workbenchItems.get(id);
            if (itemObj) {
                const isExtractedLayer = itemObj.parentId && (
                    itemObj.type === 'layer-explode' || 
                    itemObj.type === 'layer-extract' || 
                    itemObj.type === 'isolated-edit' || 
                    itemObj.type === 'extraction'
                );
                if (isExtractedLayer) {
                    state.currentActiveWorkbenchItemId = id;
                    if (typeof window.showFloatingFusionEditor === 'function') {
                        window.showFloatingFusionEditor(id);
                    }
                    return;
                }
            }
            
            // 始终显示工具箱，因为它内部会根据类型判断是否显示 fusion editor
            window.showWorkbenchToolbox(id);
        });

        wrapper.addEventListener('wheel', (e) => {
            if (e.shiftKey && wrapper.matches(':hover')) {
                e.preventDefault();
                e.stopPropagation();
                
                const transform = wrapper.style.transform;
                let currentRotation = 0;
                const rotateMatch = transform.match(/rotate\(([-\d.]+)deg\)/);
                if (rotateMatch) {
                    currentRotation = parseFloat(rotateMatch[1]);
                }
                
                // Adjust rotation speed
                const delta = e.deltaY > 0 ? -1 : 1;
                const newRotation = currentRotation + delta * 5; // 5 degrees per wheel tick
                
                // Preserve scale
                let scaleMatch = transform.match(/scale\(([\d.]+)\)/);
                let scale = scaleMatch ? scaleMatch[1] : 1;
                
                // Update state
                if (window.mvrRuntime) {
                    const workspace = window.mvrRuntime.getCurrentWorkspace();
                    if (workspace && workspace.dispatcher) {
                        const asset = workspace.currentState.assetRegistry.get(id);
                        if (asset) {
                            const newTransform = { ...asset.transform, rotation: newRotation };
                            workspace.dispatcher.dispatch({ type: 'UPDATE_TRANSFORM', payload: { uid: id, transform: newTransform } });
                        }
                    }
                }
                
                // Update state if needed
            const item = workbenchItems.get(id);
            if (item && item.scene && item.scene.transform) {
                item.scene.transform.rotation = newRotation;
            }
                
                if (window.updateParentBBox) {
                    window.updateParentBBox(id);
                }
                if (window.historyManager) window.historyManager.pushState();
            }
        }, { passive: false });

        wrapper.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            if (window.enterCroppingMode) {
                window.enterCroppingMode(id);
            }
        });
        
        if (window.drawGenealogyConnections) {
            requestAnimationFrame(window.drawGenealogyConnections);
        }
        if (window.checkProximity) {
            window.checkProximity();
        }
        if (window.historyManager) window.historyManager.pushState();

        // 确保会话存在，且仅在非恢复状态下触发
        if (!window.isRestoringSession && !window.suppressSessionAutoCreate && !(window.historyManager && window.historyManager.isRestoring)) {
            if (typeof window.ensureSessionExists === 'function') {
                Promise.resolve(window.ensureSessionExists('新建项目'))
                    .then(async () => {
                        // 触发异步重命名
                        if (typeof window.triggerAsyncSessionRename === 'function') {
                            window.triggerAsyncSessionRename();
                        }

                        if (state.currentSessionId) {
                            const currentSession = window.state?.sessions?.find(s => s.id === state.currentSessionId);
                            if (currentSession && dbHelper) {
                                await dbHelper.saveSession(currentSession);
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Failed to persist uploaded workbench image:', error);
                    });
            } else {
                if (typeof window.triggerAsyncSessionRename === 'function') {
                    window.triggerAsyncSessionRename();
                }

                if (state.currentSessionId) {
                    const currentSession = window.state?.sessions?.find(s => s.id === state.currentSessionId);
                    if (currentSession && dbHelper) {
                        dbHelper.saveSession(currentSession).catch(error => {
                            console.error('Failed to persist uploaded workbench image:', error);
                        });
                    }
                }
            }
        }

        return id;
    });
}

export async function deleteWorkbenchItem(id, skipConfirm = false, isBatch = false) {
    const performDelete = () => {
        const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
        const uidsToRemove = [];
        const itemsToRemove = new Set();

        const collectItemAndChildren = (targetId) => {
            if (itemsToRemove.has(targetId)) return;
            itemsToRemove.add(targetId);
            workbenchItems.forEach((childItem, childId) => {
                if (childItem.parentId === targetId) {
                    collectItemAndChildren(childId);
                }
            });
        };

        collectItemAndChildren(id);

        itemsToRemove.forEach(targetId => {
            const item = workbenchItems.get(targetId);
            if (item && item.el) {
                item.el.remove();
            }
            invalidatePendingWorkbenchUpload(targetId, 'delete_workbench_item');
            if (workspace) {
                const hasAsset = workspace.currentState.assetRegistry.get(targetId);
                if (hasAsset) {
                    uidsToRemove.push(targetId);
                }
            }
            // Clear from fileToWorkbenchIdMap
            fileToWorkbenchIdMap.forEach((val, key) => {
                if (val === targetId) {
                    fileToWorkbenchIdMap.delete(key);
                }
            });
            workbenchItems.delete(targetId);
            selectedWorkbenchItems.delete(targetId);
        });

        if (workspace && uidsToRemove.length > 0) {
            workspace.dispatcher.dispatch({
                type: 'REMOVE_ASSETS',
                payload: { uids: uidsToRemove }
            });
        }

        itemsToRemove.forEach(targetId => {
            if (window.currentAdjustingShape && window.currentAdjustingShape.dataset.itemId === targetId) {
                const shapeAdjuster = document.getElementById('shapeAdjuster');
                if (shapeAdjuster) shapeAdjuster.style.display = 'none';
                window.currentAdjustingShape = null;
            }
            if (state.currentActiveWorkbenchItemId === targetId) {
                if (typeof window.closeLayerManagerModal === 'function') {
                    window.closeLayerManagerModal();
                }
            }
        });

        if (window.checkProximity) window.checkProximity(); 
        if (window.drawGenealogyConnections) window.drawGenealogyConnections();

        if (workbenchItems.size === 0) {
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-workbench-state';
            emptyState.innerHTML = `
                <i class="fas fa-image"></i>
                <p>上传或生成的图片将出现在这里</p>
                <p style="font-size: 12px; margin-top: 10px;">拖拽图片重叠可触发融合反应</p>
            `;
            document.getElementById('workbench').appendChild(emptyState);
        }
        if (!isBatch) {
            if (window.updateSelectedItems) window.updateSelectedItems();
            if (window.historyManager) window.historyManager.pushState();
        }
    };

    if (skipConfirm) {
        performDelete();
    } else {
        const confirmed = await window.showCustomConfirm('确定要删除这个项目吗？');
        if (confirmed) {
            performDelete();
        }
    }
}

export async function handleUploadImage() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;
    input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            // 模仿对话框上传逻辑：不手动设置位置，让 addImageToWorkbench 使用 findNextPosition 自动排列
            // 这样可以避免重叠，并且不会遮盖原有图片。使用 await 确保顺序排列。
            for (const file of files) {
                await addImageToWorkbench(file, file.name);
            }
        }
    };
    input.click();
}

export async function clearWorkbench(skipConfirm = false) {
    const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
    const hasRuntimeAssets = !!workspace && workspace.currentState.assetRegistry.getAll().length > 0;

    if (workbenchItems.size === 0 && !hasRuntimeAssets) return;
    
    let confirmed = skipConfirm;
    if (!confirmed) {
        confirmed = await window.showCustomConfirm('确定要清空工作台吗？所有未保存的内容都将丢失。');
    }
    
    if (confirmed) {
        if (state.currentSessionId) {
            window.__allowEmptyWorkbenchSessionSave = {
                sessionId: state.currentSessionId,
                expiresAt: Date.now() + 5000
            };
        }

        workbenchItems.forEach((item) => {
            if (item.el) item.el.remove();
        });
        workbenchItems.clear();
        selectedWorkbenchItems.clear();
        fileToWorkbenchIdMap.clear();
        state.workbenchItemCount = 0;
        window.workbenchItemCount = 0;
        
        if (typeof window.closeLayerManagerModal === 'function') {
            window.closeLayerManagerModal();
        }

        // Clear runtime state
        if (workspace) {
            workspace.dispatcher.dispatch({
                type: 'CLEAR_WORKSPACE',
                payload: {},
                meta: { silent: !!window.isRestoringSession }
            });
        }

        const emptyState = document.createElement('div');
        emptyState.className = 'empty-workbench-state';
        emptyState.innerHTML = `
            <i class="fas fa-image"></i>
            <p>上传或生成的图片将出现在这里</p>
            <p style="font-size: 12px; margin-top: 10px;">拖拽图片重叠可触发融合反应</p>
        `;
        document.getElementById('workbench').appendChild(emptyState);
        
        if (window.updateSelectedItems) window.updateSelectedItems();
        if (window.drawGenealogyConnections) window.drawGenealogyConnections();
        if (window.historyManager && !window.isRestoringSession) window.historyManager.pushState();
    }
}
