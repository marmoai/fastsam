import { state } from '../core/state.js';
import { createGroupLabel } from './workbench/notes.js';
import { autoOrganizeToGrid } from './workbench/layout.js';
import { startDrawingShape } from './workbench/shapes.js';
import { handleUploadImage } from './workbench/items.js';
import { editLayerAsset } from './workbench/layer-assets.js';
import localforage from 'localforage';
import { uploadImageToOSS } from '../services/ossService.js';
import { runtime } from '../runtime/CoreRuntime';

function isRuntimeManagedWorkbenchItem(item) {
    if (!item) return false;
    const type = item.type || 'image';
    return !['text-note', 'group-label', 'shape', 'atmosphere'].includes(type);
}

export function initContextMenu() {
    const workbenchContextMenu = document.getElementById('workbenchContextMenu');
    window.workbenchContextMenu = workbenchContextMenu;
    const workbenchPanel = document.getElementById('workbench');
    let contextMenuX = 0;
    let contextMenuY = 0;
    let currentRightClickedLayer = null;
    let currentRightClickedItem = null;

    if (!workbenchContextMenu || !workbenchPanel) return;

    workbenchPanel.addEventListener('contextmenu', (e) => {
        // Only intercept if we are inside the workbench grid or background, not the header
        if (e.target.closest('.workbench-header')) return;
        
        e.preventDefault(); // Disable default context menu
        
        const layerEl = e.target.closest('.canvas-layer');
        const workbenchItemEl = e.target.closest('.workbench-item');
        
        const ctxMenuSaveAsset = document.getElementById('ctxMenuSaveAsset');
        const ctxMenuCopyStyle = document.getElementById('ctxMenuCopyStyle');
        const ctxMenuPasteStyle = document.getElementById('ctxMenuPasteStyle');
        
        if (layerEl) {
            currentRightClickedLayer = layerEl;
            currentRightClickedItem = layerEl.closest('.workbench-item');
            if (ctxMenuSaveAsset) ctxMenuSaveAsset.style.display = 'block';
            if (ctxMenuCopyStyle) ctxMenuCopyStyle.style.display = 'block';
            if (ctxMenuPasteStyle) ctxMenuPasteStyle.style.display = window.copiedStyleLogic ? 'block' : 'none';
        } else if (workbenchItemEl) {
            currentRightClickedLayer = null;
            currentRightClickedItem = workbenchItemEl;
            if (ctxMenuSaveAsset) ctxMenuSaveAsset.style.display = 'block';
            if (ctxMenuCopyStyle) ctxMenuCopyStyle.style.display = 'block';
            if (ctxMenuPasteStyle) ctxMenuPasteStyle.style.display = window.copiedStyleLogic ? 'block' : 'none';
        } else {
            currentRightClickedLayer = null;
            currentRightClickedItem = null;
            if (ctxMenuSaveAsset) ctxMenuSaveAsset.style.display = 'none';
            if (ctxMenuCopyStyle) ctxMenuCopyStyle.style.display = 'none';
            if (ctxMenuPasteStyle) ctxMenuPasteStyle.style.display = 'none';
        }

        contextMenuX = e.clientX;
        contextMenuY = e.clientY;
        
        workbenchContextMenu.style.display = 'block';
        
        requestAnimationFrame(() => {
            const rect = workbenchContextMenu.getBoundingClientRect();
            let posX = contextMenuX;
            let posY = contextMenuY;
            
            if (posX + rect.width > window.innerWidth) {
                posX = window.innerWidth - rect.width;
            }
            if (posY + rect.height > window.innerHeight) {
                posY = window.innerHeight - rect.height;
            }
            
            workbenchContextMenu.style.left = `${posX}px`;
            workbenchContextMenu.style.top = `${posY}px`;
        });
    });

    document.addEventListener('click', (e) => {
        if (workbenchContextMenu.style.display === 'block') {
            workbenchContextMenu.style.display = 'none';
        }
    });

    document.getElementById('ctxMenuGroup').addEventListener('click', () => {
        workbenchContextMenu.style.display = 'none';
        createGroupLabel();
    });

    document.getElementById('ctxMenuAutoGrid').addEventListener('click', () => {
        autoOrganizeToGrid();
    });

    document.getElementById('ctxMenuAddText').addEventListener('click', () => {
        workbenchContextMenu.style.display = 'none';
        startDrawingShape('text');
    });



    document.getElementById('ctxMenuShapeRect').addEventListener('click', (e) => { e.stopPropagation(); workbenchContextMenu.style.display = 'none'; startDrawingShape('rect'); });
    document.getElementById('ctxMenuShapeEllipse').addEventListener('click', (e) => { e.stopPropagation(); workbenchContextMenu.style.display = 'none'; startDrawingShape('ellipse'); });
    document.getElementById('ctxMenuShapeLine').addEventListener('click', (e) => { e.stopPropagation(); workbenchContextMenu.style.display = 'none'; startDrawingShape('line'); });
    document.getElementById('ctxMenuShapeArrow').addEventListener('click', (e) => { e.stopPropagation(); workbenchContextMenu.style.display = 'none'; startDrawingShape('arrow'); });

    document.getElementById('ctxMenuUpload').addEventListener('click', () => {
        handleUploadImage();
    });

    document.getElementById('ctxMenuViewAll').addEventListener('click', () => {
        workbenchContextMenu.style.display = 'none';
        if (window.viewAllWorkbenchItems) window.viewAllWorkbenchItems();
    });

    document.getElementById('ctxMenuSaveState').addEventListener('click', async () => {
        if (!state.currentSessionId) {
            alert('请先开始一个对话会话');
            return;
        }
        const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
        if (!currentSession) return;

        const savedState = [];
        state.workbenchItems.forEach((item, id) => {
            if (!item?.el || isRuntimeManagedWorkbenchItem(item)) return;
            const el = item.el;
            const rect = {
                left: el.style.left,
                top: el.style.top,
                width: el.style.width,
                height: el.style.height,
                zIndex: el.style.zIndex
            };
            
            let content = null;
            let fontSize = null;
            let fontColor = null;
            let fontFamily = null;
            let fontWeight = null;
            let fontStyle = null;
            let lineHeight = null;
            let letterSpacing = null;
            let textAlign = null;
            let textShadow = null;
            let WebkitTextStroke = null;
            let customCss = null;
            let fillColor = null;
            let borderColor = null;
            let borderWidth = null;
            let borderRadius = null;
            let clipPath = null;
            if (item.type === 'text-note' || item.type === 'group-label') {
                const contentEl = el.querySelector('.note-content') || el.querySelector('.label-content');
                content = contentEl ? contentEl.innerHTML : '';
                if (item.type === 'text-note') {
                    fontSize = contentEl ? contentEl.style.fontSize : null;
                    fontColor = contentEl ? contentEl.style.color : null;
                    fontFamily = contentEl ? contentEl.style.fontFamily : null;
                    fontWeight = contentEl ? contentEl.style.fontWeight : null;
                    fontStyle = contentEl ? contentEl.style.fontStyle : null;
                    lineHeight = contentEl ? contentEl.style.lineHeight : null;
                    letterSpacing = contentEl ? contentEl.style.letterSpacing : null;
                    textAlign = contentEl ? contentEl.style.textAlign : null;
                    textShadow = contentEl ? contentEl.style.textShadow : null;
                    WebkitTextStroke = contentEl ? contentEl.style.webkitTextStroke : null;
                    customCss = item.css || null;
                }
            } else if (item.type === 'shape') {
                const svgEl = el.querySelector('svg');
                content = svgEl ? svgEl.outerHTML : '';
                fillColor = el.style.backgroundColor || null;
                borderColor = el.style.borderColor || null;
                borderWidth = el.style.borderWidth || null;
                borderRadius = el.style.borderRadius || null;
                clipPath = el.style.clipPath || el.style.webkitClipPath || null;
            }
            
            savedState.push({
                id: id,
                type: item.type || 'image',
                shapeType: item.shapeType,
                dataUrl: item.dataUrl || (item.el.querySelector('img') ? item.el.querySelector('img').src : null),
                rect: rect,
                content: content,
                fontSize: fontSize,
                fontColor: fontColor,
                fontFamily: fontFamily,
                fontWeight: fontWeight,
                fontStyle: fontStyle,
                lineHeight: lineHeight,
                letterSpacing: letterSpacing,
                textAlign: textAlign,
                textShadow: textShadow,
                WebkitTextStroke: WebkitTextStroke,
                css: customCss,
                fillColor: fillColor,
                borderColor: borderColor,
                borderWidth: borderWidth,
                borderRadius: borderRadius,
                clipPath: clipPath,
                genealogy: item.genealogy,
                parentId: item.parentId,
                layerName: item.layerName,
                originalBbox: item.originalBbox,
                key: item.key,
                originalDataUrl: item.originalDataUrl,
                cleanPlateDataUrl: item.cleanPlateDataUrl,
                cleanPlateStatus: item.cleanPlateStatus
            });
        });

        currentSession.workbenchState = savedState;
        const workspace = runtime.getCurrentWorkspace();
        if (workspace) {
            currentSession.runtimeWorkspace = {
                projectId: workspace.projectId,
                name: workspace.name,
                currentState: {
                    stateId: workspace.currentState.stateId,
                    canvasState: workspace.currentState.canvasState,
                    assets: workspace.currentState.assetRegistry.getAll(),
                    nodes: workspace.currentState.sceneGraph.getNodes(),
                    edges: workspace.currentState.sceneGraph.getAllEdges()
                },
                viewport: {
                    zoom: state.workbenchZoom,
                    panOffsetX: state.panOffsetX,
                    panOffsetY: state.panOffsetY
                },
                decisionGraph: workspace.decisionGraph.getHistory()
            };
        }
        if (window.dbHelper) await window.dbHelper.saveSession(currentSession);
        
        // Show a temporary toast message
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
            background: rgba(42, 92, 130, 0.9); color: white; padding: 12px 24px;
            border-radius: 8px; font-size: 14px; z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); transition: opacity 0.3s;
        `;
        toast.innerHTML = '<i class="fas fa-check-circle" style="margin-right: 8px;"></i> 画板状态已保存';
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    });

    document.getElementById('ctxMenuSaveAsset')?.addEventListener('click', async () => {
        workbenchContextMenu.style.display = 'none';
        
        if (!currentRightClickedItem) return;
        
        const itemId = currentRightClickedItem.dataset.itemId || currentRightClickedItem.id;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        let imageUrl = null;
        let assetName = 'Extracted Asset';
        let assetCategory = 'unknown';

        if (currentRightClickedLayer) {
            const layerIndex = parseInt(currentRightClickedLayer.dataset.layerIndex, 10);
            let layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
            if (!layers || !layers[layerIndex]) return;
            
            const layerData = layers[layerIndex];
            if (layerData.activeVersionId && layerData.versions) {
                const v = layerData.versions.find(v => v.id === layerData.activeVersionId);
                if (v && v.cutoutUrl) imageUrl = v.cutoutUrl;
            }
            if (!imageUrl && layerData.cutoutUrl) imageUrl = layerData.cutoutUrl;
            
            assetName = layerData.name || `图层 ${layerIndex}`;
            assetCategory = layerData.category || 'unknown';
        } else {
            // It's a whole workbench item (metadata or original dataurl)
            imageUrl = item.dataUrl;
            assetName = item.label || '独立素材';
            assetCategory = item.type || 'unknown';
        }
        
        if (!imageUrl) {
            alert('该对象尚未生成内容或图像');
            return;
        }

        // Upload to OSS if it's not already an OSS URL (or if needed)
        let finalImageUrl = imageUrl;
        try {
            if (imageUrl.startsWith('data:') || imageUrl.startsWith('blob:')) {
                finalImageUrl = await uploadImageToOSS(imageUrl);
            }
        } catch (ossErr) {
            console.error('OSS 上传失败:', ossErr);
            alert('素材持久化存储失败（OSS上传失败），将仅保存在本地 IndexedDB');
        }

        // MVR Asset Registration
        if (window.mvrRuntime && typeof window.mvrRuntime.getCurrentWorkspace === 'function') {
            const workspace = window.mvrRuntime.getCurrentWorkspace();
            if (workspace) {
                const assetId = `independent_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
                const newAsset = {
                    uid: assetId,
                    type: assetCategory,
                    sourceImage: finalImageUrl,
                    masks: [],
                    variants: [],
                    metadata: {
                        createdAt: Date.now(),
                        updatedAt: Date.now(),
                        creatorId: 'local_user',
                        usageCount: 0,
                        tags: [assetName]
                    }
                };
                
                workspace.currentState.assetRegistry.register(newAsset);
                await window.mvrRuntime.saveCurrentWorkspace();
                console.log("[MVR] Asset registered and workspace saved to IndexedDB.");
            }
        }

        const mvrAssetsDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'my_assets_library' });
        
        const assetId = `independent_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
        const independentAsset = {
            uid: assetId,
            type: assetCategory,
            sourceImage: finalImageUrl,
            name: assetName,
            metadata: {
                createdAt: Date.now(),
                tags: [assetName]
            }
        };
        
        await mvrAssetsDB.setItem(assetId, independentAsset);
        
        // Show a temporary toast message
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
            background: rgba(245, 158, 11, 0.9); color: white; padding: 12px 24px;
            border-radius: 8px; font-size: 14px; z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); transition: opacity 0.3s;
        `;
        toast.innerHTML = '<i class="fas fa-star" style="margin-right: 8px;"></i> 已存入独立资产库';
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 2000);
        
        // 触发一下 UI 更新，如果需要
        if (window.renderMyAssetsPanel) {
            window.renderMyAssetsPanel();
        }
    });

    document.getElementById('ctxMenuCopyStyle')?.addEventListener('click', () => {
        workbenchContextMenu.style.display = 'none';
        
        if (!currentRightClickedItem) return;
        
        const itemId = currentRightClickedItem.dataset.itemId || currentRightClickedItem.id;
        const item = state.workbenchItems.get(itemId);
        if (!item) return;

        let styleLogic = null;

        if (currentRightClickedLayer) {
            const layerIndex = parseInt(currentRightClickedLayer.dataset.layerIndex, 10);
            let layers = item.scene && item.scene.layers ? item.scene.layers : item.layers;
            if (!layers || !layers[layerIndex]) return;
            const layerData = layers[layerIndex];
            styleLogic = layerData.promptHint || item.genealogy?.prompt || item.generationParams?.prompt;
        } else {
            styleLogic = item.genealogy?.prompt || item.generationParams?.prompt;
        }

        if (styleLogic) {
            window.copiedStyleLogic = styleLogic;
            
            const toast = document.createElement('div');
            toast.style.cssText = `
                position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
                background: rgba(42, 92, 130, 0.9); color: white; padding: 12px 24px;
                border-radius: 8px; font-size: 14px; z-index: 10000;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15); transition: opacity 0.3s;
            `;
            toast.innerHTML = '<i class="fas fa-copy" style="margin-right: 8px;"></i> 设计决策已复制：' + styleLogic;
            document.body.appendChild(toast);
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }, 2000);
        } else {
            alert('该对象并没有记录任何可以提取的设计决策。');
        }
    });

    document.getElementById('ctxMenuPasteStyle')?.addEventListener('click', async () => {
        workbenchContextMenu.style.display = 'none';
        
        if (!currentRightClickedItem || !window.copiedStyleLogic) return;
        
        const itemId = currentRightClickedItem.dataset.itemId || currentRightClickedItem.id;
        const promptParams = window.copiedStyleLogic;

        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
            background: rgba(16, 185, 129, 0.9); color: white; padding: 12px 24px;
            border-radius: 8px; font-size: 14px; z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); transition: opacity 0.3s;
        `;
        toast.innerHTML = '<i class="fas fa-paste" style="margin-right: 8px;"></i> 正在应用设计决策...';
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
        
        if (currentRightClickedLayer) {
            const layerIndex = parseInt(currentRightClickedLayer.dataset.layerIndex, 10);
            await editLayerAsset(itemId, layerIndex, promptParams);
        } else {
            if (window.handleIsolatedAssetEdit) {
                await window.handleIsolatedAssetEdit(itemId, promptParams);
            } else {
                alert('该对象不支持直接粘贴决策');
            }
        }
    });
}
