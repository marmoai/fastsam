import { state } from './state.js';
import { saveWorkbenchStateToOSS, getWorkbenchStateFromOSS } from '../services/ossService.js';
import { runtime } from '../runtime/CoreRuntime';

function isRuntimeManagedWorkbenchItem(item) {
    if (!item) return false;
    const type = item.type || 'image';
    return !['text-note', 'group-label', 'shape', 'atmosphere'].includes(type);
}

class HistoryManager {
    constructor() {
        this.undoStack = [];
        this.redoStack = [];
        this.maxSize = 50;
        this.isRestoring = false;
        this.debounceTimer = null;
    }

    clear() {
        this.undoStack = [];
        this.redoStack = [];
    }

    pushState() {
        if (this.isRestoring || window.isRestoringSession || state.isCropping) return;
        
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        this.debounceTimer = setTimeout(async () => {
            // Double check cropping state inside the timeout to prevent capturing temporary UI
            if (state.isCropping) return;

            const snapshot = this._createSnapshot(state.workbenchItems);
            
            if (this.undoStack.length > 0) {
                const lastSnapshot = this.undoStack[this.undoStack.length - 1];
                if (this._isEqual(lastSnapshot, snapshot)) {
                    return;
                }
            }

            this.undoStack.push(snapshot);
            if (this.undoStack.length > this.maxSize) {
                this.undoStack.shift();
            }
            this.redoStack = [];

            // 同步到云端
            try {
                await this.syncToCloud();
            } catch (e) {
                console.error('同步状态到云端失败', e);
            }
        }, 100);
    }

    undo() {
        if (this.undoStack.length <= 1) return;
        
        const currentState = this.undoStack.pop();
        this.redoStack.push(currentState);
        
        const previousState = this.undoStack[this.undoStack.length - 1];
        if (previousState) {
            this._restoreSnapshot(previousState);
        }
    }

    redo() {
        if (this.redoStack.length === 0) return;
        
        const nextState = this.redoStack.pop();
        this.undoStack.push(nextState);
        this._restoreSnapshot(nextState);
    }

    _createSnapshot(stateMap) {
        const snapshot = new Map();
        stateMap.forEach((item, id) => {
            if (!item.el) return;
            
            // Temporary clean up of any crop elements that might be in the DOM
            // but not yet caught by state.isCropping (safety measure)
            const elClone = item.el.cloneNode(true);
            elClone.querySelectorAll('.crop-handle, .crop-dimmer, .crop-border').forEach(child => child.remove());
            elClone.classList.remove('cropping');

            snapshot.set(id, {
                ...item, // Keep references to file, dataUrl, el, etc.
                style: {
                    left: item.el.style.left,
                    top: item.el.style.top,
                    width: item.el.style.width,
                    height: item.el.style.height,
                    transform: item.el.style.transform,
                    zIndex: item.el.style.zIndex,
                    display: item.el.style.display
                },
                className: elClone.className,
                content: elClone.innerHTML
            });
        });
        return snapshot;
    }

    _isEqual(snap1, snap2) {
        if (snap1.size !== snap2.size) return false;
        for (let [id, item1] of snap1) {
            const item2 = snap2.get(id);
            if (!item2) return false;
            if (item1.style.left !== item2.style.left ||
                item1.style.top !== item2.style.top ||
                item1.style.width !== item2.style.width ||
                item1.style.height !== item2.style.height ||
                item1.style.transform !== item2.style.transform ||
                item1.style.zIndex !== item2.style.zIndex ||
                item1.dataUrl !== item2.dataUrl ||
                item1.content !== item2.content) {
                return false;
            }
        }
        return true;
    }

    _restoreSnapshot(snapshot) {
        this.isRestoring = true;
        const workbenchGrid = document.getElementById('workbenchGrid');
        if (!workbenchGrid) {
            this.isRestoring = false;
            return;
        }

        // 1. Remove items that exist in current state but not in snapshot
        state.workbenchItems.forEach((item, id) => {
            if (!snapshot.has(id)) {
                if (typeof window.invalidatePendingWorkbenchUpload === 'function') {
                    window.invalidatePendingWorkbenchUpload(id, 'history_restore_removed');
                }
                if (item.el && item.el.parentNode) {
                    item.el.parentNode.removeChild(item.el);
                }
                state.workbenchItems.delete(id);
                state.selectedWorkbenchItems.delete(id);
            }
        });

        // 2. Add or update items from snapshot
        snapshot.forEach((snapItem, id) => {
            let currentItem = state.workbenchItems.get(id);

            if (typeof window.invalidatePendingWorkbenchUpload === 'function') {
                window.invalidatePendingWorkbenchUpload(id, 'history_restore_reconcile');
            }
            
            if (!currentItem) {
                // Item was deleted, restore it
                currentItem = { ...snapItem };
                state.workbenchItems.set(id, currentItem);
                workbenchGrid.appendChild(currentItem.el);
            }
            
            // Restore styles
            if (currentItem.el && snapItem.style) {
                Object.assign(currentItem.el.style, snapItem.style);
            }

            // Restore className
            if (currentItem.el && snapItem.className) {
                currentItem.el.className = snapItem.className;
            }
            
            // Restore content
            if (currentItem.el && snapItem.content !== undefined && currentItem.el.innerHTML !== snapItem.content) {
                currentItem.el.innerHTML = snapItem.content;
            }
            
            // Restore other properties
            Object.assign(currentItem, snapItem);
        });

        // Redraw lines
        if (typeof window.drawGenealogyConnections === 'function') {
            window.drawGenealogyConnections();
        }

        // Update selection styles
        document.querySelectorAll('.workbench-item').forEach(el => {
            if (state.selectedWorkbenchItems.has(el.dataset.itemId)) {
                el.classList.add('selected');
            } else {
                el.classList.remove('selected');
            }
        });

        this.isRestoring = false;
        
        // 恢复后也同步一次云端
        this.syncToCloud().catch(console.error);
    }

    async syncToCloud() {
        if (!state.currentSessionId || !window.dbHelper) return;
        
        const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
        if (!currentSession) return;

        const savedState = [];
        state.workbenchItems.forEach((item, id) => {
            if (!item.el) return;
            if (isRuntimeManagedWorkbenchItem(item)) return;
            const el = item.el;
            const rect = {
                left: el.style.left,
                top: el.style.top,
                width: el.style.width,
                height: el.style.height,
                zIndex: el.style.zIndex,
                display: el.style.display
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
                if (contentEl) {
                    // Extract true inner HTML by bypassing any nested tags deeply
                    let innerContent = contentEl.innerHTML;
                    let tempDiv = document.createElement('div');
                    tempDiv.innerHTML = innerContent;
                    let nested = tempDiv.querySelector('.note-content') || tempDiv.querySelector('.label-content');
                    while (nested) {
                        innerContent = nested.innerHTML;
                        tempDiv.innerHTML = innerContent;
                        nested = tempDiv.querySelector('.note-content') || tempDiv.querySelector('.label-content');
                    }
                    content = innerContent;
                } else {
                    content = '';
                }
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
                layers: item.layers,
                scene: item.scene,
                semanticViews: item.semanticViews,
                hasFullSemanticAnalysis: item.hasFullSemanticAnalysis,
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
                decisionGraph: workspace.decisionGraph.getHistory()
            };
        }
        await window.dbHelper.saveSession(currentSession);
    }

    async loadFromCloud() {
        try {
            const cloudState = await getWorkbenchStateFromOSS();
            if (!cloudState || !cloudState.items) return false;

            this.isRestoring = true;
            
            // 1. Clear current workbench
            if (typeof window.clearWorkbench === 'function') {
                await window.clearWorkbench(true);
            }

            // 2. Restore zoom and pan
            if (cloudState.zoom !== undefined) {
                state.workbenchZoom = cloudState.zoom;
                if (typeof window.applyWorkbenchZoom === 'function') {
                    window.applyWorkbenchZoom();
                }
            }
            if (cloudState.pan) {
                state.workbenchPan = cloudState.pan;
                if (typeof window.applyPanTransform === 'function') {
                    window.applyPanTransform();
                }
            }

            // 3. Restore items
            for (const item of cloudState.items) {
                try {
                    const style = item.style || {};
                    const left = style.left || '0px';
                    const top = style.top || '0px';
                    const width = style.width || '200px';
                    const height = style.height || '200px';
                    const zIndex = style.zIndex || '1';

                    if (item.type === 'text-note') {
                        if (typeof window.addTextNoteToWorkbench === 'function') {
                            window.addTextNoteToWorkbench(0, 0, {
                                ...item,
                                left: left,
                                top: top,
                                width: parseFloat(width),
                                height: parseFloat(height),
                                zIndex: zIndex
                            });
                        }
                    } else if (item.type === 'group-label') {
                        if (typeof window.restoreGroupLabelToWorkbench === 'function') {
                            window.restoreGroupLabelToWorkbench({
                                ...item,
                                left: left,
                                top: top,
                                width: width,
                                height: height,
                                zIndex: zIndex
                            });
                        }
                    } else if (item.type === 'shape') {
                        if (typeof window.restoreShapeToWorkbench === 'function') {
                            window.restoreShapeToWorkbench({
                                ...item,
                                left: left,
                                top: top,
                                width: width,
                                height: height,
                                zIndex: zIndex
                            });
                        }
                    } else if (item.type === 'atmosphere') {
                        if (typeof window.addAtmosphereNode === 'function') {
                            window.addAtmosphereNode(item.metadata?.key || 'sunny', parseFloat(left), parseFloat(top));
                        }
                    } else {
                        // Default to image item
                        if (typeof window.addImageToWorkbench === 'function') {
                            await window.addImageToWorkbench(null, item.label || '', {
                                id: item.id,
                                dataUrl: item.dataUrl,
                                left: left,
                                top: top,
                                initialWidth: parseFloat(width),
                                initialHeight: parseFloat(height),
                                zIndex: zIndex,
                                genealogy: item.genealogy,
                                metadata: item.metadata,
                                type: item.type,
                                fusionProperties: item.fusionProperties
                            });
                        }
                    }
                } catch (err) {
                    console.error('Failed to restore item from cloud state:', item, err);
                }
            }

            // 4. Finalize
            if (typeof window.drawGenealogyConnections === 'function') {
                window.drawGenealogyConnections();
            }

            this.undoStack = [this._createSnapshot(state.workbenchItems)];
            this.redoStack = [];
            this.isRestoring = false;
            return true;
        } catch (e) {
            console.error('从云端加载状态失败', e);
            this.isRestoring = false;
            return false;
        }
    }

    _createSerializableState(stateMap) {
        const items = [];
        stateMap.forEach((item, id) => {
            if (!item.el) return;
            items.push({
                id,
                type: item.type,
                label: item.label,
                dataUrl: item.dataUrl,
                style: {
                    left: item.el.style.left,
                    top: item.el.style.top,
                    width: item.el.style.width,
                    height: item.el.style.height,
                    zIndex: item.el.style.zIndex,
                    transform: item.el.style.transform
                },
                metadata: item.metadata,
                genealogy: item.genealogy,
                fusionProperties: item.fusionProperties,
                content: (item.type === 'text-note' || item.type === 'group-label' || item.type === 'shape') ? item.el.innerHTML : null
            });
        });
        return {
            items,
            zoom: state.workbenchZoom,
            pan: state.workbenchPan
        };
    }
}

export const historyManager = new HistoryManager();
window.historyManager = historyManager;
