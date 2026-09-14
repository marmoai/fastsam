import { updateHeader } from './ui/header.js';
import { runtime } from './runtime/CoreRuntime';
import { memoryLayer } from './runtime/CreativeMemoryLayer';
import { implicitMemoryEngine } from './runtime/ImplicitMemoryEngine';
import { resultFeedbackRuntime } from './runtime/ResultFeedbackRuntime';
import { interactionAttributionRuntime } from './runtime/InteractionAttributionRuntime';
import { initPredictivePromptEngine } from './ui/predictive-prompt.js';
import localforage from 'localforage';
import {
    handlePreviewDragStart, handlePreviewDragEnter, handlePreviewDragLeave,
    handlePreviewDragOver, handlePreviewDragEnd, handlePreviewDrop,
    handleTouchDragStart, initWorkbenchDrop
} from './ui/drag-drop.js';
import {
    updateWorkbenchItemsScale, resetWorkbenchZoom, updateZoomIndicator,
    setupKeyboardShortcuts, zoomIn, zoomOut, resetZoom,
    initWorkbenchPan, handlePanStart, processPanMove, handlePanMove,
    handlePanEnd, applyPanTransform, resetPan, applyWorkbenchZoom,
    viewAllWorkbenchItems, screenToWorkbenchCoord, updateSelectedItems,
    handleWorkbenchDragStart, handleResizeStart, showWorkbenchToolbox,
    initGuideLines, initWorkbenchZoom, handleWorkbenchWheel, updateBackgroundGrid, setupSelectionBox
} from './ui/interaction.js';
import {
    getIntersectionArea, checkProximity, initInjectionEngine
} from './ui/injection-engine.js';
import {
    showCustomConfirm, openMagicWandModal, closeMagicWandModal, handleMagicWandAction,
    openCameraAngleModal, closeCameraAngleModal, updateCameraAnglePreview,
    showLayerManagerModal, showVideoPromptModal, initCameraAngleModal
} from './ui/modals.js';
import {
    showFloatingFusionEditor, hideFloatingFusionEditor, showPreciseEditDialog,
    preciseEditMode, startPreciseEditMode
} from './ui/fusion-editor.js';
import { setupToolboxEvents } from './ui/toolboxes.js';
import {
    triggerLayerExplosion, executeAgentLayerExtraction, undoAgentLayerExtraction, commitAgentLayerExtraction, releaseAgentLayerExtraction,
    executeAgentCapability, undoAgentCapability, commitAgentCapabilities, releaseAgentCapabilities,
    handleQuickFusionSync, handleIsolatedAssetEdit, performPreciseEdit
} from './ui/layer-manager.js';
import { initSidebar, sidebarState } from './ui/sidebar.js';
import {
    deleteSelectedItems, findNextPosition, calculateSmartPosition, addImageToWorkbench,
    autoOrganizeToGrid, createGroupLabel, restoreGroupLabelToWorkbench,
    addTextNoteToWorkbench, handleUploadImage, deleteWorkbenchItem,
    clearWorkbench, addAtmosphereNode
} from './ui/workbench-core.js';
import { buildMessageContentHTML, addMessage, renderMessages, yieldToChatPaint, appendSuggestionButtons, buildBotFallbackText, renderAgentDebateInChat } from './ui/chat-panel.js';
import { handleAssetReuseRequest, isAssetReuseRequest, handleLayerExtractionRequest, isLayerExtractionRequest } from './ui/agent-task-controller.js';
import { initChatSourceRail } from './ui/chat-source-rail.js';
    
    import { Modality } from "@google/genai";
    import { AGENTS } from "/src/ai-services/agents.js";
    import { ai, getTextModel, getImageModel, generateTextWithSearch, generateLatentImage, createChatSession, generateVeoVideo, generateTextWithGemini, clearChatSession } from "/src/ai-services/gemini-client.js";
    import { editOrQueryImageWithGemini, editOrQueryImageWithGemini_Multiple, generateImage, generateSessionTitle, analyzeWithAgent, getSmartSuggestions, analyzeImageLayers, generateRelitImage, generatePreciseEditImage, generateVisualSearch, classifyImageCategory, planGraph, buildSeriesContinuityAnchor } from "/src/ai-services/skills-engine.js";
    import { SNAP_THRESHOLD, ZOOM_MIN, ZOOM_MAX, ZOOM_STEP, RESIZE_RADIUS, OUTPAINT_RADIUS, CANVAS_CENTER, DEFAULT_ZOOM, RATIO_MAP, ATMOSPHERE_OPTS, isAgentRuntimeFeatureEnabled } from "/src/core/config.js";
    import { state } from "/src/core/state.js";
    import { getExplicitRequestedImageCount, isRemovalRequest, isMaterialRequest, fileToDataURL, fileToBase64, dataURLToFile, dataURLtoFileSync, blobToBase64, addWatermark, getProxiedUrl, isInvalidImageSrc, showToast } from "/src/core/utils.js";
    import { determineConversationIntent, executeConversationIntent, composeExecutionPrompt, resolveExecutionPrompt, mergeRegeneratePrompt, getIntentPlaceholderText, shouldUseImagePlaceholder } from "/src/ai-services/intent-router.js";
    import { interpretConversationTurn } from "/src/ai-services/conversation-interpreter.js";
    import { resolveStateConflict } from "/src/ai-services/state-conflict-resolver.js";
    import { getAssistantWorkspaceContext } from "/src/services/workspace-context.js";
    import { recordWorkspaceAction } from "/src/services/workspace-context.js";
    import { applyWorkspaceCommand, buildWorkspaceCommandResponse, detectWorkspaceCommand } from "/src/services/workspace-intents.js";
    import { executeLensSearch, looksLikeLensSearchRequest } from "/src/services/capability-orchestrator.js";
    import { createDefaultWorkingMemory, ensureSessionWorkingMemory, mergeInterpreterResultIntoSession, finalizeWorkingMemoryWithAssistantResult, setContinuityAnchor } from "/src/ai-services/session-working-memory.js";
    import { shouldUseQwenLocalImageRouting, getQwenLocalRoutingDiagnostics } from "/src/ai-services/qwen-local-routing.js";
    import { initMaskDrawer, openMaskEditor, closeMaskEditor, applyMask } from "/src/graphics/mask-drawer.js";
    import { initRelightEngine, updateRelightingPreview, applyRelighting } from "/src/graphics/relight-engine.js";
    import { initGenealogyLines, drawGenealogyConnections } from "/src/graphics/genealogy-lines.js";
    import { createImageWithHole, cropImageByBox } from "/src/graphics/image-processor.js";
    import { initLatentCanvas, setupLatentCanvas } from "/src/graphics/latent-canvas.js";
import { MarmoLens } from "./ui/marmo-lens.js";
import { initAtmospherePalette, applyAtmosphereToImage } from "./ui/atmosphere.js";
import { initDecisionLog } from "./ui/decision-log.js";
import { historyManager } from './core/history.js';
import { loadSession, startNewSession, dbHelper, setLastActiveSessionId } from "./core/session.js";
import { graphStore } from './engine/graph-store.js';
import { graphRunner } from './engine/graph-runner.js';
import { NODE_REGISTRY } from './engine/node-registry.js';
import { executeNode } from './engine/node-executor.js';
import { installCrashDiagnostics } from './core/crash-diagnostics.js';
import { initAgentVisualSurface } from './ui/agent-visual-surface.js';
import { initAgentDock } from './ui/agent-dock.js';

// Install before application restoration so failures during startup and long
// running Magic Layers jobs leave a recoverable diagnostic trail.
installCrashDiagnostics();

// Expose to window for debugging and future Agent use
window.graphStore = graphStore;
window.graphRunner = graphRunner;
window.NODE_REGISTRY = NODE_REGISTRY;
window.executeNode = executeNode;

    // --- UI Elements ---
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const modelSelect = document.getElementById('modelSelect');
    const thinkingLevelToggle = document.getElementById('thinkingLevelToggle');
    const plusMenuBtn = document.getElementById('plusMenuBtn');
    const plusMenu = document.getElementById('plusMenu');
    const menuUploadBtn = document.getElementById('menuUploadBtn');
    const menuWorkflowBtn = document.getElementById('menuWorkflowBtn');
    const mainImageUpload = document.getElementById('mainImageUpload');
    const mainImagePreview = document.getElementById('mainImagePreview');
    const historyList = document.getElementById('historyList');
    const emptyState = document.getElementById('emptyState');
    
    let isWorkflowMode = false;
    const newSessionBtn = document.getElementById('newSessionBtn');
    const maskEditorModal = document.getElementById('maskEditorModal');
    const maskCanvasContainer = document.getElementById('maskCanvasContainer');
    const closeMaskEditorBtn = document.getElementById('closeMaskEditor');
    const cancelMaskBtn = document.getElementById('cancelMask');
    const applyMaskBtn = document.getElementById('applyMask');
    const magicWandModal = document.getElementById('magicWandModal');
    const workbenchGrid = document.getElementById('workbenchGrid');
    const workbenchToolbox = document.getElementById('workbenchToolbox');
    const injectionMenu = document.getElementById('injectionMenu');

    // --- NEW: Camera Angle Elements ---
    const cameraAngleModal = document.getElementById('cameraAngleModal');
    const closeCameraAngleBtn = document.getElementById('closeCameraAngleBtn');
    const cameraRotate = document.getElementById('cameraRotate');
    const cameraVertical = document.getElementById('cameraVertical');
    const cameraZoom = document.getElementById('cameraZoom');
    const cameraRotateVal = document.getElementById('cameraRotateVal');
    const cameraVerticalVal = document.getElementById('cameraVerticalVal');
    const cameraZoomVal = document.getElementById('cameraZoomVal');
    const cameraPreviewImg = document.getElementById('cameraPreviewImg');
    const resetCameraBtn = document.getElementById('resetCameraBtn');
    const applyCameraAngleBtn = document.getElementById('applyCameraAngleBtn');

    // --- NEW: Layer Manager Elements ---
    const layerManagerModal = document.getElementById('layerManagerPanel');
    const closeLayerModalBtn = document.getElementById('closeLayerPanel');
    const layerList = document.getElementById('layerList');

    // --- NEW: Asset Library Elements ---
    const assetLibraryGrid = document.getElementById('assetLibraryGrid');
    const emptyLibraryState = document.getElementById('emptyLibraryState');

    // --- Latent Canvas Elements ---
    const latentCanvasModal = document.getElementById('latentCanvasModal');
    const latentSketchCanvas = document.getElementById('latentSketchCanvas');
    const latentResultImg = document.getElementById('latentResultImg');
    const latentLoader = document.getElementById('latentLoader');
    const latentPromptInput = document.getElementById('latentPrompt');
    const openLatentCanvasBtn = document.getElementById('openLatentCanvasBtn');

    // 获取DOM元素
    const workbenchZoomContainer = document.getElementById('workbenchZoomContainer');
    const zoomIndicator = document.getElementById('zoomIndicator'); // May be null if element doesn't exist
    /*const resetZoomBtn = document.getElementById('resetZoom');*/

    // --- State (Migrated to state.js) ---
    const stateProps = [
        'mainImageFile', 'referenceImageFiles', 'maskDataUrl', 'isSending', 
        'lastGeneratedImageForEditing', 'isContextPreviewHidden', 'sessions', 
        'currentSessionId', 'pendingBaseImageShare', 'lastGenerationContext', 
        'userAssets', 'draggedWorkbenchItem', 'currentActiveWorkbenchItemId', 
        'currentIntentLock', 'selectionBox', 'isSelecting', 'selectionStart', 
        'isCtrlPressed', 'isAltPressed', 'workbenchItemCount', 'isPanning', 
        'isSpacePressed', 'panStartX', 'panStartY', 'panOffsetX', 'panOffsetY', 
        'workbenchZoom'
    ];
    
    stateProps.forEach(prop => {
        Object.defineProperty(window, prop, {
            get: () => state[prop],
            set: (val) => state[prop] = val,
            configurable: true
        });
    });

    const { 
        pendingReferenceImageShares, workbenchItems, selectedWorkbenchItems, fileToWorkbenchIdMap, latentState, maskEditor 
    } = state;
    
    window.state = state;
    window.pendingReferenceImageShares = pendingReferenceImageShares;
    window.workbenchItems = workbenchItems;
    window.selectedWorkbenchItems = selectedWorkbenchItems;
    window.fileToWorkbenchIdMap = fileToWorkbenchIdMap;
    window.latentState = latentState;
    window.maskEditor = maskEditor;

    // Undo Redo buttons
    const undoBtn = document.getElementById('undoBtn');
    const redoBtn = document.getElementById('redoBtn');

    function updateHistoryButtons() {
        if (window.historyManager) {
            const canUndo = window.historyManager.undoStack.length > 1;
            const canRedo = window.historyManager.redoStack.length > 0;
            undoBtn.disabled = !canUndo;
            redoBtn.disabled = !canRedo;
            undoBtn.style.opacity = undoBtn.disabled ? '0.3' : '1';
            redoBtn.style.opacity = redoBtn.disabled ? '0.3' : '1';
            undoBtn.style.cursor = undoBtn.disabled ? 'default' : 'pointer';
            redoBtn.style.cursor = redoBtn.disabled ? 'default' : 'pointer';
        }
    }

    if (undoBtn) {
        undoBtn.addEventListener('click', () => {
            Promise.resolve(window.historyManager?.undo()).finally(updateHistoryButtons);
        });
    }

    if (redoBtn) {
        redoBtn.addEventListener('click', () => {
            Promise.resolve(window.historyManager?.redo()).finally(updateHistoryButtons);
        });
    }

    // Subscribe to state changes to update buttons
    setInterval(updateHistoryButtons, 500); // Quick hack to keep history buttons updated, better to subscribe to the runtime's notify but let's just make it work for now. Alternatively we can hook into subscribe.

    runtime.restoreWorkspace().then(ws => {
        if (ws) {
            ws.currentState.subscribe(updateHistoryButtons);
        } else {
            const initialWs = runtime.createWorkspace('default', 'Default Workspace');
            initialWs.snapshot();
            initialWs.currentState.subscribe(updateHistoryButtons);
        }
        updateHistoryButtons();
    });


    // 添加键盘事件监听
    document.addEventListener('keydown', (e) => {
        // Handle Delete or Backspace for deletion
        const isDeleteKey = e.key === 'Delete' || e.key === 'Backspace';
        
        if (isDeleteKey && selectedWorkbenchItems.size > 0) {
            // Check if we are editing a text note or input - if so, don't delete the item
            if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA' || document.activeElement.contentEditable === "true")) {
                return;
            }
            e.preventDefault();
            deleteSelectedItems(); // Show confirmation dialog
        }

        // Handle Arrow keys for moving selected items
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key) && selectedWorkbenchItems.size > 0) {
            // Check if we are editing a text note or input - if so, let the default behavior (caret movement) happen
            if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA' || document.activeElement.contentEditable === "true")) {
                return;
            }
            e.preventDefault(); // Prevent default scroll behavior
            
            // Move by 10 pixels if Shift is held down, else 1 pixel
            const movement = e.shiftKey ? 10 : 1; 
            const currentZoom = state.workbenchZoom || 1;
            const adjustedMovement = movement / currentZoom;

            const workspace = window.mvrRuntime ? window.mvrRuntime.getCurrentWorkspace() : null;
            const transformsToBatch = [];
            const runtimeMoveRoots = new Set();

            selectedWorkbenchItems.forEach(id => {
                const item = workbenchItems.get(id);
                if (item && item.el) {
                    const asset = workspace ? workspace.currentState.assetRegistry.get(id) : null;
                    const isNormalAsset = asset && asset.transform;
                    
                    if (isNormalAsset && item.parentId && selectedWorkbenchItems.has(item.parentId)) {
                        return;
                    }
                    
                    let left = 0;
                    let top = 0;
                    
                    if (isNormalAsset) {
                        left = asset.transform.x ?? (parseFloat(item.el.style.left) || 0);
                        top = asset.transform.y ?? (parseFloat(item.el.style.top) || 0);
                    } else {
                        left = parseFloat(item.el.style.left) || 0;
                        top = parseFloat(item.el.style.top) || 0;
                    }
                    
                    if (e.key === 'ArrowUp') top -= adjustedMovement;
                    if (e.key === 'ArrowDown') top += adjustedMovement;
                    if (e.key === 'ArrowLeft') left -= adjustedMovement;
                    if (e.key === 'ArrowRight') left += adjustedMovement;
                    
                    if (isNormalAsset) {
                        transformsToBatch.push({
                            uid: id,
                            transform: { x: left, y: top }
                        });
                        runtimeMoveRoots.add(id);
                    } else {
                        item.el.style.top = `${top}px`;
                        item.el.style.left = `${left}px`;
                    }
                    
                    if (window.updateParentBBox) {
                        window.updateParentBBox(id);
                    }

                    if (typeof window.drawGenealogyConnections === 'function') {
                        window.drawGenealogyConnections();
                    }
                    
                    if (item.el.classList.contains('atmosphere-node') && typeof window.updateRelightingPreview === 'function') {
                        window.updateRelightingPreview(item);
                    }
                }
            });
            
            if (workspace && transformsToBatch.length > 0) {
                workspace.dispatcher.dispatch({
                    type: 'BATCH_UPDATE_TRANSFORMS',
                    payload: { transforms: transformsToBatch },
                    meta: { skipSnapshot: true, persistWithoutSnapshot: true }
                });

                // Dispatcher cascades parent movement to linked children. The
                // runtime is authoritative, so reconcile only the moved roots
                // and their descendants after the complete action queue settles.
                const runtimeIdsToSync = new Set(runtimeMoveRoots);
                const pendingIds = [...runtimeMoveRoots];
                while (pendingIds.length > 0) {
                    const parentId = pendingIds.shift();
                    const edges = workspace.currentState.sceneGraph.getEdgesForNode(parentId);
                    edges.forEach((edge) => {
                        if (edge.sourceId !== parentId || edge.relationType !== 'parent_of') return;
                        if (runtimeIdsToSync.has(edge.targetId)) return;
                        if (!workspace.currentState.assetRegistry.get(edge.targetId)) return;
                        runtimeIdsToSync.add(edge.targetId);
                        pendingIds.push(edge.targetId);
                    });
                }

                runtimeIdsToSync.forEach((id) => {
                    if (typeof window.syncDOMToScene === 'function') {
                        window.syncDOMToScene(id);
                    }
                });
            }

            if (window.historyManager) {
                window.historyManager.pushTransformState();
            }
        }

        if (e.ctrlKey || e.metaKey) {
            isCtrlPressed = true;
            
        }
        if (e.altKey) {
            isAltPressed = true;
        }
    });

    // Model and Thinking Level listeners
    modelSelect.addEventListener('change', (e) => {
        state.selectedModel = e.target.value;
    });

    thinkingLevelToggle.addEventListener('click', () => {
        state.thinkingLevel = state.thinkingLevel === 'LOW' ? 'MEDIUM' : 'LOW';
        thinkingLevelToggle.innerHTML = `<i class="fas fa-brain"></i> 思考: ${state.thinkingLevel}`;
        if (state.thinkingLevel === 'MEDIUM') {
            thinkingLevelToggle.classList.add('active');
        } else {
            thinkingLevelToggle.classList.remove('active');
        }
    });

    document.addEventListener('keyup', (e) => {
        if (!e.ctrlKey && !e.metaKey) {
            isCtrlPressed = false;
        }
        if (!e.altKey) {
            isAltPressed = false;
        }
    });
    
    // --- IndexedDB Helper ---
    // Moved to src/core/session.js


    // === 修改后 ===




    // 删除选中项的函数

    // --- Genealogy Logic ---




// ==================== 工作台右键菜单与文本便签功能 (Migrated to ui/context-menu.js) ====================


// ==================== 工作台拖拽逻辑 ====================



// Context Menu Setup
// ==================== 核心修改 END (2/4) ====================

    async function saveSessionToDB(session) {
        if (dbHelper) {
            return await dbHelper.saveSession(session);
        }
    }

    function serializeWorkspaceSnapshot(workspace) {
        if (!workspace) return null;
        return {
            projectId: workspace.projectId,
            name: workspace.name,
            currentState: {
                stateId: workspace.currentState.stateId,
                canvasState: workspace.currentState.canvasState,
                assets: workspace.currentState.assetRegistry.getAll().map(asset => {
                    const serializableAsset = { ...asset };
                    delete serializableAsset.runtimeDisplayUrl;
                    return serializableAsset;
                }),
                nodes: workspace.currentState.sceneGraph.getNodes(),
                edges: workspace.currentState.sceneGraph.getAllEdges()
            },
            decisionGraph: workspace.decisionGraph.getHistory()
        };
    }

    async function addAssetToLibrary(asset) {
        if (dbHelper) {
            await dbHelper.saveAsset(asset);
            if (typeof renderAssetLibrary === 'function') {
                await renderAssetLibrary();
            }
        }
    }

window.handleResizeStart = handleResizeStart;
window.handleWorkbenchDragStart = handleWorkbenchDragStart;
window.showWorkbenchToolbox = showWorkbenchToolbox;
window.checkProximity = checkProximity;
window.addMessage = addMessage;
window.updateWorkbenchItemsScale = updateWorkbenchItemsScale;
window.resetPan = resetPan;
window.updateZoomIndicator = updateZoomIndicator;
window.applyPanTransform = applyPanTransform;
window.screenToWorkbenchCoord = screenToWorkbenchCoord;
window.updateSelectedItems = updateSelectedItems;
window.saveSessionToDB = saveSessionToDB;
window.addAssetToLibrary = addAssetToLibrary;
window.generateSessionTitle = generateSessionTitle;
window.dbHelper = dbHelper;
window.addImageToWorkbench = addImageToWorkbench;

// Workbench actions update internal assistant context. Only final media/results
// are shown in chat; the action itself is not fabricated as a user message.
window.addWorkbenchActionToChat = async (actionName, prompt, resultImageSrc, retryCallback, resultHtml = null) => {
    const timestamp = Date.now();
    const actionRecord = {
        actionName,
        prompt: prompt || '',
        itemId: state.currentActiveWorkbenchItemId || null,
        hasResult: !!(resultImageSrc || resultHtml),
        timestamp
    };

    recordWorkspaceAction(state, actionRecord);

    let currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    if ((resultImageSrc || resultHtml) && !currentSession) {
        await startNewSession(actionName);
        currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    }
    
    // --- Sync to MVR DecisionGraph ---
    if (window.mvrRuntime) {
        const workspace = window.mvrRuntime.getCurrentWorkspace();
        if (workspace) {
            workspace.decisionGraph.addLog({
                logId: `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                action: 'custom',
                prompt: prompt,
                context: actionName,
                changePayload: actionRecord,
                timestamp
            });
            window.mvrRuntime.saveCurrentWorkspace();
        }
    }
    // --------------------------------

    // Render only the useful final artifact, not a synthetic user/assistant pair.
    if (resultImageSrc || resultHtml) {
        const resultMessage = {
            sender: 'bot',
            type: resultImageSrc ? 'image' : 'html',
            content: `✅ ${actionName}完成。`,
            ...(resultImageSrc ? { imageData: { src: resultImageSrc } } : {}),
            ...(resultHtml ? { content: resultHtml } : {})
        };
        addMessage(resultMessage);
    }

    if (currentSession) {
        renderHistoryList();
        if (dbHelper) await dbHelper.saveSession(currentSession);
    }

    // 更新 lastGenerationContext 以支持重试
    if (retryCallback) {
        state.lastGenerationContext = {
            isWorkbenchAction: true,
            actionName,
            prompt,
            retryCallback
        };
    }
    
    if (typeof triggerAsyncSessionRename === 'function') {
        triggerAsyncSessionRename();
    }
};

window.workbenchGrid = document.getElementById('workbenchGrid');
window.viewAllWorkbenchItems = viewAllWorkbenchItems;
window.applyAtmosphereToImage = applyAtmosphereToImage;
window.updateRelightingPreview = updateRelightingPreview;
window.getIntersectionArea = getIntersectionArea;
window.drawGenealogyConnections = drawGenealogyConnections;
window.fileToDataURL = fileToDataURL;


    // 显示工具箱浮窗


    window.hideWorkbenchToolbox = hideWorkbenchToolbox;
    function hideWorkbenchToolbox() {
        workbenchToolbox.style.display = 'none';
        // Hiding the toolbar must not clear the selected visual object. The
        // Visual Object surface is selection-driven; the toolbar is only a
        // transient control for standalone images.
        // Turn the legacy floating editor into idle/virtual-hidden instead of completely destroying it.
        const floatingEditor = document.getElementById('floatingFusionEditor');
        if (floatingEditor) {
            floatingEditor.classList.add('idle');
        }
        import('./ui/ai-suggestions.js').then(({ hideAISuggestionCompanion }) => {
            hideAISuggestionCompanion();
        }).catch(err => {});
    }





    // (Migrated to src/ui/workbench/items.js)



    function getWorkbenchItemContextSource(item) {
        if (!item) return null;
        const candidate = item.file || item.dataUrl || item.originalDataUrl || item.el?.querySelector('img')?.src || null;
        if (typeof candidate === 'string' && isInvalidImageSrc(candidate)) {
            return null;
        }
        return candidate || null;
    }

    function hasMatchingContextSource(collection, source) {
        return collection.some(existing => existing === source);
    }

    async function handleSelectedItems() {
        if (selectedWorkbenchItems.size === 0) {
            return { syncedCount: 0, skippedCount: 0, selectedImageCount: 0 };
        }
        
        const selectedImageItems = Array.from(selectedWorkbenchItems)
            .map(id => workbenchItems.get(id))
            .filter(item => item)
            .map(item => ({ item, source: getWorkbenchItemContextSource(item) }))
            .filter(({ source }) => !!source);

        const skippedCount = selectedWorkbenchItems.size - selectedImageItems.length;

        if (selectedImageItems.length === 0) {
            updateSendBtnState();
            return { syncedCount: 0, skippedCount, selectedImageCount: 0 };
        }

        let syncedCount = 0;

        if (mainImageFile) {
            // 已经有主图，所有选中项都作为参考图
            selectedImageItems.forEach(({ source }) => {
                if (source !== mainImageFile && !hasMatchingContextSource(referenceImageFiles, source)) {
                    referenceImageFiles.push(source);
                    pendingReferenceImageShares.add(source);
                    syncedCount += 1;
                }
            });
        } else {
            // 没有主图，第一个选中项作为主图，其他作为参考图
            const [firstItem, ...otherItems] = selectedImageItems;
            
            mainImageFile = firstItem.source;
            pendingBaseImageShare = true;
            syncedCount += 1;
            
            otherItems.forEach(({ source }) => {
                if (!hasMatchingContextSource(referenceImageFiles, source)) {
                    referenceImageFiles.push(source);
                    pendingReferenceImageShares.add(source);
                    syncedCount += 1;
                }
            });
        }
        
        await updateImagePreview();
        updateSendBtnState();
        return { syncedCount, skippedCount, selectedImageCount: selectedImageItems.length };
    }

// 手动将当前选中的图片同步到聊天上下文
async function pushSelectedToChat() {
    if (selectedWorkbenchItems.size === 0) return;
    
    // 1. 执行原有的同步逻辑
    const { syncedCount, skippedCount, selectedImageCount } = await handleSelectedItems();

    if (syncedCount === 0) {
        let noImageMessage = '⚠️ 选中的项目里没有可加入对话的图片。';
        if (selectedImageCount > 0) {
            noImageMessage = skippedCount > 0
                ? `⚠️ 选中的图片已经在对话上下文里了，另有 ${skippedCount} 个非图片项目未加入。`
                : '⚠️ 选中的图片已经在对话上下文里了。';
        }
        addMessage({ sender: 'bot', content: noImageMessage });
        return;
    }
    
    // 2. 视觉反馈：选中的图片闪烁绿光，表示“成功收录”
    selectedWorkbenchItems.forEach(id => {
        const item = workbenchItems.get(id);
        if (item && item.el) {
            const originalShadow = item.el.style.boxShadow;
            const originalBorder = item.el.style.borderColor;
            
            item.el.style.boxShadow = "0 0 20px #4CAF50";
            item.el.style.borderColor = "#4CAF50";
            
            setTimeout(() => {
                item.el.style.boxShadow = originalShadow;
                item.el.style.borderColor = originalBorder;
            }, 800);
        }
    });

    // 3. 提示用户
    const skippedSuffix = skippedCount > 0 ? `，另有 ${skippedCount} 个非图片项目未加入` : '';
    addMessage({ sender: 'bot', content: `✅ 已将选中的 ${syncedCount} 张图片加入对话上下文${skippedSuffix}。` });
    
    // 4. (可选) 推送后清除工作台的蓝色选中状态，保持干净
    // selectedWorkbenchItems.clear();
    // document.querySelectorAll('.workbench-item').forEach(el => el.classList.remove('selected'));
}

    // --- Core Functions ---
    
    

    
    // --- Latent Canvas Functions ---
    // Moved to src/graphics/latent-canvas.js

    // --- Multi-Agent Debate Logic ---
    // Moved to src/ui/debate.js

    // --- UI Interaction Logic ---
    function updateSendBtnState() {
        const effectiveBaseImage = mainImageFile || lastGeneratedImageForEditing;
        sendBtn.disabled = isSending || (userInput.value.trim().length === 0 && !effectiveBaseImage);
    }

    function handleSuggestionClick(prompt) {
        userInput.value = prompt;
        userInput.focus();
        updateSendBtnState();
    }
    

    async function createImagePreviewItem(file, label, options = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'preview-item-wrapper';
        const fileId = typeof file === 'string' ? file : (file.name + file.lastModified);
        wrapper.dataset.fileId = fileId;
        wrapper.dataset.previewRole = options.role || '';
        if (options.role === 'ref') wrapper.dataset.refIndex = options.refIndex;

        const img = document.createElement('img');
        img.src = await fileToDataURL(file);
        img.className = 'generated-image';
        wrapper.appendChild(img);
        
        if (label) {
            const labelEl = document.createElement('div');
            labelEl.className = 'preview-item-label';
            labelEl.textContent = label;
            wrapper.appendChild(labelEl);
        }

        const removeBtn = document.createElement('button');
        removeBtn.innerHTML = '&times;';
        removeBtn.style.cssText = 'position: absolute; top: -5px; right: -5px; background: rgba(0,0,0,0.6); color: white; border: none; border-radius: 50%; width: 18px; height: 18px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px; line-height: 1; z-index: 10;';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            if (file === mainImageFile) {
                mainImageFile = null; maskDataUrl = null; pendingBaseImageShare = false;
            } else if (file === lastGeneratedImageForEditing) {
                isContextPreviewHidden = true;
            } else {
                referenceImageFiles = referenceImageFiles.filter(f => f !== file);
                pendingReferenceImageShares.delete(file);
            }
            updateImagePreview(); updateSendBtnState();
        };
        wrapper.appendChild(removeBtn);
        
        if (label === 'Base' || label === 'Context') {
             if (maskDataUrl) wrapper.classList.add('has-mask');
            const maskBtn = document.createElement('button');
            maskBtn.className = 'mask-edit-btn';
            maskBtn.textContent = maskDataUrl ? '编辑蒙版' : '绘制蒙版';
            maskBtn.onclick = (e) => { e.stopPropagation(); openMaskEditor(file); };
            wrapper.appendChild(maskBtn);
        }

        const saveBtn = document.createElement('button');
        saveBtn.className = 'save-to-library-btn';
        saveBtn.innerHTML = '<i class="fas fa-star"></i>';
        const isSaved = userAssets.some(asset => asset.originalId === fileId);
        if (isSaved) saveBtn.classList.add('saved');
        
        saveBtn.onclick = async (e) => {
            e.stopPropagation();
            const dataUrl = img.src;
            const asset = { id: Date.now(), dataUrl: dataUrl, originalId: fileId };
            await dbHelper.saveAsset(asset);
            await renderAssetLibrary();
            saveBtn.classList.add('saved');
            alert('素材已保存到我的素材库！');
        };
        wrapper.appendChild(saveBtn);

        wrapper.draggable = true;
        wrapper.addEventListener('dragstart', (event) => handlePreviewDragStart(event, { role: options.role || 'ref', index: options.refIndex ?? 0, fileId: wrapper.dataset.fileId }));
        wrapper.addEventListener('dragenter', handlePreviewDragEnter);
        wrapper.addEventListener('dragleave', handlePreviewDragLeave);
        wrapper.addEventListener('dragover', handlePreviewDragOver);
        wrapper.addEventListener('drop', (event) => handlePreviewDrop(event, { role: options.role || 'ref', index: options.refIndex ?? 0 }));
        wrapper.addEventListener('dragend', handlePreviewDragEnd);
        wrapper.addEventListener('touchstart', (event) => handleTouchDragStart(event, { role: options.role, refIndex: options.refIndex }));

        return wrapper;
    }
    


    window.updateImagePreview = updateImagePreview;
    window.updateSendBtnState = updateSendBtnState;
    window.handleSend = handleSend;

    async function updateImagePreview() {
        mainImagePreview.innerHTML = '';
        let hasContent = false;
        
        if (mainImageFile) {
            mainImagePreview.appendChild(await createImagePreviewItem(mainImageFile, 'Base', { role: 'base' }));
            hasContent = true;
        } else if (lastGeneratedImageForEditing && !isContextPreviewHidden) {
            mainImagePreview.appendChild(await createImagePreviewItem(lastGeneratedImageForEditing, 'Context', { role: 'base' }));
            hasContent = true;
        }

        for (const [index, file] of referenceImageFiles.entries()) {
            mainImagePreview.appendChild(await createImagePreviewItem(file, 'Ref', { role: 'ref', refIndex: index }));
            hasContent = true;
        }
        
        mainImagePreview.style.display = hasContent ? 'flex' : 'none';
    }
    
    function renderHistoryList() {
        historyList.innerHTML = '';
        if (sessions.length === 0) {
            emptyState.style.display = 'flex';
        } else {
            emptyState.style.display = 'none';
            sessions.forEach(session => {
                const li = document.createElement('li');
                li.dataset.sessionId = session.id;
                if (session.id === currentSessionId) li.classList.add('active');
                li.innerHTML = `<span class="history-title">${session.title}</span><button class="history-more-btn">...</button>`;
                historyList.appendChild(li);
            });
        }
    }
    window.renderHistoryList = renderHistoryList;
    
    // --- 修改后的 loadSession 开头清除逻辑 ---
    // Moved to src/core/session.js

    function isSearchGroundingNeeded(text) {
        const keywords = ['新款', '最新', '当前', '最近', '2024', '新闻', '事件'];
        return keywords.some(keyword => text.includes(keyword));
    }

    function handleRegenerate() {
        if (isSending || !lastGenerationContext) return;
        
        const actualInput = userInput.value.trim();
        const displayContent = actualInput || '[ 指令: 重新生成 ]';

        if (lastGenerationContext.isWorkbenchAction && typeof lastGenerationContext.retryCallback === 'function') {
            const currentSession = sessions.find(s => s.id === currentSessionId);
            if (currentSession) {
                const userMsg = { sender: 'user', type: 'text', content: displayContent };
                currentSession.messages.push(userMsg);
                addMessage(userMsg);
            }
            
            const newPrompt = mergeRegeneratePrompt(
                lastGenerationContext.resolvedPrompt || lastGenerationContext.prompt,
                actualInput
            );
            userInput.value = '';
            userInput.style.height = '';
            
            lastGenerationContext.retryCallback(newPrompt);
            return;
        }

        isSending = true; 
        updateSendBtnState();
        
        const { baseImage, referenceImages, mask, prompt, corePrompt, isGenTask, isEditTask } = lastGenerationContext;
        const currentSession = sessions.find(s => s.id === currentSessionId);
        if (!currentSession) { isSending = false; updateSendBtnState(); return; }

        const regenUserMessage = { sender: 'user', type: 'text', content: displayContent };
        currentSession.messages.push(regenUserMessage); 
        addMessage(regenUserMessage);
        
        userInput.value = '';
        userInput.style.height = '';
        
        let placeholderDiv = document.createElement('div');
        placeholderDiv.className = 'message bot-message';
        const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
        
        const requestedImageCount = getExplicitRequestedImageCount(corePrompt || prompt, 6);
        let placeholderText = requestedImageCount > 1 ? "正在为您生成多个版本..." : "收到，正在为您重新生成...";
        
        let placeholderContent = `<div class="image-placeholder-container"><div class="image-placeholder-box"><div class="spinner"></div></div><div class="placeholder-text">${placeholderText}</div></div>`;
        placeholderDiv.innerHTML = botAvatar + placeholderContent;
        chatMessages.appendChild(placeholderDiv); 
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        (async () => {
            try {
                const retryCorePrompt = mergeRegeneratePrompt(corePrompt || prompt, actualInput);
                const retryPrompt = composeExecutionPrompt(
                    { route: isEditTask ? 'image_edit' : 'image_generation' },
                    {
                        prompt: retryCorePrompt,
                        baseImage,
                        referenceImages,
                        mask,
                        history: currentSession.messages,
                        workingMemory: currentSession.workingMemory,
                        workspaceContext: getAssistantWorkspaceContext(state)
                    }
                );
                let result;
                if (isEditTask) {
                    if (requestedImageCount > 1) result = await editOrQueryImageWithGemini_Multiple(retryPrompt, baseImage, referenceImages, mask, null, requestedImageCount);
                    else result = await editOrQueryImageWithGemini(retryPrompt, baseImage, referenceImages, mask);
                } else if (isGenTask) {
                    result = await generateImage(retryPrompt, '1:1', {
                        baseImage,
                        referenceImages,
                        history: currentSession.messages,
                        imageCount: requestedImageCount
                    });
                } else throw new Error("Regeneration is only available for image tasks.");
                
                if (result.success) {
                    const botMessage = { sender: 'bot', type: 'bot-rich' };

                    if (Array.isArray(result.imageData)) {
                        botMessage.imageData = await Promise.all(result.imageData.map(async (img) => {
                            const originalSrc = `data:${img.mimeType};base64,${img.imageData}`;
                            return { src: await addWatermark(originalSrc), mimeType: img.mimeType };
                        }));
                        botMessage.content = result.text || '已为您生成多个版本，请选用一张以继续。';
                        
                        // 【同步点1】第一时间显示聊天框内容
                        placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                        currentSession.messages.push(botMessage);
                        await updateImagePreview();

                        // 【同步点2】瞬间同步至工作台 (使用同步转换器)
                        botMessage.imageData.forEach(async (imgData) => {
                            try {
                                const file = dataURLtoFileSync(imgData.src, `regen-${Date.now()}.png`);
                                await addImageToWorkbench(file, 'AI生成', {
                                    dataUrl: imgData.src,
                                    parentId: baseImage ? fileToWorkbenchIdMap.get(baseImage) : null,
                                    generationParams: { prompt: retryPrompt }
                                });
                            } catch(e) {}
                        });

                        // 【后台任务】非阻塞加载建议按钮
                        getSmartSuggestions(result.imageData[0].imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                        lastGeneratedImageForEditing = null;

                    } else if (result.imageData) {
                        const watermarkedImgSrc = await addWatermark(`data:${result.mimeType};base64,${result.imageData}`);
                        botMessage.imageData = { src: watermarkedImgSrc };
                        botMessage.content = result.text || buildBotFallbackText(retryPrompt, {isEditTask, isGenTask});

                        // 【同步点1】对话框上屏
                        placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                        currentSession.messages.push(botMessage);
                        await updateImagePreview();

                        // 【同步点2】工作台瞬间同步
                        const newFile = dataURLtoFileSync(watermarkedImgSrc, `regen-${Date.now()}.png`);
                        // 不再自动把 AI 结果设为下一轮对话底图，避免后续纯文本请求被误路由到图片链路
                        lastGeneratedImageForEditing = null;
                        mainImageFile = null;

                        let finalX = 50000, finalY = 50000;
                        const baseId = fileToWorkbenchIdMap.get(baseImage);
                        if (baseId) {
                            const baseItem = workbenchItems.get(baseId);
                            if (baseItem && baseItem.el) {
                                const baseW = parseFloat(baseItem.el.style.width) || 300;
                                finalX = parseFloat(baseItem.el.style.left) + (baseW * 0.35);
                                finalY = parseFloat(baseItem.el.style.top) + (baseW * 0.35);
                            }
                        }

                        await addImageToWorkbench(newFile, '重新生成', {
                            x: finalX, y: finalY,
                            dataUrl: watermarkedImgSrc,
                            parentId: baseId,
                            type: 'regenerate',
                            generationParams: { prompt: retryPrompt }
                        });

                        // 【后台任务】加载建议
                        getSmartSuggestions(result.imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                    }
                }
            } catch (error) {
                placeholderDiv.innerHTML = botAvatar + `<div>重新生成失败: ${error.message}</div>`;
            } finally {
                isSending = false; updateSendBtnState(); await dbHelper.saveSession(currentSession); renderHistoryList();
            }
        })();
    }

    const GENERIC_TITLES = [
        '新建项目', '图片对话', '新对话', 'Untitled', 'New Project', 'New Chat', '开启新对话',
        '图片创作', '上传创作', '图片项目', '未命名项目', '创作项目', '图片编辑', '编辑项目'
    ];

    async function cleanupPlaceholderSessions() {
        const placeholderCandidates = sessions.filter(session => {
            const isGeneric = GENERIC_TITLES.includes(session.title);
            return isGeneric;
        });

        if (placeholderCandidates.length === 0) return;

        const placeholderSessions = [];
        for (const session of placeholderCandidates) {
            const fullSession = await dbHelper.getSessionData(session.id);
            // 云端清单不包含重数据；详情尚未加载时不能判定为本地空会话。
            if (!fullSession) continue;
            const mergedSession = { ...session, ...(fullSession || {}) };
            const messageCount = Array.isArray(mergedSession.messages) ? mergedSession.messages.length : 0;
            const runtimeAssetCount = Array.isArray(mergedSession.runtimeWorkspace?.currentState?.assets)
                ? mergedSession.runtimeWorkspace.currentState.assets.length
                : 0;
            const legacyImageCount = Array.isArray(mergedSession.workbenchState)
                ? mergedSession.workbenchState.filter(item => item?.dataUrl).length
                : 0;

            if (messageCount === 0 && runtimeAssetCount === 0 && legacyImageCount === 0) {
                placeholderSessions.push(session);
            }
        }

        if (placeholderSessions.length === 0) return;

        for (const session of placeholderSessions) {
            await dbHelper.deleteSession(session.id);
        }

        sessions = sessions.filter(session => !placeholderSessions.some(item => item.id === session.id));
    }

    async function ensureSessionExists(initialTitle = '新建项目') {
        if (!state.currentSessionId) {
            state.currentSessionId = `session-${Date.now()}`;
            
            const isAutoRenamed = !GENERIC_TITLES.includes(initialTitle);

            const newSession = { 
                id: state.currentSessionId, 
                title: initialTitle, 
                timestamp: Date.now(), 
                messages: [],
                isAutoRenamed: isAutoRenamed,
                workingMemory: createDefaultWorkingMemory()
            };
            sessions.unshift(newSession);
            await dbHelper.saveSession(newSession);
            setLastActiveSessionId(newSession.id);
            renderHistoryList();
            return true;
        }
        return false;
    }

    async function appendAgentTaskUserMessage(text) {
        await ensureSessionExists('资产任务');
        const currentSession = sessions.find(session => session.id === currentSessionId);
        if (!currentSession) return;
        ensureSessionWorkingMemory(currentSession);
        const userMessage = { sender: 'user', type: 'text', content: text };
        currentSession.messages.push(userMessage);
        addMessage(userMessage);
        currentSession.updatedAt = Date.now();
        userInput.value = '';
        userInput.style.height = '';
        renderHistoryList();
        await dbHelper.saveSession(currentSession);
    }

    async function persistAgentTaskResult() {
        const currentSession = sessions.find(session => session.id === currentSessionId);
        if (currentSession) {
            currentSession.updatedAt = Date.now();
            await dbHelper.saveSession(currentSession);
            renderHistoryList();
        }
    }

    let renameTimeout = null;
    async function triggerAsyncSessionRename() {
        if (!state.currentSessionId) return;
        
        if (renameTimeout) clearTimeout(renameTimeout);
        
        renameTimeout = setTimeout(async () => {
            const currentSession = sessions.find(s => s.id === state.currentSessionId);
            if (!currentSession) return;
            if (!Array.isArray(currentSession.messages)) {
                currentSession.messages = [];
            }
            
            if (currentSession.isAutoRenamed) {
                return; // Already auto-renamed by a real prompt or action
            }

            try {
                let contextText = "";
                let isRealPrompt = false;
                
                const actionKeywords = ['正在', '完成', '生成', '提取', '拆解', '融合', '扩图', '重光照', '重绘', '编辑', '处理'];
                
                // Get text from recent messages, ignoring system-like bracketed messages
                const meaningfulMsgs = currentSession.messages.filter(m => {
                    if (m.type !== 'text') return false;
                    const text = m.content.trim();
                    if (text.includes('[ 已选用图片 ]')) return false;
                    if (text.includes('图片已上传')) return false;
                    if (m.sender === 'user') return true;
                    // For bot messages, check if it contains action keywords
                    if (m.sender === 'bot' && actionKeywords.some(kw => text.includes(kw))) return true;
                    return false;
                });

                const recentMsgs = meaningfulMsgs.slice(-3);
                
                if (recentMsgs.length > 0) {
                    contextText = recentMsgs.map(m => `${m.sender === 'user' ? '用户' : '系统'}: ${m.content}`).join('\n');
                    isRealPrompt = true;
                }
                
                // If no real prompt, try to describe workbench
                if (!contextText && workbenchItems.size > 0) {
                    contextText = "用户上传了图片并正在进行创作";
                    isRealPrompt = false;
                }

                if (!contextText) return;
                
                // Avoid redundant calls if context hasn't changed for this session
                if (currentSession.lastRenameContext === contextText) return;

                const smartTitle = await generateSessionTitle(contextText);
                if (smartTitle) {
                    currentSession.title = smartTitle;
                    currentSession.lastRenameContext = contextText;
                    currentSession.updatedAt = Date.now();
                    
                    // Only lock the name if it was generated from a real user prompt or tool action.
                    if (isRealPrompt) {
                        currentSession.isAutoRenamed = true; 
                    }
                    
                    await dbHelper.saveSession(currentSession);
                    renderHistoryList();
                    if (currentSession.id === state.currentSessionId) updateHeader(currentSession);
                }
            } catch (e) {
                console.error("Async rename failed:", e);
            }
        }, 2000); // 2 seconds delay to wait for operations to settle
    }

    window.triggerAsyncSessionRename = triggerAsyncSessionRename;
    window.ensureSessionExists = ensureSessionExists;

    async function executeWorkflow(textFromInput, userMessages, baseImageDataUrl, referenceImages = [], maskDataUrl = null) {
        if (!textFromInput && (!userMessages || userMessages.length === 0)) return;

        isSending = true;
        updateSendBtnState();

        // 1. Add user message to chat
        await ensureSessionExists('AI 流程');
        const currentSession = sessions.find(s => s.id === currentSessionId);
        const hadSessionMessages = currentSession.messages.length > 0;
        
        let messagesToAppend = userMessages || [];
        if (userMessages && userMessages.length > 0) {
            userMessages.forEach(msg => {
                if (msg.type === 'text') {
                    msg.content = `[AI 流程策划] ${msg.content}`;
                }
                currentSession.messages.push(msg);
            });
        } else {
            const userMsg = { sender: 'user', type: 'text', content: `[AI 流程策划] ${textFromInput}` };
            currentSession.messages.push(userMsg);
            messagesToAppend = [userMsg];
        }
        
        if (hadSessionMessages) {
            messagesToAppend.forEach(message => addMessage(message));
        } else {
            renderMessages(currentSession.messages);
        }
        userInput.value = '';
        userInput.style.height = '';
        
        // Clear image states
        mainImageFile = null;
        pendingBaseImageShare = false;
        pendingReferenceImageShares.clear();
        mainImagePreview.innerHTML = '';
        mainImagePreview.style.display = 'none';

        // 2. Show planning placeholder
        let placeholderDiv = document.createElement('div');
        placeholderDiv.className = 'message bot-message';
        const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
        placeholderDiv.innerHTML = botAvatar + `<div class="placeholder-text">正在策划 AI 流程...</div>`;
        chatMessages.appendChild(placeholderDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        await yieldToChatPaint();

        try {
            // 3. Get Selected Scenario
            const selectedScenario = window.selectedScenarioType || 'interior';
            
            let plan = { nodes: [], edges: [] };
            let scenarioTitle = "";

            if (selectedScenario === 'interior') {
                scenarioTitle = "室内设计提案全案";
                plan.nodes = [
                    { id: "node1", type: baseImageDataUrl ? "edit-image" : "generate-image", inputs: { prompt: textFromInput || "优化为极简侘寂风室内设计效果图，提升光影质感" } },
                    { id: "node2", type: "multiview", inputs: { prompt: "生成不同视角的九宫格展示" } },
                    { id: "node3", type: "analyze", inputs: { agentType: "suggest" } },
                    { id: "node4", type: "video-generation", inputs: { prompt: "空间漫游，光影流转" } }
                ];
                plan.edges = [
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node2", inputId: "image" } },
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node3", inputId: "image" } },
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node4", inputId: "image" } }
                ];
            } else if (selectedScenario === 'ecommerce') {
                scenarioTitle = "电商爆款上新全案";
                plan.nodes = [
                    { id: "node1", type: baseImageDataUrl ? "edit-image" : "generate-image", inputs: { prompt: textFromInput || "生成带高级背景的商品海报，突出质感" } },
                    { id: "node2", type: "analyze", inputs: { agentType: "describe" } },
                    { id: "node3", type: "edit-image", inputs: { prompt: "生成该商品的3种其他颜色变体" } },
                    { id: "node4", type: "video-generation", inputs: { prompt: "商品展示动态视频，镜头拉近，光效扫过" } }
                ];
                plan.edges = [
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node2", inputId: "image" } },
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node3", inputId: "image" } },
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node4", inputId: "image" } }
                ];
            } else if (selectedScenario === 'ip_creation') {
                scenarioTitle = "自媒体/IP角色孵化全案";
                plan.nodes = [
                    { id: "node1", type: baseImageDataUrl ? "edit-image" : "generate-image", inputs: { prompt: textFromInput || "超高清角色正面立绘" } },
                    { id: "node2", type: "multiview", inputs: { prompt: "生成角色的三视图（正、侧、背）和表情包" } },
                    { id: "node3", type: "analyze", inputs: { agentType: "describe" } }
                ];
                plan.edges = [
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node2", inputId: "image" } },
                    { from: { nodeId: "node1", outputId: "image" }, to: { nodeId: "node3", inputId: "image" } }
                ];
            }

            if (!plan || !plan.nodes) throw new Error("无法生成场景全案计划");

            // 4. Update placeholder with plan details
            let planHtml = `<div class="workflow-plan">
                <div style="font-weight: bold; margin-bottom: 8px;">已为您启动【${scenarioTitle}】：</div>
                <ul style="margin: 0; padding-left: 20px;">
                    ${plan.nodes.map((n, i) => `<li>步骤 ${i+1}: ${NODE_REGISTRY[n.type]?.title || n.type}</li>`).join('')}
                </ul>
                <div style="margin-top: 8px; color: #666; font-size: 12px;">正在按场景预设自动执行...</div>
            </div>`;
            placeholderDiv.innerHTML = botAvatar + planHtml;

            // 5. Build and Run Graph
            graphStore.nodes.clear();
            graphStore.edges = [];
            graphStore.results.clear();

            plan.nodes.forEach(n => graphStore.addNode(n));
            plan.edges.forEach(e => {
                let from = e.from;
                let to = e.to;
                
                if (typeof from === 'string') from = { nodeId: from, outputId: 'image' };
                else if (from && !from.nodeId && from.id) from.nodeId = from.id;
                
                if (typeof to === 'string') to = { nodeId: to, inputId: 'image' };
                else if (to && !to.nodeId && to.id) to.nodeId = to.id;
                
                if (from && !from.outputId) from.outputId = 'image';
                if (to && !to.inputId) to.inputId = 'image';
                
                if (from && to && from.nodeId && to.nodeId) {
                    graphStore.addEdge(from, to);
                }
            });

            // Remove edges from non-existent nodes (e.g., if planner hallucinated an "input" node)
            graphStore.edges = graphStore.edges.filter(e => graphStore.nodes.has(e.from.nodeId) && graphStore.nodes.has(e.to.nodeId));

            // Set initial inputs if needed (e.g., if the first node needs an image from the workbench)
            if (baseImageDataUrl) {
                const imageNodes = Array.from(graphStore.nodes.values()).filter(n => 
                    NODE_REGISTRY[n.type]?.inputs.some(i => i.id === 'image')
                );
                
                imageNodes.forEach(node => {
                    const hasIncomingImageEdge = graphStore.edges.some(e => e.to.nodeId === node.id && e.to.inputId === 'image');
                    if (!hasIncomingImageEdge) {
                        node.inputs = node.inputs || {};
                        node.inputs.image = baseImageDataUrl;
                    }
                });
            }

            if (referenceImages && referenceImages.length > 0) {
                const refNodes = Array.from(graphStore.nodes.values()).filter(n => 
                    NODE_REGISTRY[n.type]?.inputs.some(i => i.id === 'referenceImages')
                );
                refNodes.forEach(node => {
                    node.inputs = node.inputs || {};
                    node.inputs.referenceImages = referenceImages;
                });
            }

            if (maskDataUrl) {
                const maskNodes = Array.from(graphStore.nodes.values()).filter(n => 
                    NODE_REGISTRY[n.type]?.inputs.some(i => i.id === 'mask')
                );
                maskNodes.forEach(node => {
                    node.inputs = node.inputs || {};
                    node.inputs.mask = maskDataUrl;
                });
            }
            
            await graphRunner.runGraph();

            // 6. Show final results (All nodes)
            let content = `【${scenarioTitle}】执行完毕！\n\n以下是为您生成的全套方案：\n\n`;
            let finalImageResult = null;
            
            plan.nodes.forEach((node, index) => {
                const result = graphStore.getNodeResult(node.id);
                const nodeTitle = NODE_REGISTRY[node.type]?.title || node.type;
                content += `**步骤 ${index + 1}: ${nodeTitle}**\n`;
                
                if (result) {
                    if (result.image) {
                        content += `[图像已生成，请在工作台查看]\n`;
                        finalImageResult = result.image; // Keep track of the last image generated
                    } else if (result.video) {
                        content += `[视频已生成，请在工作台查看]\n`;
                    } else if (result.analysis) {
                        let analysisText = typeof result.analysis === 'object' ? JSON.stringify(result.analysis, null, 2) : result.analysis;
                        content += `${analysisText}\n`;
                    } else if (result.layers) {
                        content += `共发现 ${result.layers.length} 个视觉元素。\n`;
                    } else {
                        content += `执行成功。\n`;
                    }
                } else {
                    content += `执行失败或无输出。\n`;
                }
                content += `\n`;
            });

            const botMessage = { 
                sender: 'bot', 
                type: 'bot-rich', 
                content: content 
            };

            if (finalImageResult) {
                const watermarkedImgSrc = await addWatermark(finalImageResult);
                botMessage.imageData = { src: watermarkedImgSrc };
                
                // Add to workbench
                const newFile = dataURLtoFileSync(watermarkedImgSrc, `workflow-${Date.now()}.png`);
                await addImageToWorkbench(newFile, '全案生成', {
                    dataUrl: watermarkedImgSrc,
                    generationParams: { prompt: textFromInput, workflow: plan }
                });
            } else if (finalResult && finalResult.video) {
                const videoHtml = `
                    <div style="margin-top: 10px; width: 100%;">
                        <video src="${finalResult.video}" controls autoplay loop muted playsinline 
                            style="width: 100%; max-width: 300px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); background: #000;">
                        </video>
                        <div style="font-size: 12px; color: #666; margin-top: 8px; display:flex; justify-content:space-between; align-items:center;">
                            <span>✅ 视频生成成功</span>
                            <a href="${finalResult.video}" download="veo-${Date.now()}.mp4" style="color:#8b5cf6; text-decoration:none; font-weight:500;">
                                <i class="fas fa-download"></i> 下载
                            </a>
                        </div>
                    </div>
                `;
                botMessage.content += videoHtml;
            }

            placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
            currentSession.messages.push(botMessage);

        } catch (error) {
            console.error("Workflow failed:", error);
            let errorHtml = `<div style="color: #d32f2f;">流程执行失败: ${error.message}</div>`;
            
            if (error.message.includes("PERMISSION_DENIED") || error.message.includes("permission")) {
                errorHtml += `
                    <div style="margin-top: 12px; padding: 12px; background: #fff5f5; border: 1px solid #feb2b2; border-radius: 8px; font-size: 13px;">
                        <p style="margin: 0 0 8px 0; font-weight: bold;">⚠️ 权限不足</p>
                        <p style="margin: 0 0 12px 0;">生成高清图像或视频需要连接您的付费 API Key。</p>
                        <button onclick="window.aistudio.openSelectKey()" style="background: #8b5cf6; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                            连接 API Key
                        </button>
                    </div>
                `;
            }
            placeholderDiv.innerHTML = botAvatar + errorHtml;
        } finally {
            isSending = false;
            updateSendBtnState();
            await dbHelper.saveSession(currentSession);
            renderHistoryList();
        }
    }

    async function handleSend() {
        let textFromInput = userInput.value.trim();
        
        if (currentIntentLock === 'material') {
            if (!textFromInput) textFromInput = "替换材质";
            else if (!isMaterialRequest(textFromInput)) textFromInput = "材质替换为：" + textFromInput;
        } else if (currentIntentLock === 'erase') {
            if (!textFromInput) textFromInput = "移除选中区域";
            else if (!isRemovalRequest(textFromInput)) textFromInput = "局部重绘并移除：" + textFromInput;
        }
        currentIntentLock = null;

        // Feature-flagged bridge to the Runtime Job protocol. The existing
        // image/chat pipeline remains untouched when this flag is disabled.
        if (
            isAgentRuntimeFeatureEnabled('assetReuseFromChat') &&
            isAssetReuseRequest(textFromInput) &&
            !isWorkflowMode &&
            !mainImageFile &&
            !lastGeneratedImageForEditing &&
            !pendingBaseImageShare &&
            referenceImageFiles.length === 0
        ) {
            if (isSending) return;
            isSending = true;
            updateSendBtnState();
            try {
                await handleAssetReuseRequest({
                    text: textFromInput,
                    runtime,
                    onUserMessage: appendAgentTaskUserMessage,
                    onFinished: persistAgentTaskResult
                });
            } catch (error) {
                addMessage({ sender: 'bot', content: `资产任务启动失败：${error?.message || error}` });
            } finally {
                isSending = false;
                updateSendBtnState();
            }
            return;
        }

        if (
            isAgentRuntimeFeatureEnabled('magicLayersCommand') &&
            isLayerExtractionRequest(textFromInput) &&
            !isWorkflowMode &&
            !mainImageFile &&
            !lastGeneratedImageForEditing &&
            !pendingBaseImageShare &&
            referenceImageFiles.length === 0
        ) {
            if (isSending) return;
            isSending = true;
            updateSendBtnState();
            try {
                const result = await handleLayerExtractionRequest({
                    text: textFromInput,
                    runtime,
                    onUserMessage: appendAgentTaskUserMessage,
                    onFinished: persistAgentTaskResult
                });
                if (result) {
                    userInput.value = '';
                    userInput.style.height = '';
                    return;
                }
            } catch (error) {
                addMessage({ sender: 'bot', content: `图层提取任务启动失败：${error?.message || error}` });
            } finally {
                isSending = false;
                updateSendBtnState();
            }
            return;
        }

        const regenerationKeywords = ["不满意", "重来", "再试一次", "重新生成", "换一个", "另一个版本"];
        if (regenerationKeywords.some(k => textFromInput.toLowerCase().includes(k)) && lastGenerationContext) {
            handleRegenerate(); return;
        }
        
        // ... (精准修图逻辑保持不变)
        if (preciseEditMode.pendingBox && preciseEditMode.pendingItemId && textFromInput) {
            const promptText = textFromInput.replace(/^修改框选区域：\s*/, '');
            if (promptText) {
                if (!currentSessionId) {
                    await ensureSessionExists('精准修图');
                }
                const currentSession = sessions.find(s => s.id === currentSessionId);
                const hadSessionMessages = currentSession.messages.length > 0;
                const userMsg = { sender: 'user', type: 'text', content: textFromInput };
                currentSession.messages.push(userMsg);
                if (hadSessionMessages) addMessage(userMsg);
                else renderMessages(currentSession.messages);
                renderHistoryList();
                userInput.value = ''; userInput.style.height = '';
                const box = preciseEditMode.pendingBox; const itemId = preciseEditMode.pendingItemId;
                preciseEditMode.pendingBox = null; preciseEditMode.pendingItemId = null;
                await yieldToChatPaint();
                await performPreciseEdit(itemId, box, promptText); return;
            }
        }
        
        let effectiveBaseImage = mainImageFile || lastGeneratedImageForEditing;
        if (isSending || (!textFromInput && !effectiveBaseImage)) return;
        
        if (isWorkflowMode) {
            let userMessages = [];
            if (textFromInput) userMessages.push({ sender: 'user', type: 'text', content: textFromInput });
            if (pendingBaseImageShare && mainImageFile) {
                userMessages.push({ sender: 'user', type: 'image', imageData: { src: await fileToDataURL(mainImageFile) } });
            }
            for (const file of [...pendingReferenceImageShares]) {
                userMessages.push({ sender: 'user', type: 'image', imageData: { src: await fileToDataURL(file) } });
            }
            const baseImageDataUrl = effectiveBaseImage ? await fileToDataURL(effectiveBaseImage) : null;
            
            const referenceImagesSent = [...referenceImageFiles];
            const maskSent = maskDataUrl;
            
            await executeWorkflow(textFromInput, userMessages, baseImageDataUrl, referenceImagesSent, maskSent);
            
            mainImageFile = null; referenceImageFiles = []; maskDataUrl = null;
            pendingBaseImageShare = false; pendingReferenceImageShares.clear();
            await updateImagePreview();
            
            return;
        }

        isSending = true; updateSendBtnState();

        let targetAspectRatio = '1:1';
        let targetFusionProperties = null;

        if (state.pendingImplicitMemoryOverride) {
            if (state.pendingImplicitMemoryOverride.aspectRatio) {
                targetAspectRatio = state.pendingImplicitMemoryOverride.aspectRatio;
            }
            if (state.pendingImplicitMemoryOverride.fusionProperties) {
                targetFusionProperties = JSON.parse(JSON.stringify(state.pendingImplicitMemoryOverride.fusionProperties));
            }
            // Clear so it is a one-off preloading
            state.pendingImplicitMemoryOverride = null;
            console.log("[ImplicitMemory] Applied preloaded parameter overrides:", { targetAspectRatio, targetFusionProperties });
        }

        // 构建用户消息
        let userMessages = [];
        if (textFromInput) userMessages.push({ sender: 'user', type: 'text', content: textFromInput });
        if (pendingBaseImageShare && mainImageFile) {
            userMessages.push({ sender: 'user', type: 'image', imageData: { src: await fileToDataURL(mainImageFile) } });
            pendingBaseImageShare = false;
        }
        for (const file of [...pendingReferenceImageShares]) {
            userMessages.push({ sender: 'user', type: 'image', imageData: { src: await fileToDataURL(file) } });
            pendingReferenceImageShares.delete(file);
        }
        
        if (!currentSessionId) {
            await ensureSessionExists('图片对话');
        }

        const currentSession = sessions.find(s => s.id === currentSessionId);
        ensureSessionWorkingMemory(currentSession);
        const hadSessionMessages = currentSession.messages.length > 0;
        currentSession.updatedAt = Date.now();
        currentSession.messages.push(...userMessages);
        if (hadSessionMessages) {
            userMessages.forEach(message => addMessage(message));
        } else {
            renderMessages(currentSession.messages);
        }
        renderHistoryList();
        userInput.value = ''; userInput.style.height = '';

        const qwenRoutingContext = {
            model: getImageModel(),
            text: textFromInput,
            baseImage: effectiveBaseImage,
            mask: maskDataUrl
        };
        const useQwenLocalImageRouting = shouldUseQwenLocalImageRouting(qwenRoutingContext);
        console.log('[Qwen Local Routing] decision', getQwenLocalRoutingDiagnostics(qwenRoutingContext));
        if (!useQwenLocalImageRouting) triggerAsyncSessionRename();
        
        const referenceImagesSent = [...referenceImageFiles];
        const maskSent = maskDataUrl;
        let placeholderDiv = document.createElement('div');
        placeholderDiv.className = 'message bot-message';
        const botAvatar = `<img src="https://www.marmoai.cn/images/avatars/WeChat84b8e05cc8464bb089de1c46bed38809.jpg" alt="小M" style="width:32px;height:32px;border-radius:50%; flex-shrink: 0;">`;
        placeholderDiv.innerHTML = botAvatar + `<div>思考中...</div>`;
        chatMessages.appendChild(placeholderDiv); chatMessages.scrollTop = chatMessages.scrollHeight;
        await yieldToChatPaint();
        
        mainImageFile = null; referenceImageFiles = []; maskDataUrl = null;
        await updateImagePreview();

        try {
            let workspaceContext = getAssistantWorkspaceContext(state);
            const interpretation = await interpretConversationTurn({
                text: textFromInput,
                baseImage: effectiveBaseImage,
                referenceImages: referenceImagesSent,
                mask: maskSent,
                history: currentSession.messages,
                workingMemory: currentSession.workingMemory,
                selectedModel: state.selectedModel,
                workspaceContext,
                skipModel: useQwenLocalImageRouting
            });
            const resolvedInterpretation = resolveStateConflict({
                text: textFromInput,
                baseImage: effectiveBaseImage,
                referenceImages: referenceImagesSent,
                mask: maskSent,
                history: currentSession.messages,
                workingMemory: currentSession.workingMemory,
                workspaceContext
            }, interpretation);
            const updatedWorkingMemory = mergeInterpreterResultIntoSession(currentSession, resolvedInterpretation);

            const routeDecision = await determineConversationIntent({
                text: textFromInput,
                baseImage: effectiveBaseImage,
                referenceImages: referenceImagesSent,
                mask: maskSent,
                history: currentSession.messages,
                selectedModel: state.selectedModel,
                workingMemory: updatedWorkingMemory,
                workspaceContext,
                localImageRouting: useQwenLocalImageRouting
            });

            const workspaceCommand = detectWorkspaceCommand(textFromInput);
            if (workspaceCommand) {
                workspaceContext = applyWorkspaceCommand(state, workspaceCommand, textFromInput);
            }
            const lensSearchRequest = looksLikeLensSearchRequest(textFromInput);
            let lensSearchResult = null;

            const isGenTask = routeDecision.route === 'image_generation';
            const isEditTask = routeDecision.route === 'image_edit';
            const isQueryTask = routeDecision.route === 'image_query';
            const shouldFinalizeBriefFromResponse = resolvedInterpretation.briefSource === 'assistant_result' && !isGenTask && !isEditTask;

            const stateObj = historyManager._createSerializableState(state.workbenchItems);
            const canvasState = stateObj.items.map(item => ({
                id: item.id,
                type: item.type,
                label: item.label,
                position: { left: item.style.left, top: item.style.top },
                size: { width: item.style.width, height: item.style.height },
                zIndex: item.style.zIndex,
                content: item.type === 'text-note' ? item.content : undefined
            }));

            const resolvedExecutionPrompt = resolveExecutionPrompt(routeDecision, {
                prompt: textFromInput,
                baseImage: effectiveBaseImage,
                referenceImages: referenceImagesSent,
                mask: maskSent,
                targetAspectRatio: targetAspectRatio || '1:1',
                history: currentSession.messages,
                sessionId: currentSessionId,
                canvasState,
                workingMemory: updatedWorkingMemory,
                workspaceContext
            });
            const executionPrompt = composeExecutionPrompt(routeDecision, {
                prompt: resolvedExecutionPrompt || textFromInput,
                baseImage: effectiveBaseImage,
                referenceImages: referenceImagesSent,
                mask: maskSent,
                targetAspectRatio: targetAspectRatio || '1:1',
                history: currentSession.messages,
                sessionId: currentSessionId,
                canvasState,
                workingMemory: updatedWorkingMemory,
                workspaceContext
            });

            lastGenerationContext = (isGenTask || isEditTask) ? {
                prompt: textFromInput,
                corePrompt: resolvedExecutionPrompt || textFromInput,
                resolvedPrompt: executionPrompt,
                baseImage: effectiveBaseImage,
                referenceImages: [...referenceImagesSent],
                mask: maskSent,
                isGenTask,
                isEditTask
            } : null;

            const placeholderText = lensSearchRequest
                ? '正在搜索同款...'
                : getIntentPlaceholderText(routeDecision, {
                prompt: textFromInput,
                mask: maskSent
            });
            const placeholderContent = shouldUseImagePlaceholder(routeDecision)
                ? `<div class="image-placeholder-container"><div class="image-placeholder-box"><div class="spinner"></div></div><div class="placeholder-text">${placeholderText}</div></div>`
                : `<div>${placeholderText}</div>`;
            placeholderDiv.innerHTML = botAvatar + placeholderContent;

            if (lensSearchRequest) {
                lensSearchResult = await executeLensSearch(state, textFromInput, workspaceContext);
            }

            let result = workspaceCommand
                ? {
                    success: true,
                    text: buildWorkspaceCommandResponse(workspaceCommand, workspaceContext),
                    functionCalls: []
                }
                : lensSearchResult
                ? lensSearchResult
                : await executeConversationIntent(routeDecision, {
                    prompt: resolvedExecutionPrompt || textFromInput,
                    composedPrompt: executionPrompt,
                    baseImage: effectiveBaseImage,
                    referenceImages: referenceImagesSent,
                    mask: maskSent,
                    targetAspectRatio: targetAspectRatio || '1:1',
                    history: currentSession.messages,
                    sessionId: currentSessionId,
                    canvasState,
                    workingMemory: updatedWorkingMemory,
                    workspaceContext
                });
            
            if (routeDecision.route === 'text_chat' && result.functionCalls && result.functionCalls.length > 0) {
                let originalText = result.text;
                let hasEdit = false;
                for (const call of result.functionCalls) {
                    const { name, args } = call;
                    if (name === 'edit_image') {
                        const targetItem = state.workbenchItems.get(args.itemId);
                        if (targetItem && targetItem.file) {
                            effectiveBaseImage = targetItem.file;
                            if (!hasEdit) {
                                placeholderDiv.innerHTML = botAvatar + `<div class="image-placeholder-container"><div class="image-placeholder-box"><div class="spinner"></div></div><div class="placeholder-text">正在编辑图片...</div></div>`;
                                hasEdit = true;
                            }
                            const requestedCount = getExplicitRequestedImageCount(args.prompt || '', 6);
                            result = requestedCount > 1
                                ? await editOrQueryImageWithGemini_Multiple(args.prompt, targetItem.file, [], null, null, requestedCount)
                                : await editOrQueryImageWithGemini(args.prompt, targetItem.file, [], null);
                            if (!result.text) result.text = originalText !== "正在执行操作..." ? originalText : "已为您修改图片。";
                        } else {
                            if (!result.text) result.text = originalText !== "正在执行操作..." ? originalText : "找不到指定的图片进行编辑。";
                        }
                    } else if (name === 'manipulate_item') {
                        const targetItem = state.workbenchItems.get(args.itemId);
                        if (targetItem && targetItem.el) {
                            if (args.action === 'delete') {
                                targetItem.el.remove();
                                state.workbenchItems.delete(args.itemId);
                            } else if (args.action === 'move') {
                                try {
                                    const val = typeof args.value === 'string' ? JSON.parse(args.value) : args.value;
                                    if (val.x !== undefined) targetItem.el.style.left = val.x + 'px';
                                    if (val.y !== undefined) targetItem.el.style.top = val.y + 'px';
                                } catch(e) {}
                            } else if (args.action === 'resize') {
                                try {
                                    const val = typeof args.value === 'string' ? JSON.parse(args.value) : args.value;
                                    if (val.width !== undefined) targetItem.el.style.width = val.width + 'px';
                                    if (val.height !== undefined) targetItem.el.style.height = val.height + 'px';
                                } catch(e) {}
                            }
                        } else {
                            if (!result.text || result.text === "正在执行操作...") result.text = "找不到指定的元素进行操作。";
                        }
                    }
                }
                if (result.functionCalls.some(c => c.name === 'manipulate_item')) {
                    historyManager.pushState();
                    if (result.text === "正在执行操作...") result.text = "已为您更新工作台。";
                }
            }

            if (result.success || result.handled) {
                const botMessage = { sender: 'bot', type: 'bot-rich' };
                
                if (Array.isArray(result.imageData)) {
                    botMessage.imageData = await Promise.all(result.imageData.map(async (img) => {
                        const watermarkedSrc = await addWatermark(`data:${img.mimeType};base64,${img.imageData}`);
                        // 【瞬间同步】工作台
                        try {
                            const file = dataURLtoFileSync(watermarkedSrc, `gen-${Date.now()}.png`);
                            await addImageToWorkbench(file, 'AI生成', {
                                dataUrl: watermarkedSrc,
                                parentId: effectiveBaseImage ? fileToWorkbenchIdMap.get(effectiveBaseImage) : null,
                                generationParams: { prompt: executionPrompt },
                                fusionProperties: targetFusionProperties || undefined
                            });
                        } catch(e){}
                        return { src: watermarkedSrc, mimeType: img.mimeType };
                    }));
                    botMessage.content = result.text || '已为您生成多个版本，请选用一张以继续。';
                    
                    // 对话框上屏
                    placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                    currentSession.updatedAt = Date.now();
                    currentSession.messages.push(botMessage);
                    if (shouldFinalizeBriefFromResponse) {
                        finalizeWorkingMemoryWithAssistantResult(currentSession, resolvedInterpretation, botMessage.content || '');
                    }
                    await updateImagePreview();
                    // 异步建议
                    getSmartSuggestions(result.imageData[0].imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                    lastGeneratedImageForEditing = null;
                } else if (result.imageData) {
                    const watermarkedImgSrc = await addWatermark(`data:${result.mimeType};base64,${result.imageData}`);
                    botMessage.imageData = { src: watermarkedImgSrc };
                    botMessage.content = result.text || buildBotFallbackText(executionPrompt, {isEditTask, isGenTask});
                    
                    // 【同步点1】对话框显示
                    placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                    currentSession.updatedAt = Date.now();
                    currentSession.messages.push(botMessage);
                    if (shouldFinalizeBriefFromResponse) {
                        finalizeWorkingMemoryWithAssistantResult(currentSession, resolvedInterpretation, botMessage.content || '');
                    }
                    await updateImagePreview();

                    // 【同步点2】工作台瞬间显示
                    try {
                        const generatedFile = dataURLtoFileSync(watermarkedImgSrc, `gen-${Date.now()}.png`);
                        // 不再自动把 AI 结果设为下一轮对话底图，避免后续纯文本请求被误路由到图片链路
                        lastGeneratedImageForEditing = null;
                        mainImageFile = null;
                        await addImageToWorkbench(generatedFile, 'AI生成', {
                            dataUrl: watermarkedImgSrc,
                            parentId: effectiveBaseImage ? fileToWorkbenchIdMap.get(effectiveBaseImage) : null,
                            generationParams: { prompt: executionPrompt },
                            fusionProperties: targetFusionProperties || undefined
                        });
                    } catch(e){}
                    
                    // 【后台】加载建议
                    getSmartSuggestions(result.imageData).then(s => appendSuggestionButtons(placeholderDiv, s, handleSuggestionClick));
                } else {
                    botMessage.content = result.text;
                    placeholderDiv.innerHTML = botAvatar + buildMessageContentHTML(botMessage);
                    currentSession.updatedAt = Date.now();
                    currentSession.messages.push(botMessage);
                    if (shouldFinalizeBriefFromResponse) {
                        finalizeWorkingMemoryWithAssistantResult(currentSession, resolvedInterpretation, botMessage.content || '');
                    }
                    if (!isQueryTask) lastGeneratedImageForEditing = null;
                }
            }
        } catch (error) {
            placeholderDiv.innerHTML = botAvatar + `<div>处理失败: ${error.message}</div>`;
        } finally {
            isSending = false; updateSendBtnState(); await dbHelper.saveSession(currentSession); renderHistoryList();
        }
    }
    
    // Removed let closeMobileMenu = () => {};

    function triggerAIActiveGuidance() {
        const currentSession = sessions.find(s => s.id === currentSessionId);
        if (!currentSession || !Array.isArray(currentSession.messages) || currentSession.messages.length > 2) return;

        const messageDiv = addMessage({ sender: 'bot', content: '图片已上传。您可以让我先读图、提炼结构、整理成文本总纲，或者继续编辑与生成：' });
        appendSuggestionButtons(messageDiv, [
            { label: '绘制蒙版并替换', action: () => { if(mainImageFile) openMaskEditor(mainImageFile) } },
            { label: '提炼结构总纲', prompt: '请把这张图整理成后续都要遵循的结构化文本总纲' },
            { label: '咨询这张图片', prompt: '详细描述这张图的内容' },
        ], handleSuggestionClick);
    }
    
    // Make it globally accessible for context-menu to trigger it
    window.renderMyAssetsPanel = renderAssetLibrary;

    async function renderAssetLibrary() {
        userAssets = await dbHelper.getAllAssets();
        
        // --- INTEGRATING MVR ASSET RUNTIME ---
        let independentAssets = [];
        try {
            const mvrAssetsDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'my_assets_library' });
            const keys = await mvrAssetsDB.keys();
            for (const key of keys) {
                const asset = await mvrAssetsDB.getItem(key);
                if (asset) {
                    independentAssets.push(asset);
                }
            }
        } catch (e) {
            console.error("Failed to load independent assets", e);
        }
        // -------------------------------------

        assetLibraryGrid.innerHTML = '';
        emptyLibraryState.style.display = (userAssets.length === 0 && independentAssets.length === 0) ? 'block' : 'none';

        // Render old assets
        userAssets.forEach(asset => {
            const rawAssetUrl = asset.dataUrl || asset.sourceImage;
            const assetSrc = getProxiedUrl(rawAssetUrl);
            if (!assetSrc) {
                console.warn('[AssetLibrary] Skipping saved asset with invalid dataUrl:', asset);
                return;
            }

            const item = document.createElement('div');
            item.className = 'asset-item';
            item.innerHTML = `
                <img src="${assetSrc}" class="generated-image" alt="Saved Asset">
                <button class="delete-asset-btn" data-id="${asset.id}">&times;</button>
            `;
            item.addEventListener('click', async (e) => {
                const deleteButton = e.target.closest('.delete-asset-btn');
                if (deleteButton) {
                    e.stopPropagation();
                    const idToDelete = parseInt(deleteButton.dataset.id, 10);
                    const confirmed = await showCustomConfirm('确定要从素材库删除这张图片吗？');
                    if (confirmed) {
                        await dbHelper.deleteAsset(idToDelete);
                        await renderAssetLibrary();
                        await updateImagePreview();
                    }
                } else {
                    const file = await dataURLToFile(rawAssetUrl, `asset-${asset.id}.png`);
                    referenceImageFiles.push(file);
                    pendingReferenceImageShares.add(file);
                    await updateImagePreview();
                    updateSendBtnState();
                }
            });
            assetLibraryGrid.appendChild(item);
        });

        // Render MVR independent assets
        independentAssets.forEach(asset => {
            const assetSrc = getProxiedUrl(asset.sourceImage);
            if (isInvalidImageSrc(asset.sourceImage) || !assetSrc) {
                console.warn('[AssetLibrary] Skipping independent asset with invalid sourceImage:', asset);
                return;
            }

            const item = document.createElement('div');
            item.className = 'asset-item mvr-asset';
            item.style.border = '2px solid #f59e0b';
            item.draggable = true;
            item.innerHTML = `
                <img src="${assetSrc}" class="generated-image" alt="${asset.name}" title="${asset.name}">
                <button class="delete-asset-btn" data-uid="${asset.uid}">&times;</button>
                <div style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.5); color: white; font-size: 10px; padding: 2px 4px; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${asset.name}</div>
            `;
            
            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', JSON.stringify({
                    type: 'independent_asset',
                    uid: asset.uid,
                    name: asset.name,
                    sourceImage: asset.sourceImage,
                    category: asset.type
                }));
                e.dataTransfer.effectAllowed = 'copy';
            });
            
            item.addEventListener('click', async (e) => {
                const deleteButton = e.target.closest('.delete-asset-btn');
                if (deleteButton) {
                    e.stopPropagation();
                    const uidToDelete = deleteButton.dataset.uid;
                    const confirmed = await showCustomConfirm('确定要删除这个独立资产吗？');
                    if (confirmed) {
                        const mvrAssetsDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'my_assets_library' });
                        await mvrAssetsDB.removeItem(uidToDelete);
                        await renderAssetLibrary();
                    }
                } else {
                    const file = await dataURLToFile(asset.sourceImage, `mvr-asset-${Date.now()}.png`);
                    referenceImageFiles.push(file);
                    pendingReferenceImageShares.add(file);
                    await updateImagePreview();
                    updateSendBtnState();
                    showToast('已添加到聊天参考图', 2000, true);
                }
            });
            assetLibraryGrid.appendChild(item);
        });
    }

    async function initializeApp() {
        // Initialize Core OS Tools
        initDecisionLog();
        
        // Initialize Latent Canvas
        initLatentCanvas({
            addImageToWorkbench: addImageToWorkbench,
            addMessage: addMessage
        });
        setupLatentCanvas();
        
        // Initialize Mask Drawer
        initMaskDrawer({
            maskEditorModal: document.getElementById('maskEditorModal'),
            maskCanvasContainer: document.getElementById('maskCanvasContainer'),
            closeMaskEditorBtn: document.getElementById('closeMaskEditor'),
            cancelMaskBtn: document.getElementById('cancelMask'),
            applyMaskBtn: document.getElementById('applyMask')
        }, {
            updateImagePreview: updateImagePreview,
            updateSendBtnState: updateSendBtnState
        });

        // Initialize Relight Engine
        initRelightEngine({
            addImageToWorkbench: addImageToWorkbench,
            addMessage: addMessage
        });

        // Initialize Genealogy Lines
        initGenealogyLines({
            addImageToWorkbench: addImageToWorkbench,
            addMessage: addMessage
        });

        const isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
        if (isTouchDevice) {
            const lassoHint = document.querySelector('.lasso-hint');
            if (lassoHint) lassoHint.textContent = '用手指沿物体边缘拖拽完成套索';
        }

        await dbHelper.init();
        window.suppressSessionAutoCreate = true;
        window.isInitializingAppRestore = true;
        try {
            // --- INTEGRATING MVR ASSET RUNTIME ---
            window.mvrRuntime = runtime; // Set it early
            window.memoryLayer = memoryLayer; // Expose memoryLayer
            runtime.setAgentAssetResolver(async ({ projectId, assetId }) => {
                const persistedSession = await dbHelper.getSessionData(projectId);
                return persistedSession?.runtimeWorkspace?.currentState?.assets?.find(asset => asset.uid === assetId) || null;
            });
            runtime.setAgentVerificationHooks({
                verify: async ({ workspace: targetWorkspace, assetIds }) => {
                    const checks = [];
                    const closeEnough = (actual, expected) => Number.isFinite(actual) && Math.abs(actual - expected) <= 1;
                    const waitForImage = (image) => {
                        if (!image || image.complete) return Promise.resolve();
                        return new Promise(resolve => {
                            let settled = false;
                            const finish = () => {
                                if (settled) return;
                                settled = true;
                                image.removeEventListener('load', finish);
                                image.removeEventListener('error', finish);
                                resolve();
                            };
                            image.addEventListener('load', finish, { once: true });
                            image.addEventListener('error', finish, { once: true });
                            setTimeout(finish, 1200);
                        });
                    };
                    for (const assetId of assetIds) {
                        const asset = targetWorkspace.currentState.assetRegistry.get(assetId);
                        const escapedAssetId = typeof CSS !== 'undefined' && typeof CSS.escape === 'function'
                            ? CSS.escape(assetId)
                            : String(assetId).replace(/(["\\])/g, '\\$1');
                        const element = document.querySelector(`[data-item-id="${escapedAssetId}"]`);
                        const image = element?.querySelector('img');
                        await waitForImage(image);
                        const hasElement = Boolean(element);
                        const hasImageReference = Boolean(image?.getAttribute('src') || image?.src);
                        checks.push({
                            id: `dom_asset_${assetId}`,
                            passed: hasElement,
                            message: hasElement ? `Workspace element ${assetId} is present.` : `Workspace element ${assetId} is missing.`
                        });
                        checks.push({
                            id: `image_reference_${assetId}`,
                            passed: hasImageReference,
                            message: hasImageReference ? `Image reference ${assetId} is bound.` : `Image reference ${assetId} is missing.`
                        });
                        const imageLoaded = Boolean(image && image.complete && (
                            image.naturalWidth > 0 || /^(data:|blob:)/i.test(image.currentSrc || image.src || '')
                        ));
                        checks.push({
                            id: `image_loaded_${assetId}`,
                            passed: imageLoaded,
                            message: imageLoaded ? `Image ${assetId} is loadable.` : `Image ${assetId} has not finished loading.`
                        });
                        if (!asset?.transform || !element) continue;
                        const style = getComputedStyle(element);
                        const transformMatches =
                            closeEnough(parseFloat(element.style.left || style.left), asset.transform.x) &&
                            closeEnough(parseFloat(element.style.top || style.top), asset.transform.y) &&
                            closeEnough(parseFloat(element.style.width || style.width), asset.transform.width) &&
                            closeEnough(parseFloat(element.style.height || style.height), asset.transform.height) &&
                            String(element.style.zIndex || style.zIndex || '') === String(asset.transform.zIndex);
                        checks.push({
                            id: `transform_${assetId}`,
                            passed: transformMatches,
                            message: transformMatches ? `Transform ${assetId} matches Runtime.` : `Transform ${assetId} differs from Runtime.`
                        });
                    }
                    return checks;
                },
                repair: isAgentRuntimeFeatureEnabled('verificationAutoRepair') ? async () => {
                    if (typeof window.hydrateWorkbench === 'function') await window.hydrateWorkbench();
                    if (typeof window.reconcileAllAssets === 'function') window.reconcileAllAssets();
                } : undefined
            });
            runtime.setAgentLayerExtractionHooks(isAgentRuntimeFeatureEnabled('magicLayersCommand') ? {
                execute: executeAgentLayerExtraction,
                undo: undoAgentLayerExtraction,
                commit: commitAgentLayerExtraction,
                release: releaseAgentLayerExtraction
            } : {});
            runtime.setAgentCapabilityHooks(isAgentRuntimeFeatureEnabled('capabilityCommands') || isAgentRuntimeFeatureEnabled('objectEditing') ? {
                execute: executeAgentCapability,
                undo: undoAgentCapability,
                commit: commitAgentCapabilities,
                release: releaseAgentCapabilities
            } : {});
            if (isAgentRuntimeFeatureEnabled('agentTelemetry')) {
                runtime.getAgentRuntime().subscribe(event => {
                    console.info('[AgentTelemetry]', event);
                });
            }
            runtime.setAgentCommitPersistence(async () => {
                const currentSession = state.sessions.find(session => session.id === state.currentSessionId);
                if (currentSession && dbHelper?.saveSession) await dbHelper.saveSession(currentSession);
            });
            const workspace = await runtime.restoreWorkspace();
            if (workspace) {
                console.log("[MVR] Workspace re-initialized on app start");
                window.__startupWorkspaceSnapshot = serializeWorkspaceSnapshot(workspace);
            } else {
                console.log("[MVR] No existing workspace, creating default");
                const initialWs = runtime.createWorkspace('default-workspace', 'Local Session');
                window.__startupWorkspaceSnapshot = null;
            }
            // -------------------------------------
            
            // 恢复资产库
            let assets = await dbHelper.getAllAssets();
            if (assets.length === 0) {
                console.log("本地没有资产，尝试从云端恢复...");
                const restoredAssets = await dbHelper.restoreAssetsFromCloud();
                if (restoredAssets && restoredAssets.length > 0) {
                    console.log("资产恢复成功");
                }
            }
            await renderAssetLibrary();
            
            sessions = await dbHelper.getAllSessions();

            // 每次启动都与云端会话对账，避免本地旧副本覆盖云端较新的项目名/聊天记录/工作台快照
            const restoredSessions = await dbHelper.restoreSessionsFromCloud({ pruneMissingLocal: true });
            if (restoredSessions && restoredSessions.length > 0) {
                sessions = await dbHelper.getAllSessions();
            }

            await cleanupPlaceholderSessions();
            sessions = await dbHelper.getAllSessions();
            // Backfill the lightweight Agent catalog one session at a time so
            // historical projects become searchable after an upgrade without
            // retaining all image payloads in memory or in the index.
            await dbHelper.rebuildAssetCatalog(sessions);
            state.sessions = sessions;

            renderHistoryList();
            
            if (sessions.length > 0) {
                const bestSession = await dbHelper.getBestSessionToOpen();
                const sessionIdToOpen = bestSession?.id || sessions[0].id;
                await loadSession(sessionIdToOpen);
            } else {
                startNewSession();
            }
            initAgentVisualSurface({
                runtime,
                onUserMessage: appendAgentTaskUserMessage,
                onFinished: persistAgentTaskResult,
                onFallback: async (text) => {
                    userInput.value = text;
                    await handleSend();
                }
            });
        } finally {
            window.isInitializingAppRestore = false;
            window.suppressSessionAutoCreate = false;
        }
        
        initSidebar();
        initAgentDock({ runtime });
        initCameraAngleModal(handleSend);
        setupSelectionBox(); 

        initWorkbenchZoom();
        initAtmospherePalette(workbenchGrid); // Initialize new Atmosphere features
        initInjectionEngine();
        
        // Initialize dynamic implicit predictive prompt overlay
        initPredictivePromptEngine(userInput);

        userInput.addEventListener('input', () => { 
            userInput.style.height = ''; // Reset to min-height to calculate correct scrollHeight
            const newHeight = Math.min(userInput.scrollHeight, 200);
            userInput.style.height = `${newHeight}px`; 
            updateSendBtnState(); 
        });
        
        userInput.addEventListener('keydown', (e) => { 
            if (e.key === 'Enter' && !e.shiftKey) { 
                e.preventDefault(); 
                handleSend(); 
            } 
        });
        
        if (sendBtn) sendBtn.addEventListener('click', handleSend);
        
        if (plusMenuBtn) {
            plusMenuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                plusMenu.classList.toggle('hidden');
            });
        }
        
        document.addEventListener('click', (e) => {
            if (plusMenu && !plusMenu.contains(e.target) && !plusMenuBtn.contains(e.target) && (!scenarioSubMenu || !scenarioSubMenu.contains(e.target))) {
                plusMenu.classList.add('hidden');
                if (scenarioSubMenu) scenarioSubMenu.style.display = 'none';
            }
        });

        if (menuUploadBtn) {
            menuUploadBtn.addEventListener('click', () => {
                plusMenu.classList.add('hidden');
                if (scenarioSubMenu) scenarioSubMenu.style.display = 'none';
                mainImageUpload.click();
            });
        }

        const workflowMenuContainer = document.getElementById('workflowMenuContainer');
        const scenarioSubMenu = document.getElementById('scenarioSubMenu');
        const exitWorkflowBtn = document.getElementById('exitWorkflowBtn');
        const scenarioOptionBtns = document.querySelectorAll('.scenario-option-btn');
        
        window.selectedScenarioType = 'interior'; // Default

        if (workflowMenuContainer && scenarioSubMenu) {
            workflowMenuContainer.addEventListener('mouseenter', () => {
                scenarioSubMenu.style.display = 'flex';
            });
            workflowMenuContainer.addEventListener('mouseleave', () => {
                scenarioSubMenu.style.display = 'none';
            });
            // Fallback for click
            const menuWorkflowBtn = document.getElementById('menuWorkflowBtn');
            if (menuWorkflowBtn) {
                menuWorkflowBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isHidden = scenarioSubMenu.style.display === 'none';
                    scenarioSubMenu.style.display = isHidden ? 'flex' : 'none';
                });
            }
        }

        if (scenarioOptionBtns) {
            scenarioOptionBtns.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    window.selectedScenarioType = btn.getAttribute('data-scenario');
                    const scenarioName = btn.innerText.trim();
                    
                    isWorkflowMode = true;
                    userInput.placeholder = `[${scenarioName}] 输入需求，AI将自动执行全案...`;
                    
                    plusMenu.classList.add('hidden');
                    scenarioSubMenu.style.display = 'none';
                    
                    if (workflowMenuContainer) workflowMenuContainer.style.display = 'none';
                    if (exitWorkflowBtn) exitWorkflowBtn.classList.remove('hidden');
                });
            });
        }

        if (exitWorkflowBtn) {
            exitWorkflowBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                isWorkflowMode = false;
                userInput.placeholder = "输入您的指令...（例：生成一张风景画，或上传图片进行编辑）";
                
                plusMenu.classList.add('hidden');
                exitWorkflowBtn.classList.add('hidden');
                if (workflowMenuContainer) workflowMenuContainer.style.display = 'block';
            });
        }
        
        if (mainImageUpload) {
            mainImageUpload.addEventListener('change', async (e) => {
                const incomingFiles = Array.from(e.target.files);
                if (incomingFiles.length === 0) return;
                
                // 确保会话存在
                await ensureSessionExists('新建项目');
                triggerAsyncSessionRename();

                incomingFiles.forEach((file, index) => {
                    addImageToWorkbench(file, '上传图片');
                    
                    if (!mainImageFile && index === 0) {
                        mainImageFile = file;
                        pendingBaseImageShare = true;
                        isContextPreviewHidden = false;
                        maskDataUrl = null;
                    } 
                    else {
                        referenceImageFiles.push(file);
                        pendingReferenceImageShares.add(file);
                    }
                });

                if (mainImageFile && incomingFiles[0].type.startsWith('image/')) {
                    const currentSession = sessions.find(s => s.id === currentSessionId);
                    if (!currentSession || !Array.isArray(currentSession.messages) || currentSession.messages.length <= 2) {
                        setTimeout(triggerAIActiveGuidance, 100);
                    }
                }

                e.target.value = '';
                await updateImagePreview();
                updateSendBtnState();
                
                // 触发异步重命名
                triggerAsyncSessionRename();
            });
        }

        // Sidebar logic moved to sidebar.js
        
        function updateNewSessionBtnText() {
        const btn = document.getElementById('newSessionBtn');
        if (!btn) return;
        if (window.innerWidth > 1024) {
            btn.innerHTML = '<i class="fas fa-plus"></i> 新建项目';
        } else {
            btn.innerHTML = '开启新对话';
        }
    }

    if (newSessionBtn) newSessionBtn.addEventListener('click', startNewSession);
    const newProjectMiniBtn = document.getElementById('newProjectMiniBtn');
    if (newProjectMiniBtn) newProjectMiniBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        startNewSession();
    });
    window.addEventListener('resize', updateNewSessionBtnText);
    updateNewSessionBtnText();
        
        const clearWorkbenchBtn = document.getElementById('clearWorkbench');
        if (clearWorkbenchBtn) {
            clearWorkbenchBtn.addEventListener('click', clearWorkbench);
        }

        if (closeLayerModalBtn) {
            closeLayerModalBtn.addEventListener('click', () => {
                if (typeof window.closeLayerManagerModal === 'function') {
                    window.closeLayerManagerModal();
                }
            });
        }

        const toggleCanvasStyleBtn = document.getElementById('toggleCanvasStyle');
        if (toggleCanvasStyleBtn) {
            toggleCanvasStyleBtn.addEventListener('click', () => {
                document.body.classList.toggle('dark-mode');
                const btnIcon = toggleCanvasStyleBtn.querySelector('i');
                if (document.body.classList.contains('dark-mode')) {
                    btnIcon.className = 'fas fa-braille';
                } else {
                    btnIcon.className = 'fas fa-border-all';
                }
            });
        }

        // --- 工作台工具箱按钮事件 (Robust Binding) ---
        setupToolboxEvents({
            workbenchToolbox,
            workbenchGrid,
            workbenchZoomContainer,
            pushSelectedToChat
        });

        // --- 核心修改：基于语义图层的物理拆解 ---
        // (Migrated to src/ui/layer-manager.js)


        document.body.addEventListener('click', (e) => {
            if (e.target.classList.contains('image-message')) {
                let modal = document.getElementById('imagePreviewModal');
                if (!modal) {
                    modal = document.createElement('div'); modal.id = 'imagePreviewModal';
                    modal.style.cssText = 'display:none; position:fixed; z-index:99999; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.85); align-items:center; justify-content:center; cursor: zoom-out;';
                    modal.innerHTML = `<img id="imagePreviewModalImg" src="" class="generated-image" style="max-width:90vw;max-height:90vh;border-radius:12px;box-shadow:0 4px 32px #0008; cursor: default;"><button id="imagePreviewModalClose" style="position:absolute;top:32px;right:48px;font-size:2.2rem;color:#fff;background:none;border:none;cursor:pointer;z-index:2;">&times;</button>`;
                    document.body.appendChild(modal);
                }
                const modalImg = document.getElementById('imagePreviewModalImg'), closeBtn = document.getElementById('imagePreviewModalClose');
                if (closeBtn && !closeBtn.dataset.bound) {
                    closeBtn.dataset.bound = 'true';
                    closeBtn.addEventListener('click', () => { document.getElementById('imagePreviewModal').style.display = 'none'; });
                }
                modalImg.src = e.target.src; modal.style.display = 'flex'; return;
            }
            if (e.target.id === 'imagePreviewModal') { e.target.style.display = 'none'; return; }
            const historyMenu = document.querySelector('.history-menu');
            if (historyMenu && !historyMenu.contains(e.target) && !e.target.closest('.history-more-btn')) historyMenu.remove();
            if (magicWandModal?.style.display === 'block' && !magicWandModal.contains(e.target) && !e.target.closest('#mainImagePreview')) closeMagicWandModal();
            
            // 点击外部隐藏工具箱
            const floatingEditor = document.getElementById('floatingFusionEditor');
            const isClickingCapsule = e.target.closest('#aiInspirationCapsule') || e.target.closest('#capsuleSpeechTooltip');
            if (workbenchToolbox && workbenchToolbox.style.display === 'flex' && !workbenchToolbox.contains(e.target) && !e.target.closest('.workbench-item') && !(floatingEditor && floatingEditor.contains(e.target)) && !isClickingCapsule) {
                hideWorkbenchToolbox();
            }

            // 点击外部隐藏悬浮编辑框（双重保险）- 改变为转为虚隐状态
            if (floatingEditor && !floatingEditor.contains(e.target) && !e.target.closest('.workbench-item') && e.target.closest('#workbenchGrid') && !isClickingCapsule) {
                floatingEditor.classList.add('idle');
            }
            
        });
        
        if (historyList) {
            historyList.addEventListener('click', (e) => {
                const li = e.target.closest('li'); if (!li) return;
                const sessionId = li.dataset.sessionId;
                if (e.target.classList.contains('history-more-btn')) {
                    e.stopPropagation();
                    document.querySelector('.history-menu')?.remove();
                    const menu = document.createElement('div'); menu.className = 'history-menu';
                    menu.innerHTML = `
                        <div class="history-menu-item rename" data-session-id="${sessionId}"><i class="fas fa-pencil-alt"></i> 重命名</div>
                        <div class="history-menu-item share" data-session-id="${sessionId}"><i class="fas fa-share-alt"></i> 分享</div>
                        <div class="history-menu-item delete" data-session-id="${sessionId}"><i class="fas fa-trash-alt"></i> 删除</div>
                    `;
                    menu.addEventListener('click', (menuEvent) => {
                        menuEvent.stopPropagation();
                        const actionTarget = menuEvent.target.closest('.history-menu-item');
                        if (!actionTarget) return;

                        const id = actionTarget.dataset.sessionId;

                        if (actionTarget.classList.contains('delete')) {
                            const deleteModal = document.getElementById('deleteConfirmModal');
                            const confirmBtn = document.getElementById('confirmDeleteBtn');
                            if (deleteModal && confirmBtn) {
                                confirmBtn.dataset.sessionId = id;
                                deleteModal.style.display = 'flex';
                            }
                        } else if (actionTarget.classList.contains('rename')) {
                            console.log('Rename session:', id);
                            alert('重命名功能正在开发中！');
                        } else if (actionTarget.classList.contains('share')) {
                            console.log('Share session:', id);
                            alert('分享功能正在开发中！');
                        }
                        menu.remove();
                    });
                    document.body.appendChild(menu);
                    const rect = e.target.getBoundingClientRect();
                    menu.style.cssText = `position: absolute; display: block; top: ${rect.bottom + window.scrollY}px; left: ${rect.right + window.scrollX - menu.offsetWidth}px;`;
                } else {
                    loadSession(sessionId);
                }
            });
        }

        if (chatMessages) {
            chatMessages.addEventListener('click', async (e) => {
                const selectBtn = e.target.closest('.select-image-btn');
                if (!selectBtn || isSending) return;

                const imgSrc = selectBtn.dataset.src;
                const mimeType = selectBtn.dataset.mime;
                if (!imgSrc) return;

                try {
                    isSending = true;
                    selectBtn.textContent = '已选用';
                    selectBtn.style.background = '#4CAF50';

                    // Register implicit satisfaction of the prompting chain with full image source context and state
                    try {
                        await implicitMemoryEngine.registerImplicitSatisfaction(null, imgSrc, state);
                    } catch (e) {
                        console.error("[ImplicitMemory] Error registering satisfaction:", e);
                    }
                    try {
                        const source = interactionAttributionRuntime.resolveSource({
                            fallbackSourceType: 'manual'
                        });
                        resultFeedbackRuntime.recordEvent({
                            type: 'result_selected',
                            sessionId: currentSessionId || undefined,
                            projectId: runtime.getCurrentWorkspace()?.projectId || undefined,
                            sourceType: source.sourceType,
                            sourceId: source.sourceId,
                            context: {
                                workflowStage: 'chat_selection'
                            },
                            metadata: {
                                imageSrc: imgSrc,
                                attributionMetadata: source.metadata || null
                            }
                        });
                    } catch (e) {
                        console.error('[ResultFeedback] Failed to record result_selected:', e);
                    }

                    const selectedFile = await dataURLToFile(imgSrc, `selected-${Date.now()}.png`);
                    
                    lastGeneratedImageForEditing = selectedFile;
                    mainImageFile = selectedFile;
                    
                    isContextPreviewHidden = false;
                    maskDataUrl = null;
                    referenceImageFiles = [];
                    pendingBaseImageShare = false;
                    pendingReferenceImageShares.clear();
                    
                    await updateImagePreview();
                    mainImagePreview.style.display = 'flex';
                    
                    const currentSession = sessions.find(s => s.id === currentSessionId);
                    const userMessage = { sender: 'user', type: 'text', content: `[ 已选用图片 ]` };
                    addMessage(userMessage);
                    if (currentSession) {
                        currentSession.messages.push(userMessage);
                        try {
                            const continuityAnchor = await buildSeriesContinuityAnchor(
                                selectedFile,
                                currentSession.messages,
                                {
                                    promptHint: '这张图已被用户明确选用，后续连续性项目应继承其中可复用的系列视觉系统，而不是机械复制当前场景。',
                                    anchorSource: 'selected_image'
                                }
                            );
                            if (continuityAnchor) {
                                setContinuityAnchor(currentSession, continuityAnchor, { replace: true });
                            }
                        } catch (anchorError) {
                            console.warn('Failed to build continuity anchor from selected image:', anchorError);
                        }
                        await dbHelper.saveSession(currentSession);
                    }
                    
                } catch (error) {
                    console.error("Error selecting image:", error);
                    alert("选择图片时出错，请重试。");
                } finally {
                    isSending = false;
                    updateSendBtnState();
                }
            });
        }

        if (magicWandModal) {
            magicWandModal.addEventListener('click', (e) => { 
                if (e.target.dataset.action) {
                    handleMagicWandAction(e.target.dataset.action, {
                        mainImageFile,
                        lastGeneratedImageForEditing,
                        userInput,
                        updateSendBtnState
                    });
                }
            });
        }

        // Mobile menu logic moved to sidebar.js

        const creditCounter = document.getElementById('creditCounter'), creditModal = document.getElementById('creditModal'), closeCreditModalBtn = document.getElementById('closeCreditModal');
        if (creditCounter && creditModal) creditCounter.addEventListener('click', () => { creditModal.style.display = 'flex'; });
        if (closeCreditModalBtn) closeCreditModalBtn.addEventListener('click', () => { creditModal.style.display = 'none'; });
        if (creditModal) creditModal.addEventListener('click', (e) => { if (e.target === creditModal) creditModal.style.display = 'none'; });

        const deleteModal = document.getElementById('deleteConfirmModal');
        const closeDeleteModalBtn = document.getElementById('closeDeleteModal');
        const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
        const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');

        const hideDeleteModal = () => {
            if (deleteModal) deleteModal.style.display = 'none';
        };

        if (closeDeleteModalBtn) closeDeleteModalBtn.addEventListener('click', hideDeleteModal);
        if (cancelDeleteBtn) cancelDeleteBtn.addEventListener('click', hideDeleteModal);
        if (deleteModal) {
            deleteModal.addEventListener('click', (e) => {
                if (e.target === deleteModal) {
                    hideDeleteModal();
                }
            });
        }

        if (confirmDeleteBtn) {
            confirmDeleteBtn.addEventListener('click', async () => {
                const idToDelete = confirmDeleteBtn.dataset.sessionId;
                if (idToDelete) {
                    await dbHelper.deleteSession(idToDelete);
                    sessions = sessions.filter(s => s.id !== idToDelete);
                    clearChatSession(idToDelete);
                    if (currentSessionId === idToDelete) await startNewSession();
                    renderHistoryList();
                }
                hideDeleteModal();
            });
        }
    }

    // --- 世界构建核心逻辑 (World Builder Logic) ---

    // --- Atmosphere / Environment Logic ---
    // (Migrated to src/ui/atmosphere.js)

    /**
     * 执行精准修图
     * (Migrated to src/ui/layer-manager.js)
     */


    /**
     * 新增：执行快捷融合逻辑
     * (Migrated to src/ui/layer-manager.js)
     */


    // === 将缩放函数导出到全局作用域 ===
    window.zoomIn = zoomIn;
    window.zoomOut = zoomOut;
    window.resetZoom = resetZoom;
    
    // --- Integration: Add Button to Toolbox ---
    document.addEventListener('DOMContentLoaded', () => {
        initChatSourceRail();
        initializeApp();
        
        // Initialize MarmoLens
        window.marmoLens = new MarmoLens(ai);
        
        // Add "Lens" button to toolbox HTML dynamically if not present
        const toolbox = document.getElementById('workbenchToolbox');
        if (toolbox && !toolbox.querySelector('.lens-btn')) {
            const lensBtn = document.createElement('button');
            lensBtn.className = 'toolbox-btn lens-btn';
            lensBtn.title = "智能识别与搜同款";
            lensBtn.innerHTML = '<i class="fas fa-search"></i>'; // Lens icon replacement
            lensBtn.style.color = '#8b5cf6'; // Lens purple
            
            lensBtn.onclick = () => {
                if (currentActiveWorkbenchItemId) {
                    // Hide toolbox
                    toolbox.style.display = 'none';
                    // Trigger Lens Manual Selection
                    window.marmoLens.startSelectionMode(currentActiveWorkbenchItemId);
                }
            };
            
            // Insert before Delete button
            const deleteBtn = toolbox.querySelector('.delete');
            toolbox.insertBefore(lensBtn, deleteBtn);
        }
    });
