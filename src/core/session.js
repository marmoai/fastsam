import { state } from './state.js';
import { addMessage, renderMessages } from '../ui/chat-panel.js';
import { dataURLToFile, fileToDataURL } from './utils.js';
import { addTextNoteToWorkbench, restoreGroupLabelToWorkbench, addAtmosphereNode, addImageToWorkbench } from '../ui/workbench-core.js';
import { sidebarState } from '../ui/sidebar.js';
import { closeMagicWandModal } from '../ui/modals.js';
import { saveSessionsToOSS, getSessionsFromOSS, uploadImageToOSS } from '../services/ossService.js';
import { updateHeader } from '../ui/header.js';
import { runtime } from '../runtime/CoreRuntime';

import localforage from 'localforage';

const sessionsMetaDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'sessionsMeta' });
const sessionDataDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'sessionData' });
const assetsDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'assets' });
const imageCacheDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'imageCache' });

const dbWorker = new Worker(new URL('./db-worker.js', import.meta.url), { type: 'module' });
const pendingJobs = new Map();
let jobIdCounter = 0;

function getSessionRevision(session) {
    if (!session || typeof session !== 'object') return 0;
    return Math.max(
        Number(session.updatedAt) || 0,
        Number(session.timestamp) || 0
    );
}

function getSessionContentSummary(session) {
    if (!session || typeof session !== 'object') {
        return {
            messageCount: 0,
            messageImageCount: 0,
            runtimeAssetCount: 0,
            legacyImageCount: 0,
            nonRuntimeItemCount: 0,
            hasContent: false,
            score: 0
        };
    }

    const messages = Array.isArray(session.messages) ? session.messages : [];
    const runtimeAssets = Array.isArray(session.runtimeWorkspace?.currentState?.assets)
        ? session.runtimeWorkspace.currentState.assets
        : [];
    const renderableRuntimeAssetCount = runtimeAssets.filter(asset =>
        asset?.sourceImage || asset?.sourceImageBlob || asset?.originalDataUrl || asset?.originalBlob
    ).length;
    const workbenchState = Array.isArray(session.workbenchState) ? session.workbenchState : [];
    const legacyImageCount = workbenchState.filter(item => {
        if (!isRuntimeManagedWorkbenchItem(item)) return false;
        return !!(item?.dataUrl || item?.blob || item?.sourceImage || item?.sourceImageBlob);
    }).length;
    const nonRuntimeItemCount = workbenchState.filter(item => !isRuntimeManagedWorkbenchItem(item)).length;
    const messageImageCount = messages.reduce((count, msg) => {
        if (!msg?.imageData) return count;
        const images = Array.isArray(msg.imageData) ? msg.imageData : [msg.imageData];
        return count + images.filter(img => img?.src || img?.blob).length;
    }, 0);
    const messageCount = messages.length;
    const runtimeAssetCount = runtimeAssets.length;

    const score =
        (renderableRuntimeAssetCount * 1000) +
        (legacyImageCount * 800) +
        (messageCount * 50) +
        (messageImageCount * 25) +
        (nonRuntimeItemCount * 10);

    return {
        messageCount,
        messageImageCount,
        runtimeAssetCount,
        renderableRuntimeAssetCount,
        legacyImageCount,
        nonRuntimeItemCount,
        hasContent: score > 0,
        score
    };
}

dbWorker.onmessage = (e) => {
    const { id, result, error } = e.data;
    if (pendingJobs.has(id)) {
        const { resolve, reject } = pendingJobs.get(id);
        pendingJobs.delete(id);
        if (error) reject(new Error(error));
        else resolve(result);
    }
};

function runWorkerJob(type, payload) {
    return new Promise((resolve, reject) => {
        const id = jobIdCounter++;
        pendingJobs.set(id, { resolve, reject });
        dbWorker.postMessage({ type, payload, id });
    });
}

function isRuntimeManagedWorkbenchItem(item) {
    if (!item) return false;
    const type = item.type || 'image';
    return !['text-note', 'group-label', 'shape', 'atmosphere'].includes(type);
}

function serializeNonRuntimeWorkbenchItems() {
    const savedState = [];

    state.workbenchItems.forEach((item, id) => {
        if (!item?.el || isRuntimeManagedWorkbenchItem(item)) return;

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
                let innerContent = contentEl.innerHTML;
                const tempDiv = document.createElement('div');
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
            id,
            type: item.type,
            shapeType: item.shapeType,
            rect,
            content,
            fontSize,
            fontColor,
            fontFamily,
            fontWeight,
            fontStyle,
            lineHeight,
            letterSpacing,
            textAlign,
            textShadow,
            WebkitTextStroke,
            css: customCss,
            fillColor,
            borderColor,
            borderWidth,
            borderRadius,
            clipPath,
            genealogy: item.genealogy,
            parentId: item.parentId,
            layerName: item.layerName,
            originalBbox: item.originalBbox,
            key: item.key
        });
    });

    return savedState;
}

function serializeCurrentRuntimeWorkspace() {
    const workspace = runtime.getCurrentWorkspace();
    if (!workspace) return null;

    return {
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

function hydrateRuntimeWorkspaceFromSession(session) {
    const snapshot = session?.runtimeWorkspace;
    if (!snapshot?.currentState) return false;

    const projectId = session.id || snapshot.projectId || 'default-workspace';
    const name = session.title || snapshot.name || 'Local Session';
    const workspace = runtime.createWorkspace(projectId, name);

    workspace.currentState.stateId = snapshot.currentState.stateId || `state_${Date.now()}`;
    workspace.currentState.canvasState = snapshot.currentState.canvasState || workspace.currentState.canvasState;

    if (Array.isArray(snapshot.currentState.assets)) {
        for (const asset of snapshot.currentState.assets) {
            workspace.currentState.assetRegistry.register(asset);
            workspace.currentState.sceneGraph.addNode(asset.uid);
        }
    }

    const nodes = snapshot.currentState.nodes || snapshot.currentState.sceneNodes || [];
    if (Array.isArray(nodes)) {
        for (const node of nodes) {
            workspace.currentState.sceneGraph.addNode(node);
        }
    }

    const edges = snapshot.currentState.edges || snapshot.currentState.sceneEdges || [];
    if (Array.isArray(edges)) {
        for (const edge of edges) {
            const sourceExists = workspace.currentState.assetRegistry.get(edge.sourceId) || nodes.includes(edge.sourceId);
            const targetExists = workspace.currentState.assetRegistry.get(edge.targetId) || nodes.includes(edge.targetId);
            if (sourceExists && targetExists) {
                workspace.currentState.sceneGraph.addEdge(edge);
            }
        }
    }

    if (Array.isArray(snapshot.decisionGraph)) {
        for (const log of snapshot.decisionGraph) {
            workspace.decisionGraph.addLog(log);
        }
    }

    workspace.history = [];
    workspace.historyIndex = -1;
    workspace.snapshot();
    workspace.currentState.notify();

    if (snapshot.viewport) {
        if (typeof snapshot.viewport.zoom === 'number') {
            state.workbenchZoom = snapshot.viewport.zoom;
        }
        if (typeof snapshot.viewport.panOffsetX === 'number') {
            state.panOffsetX = snapshot.viewport.panOffsetX;
        }
        if (typeof snapshot.viewport.panOffsetY === 'number') {
            state.panOffsetY = snapshot.viewport.panOffsetY;
        }
        if (typeof window.applyPanTransform === 'function') {
            window.applyPanTransform();
        }
        if (typeof window.updateZoomIndicator === 'function') {
            window.updateZoomIndicator();
        }
        if (typeof window.updateWorkbenchItemsScale === 'function') {
            window.updateWorkbenchItemsScale();
        }
        window.__hasRestoredViewport = true;
    }

    return true;
}

function extractLegacyImageWorkbenchItems(session) {
    if (!Array.isArray(session?.workbenchState)) return [];
    return session.workbenchState.filter(item => item?.dataUrl && isRuntimeManagedWorkbenchItem(item));
}

function getRenderableWorkbenchImageCount() {
    let count = 0;
    state.workbenchItems.forEach((item) => {
        if (item?.el && isRuntimeManagedWorkbenchItem(item)) {
            count += 1;
        }
    });
    return count;
}

function cloneWorkspaceSnapshot(snapshot) {
    if (!snapshot) return null;
    return JSON.parse(JSON.stringify(snapshot));
}

function cloneValue(value) {
    if (value == null) return value;
    return JSON.parse(JSON.stringify(value));
}

function cloneCloudPayloadValue(value) {
    if (value == null) return value;
    if (typeof structuredClone === 'function') {
        try {
            return structuredClone(value);
        } catch (error) {
            console.warn('structuredClone failed for cloud payload, falling back to JSON clone:', error);
        }
    }
    return cloneValue(value);
}

const MEDIA_FIELD_NAMES = new Set([
    'src',
    'dataUrl',
    'sourceImage',
    'originalDataUrl',
    'cleanPlateDataUrl',
    'image',
    'mask',
    'maskUrl',
    'cutoutUrl',
    'previewUrl'
]);

const BLOB_TARGET_FIELD = {
    blob: 'dataUrl',
    sourceImageBlob: 'sourceImage',
    originalBlob: 'originalDataUrl',
    cleanPlateBlob: 'cleanPlateDataUrl',
    maskBlob: 'mask',
    imageBlob: 'image',
    cutoutBlob: 'cutoutUrl',
    previewBlob: 'previewUrl'
};

function isBlobLike(value) {
    return typeof Blob !== 'undefined' && value instanceof Blob;
}

function isLargeEmbeddedMediaString(key, value) {
    if (typeof value !== 'string') return false;
    if (value.startsWith('data:')) return true;
    if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('blob:')) return false;
    return MEDIA_FIELD_NAMES.has(key) && value.length > 1000;
}

function collectEmbeddedMediaUploadTasks(value, addTask, visited = new WeakSet()) {
    if (!value || typeof value !== 'object') return;
    if (visited.has(value)) return;
    visited.add(value);

    if (Array.isArray(value)) {
        value.forEach(item => collectEmbeddedMediaUploadTasks(item, addTask, visited));
        return;
    }

    Object.entries(value).forEach(([key, child]) => {
        if (BLOB_TARGET_FIELD[key] && isBlobLike(child)) {
            addTask(value, key, BLOB_TARGET_FIELD[key]);
            return;
        }

        if (isLargeEmbeddedMediaString(key, child)) {
            const isRawBase64 = typeof child === 'string' && !child.startsWith('data:');
            addTask(value, key, key, isRawBase64, 'data:image/png;base64,');
            return;
        }

        if (child && typeof child === 'object') {
            collectEmbeddedMediaUploadTasks(child, addTask, visited);
        }
    });
}

function pruneLayerForCloud(layer) {
    if (!layer || typeof layer !== 'object') return;

    delete layer.blob;
    delete layer.imageBlob;
    delete layer.maskBlob;
    delete layer.cutoutBlob;
    delete layer.previewBlob;
    delete layer.sourceImageBlob;
    delete layer.originalBlob;
    delete layer.cleanPlateBlob;

    const removeIfEmbeddedMedia = (obj, key) => {
        const value = obj?.[key];
        if (typeof value !== 'string') {
            delete obj[key];
            return;
        }
        if (!value.startsWith('http://') && !value.startsWith('https://')) {
            delete obj[key];
        }
    };

    removeIfEmbeddedMedia(layer, 'image');
    removeIfEmbeddedMedia(layer, 'mask');
    removeIfEmbeddedMedia(layer, 'maskUrl');
    removeIfEmbeddedMedia(layer, 'cutoutUrl');
    removeIfEmbeddedMedia(layer, 'previewUrl');

    if (Array.isArray(layer.versions)) {
        layer.versions = layer.versions.slice(-3).map(version => {
            const slimVersion = { ...version };
            delete slimVersion.blob;
            delete slimVersion.imageBlob;
            delete slimVersion.maskBlob;
            delete slimVersion.cutoutBlob;
            delete slimVersion.previewBlob;
            delete slimVersion.sourceImageBlob;
            delete slimVersion.originalBlob;
            delete slimVersion.cleanPlateBlob;
            removeIfEmbeddedMedia(slimVersion, 'image');
            removeIfEmbeddedMedia(slimVersion, 'mask');
            removeIfEmbeddedMedia(slimVersion, 'maskUrl');
            removeIfEmbeddedMedia(slimVersion, 'cutoutUrl');
            removeIfEmbeddedMedia(slimVersion, 'previewUrl');
            return slimVersion;
        });
    }
}

function pruneSemanticViewsForCloud(semanticViews, rank) {
    if (!semanticViews || typeof semanticViews !== 'object') return;
    delete semanticViews.layerGraph;

    if (Array.isArray(semanticViews.editableSceneLayers)) {
        semanticViews.editableSceneLayers.forEach(pruneLayerForCloud);
    }

    if (Array.isArray(semanticViews.cleanPlateLayers)) {
        semanticViews.cleanPlateLayers.forEach(pruneLayerForCloud);
    }

    if (rank > 0) {
        delete semanticViews.cleanPlateLayers;
    }
}

function pruneDecisionLogForCloud(decisionLog, rank) {
    if (!Array.isArray(decisionLog)) return undefined;
    if (rank > 1) return undefined;

    return decisionLog.slice(-30).map(log => ({
        timestamp: log?.timestamp || Date.now(),
        actionType: log?.actionType || log?.action || log?.type || 'unknown',
        payload: log?.payload || {}
    }));
}

function removeEmbeddedMediaFallback(value, visited = new WeakSet()) {
    if (!value || typeof value !== 'object') return;
    if (visited.has(value)) return;
    visited.add(value);

    if (Array.isArray(value)) {
        value.forEach(item => removeEmbeddedMediaFallback(item, visited));
        return;
    }

    Object.entries(value).forEach(([key, child]) => {
        if (BLOB_TARGET_FIELD[key]) {
            delete value[key];
            return;
        }
        if (isLargeEmbeddedMediaString(key, child)) {
            console.warn(`[CloudSync] Dropping oversized embedded media field "${key}" from cloud payload because upload replacement was not available.`);
            delete value[key];
            return;
        }
        if (child && typeof child === 'object') {
            removeEmbeddedMediaFallback(child, visited);
        }
    });
}

function prepareSessionForCloudPayload(session, rank = 0) {
    const cloudSession = cloneCloudPayloadValue(session);

    const cloudDecisionLog = pruneDecisionLogForCloud(cloudSession.decisionLog, rank);
    if (cloudDecisionLog) {
        cloudSession.decisionLog = cloudDecisionLog;
    } else {
        delete cloudSession.decisionLog;
    }

    if (Array.isArray(cloudSession.workbenchState)) {
        cloudSession.workbenchState.forEach(item => {
            pruneSemanticViewsForCloud(item.semanticViews, rank);
            (item.layers || []).forEach(pruneLayerForCloud);
            (item.scene?.layers || []).forEach(pruneLayerForCloud);
        });
    }

    const runtimeWorkspace = cloudSession.runtimeWorkspace;
    if (runtimeWorkspace) {
        if (Array.isArray(runtimeWorkspace.decisionGraph)) {
            runtimeWorkspace.decisionGraph = rank === 0 ? runtimeWorkspace.decisionGraph.slice(-20) : [];
        }

        const assets = runtimeWorkspace.currentState?.assets;
        if (Array.isArray(assets)) {
            assets.forEach(asset => {
                pruneSemanticViewsForCloud(asset.semanticViews, rank);
                (asset.layers || []).forEach(pruneLayerForCloud);
                (asset.scene?.layers || []).forEach(pruneLayerForCloud);
            });
        }
    }

    return cloudSession;
}

function consumeEmptyWorkbenchPersistAllowance(sessionId) {
    const allowance = window.__allowEmptyWorkbenchSessionSave;
    if (!allowance || allowance.sessionId !== sessionId) return false;
    const notExpired = !allowance.expiresAt || allowance.expiresAt >= Date.now();
    delete window.__allowEmptyWorkbenchSessionSave;
    return notExpired;
}

function shouldUseStartupWorkspaceFallback(session, legacyImageItems) {
    if (!session) return false;
    if (window.__startupWorkspaceConsumedSessionId) return false;
    const startupSnapshot = window.__startupWorkspaceSnapshot;
    const startupAssets = startupSnapshot?.currentState?.assets;
    if (!Array.isArray(startupAssets) || startupAssets.length === 0) return false;
    if (legacyImageItems.length > 0) return false;
    if (Array.isArray(session.messages) && session.messages.some(msg => msg?.imageData)) return false;

    const ordered = [...state.sessions].sort((a, b) => getSessionRevision(b) - getSessionRevision(a));
    return ordered[0]?.id === session.id;
}

export const dbHelper = {
    db: null,
    async init() {
        return Promise.resolve();
    },
    
    async saveSession(session) {
        try {
            const now = Date.now();
            if (!session.timestamp) {
                session.timestamp = now;
            }
            const existingRawData = await sessionDataDB.getItem(session.id);
            let existingSession = null;
            if (existingRawData) {
                try {
                    existingSession = await runWorkerJob('deserializeSession', existingRawData);
                } catch (error) {
                    console.warn(`Failed to inspect existing session ${session.id} before save:`, error);
                }
            }

            const nextSession = {
                ...session,
                updatedAt: now,
                workbenchState: serializeNonRuntimeWorkbenchItems(),
                runtimeWorkspace: serializeCurrentRuntimeWorkspace()
            };

            // Attach purified decision logs to the session before saving
            if (window.memoryLayer) {
                const logs = window.memoryLayer.getLogs();
                nextSession.decisionLog = Array.isArray(logs) ? logs.slice(-30) : [];
            }

            const existingContent = getSessionContentSummary(existingSession);
            const nextContent = getSessionContentSummary(nextSession);
            const isProtectedRestorePhase = !!(window.isRestoringSession || window.isInitializingAppRestore);
            const shouldProtectExistingContent =
                existingContent.hasContent &&
                !nextContent.hasContent &&
                isProtectedRestorePhase;

            if (shouldProtectExistingContent && existingSession) {
                console.warn(`Prevented empty overwrite for session ${session.id} during restore/startup phase.`);
                Object.assign(session, existingSession, {
                    id: session.id,
                    title: session.title || existingSession.title,
                    timestamp: session.timestamp || existingSession.timestamp,
                    updatedAt: Math.max(now, existingSession.updatedAt || 0),
                    isAutoRenamed: session.isAutoRenamed || existingSession.isAutoRenamed || false
                });
                return;
            }

            const allowEmptyWorkbenchPersist = consumeEmptyWorkbenchPersistAllowance(session.id);

            if (
                existingSession &&
                existingContent.messageCount > 0 &&
                nextContent.messageCount === 0
            ) {
                nextSession.messages = cloneValue(existingSession.messages) || [];
            }

            const shouldPreserveWorkbenchContent =
                existingSession &&
                !allowEmptyWorkbenchPersist &&
                (existingContent.runtimeAssetCount > 0 || existingContent.legacyImageCount > 0) &&
                nextContent.runtimeAssetCount === 0 &&
                nextContent.legacyImageCount === 0;

            if (shouldPreserveWorkbenchContent) {
                if (existingSession.runtimeWorkspace) {
                    nextSession.runtimeWorkspace = cloneWorkspaceSnapshot(existingSession.runtimeWorkspace);
                }
                if (Array.isArray(existingSession.workbenchState)) {
                    nextSession.workbenchState = cloneValue(existingSession.workbenchState) || [];
                }
            }

            Object.assign(session, nextSession);

            // 1. Save metadata (lightweight)
            const meta = {
                id: session.id,
                title: session.title,
                timestamp: session.timestamp,
                updatedAt: session.updatedAt,
                isAutoRenamed: session.isAutoRenamed || false
            };
            await sessionsMetaDB.setItem(session.id, meta);

            // 2. Process heavy data in worker
            const processedSession = await runWorkerJob('serializeSession', session);

            // 3. Save heavy data
            await sessionDataDB.setItem(session.id, processedSession);

            // 4. 同步到云端
            this.syncSessionsToCloud();
        } catch (e) {
            console.error('Failed to save session to DB:', e);
        }
    },
    
    async getAllSessions() {
        try {
            // Only load metadata!
            const sessions = [];
            await sessionsMetaDB.iterate((value, key) => {
                sessions.push(value);
            });
            // Sort by timestamp descending
            return sessions.sort((a, b) => getSessionRevision(b) - getSessionRevision(a));
        } catch (e) {
            console.error('Failed to get sessions meta:', e);
            return [];
        }
    },
    async getSessionData(sessionId) {
        try {
            const data = await sessionDataDB.getItem(sessionId);
            if (!data) return null;
            // Deserialize in worker
            const session = await runWorkerJob('deserializeSession', data);
            return session;
        } catch (e) {
            console.error('Failed to get session data:', e);
            return null;
        }
    },
    async deleteSession(sessionId) {
        try {
            await sessionsMetaDB.removeItem(sessionId);
            await sessionDataDB.removeItem(sessionId);
            // 删除需要尽快同步到云端，避免其他设备继续恢复旧项目
            await this.syncSessionsToCloud({ immediate: true });
        } catch (e) {
            console.error('Failed to delete session:', e);
        }
    },
    // --- NEW: Cloud Sync Functions ---
    _syncTimer: null,
    _syncInFlight: null,
    async _performSessionsCloudSync() {
        if (this._syncInFlight) {
            return this._syncInFlight;
        }

        this._syncInFlight = (async () => {
            try {
                const metaList = await this.getAllSessions();
                
                const b64toBlob = async (b64Data) => {
                    const res = await fetch(b64Data);
                    return await res.blob();
                };

                // Form full sessions list ensures all images are uploaded to OSS
                const fullSessions = [];
                for (const meta of metaList) {
                    const data = await sessionDataDB.getItem(meta.id);
                    if (!data) {
                        console.warn(`Skipping cloud sync for session ${meta.id} because sessionData is missing.`);
                        continue;
                    }
                    
                    let changed = false;

                    // Prune large global decisionLog if it got saved into the session
                    if (data.decisionLog && Array.isArray(data.decisionLog) && data.decisionLog.length > 30) {
                        data.decisionLog = data.decisionLog.slice(-30);
                        changed = true;
                    }

                    const cloudData = prepareSessionForCloudPayload(data, fullSessions.length);

                    const uploadTasks = [];
                    const addTask = (obj, sourceProp, targetProp, isBase64 = false, fallbackPrefix = '', sessionId = meta.id) => {
                        let sourceData = obj[sourceProp];
                        if (!sourceData) return;
                        uploadTasks.push(async () => {
                            try {
                                if (isBase64 && typeof sourceData === 'string' && !sourceData.startsWith('data:') && !sourceData.startsWith('http') && sourceData.length > 1000) {
                                    sourceData = fallbackPrefix + sourceData;
                                }
                                const url = await uploadImageToOSS(sourceData, { sessionId });
                                obj[targetProp] = url;
                                if (sourceProp !== targetProp) {
                                    delete obj[sourceProp];
                                }
                                changed = true;
                            } catch (err) {
                                console.error(`Failed to upload layer/mask concurrently:`, err);
                            }
                        });
                    };

                    collectEmbeddedMediaUploadTasks(cloudData, addTask);
                    
                    // 处理消息中的图片
                    if (cloudData.messages) {
                        for (let msg of cloudData.messages) {
                            if (msg.imageData) {
                                const images = Array.isArray(msg.imageData) ? msg.imageData : [msg.imageData];
                                for (let img of images) {
                                    if (img.blob) {
                                        addTask(img, 'blob', 'src');
                                    } else if (img.src && img.src.startsWith('data:')) {
                                        addTask(img, 'src', 'src');
                                    }
                                }
                            }
                        }
                    }
                    
                    // 处理工作台中的非 runtime 元素
                    if (cloudData.workbenchState) {
                        for (let item of cloudData.workbenchState) {
                            if (item.blob) {
                                addTask(item, 'blob', 'dataUrl');
                            } else if (item.dataUrl && item.dataUrl.startsWith('data:')) {
                                addTask(item, 'dataUrl', 'dataUrl');
                            }

                            if (item.originalBlob) {
                                addTask(item, 'originalBlob', 'originalDataUrl');
                            } else if (item.originalDataUrl && item.originalDataUrl.startsWith('data:')) {
                                addTask(item, 'originalDataUrl', 'originalDataUrl');
                            }

                            if (item.cleanPlateBlob) {
                                addTask(item, 'cleanPlateBlob', 'cleanPlateDataUrl');
                            } else if (item.cleanPlateDataUrl && item.cleanPlateDataUrl.startsWith('data:')) {
                                addTask(item, 'cleanPlateDataUrl', 'cleanPlateDataUrl');
                            }
                            
                            const handleLayers = (layers) => {
                                if (!layers || !Array.isArray(layers)) return;
                                for (let layer of layers) {
                                    if (layer.blob) {
                                        addTask(layer, 'blob', 'image');
                                    } else if (layer.image && (layer.image.startsWith('data:') || (!layer.image.startsWith('http') && layer.image.length > 1000))) {
                                        addTask(layer, 'image', 'image', !layer.image.startsWith('data:'), 'data:image/png;base64,');
                                    }

                                    if (layer.maskBlob) {
                                        addTask(layer, 'maskBlob', 'mask');
                                    } else if (layer.mask && (layer.mask.startsWith('data:') || (!layer.mask.startsWith('http') && layer.mask.length > 1000))) {
                                        addTask(layer, 'mask', 'mask', !layer.mask.startsWith('data:'), 'data:image/png;base64,');
                                    }
                                }
                            };

                            handleLayers(item.layers);
                            if (item.scene) {
                                handleLayers(item.scene.layers);
                            }
                        }
                    }

                    // 处理 runtime 工作台快照中的图片资产
                    if (cloudData.runtimeWorkspace?.currentState?.assets) {
                        for (const asset of cloudData.runtimeWorkspace.currentState.assets) {
                            if (asset.sourceImageBlob) {
                                addTask(asset, 'sourceImageBlob', 'sourceImage');
                            } else if (asset.sourceImage && asset.sourceImage.startsWith('data:')) {
                                addTask(asset, 'sourceImage', 'sourceImage');
                            }
                            if (asset.originalBlob) {
                                addTask(asset, 'originalBlob', 'originalDataUrl');
                            } else if (asset.originalDataUrl && asset.originalDataUrl.startsWith('data:')) {
                                addTask(asset, 'originalDataUrl', 'originalDataUrl');
                            }
                            if (asset.cleanPlateBlob) {
                                addTask(asset, 'cleanPlateBlob', 'cleanPlateDataUrl');
                            } else if (asset.cleanPlateDataUrl && asset.cleanPlateDataUrl.startsWith('data:')) {
                                addTask(asset, 'cleanPlateDataUrl', 'cleanPlateDataUrl');
                            }

                            const handleLayers = (layers) => {
                                if (!layers || !Array.isArray(layers)) return;
                                for (const layer of layers) {
                                    if (layer.image && (layer.image.startsWith('data:') || (!layer.image.startsWith('http') && layer.image.length > 1000))) {
                                        addTask(layer, 'image', 'image', !layer.image.startsWith('data:'), 'data:image/png;base64,');
                                    }
                                    if (layer.mask && (layer.mask.startsWith('data:') || (!layer.mask.startsWith('http') && layer.mask.length > 1000))) {
                                        addTask(layer, 'mask', 'mask', !layer.mask.startsWith('data:'), 'data:image/png;base64,');
                                    }
                                }
                            };

                            handleLayers(asset.layers);
                            if (asset.scene?.layers) {
                                handleLayers(asset.scene.layers);
                            }
                        }
                    }

                    // Execute all captured uploads in chunks of 8 concurrent requests
                    if (uploadTasks.length > 0) {
                        const chunkSize = 8;
                        for (let i = 0; i < uploadTasks.length; i += chunkSize) {
                            const chunk = uploadTasks.slice(i, i + chunkSize);
                            await Promise.all(chunk.map(task => task()));
                        }
                    }
                    
                    if (changed) {
                        await sessionDataDB.setItem(meta.id, data);
                    }

                    removeEmbeddedMediaFallback(cloudData);

                    const mergedSession = { ...meta, ...cloudData };
                    const mergedContent = getSessionContentSummary(mergedSession);
                    if (!mergedContent.hasContent) {
                        console.warn(`Skipping cloud sync for empty shell session ${meta.id}.`);
                        continue;
                    }

                    fullSessions.push(mergedSession);
                }

                if (fullSessions.length === 0) {
                    console.warn('Skipping cloud sync because no session has recoverable content.');
                    return;
                }
                
                // Keep the payload size optimized. Sort newer sessions first.
                let finalPayloadSessions = [...fullSessions].sort((a, b) => getSessionRevision(b) - getSessionRevision(a));
                
                // Strip large debugging decisionLog for all historical sessions (i >= 2) to drastically shrink payload size
                for (let i = 2; i < finalPayloadSessions.length; i++) {
                    const s = finalPayloadSessions[i];
                    if (s.decisionLog) {
                        delete s.decisionLog;
                    }
                }

                await saveSessionsToOSS(finalPayloadSessions);
                console.log('Sessions synced to cloud successfully');
            } catch (e) {
                console.error('Failed to sync sessions to cloud:', e);
            } finally {
                this._syncInFlight = null;
            }
        })();

        return this._syncInFlight;
    },
    async syncSessionsToCloud(options = {}) {
        const { immediate = false } = options;
        if (this._syncTimer) {
            clearTimeout(this._syncTimer);
            this._syncTimer = null;
        }

        if (immediate) {
            return this._performSessionsCloudSync();
        }

        this._syncTimer = setTimeout(() => {
            this._performSessionsCloudSync();
        }, 800); // 缩短跨设备同步延迟，避免“图片已上传但会话尚未上云”
    },
    async restoreSessionsFromCloud(options = {}) {
        try {
            const { pruneMissingLocal = false } = options;
            const cloudSessions = await getSessionsFromOSS();
            if (!cloudSessions || !Array.isArray(cloudSessions)) return null;
            
            console.log(`Restoring ${cloudSessions.length} sessions from cloud...`);
            const cloudSessionIds = new Set(cloudSessions.map(session => session.id));
            const maxCloudRevision = cloudSessions.reduce((maxRevision, session) => {
                return Math.max(maxRevision, getSessionRevision(session));
            }, 0);
            
            for (const session of cloudSessions) {
                const localMeta = await sessionsMetaDB.getItem(session.id);
                const localRevision = getSessionRevision(localMeta);
                const cloudRevision = getSessionRevision(session);

                if (localMeta && localRevision > cloudRevision) {
                    continue;
                }

                const localRawData = await sessionDataDB.getItem(session.id);
                let localSession = null;
                if (localRawData) {
                    try {
                        localSession = await runWorkerJob('deserializeSession', localRawData);
                    } catch (error) {
                        console.warn(`Failed to inspect local session ${session.id} before cloud restore:`, error);
                    }
                }

                const localContent = getSessionContentSummary(localSession || localMeta);
                const cloudContent = getSessionContentSummary(session);
                const cloudLooksEmpty = !cloudContent.hasContent;
                const localHasMeaningfulContent = localContent.hasContent;

                if (localHasMeaningfulContent && cloudLooksEmpty) {
                    console.warn(`Skipping cloud overwrite for session ${session.id} because cloud copy is empty while local copy has content.`);
                    continue;
                }

                const mergedCloudSession = { ...session };
                if (localSession && localContent.messageCount > cloudContent.messageCount) {
                    mergedCloudSession.messages = cloneValue(localSession.messages) || [];
                }
                if (
                    localSession &&
                    localContent.renderableRuntimeAssetCount > cloudContent.renderableRuntimeAssetCount &&
                    Array.isArray(localSession.runtimeWorkspace?.currentState?.assets)
                ) {
                    mergedCloudSession.runtimeWorkspace = cloneWorkspaceSnapshot(localSession.runtimeWorkspace);
                }

                // 1. 保存元数据
                const meta = {
                    id: mergedCloudSession.id,
                    title: mergedCloudSession.title,
                    timestamp: mergedCloudSession.timestamp,
                    updatedAt: mergedCloudSession.updatedAt || mergedCloudSession.timestamp,
                    isAutoRenamed: mergedCloudSession.isAutoRenamed || false
                };
                await sessionsMetaDB.setItem(session.id, meta);
                
                // 2. 保存详细数据（需要序列化）
                const processedSession = await runWorkerJob('serializeSession', mergedCloudSession);
                await sessionDataDB.setItem(session.id, processedSession);
            }

            if (pruneMissingLocal && cloudSessions.length > 0) {
                const localSessions = await this.getAllSessions();
                for (const localMeta of localSessions) {
                    if (cloudSessionIds.has(localMeta.id)) continue;

                    const localRevision = getSessionRevision(localMeta);
                    if (localRevision > maxCloudRevision) {
                        console.warn(`Keeping local-only session ${localMeta.id} because it is newer than current cloud baseline.`);
                        continue;
                    }

                    console.log(`Pruning locally cached session ${localMeta.id} because it no longer exists in cloud.`);
                    await sessionsMetaDB.removeItem(localMeta.id);
                    await sessionDataDB.removeItem(localMeta.id);
                }
            }
            
            return cloudSessions;
        } catch (e) {
            console.error('Failed to restore sessions from cloud:', e);
            return null;
        }
    },
    async getBestSessionToOpen() {
        try {
            const metaSessions = await this.getAllSessions();
            if (!metaSessions.length) return null;

            let bestSession = null;
            let bestScore = -1;
            let bestRevision = -1;

            for (const meta of metaSessions) {
                const fullSession = await this.getSessionData(meta.id);
                const mergedSession = { ...meta, ...(fullSession || {}) };
                const content = getSessionContentSummary(mergedSession);
                const revision = getSessionRevision(mergedSession);

                const sessionRef = state.sessions.find(session => session.id === meta.id);
                if (sessionRef) {
                    Object.assign(sessionRef, mergedSession);
                }

                if (
                    content.score > bestScore ||
                    (content.score === bestScore && revision > bestRevision)
                ) {
                    bestSession = mergedSession;
                    bestScore = content.score;
                    bestRevision = revision;
                }
            }

            return bestSession || metaSessions[0];
        } catch (error) {
            console.error('Failed to determine best session to open:', error);
            return null;
        }
    },
    // --- NEW: Asset Library DB Functions ---
    syncAssetsTimeout: null,
    async syncAssetsToCloud() {
        return;
    },
    async restoreAssetsFromCloud() {
        return [];
    },
    async saveAsset(asset) {
        try {
            const processedAsset = await runWorkerJob('serializeAsset', asset);
            await assetsDB.setItem(asset.id, processedAsset);
            this.syncAssetsToCloud();
        } catch (e) {
            console.error('Failed to save asset:', e);
        }
    },
    async getAllAssets() {
        try {
            const assets = [];
            await assetsDB.iterate((value, key) => {
                assets.push(value);
            });
            
            // Deserialize all assets (can be done in parallel)
            const deserializedAssets = await Promise.all(
                assets.map(asset => runWorkerJob('deserializeAsset', asset))
            );
            
            return deserializedAssets.sort((a, b) => b.timestamp - a.timestamp);
        } catch (e) {
            console.error('Failed to get assets:', e);
            return [];
        }
    },
    async deleteAsset(assetId) {
        try {
            await assetsDB.removeItem(assetId);
            this.syncAssetsToCloud();
        } catch (e) {
            console.error('Failed to delete asset:', e);
        }
    },
    // --- NEW: Image Cache DB Functions ---
    async saveImageCache(url, blob) {
        try {
            await imageCacheDB.setItem(url, { blob, timestamp: Date.now() });
            // 异步触发清理，不阻塞当前保存
            this.cleanupImageCache();
        } catch (e) {
            console.error('Failed to save image to cache:', e);
        }
    },
    async getImageCache(url) {
        try {
            const item = await imageCacheDB.getItem(url);
            if (item && item.blob) {
                // 更新最后访问时间
                item.timestamp = Date.now();
                await imageCacheDB.setItem(url, item);
                return item.blob;
            }
            // 兼容旧版本的纯 Blob 格式
            if (item instanceof Blob) {
                await imageCacheDB.setItem(url, { blob: item, timestamp: Date.now() });
                return item;
            }
            return null;
        } catch (e) {
            console.error('Failed to get image from cache:', e);
            return null;
        }
    },
    async cleanupImageCache() {
        try {
            const keys = await imageCacheDB.keys();
            const MAX_CACHE_ITEMS = 100; // 最多缓存 100 张工作台图片
            if (keys.length > MAX_CACHE_ITEMS) {
                const items = [];
                await imageCacheDB.iterate((value, key) => {
                    items.push({ key, timestamp: value.timestamp || 0 });
                });
                // 按时间降序排列（最新的在前面）
                items.sort((a, b) => b.timestamp - a.timestamp);
                // 截取超出部分的旧数据
                const toDelete = items.slice(MAX_CACHE_ITEMS);
                for (const item of toDelete) {
                    await imageCacheDB.removeItem(item.key);
                }
                console.log(`Cleaned up ${toDelete.length} old images from local cache.`);
            }
        } catch (e) {
            console.error('Failed to cleanup image cache:', e);
        }
    },
    async clearImageCache() {
        try {
            await imageCacheDB.clear();
        } catch (e) {
            console.error('Failed to clear image cache:', e);
        }
    }
};

let isLoadingSession = false;

export async function loadSession(sessionId) {
    if (isLoadingSession) return;
    isLoadingSession = true;
    
    try {
        window.__hasRestoredViewport = false;
        const workbenchGrid = document.getElementById('workbenchGrid');
        window.isRestoringSession = true;
    
    // 1. 清空当前画板和对话
    if (typeof window.clearWorkbench === 'function') {
        await window.clearWorkbench(true);
    } else {
        state.workbenchItems.clear();
        state.selectedWorkbenchItems.clear();
    }
    
    document.getElementById('chatMessages').innerHTML = '';
    workbenchGrid.innerHTML = `
        <div id="lightConnector" class="light-connector"></div>
        <div id="vLine" class="guide-line v"></div>
        <div id="hLine" class="guide-line h"></div>
        <div id="guide-v" class="guide-line guide-line-v"></div>
        <div id="guide-h" class="guide-line guide-line-h"></div>
    `;
    
    // 重新添加基因谱系层（确保 ID 一致）
    const svgLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgLayer.id = "genealogyLayer";
    svgLayer.classList.add("genealogy-layer");
    svgLayer.innerHTML = '<defs>' +
        '<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#cbd5e0"/></marker>' +
        '<marker id="arrowhead-active" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#2A5C82"/></marker>' +
    '</defs>';
    workbenchGrid.appendChild(svgLayer);

    // 2. 检查并确保视口层 (#workbench) 存在空状态提示
    if (!document.querySelector('#workbench > .empty-workbench-state')) {
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-workbench-state';
        emptyState.innerHTML = `
            <i class="fas fa-image"></i>
            <p>上传或生成的图片将出现在这里</p>
            <p style="font-size: 12px; margin-top: 10px;">拖拽图片重叠可触发融合反应</p>
        `;
        document.getElementById('workbench').appendChild(emptyState);
    }

    state.mainImageFile = null;
    state.referenceImageFiles = [];
    state.maskDataUrl = null;
    state.lastGeneratedImageForEditing = null;
    state.isContextPreviewHidden = false;
    state.pendingBaseImageShare = false;
    state.pendingReferenceImageShares.clear();
    state.lastGenerationContext = null;
    closeMagicWandModal();
    if (typeof window.closeLayerManagerModal === 'function') {
        window.closeLayerManagerModal();
    }

    let session = state.sessions.find(s => s.id === sessionId);
    if (session) {
        // Fetch full data if not already loaded
        if (!session.messages) {
            const fullSession = await dbHelper.getSessionData(sessionId);
            if (fullSession) {
                Object.assign(session, fullSession);
            } else {
                session.messages = [];
            }
        }
        if (!Array.isArray(session.messages)) {
            session.messages = [];
        }
        
        state.currentSessionId = sessionId;
        updateHeader(session);
        
        if (typeof window.renderHistoryList === 'function') {
            window.renderHistoryList();
        }
        
        renderMessages(session.messages);
        
        const lastBotImageMessage = [...session.messages].reverse().find(msg => 
            msg.sender === 'bot' && msg.imageData?.src
        );
        
        if (lastBotImageMessage) {
            try {
                    state.lastGeneratedImageForEditing = await dataURLToFile(
                    lastBotImageMessage.imageData.src, 
                    `context-${Date.now()}.png`
                );
            } catch (e) { 
                console.error("Failed to restore image:", e); 
                state.lastGeneratedImageForEditing = null; 
            }
        }

        const delayedRestorations = [];
        const legacyImageItems = extractLegacyImageWorkbenchItems(session);

        let runtimeRestored = hydrateRuntimeWorkspaceFromSession(session);
        if (!runtimeRestored && shouldUseStartupWorkspaceFallback(session, legacyImageItems)) {
            const fallbackSnapshot = cloneWorkspaceSnapshot(window.__startupWorkspaceSnapshot);
            if (fallbackSnapshot) {
                session.runtimeWorkspace = fallbackSnapshot;
                runtimeRestored = hydrateRuntimeWorkspaceFromSession(session);
                if (runtimeRestored) {
                    window.__startupWorkspaceConsumedSessionId = session.id;
                    await dbHelper.saveSession(session);
                }
            }
        }
        const currentWorkspace = runtime.getCurrentWorkspace();
        const runtimeAssetCount = currentWorkspace ? currentWorkspace.currentState.assetRegistry.getAll().length : 0;
        const hasRuntimeAssets = runtimeRestored && runtimeAssetCount > 0;

        if (hasRuntimeAssets && typeof window.hydrateWorkbench === 'function') {
            await window.hydrateWorkbench();
        }
        const hasRenderableRuntimeItems = getRenderableWorkbenchImageCount() > 0;
        
        // 恢复非图片工作台元素；图片资产以 runtimeWorkspace 为准
        if (session.workbenchState) {
            for (const itemState of session.workbenchState) {
                if (itemState.type === 'text-note') {
                    delayedRestorations.push(() => addTextNoteToWorkbench(0, 0, itemState));
                } else if (itemState.type === 'group-label') {
                    delayedRestorations.push(() => restoreGroupLabelToWorkbench(itemState));
                } else if (itemState.type === 'shape') {
                    delayedRestorations.push(() => {
                        if (typeof window.restoreShapeToWorkbench === 'function') {
                            window.restoreShapeToWorkbench(itemState);
                        }
                    });
                } else if (itemState.type === 'atmosphere' && itemState.key) {
                    delayedRestorations.push(() => addAtmosphereNode(itemState.key, parseFloat(itemState.rect.left) + 30, parseFloat(itemState.rect.top) + 30));
                }
            }
        }

        if (!hasRuntimeAssets || !hasRenderableRuntimeItems) {
            const imageLoadPromises = [];
            const loadedImageHashes = new Set();
            const getImageHash = (src) => {
                if (!src) return '';
                const len = src.length;
                const mid = Math.floor(len / 2);
                return `${len}-${src.substring(mid, mid + 50)}`;
            };

            for (const itemState of legacyImageItems) {
                const hash = getImageHash(itemState.dataUrl);
                loadedImageHashes.add(hash);

                imageLoadPromises.push(
                    dataURLToFile(itemState.dataUrl, `restored-${Date.now()}.png`)
                        .then(async file => {
                            await addImageToWorkbench(file, '恢复的图片', {
                                id: itemState.id,
                                dataUrl: itemState.dataUrl,
                                left: itemState.rect.left,
                                top: itemState.rect.top,
                                initialWidth: parseFloat(itemState.rect.width),
                                initialHeight: parseFloat(itemState.rect.height),
                                zIndex: itemState.rect.zIndex,
                                genealogy: itemState.genealogy,
                                parentId: itemState.parentId,
                                layerName: itemState.layerName,
                                originalBbox: itemState.originalBbox,
                                type: itemState.type,
                                layers: itemState.layers,
                                scene: itemState.scene,
                                semanticViews: itemState.semanticViews,
                                hasFullSemanticAnalysis: itemState.hasFullSemanticAnalysis,
                                originalDataUrl: itemState.originalDataUrl,
                                cleanPlateDataUrl: itemState.cleanPlateDataUrl,
                                cleanPlateStatus: itemState.cleanPlateStatus
                            });
                            return file;
                        })
                        .catch(e => {
                            console.error('Failed to restore legacy state image:', e);
                            return null;
                        })
                );
            }

            if (!legacyImageItems.length) {
            // 如果没有保存过画板状态，则从历史记录加载所有图片
                for (const msg of session.messages) {
                    if (msg.imageData) {
                        const images = Array.isArray(msg.imageData) ? msg.imageData : [msg.imageData];
                        images.forEach(img => {
                            if (img.src) {
                                const hash = getImageHash(img.src);
                                if (loadedImageHashes.has(hash)) return;
                                loadedImageHashes.add(hash);

                                imageLoadPromises.push(
                                    dataURLToFile(img.src, `history-${Date.now()}.png`)
                                        .then(async file => {
                                            await addImageToWorkbench(
                                                file, 
                                                msg.sender === 'user' ? '上传图片' : 'AI生成'
                                            );
                                            return file;
                                        })
                                        .catch(e => {
                                            console.error('Failed to load history image:', e);
                                            return null;
                                        })
                                );
                            }
                        });
                    }
                }
            }

            await Promise.all(imageLoadPromises);
        }
        
        // Execute delayed restorations (texts, shapes, etc.) AFTER images are loaded
        // This ensures they are appended later in the DOM, maintaining z-index hierarchy naturally
        delayedRestorations.forEach(restoreFn => restoreFn());
        
        // 自动触发所有图层的渲染，确保净化效果可见
        state.workbenchItems.forEach((item, itemId) => {
            if (typeof window.renderCanvasLayers === 'function') {
                window.renderCanvasLayers(itemId);
            }
        });

        // Restored assets can be far outside the current viewport due to preserved pan/zoom.
        // Reframe the canvas after restore so the user actually sees the recovered images.
        const hasSavedViewport =
            typeof session.runtimeWorkspace?.viewport?.zoom === 'number' &&
            typeof session.runtimeWorkspace?.viewport?.panOffsetX === 'number' &&
            typeof session.runtimeWorkspace?.viewport?.panOffsetY === 'number';

        if (state.workbenchItems.size > 0 && typeof window.viewAllWorkbenchItems === 'function' && !hasSavedViewport) {
            requestAnimationFrame(() => {
                window.viewAllWorkbenchItems();
            });
        }
        
        window.isRestoringSession = false;
        
        if (window.historyManager) {
            window.historyManager.clear();
            window.historyManager.pushState();
        }
        
        if (typeof window.updateImagePreview === 'function') {
            await window.updateImagePreview();
        }
        if (typeof window.renderHistoryList === 'function') {
            window.renderHistoryList();
        }
        sidebarState.closeMobileMenu();
        
        if (state.lastGeneratedImageForEditing) {
            state.mainImageFile = state.lastGeneratedImageForEditing;
            state.pendingBaseImageShare = true;
        }
    } else {
        window.isRestoringSession = false;
    }
    } finally {
        isLoadingSession = false;
    }
}

export async function startNewSession() {
    updateHeader(null);
    state.currentSessionId = null;
    state.mainImageFile = null; state.referenceImageFiles = []; state.maskDataUrl = null;
    state.lastGeneratedImageForEditing = null; state.isContextPreviewHidden = false;
    state.pendingBaseImageShare = false; state.pendingReferenceImageShares.clear();
    state.lastGenerationContext = null; 
    if (typeof window.closeLayerManagerModal === 'function') {
        window.closeLayerManagerModal();
    }
    if (typeof window.updateImagePreview === 'function') {
        await window.updateImagePreview();
    }
    const userInput = document.getElementById('userInput');
    if (userInput) userInput.value = ''; 
    if (typeof window.updateSendBtnState === 'function') {
        window.updateSendBtnState(); 
    }
    if (typeof window.renderHistoryList === 'function') {
        window.renderHistoryList();
    }
    closeMagicWandModal(); sidebarState.closeMobileMenu();
    
    // 清空工作台
    if (typeof window.clearWorkbench === 'function') {
        await window.clearWorkbench(true); // 跳过确认
    } else {
        state.workbenchItems.clear();
        state.selectedWorkbenchItems.clear();
    }
    
    if (window.historyManager) {
        window.historyManager.clear();
        window.historyManager.pushState();
    }

    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        renderMessages([]);
    }
}
