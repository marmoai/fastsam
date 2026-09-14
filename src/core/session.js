import { state } from './state.js';
import { addMessage, renderMessages } from '../ui/chat-panel.js';
import { dataURLToFile, fileToDataURL } from './utils.js';
import { addTextNoteToWorkbench, restoreGroupLabelToWorkbench, addAtmosphereNode, addImageToWorkbench } from '../ui/workbench-core.js';
import { sidebarState } from '../ui/sidebar.js';
import { closeMagicWandModal } from '../ui/modals.js';
import { saveSessionsToOSS, getSessionsFromOSS, getSessionFromOSS, uploadImageToOSS } from '../services/ossService.js';
import { updateHeader } from '../ui/header.js';
import { runtime } from '../runtime/CoreRuntime';
import { assetCatalog } from '../runtime/AssetCatalog';

import localforage from 'localforage';

const sessionsMetaDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'sessionsMeta' });
const sessionDataDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'sessionData' });
const assetsDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'assets' });
const imageCacheDB = localforage.createInstance({ name: 'MarmoAid', storeName: 'imageCache' });
const LAST_ACTIVE_SESSION_STORAGE_KEY = 'marmo:last_active_session_id';

const dbWorker = new Worker(new URL('./db-worker.js', import.meta.url), { type: 'module' });
const pendingJobs = new Map();
let jobIdCounter = 0;

export function getLastActiveSessionId() {
    try {
        return typeof localStorage !== 'undefined'
            ? (localStorage.getItem(LAST_ACTIVE_SESSION_STORAGE_KEY) || '')
            : '';
    } catch (error) {
        console.warn('Failed to read last active session id:', error);
        return '';
    }
}

export function setLastActiveSessionId(sessionId) {
    if (!sessionId) return;
    try {
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem(LAST_ACTIVE_SESSION_STORAGE_KEY, sessionId);
        }
    } catch (error) {
        console.warn('Failed to persist last active session id:', error);
    }
}

export function clearLastActiveSessionId(sessionId = '') {
    try {
        if (typeof localStorage === 'undefined') return;
        const current = localStorage.getItem(LAST_ACTIVE_SESSION_STORAGE_KEY) || '';
        if (!sessionId || current === sessionId) {
            localStorage.removeItem(LAST_ACTIVE_SESSION_STORAGE_KEY);
        }
    } catch (error) {
        console.warn('Failed to clear last active session id:', error);
    }
}

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
            sourceTextLayerId: item.sourceTextLayerId || null,
            key: item.key
        });
    });

    return savedState;
}

function serializeCurrentRuntimeWorkspace() {
    const workspace = runtime.getCurrentWorkspace();
    if (!workspace) return null;

    // runtimeDisplayUrl can be a page-local blob URL. It must never be
    // persisted; sourceImage remains the durable OSS address.
    const assets = workspace.currentState.assetRegistry.getAll().map(asset => {
        const serializableAsset = { ...asset };
        delete serializableAsset.runtimeDisplayUrl;
        return serializableAsset;
    });

    return {
        projectId: workspace.projectId,
        name: workspace.name,
        currentState: {
            stateId: workspace.currentState.stateId,
            canvasState: workspace.currentState.canvasState,
            assets,
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
    'segmentationSourceUrl',
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
    segmentationSourceBlob: 'segmentationSourceUrl',
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

const INLINE_DATA_URL_PATTERN = /data:image\/(?:png|jpe?g|webp|gif|avif);base64,[A-Za-z0-9+/=_-]+/g;

function collectInlineDataUrlUploadTasks(value, addInlineTask, visited = new WeakSet()) {
    if (!value || typeof value !== 'object') return;
    if (visited.has(value)) return;
    visited.add(value);

    if (Array.isArray(value)) {
        value.forEach(item => collectInlineDataUrlUploadTasks(item, addInlineTask, visited));
        return;
    }

    Object.entries(value).forEach(([key, child]) => {
        // These fields are handled by the regular media walker. Scanning them
        // again would create duplicate uploads and duplicate replacements.
        if (MEDIA_FIELD_NAMES.has(key)) return;
        if (typeof child === 'string' && child.includes('data:image/')) {
            const matches = child.match(INLINE_DATA_URL_PATTERN);
            if (matches?.length) {
                addInlineTask(value, key, matches);
                return;
            }
        }
        if (child && typeof child === 'object') {
            collectInlineDataUrlUploadTasks(child, addInlineTask, visited);
        }
    });
}

function replaceInlineDataUrlsInObject(value, replacements, visited = new WeakSet()) {
    if (!value || typeof value !== 'object') return false;
    if (visited.has(value)) return false;
    visited.add(value);
    let changed = false;

    if (Array.isArray(value)) {
        value.forEach(item => {
            if (replaceInlineDataUrlsInObject(item, replacements, visited)) changed = true;
        });
        return changed;
    }

    Object.entries(value).forEach(([key, child]) => {
        if (MEDIA_FIELD_NAMES.has(key)) return;
        if (typeof child === 'string' && child.includes('data:image/')) {
            let next = child;
            replacements.forEach((replacement, source) => {
                next = next.split(source).join(replacement);
            });
            if (next !== child) {
                value[key] = next;
                changed = true;
            }
            return;
        }
        if (child && typeof child === 'object') {
            if (replaceInlineDataUrlsInObject(child, replacements, visited)) changed = true;
        }
    });
    return changed;
}

function removeInlineDataUrlFallback(value, visited = new WeakSet()) {
    if (!value || typeof value !== 'object') return;
    if (visited.has(value)) return;
    visited.add(value);

    if (Array.isArray(value)) {
        value.forEach(item => removeInlineDataUrlFallback(item, visited));
        return;
    }

    Object.entries(value).forEach(([key, child]) => {
        if (MEDIA_FIELD_NAMES.has(key)) return;
        if (typeof child === 'string' && child.includes('data:image/')) {
            value[key] = child.replace(INLINE_DATA_URL_PATTERN, '');
            return;
        }
        if (child && typeof child === 'object') {
            removeInlineDataUrlFallback(child, visited);
        }
    });
}

function removeIfEmbeddedMedia(obj, key) {
    const value = obj?.[key];
    if (typeof value !== 'string') {
        delete obj[key];
        return;
    }
    if (!value.startsWith('http://') && !value.startsWith('https://')) {
        delete obj[key];
    }
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

    if (Array.isArray(semanticViews.completionAssets)) {
        semanticViews.completionAssets = semanticViews.completionAssets.map(asset => {
            const slimAsset = { ...asset };
            if (slimAsset.observedCutout) {
                slimAsset.observedCutout = { ...slimAsset.observedCutout };
                removeIfEmbeddedMedia(slimAsset.observedCutout, 'cutoutUrl');
                removeIfEmbeddedMedia(slimAsset.observedCutout, 'maskUrl');
            }
            if (slimAsset.canonicalAsset) {
                slimAsset.canonicalAsset = { ...slimAsset.canonicalAsset };
                removeIfEmbeddedMedia(slimAsset.canonicalAsset, 'cutoutUrl');
                removeIfEmbeddedMedia(slimAsset.canonicalAsset, 'maskUrl');
                removeIfEmbeddedMedia(slimAsset.canonicalAsset, 'previewUrl');
            }
            return slimAsset;
        });
    }

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
    
    async saveSession(session, options = {}) {
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

            // Keep a lightweight cross-project index. The catalog stores only
            // IDs and metadata; image payloads remain in sessionDataDB/OSS.
            await assetCatalog.upsertWorkspaceSnapshot(nextSession.runtimeWorkspace?.currentState ? {
                projectId: nextSession.runtimeWorkspace.projectId || session.id,
                projectName: nextSession.runtimeWorkspace.name || session.title || session.id,
                assets: nextSession.runtimeWorkspace.currentState.assets || []
            } : { projectId: session.id, projectName: session.title || session.id, assets: [] });

            // 4. 普通保存继续使用延迟同步；撤销/重做恢复会显式等待
            // 立即同步，确保刷新时不会读到恢复前的 runtimeWorkspace。
            if (options.syncImmediately) {
                await this.syncSessionsToCloud({ immediate: true });
            } else {
                this.syncSessionsToCloud();
            }
            return true;
        } catch (e) {
            console.error('Failed to save session to DB:', e);
            return false;
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
    async rebuildAssetCatalog(sessionMetas = null) {
        try {
            const metas = Array.isArray(sessionMetas) ? sessionMetas : await this.getAllSessions();
            let indexedAssetCount = 0;
            for (const meta of metas) {
                const session = await this.getSessionData(meta.id);
                const runtimeWorkspace = session?.runtimeWorkspace;
                if (!runtimeWorkspace?.currentState) continue;
                const assets = Array.isArray(runtimeWorkspace.currentState.assets)
                    ? runtimeWorkspace.currentState.assets
                    : [];
                await assetCatalog.upsertWorkspaceSnapshot({
                    projectId: runtimeWorkspace.projectId || session.id || meta.id,
                    projectName: runtimeWorkspace.name || session.title || meta.title || meta.id,
                    assets
                });
                indexedAssetCount += assets.length;
            }
            return indexedAssetCount;
        } catch (error) {
            console.error('Failed to rebuild asset catalog:', error);
            return 0;
        }
    },
    async restoreSessionDataFromCloud(sessionId) {
        try {
            const cloudSession = await getSessionFromOSS(sessionId);
            if (!cloudSession) return null;

            const meta = {
                id: cloudSession.id || sessionId,
                title: cloudSession.title || '未命名项目',
                timestamp: cloudSession.timestamp || Date.now(),
                updatedAt: cloudSession.updatedAt || cloudSession.timestamp || Date.now(),
                isAutoRenamed: !!cloudSession.isAutoRenamed
            };
            await sessionsMetaDB.setItem(sessionId, meta);
            const processedSession = await runWorkerJob('serializeSession', cloudSession);
            await sessionDataDB.setItem(sessionId, processedSession);
            if (cloudSession.runtimeWorkspace?.currentState) {
                await assetCatalog.upsertWorkspaceSnapshot({
                    projectId: cloudSession.runtimeWorkspace.projectId || sessionId,
                    projectName: cloudSession.runtimeWorkspace.name || cloudSession.title || sessionId,
                    assets: cloudSession.runtimeWorkspace.currentState.assets || []
                });
            }
            return cloudSession;
        } catch (error) {
            console.error(`Failed to restore session ${sessionId} from cloud:`, error);
            return null;
        }
    },
    async deleteSession(sessionId) {
        try {
            await sessionsMetaDB.removeItem(sessionId);
            await sessionDataDB.removeItem(sessionId);
            await assetCatalog.removeProject(sessionId);
            clearLastActiveSessionId(sessionId);
            // 删除需要尽快同步到云端，避免其他设备继续恢复旧项目
            await this.syncSessionsToCloud({ immediate: true });
        } catch (e) {
            console.error('Failed to delete session:', e);
        }
    },
    // --- NEW: Cloud Sync Functions ---
    _syncTimer: null,
    _syncInFlight: null,
    _syncQueuedPromise: null,
    async _performSessionsCloudSync() {
        if (this._syncInFlight) {
            // A save can happen while an older, potentially stale payload is
            // still uploading. Queue exactly one follow-up sync so the latest
            // local session always gets a chance to replace that payload.
            if (!this._syncQueuedPromise) {
                const activeSync = this._syncInFlight;
                this._syncQueuedPromise = activeSync
                    .then(() => this._performSessionsCloudSync())
                    .finally(() => {
                        this._syncQueuedPromise = null;
                    });
            }
            return this._syncQueuedPromise;
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
                    const inlineReplacements = new Map();
                    const inlineUploadCache = new Map();
                    const addTask = (obj, sourceProp, targetProp, isBase64 = false, fallbackPrefix = '', sessionId = meta.id) => {
                        let sourceData = obj[sourceProp];
                        if (!sourceData) return;
                        uploadTasks.push(async () => {
                            try {
                                if (isBase64 && typeof sourceData === 'string' && !sourceData.startsWith('data:') && !sourceData.startsWith('http') && sourceData.length > 1000) {
                                    sourceData = fallbackPrefix + sourceData;
                                }
                                const url = await uploadImageToOSS(sourceData, {
                                    sessionId,
                                    preserveOriginal: targetProp === 'segmentationSourceUrl'
                                });
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
                    collectInlineDataUrlUploadTasks(cloudData, (obj, key, matches) => {
                        uploadTasks.push(async () => {
                            let next = String(obj[key] || '');
                            for (const sourceData of new Set(matches)) {
                                let url = inlineUploadCache.get(sourceData);
                                if (!url) {
                                    try {
                                        url = await uploadImageToOSS(sourceData, { sessionId: meta.id });
                                        inlineUploadCache.set(sourceData, url || '');
                                    } catch (error) {
                                        console.error('Failed to upload inline HTML media:', error);
                                        url = '';
                                        inlineUploadCache.set(sourceData, url);
                                    }
                                }
                                if (url) {
                                    inlineReplacements.set(sourceData, url);
                                    next = next.split(sourceData).join(url);
                                } else {
                                    next = next.split(sourceData).join('');
                                }
                            }
                            obj[key] = next;
                            changed = true;
                        });
                    });
                    
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

                            if (item.segmentationSourceBlob) {
                                addTask(item, 'segmentationSourceBlob', 'segmentationSourceUrl');
                            } else if (item.segmentationSourceUrl && item.segmentationSourceUrl.startsWith('data:')) {
                                addTask(item, 'segmentationSourceUrl', 'segmentationSourceUrl');
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
                            // This is only a current-page preview URL. Never
                            // send a blob URL to local or cloud persistence.
                            delete asset.runtimeDisplayUrl;
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
                            if (asset.segmentationSourceBlob) {
                                addTask(asset, 'segmentationSourceBlob', 'segmentationSourceUrl');
                            } else if (asset.segmentationSourceUrl && asset.segmentationSourceUrl.startsWith('data:')) {
                                addTask(asset, 'segmentationSourceUrl', 'segmentationSourceUrl');
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

                    // The OSS upload endpoint rate-limits bursts. uploadImageToOSS
                    // also has a global queue, but keep this loop sequential so a
                    // large historical session cannot create an upload burst.
                    if (uploadTasks.length > 0) {
                        const chunkSize = 1;
                        for (let i = 0; i < uploadTasks.length; i += chunkSize) {
                            const chunk = uploadTasks.slice(i, i + chunkSize);
                            await Promise.all(chunk.map(task => task()));
                        }
                    }

                    // Repair sessions that already contain a Motion HTML
                    // payload with embedded Base64, so the next sync does not
                    // rediscover and resend the same oversized content.
                    if (inlineReplacements.size > 0) {
                        if (replaceInlineDataUrlsInObject(data, inlineReplacements)) {
                            changed = true;
                        }
                    }

                    // Also scrub unresolved inline media from the local copy.
                    // Otherwise every later save would rehydrate the old HTML
                    // and retry the same oversized request forever.
                    removeInlineDataUrlFallback(data);
                    
                    if (changed) {
                        await sessionDataDB.setItem(meta.id, data);
                    }

                    removeEmbeddedMediaFallback(cloudData);
                    // If an upload failed, never send the original Base64 in
                    // the JSON request. The preview may lose that media, but
                    // cloud sync must remain below the gateway limit.
                    removeInlineDataUrlFallback(cloudData);

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

                // 启动阶段只同步清单。详情保持本地缓存；若云端更新，则丢弃
                // 旧详情，等用户真正打开该会话时再按 sessionId 下载。
                const meta = {
                    id: session.id,
                    title: session.title || '未命名项目',
                    timestamp: session.timestamp,
                    updatedAt: session.updatedAt || session.timestamp,
                    isAutoRenamed: !!session.isAutoRenamed
                };
                await sessionsMetaDB.setItem(session.id, meta);
                if (cloudRevision > localRevision) {
                    await sessionDataDB.removeItem(session.id);
                    await assetCatalog.removeProject(session.id);
                }
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
                    await assetCatalog.removeProject(localMeta.id);
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

            const lastActiveSessionId = getLastActiveSessionId();
            if (lastActiveSessionId) {
                const matchedMeta = metaSessions.find(session => session.id === lastActiveSessionId);
                if (matchedMeta) {
                    return matchedMeta;
                }
            }
            return metaSessions[0];
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

    // The workbench item module owns the single empty-state placeholder.
    window.syncWorkbenchEmptyState?.();

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
            const cloudSession = fullSession || await dbHelper.restoreSessionDataFromCloud(sessionId);
            if (cloudSession) Object.assign(session, cloudSession);
            else session.messages = [];
        }
        if (!Array.isArray(session.messages)) {
            session.messages = [];
        }
        
        state.currentSessionId = sessionId;
        setLastActiveSessionId(sessionId);
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
                                segmentationSourceUrl: itemState.segmentationSourceUrl,
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
