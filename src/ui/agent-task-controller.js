import { isAgentRuntimeFeatureEnabled } from '../core/config.js';
import { state } from '../core/state.js';
import { addAgentJobCard, addMessage } from './chat-panel.js';

const ASSET_ACTION_PATTERN = /(上次|之前|历史|项目中|复用|复制|资产库|素材库|找到|放入|放进)/i;
const ASSET_SUBJECT_PATTERN = /(资产|素材|对象|物品|沙发|椅子|桌子|灯|台灯|床|柜|sofa|chair|table|lamp)/i;
const LAYER_EXTRACTION_PATTERN = /(提取|拆解|分离|拆出|独立图层)/i;
const LAYER_EXTRACTION_OBJECT_PATTERN = /(图层|对象|物体|沙发|椅子|桌子|灯具|台灯|床|柜|sofa|chair|table|lamp|全部|所有)/i;

export function isAssetReuseRequest(text = '') {
    const normalized = String(text).trim();
    if (!normalized) return false;
    return ASSET_ACTION_PATTERN.test(normalized) && ASSET_SUBJECT_PATTERN.test(normalized);
}

export function isLayerExtractionRequest(text = '') {
    const normalized = String(text).trim();
    return Boolean(normalized) && LAYER_EXTRACTION_PATTERN.test(normalized) && LAYER_EXTRACTION_OBJECT_PATTERN.test(normalized);
}

function normalizeLayerText(value = '') {
    return String(value).toLowerCase().replace(/[\s_\-:：]/g, '');
}

export function resolveLayerExtractionRequest(text = '', appState = state) {
    const normalized = normalizeLayerText(text);
    const itemId = appState.currentActiveWorkbenchItemId || appState.workspaceContext?.activeItemId || [...appState.selectedWorkbenchItems][0];
    const item = itemId ? appState.workbenchItems.get(itemId) : null;
    if (!item) return { ok: false, reason: '当前没有可执行图层提取的工作区图片。' };

    const layers = item.scene?.layers || item.layers || [];
    const candidates = layers.map((layer, index) => ({ layer, index })).filter(({ layer, index }) => {
        const layerState = item.layerStates instanceof Map ? item.layerStates.get(index) : item.layerStates?.[index];
        const name = normalizeLayerText(layer?.name || layer?.label || '');
        const category = normalizeLayerText(layer?.category || layer?.semanticType || '');
        return Boolean(layer?.id) && layerState?.locked !== true && !/(background|背景|底板)/i.test(`${name}${category}`);
    });
    if (candidates.length === 0) return { ok: false, reason: '当前图片没有可提取的语义图层。' };

    const extractAll = /(全部|所有|所有图层|可编辑图层)/.test(String(text));
    const matched = extractAll
        ? candidates
        : candidates.filter(({ layer }) => {
            const name = normalizeLayerText(layer.name || layer.label || '');
            if (name && normalized.includes(name)) return true;
            if (normalized.includes('灯具') && name.includes('灯')) return true;
            if (normalized.includes('家具') && /(沙发|椅|桌|床|柜)/.test(name)) return true;
            const objectKeywords = String(text).match(/沙发|椅子?|桌子?|台灯|落地灯|灯具|床|柜子?|窗帘|植物|花瓶/gi) || [];
            if (objectKeywords.some(keyword => name.includes(normalizeLayerText(keyword)))) return true;
            return false;
        });
    if (matched.length === 0) {
        return {
            ok: false,
            reason: `没有在当前图层列表中匹配到“${text}”对应的对象，请先在语义列表中选择图层。`
        };
    }
    return {
        ok: true,
        itemId,
        layerIds: matched.map(({ layer }) => layer.id),
        layerNames: matched.map(({ layer }) => layer.name || layer.label || layer.id),
        item
    };
}

export function parseAssetReuseRequest(text = '') {
    const normalized = String(text).trim();
    const subjectMatch = normalized.match(/(?:项目中的|项目的|其中的|里面的|中的|找到|找|复制|复用)\s*(?:一个|一件)?(.{1,16}?)(?=放入|放进|复制|复用|缩小|放大|[，。,.!?！？]|$)/i);
    const query = subjectMatch?.[1]
        ? subjectMatch[1].replace(/^(一个|一件|这个|那个)/, '')
        : normalized
            .replace(/上次项目(?:中的|的)?|当前房间|当前场景|当前工作区|放入|放进|缩小\s*\d+%?|放大\s*\d+%?/gi, ' ')
            .trim();

    const scaleMatch = normalized.match(/(缩小|放大)\s*(\d+(?:\.\d+)?)\s*%/);
    const percentage = scaleMatch ? Number(scaleMatch[2]) / 100 : null;
    const scale = percentage === null ? undefined : scaleMatch[1] === '缩小' ? 1 - percentage : 1 + percentage;

    return {
        query: query || normalized,
        scale: scale && scale > 0 ? scale : undefined,
        placement: {}
    };
}

function refreshWorkbench() {
    if (typeof window !== 'undefined' && typeof window.hydrateWorkbench === 'function') {
        return Promise.resolve(window.hydrateWorkbench());
    }
    return Promise.resolve();
}

async function runAgentJobInChat({ agent, job, onFinished, onJobCreated }) {
    let card = null;
    let unsubscribe = null;
    if (onJobCreated) await onJobCreated(job);
    const updateCard = (nextJob) => {
        if (card) card.update(nextJob);
        if (unsubscribe && ['completed', 'cancelled'].includes(nextJob?.status)) {
            unsubscribe();
            unsubscribe = null;
        }
        return nextJob;
    };
    const finish = async (result) => {
        updateCard(result);
        if (result.status === 'waiting_confirmation' || result.status === 'completed') await refreshWorkbench();
        if (onFinished) await onFinished(result);
        return result;
    };
    const execute = () => agent.executeJob(job.id).then(finish);

    card = addAgentJobCard(job, {
        onConfirm: async () => finish(await agent.confirmJob(job.id)),
        onCancel: async () => finish(await agent.cancelJob(job.id)),
        onRetry: async () => finish(await agent.retryJob(job.id)),
        onError: (error) => {
            const current = agent.getJob(job.id) || job;
            updateCard({ ...current, error: { message: error?.message || String(error) } });
        }
    });
    unsubscribe = agent.subscribe(event => {
        if (event.jobId !== job.id) return;
        const current = agent.getJob(job.id);
        if (current) updateCard(current);
    });
    return execute();
}

export async function startLayerExtractionJob({ runtime, itemId, layerIds = [], goal = '提取选定语义图层', onFinished } = {}) {
    if (!runtime?.getAgentRuntime) throw new Error('Agent Runtime is unavailable.');
    if (!itemId || !Array.isArray(layerIds) || layerIds.length === 0) {
        throw new Error('Layer extraction requires at least one semantic layer.');
    }
    const agent = runtime.getAgentRuntime();
    const job = agent.createLayerExtractionJob(goal, itemId, {
        layerIds: [...new Set(layerIds.filter(Boolean))],
        requiresConfirmation: true
    });
    return runAgentJobInChat({ agent, job, onFinished });
}

export async function handleLayerExtractionRequest({ text, runtime, onUserMessage, onFinished } = {}) {
    if (!isAgentRuntimeFeatureEnabled('magicLayersCommand') || !isLayerExtractionRequest(text)) return null;
    if (onUserMessage) await onUserMessage(text);
    const target = resolveLayerExtractionRequest(text, state);
    if (!target.ok) {
        addMessage({ sender: 'bot', type: 'text', content: `⚠️ ${target.reason}` });
        return { status: 'needs_selection', reason: target.reason };
    }
    return startLayerExtractionJob({
        runtime,
        itemId: target.itemId,
        layerIds: target.layerIds,
        goal: `提取${target.layerNames.join('、')}`,
        onFinished
    });
}

/**
 * Chat-to-Agent adapter. It owns no workspace state: all mutations go through
 * AgentRuntime, and this module only coordinates the user-facing card.
 */
export async function handleAssetReuseRequest({
    text,
    runtime,
    onUserMessage,
    onFinished
}) {
    if (!isAgentRuntimeFeatureEnabled('assetReuseFromChat') || !isAssetReuseRequest(text)) return null;

    if (!runtime?.getAgentRuntime) throw new Error('Agent Runtime is unavailable.');
    if (onUserMessage) await onUserMessage(text);

    const parsed = parseAssetReuseRequest(text);
    const agent = runtime.getAgentRuntime();
    const job = agent.createAssetReuseJob(text, parsed.query, {
        placement: parsed.placement,
        scale: parsed.scale,
        requiresConfirmation: true
    });

    return runAgentJobInChat({ agent, job, onFinished });
}

export async function startObjectEditJob({
    runtime,
    itemId,
    prompt,
    mode = 'variant',
    onUserMessage,
    onFinished,
    onJobCreated
} = {}) {
    if (!runtime?.getAgentRuntime) throw new Error('Agent Runtime is unavailable.');
    if (!itemId || !String(prompt || '').trim()) throw new Error('Object editing requires a target and a prompt.');
    const agent = runtime.getAgentRuntime();
    const label = mode === 'current_layer' ? '更新当前图层' : '生成对象新版本';
    const job = agent.createObjectEditJob(
        `${label}：${String(prompt).trim()}`,
        itemId,
        String(prompt).trim(),
        mode === 'current_layer' ? 'current_layer' : 'variant',
        { requiresConfirmation: true }
    );
    if (onUserMessage) await onUserMessage(`编辑对象：${String(prompt).trim()}`);
    return runAgentJobInChat({ agent, job, onFinished, onJobCreated });
}
