import { state } from '../core/state.js';
import { handleAssetReuseRequest, handleLayerExtractionRequest, startObjectEditJob } from './agent-task-controller.js';
import { SemanticRecommender } from '../runtime/SemanticRecommender';
import { describeTaskBucket, inferTaskBucketFromText } from '../runtime/taskBuckets';
import { resultFeedbackRuntime } from '../runtime/ResultFeedbackRuntime';

const STATUS_LABELS = {
    planned: '已规划',
    running: '执行中',
    verifying: '验证中',
    waiting_confirmation: '等待确认',
    committing: '提交中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消'
};

function text(node, value) {
    if (node) node.textContent = value == null ? '' : String(value);
}

function renderFlow(container, labels, className = '') {
    if (!container) return;
    container.replaceChildren();
    labels.forEach((label, index) => {
        const step = document.createElement('span');
        step.className = className;
        step.textContent = label;
        container.appendChild(step);
        if (index < labels.length - 1) {
            const arrow = document.createElement('i');
            arrow.className = 'fas fa-arrow-right';
            arrow.setAttribute('aria-hidden', 'true');
            container.appendChild(arrow);
        }
    });
}

function currentAsset() {
    const itemId = state.currentActiveWorkbenchItemId || state.workspaceContext?.activeItemId;
    if (!itemId || !state.selectedWorkbenchItems?.has(itemId)) return { itemId: null, item: null, asset: null };
    return {
        itemId,
        item: state.workbenchItems.get(itemId) || null,
        asset: window.mvrRuntime?.getCurrentWorkspace?.()?.currentState?.assetRegistry?.get(itemId) || null
    };
}

function isObjectLayer(item) {
    return ['layer-explode', 'layer-extract', 'isolated-edit', 'extraction']
        .includes(String(item.type || '').toLowerCase());
}

function layerNamesForJob(job) {
    const step = job?.plan?.steps?.[0];
    const ids = step?.params?.layerIds || [];
    const item = state.workbenchItems.get(step?.params?.assetId);
    const layers = item?.scene?.layers || item?.layers || [];
    return ids.map(id => layers.find(layer => layer?.id === id)?.name || id).filter(Boolean);
}

function createPlanView(root) {
    if (!root) return null;
    root.innerHTML = `
        <div class="agent-task-capsule-head">
            <div>
                <span class="agent-visual-kicker">AGENT TASK</span>
                <strong class="agent-visual-plan-title"></strong>
            </div>
            <span class="agent-visual-plan-status"></span>
        </div>
        <div class="agent-task-capsule-summary" aria-label="Agent 任务摘要">
            <span data-plan-source></span>
            <span data-plan-version></span>
            <span data-plan-target></span>
            <span data-plan-relation></span>
        </div>
        <div class="agent-visual-plan-details" hidden>
            <div class="agent-visual-plan-route"></div>
            <div class="agent-visual-plan-grid">
                <div><span>来源对象</span><strong data-plan-source-detail></strong></div>
                <div><span>目标场景</span><strong data-plan-target-detail></strong></div>
                <div><span>关系变化</span><strong data-plan-relation-detail></strong></div>
                <div><span>安全策略</span><strong data-plan-risk-detail></strong></div>
            </div>
            <div class="agent-visual-plan-steps"></div>
        </div>
        <span class="agent-task-capsule-hint">点击查看任务详情</span>
    `;
    root.hidden = true;
    root.setAttribute('role', 'button');
    root.setAttribute('tabindex', '0');
    root.setAttribute('aria-label', '打开 Agent 任务详情');
    return root;
}

function updatePlan(root, job) {
    if (!root || !job) return;
    root.hidden = false;
    root.classList.add('is-visible');
    root.dataset.jobId = job.id || '';
    root.dataset.status = job.status || '';
    root.classList.toggle('is-running', ['created', 'planned', 'running', 'verifying', 'committing'].includes(job.status));
    root.classList.toggle('is-waiting', job.status === 'waiting_confirmation');
    root.classList.toggle('is-completed', job.status === 'completed');
    root.classList.toggle('is-failed', job.status === 'failed');
    text(root.querySelector('.agent-visual-plan-title'), job.goal || '视觉任务');
    text(root.querySelector('.agent-visual-plan-status'), STATUS_LABELS[job.status] || job.status);

    const resultRefs = job.result?.commandResults?.flatMap(result => Object.entries(result.outputRefs || {})) || [];
    const ref = key => resultRefs.find(([name]) => name === key)?.[1];
    const layerNames = layerNamesForJob(job);
    const source = ref('sourceAssetId') || (layerNames.length ? layerNames.join('、') : job.plan?.steps?.[0]?.params?.query?.text || '待解析资产');
    const workspace = window.mvrRuntime?.getCurrentWorkspace?.();
    const target = workspace?.name
        ? `当前场景 · ${workspace.name}`
        : ref('projectId') || workspace?.projectId || '当前 Workspace';
    const relation = job.plan?.type === 'layer_extraction'
        ? (layerNames.length ? `提取 ${layerNames.join('、')}` : '生成独立可编辑图层')
        : '资产 → 当前场景 / 保持可编辑';
    const risk = job.verification?.passed === false
        ? '验证未通过，暂不提交'
        : job.plan?.requiresConfirmation === false ? '验证后自动提交' : '验证后等待确认';
    const sourceLabel = layerNames.length
        ? layerNames.join('、')
        : job.plan?.type === 'asset_reuse'
            ? `复用 · ${job.plan?.steps?.[0]?.params?.query?.text || '资产'}`
            : job.plan?.steps?.[0]?.params?.prompt || job.goal || '当前对象';
    const versionAssetId = ref('newAssetId') || ref('assetId') || job.targetAssets?.[0];
    const versionAsset = versionAssetId
        ? window.mvrRuntime?.getCurrentWorkspace?.()?.currentState?.assetRegistry?.get(versionAssetId)
        : null;
    const version = ref('version') || versionAsset?.version || (job.plan?.type === 'capability' ? '生成中' : '待确认');
    const versionLabel = ['生成中', '待确认'].includes(String(version))
        ? String(version)
        : String(version).startsWith('V') ? String(version) : `V${version}`;
    text(root.querySelector('[data-plan-source]'), sourceLabel);
    text(root.querySelector('[data-plan-version]'), versionLabel);
    text(root.querySelector('[data-plan-target]'), target);
    text(root.querySelector('[data-plan-relation]'), relation);
    text(root.querySelector('[data-plan-source-detail]'), source);
    text(root.querySelector('[data-plan-target-detail]'), target);
    text(root.querySelector('[data-plan-relation-detail]'), relation);
    text(root.querySelector('[data-plan-risk-detail]'), risk);
    root.querySelector('[data-plan-source]')?.setAttribute('title', sourceLabel);
    root.querySelector('[data-plan-target]')?.setAttribute('title', target);
    root.querySelector('[data-plan-relation]')?.setAttribute('title', relation);

    const route = root.querySelector('.agent-visual-plan-route');
    const editStep = job.plan?.steps?.find(step => step.commandType === 'edit_asset');
    const routeLabels = editStep
        ? ['解析对象', '选择编辑能力', editStep.label || '执行编辑']
        : (job.plan?.steps || []).map(step => step.label || step.commandType);
    if (job.verification) routeLabels.push(job.verification.passed ? '验证通过' : '重新验证');
    if (job.status === 'waiting_confirmation') routeLabels.push(job.verification?.passed ? '等待确认' : '等待处理');
    if (job.status === 'completed') routeLabels.push('已提交');
    renderFlow(route, routeLabels);

    const steps = root.querySelector('.agent-visual-plan-steps');
    steps.replaceChildren();
    (job.plan?.steps || []).forEach((step, index) => {
        const done = job.status === 'completed' || index < job.currentStep;
        const active = index === job.currentStep && !done;
        const item = document.createElement('span');
        item.className = done ? 'is-done' : active ? 'is-active' : '';
        const marker = document.createElement('b');
        marker.textContent = done ? '✓' : String(index + 1);
        item.append(marker, document.createTextNode(step.label || step.commandType));
        steps.appendChild(item);
    });
}

function highlightJobAssets(job) {
    document.querySelectorAll('.agent-plan-highlight').forEach(node => node.classList.remove('agent-plan-highlight'));
    const ids = job?.targetAssets || job?.result?.assetIds || [];
    ids.forEach(id => {
        const escaped = typeof CSS !== 'undefined' && typeof CSS.escape === 'function'
            ? CSS.escape(id)
            : String(id).replace(/(["\\])/g, '\\$1');
        const node = document.querySelector(`[data-item-id="${escaped}"]`);
        node?.classList.add('agent-plan-highlight');
    });
}

function clearJobHighlights() {
    document.querySelectorAll('.agent-plan-highlight').forEach(node => node.classList.remove('agent-plan-highlight'));
}

export function initAgentVisualSurface({ runtime, onFallback, onUserMessage, onFinished } = {}) {
    const targetBar = document.getElementById('agentVisualTargetBar');
    const input = document.getElementById('agentVisualTargetInput');
    const submit = document.getElementById('agentVisualTargetSubmit');
    const context = document.getElementById('agentVisualTargetContext');
    const contextLabel = context?.querySelector('[data-target-context-label]');
    const contextId = context?.querySelector('[data-target-context-id]');
    const editControls = targetBar?.querySelector('[data-target-edit-controls]');
    const modeButtons = [...(targetBar?.querySelectorAll('[data-target-mode]') || [])];
    const suggestions = targetBar?.querySelector('[data-target-suggestions]');
    const suggestionsLabel = targetBar?.querySelector('[data-target-suggestions-label]');
    const suggestionButtons = [...(suggestions?.querySelectorAll('[data-object-suggestion]') || [])];
    const plan = createPlanView(document.getElementById('agentVisualPlan'));
    if (!targetBar || !input || !submit || !plan || !contextLabel || !contextId || !editControls) return () => {};

    document.body.classList.add('agent-visual-workspace-mode');

    let editMode = 'variant';
    let lastObjectId = null;
    let visibleSuggestions = [];
    let acceptedSuggestion = null;
    const shownSuggestionKeys = new Set();
    const pendingRecommendationJobs = new Map();

    const suggestionIdFor = value => `target_bar_${String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[^\w\u4e00-\u9fff]+/g, '_')
        .slice(0, 48)}`;

    const recordSuggestionEvent = (type, suggestion, object, extra = {}) => {
        if (!suggestion?.patternId || !object?.itemId) return;
        const workspace = window.mvrRuntime?.getCurrentWorkspace?.();
        const item = object.item || {};
        const asset = object.asset || {};
        resultFeedbackRuntime.recordEvent({
            type,
            assetUid: asset.uid || object.itemId,
            layerId: item.sourceLayerId || item.layerId || undefined,
            sessionId: state.currentSessionId || undefined,
            projectId: workspace?.projectId || undefined,
            sourceType: 'recommendation',
            sourceId: suggestion.patternId,
            context: {
                taskType: suggestion.taskType || 'general',
                semanticType: item.semanticType || asset.semanticType || undefined,
                designRole: item.designRole || asset.designRole || undefined,
                workflowStage: 'agent_visual_target_bar',
                layerName: item.layerName || asset.layerName || item.name || undefined
            },
            metadata: {
                attributionMetadata: {
                    patternId: suggestion.patternId,
                    suggestionText: suggestion.text,
                    surface: 'agent_visual_target_bar',
                    phase: type
                },
                ...extra
            }
        });
    };

    const recordRecommendationOutcome = (jobId, job) => {
        const pending = pendingRecommendationJobs.get(jobId);
        if (!pending || !['completed', 'failed', 'cancelled'].includes(job?.status)) return;
        recordSuggestionEvent(
            job.status === 'completed' ? 'recommendation_executed' : 'recommendation_failed',
            pending.suggestion,
            pending.object,
            { jobId, jobStatus: job.status }
        );
        pendingRecommendationJobs.delete(jobId);
    };

    const fallbackSuggestionsFor = value => {
        const text = String(value || '').toLowerCase();
        if (/(材质|颜色|换成|布料|木纹|金属)/.test(text)) {
            return ['换成米白材质', '换成暖灰材质', '保留构图，提升材质质感'];
        }
        if (/(电商|商品|展示|广告|海报)/.test(text)) {
            return ['生成电商展示版本', '保留构图，提升商品质感', '整理为干净的广告展示图'];
        }
        if (/(光|质感|清晰|高级|氛围)/.test(text)) {
            return ['保留构图，提升质感', '增强自然光和接触阴影', '让材质细节更清晰'];
        }
        return ['换成米白材质', '保留构图，提升质感', '生成电商展示版本'];
    };

    const buildRecommendationAsset = (object, prompt) => {
        const item = object?.item || {};
        const asset = object?.asset || {};
        const parent = item.parentId ? state.workbenchItems.get(item.parentId) : null;
        const layers = parent?.scene?.layers || parent?.layers || item.scene?.layers || item.layers || [];
        const tags = [
            ...(Array.isArray(asset?.metadata?.tags) ? asset.metadata.tags : []),
            ...(Array.isArray(item?.metadata?.tags) ? item.metadata.tags : []),
            item.layerName,
            item.semanticType,
            item.designRole,
            asset.semanticType,
            asset.designRole
        ].filter(Boolean).map(String);

        return {
            uid: asset.uid || object.itemId,
            type: item.type || asset.type || 'image',
            name: item.layerName || item.name || asset.name || asset.label || '',
            label: item.label || asset.label || '',
            layerName: item.layerName || asset.layerName || '',
            semanticType: item.semanticType || asset.semanticType || '',
            designRole: item.designRole || asset.designRole || '',
            prompt: [asset.prompt, item.prompt].filter(Boolean).join(' '),
            fusionProperties: asset.fusionProperties || item.fusionProperties,
            metadata: { tags: [...new Set(tags)].slice(0, 20) },
            layers: Array.isArray(layers)
                ? layers.slice(0, 30).map(layer => ({
                    name: layer?.name,
                    semanticType: layer?.semanticType,
                    designRole: layer?.designRole,
                    reasoning: layer?.reasoning
                }))
                : [],
            currentPrompt: prompt || ''
        };
    };

    const suggestionsFor = value => {
        const object = selectedObject();
        const taskType = inferTaskBucketFromText(value);
        const fallback = fallbackSuggestionsFor(value).map(text => ({
            text,
            patternId: suggestionIdFor(text),
            taskType: taskType || 'general',
            source: 'fallback'
        }));
        if (!object.itemId) return { values: fallback, label: '智能建议 · 根据当前对象和指令' };

        const recommendationAsset = buildRecommendationAsset(object, value);
        const { recommended, others } = SemanticRecommender.recommendWithInsights(
            recommendationAsset,
            { prompt: value, taskType }
        );
        const entries = [...recommended, ...others];
        const learned = entries
            .map(entry => {
                const pattern = entry.pattern || entry;
                // Learned NL prompts are already user-facing commands; static
                // design patterns have concise Chinese names intended for UI.
                const text = String(pattern.id || '').startsWith('dynamic_nl_intent_')
                    ? pattern.intent
                    : pattern.name;
                return {
                    text,
                    patternId: pattern.id || suggestionIdFor(text),
                    taskType: entry.bucket || taskType || 'general',
                    source: 'semantic_recommender'
                };
            })
            .map(entry => ({
                ...entry,
                text: String(entry.text || '').replace(/^[\p{Extended_Pictographic}\s]+/u, '').trim()
            }))
            .filter(entry => entry.text);
        // Historical preferences personalize the first two slots, while the
        // current instruction always contributes at least one immediate,
        // task-specific suggestion.
        const values = [...learned.slice(0, 2), ...fallback]
            .filter((entry, index, list) => list.findIndex(candidate => candidate.text === entry.text) === index)
            .slice(0, 3);
        const taskLabel = taskType ? `${describeTaskBucket(taskType)}任务` : '当前对象';
        return {
            values: values.length > 0 ? values : fallback,
            label: `智能建议 · ${taskLabel}，已结合历史偏好`
        };
    };

    const renderSuggestions = value => {
        const { values, label } = suggestionsFor(value);
        visibleSuggestions = values;
        const object = selectedObject();
        suggestionButtons.forEach((button, index) => {
            const suggestion = values[index];
            if (!suggestion) {
                button.hidden = true;
                return;
            }
            button.hidden = false;
            button.dataset.objectSuggestion = suggestion.text;
            button.dataset.suggestionPatternId = suggestion.patternId;
            button.textContent = suggestion.text;
            const shownKey = `${object.itemId}:${suggestion.patternId}:${suggestion.taskType || 'general'}`;
            if (!shownSuggestionKeys.has(shownKey)) {
                shownSuggestionKeys.add(shownKey);
                recordSuggestionEvent('recommendation_shown', suggestion, object);
            }
        });
        if (suggestionsLabel) {
            suggestionsLabel.textContent = label;
        }
    };

    const openSuggestions = () => {
        if (!suggestions || !selectedObject().itemId) return;
        renderSuggestions(input.value);
        suggestions.hidden = false;
    };

    const selectedObject = () => {
        const { itemId, item, asset } = currentAsset();
        return itemId && isObjectLayer(item) ? { itemId, item, asset } : { itemId: null, item: null, asset: null };
    };
    const canUpdateCurrentLayer = item => {
        const parent = item?.parentId ? state.workbenchItems.get(item.parentId) : null;
        const layers = parent?.scene?.layers || parent?.layers || [];
        const sourceLayerId = item?.sourceLayerId || item?.layerId;
        return Boolean(
            item?.parentId && parent && Array.isArray(item.originalBbox) && item.originalBbox.length === 4 &&
            ((sourceLayerId && layers.some(layer => layer?.id === sourceLayerId)) ||
                (item.layerName && layers.some(layer => layer?.name === item.layerName)) ||
                layers.some(layer => Array.isArray(layer?.bbox) && layer.bbox.length === 4 &&
                    layer.bbox.every((value, index) => Math.abs(Number(value) - Number(item.originalBbox[index])) <= 2)))
        );
    };
    const closeSuggestions = () => {
        if (!suggestions) return;
        suggestions.hidden = true;
    };
    const setEditMode = mode => {
        if (mode === 'current_layer' && !canUpdateCurrentLayer(selectedObject().item)) return;
        editMode = mode;
        modeButtons.forEach(button => button.classList.toggle('is-active', button.dataset.targetMode === mode));
    };

    const updateContext = () => {
        const { item, asset } = currentAsset();
        const object = item && isObjectLayer(item);
        const objectId = object ? currentAsset().itemId : null;
        const name = asset?.layerName || item?.layerName || item?.label || item?.name || (item ? '当前图片' : '未选择对象');
        contextLabel.textContent = object ? `当前图层 · ${name}` : item ? `当前图片 · ${name}` : '当前场景';
        contextId.textContent = object && (asset?.uid || currentAsset().itemId) ? ` · ${asset?.uid || currentAsset().itemId}` : '';
        editControls.hidden = !object;
        submit.querySelector('span').textContent = object ? '生成计划' : '执行';
        input.placeholder = object ? '描述你想如何修改当前图层…' : '描述要添加、提取或修改的内容…';
        const updateButton = targetBar.querySelector('[data-target-mode="current_layer"]');
        if (updateButton) {
            const enabled = object && canUpdateCurrentLayer(item);
            updateButton.disabled = !enabled;
            updateButton.title = enabled ? '' : '只有来自 Magic Layers 的可回写图层支持此操作';
            if (!enabled && editMode === 'current_layer') setEditMode('variant');
        }
        if (!object) {
            lastObjectId = null;
            closeSuggestions();
        } else if (objectId !== lastObjectId) {
            lastObjectId = objectId;
            openSuggestions();
        }
    };
    updateContext();
    const interval = setInterval(updateContext, 700);
    const onSelection = () => {
        // The target bar is the only object-editing surface. Close any legacy
        // drawer that may have been opened by an older capsule/recommender
        // callback before refreshing the selected-object context.
        if (document.getElementById('floatingFusionEditor')) {
            window.hideFloatingFusionEditor?.();
        }
        updateContext();
    };
    window.addEventListener('marmo:workspace-selection-changed', onSelection);

    const agent = runtime?.getAgentRuntime?.();
    const autoOpenedJobs = new Set();
    const shouldOpenChatFor = job => {
        if (!job?.plan) return false;
        const isAttentionState = ['waiting_confirmation', 'failed'].includes(job.status);
        const isComplexRunning = job.plan.type === 'asset_reuse' &&
            ['running', 'verifying', 'committing'].includes(job.status);
        return isAttentionState || isComplexRunning;
    };
    const openChatForJobIfNeeded = job => {
        if (!shouldOpenChatFor(job) || autoOpenedJobs.has(job.id)) return;
        autoOpenedJobs.add(job.id);
        window.setTimeout(() => window.toggleChat?.(true), 0);
    };
    const unsubscribe = agent?.subscribe(event => {
        const job = agent.getJob(event.jobId);
        if (job) {
            updatePlan(plan, job);
            openChatForJobIfNeeded(job);
            if (['completed', 'cancelled'].includes(job.status)) {
                clearJobHighlights();
            } else {
                highlightJobAssets(job);
            }
            recordRecommendationOutcome(event.jobId, job);
            if (['completed', 'cancelled'].includes(job.status)) {
                setTimeout(() => {
                    if (agent.getJob(event.jobId)?.status === job.status) {
                        plan.classList.remove('is-visible');
                        plan.hidden = true;
                    }
                }, 7000);
            }
        }
    });

    const openPlanDetails = event => {
        event.preventDefault();
        event.stopPropagation();
        window.toggleChat?.(true);
        const jobId = plan.dataset.jobId;
        if (jobId) {
            window.setTimeout(() => {
                window.dispatchEvent(new CustomEvent('marmo:agent-task-focus', { detail: { jobId } }));
            }, 80);
        }
    };
    plan.addEventListener('click', openPlanDetails);
    const openPlanWithKeyboard = event => {
        if (event.key === 'Enter' || event.key === ' ') openPlanDetails(event);
    };
    plan.addEventListener('keydown', openPlanWithKeyboard);

    const submitTarget = async () => {
        const value = input.value.trim();
        if (!value || submit.disabled) return;
        submit.disabled = true;
        targetBar.classList.add('is-busy');
        const object = selectedObject();
        const recommendation = acceptedSuggestion?.text === value ? acceptedSuggestion : null;
        let trackedJobId = null;
        try {
            let result;
            if (object.itemId) {
                result = await startObjectEditJob({
                    runtime,
                    itemId: object.itemId,
                    prompt: value,
                    mode: editMode,
                    onUserMessage,
                    onFinished,
                    onJobCreated: job => {
                        trackedJobId = job.id;
                        if (recommendation) {
                            pendingRecommendationJobs.set(job.id, { suggestion: recommendation, object });
                        }
                    }
                });
            } else {
                const args = { text: value, runtime, onUserMessage, onFinished };
                const layerResult = await handleLayerExtractionRequest(args);
                result = layerResult || await handleAssetReuseRequest(args);
            }
            input.value = '';
            closeSuggestions();
            if (recommendation && result?.id) recordRecommendationOutcome(result.id, result);
            acceptedSuggestion = null;
            if (!result && onFallback) await onFallback(value);
        } catch (error) {
            if (recommendation && !trackedJobId) {
                recordSuggestionEvent('recommendation_failed', recommendation, object, {
                    error: error?.message || String(error)
                });
            }
            acceptedSuggestion = null;
            if (onFallback) await onFallback(value, error);
        } finally {
            targetBar.classList.remove('is-busy');
            submit.disabled = false;
        }
    };

    modeButtons.forEach(button => button.addEventListener('click', () => setEditMode(button.dataset.targetMode)));
    suggestionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const suggestion = visibleSuggestions.find(entry => entry.text === button.dataset.objectSuggestion);
            if (!suggestion) return;
            acceptedSuggestion = suggestion;
            recordSuggestionEvent('recommendation_accepted', suggestion, selectedObject());
            input.value = suggestion.text;
            closeSuggestions();
            input.focus();
        });
    });
    document.addEventListener('click', closeSuggestions);
    targetBar.addEventListener('click', event => event.stopPropagation());

    submit.addEventListener('click', submitTarget);
    input.addEventListener('keydown', event => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            submitTarget();
        }
    });
    input.addEventListener('input', () => {
        if (acceptedSuggestion && input.value.trim() !== acceptedSuggestion.text) acceptedSuggestion = null;
        if (selectedObject().itemId) openSuggestions();
    });
    input.addEventListener('focus', () => targetBar.classList.add('is-focused'));
    input.addEventListener('blur', () => targetBar.classList.remove('is-focused'));

    return () => {
        clearInterval(interval);
        window.removeEventListener('marmo:workspace-selection-changed', onSelection);
        document.removeEventListener('click', closeSuggestions);
        plan.removeEventListener('click', openPlanDetails);
        plan.removeEventListener('keydown', openPlanWithKeyboard);
        unsubscribe?.();
        document.body.classList.remove('agent-visual-workspace-mode');
    };
}
