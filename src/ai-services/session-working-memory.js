const MAX_CONSTRAINTS = 6;
const MAX_STYLE_ANCHORS = 4;
const MAX_CONTINUITY_ITEMS = 5;
const MAX_TEXT_LENGTH = 600;

export const PROJECT_STATE_SCALAR_FIELDS = [
    'projectType',
    'subject',
    'primaryGoal',
    'currentDeliverable',
    'targetAudience',
    'timeframe',
    'budget'
];

export const PROJECT_STATE_LIST_FIELDS = [
    'coreRequirements',
    'mustKeep',
    'avoid',
    'styleAnchors',
    'referenceNotes'
];

export const PROJECT_STATE_FIELDS = [
    ...PROJECT_STATE_SCALAR_FIELDS,
    ...PROJECT_STATE_LIST_FIELDS
];

const PROJECT_STATE_FIELD_SET = new Set(PROJECT_STATE_FIELDS);

function normalizeText(value, maxLength = MAX_TEXT_LENGTH) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (!text) return '';
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function dedupeTextList(values = [], maxItems = MAX_CONSTRAINTS, maxLength = 160) {
    const normalized = [];
    const seen = new Set();

    for (const value of values) {
        const text = normalizeText(value, maxLength);
        if (!text) continue;
        const key = text.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        normalized.push(text);
        if (normalized.length >= maxItems) break;
    }

    return normalized;
}

function normalizeProjectStateList(field, values = []) {
    if (field === 'styleAnchors') {
        return dedupeTextList(values, MAX_STYLE_ANCHORS, 100);
    }

    if (field === 'referenceNotes') {
        return dedupeTextList(values, 4, 120);
    }

    return dedupeTextList(values, MAX_CONSTRAINTS, 140);
}

function normalizeContinuityStateList(values = [], maxItems = MAX_CONTINUITY_ITEMS, maxLength = 140) {
    return dedupeTextList(values, maxItems, maxLength);
}

export function createDefaultContinuityState() {
    return {
        anchorSummary: '',
        preserveElements: [],
        variableElements: [],
        usageNotes: [],
        anchorSource: 'none',
        updatedAt: 0
    };
}

export function normalizeContinuityState(continuityState = {}) {
    return {
        anchorSummary: normalizeText(continuityState?.anchorSummary, 320),
        preserveElements: normalizeContinuityStateList(continuityState?.preserveElements || [], MAX_CONTINUITY_ITEMS, 120),
        variableElements: normalizeContinuityStateList(continuityState?.variableElements || [], MAX_CONTINUITY_ITEMS, 120),
        usageNotes: normalizeContinuityStateList(continuityState?.usageNotes || [], 4, 140),
        anchorSource: normalizeText(continuityState?.anchorSource, 40) || 'none',
        updatedAt: Number(continuityState?.updatedAt) || 0
    };
}

export function hasUsefulContinuityState(continuityState) {
    if (!continuityState) return false;

    return !!(
        normalizeText(continuityState.anchorSummary, 320)
        || (Array.isArray(continuityState.preserveElements) && continuityState.preserveElements.length > 0)
        || (Array.isArray(continuityState.variableElements) && continuityState.variableElements.length > 0)
        || (Array.isArray(continuityState.usageNotes) && continuityState.usageNotes.length > 0)
    );
}

function summarizeContinuityState(continuityState) {
    const normalized = normalizeContinuityState(continuityState);
    if (!hasUsefulContinuityState(normalized)) return [];

    const lines = ['系列连续性锚点：'];
    if (normalized.anchorSummary) lines.push(`系列视觉系统：${normalized.anchorSummary}`);
    if (normalized.preserveElements.length > 0) lines.push(`必须继承：${normalized.preserveElements.join('；')}`);
    if (normalized.variableElements.length > 0) lines.push(`允许变化：${normalized.variableElements.join('；')}`);
    if (normalized.usageNotes.length > 0) lines.push(`执行提示：${normalized.usageNotes.join('；')}`);
    return lines;
}

function mergeContinuityState(currentState, patch = {}, options = {}) {
    const replace = options.replace !== false;
    const current = normalizeContinuityState(currentState || {});
    const normalizedPatch = normalizeContinuityState({
        ...patch,
        anchorSource: patch?.anchorSource || current.anchorSource || 'none',
        updatedAt: patch?.updatedAt || Date.now()
    });

    if (replace) {
        return normalizeContinuityState({
            ...createDefaultContinuityState(),
            ...normalizedPatch
        });
    }

    return normalizeContinuityState({
        anchorSummary: normalizedPatch.anchorSummary || current.anchorSummary,
        preserveElements: normalizedPatch.preserveElements.length > 0
            ? normalizedPatch.preserveElements
            : current.preserveElements,
        variableElements: normalizedPatch.variableElements.length > 0
            ? normalizedPatch.variableElements
            : current.variableElements,
        usageNotes: normalizedPatch.usageNotes.length > 0
            ? normalizedPatch.usageNotes
            : current.usageNotes,
        anchorSource: normalizedPatch.anchorSource || current.anchorSource || 'none',
        updatedAt: normalizedPatch.updatedAt || current.updatedAt || Date.now()
    });
}

export function createDefaultProjectState() {
    return {
        projectType: '',
        subject: '',
        primaryGoal: '',
        currentDeliverable: '',
        targetAudience: '',
        timeframe: '',
        budget: '',
        coreRequirements: [],
        mustKeep: [],
        avoid: [],
        styleAnchors: [],
        referenceNotes: []
    };
}

export function normalizeProjectState(projectState = {}) {
    const normalized = createDefaultProjectState();

    for (const field of PROJECT_STATE_SCALAR_FIELDS) {
        normalized[field] = normalizeText(projectState?.[field], field === 'primaryGoal' ? 240 : 160);
    }

    for (const field of PROJECT_STATE_LIST_FIELDS) {
        normalized[field] = normalizeProjectStateList(field, projectState?.[field] || []);
    }

    return normalized;
}

export function normalizeProjectStatePatch(projectStatePatch = {}) {
    const patch = {};

    if (!projectStatePatch || typeof projectStatePatch !== 'object') {
        return patch;
    }

    for (const field of PROJECT_STATE_SCALAR_FIELDS) {
        if (!Object.prototype.hasOwnProperty.call(projectStatePatch, field)) continue;
        patch[field] = normalizeText(projectStatePatch[field], field === 'primaryGoal' ? 240 : 160);
    }

    for (const field of PROJECT_STATE_LIST_FIELDS) {
        if (!Object.prototype.hasOwnProperty.call(projectStatePatch, field)) continue;
        patch[field] = normalizeProjectStateList(field, projectStatePatch[field] || []);
    }

    return patch;
}

function buildLegacyProjectState(memory = {}) {
    return normalizeProjectState({
        primaryGoal: memory.activeTask || '',
        coreRequirements: memory.constraints || [],
        styleAnchors: memory.styleAnchors || []
    });
}

function syncLegacyFields(memory) {
    const projectState = normalizeProjectState(memory.projectState || {});
    memory.projectState = projectState;
    memory.activeTask = projectState.currentDeliverable || projectState.primaryGoal || normalizeText(memory.activeTask, 240);
    memory.constraints = dedupeTextList([
        ...(projectState.coreRequirements || []),
        ...(projectState.mustKeep || [])
    ], MAX_CONSTRAINTS, 140);
    memory.styleAnchors = normalizeProjectStateList('styleAnchors', projectState.styleAnchors || []);
    memory.referencePolicy = normalizeText(memory.referencePolicy, 80) || 'none';
    return memory;
}

function hasUsefulProjectState(projectState) {
    if (!projectState) return false;

    for (const field of PROJECT_STATE_SCALAR_FIELDS) {
        if (normalizeText(projectState[field], 240)) return true;
    }

    for (const field of PROJECT_STATE_LIST_FIELDS) {
        if (Array.isArray(projectState[field]) && projectState[field].length > 0) return true;
    }

    return false;
}

function summarizeProjectState(projectState) {
    if (!hasUsefulProjectState(projectState)) return [];

    const lines = [];

    if (projectState.projectType) lines.push(`项目类型：${normalizeText(projectState.projectType, 80)}`);
    if (projectState.subject) lines.push(`项目主题：${normalizeText(projectState.subject, 120)}`);
    if (projectState.primaryGoal) lines.push(`项目目标：${normalizeText(projectState.primaryGoal, 220)}`);
    if (projectState.currentDeliverable) lines.push(`当前交付：${normalizeText(projectState.currentDeliverable, 180)}`);
    if (projectState.targetAudience) lines.push(`目标对象：${normalizeText(projectState.targetAudience, 120)}`);
    if (projectState.timeframe) lines.push(`项目周期：${normalizeText(projectState.timeframe, 80)}`);
    if (projectState.budget) lines.push(`预算/资源：${normalizeText(projectState.budget, 80)}`);
    if (projectState.coreRequirements.length > 0) lines.push(`核心要求：${projectState.coreRequirements.join('；')}`);
    if (projectState.mustKeep.length > 0) lines.push(`必须保留：${projectState.mustKeep.join('；')}`);
    if (projectState.avoid.length > 0) lines.push(`避免事项：${projectState.avoid.join('；')}`);
    if (projectState.styleAnchors.length > 0) lines.push(`风格锚点：${projectState.styleAnchors.join('；')}`);
    if (projectState.referenceNotes.length > 0) lines.push(`参考备注：${projectState.referenceNotes.join('；')}`);

    return lines;
}

function applyProjectStatePatch(currentState, rawPatch, replaceFields = [], clearFields = []) {
    const nextState = normalizeProjectState(currentState || {});
    const patch = normalizeProjectStatePatch(rawPatch);
    const replaceSet = new Set(
        (Array.isArray(replaceFields) ? replaceFields : []).filter(field => PROJECT_STATE_FIELD_SET.has(field))
    );
    const clearSet = new Set(
        (Array.isArray(clearFields) ? clearFields : []).filter(field => PROJECT_STATE_FIELD_SET.has(field))
    );

    for (const field of clearSet) {
        nextState[field] = PROJECT_STATE_FIELD_SET.has(field) && PROJECT_STATE_LIST_FIELDS.includes(field) ? [] : '';
    }

    for (const field of PROJECT_STATE_SCALAR_FIELDS) {
        if (!Object.prototype.hasOwnProperty.call(patch, field)) continue;
        nextState[field] = patch[field];
    }

    for (const field of PROJECT_STATE_LIST_FIELDS) {
        if (!Object.prototype.hasOwnProperty.call(patch, field)) continue;
        nextState[field] = replaceSet.has(field)
            ? normalizeProjectStateList(field, patch[field])
            : normalizeProjectStateList(field, [
                ...(nextState[field] || []),
                ...(patch[field] || [])
            ]);
    }

    return normalizeProjectState(nextState);
}

export function createDefaultWorkingMemory() {
    return {
        version: 3,
        mode: 'single_turn',
        continuity: 'fresh',
        activeTask: '',
        canonicalBrief: '',
        briefLocked: false,
        briefSource: 'none',
        constraints: [],
        styleAnchors: [],
        referencePolicy: 'none',
        projectState: createDefaultProjectState(),
        continuityState: createDefaultContinuityState(),
        lastInterpreterReason: '',
        lastUserIntent: '',
        pendingBriefCommit: null,
        updatedAt: 0
    };
}

export function ensureSessionWorkingMemory(session) {
    if (!session) return createDefaultWorkingMemory();

    if (!session.workingMemory || typeof session.workingMemory !== 'object') {
        session.workingMemory = createDefaultWorkingMemory();
    } else {
        const mergedMemory = {
            ...createDefaultWorkingMemory(),
            ...session.workingMemory
        };

        const legacyProjectState = buildLegacyProjectState(mergedMemory);
        mergedMemory.projectState = normalizeProjectState(
            hasUsefulProjectState(mergedMemory.projectState) ? mergedMemory.projectState : legacyProjectState
        );
        mergedMemory.continuityState = normalizeContinuityState(mergedMemory.continuityState || {});
        mergedMemory.canonicalBrief = normalizeText(mergedMemory.canonicalBrief, MAX_TEXT_LENGTH);
        mergedMemory.lastInterpreterReason = normalizeText(mergedMemory.lastInterpreterReason, 240);
        mergedMemory.lastUserIntent = normalizeText(mergedMemory.lastUserIntent, 80);
        session.workingMemory = syncLegacyFields(mergedMemory);
    }

    return session.workingMemory;
}

export function hasUsefulWorkingMemory(memory) {
    if (!memory) return false;

    return !!(
        normalizeText(memory.canonicalBrief) ||
        hasUsefulContinuityState(memory.continuityState) ||
        hasUsefulProjectState(memory.projectState) ||
        normalizeText(memory.activeTask, 240) ||
        (Array.isArray(memory.constraints) && memory.constraints.length > 0) ||
        (Array.isArray(memory.styleAnchors) && memory.styleAnchors.length > 0)
    );
}

export function summarizeWorkingMemory(memory, options = {}) {
    if (!hasUsefulWorkingMemory(memory)) return '';

    const prefix = options.prefix || '当前会话工作记忆';
    const lines = [prefix];

    if (memory.briefLocked && memory.canonicalBrief) {
        lines.push(`已锁定总纲：${normalizeText(memory.canonicalBrief, 420)}`);
    }

    const continuityLines = summarizeContinuityState(memory.continuityState || {});
    if (continuityLines.length > 0) {
        lines.push(...continuityLines);
    }

    const projectStateLines = summarizeProjectState(normalizeProjectState(memory.projectState || {}));
    if (projectStateLines.length > 0) {
        lines.push(...projectStateLines);
    } else if (memory.activeTask) {
        lines.push(`当前主任务：${normalizeText(memory.activeTask, 240)}`);
    }

    if (memory.referencePolicy && memory.referencePolicy !== 'none') {
        lines.push(`参考策略：${memory.referencePolicy}`);
    }

    return lines.join('\n');
}

export function applyWorkingMemoryToPrompt(prompt, memory, options = {}) {
    const normalizedPrompt = String(prompt || '').trim();
    const summary = summarizeWorkingMemory(memory, {
        prefix: options.prefix || '当前会话工作记忆（高优先级）'
    });

    if (!summary) return normalizedPrompt;
    if (normalizedPrompt.includes('当前会话工作记忆（高优先级）')) return normalizedPrompt;

    return [
        summary,
        options.instruction || '除非用户明确表示切换任务、推翻既有基线或要求重置方向，否则请优先遵守以上内容。',
        normalizedPrompt ? `当前请求：${normalizedPrompt}` : ''
    ].filter(Boolean).join('\n\n');
}

export function mergeInterpreterResultIntoSession(session, interpretation) {
    const memory = ensureSessionWorkingMemory(session);
    if (!interpretation || typeof interpretation !== 'object') return memory;

    if (interpretation.shouldResetMemory) {
        session.workingMemory = createDefaultWorkingMemory();
    }

    const nextMemory = ensureSessionWorkingMemory(session);
    nextMemory.mode = interpretation.mode || nextMemory.mode;
    nextMemory.continuity = interpretation.continuity || nextMemory.continuity;
    nextMemory.referencePolicy = interpretation.referencePolicy || nextMemory.referencePolicy;
    nextMemory.lastInterpreterReason = normalizeText(interpretation.reason, 240);
    nextMemory.lastUserIntent = normalizeText(interpretation.userIntent, 80);
    nextMemory.updatedAt = Date.now();

    const canonicalGoal = normalizeText(interpretation.canonicalGoal, 420);
    if (canonicalGoal) {
        nextMemory.activeTask = canonicalGoal;
    }

    nextMemory.projectState = applyProjectStatePatch(
        nextMemory.projectState,
        interpretation.projectStatePatch,
        interpretation.replaceProjectStateFields,
        interpretation.clearProjectStateFields
    );

    if (canonicalGoal && !nextMemory.projectState.primaryGoal) {
        nextMemory.projectState.primaryGoal = normalizeText(canonicalGoal, 240);
    }

    if (interpretation.shouldPersistBrief) {
        if (interpretation.briefSource === 'user_request' && canonicalGoal) {
            nextMemory.canonicalBrief = canonicalGoal;
            nextMemory.briefLocked = true;
            nextMemory.briefSource = 'user_request';
            nextMemory.pendingBriefCommit = null;
        } else if (interpretation.briefSource === 'assistant_result') {
            nextMemory.pendingBriefCommit = {
                source: 'assistant_result',
                requestedGoal: canonicalGoal || nextMemory.activeTask || '',
                createdAt: Date.now()
            };
        }
    }

    if (interpretation.shouldUnlockBrief) {
        nextMemory.canonicalBrief = '';
        nextMemory.briefLocked = false;
        nextMemory.briefSource = 'none';
        nextMemory.pendingBriefCommit = null;
    }

    session.workingMemory = syncLegacyFields(nextMemory);
    return session.workingMemory;
}

export function setContinuityAnchor(session, continuityPatch = {}, options = {}) {
    const memory = ensureSessionWorkingMemory(session);
    memory.continuityState = mergeContinuityState(memory.continuityState, continuityPatch, options);
    memory.updatedAt = Date.now();
    session.workingMemory = syncLegacyFields(memory);
    return session.workingMemory;
}

export function finalizeWorkingMemoryWithAssistantResult(session, interpretation, assistantText) {
    const memory = ensureSessionWorkingMemory(session);
    const resultText = normalizeText(assistantText, 900);
    if (!resultText) return memory;

    if (memory.pendingBriefCommit?.source === 'assistant_result') {
        memory.canonicalBrief = resultText;
        memory.briefLocked = true;
        memory.briefSource = 'assistant_result';
        memory.pendingBriefCommit = null;
        memory.updatedAt = Date.now();
    }

    if (!memory.projectState.primaryGoal && interpretation?.canonicalGoal) {
        memory.projectState.primaryGoal = normalizeText(interpretation.canonicalGoal, 240);
    }

    session.workingMemory = syncLegacyFields(memory);
    return session.workingMemory;
}
