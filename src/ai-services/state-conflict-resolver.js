import {
    PROJECT_STATE_FIELDS,
    PROJECT_STATE_LIST_FIELDS,
    PROJECT_STATE_SCALAR_FIELDS
} from "./session-working-memory.js";

const STATE_TRANSITIONS = new Set(['supplement', 'replace', 'overturn', 'new_project_switch']);
const LIST_FIELD_SET = new Set(PROJECT_STATE_LIST_FIELDS);

function normalizeText(value, maxLength = 240) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (!text) return '';
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function normalizeFieldList(values = []) {
    if (!Array.isArray(values)) return [];
    const result = [];
    const seen = new Set();

    for (const value of values) {
        const text = normalizeText(value, 40);
        if (!text || !PROJECT_STATE_FIELDS.includes(text) || seen.has(text)) continue;
        seen.add(text);
        result.push(text);
    }

    return result;
}

function hasExplicitNewProjectCue(text = '') {
    return /全新的项目|新项目|换个项目|另一个项目|重新开一个项目|这个项目先停|我们做另一个/.test(text);
}

function hasExplicitOverturnCue(text = '') {
    return /推翻之前|全部推翻|之前作废|不要按之前|不再按之前|旧的作废|全部重来|完全重做|整个方向换掉|清空之前|从零开始|之前那套不要了/.test(text);
}

function hasExplicitReplaceCue(text = '') {
    return /按新要求|以这个为准|改成|改为|换成|更新为|改按这个|不要参考上一次|不基于上一次|这次按这个|重新定为|从.*改到|提升到|降低到|改走/.test(text);
}

function hasExplicitSupplementCue(text = '') {
    return /补充|另外|再加|还要|同时|并且|也要|顺便|再补一个|在原来基础上|保留原来/.test(text);
}

function hasAnyPatchValues(projectStatePatch = {}) {
    return Object.values(projectStatePatch || {}).some(value => {
        if (Array.isArray(value)) return value.length > 0;
        return !!normalizeText(value, 120);
    });
}

function detectChangedScalarFields(currentState = {}, patch = {}) {
    const result = [];

    for (const field of PROJECT_STATE_SCALAR_FIELDS) {
        if (!Object.prototype.hasOwnProperty.call(patch, field)) continue;
        const nextValue = normalizeText(patch[field], 160);
        if (!nextValue) continue;
        const currentValue = normalizeText(currentState?.[field], 160);
        if (currentValue && currentValue !== nextValue) {
            result.push(field);
        }
    }

    return result;
}

function detectListFieldsWithContent(projectStatePatch = {}) {
    return PROJECT_STATE_LIST_FIELDS.filter(field => Array.isArray(projectStatePatch?.[field]) && projectStatePatch[field].length > 0);
}

function inferTransition(text = '', interpretation = {}, currentMemory = {}) {
    if (interpretation.stateTransition && STATE_TRANSITIONS.has(interpretation.stateTransition)) {
        return interpretation.stateTransition;
    }

    if (interpretation.shouldResetMemory || interpretation.continuity === 'switching' || hasExplicitNewProjectCue(text)) {
        return 'new_project_switch';
    }

    if (hasExplicitOverturnCue(text)) {
        return 'overturn';
    }

    const scalarChanges = detectChangedScalarFields(currentMemory?.projectState || {}, interpretation.projectStatePatch || {});
    const listFieldsWithContent = detectListFieldsWithContent(interpretation.projectStatePatch || {});
    const hasReplaceFields = Array.isArray(interpretation.replaceProjectStateFields) && interpretation.replaceProjectStateFields.length > 0;

    if (hasExplicitReplaceCue(text) || scalarChanges.length > 0 || hasReplaceFields) {
        return 'replace';
    }

    if (hasExplicitSupplementCue(text) || listFieldsWithContent.length > 0 || hasAnyPatchValues(interpretation.projectStatePatch)) {
        return 'supplement';
    }

    return 'supplement';
}

export function resolveStateConflict(context = {}, interpretation = {}) {
    const text = normalizeText(context.text, 240);
    const currentMemory = context.workingMemory || {};
    const next = {
        ...interpretation,
        projectStatePatch: interpretation.projectStatePatch || {},
        replaceProjectStateFields: normalizeFieldList(interpretation.replaceProjectStateFields || []),
        clearProjectStateFields: normalizeFieldList(interpretation.clearProjectStateFields || []),
        shouldUnlockBrief: !!interpretation.shouldUnlockBrief
    };

    const transition = inferTransition(text, interpretation, currentMemory);
    next.stateTransition = transition;

    if (transition === 'new_project_switch') {
        next.shouldResetMemory = true;
        next.shouldUnlockBrief = true;
        next.clearProjectStateFields = [];
        next.replaceProjectStateFields = normalizeFieldList([
            ...next.replaceProjectStateFields,
            ...PROJECT_STATE_LIST_FIELDS
        ]);
        return next;
    }

    if (transition === 'overturn') {
        const touchedListFields = detectListFieldsWithContent(next.projectStatePatch);
        const touchedScalarFields = detectChangedScalarFields(currentMemory?.projectState || {}, next.projectStatePatch);
        const defaultClearFields = [
            'coreRequirements',
            'mustKeep',
            'avoid',
            'styleAnchors',
            'referenceNotes',
            'currentDeliverable'
        ];

        next.shouldUnlockBrief = true;
        next.clearProjectStateFields = normalizeFieldList([
            ...next.clearProjectStateFields,
            ...defaultClearFields,
            ...touchedScalarFields
        ]);
        next.replaceProjectStateFields = normalizeFieldList([
            ...next.replaceProjectStateFields,
            ...touchedListFields
        ]);
        return next;
    }

    if (transition === 'replace') {
        const scalarChanges = detectChangedScalarFields(currentMemory?.projectState || {}, next.projectStatePatch);
        const listFieldsWithContent = detectListFieldsWithContent(next.projectStatePatch);

        next.replaceProjectStateFields = normalizeFieldList([
            ...next.replaceProjectStateFields,
            ...scalarChanges,
            ...listFieldsWithContent
        ]);
        return next;
    }

    next.replaceProjectStateFields = normalizeFieldList(next.replaceProjectStateFields);
    return next;
}
