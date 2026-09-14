import { Type } from "@google/genai";
import { getTextModel, proxyGenerateContent } from "./gemini-client.js";
import * as prompts from "./prompts.js";
import {
    summarizeWorkingMemory,
    PROJECT_STATE_SCALAR_FIELDS,
    PROJECT_STATE_LIST_FIELDS,
    normalizeProjectStatePatch
} from "./session-working-memory.js";

const VALID_MODES = new Set(['single_turn', 'ongoing_project']);
const VALID_CONTINUITIES = new Set(['fresh', 'continuing', 'switching']);
const VALID_BRIEF_SOURCES = new Set(['none', 'user_request', 'assistant_result']);
const VALID_REFERENCE_POLICIES = new Set(['none', 'supporting_reference', 'authoritative_reference']);
const PROJECT_STATE_FIELD_SET = new Set([
    ...PROJECT_STATE_SCALAR_FIELDS,
    ...PROJECT_STATE_LIST_FIELDS
]);

function clampConfidence(value) {
    const num = Number(value);
    if (Number.isNaN(num)) return 0.5;
    return Math.min(1, Math.max(0, num));
}

function normalizeText(value, maxLength = 240) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (!text) return '';
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function normalizeList(values = [], maxItems = 6, maxLength = 120) {
    const result = [];
    const seen = new Set();
    for (const value of values) {
        const text = normalizeText(value, maxLength);
        if (!text) continue;
        const key = text.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        result.push(text);
        if (result.length >= maxItems) break;
    }
    return result;
}

function normalizeProjectStateFieldReplacements(values = []) {
    if (!Array.isArray(values)) return [];
    const result = [];
    const seen = new Set();

    for (const value of values) {
        const text = normalizeText(value, 40);
        if (!text || !PROJECT_STATE_FIELD_SET.has(text)) continue;
        if (seen.has(text)) continue;
        seen.add(text);
        result.push(text);
    }

    return result;
}

function buildHistoryTranscript(history = [], maxMessages = 8) {
    return history
        .filter(msg => msg && msg.content && (msg.type === 'text' || msg.type === 'bot-rich'))
        .slice(-maxMessages)
        .map(msg => `${msg.sender === 'user' ? '用户' : '助手'}：${normalizeText(msg.content, 220)}`)
        .join('\n');
}

function fallbackInterpretation(context = {}, errorMessage = '') {
    const text = normalizeText(context.text, 240);
    const hasBaseImage = !!context.baseImage;
    const existingMemory = context.workingMemory;

    const isNewProject = /全新的项目|新项目|换个项目|另一个项目|重新开一个项目/.test(text);
    const wantsPersistentBrief = /后面.*遵循|后续.*遵循|作为.*总纲|作为.*基准|后面.*都按这个|之后.*都按这个|统一按照/.test(text);
    const wantsSummary = /总结|整理|提炼|梳理|归纳|转成文本/.test(text);

    return {
        mode: isNewProject || wantsPersistentBrief ? 'ongoing_project' : (existingMemory?.mode || 'single_turn'),
        userIntent: hasBaseImage && wantsSummary ? 'analyze_reference' : 'discuss',
        continuity: isNewProject ? 'switching' : 'continuing',
        shouldResetMemory: isNewProject,
        shouldPersistBrief: wantsPersistentBrief,
        briefSource: wantsPersistentBrief && hasBaseImage && wantsSummary ? 'assistant_result' : (wantsPersistentBrief ? 'user_request' : 'none'),
        canonicalGoal: text,
        referencePolicy: wantsPersistentBrief ? 'authoritative_reference' : (hasBaseImage ? 'supporting_reference' : 'none'),
        projectStatePatch: {
            primaryGoal: text
        },
        replaceProjectStateFields: isNewProject ? ['primaryGoal', 'coreRequirements', 'mustKeep', 'avoid', 'styleAnchors', 'referenceNotes'] : [],
        reason: `解释器不可用，使用保守回退。${errorMessage}`.trim(),
        confidence: 0.4
    };
}

function normalizeInterpretation(raw, context = {}, source = 'llm') {
    const normalized = {
        mode: VALID_MODES.has(raw?.mode) ? raw.mode : 'single_turn',
        userIntent: normalizeText(raw?.userIntent, 80) || 'discuss',
        continuity: VALID_CONTINUITIES.has(raw?.continuity) ? raw.continuity : 'continuing',
        shouldResetMemory: !!raw?.shouldResetMemory,
        shouldPersistBrief: !!raw?.shouldPersistBrief,
        briefSource: VALID_BRIEF_SOURCES.has(raw?.briefSource) ? raw.briefSource : 'none',
        canonicalGoal: normalizeText(raw?.canonicalGoal, 420),
        referencePolicy: VALID_REFERENCE_POLICIES.has(raw?.referencePolicy) ? raw.referencePolicy : 'none',
        projectStatePatch: normalizeProjectStatePatch(raw?.projectStatePatch || {}),
        replaceProjectStateFields: normalizeProjectStateFieldReplacements(raw?.replaceProjectStateFields || []),
        reason: normalizeText(raw?.reason, 240),
        confidence: clampConfidence(raw?.confidence),
        source
    };

    if (!normalized.reason) {
        normalized.reason = '解释器已更新当前会话任务理解。';
    }

    return normalized;
}

function buildInterpreterPrompt(context = {}) {
    const workingMemorySummary = summarizeWorkingMemory(context.workingMemory, {
        prefix: '已有会话工作记忆'
    });

    const payload = {
        user_text: normalizeText(context.text, 240),
        has_base_image: !!context.baseImage,
        reference_image_count: Array.isArray(context.referenceImages) ? context.referenceImages.length : 0,
        has_mask: !!context.mask,
        workspace_context: context.workspaceContext || null,
        existing_working_memory: workingMemorySummary,
        existing_project_state: context.workingMemory?.projectState || null,
        locked_brief: context.workingMemory?.briefLocked ? normalizeText(context.workingMemory?.canonicalBrief, 320) : '',
        recent_history: buildHistoryTranscript(context.history || [])
    };

    return [
        '请阅读当前用户输入、最近上下文与已有会话工作记忆。',
        'workspace_context 是 Workbench 的内部只读状态，不是聊天消息；请用它理解当前图片、图层和最近工作台动作，但不要把它当成用户刚刚说过的话。',
        '你的任务不是选择模型路由，而是判断当前会话的“任务状态”是否需要更新。',
        '尤其要识别：是否进入长期项目、是否切换任务、是否出现应长期遵守的约束、是否应把本轮或下一轮产出的文本升级为项目总纲。',
        '上下文 JSON：',
        JSON.stringify(payload, null, 2)
    ].join('\n');
}

export async function interpretConversationTurn(context = {}) {
    if (context.skipModel === true) {
        return fallbackInterpretation(context, '当前为 Qwen 图片任务本地直通，跳过会话解释器模型。');
    }

    try {
        const response = await proxyGenerateContent({
            model: getTextModel(),
            systemInstruction: {
                parts: [{ text: prompts.getConversationInterpreterSystemInstruction() }]
            },
            contents: [{ role: 'user', parts: [{ text: buildInterpreterPrompt(context) }] }],
            config: {
                responseMimeType: 'application/json',
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        mode: { type: Type.STRING },
                        userIntent: { type: Type.STRING },
                        continuity: { type: Type.STRING },
                        shouldResetMemory: { type: Type.BOOLEAN },
                        shouldPersistBrief: { type: Type.BOOLEAN },
                        briefSource: { type: Type.STRING },
                        canonicalGoal: { type: Type.STRING },
                        projectStatePatch: {
                            type: Type.OBJECT,
                            properties: {
                                projectType: { type: Type.STRING },
                                subject: { type: Type.STRING },
                                primaryGoal: { type: Type.STRING },
                                currentDeliverable: { type: Type.STRING },
                                targetAudience: { type: Type.STRING },
                                timeframe: { type: Type.STRING },
                                budget: { type: Type.STRING },
                                coreRequirements: {
                                    type: Type.ARRAY,
                                    items: { type: Type.STRING }
                                },
                                mustKeep: {
                                    type: Type.ARRAY,
                                    items: { type: Type.STRING }
                                },
                                avoid: {
                                    type: Type.ARRAY,
                                    items: { type: Type.STRING }
                                },
                                styleAnchors: {
                                    type: Type.ARRAY,
                                    items: { type: Type.STRING }
                                },
                                referenceNotes: {
                                    type: Type.ARRAY,
                                    items: { type: Type.STRING }
                                }
                            }
                        },
                        replaceProjectStateFields: {
                            type: Type.ARRAY,
                            items: { type: Type.STRING }
                        },
                        referencePolicy: { type: Type.STRING },
                        reason: { type: Type.STRING },
                        confidence: { type: Type.NUMBER }
                    },
                    required: ['mode', 'userIntent', 'continuity', 'shouldResetMemory', 'shouldPersistBrief', 'briefSource', 'canonicalGoal', 'projectStatePatch', 'replaceProjectStateFields', 'referencePolicy', 'reason', 'confidence']
                }
            }
        });

        const text = response?.text?.trim?.() || '';
        if (!text) {
            return fallbackInterpretation(context, '解释器返回空内容。');
        }

        const cleaned = text.replace(/```json|```/g, '').trim();
        const parsed = JSON.parse(cleaned);
        return normalizeInterpretation(parsed, context, 'llm_interpreter');
    } catch (error) {
        return fallbackInterpretation(context, `解释失败: ${error.message}`);
    }
}
