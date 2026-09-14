import { describe, it, expect } from 'vitest';
import { applyConversationHardGuards } from '../ai-services/hard-guards.js';
import { fallbackIntentDecision, normalizeIntentDecision } from '../ai-services/llm-intent-classifier.js';
import { composeReferenceAwareGenerationPrompt } from '../ai-services/skills-engine.js';
import { resolveExecutionPrompt, composeExecutionPrompt, classifyRegenerateInstruction, mergeRegeneratePrompt } from '../ai-services/intent-executor.js';
import { resolveStateConflict } from '../ai-services/state-conflict-resolver.js';
import {
    createDefaultWorkingMemory,
    ensureSessionWorkingMemory,
    mergeInterpreterResultIntoSession,
    finalizeWorkingMemoryWithAssistantResult,
    applyWorkingMemoryToPrompt,
    setContinuityAnchor
} from '../ai-services/session-working-memory.js';

describe('conversation routing architecture', () => {
    it('uses hard guards only for high-risk image contexts', () => {
        expect(applyConversationHardGuards({
            text: '继续',
            baseImage: { name: 'base.png' },
            mask: 'mask-data'
        })?.route).toBe('image_edit');

        expect(applyConversationHardGuards({
            text: '',
            baseImage: { name: 'base.png' }
        })?.route).toBe('image_query');

        expect(applyConversationHardGuards({
            text: '帮我总结一下',
            baseImage: null
        })).toBeNull();
    });

    it('continues image production tasks when the user gives short execution follow-ups', () => {
        expect(applyConversationHardGuards({
            text: '按这个改，开始吧',
            baseImage: { name: 'cover.png' },
            history: [
                { sender: 'user', type: 'text', content: '把第一关封面重新做一版，风格跟总图统一' },
                { sender: 'bot', type: 'bot-rich', content: '我会按这个方向重做封面视觉。' }
            ]
        })?.route).toBe('image_edit');

        expect(applyConversationHardGuards({
            text: '重新生成一版',
            baseImage: null,
            history: [
                { sender: 'user', type: 'text', content: '给我做一个暑期课程封面' }
            ]
        })?.route).toBe('image_generation');
    });

    it('does not force edit when the user wants a new image derived from a reference image', () => {
        expect(applyConversationHardGuards({
            text: '参考这张图继续第三部分，生成一张新的页面，不要复制上一张构图',
            baseImage: { name: 'chapter-2.png' },
            history: [
                { sender: 'user', type: 'text', content: '先做第二部分的页面' },
                { sender: 'bot', type: 'bot-rich', content: '我先延续同系列风格完成第二部分。' }
            ]
        })?.route).toBe('image_generation');
    });

    it('keeps forcing edit only for explicit same-image modification requests', () => {
        expect(applyConversationHardGuards({
            text: '在这张图基础上继续修改，把标题区换成绿色',
            baseImage: { name: 'chapter-2.png' }
        })?.route).toBe('image_edit');
    });

    it('does not force execution for discussion-only continuations', () => {
        expect(applyConversationHardGuards({
            text: '继续',
            baseImage: null,
            history: [
                { sender: 'user', type: 'text', content: '帮我总结这张图的结构' },
                { sender: 'bot', type: 'bot-rich', content: '我先把结构分成三部分。' }
            ]
        })).toBeNull();
    });

    it('expands short execution commands into contextual image prompts', () => {
        const prompt = resolveExecutionPrompt(
            { route: 'image_generation' },
            {
                prompt: '生成吧，开始',
                history: [
                    { sender: 'user', type: 'text', content: '保持原汁原味的风格，确保第一关小火车村的封面和总图视觉完全一致，并把关卡介绍和任务总览合理排版进去。' },
                    { sender: 'bot', type: 'bot-rich', content: '我会保持总图风格一致，重做第一关封面，并把关卡介绍和任务总览排版进去。' },
                    { sender: 'user', type: 'text', content: '生成吧，开始' }
                ]
            }
        );

        expect(prompt).toContain('第一关小火车村');
        expect(prompt).toContain('总图视觉完全一致');
        expect(prompt).toContain('当前执行口令');
    });

    it('keeps normal detailed prompts unchanged', () => {
        const prompt = resolveExecutionPrompt(
            { route: 'image_generation' },
            {
                prompt: '生成一张夏令营课程封面，扁平插画风格，主标题醒目',
                history: []
            }
        );

        expect(prompt).toBe('生成一张夏令营课程封面，扁平插画风格，主标题醒目');
    });

    it('injects visual reference constraints into generation prompts when available', () => {
        const prompt = composeReferenceAwareGenerationPrompt(
            '继续第三关的图片生成',
            '保持同系列关卡插画语言\n延续第二关的版式层级\n不要直接复制上一张构图'
        );

        expect(prompt).toContain('视觉参考约束');
        expect(prompt).toContain('保持同系列关卡插画语言');
        expect(prompt).toContain('继续第三关的图片生成');
    });

    it('stores a selected image as a continuity anchor for later series tasks', () => {
        const session = { id: 's-anchor', messages: [], workingMemory: createDefaultWorkingMemory() };

        setContinuityAnchor(session, {
            anchorSummary: '儿童课程关卡系列，保持同一套扁平插画、信息卡片和角色比例系统。',
            preserveElements: ['角色比例统一', '卡片式关卡信息模块', '明亮扁平插画质感'],
            variableElements: ['每一关可更换主场景', '局部主色可随关卡主题调整'],
            usageNotes: ['新页面不要照搬上一张构图']
        });

        expect(session.workingMemory.continuityState.anchorSummary).toContain('儿童课程关卡系列');
        expect(session.workingMemory.continuityState.preserveElements).toContain('角色比例统一');
        expect(session.workingMemory.continuityState.variableElements).toContain('每一关可更换主场景');
    });

    it('builds continuity-aware prompts for short series continuation commands', () => {
        const session = { id: 's-continuity', messages: [], workingMemory: createDefaultWorkingMemory() };
        setContinuityAnchor(session, {
            anchorSummary: '同一套儿童关卡课程视觉系统，强调统一角色、版式和趣味信息层级。',
            preserveElements: ['统一角色设定', '一致的信息卡片结构', '同系列插画语言'],
            variableElements: ['每一关可以切换新场景', '局部色彩随当前主题变化'],
            usageNotes: ['不要复制上一张的具体构图']
        });

        const prompt = composeExecutionPrompt(
            { route: 'image_edit' },
            {
                prompt: '继续第三关',
                baseImage: { name: 'level-2.png' },
                history: [
                    { sender: 'user', type: 'text', content: '第二关已经确认，保留这个系列系统继续往下做。' }
                ],
                workingMemory: session.workingMemory
            }
        );

        expect(prompt).toContain('系列连续性执行原则');
        expect(prompt).toContain('必须继承：统一角色设定');
        expect(prompt).toContain('本轮允许变化：每一关可以切换新场景');
        expect(prompt).toContain('不要机械复制上一张图的具体构图');
        expect(prompt).toContain('当前执行任务');
    });

    it('treats plain regenerate as full prompt reuse', () => {
        const strategy = classifyRegenerateInstruction('重新生成');
        expect(strategy.mode).toBe('reuse');
        expect(mergeRegeneratePrompt('原始提示词', '重新生成')).toBe('原始提示词');
    });

    it('treats regenerate with added requirements as explicit merge', () => {
        const strategy = classifyRegenerateInstruction('重新生成，并且把标题做得更醒目');
        expect(strategy.mode).toBe('merge');
        expect(mergeRegeneratePrompt('原始提示词', '重新生成，并且把标题做得更醒目')).toContain('原始提示词');
        expect(mergeRegeneratePrompt('原始提示词', '重新生成，并且把标题做得更醒目')).toContain('把标题做得更醒目');
    });

    it('treats regenerate by new requirements as explicit override', () => {
        const strategy = classifyRegenerateInstruction('按新要求重新生成：做成蓝绿色科技风，去掉插画感');
        expect(strategy.mode).toBe('override');
        expect(mergeRegeneratePrompt('原始提示词', '按新要求重新生成：做成蓝绿色科技风，去掉插画感')).toBe('做成蓝绿色科技风，去掉插画感');
    });

    it('stores locked working memory and injects it into later prompts', () => {
        const session = { id: 's1', messages: [], workingMemory: createDefaultWorkingMemory() };
        mergeInterpreterResultIntoSession(session, {
            mode: 'ongoing_project',
            userIntent: 'plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: true,
            briefSource: 'user_request',
            canonicalGoal: '暑期60天幼小衔接课程项目，后续所有内容都要遵循课程结构总纲。',
            referencePolicy: 'authoritative_reference',
            projectStatePatch: {
                subject: '暑期60天幼小衔接课程',
                timeframe: '暑期60天',
                primaryGoal: '后续所有内容都要遵循课程结构总纲',
                coreRequirements: ['保持课程结构一致', '后续生图与讨论都基于同一能力树'],
                styleAnchors: ['原汁原味', '结构统一']
            },
            replaceProjectStateFields: ['coreRequirements', 'styleAnchors'],
            reason: '用户要求将当前内容作为后续统一基线。',
            confidence: 0.95
        });

        const prompt = applyWorkingMemoryToPrompt('继续生成第一关封面', session.workingMemory);
        expect(prompt).toContain('已锁定总纲');
        expect(prompt).toContain('课程结构总纲');
        expect(prompt).toContain('项目周期：暑期60天');
        expect(prompt).toContain('核心要求：保持课程结构一致');
        expect(prompt).toContain('继续生成第一关封面');
    });

    it('commits assistant summaries into locked briefs when requested', () => {
        const session = { id: 's2', messages: [] };
        ensureSessionWorkingMemory(session);

        mergeInterpreterResultIntoSession(session, {
            mode: 'ongoing_project',
            userIntent: 'analyze_reference',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: true,
            briefSource: 'assistant_result',
            canonicalGoal: '先从课程结构图提炼一份后续统一遵循的文本总纲。',
            referencePolicy: 'authoritative_reference',
            projectStatePatch: {
                currentDeliverable: '提炼课程结构图为后续统一遵循的文本总纲'
            },
            replaceProjectStateFields: [],
            reason: '用户希望把本轮提炼结果作为后续统一依据。',
            confidence: 0.88
        });

        finalizeWorkingMemoryWithAssistantResult(
            session,
            null,
            '项目总纲：课程分为多个关卡，后续所有封面、地图与教学内容都要围绕同一能力递进。'
        );

        expect(session.workingMemory.briefLocked).toBe(true);
        expect(session.workingMemory.canonicalBrief).toContain('项目总纲');
        expect(session.workingMemory.pendingBriefCommit).toBeNull();
    });

    it('replaces scalar project state fields when the user updates them later', () => {
        const session = { id: 's3', messages: [], workingMemory: createDefaultWorkingMemory() };

        mergeInterpreterResultIntoSession(session, {
            mode: 'ongoing_project',
            userIntent: 'plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: false,
            briefSource: 'none',
            canonicalGoal: '先按100元预算做第一版方案',
            referencePolicy: 'none',
            projectStatePatch: {
                budget: '100元',
                primaryGoal: '做第一版方案'
            },
            replaceProjectStateFields: [],
            reason: '用户先给出初始预算。',
            confidence: 0.8
        });

        mergeInterpreterResultIntoSession(session, {
            mode: 'ongoing_project',
            userIntent: 'adjust_plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: false,
            briefSource: 'none',
            canonicalGoal: '预算提升到1000元，方案也一起升级',
            referencePolicy: 'none',
            projectStatePatch: {
                budget: '1000元',
                coreRequirements: ['方案升级']
            },
            replaceProjectStateFields: ['coreRequirements'],
            reason: '用户明确提高预算并升级要求。',
            confidence: 0.9
        });

        expect(session.workingMemory.projectState.budget).toBe('1000元');
        expect(session.workingMemory.projectState.coreRequirements).toEqual(['方案升级']);
    });

    it('treats additive requests as supplements that keep existing state', () => {
        const session = { id: 's4', messages: [], workingMemory: createDefaultWorkingMemory() };

        mergeInterpreterResultIntoSession(session, {
            mode: 'ongoing_project',
            userIntent: 'plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: false,
            briefSource: 'none',
            canonicalGoal: '做课程封面',
            referencePolicy: 'none',
            projectStatePatch: {
                coreRequirements: ['保留课程结构']
            },
            replaceProjectStateFields: [],
            reason: '初始化要求',
            confidence: 0.8
        });

        const resolved = resolveStateConflict({
            text: '另外再加一点游戏感和趣味性',
            workingMemory: session.workingMemory
        }, {
            mode: 'ongoing_project',
            userIntent: 'adjust_plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: false,
            briefSource: 'none',
            canonicalGoal: '补充趣味性要求',
            referencePolicy: 'none',
            projectStatePatch: {
                coreRequirements: ['增加游戏感和趣味性']
            },
            replaceProjectStateFields: [],
            reason: '补充要求',
            confidence: 0.85
        });

        expect(resolved.stateTransition).toBe('supplement');
        mergeInterpreterResultIntoSession(session, resolved);
        expect(session.workingMemory.projectState.coreRequirements).toEqual(['保留课程结构', '增加游戏感和趣味性']);
    });

    it('treats explicit overturns as clearing old baseline fields', () => {
        const session = { id: 's5', messages: [], workingMemory: createDefaultWorkingMemory() };

        mergeInterpreterResultIntoSession(session, {
            mode: 'ongoing_project',
            userIntent: 'plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: true,
            briefSource: 'user_request',
            canonicalGoal: '旧方向总纲',
            referencePolicy: 'authoritative_reference',
            projectStatePatch: {
                currentDeliverable: '第一关封面',
                coreRequirements: ['保留旧结构'],
                styleAnchors: ['温暖手绘']
            },
            replaceProjectStateFields: ['coreRequirements', 'styleAnchors'],
            reason: '初始化旧方向',
            confidence: 0.9
        });

        const resolved = resolveStateConflict({
            text: '推翻之前的风格和要求，这次全部重来，改成蓝绿色科技风',
            workingMemory: session.workingMemory
        }, {
            mode: 'ongoing_project',
            userIntent: 'adjust_plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: false,
            briefSource: 'none',
            canonicalGoal: '改成蓝绿色科技风的新方向',
            referencePolicy: 'authoritative_reference',
            projectStatePatch: {
                styleAnchors: ['蓝绿色科技风']
            },
            replaceProjectStateFields: [],
            reason: '用户推翻旧方向',
            confidence: 0.92
        });

        expect(resolved.stateTransition).toBe('overturn');
        expect(resolved.shouldUnlockBrief).toBe(true);
        mergeInterpreterResultIntoSession(session, resolved);
        expect(session.workingMemory.briefLocked).toBe(false);
        expect(session.workingMemory.projectState.currentDeliverable).toBe('');
        expect(session.workingMemory.projectState.styleAnchors).toEqual(['蓝绿色科技风']);
        expect(session.workingMemory.projectState.coreRequirements).toEqual([]);
    });

    it('treats new project switches as resetting prior project state', () => {
        const session = { id: 's6', messages: [], workingMemory: createDefaultWorkingMemory() };

        mergeInterpreterResultIntoSession(session, {
            mode: 'ongoing_project',
            userIntent: 'plan',
            continuity: 'continuing',
            shouldResetMemory: false,
            shouldPersistBrief: false,
            briefSource: 'none',
            canonicalGoal: '旧课程项目',
            referencePolicy: 'none',
            projectStatePatch: {
                subject: '旧课程',
                budget: '100元',
                coreRequirements: ['旧要求']
            },
            replaceProjectStateFields: ['coreRequirements'],
            reason: '初始化旧项目',
            confidence: 0.8
        });

        const resolved = resolveStateConflict({
            text: '我们换个项目，重新开一个项目，做中秋节活动海报',
            workingMemory: session.workingMemory
        }, {
            mode: 'ongoing_project',
            userIntent: 'plan',
            continuity: 'switching',
            shouldResetMemory: false,
            shouldPersistBrief: false,
            briefSource: 'none',
            canonicalGoal: '中秋节活动海报项目',
            referencePolicy: 'none',
            projectStatePatch: {
                subject: '中秋节活动海报',
                primaryGoal: '做新的活动海报'
            },
            replaceProjectStateFields: [],
            reason: '切换到新项目',
            confidence: 0.95
        });

        expect(resolved.stateTransition).toBe('new_project_switch');
        expect(resolved.shouldResetMemory).toBe(true);
        mergeInterpreterResultIntoSession(session, resolved);
        expect(session.workingMemory.projectState.subject).toBe('中秋节活动海报');
        expect(session.workingMemory.projectState.budget).toBe('');
        expect(session.workingMemory.projectState.coreRequirements).toEqual([]);
    });

    it('normalizes invalid classifier output conservatively', () => {
        const normalized = normalizeIntentDecision({
            route: 'image_query',
            imageQueryMode: 'unknown_mode',
            confidence: 2,
            reason: 'test'
        }, {
            text: '这张图主要讲了什么？',
            baseImage: { name: 'base.png' }
        });

        expect(normalized.route).toBe('image_query');
        expect(normalized.imageQueryMode).toBe('answer_question');
        expect(normalized.confidence).toBe(1);
    });

    it('falls back safely when classifier is unavailable', () => {
        expect(fallbackIntentDecision({
            text: '帮我总结这张图的结构',
            baseImage: { name: 'base.png' }
        }).route).toBe('image_query');

        expect(fallbackIntentDecision({
            text: '生成一张未来城市海报',
            baseImage: null
        }).route).toBe('image_generation');

        expect(fallbackIntentDecision({
            text: '参考这张图继续做下一张，不要复制上一张构图',
            baseImage: { name: 'cover.png' }
        }).route).toBe('image_generation');

        expect(fallbackIntentDecision({
            text: '我们先想想方案',
            baseImage: null
        }).route).toBe('text_chat');
    });
});
