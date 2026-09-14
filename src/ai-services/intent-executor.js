import { getExplicitRequestedImageCount, isImageGenerationRequest, isMaterialRequest, isRemovalRequest } from "../core/utils.js";
import { generateTextWithGemini, getImageModel, isQwenImageFamilyModel } from "./gemini-client.js";
import { editOrQueryImageWithGemini, editOrQueryImageWithGemini_Multiple, generateImage, queryImageWithGemini } from "./skills-engine.js";
import { applyConversationHardGuards } from "./hard-guards.js";
import { classifyConversationIntent, fallbackIntentDecision } from "./llm-intent-classifier.js";
import { applyWorkingMemoryToPrompt } from "./session-working-memory.js";

export async function determineConversationIntent(context = {}) {
    const hardGuardDecision = applyConversationHardGuards({
        text: context.text,
        baseImage: context.baseImage,
        referenceImages: context.referenceImages,
        mask: context.mask,
        referenceImagesCount: Array.isArray(context.referenceImages) ? context.referenceImages.length : 0,
        history: context.history,
        workingMemory: context.workingMemory
    });

    if (hardGuardDecision) {
        return hardGuardDecision;
    }

    if (context.localImageRouting === true && isQwenImageFamilyModel(getImageModel())) {
        const text = String(context.text || '').trim();
        const hasBaseImage = !!context.baseImage;
        const editSignal = !!context.mask
            || isMaterialRequest(text)
            || isRemovalRequest(text)
            || /修改|替换|编辑|修图|重绘|扩图|调整|改成|改为|换成|加上|移除|删除|擦掉|去掉|局部重绘|图片处理|图像处理/.test(text);

        if (hasBaseImage && editSignal) {
            return {
                route: 'image_edit',
                imageQueryMode: null,
                confidence: 0.95,
                reason: 'Qwen 图片任务本地路由：检测到明确的当前图片编辑请求。',
                source: 'qwen_local'
            };
        }

        if (isImageGenerationRequest(text)) {
            return {
                route: 'image_generation',
                imageQueryMode: null,
                confidence: 0.9,
                reason: 'Qwen 图片任务本地路由：检测到明确的图片生成请求。',
                source: 'qwen_local'
            };
        }

        return fallbackIntentDecision(context, 'Qwen 图片任务无法由本地规则确认类型。');
    }

    return await classifyConversationIntent(context);
}

export function getRequestedImageCount(prompt = '') {
    return getExplicitRequestedImageCount(prompt, 6);
}

function isBracketedSystemLikeText(text = '') {
    return /^\s*\[.*\]\s*/.test(text);
}

function isExecutionContinuationPrompt(text = '') {
    const normalized = String(text || '').trim();
    if (!normalized) return false;

    return /开始吧|生成吧|继续|继续做|往下做|直接做|先做|按这个改|照这个改|就按这个|按这个来|去做吧|改一下|优化一下|那就做|就这样做|重新生成|重来一版|再来一版|换一版/.test(normalized)
        || normalized.length <= 8;
}

function collectRecentUserRequirements(history = [], currentPrompt = '', limit = 2) {
    const results = [];
    let skippedCurrent = false;

    for (let i = history.length - 1; i >= 0; i -= 1) {
        const msg = history[i];
        if (!msg || msg.sender !== 'user' || msg.type !== 'text' || !msg.content) continue;

        const content = String(msg.content).trim();
        if (!content || isBracketedSystemLikeText(content)) continue;

        if (!skippedCurrent && currentPrompt && content === currentPrompt.trim()) {
            skippedCurrent = true;
            continue;
        }

        if (content.length < 6 && !/[，。；：、]/.test(content)) continue;

        results.push(content);
        if (results.length >= limit) break;
    }

    return results.reverse();
}

function getRecentAssistantSummary(history = [], currentPrompt = '') {
    for (let i = history.length - 1; i >= 0; i -= 1) {
        const msg = history[i];
        if (!msg || msg.sender !== 'bot' || !msg.content) continue;
        const content = String(msg.content).trim();
        if (!content || content === currentPrompt.trim()) continue;
        return content;
    }

    return '';
}

export function buildWorkspaceContextPrompt(workspaceContext = {}) {
    if (!workspaceContext || typeof workspaceContext !== 'object') return '';

    const activeItem = workspaceContext.activeItem;
    const selectedItems = Array.isArray(workspaceContext.selectedItems)
        ? workspaceContext.selectedItems
        : [];
    const recentActions = Array.isArray(workspaceContext.recentActions)
        ? workspaceContext.recentActions.slice(-5)
        : [];

    const lines = [
        '当前 Workbench 状态（内部上下文，不是用户消息）',
        workspaceContext.sceneMode && workspaceContext.sceneMode !== 'unknown'
            ? `场景模式：${workspaceContext.sceneMode}`
            : '',
        activeItem
            ? `当前图片：${activeItem.name || '未命名图片'}，图层 ${activeItem.layerCount || 0} 个，已提取 ${activeItem.extractedLayerCount || 0} 个，语义分析：${activeItem.hasFullSemanticAnalysis ? '已完成' : '未完成'}，clean plate：${activeItem.hasCleanPlate ? '已有' : activeItem.cleanPlateStatus || '未完成'}`
            : '当前没有活动图片',
        selectedItems.length > 0
            ? `选中图片：${selectedItems.map(item => item.name || item.id).filter(Boolean).join('、')}`
            : '当前没有选中图片',
        activeItem?.editableLayerNames?.length > 0
            ? `可编辑图层：${activeItem.editableLayerNames.join('、')}`
            : '',
        activeItem?.selectedLayers?.length > 0
            ? `当前选中图层：${activeItem.selectedLayers.map(layer => layer.name).join('、')}`
            : '当前没有选中的图层',
        '可回答范围：当前图片与图层状态、已提取资产、clean plate、最近工作台动作，以及基于这些状态的下一步建议。',
        recentActions.length > 0
            ? `最近工作台动作：${recentActions.map(action => action.actionName).filter(Boolean).join('、')}`
            : '最近没有记录的工作台动作'
    ].filter(Boolean);

    return lines.join('\n');
}

function hasContinuityAnchor(memory = {}) {
    const continuityState = memory?.continuityState || {};
    return !!(
        String(continuityState.anchorSummary || '').trim()
        || (Array.isArray(continuityState.preserveElements) && continuityState.preserveElements.length > 0)
        || (Array.isArray(continuityState.variableElements) && continuityState.variableElements.length > 0)
    );
}

export function looksLikeSeriesContinuationRequest(prompt = '', workingMemory = {}, history = []) {
    const normalized = String(prompt || '').trim();
    if (!hasContinuityAnchor(workingMemory)) return false;

    if (!normalized) {
        return Array.isArray(history) && history.length > 0;
    }

    if (/(继续|延续|同系列|系列一致|新的一页|新页面|新场景|新关卡|下一页|下一张|下一关|下一章节|下一部分|第三关|第四关|基于上一张继续|参考上一张继续)/.test(normalized)) {
        return true;
    }

    return normalized.length <= 10 && isExecutionContinuationPrompt(normalized);
}

export function composeContinuityAwarePrompt(prompt, decision, executionContext = {}) {
    const normalizedPrompt = String(prompt || '').trim();
    if (!normalizedPrompt || !decision?.route) return normalizedPrompt;
    if (normalizedPrompt.includes('系列连续性执行原则')) return normalizedPrompt;
    if (decision.route !== 'image_generation' && decision.route !== 'image_edit') return normalizedPrompt;

    const continuityState = executionContext.workingMemory?.continuityState || {};
    if (!hasContinuityAnchor(executionContext.workingMemory)) {
        return normalizedPrompt;
    }

    const isSeriesContinuation = looksLikeSeriesContinuationRequest(
        normalizedPrompt,
        executionContext.workingMemory,
        executionContext.history || []
    );

    const continuityLines = [
        '系列连续性执行原则（高优先级）',
        continuityState.anchorSummary ? `系列视觉系统：${continuityState.anchorSummary}` : '',
        Array.isArray(continuityState.preserveElements) && continuityState.preserveElements.length > 0
            ? `必须继承：${continuityState.preserveElements.join('；')}`
            : '',
        Array.isArray(continuityState.variableElements) && continuityState.variableElements.length > 0
            ? `本轮允许变化：${continuityState.variableElements.join('；')}`
            : '',
        Array.isArray(continuityState.usageNotes) && continuityState.usageNotes.length > 0
            ? `执行提示：${continuityState.usageNotes.join('；')}`
            : '',
        decision.route === 'image_generation'
            ? (isSeriesContinuation
                ? '这是连续性项目中的新页面/新关卡生成任务。请继承系列系统，但不要机械复制上一张图的具体构图、局部场景、道具摆位或一次性色彩分布，除非用户明确要求。'
                : '如果当前图只是参考，请继承系列视觉系统与角色逻辑，不要把参考图逐像素照搬。')
            : (isSeriesContinuation
                ? '这是连续性项目中的延展编辑任务。允许为了新页面/新关卡目标调整场景、镜头、主色和内容结构，但仍要保持系列系统统一；不要机械复制上一张图的具体构图、局部场景和一次性道具。'
                : '如果用户只是要求在当前图上修改，请优先保留当前画面成立的结构，只改明确点名的部分。')
    ].filter(Boolean);

    return [
        continuityLines.join('\n'),
        normalizedPrompt ? `当前执行任务：${normalizedPrompt}` : ''
    ].filter(Boolean).join('\n\n');
}

export function resolveExecutionPrompt(decision, executionContext = {}) {
    const prompt = String(executionContext.prompt || '').trim();
    if (!decision?.route || !prompt) return prompt;
    if (prompt.includes('请严格执行下面已经确认的视觉需求')) return prompt;

    if (decision.route !== 'image_generation' && decision.route !== 'image_edit') {
        return prompt;
    }

    if (!isExecutionContinuationPrompt(prompt)) {
        return prompt;
    }

    const requirementMessages = collectRecentUserRequirements(executionContext.history || [], prompt, 2);
    const assistantSummary = getRecentAssistantSummary(executionContext.history || [], prompt);

    if (requirementMessages.length === 0 && !assistantSummary) {
        return prompt;
    }

    const taskLabel = decision.route === 'image_edit' ? '基于当前图片执行修改' : '生成一张新图片';

    return [
        '请严格执行下面已经确认的视觉需求，不要脱离主题自由发挥。',
        `任务类型：${taskLabel}`,
        requirementMessages.length > 0 ? `已确认需求：\n${requirementMessages.join('\n')}` : '',
        assistantSummary ? `最近确认摘要：\n${assistantSummary}` : '',
        `当前执行口令：${prompt}`,
        '如果当前这句话只是“开始吧”“生成吧”“继续”“按这个改”之类的执行口令，请以上面已经确认的需求为准直接产出结果。'
    ].filter(Boolean).join('\n\n');
}

export function composeExecutionPrompt(decision, executionContext = {}) {
    const resolvedPrompt = resolveExecutionPrompt(decision, executionContext);
    const continuityAwarePrompt = composeContinuityAwarePrompt(
        resolvedPrompt || executionContext.prompt,
        decision,
        executionContext
    );

    const promptWithWorkingMemory = applyWorkingMemoryToPrompt(
        continuityAwarePrompt || resolvedPrompt || executionContext.prompt,
        executionContext.workingMemory,
        {
            prefix: '当前会话工作记忆（高优先级）',
            instruction: decision?.route === 'image_generation' || decision?.route === 'image_edit'
                ? '如果当前请求与既有总纲并不冲突，请优先保持系列连续性、角色设定和视觉系统；只有当用户明确切换项目、推翻基线或要求重置方向时，才覆盖这些约束。'
                : '如果当前请求与既有总纲并不冲突，请优先保持连续性；只有当用户明确切换任务或推翻既有基线时，才覆盖这些约束。'
        }
    ) || continuityAwarePrompt || resolvedPrompt || executionContext.prompt;

    if (decision?.route !== 'text_chat') return promptWithWorkingMemory;

    const workspacePrompt = buildWorkspaceContextPrompt(executionContext.workspaceContext);
    return workspacePrompt
        ? `${workspacePrompt}\n\n用户当前问题：${promptWithWorkingMemory}`
        : promptWithWorkingMemory;
}

export function classifyRegenerateInstruction(actualInput = '') {
    const normalizedInput = String(actualInput || '').trim();
    if (!normalizedInput) {
        return {
            mode: 'reuse',
            normalizedInput,
            addition: '',
            overridePrompt: ''
        };
    }

    const explicitOverridePatterns = [
        /按新要求重新生成/,
        /按以下要求重新生成/,
        /按照?新的?要求重新生成/,
        /重新生成，?按以下要求/,
        /重新生成，?按这个要求/,
        /改按这个要求重新生成/,
        /不要参考上一次/,
        /不要沿用上一次/,
        /不基于上一次/,
        /覆盖旧(?:的)?\s*prompt/,
        /覆盖上一次(?:的)?(?:提示词|prompt)?/,
        /这次按新要求/
    ];

    if (explicitOverridePatterns.some(pattern => pattern.test(normalizedInput))) {
        const overridePrompt = normalizedInput
            .replace(/^按新要求重新生成[：:\s,，。；;]*/,'')
            .replace(/^按以下要求重新生成[：:\s,，。；;]*/,'')
            .replace(/^按照?新的?要求重新生成[：:\s,，。；;]*/,'')
            .replace(/^重新生成[：:\s,，。；;]*按以下要求[：:\s,，。；;]*/,'')
            .replace(/^重新生成[：:\s,，。；;]*按这个要求[：:\s,，。；;]*/,'')
            .replace(/^改按这个要求重新生成[：:\s,，。；;]*/,'')
            .replace(/^这次按新要求[：:\s,，。；;]*/,'')
            .trim();

        return {
            mode: 'override',
            normalizedInput,
            addition: '',
            overridePrompt: overridePrompt || normalizedInput
        };
    }

    const pureRegenerateCommand = /^(不满意|重来|再试一次|重新生成|换一个|另一个版本|重来一版|再来一版|换一版)[!！,，。.\s]*$/;
    if (pureRegenerateCommand.test(normalizedInput)) {
        return {
            mode: 'reuse',
            normalizedInput,
            addition: '',
            overridePrompt: ''
        };
    }

    const mergeAddition = normalizedInput
        .replace(/^(请)?重新生成(一版)?[：:\s,，。；;]*/, '')
        .replace(/^(请)?重来(一版)?[：:\s,，。；;]*/, '')
        .replace(/^(请)?再来一版[：:\s,，。；;]*/, '')
        .replace(/^(请)?换一版[：:\s,，。；;]*/, '')
        .replace(/^(请)?换一个版本[：:\s,，。；;]*/, '')
        .trim();

    return {
        mode: 'merge',
        normalizedInput,
        addition: mergeAddition || normalizedInput,
        overridePrompt: ''
    };
}

export function mergeRegeneratePrompt(basePrompt = '', actualInput = '') {
    const strategy = classifyRegenerateInstruction(actualInput);
    if (strategy.mode === 'reuse') {
        return basePrompt;
    }

    if (strategy.mode === 'override') {
        return strategy.overridePrompt || basePrompt;
    }

    if (!basePrompt) {
        return strategy.addition || strategy.normalizedInput;
    }

    return `${basePrompt}\n补充要求：${strategy.addition || strategy.normalizedInput}`;
}

export function getIntentPlaceholderText(decision, executionContext = {}) {
    if (!decision) return '思考中...';

    if (decision.route === 'image_generation') {
        return '正在生成图片...';
    }

    if (decision.route === 'image_edit') {
        return getRequestedImageCount(executionContext.prompt) > 1
            ? '正在为您生成多个版本...'
            : '正在修改图片...';
    }

    if (decision.route === 'image_query') {
        return '正在阅读图片...';
    }

    return '思考中...';
}

export function shouldUseImagePlaceholder(decision) {
    return decision?.route === 'image_generation' || decision?.route === 'image_edit';
}

export async function executeConversationIntent(decision, executionContext = {}) {
    if (!decision || !decision.route) {
        throw new Error('Missing route decision.');
    }

    const executionPrompt = executionContext.composedPrompt || composeExecutionPrompt(decision, executionContext);

    switch (decision.route) {
        case 'image_generation':
            return await generateImage(
                executionPrompt,
                executionContext.targetAspectRatio || '1:1',
                {
                    baseImage: executionContext.baseImage || null,
                    referenceImages: executionContext.referenceImages || [],
                    history: executionContext.history || [],
                    imageCount: getRequestedImageCount(executionPrompt)
                }
            );
        case 'image_edit':
            if (getRequestedImageCount(executionPrompt) > 1) {
                return await editOrQueryImageWithGemini_Multiple(
                    executionPrompt,
                    executionContext.baseImage,
                    executionContext.referenceImages || [],
                    executionContext.mask || null,
                    executionContext.targetAspectRatio || '1:1',
                    getRequestedImageCount(executionPrompt)
                );
            }

            return await editOrQueryImageWithGemini(
                executionPrompt,
                executionContext.baseImage,
                executionContext.referenceImages || [],
                executionContext.mask || null,
                executionContext.targetAspectRatio || '1:1'
            );
        case 'image_query':
            return await queryImageWithGemini(
                executionPrompt,
                executionContext.baseImage,
                executionContext.referenceImages || [],
                executionContext.history || [],
                {
                    imageQueryMode: decision.imageQueryMode,
                    reason: decision.reason
                }
            );
        case 'text_chat':
        default:
            return await generateTextWithGemini(
                executionPrompt,
                executionContext.sessionId,
                executionContext.history || [],
                executionContext.canvasState
            );
    }
}
