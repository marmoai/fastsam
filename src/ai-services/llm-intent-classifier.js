import { Type } from "@google/genai";
import { isImageGenerationRequest, isMaterialRequest, isRemovalRequest } from "../core/utils.js";
import { getTextModel, proxyGenerateContent } from "./gemini-client.js";
import * as prompts from "./prompts.js";
import { summarizeWorkingMemory } from "./session-working-memory.js";
import { looksLikeReferenceDrivenGenerationRequest, looksLikeSameImageEditRequest, looksLikeWorkspaceStatusRequest, looksLikeWorkspaceModeSwitchRequest } from "./hard-guards.js";
import { looksLikeLensSearchRequest } from "../services/capability-orchestrator.js";

const VALID_ROUTES = new Set(['text_chat', 'image_generation', 'image_edit', 'image_query']);
const VALID_IMAGE_QUERY_MODES = new Set(['extract_text', 'summarize_content', 'answer_question', 'describe_image']);

function clampConfidence(value) {
    const num = Number(value);
    if (Number.isNaN(num)) return 0.5;
    return Math.min(1, Math.max(0, num));
}

function trimMessageContent(content, maxLength = 280) {
    const text = String(content || '').replace(/\s+/g, ' ').trim();
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function buildHistoryTranscript(history = [], maxMessages = 6) {
    return history
        .filter(msg => msg && msg.content && (msg.type === 'text' || msg.type === 'bot-rich'))
        .slice(-maxMessages)
        .map(msg => `${msg.sender === 'user' ? '用户' : '助手'}：${trimMessageContent(msg.content)}`);
}

function inferFallbackImageQueryMode(text = '') {
    const normalized = text.toLowerCase();
    if (/ocr|提取|转写|抄出来|摘出来|读取|识别文字|图里写了什么|图片里写了什么|整理成文本|转成文本|读一下文字/.test(normalized)) {
        return 'extract_text';
    }
    if (/总结|概括|归纳|提炼|梳理|摘要|说重点|结构|规划/.test(normalized)) {
        return 'summarize_content';
    }
    if (/为什么|怎么|是否|能不能|可不可以|什么意思|这是什么|图里有什么|图片里有什么|哪里|哪个|谁|吗|\?|\？/.test(normalized)) {
        return 'answer_question';
    }
    return 'describe_image';
}

function looksLikeEditFallback(text = '') {
    return looksLikeSameImageEditRequest(text) ||
        isMaterialRequest(text) ||
        isRemovalRequest(text) ||
        /修改|替换|编辑|重绘|扩图|调整|改成|改为|换成|加上|加个|加一|移除|删除|擦掉|去掉|颜色|材质|纹理|风格/.test(text.toLowerCase());
}

export function fallbackIntentDecision(context = {}, errorMessage = '') {
    const text = (context.text || '').trim();
    const hasBaseImage = !!context.baseImage;

    if (looksLikeWorkspaceStatusRequest(text) || looksLikeWorkspaceModeSwitchRequest(text) || looksLikeLensSearchRequest(text)) {
        return {
            route: 'text_chat',
            imageQueryMode: null,
            confidence: 0.99,
            reason: `用户询问当前 Workbench 状态或能力，使用只读状态回答。${errorMessage ? ` ${errorMessage}` : ''}`.trim(),
            source: 'fallback'
        };
    }

    if (hasBaseImage) {
        if (looksLikeReferenceDrivenGenerationRequest(text)) {
            return {
                route: 'image_generation',
                imageQueryMode: null,
                confidence: 0.45,
                reason: `分类器不可用，回退为参考图驱动的新图生成判断。${errorMessage ? ` ${errorMessage}` : ''}`.trim(),
                source: 'fallback'
            };
        }

        if (looksLikeEditFallback(text)) {
            return {
                route: 'image_edit',
                imageQueryMode: null,
                confidence: 0.45,
                reason: `分类器不可用，回退为保守编辑判断。${errorMessage ? ` ${errorMessage}` : ''}`.trim(),
                source: 'fallback'
            };
        }

        return {
            route: 'image_query',
            imageQueryMode: inferFallbackImageQueryMode(text),
            confidence: 0.45,
            reason: `分类器不可用，回退为保守图片理解判断。${errorMessage ? ` ${errorMessage}` : ''}`.trim(),
            source: 'fallback'
        };
    }

    if (isImageGenerationRequest(text)) {
        return {
            route: 'image_generation',
            imageQueryMode: null,
            confidence: 0.4,
            reason: `分类器不可用，回退为图片生成判断。${errorMessage ? ` ${errorMessage}` : ''}`.trim(),
            source: 'fallback'
        };
    }

    return {
        route: 'text_chat',
        imageQueryMode: null,
        confidence: 0.4,
        reason: `分类器不可用，回退为普通对话。${errorMessage ? ` ${errorMessage}` : ''}`.trim(),
        source: 'fallback'
    };
}

export function normalizeIntentDecision(rawDecision, context = {}, source = 'llm_classifier') {
    const normalized = {
        route: VALID_ROUTES.has(rawDecision?.route) ? rawDecision.route : null,
        imageQueryMode: VALID_IMAGE_QUERY_MODES.has(rawDecision?.imageQueryMode) ? rawDecision.imageQueryMode : null,
        confidence: clampConfidence(rawDecision?.confidence),
        reason: typeof rawDecision?.reason === 'string' ? rawDecision.reason.trim() : '',
        source
    };

    if (!normalized.route) {
        return fallbackIntentDecision(context, '分类器返回了无效 route。');
    }

    if ((normalized.route === 'image_edit' || normalized.route === 'image_query') && !context.baseImage) {
        return {
            route: 'text_chat',
            imageQueryMode: null,
            confidence: Math.min(normalized.confidence, 0.5),
            reason: '用户当前没有提供底图，无法直接执行图片编辑或图片理解，转为普通对话以便继续澄清。',
            source
        };
    }

    if (normalized.route === 'image_query' && !normalized.imageQueryMode) {
        normalized.imageQueryMode = inferFallbackImageQueryMode(context.text || '');
    }

    if (normalized.route !== 'image_query') {
        normalized.imageQueryMode = null;
    }

    if (!normalized.reason) {
        normalized.reason = `模型判断当前任务属于 ${normalized.route}。`;
    }

    return normalized;
}

function buildClassifierPrompt(context = {}) {
    const historyTranscript = buildHistoryTranscript(context.history || []);
    const contextPayload = {
        user_text: (context.text || '').trim(),
        has_base_image: !!context.baseImage,
        reference_image_count: Array.isArray(context.referenceImages) ? context.referenceImages.length : 0,
        has_mask: !!context.mask,
        selected_model_suite: context.selectedModel || '',
        workspace_context: context.workspaceContext || null,
        working_memory: summarizeWorkingMemory(context.workingMemory, { prefix: '当前会话工作记忆' }),
        recent_history: historyTranscript
    };

    return [
        '请根据以下上下文判断用户当前最想让小M做什么。',
        '不要考虑底层 API 实现，只判断任务意图。',
        'workspace_context 是 Workbench 的内部只读状态，不是聊天消息；可以据此判断当前图片、图层和最近动作，但不要把这些状态伪装成用户消息。',
        '如果用户是在继续前文任务，例如“开始吧”“继续”“先做第一部分”，必须结合 recent_history 推断。',
        '如果用户提供了图片，也不要自动视为 edit。先判断这张图是要被直接修改，还是仅作为风格/结构参考来生成一张新图。',
        '判断要点：如果目标是继续修改同一张图的像素、构图或局部内容，选 image_edit；如果目标是参考现有图的风格、元素或世界观，继续做下一张/下一部分/另一张新图，选 image_generation。',
        '上下文 JSON：',
        JSON.stringify(contextPayload, null, 2)
    ].join('\n');
}

export async function classifyConversationIntent(context = {}) {
    const response = await proxyGenerateContent({
        model: getTextModel(),
        systemInstruction: {
            parts: [{ text: prompts.getIntentClassifierSystemInstruction() }]
        },
        contents: [{ role: 'user', parts: [{ text: buildClassifierPrompt(context) }] }],
        config: {
            responseMimeType: 'application/json',
            responseSchema: {
                type: Type.OBJECT,
                properties: {
                    route: { type: Type.STRING },
                    imageQueryMode: { type: Type.STRING },
                    confidence: { type: Type.NUMBER },
                    reason: { type: Type.STRING }
                },
                required: ['route', 'confidence', 'reason']
            }
        }
    });

    const text = response?.text?.trim?.() || '';
    if (!text) {
        return fallbackIntentDecision(context, '分类器返回空内容。');
    }

    try {
        const cleaned = text.replace(/```json|```/g, '').trim();
        const parsed = JSON.parse(cleaned);
        return normalizeIntentDecision(parsed, context, 'llm_classifier');
    } catch (error) {
        return fallbackIntentDecision(context, `分类器 JSON 解析失败: ${error.message}`);
    }
}
