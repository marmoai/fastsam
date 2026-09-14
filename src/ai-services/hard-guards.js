import { looksLikeLensSearchRequest } from '../services/capability-orchestrator.js';

function buildDecision(route, overrides = {}) {
    return {
        route,
        imageQueryMode: null,
        confidence: 1,
        reason: '',
        source: 'hard_guard',
        ...overrides
    };
}

function getRecentTranscript(history = [], maxMessages = 6) {
    return history
        .filter(msg => msg && msg.content && (msg.type === 'text' || msg.type === 'bot-rich'))
        .slice(-maxMessages)
        .map(msg => String(msg.content).trim())
        .join('\n');
}

function isExplicitRegenerateRequest(text = '') {
    return /重新生成|重做一版|重做一下|重来一版|再来一版|换一版|换个版本|出一版|做一版|重新出图/.test(text);
}

function isContinuationExecutionRequest(text = '') {
    return /开始吧|继续|继续做|往下做|直接做|先做|按这个改|照这个改|就按这个|按这个来|去做吧|改一下|优化一下|那就做|就这样做/.test(text);
}

export function looksLikeWorkspaceStatusRequest(text = '') {
    const normalized = String(text || '').trim().toLowerCase();
    if (!normalized) return false;

    return /当前工作台|工作台状态|工作台上|当前有哪些图层|有哪些图层|哪些图层已提取|已提取.*图层|最近.*工作台|刚才.*工作台|现在能做什么|这张图现在能做什么|当前选中|当前图片.*图层|clean\s*plate|清理底图|语义分析.*完成/.test(normalized);
}

export function looksLikeWorkspaceModeSwitchRequest(text = '') {
    const normalized = String(text || '').trim();
    if (!normalized) return false;

    return /切换到?|进入|改走|使用|按/.test(normalized)
        && /室内设计模式|室内模式|室内设计|平面广告模式|平面设计模式|平面模式|广告模式|平面广告/.test(normalized);
}

export function looksLikeSameImageEditRequest(text = '') {
    const normalized = String(text || '').trim();
    if (!normalized) return false;

    return /修改这张|改这张|在这张(图片|图)?基础上|基于这张(图片|图)|沿用这张|保留这张|原图|当前图|局部重绘|局部修改|替换掉|把这张图改成|对这张图|按这个改|照这个改/.test(normalized)
        || (/修改|替换|编辑|重绘|扩图|调整|改成|改为|换成|加上|加个|加一|移除|删除|擦掉|去掉|颜色|材质|纹理/.test(normalized)
            && /这张|当前|原图|局部|基础上/.test(normalized));
}

export function looksLikeReferenceDrivenGenerationRequest(text = '') {
    const normalized = String(text || '').trim();
    if (!normalized) return false;

    return /参考这张|参考上一张|参考上一个版本|按这个风格继续|按这张风格继续|同风格继续|延续.*风格|保持.*风格.*(继续|生成|做)|下一张|下一页|下一部分|下一关|第二张|第三张|新的一张|另一张|新画面|新场景|不要复制构图|不要沿用原构图|重新开始.*生成/.test(normalized);
}

function hasRecentImageProductionContext(history = []) {
    const transcript = getRecentTranscript(history);
    if (!transcript) return false;

    const productionSignals = [
        /生成|出图|做图|重绘|重做|重生成/,
        /封面|海报|主视觉|版式|排版|插画|效果图/,
        /改图|修改图片|替换|换成|调整视觉|统一风格/
    ];

    const discussionOnlySignals = [
        /总结|概括|提取文字|OCR|整理成文本|翻译|解释这张图|这张图讲了什么/
    ];

    const hasProductionSignal = productionSignals.some(pattern => pattern.test(transcript));
    const looksMostlyDiscussion = discussionOnlySignals.some(pattern => pattern.test(transcript)) && !hasProductionSignal;

    return hasProductionSignal && !looksMostlyDiscussion;
}

export function applyConversationHardGuards(context = {}) {
    const text = (context.text || '').trim();
    const hasBaseImage = !!context.baseImage;
    const hasMask = !!(context.hasMask || context.mask);
    const referenceImagesCount = typeof context.referenceImagesCount === 'number'
        ? context.referenceImagesCount
        : Array.isArray(context.referenceImages) ? context.referenceImages.length : 0;
    const recentHistory = Array.isArray(context.history) ? context.history : [];

    if (looksLikeWorkspaceStatusRequest(text) || looksLikeWorkspaceModeSwitchRequest(text) || looksLikeLensSearchRequest(text)) {
        return buildDecision('text_chat', {
            confidence: 0.99,
            reason: looksLikeLensSearchRequest(text)
                ? '用户要求对 Workbench 对象执行 Lens 同款搜索，进入能力调度链路。'
                : looksLikeWorkspaceModeSwitchRequest(text)
                ? '用户要求切换 Workbench 场景模式，进入工作台动作链路。'
                : '用户询问当前 Workbench 状态或可执行能力，保持只读工作台问答链路。'
        });
    }

    if (!text && hasBaseImage) {
        return buildDecision('image_query', {
            imageQueryMode: 'describe_image',
            reason: '用户已提供底图但未输入文本，默认进入图片理解。'
        });
    }

    if (hasBaseImage && hasMask) {
        return buildDecision('image_edit', {
            reason: '存在蒙版，必须进入图片编辑链路。'
        });
    }

    if (hasBaseImage && looksLikeReferenceDrivenGenerationRequest(text)) {
        return buildDecision('image_generation', {
            confidence: 0.96,
            reason: '用户把当前图片作为风格或结构参考来继续生成新图，进入图片生成链路。'
        });
    }

    if (hasBaseImage && referenceImagesCount > 0) {
        if (looksLikeSameImageEditRequest(text)) {
            return buildDecision('image_edit', {
                confidence: 0.98,
                reason: '用户明确要求修改当前图片，同时存在参考图，进入图片编辑链路。'
            });
        }

        if (looksLikeReferenceDrivenGenerationRequest(text)) {
            return buildDecision('image_generation', {
                confidence: 0.96,
                reason: '用户把当前图片作为视觉参考来继续生成新图，进入图片生成链路。'
            });
        }
    }

    if (isExplicitRegenerateRequest(text)) {
        if (hasBaseImage && looksLikeReferenceDrivenGenerationRequest(text)) {
            return buildDecision('image_generation', {
                confidence: 0.97,
                reason: '用户要求参考现有图片重新出一张新图，进入图片生成链路。'
            });
        }

        if (hasBaseImage && looksLikeSameImageEditRequest(text)) {
            return buildDecision('image_edit', {
                confidence: 0.97,
                reason: '用户明确要求在当前图片上下文中重做版本，进入图片编辑链路。'
            });
        }

        return buildDecision('image_generation', {
            confidence: 0.97,
            reason: hasBaseImage
                ? '用户明确要求重新生成版本，但未说明要直接修改当前底图，优先按新图生成处理。'
                : '用户明确要求重新生成新版本，直接进入图片生成链路。'
        });
    }

    if (isContinuationExecutionRequest(text) && hasRecentImageProductionContext(recentHistory)) {
        if (hasBaseImage && looksLikeSameImageEditRequest(text)) {
            return buildDecision('image_edit', {
                confidence: 0.93,
                reason: '用户在延续前文，并明确要继续修改当前图片。'
            });
        }

        if (hasBaseImage && looksLikeReferenceDrivenGenerationRequest(text)) {
            return buildDecision('image_generation', {
                confidence: 0.93,
                reason: '用户在延续前文，并把当前图片作为参考继续生成新图。'
            });
        }

        if (!hasBaseImage) {
            return buildDecision('image_generation', {
                confidence: 0.93,
                reason: '用户在延续前文的图片制作任务，当前应继续执行图片生成。'
            });
        }

        return null;
    }

    if (hasBaseImage && looksLikeSameImageEditRequest(text)) {
        return buildDecision('image_edit', {
            confidence: 0.93,
            reason: '用户明确要求修改当前图片，直接进入图片编辑链路。'
        });
    }

    return null;
}
