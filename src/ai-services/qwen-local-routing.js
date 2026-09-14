import { isMaterialRequest, isRemovalRequest } from '../core/utils.js';
import { isQwenImageFamilyModel } from './gemini-client.js';

// Temporary switch while the Gemini endpoint is under maintenance.
// Set to false after Gemini is restored to bring back the normal interpreter path.
export const ENABLE_QWEN_LOCAL_ROUTING_BYPASS = false;

const EDIT_KEYWORDS = /修改|替换|编辑|修图|重绘|扩图|调整|改成|改为|换成|加上|移除|删除|擦掉|去掉|局部重绘|图片处理|图像处理/;
const GENERATION_KEYWORDS = /生成|生图|出图|做图|绘制|创造|创作|重绘|重新生成|再生成|来一张|画一张|画一个/;
const LENS_KEYWORDS = /搜索同款|搜同款|以图搜图|镜像搜索|lens|灵感仓/iu;

export function shouldUseQwenLocalImageRouting({ model, text = '', baseImage = null, mask = null } = {}) {
    if (!ENABLE_QWEN_LOCAL_ROUTING_BYPASS || !isQwenImageFamilyModel(model)) return false;

    const normalized = String(text || '').trim();
    if (LENS_KEYWORDS.test(normalized)) return false;

    const generationSignal = GENERATION_KEYWORDS.test(normalized);
    const editSignal = !!mask
        || isMaterialRequest(normalized)
        || isRemovalRequest(normalized)
        || EDIT_KEYWORDS.test(normalized);

    return generationSignal || (!!baseImage && editSignal);
}

export function getQwenLocalRoutingDiagnostics({ model, text = '', baseImage = null, mask = null } = {}) {
    return {
        enabled: ENABLE_QWEN_LOCAL_ROUTING_BYPASS,
        selectedModel: typeof document !== 'undefined' ? document.getElementById('modelSelect')?.value || null : null,
        imageModel: model || null,
        text: String(text || '').trim(),
        hasBaseImage: !!baseImage,
        hasMask: !!mask,
        useLocalRouting: shouldUseQwenLocalImageRouting({ model, text, baseImage, mask })
    };
}
