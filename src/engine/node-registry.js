/**
 * Node Registry
 * Defines all available node types in the MarmoAid Node Graph system.
 * Each node type defines its inputs, outputs, and the skill it maps to.
 */

export const NODE_REGISTRY = {
    "generate-image": {
        title: "灵感生成",
        description: "根据文字描述生成全新的图像",
        inputs: [
            { id: "prompt", type: "string", label: "提示词" },
            { id: "aspectRatio", type: "string", label: "宽高比", default: "1:1" }
        ],
        outputs: [
            { id: "image", type: "image", label: "生成图像" }
        ],
        skill: "generateImage"
    },
    "edit-image": {
        title: "指令编辑",
        description: "对现有图像进行全局风格或内容的通用修改",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "修改指令" },
            { id: "mask", type: "mask", label: "蒙版", optional: true },
            { id: "referenceImages", type: "array", label: "参考图", optional: true }
        ],
        outputs: [
            { id: "image", type: "image", label: "编辑结果" }
        ],
        skill: "editOrQueryImageWithGemini"
    },
    "material-replace": {
        title: "材质替换",
        description: "替换图像中特定区域或物体的材质、纹理或表面属性",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "材质指令" },
            { id: "mask", type: "mask", label: "蒙版", optional: true },
            { id: "referenceImages", type: "array", label: "参考图", optional: true }
        ],
        outputs: [
            { id: "image", type: "image", label: "替换结果" }
        ],
        skill: "editOrQueryImageWithGemini"
    },
    "inpainting": {
        title: "局部重绘",
        description: "对图像的特定局部区域进行内容重绘、消除或填充",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "重绘指令" },
            { id: "mask", type: "mask", label: "蒙版", optional: true },
            { id: "referenceImages", type: "array", label: "参考图", optional: true }
        ],
        outputs: [
            { id: "image", type: "image", label: "重绘结果" }
        ],
        skill: "editOrQueryImageWithGemini"
    },
    "relight": {
        title: "重光照",
        description: "调整图像的光影效果",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "光影指令" }
        ],
        outputs: [
            { id: "image", type: "image", label: "光影结果" }
        ],
        skill: "generateRelitImage"
    },
    "analyze": {
        title: "智能分析",
        description: "AI 代理对图像进行多维分析",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "agentType", type: "string", label: "代理类型", enum: ["describe", "critique", "suggest"] }
        ],
        outputs: [
            { id: "analysis", type: "object", label: "分析结果" }
        ],
        skill: "analyzeWithAgent"
    },
    "visual-search": {
        title: "视觉搜索",
        description: "识别图中物体并搜索相关信息",
        inputs: [
            { id: "image", type: "image", label: "原图" }
        ],
        outputs: [
            { id: "data", type: "object", label: "搜索数据" },
            { id: "links", type: "array", label: "网页链接" }
        ],
        skill: "generateVisualSearch"
    },
    "layer-analysis": {
        title: "语义拆解",
        description: "自动识别并拆解图像中的视觉元素",
        inputs: [
            { id: "image", type: "image", label: "原图" }
        ],
        outputs: [
            { id: "layers", type: "array", label: "图层数据" }
        ],
        skill: "analyzeImageLayers"
    },
    "crop": {
        title: "局部裁剪",
        description: "根据图层名称从原图中裁剪出特定部分",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "layers", type: "array", label: "图层数据" },
            { id: "layerName", type: "string", label: "目标图层名称" }
        ],
        outputs: [
            { id: "image", type: "image", label: "裁剪结果" }
        ],
        skill: "cropImageByLayer"
    },
    "outpaint": {
        title: "外扩图片",
        description: "向外扩展图像的边界并补充内容",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "扩展指令", default: "请将这张图的边界向外扩展，保持原有风格" },
            { id: "aspectRatio", type: "string", label: "目标比例", enum: ["1:1", "16:9", "9:16", "4:3", "3:4"], default: "1:1" }
        ],
        outputs: [
            { id: "image", type: "image", label: "外扩结果" }
        ],
        skill: "outpaintImage"
    },
    "camera-lens": {
        title: "相机镜头",
        description: "调整图像的相机视角和镜头效果",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "镜头指令", default: "请调整这张图片的相机视角。必须严格执行新的视角！在改变透视和构图的同时，尽可能保留原主体的人物/物体特征和环境氛围，但绝对不要直接复制原图的角度。" }
        ],
        outputs: [
            { id: "image", type: "image", label: "镜头结果" }
        ],
        skill: "applyCameraLens"
    },
    "video-generation": {
        title: "视频生成",
        description: "将静态图像转换为动态视频",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "视频指令" },
            { id: "aspectRatio", type: "string", label: "视频比例", enum: ["16:9", "9:16"], default: "16:9" }
        ],
        outputs: [
            { id: "video", type: "video", label: "生成视频" }
        ],
        skill: "generateVideo"
    },
    "upscale": {
        title: "高清增强",
        description: "提升图像分辨率并补充细节",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "增强指令", default: "Image Restoration & Reconstruction: Redraw this low-quality image into a pristine, ultra-high-resolution (4K) masterpiece. Aggressively remove all blur, noise, and compression artifacts. CRITICAL: Do not just sharpen the existing pixels. Instead, synthesise and hallucinate missing high-frequency details (such as skin texture, hair strands, fabric patterns, and sharp edges) that are lost in the original. Re-imagine the subject with perfect focus and clarity while keeping the original subject identity, pose, and overall composition intact. The output must look like a sharp, professional commercial photograph taken with a modern high-end DSLR." }
        ],
        outputs: [
            { id: "image", type: "image", label: "增强结果" }
        ],
        skill: "upscaleImage"
    },
    "multiview": {
        title: "多维视图",
        description: "智能识别场景并生成多角度视图",
        inputs: [
            { id: "image", type: "image", label: "原图" },
            { id: "prompt", type: "string", label: "视图指令", optional: true }
        ],
        outputs: [
            { id: "image", type: "image", label: "多维结果" }
        ],
        skill: "generateMultiview"
    }
};
