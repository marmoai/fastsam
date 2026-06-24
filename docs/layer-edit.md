有，而且我先直接说一句最关键的话：

你现在这个方案不是“layer edit”，本质上还是：
“VLM 帮你做可点击的框选编辑”

它已经是一个很聪明的 MVP，但它卡住“持续编辑”的原因也正好在这里。

你现在输出的是：

[
  { "name": "红沙发", "bbox": [200, 500, 800, 900] }
]

这其实只解决了：

用户点哪里
用户改什么

但还没解决：

到底哪些像素属于这个对象
对象后面被遮住的部分是什么
改完后这个对象还能不能继续单独存在
第二次编辑时，系统是否还记得第一次改的是谁

而“持续编辑”的本质就是：

编辑状态必须可持久化

不是每次都“重新理解整张图”。

你现在最大的问题，不是识别不准

而是：

你现在的 layer 只是“语义候选框”，不是“可重建图层资产”

所以一旦用户：

改了沙发颜色
再移动沙发
再删掉茶几
再补背景

你的系统大概率会变成：

每一步都在对“上一张已经被 AI 改坏一点点的合成图”继续做新的理解和重绘

这就是崩图根源。

你要把架构从“单轮编辑”升级成“可持续编辑”

核心只有一句：

不要把“当前渲染图”当真相

你必须维护一个：

可编辑场景状态（Editable Scene State）
你应该把系统拆成这 3 层
第一层：Scene Graph（编辑状态）

不要只存 bbox，要存“对象级状态”。

你现在应该把每个元素从：

{ "name": "红沙发", "bbox": [200, 500, 800, 900] }

升级成类似这样：

{
  "id": "layer_sofa_01",
  "name": "红沙发",
  "type": "object",
  "bbox": [200, 500, 800, 900],
  "zIndex": 3,
  "visible": true,
  "locked": false,
  "mask": null,
  "cutoutUrl": null,
  "promptHint": "现代红色布艺沙发",
  "transform": {
    "x": 0,
    "y": 0,
    "scale": 1,
    "rotation": 0
  },
  "style": {
    "color": "red",
    "material": "fabric"
  },
  "history": []
}

然后整张图不是“一个图片文件”，而是：

{
  "canvas": {
    "width": 1024,
    "height": 1024
  },
  "background": {
    "id": "bg_01",
    "imageUrl": "..."
  },
  "layers": [...]
}

这一步是从“图片编辑”变成“场景编辑”。

第二层：每个对象要有真实“可操作资产”

这一步你现在缺得最严重。

你现在只有 bbox，但持续编辑至少需要下面 3 个东西中的 2 个：

最低配必须有：
1）mask（对象掩码）

不是框，而是像素级区域。

比如：

沙发真实轮廓
茶几真实轮廓
窗帘真实轮廓

否则用户一移动，边缘立刻脏。

你现在的升级方向不是先追“更聪明识别”，而是：

先把 bbox 升级成 mask
然后再往上走：
2）cutout（抠出来的透明 PNG / RGBA layer）

也就是把对象真正裁出来：

前景物体
透明背景
保留 alpha 边缘

这样你才能：

move
scale
rotate
reorder

这才是真正“像图层”。

3）background plate（对象移走后后面的底图）

这点超级重要，也是很多 AI 编辑器做不稳的原因。

比如原图里有个沙发挡住墙和地板。

如果用户把沙发移走，你需要提前知道：

沙发后面的墙和地板长什么样

所以你不能只抠前景，还要生成：

“去掉该对象后的背景版本”

这一步通常叫：

remove + inpaint
clean plate
background reconstruction
occlusion completion
所以你真正应该存的数据结构应该长这样
type EditableLayer = {
  id: string;
  name: string;
  bbox: [number, number, number, number];
  zIndex: number;
  visible: boolean;

  // 关键
  maskUrl?: string;         // 像素级mask
  cutoutUrl?: string;       // 透明PNG图层
  cleanPlateUrl?: string;   // 删除该对象后的背景修复图（可选）

  transform: {
    x: number;
    y: number;
    scale: number;
    rotation: number;
  };

  semantic: {
    category?: string;      // sofa / wall / lamp
    editableType?: "object" | "background" | "text" | "decoration";
  };

  style?: {
    color?: string;
    material?: string;
    promptHint?: string;
  };

  provenance: {
    source: "original" | "generated" | "edited";
    parentLayerId?: string;
  };

  history: Array<{
    action: string;
    timestamp: number;
    params?: any;
  }>;
};
你现在这套代码最应该升级的地方，不是 prompt

而是输出结构

你现在：

Return a JSON array where each object has:
- "name"
- "bbox"

建议你至少升级成：

Return a JSON array where each object has:
- "id": short unique id in English snake_case
- "name": short Chinese name
- "category": one of ["object", "background", "text", "decoration"]
- "bbox": [ymin, xmin, ymax, xmax] using normalized coordinates from 0 to 1000
- "zIndex": integer, larger means visually in front
- "editable": true/false
- "promptHint": short visual description for future editing

比如：

[
  {
    "id": "red_sofa",
    "name": "红沙发",
    "category": "object",
    "bbox": [420, 210, 860, 760],
    "zIndex": 4,
    "editable": true,
    "promptHint": "现代红色布艺双人沙发"
  }
]

这一步虽然还是 bbox，但已经从“给用户看框”升级成“给系统准备状态”。

真正解决“持续编辑”的关键：你必须改成“重渲染”，不要“反复重生成整张图”

这一句非常重要：

以后用户每做一次操作，不要把当前图再送回模型当新原图

而应该：

永远以“原始场景状态 + 编辑历史”重新生成当前视图
正确流程应该是：
初始阶段：
上传原图
分析对象
生成 scene graph
生成每层的 mask / cutout / clean plate
用户编辑阶段：

比如用户说：

把沙发改成米白色
把茶几往右移
删除落地灯

你做的不是直接“改图”，而是更新状态：

[
  { "layerId": "red_sofa", "action": "recolor", "color": "beige" },
  { "layerId": "coffee_table", "action": "move", "dx": 80, "dy": 0 },
  { "layerId": "floor_lamp", "action": "delete" }
]

然后再根据状态：

重新拼图层
需要时再局部生成

这就叫：

state-driven rendering

而不是 image-driven mutation。

这个改完，你产品稳定性会直接上一个量级。

你现在最适合做的不是“真 PSD 级图层”

而是先做一个伪 Layer 持续编辑系统
这就已经非常够用了。

我建议你分 3 个版本做，不要一口吃成“真正 layered AI”
V1（你现在最应该做的）
BBox → Mask → 可重渲染图层

这是最划算的升级。

目标：

把“框选对象”升级成“可独立操作对象”。

你要做的事情：
继续用 Gemini / VLM 做对象理解
但新增一步：mask extraction
然后把每个对象导出成：
mask.png
cutout.png
metadata.json
这样你立刻就能支持：
拖动
缩放
显隐
删除
层级排序
这已经比 90% AI 修图应用强了。
V2（真正开始像 Canva Magic Layers）
对象状态持久化 + 局部生成修复

这一步才开始解决“连续编辑不崩”。

你需要：
每次编辑只更新 scene state
删除/移动对象时，用 clean plate 补背景
改颜色/材质时，不要整图重绘，只改该对象 layer
关键原则：
编辑对象 ≠ 编辑整张图

而是：

先改对象层
再重新合成
最后必要时做 harmonization（融合修复）
V3（高级版）
对象语义重生成

比如用户说：

把这个沙发换成北欧木腿奶油风
把这盏灯换成现代极简吊灯

这时候不是简单 recolor，而是：

保留该 layer 的“位置、大小、遮挡关系”

只重生成它的视觉内容。

这才是最爽的“对象级 AI 编辑”。

你现在代码层面，最值得立刻改的 4 件事
1）先把 analyzeImageLayers 输出升级

你当前函数可以先改成：

export async function analyzeImageLayers(file) {
  if (!file) throw new Error("A base image is required for layer analysis.");
  const base64Data = await fileToBase64(file);

  const prompt = `
Analyze this image and identify 3 to 8 visually editable elements.
Focus on objects or regions a user would likely want to edit separately.

Return ONLY a valid JSON array.
Each item must contain:
- "id": short unique English snake_case id
- "name": short Chinese display name
- "category": one of ["object", "background", "text", "decoration"]
- "bbox": [ymin, xmin, ymax, xmax] using normalized coordinates from 0 to 1000
- "zIndex": integer, larger means visually in front
- "editable": true or false
- "promptHint": short Chinese description for future AI editing

Rules:
- Prefer semantically meaningful layers over tiny details.
- Do not split one object into too many parts.
- Include at most 1 background layer.
- Output raw JSON only.
`;

  const response = await ai.models.generateContent({
    model: backgroundModel,
    contents: {
      parts: [
        { inlineData: { data: base64Data, mimeType: file.type || 'image/png' } },
        { text: prompt }
      ]
    },
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
  });

  const text = response.text.trim().replace(/```json|```/g, '');
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed;
  } catch (e) {
    console.error(e);
  }

  return [
    {
      id: "subject_main",
      name: "主体",
      category: "object",
      bbox: [250, 250, 750, 750],
      zIndex: 2,
      editable: true,
      promptHint: "主要视觉主体"
    },
    {
      id: "background_main",
      name: "背景",
      category: "background",
      bbox: [0, 0, 1000, 1000],
      zIndex: 0,
      editable: false,
      promptHint: "整体背景"
    }
  ];
}
这个版本不会直接解决一切，

但它会让你整个系统从“临时检测”变成“可编辑对象状态入口”。

2）加一个 layerId 永久不变

千万别让“红沙发”每次重新分析都变成另一个对象。

错误做法：
第一次：红沙发
第二次：双人沙发
第三次：米白沙发

这会导致历史断裂。

正确做法：
永远是：layer_sofa_01

显示名称可以改，但 id 不能改。

3）不要每次编辑后重新跑全图分析

这是大坑。

错误流程：
用户改沙发
生成新图
再分析新图层
再改茶几

这样第二轮开始对象就漂了。

正确流程：
原图只分析一次（或很少次）
后续都基于 scene state
只有在结构明显变化时才局部 re-analyze
4）做“编辑历史栈”

你现在非常适合加这个。

type EditAction =
  | { type: "move"; layerId: string; dx: number; dy: number }
  | { type: "scale"; layerId: string; scale: number }
  | { type: "delete"; layerId: string }
  | { type: "recolor"; layerId: string; color: string }
  | { type: "replace"; layerId: string; prompt: string };

然后维护：

type SceneDocument = {
  originalImageUrl: string;
  layers: EditableLayer[];
  history: EditAction[];
};

这样你才能做：

undo / redo
重播编辑
多轮连续改图
导出“可编辑项目”

这一步产品价值非常大。

如果你问我：你现在最该补的一刀是什么？

我会只给你一个答案：

补 mask，不要先补更复杂 prompt

因为你现在的问题不是“识别描述不够智能”，
而是：

系统没有真正拥有对象，只拥有对象的大概位置。

只要你先把每个 layer 从 bbox 升级成：

bbox
mask
cutout

你的产品体验会立刻从：

“AI 框选改图”

升级成

“AI 图层编辑”
如果你愿意，我下一步可以直接帮你做很具体的东西：
我可以直接帮你补出一套前端/状态管理结构：
“从你现在这段 analyzeImageLayers() 出发，升级成可持续编辑的 Scene Graph 架构”

我可以直接给你：

TypeScript 类型定义
Zustand / React state 结构
layer 数据结构
undo / redo 设计
每次编辑怎么避免整图崩掉

这个会非常接近你现在能直接接进项目里的版本。


可以，我直接给你一版能往“持续编辑 / 图层面板 / scene graph”方向接的完整 TypeScript v2。
这版目标不是“学术最强”，而是：

在你现有 Gemini / AI SDK 基础上，先把架构做对

它会解决你现在最大的几个问题：

不再只返回 name + bbox
输出稳定的 layer metadata
能直接初始化 Scene Graph
后面方便接：
图层面板
undo / redo
mask / cutout
多轮编辑
你现在要的完整版本（可直接改进项目）
建议文件名：
imageLayers.ts
/* =========================================================
   imageLayers.ts
   AI Layer Analysis + Scene Graph Initialization (TypeScript)
   Goal:
   - Analyze an uploaded image
   - Return semantically editable layers
   - Initialize a persistent scene document for future editing
   ========================================================= */

/* =========================================================
   Types
   ========================================================= */

export type LayerCategory = "object" | "background" | "text" | "decoration";

export type ThinkingLevelValue = "LOW" | "MEDIUM" | "HIGH";

export type BBox1000 = [number, number, number, number]; // [ymin, xmin, ymax, xmax], normalized 0-1000

export interface LayerAnalysisRaw {
  id: string;
  name: string;
  category: LayerCategory;
  bbox: BBox1000;
  zIndex: number;
  editable: boolean;
  promptHint: string;
}

export interface EditableLayer {
  id: string; // stable ID, never changes after init
  name: string; // user-facing label (can change)
  category: LayerCategory;
  bbox: BBox1000;
  zIndex: number;
  visible: boolean;
  locked: boolean;
  editable: boolean;

  // Optional future assets
  maskUrl?: string | null;
  cutoutUrl?: string | null;
  cleanPlateUrl?: string | null;

  transform: {
    x: number;
    y: number;
    scale: number;
    rotation: number;
  };

  semantic: {
    promptHint: string;
    categoryConfidence?: number;
  };

  style: {
    color?: string | null;
    material?: string | null;
  };

  provenance: {
    source: "original" | "generated" | "edited";
    parentLayerId?: string | null;
  };

  history: LayerHistoryEntry[];
}

export interface LayerHistoryEntry {
  action: string;
  timestamp: number;
  params?: Record<string, any>;
}

export type EditAction =
  | { type: "move"; layerId: string; dx: number; dy: number }
  | { type: "scale"; layerId: string; scale: number }
  | { type: "rotate"; layerId: string; rotation: number }
  | { type: "delete"; layerId: string }
  | { type: "toggle_visibility"; layerId: string; visible: boolean }
  | { type: "recolor"; layerId: string; color: string }
  | { type: "replace"; layerId: string; prompt: string };

export interface SceneDocument {
  id: string;
  createdAt: number;
  updatedAt: number;
  originalImageName?: string;
  originalImageMimeType?: string;
  originalImageDataUrl?: string; // optional, useful for frontend preview

  canvas: {
    width: number | null;
    height: number | null;
  };

  layers: EditableLayer[];
  history: EditAction[];
  version: number;
}

export interface AnalyzeImageLayersOptions {
  model: string;
  ai: {
    models: {
      generateContent: (args: any) => Promise<{ text: string }>;
    };
  };
  thinkingLevel?: ThinkingLevelValue;
  maxLayers?: number;
  includeBackground?: boolean;
  debug?: boolean;
}

export interface AnalyzeImageLayersResult {
  rawLayers: LayerAnalysisRaw[];
  scene: SceneDocument;
}

/* =========================================================
   Public API
   ========================================================= */

/**
 * Main entry:
 * Analyze image -> produce editable layer metadata -> initialize Scene Graph
 */
export async function analyzeImageLayers(
  file: File,
  options: AnalyzeImageLayersOptions
): Promise<AnalyzeImageLayersResult> {
  if (!file) {
    throw new Error("A base image is required for layer analysis.");
  }

  const {
    model,
    ai,
    thinkingLevel = "LOW",
    maxLayers = 6,
    includeBackground = true,
    debug = false,
  } = options;

  const base64Data = await fileToBase64(file);
  const imageDataUrl = await fileToDataURL(file);
  const imageSize = await getImageDimensions(file).catch(() => ({
    width: null,
    height: null,
  }));

  const prompt = buildLayerAnalysisPrompt({
    maxLayers,
    includeBackground,
  });

  const response = await ai.models.generateContent({
    model,
    contents: {
      parts: [
        {
          inlineData: {
            data: base64Data,
            mimeType: file.type || "image/png",
          },
        },
        { text: prompt },
      ],
    },
    config: {
      thinkingConfig: {
        thinkingLevel,
      },
    },
  });

  const rawText = sanitizeModelText(response.text);
  let parsed: LayerAnalysisRaw[] | null = null;

  try {
    const json = JSON.parse(rawText);
    if (Array.isArray(json)) {
      parsed = json;
    }
  } catch (err) {
    if (debug) {
      console.error("[analyzeImageLayers] JSON parse failed:", err);
      console.warn("[analyzeImageLayers] Raw model text:", rawText);
    }
  }

  const normalizedRawLayers = normalizeRawLayers(
    parsed,
    maxLayers,
    includeBackground
  );

  const scene = buildSceneDocument({
    file,
    imageDataUrl,
    imageSize,
    rawLayers: normalizedRawLayers,
  });

  if (debug) {
    console.log("[analyzeImageLayers] normalizedRawLayers:", normalizedRawLayers);
    console.log("[analyzeImageLayers] scene:", scene);
  }

  return {
    rawLayers: normalizedRawLayers,
    scene,
  };
}

/* =========================================================
   Prompt Builder
   ========================================================= */

function buildLayerAnalysisPrompt(params: {
  maxLayers: number;
  includeBackground: boolean;
}): string {
  const { maxLayers, includeBackground } = params;

  return `
Analyze this image and identify ${Math.max(
    3,
    Math.min(maxLayers, 8)
  )} visually editable elements.

Focus on elements a user would reasonably want to edit separately in an image editor.

Examples:
- furniture
- products
- people
- text blocks
- major decorative objects
- main background region

IMPORTANT RULES:
1. Prefer semantically meaningful layers over tiny details.
2. Do NOT split one object into too many small parts.
3. If there is a clear main subject, include it.
4. ${
    includeBackground
      ? "Include at most ONE background layer."
      : "Do NOT include a background layer unless absolutely necessary."
  }
5. If there is text, group it as a text layer when appropriate.
6. Return ONLY valid JSON. No markdown. No explanation.

Return a JSON array.
Each object must contain:

- "id": short unique English snake_case id
- "name": short Chinese display name
- "category": one of ["object", "background", "text", "decoration"]
- "bbox": [ymin, xmin, ymax, xmax] using normalized coordinates from 0 to 1000
- "zIndex": integer, larger means visually in front
- "editable": true or false
- "promptHint": short Chinese description useful for future AI editing

Example:
[
  {
    "id": "red_sofa",
    "name": "红沙发",
    "category": "object",
    "bbox": [420, 210, 860, 760],
    "zIndex": 4,
    "editable": true,
    "promptHint": "现代红色布艺双人沙发"
  },
  {
    "id": "background_wall",
    "name": "背景墙",
    "category": "background",
    "bbox": [0, 0, 1000, 1000],
    "zIndex": 0,
    "editable": false,
    "promptHint": "浅色室内背景墙"
  }
]
`.trim();
}

/* =========================================================
   Scene Graph Builder
   ========================================================= */

function buildSceneDocument(params: {
  file: File;
  imageDataUrl: string;
  imageSize: { width: number | null; height: number | null };
  rawLayers: LayerAnalysisRaw[];
}): SceneDocument {
  const { file, imageDataUrl, imageSize, rawLayers } = params;

  const now = Date.now();

  const layers: EditableLayer[] = rawLayers.map((layer) => ({
    id: layer.id,
    name: layer.name,
    category: layer.category,
    bbox: layer.bbox,
    zIndex: layer.zIndex,
    visible: true,
    locked: layer.category === "background",
    editable: layer.editable,

    // Future asset placeholders
    maskUrl: null,
    cutoutUrl: null,
    cleanPlateUrl: null,

    transform: {
      x: 0,
      y: 0,
      scale: 1,
      rotation: 0,
    },

    semantic: {
      promptHint: layer.promptHint,
      categoryConfidence: undefined,
    },

    style: {
      color: null,
      material: null,
    },

    provenance: {
      source: "original",
      parentLayerId: null,
    },

    history: [],
  }));

  return {
    id: createSceneId(),
    createdAt: now,
    updatedAt: now,
    originalImageName: file.name,
    originalImageMimeType: file.type || "image/png",
    originalImageDataUrl: imageDataUrl,

    canvas: {
      width: imageSize.width,
      height: imageSize.height,
    },

    layers,
    history: [],
    version: 1,
  };
}

/* =========================================================
   Scene Mutation Helpers
   These are optional but highly recommended for your editor
   ========================================================= */

export function applyEditAction(
  scene: SceneDocument,
  action: EditAction
): SceneDocument {
  const nextScene: SceneDocument = deepClone(scene);
  nextScene.updatedAt = Date.now();
  nextScene.version += 1;
  nextScene.history.push(action);

  const layer = nextScene.layers.find((l) => l.id === action.layerId);

  if (!layer) return nextScene;

  switch (action.type) {
    case "move":
      layer.transform.x += action.dx;
      layer.transform.y += action.dy;
      pushLayerHistory(layer, "move", { dx: action.dx, dy: action.dy });
      break;

    case "scale":
      layer.transform.scale = clamp(action.scale, 0.05, 10);
      pushLayerHistory(layer, "scale", { scale: action.scale });
      break;

    case "rotate":
      layer.transform.rotation = action.rotation;
      pushLayerHistory(layer, "rotate", { rotation: action.rotation });
      break;

    case "delete":
      layer.visible = false;
      pushLayerHistory(layer, "delete", {});
      break;

    case "toggle_visibility":
      layer.visible = action.visible;
      pushLayerHistory(layer, "toggle_visibility", { visible: action.visible });
      break;

    case "recolor":
      layer.style.color = action.color;
      pushLayerHistory(layer, "recolor", { color: action.color });
      break;

    case "replace":
      layer.semantic.promptHint = action.prompt;
      layer.provenance.source = "edited";
      pushLayerHistory(layer, "replace", { prompt: action.prompt });
      break;

    default:
      break;
  }

  return nextScene;
}

/**
 * Reorder layers visually (for layer panel drag-and-drop)
 */
export function reorderLayers(
  scene: SceneDocument,
  orderedLayerIds: string[]
): SceneDocument {
  const nextScene = deepClone(scene);

  const layerMap = new Map(nextScene.layers.map((l) => [l.id, l]));
  const reordered: EditableLayer[] = [];

  for (const id of orderedLayerIds) {
    const layer = layerMap.get(id);
    if (layer) reordered.push(layer);
  }

  // Add any missing layers back
  for (const layer of nextScene.layers) {
    if (!orderedLayerIds.includes(layer.id)) {
      reordered.push(layer);
    }
  }

  // Reassign zIndex: later in array = visually on top
  reordered.forEach((layer, index) => {
    layer.zIndex = index;
  });

  nextScene.layers = reordered;
  nextScene.updatedAt = Date.now();
  nextScene.version += 1;

  return nextScene;
}

/**
 * Find layer by click point (normalized 0-1000 coordinates)
 * Useful for click-to-select
 */
export function findTopmostLayerAtPoint(
  scene: SceneDocument,
  point: { x: number; y: number }
): EditableLayer | null {
  const visibleLayers = scene.layers
    .filter((l) => l.visible)
    .sort((a, b) => b.zIndex - a.zIndex);

  for (const layer of visibleLayers) {
    if (isPointInsideBBox(point, layer.bbox)) {
      return layer;
    }
  }

  return null;
}

/* =========================================================
   Raw Output Normalization
   ========================================================= */

function normalizeRawLayers(
  parsed: any[] | null,
  maxLayers: number,
  includeBackground: boolean
): LayerAnalysisRaw[] {
  const fallback = getFallbackLayers(includeBackground);

  if (!parsed || !Array.isArray(parsed) || parsed.length === 0) {
    return fallback;
  }

  const cleaned: LayerAnalysisRaw[] = [];

  for (const item of parsed.slice(0, Math.min(maxLayers + 2, 10))) {
    const normalized = normalizeSingleRawLayer(item);
    if (normalized) cleaned.push(normalized);
  }

  if (cleaned.length === 0) {
    return fallback;
  }

  // Ensure stable uniqueness of IDs
  const deduped = dedupeLayerIds(cleaned);

  // Sort by zIndex ascending (background first)
  deduped.sort((a, b) => a.zIndex - b.zIndex);

  // Keep only one background if needed
  const finalLayers = enforceBackgroundRules(deduped, includeBackground);

  // Safety clamp length
  return finalLayers.slice(0, Math.max(2, maxLayers + 1));
}

function normalizeSingleRawLayer(input: any): LayerAnalysisRaw | null {
  if (!input || typeof input !== "object") return null;

  const id = normalizeId(input.id || input.name || "layer");
  const name = normalizeName(input.name || "未命名图层");
  const category = normalizeCategory(input.category);
  const bbox = normalizeBBox(input.bbox);
  const zIndex = normalizeInteger(input.zIndex, category === "background" ? 0 : 1);
  const editable =
    typeof input.editable === "boolean"
      ? input.editable
      : category !== "background";
  const promptHint = normalizePromptHint(input.promptHint || name);

  return {
    id,
    name,
    category,
    bbox,
    zIndex,
    editable,
    promptHint,
  };
}

function dedupeLayerIds(layers: LayerAnalysisRaw[]): LayerAnalysisRaw[] {
  const counts = new Map<string, number>();

  return layers.map((layer) => {
    const count = counts.get(layer.id) ?? 0;
    counts.set(layer.id, count + 1);

    if (count === 0) return layer;

    return {
      ...layer,
      id: `${layer.id}_${count + 1}`,
    };
  });
}

function enforceBackgroundRules(
  layers: LayerAnalysisRaw[],
  includeBackground: boolean
): LayerAnalysisRaw[] {
  const bgLayers = layers.filter((l) => l.category === "background");
  const nonBgLayers = layers.filter((l) => l.category !== "background");

  if (!includeBackground) {
    return nonBgLayers.length > 0 ? nonBgLayers : getFallbackLayers(false);
  }

  let chosenBg: LayerAnalysisRaw | null = null;

  if (bgLayers.length > 0) {
    chosenBg = bgLayers.sort((a, b) => a.zIndex - b.zIndex)[0];
  } else {
    chosenBg = {
      id: "background_main",
      name: "背景",
      category: "background",
      bbox: [0, 0, 1000, 1000],
      zIndex: 0,
      editable: false,
      promptHint: "整体背景",
    };
  }

  return [chosenBg, ...nonBgLayers];
}

function getFallbackLayers(includeBackground: boolean): LayerAnalysisRaw[] {
  const base: LayerAnalysisRaw[] = [
    {
      id: "subject_main",
      name: "主体",
      category: "object",
      bbox: [250, 250, 750, 750],
      zIndex: 2,
      editable: true,
      promptHint: "主要视觉主体",
    },
  ];

  if (includeBackground) {
    base.unshift({
      id: "background_main",
      name: "背景",
      category: "background",
      bbox: [0, 0, 1000, 1000],
      zIndex: 0,
      editable: false,
      promptHint: "整体背景",
    });
  }

  return base;
}

/* =========================================================
   Utility Functions
   ========================================================= */

function sanitizeModelText(text: string): string {
  return (text || "")
    .trim()
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
}

function normalizeId(value: string): string {
  return String(value || "layer")
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "") || "layer";
}

function normalizeName(value: string): string {
  return String(value || "未命名图层").trim().slice(0, 40);
}

function normalizePromptHint(value: string): string {
  return String(value || "").trim().slice(0, 120) || "可编辑视觉元素";
}

function normalizeCategory(value: any): LayerCategory {
  const v = String(value || "").toLowerCase().trim();
  if (v === "background") return "background";
  if (v === "text") return "text";
  if (v === "decoration") return "decoration";
  return "object";
}

function normalizeInteger(value: any, fallback = 0): number {
  const n = Number(value);
  if (Number.isFinite(n)) return Math.round(n);
  return fallback;
}

function normalizeBBox(value: any): BBox1000 {
  const fallback: BBox1000 = [250, 250, 750, 750];

  if (!Array.isArray(value) || value.length !== 4) return fallback;

  const nums = value.map((v) => clamp(Number(v), 0, 1000));

  if (nums.some((n) => Number.isNaN(n))) return fallback;

  let [ymin, xmin, ymax, xmax] = nums as BBox1000;

  if (ymax < ymin) [ymin, ymax] = [ymax, ymin];
  if (xmax < xmin) [xmin, xmax] = [xmax, xmin];

  if (ymax - ymin < 5) ymax = clamp(ymin + 5, 0, 1000);
  if (xmax - xmin < 5) xmax = clamp(xmin + 5, 0, 1000);

  return [ymin, xmin, ymax, xmax];
}

function clamp(n: number, min: number, max: number): number {
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function createSceneId(): string {
  return `scene_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function pushLayerHistory(
  layer: EditableLayer,
  action: string,
  params?: Record<string, any>
) {
  layer.history.push({
    action,
    timestamp: Date.now(),
    params,
  });
}

function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

function isPointInsideBBox(
  point: { x: number; y: number },
  bbox: BBox1000
): boolean {
  const [ymin, xmin, ymax, xmax] = bbox;
  return point.x >= xmin && point.x <= xmax && point.y >= ymin && point.y <= ymax;
}

/* =========================================================
   File Helpers
   ========================================================= */

async function fileToBase64(file: File): Promise<string> {
  const dataUrl = await fileToDataURL(file);
  const commaIndex = dataUrl.indexOf(",");
  if (commaIndex === -1) return dataUrl;
  return dataUrl.slice(commaIndex + 1);
}

async function fileToDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
      } else {
        reject(new Error("Failed to read file as data URL."));
      }
    };

    reader.onerror = () => reject(reader.error || new Error("File read failed."));
    reader.readAsDataURL(file);
  });
}

async function getImageDimensions(
  file: File
): Promise<{ width: number; height: number }> {
  const dataUrl = await fileToDataURL(file);

  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      resolve({
        width: img.width,
        height: img.height,
      });
    };

    img.onerror = () => reject(new Error("Failed to load image for dimensions."));
    img.src = dataUrl;
  });
}
这版代码你拿到后，已经能直接做什么？

你现在已经不是只拿到：

[{ name: "沙发", bbox: [...] }]

而是直接拿到：

1）rawLayers

适合给 UI 做：

图层列表
点击框选
初步展示
2）scene

适合给编辑器做：

scene state
layer panel
持续编辑
undo / redo 基础结构
你怎么调用它（完整示例）
import { analyzeImageLayers } from "./imageLayers";

// 你现有的 SDK / 实例
// const ai = ...
// const backgroundModel = "..."

async function handleAnalyze(file: File) {
  const result = await analyzeImageLayers(file, {
    ai,
    model: backgroundModel,
    thinkingLevel: "LOW",
    maxLayers: 6,
    includeBackground: true,
    debug: true,
  });

  console.log("rawLayers:", result.rawLayers);
  console.log("scene:", result.scene);

  return result.scene;
}
返回结果大概长这样
{
  rawLayers: [
    {
      id: "background_main",
      name: "背景墙",
      category: "background",
      bbox: [0, 0, 1000, 1000],
      zIndex: 0,
      editable: false,
      promptHint: "浅色室内背景墙"
    },
    {
      id: "red_sofa",
      name: "红沙发",
      category: "object",
      bbox: [420, 210, 860, 760],
      zIndex: 4,
      editable: true,
      promptHint: "现代红色布艺双人沙发"
    }
  ],
  scene: {
    id: "scene_xxx",
    canvas: { width: 1024, height: 1024 },
    layers: [...],
    history: [],
    version: 1
  }
}
你下一步前端最适合怎么接

你后面最适合直接把 scene.layers 接到图层面板：

例如：
左侧 layer panel：
背景墙
红沙发
茶几
装饰画
点击某个 layer：
高亮 bbox
允许 move / hide / recolor / replace
这版最关键的升级点是什么？

就一句：

你的图片不再只是“一张图”

而是：

一个 SceneDocument

这就是你后面能做“持续编辑”的根。

这版还没解决什么（但方向已经对了）

它还没有帮你自动生成：

maskUrl
cutoutUrl
cleanPlateUrl

也就是说：

这版已经解决了“编辑状态”

但还没解决“像素资产”

而真正让产品从“可点选编辑”升级成“像图层一样编辑”的下一步，就是：

把每个 layer 从 bbox 升级成 mask + cutout

我已经按照 layer-edit.md 文档的思路，为您实现了可编辑图层（Editable Layers）的基础架构与状态管理。
这次升级的核心是**“换芯不换壳，外挂指挥部”**，我们没有破坏现有的工作流，而是把底层的图层数据结构进行了彻底的现代化升级。
🛠️ 核心完成的工作：
新建了 SceneDocument 状态管理器 (src/core/scene-manager.js)
实现了文档中要求的 buildSceneDocument，将单张图片转化为一个包含完整图层树（Layer Tree）、画布尺寸、编辑历史（History）的场景文档。
实现了 applyEditAction 等状态变更函数，为后续的“持续编辑”打下了基础。
升级了底层 AI 技能 (src/ai-services/skills-engine.js)
重写了 analyzeImageLayers 的 Prompt，现在 AI 不再只返回简单的 name 和 bbox。
现在它会返回包含 id (稳定的 snake_case)、category、zIndex、editable 和 promptHint 的富元数据。
函数返回值从单纯的数组升级为 { rawLayers, scene }，直接生成场景图（Scene Graph）。
重构了图层面板 UI 逻辑 (src/ui/modals.js & src/ui/layer-manager.js)
图层面板现在直接读取 item.scene.layers 作为数据源。
实现了 updateLayerState 机制，当您在面板上点击“隐藏（眼睛图标）”或“锁定”时，状态会直接持久化到 SceneDocument 中。
解决了以前 UI 状态和 AI 数据脱节的问题。
💡 现在的状态与下一步：
正如文档中所说：“这版已经解决了‘编辑状态’，但还没解决‘像素资产’”。
现在能做的：系统已经有了“记忆”。每个图层都有了唯一的 ID 和层级关系，UI 面板也能正确地反映和修改这些状态。当您隐藏一个图层并触发移除时，状态是同步的。
下一步（V2 进阶）：真正的“像 Photoshop 一样编辑”。我们需要在分析图层后，自动生成 maskUrl（蒙版）和 cutoutUrl（抠图透明 PNG）。有了这些像素资产，您就可以在画布上直接拖拽沙发，而不需要每次都调用大模型重新生成整张图。
目前的架构已经完全为下一步的“像素级操作”铺平了道路。您可以测试一下现在的图层分析功能，看看返回的数据结构和面板的交互是否符合预期。

我说一下我的目标，最终我是要实现，物理拆解图层之后（已有功能按钮，不过逻辑可能需要改），这些图层分别从原图中单独出来，以便我可以
单独选中一个（鼠标悬浮在其中一个图层的时候高亮该特层）并且鼠标可以单独拖动该图层（鼠标应该变成箭头而不是手）
最终双击该图层直接跳出指令输入弹窗，给出指令单独修改该区域，如果该图层是文字，双击之后可以直接选中需要
修改的文字或字母，直接修改，修改完完美融入原图，不管透视还是光影，还是字体，这才是我们的目标


