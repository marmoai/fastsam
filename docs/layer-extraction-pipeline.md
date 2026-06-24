# 语义分析与混合抠图图层提取管线 (Hybrid Matting Pipeline) 架构文档

本文档详细梳理了当前系统中涉及抠图及AI图层处理的 API 调用位置，以及从“图像语义分析”到“单图层独立提取”的完整技术流转。系统采用的架构为 **“云端AI重绘 + 本地纯色键控解算”** 的混合抠图管线。

## 核心架构演变：从单线 API 到多策略智能调度分流系统 (Strategy Dispatcher)

系统当前的极客架构已经超越了单纯的“全量塞给大模型抠图”，而是演进成为了一个 **AI + CV 融合重建引擎**。它的核心思想不是“如何用算法完全替代 Gemini”，而是 **“如何根据图像局部特征进行动态的路径阻断与分流”**。


### 架构拓扑
```text
            ┌──────────────┐
            │  图像输入     │
            └──────┬───────┘
                   │
        ┌──────────▼──────────┐
        │ Strategy Dispatcher │  <-- 智能场景分析与判定分流
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
     CV路径                Hybrid路径
 (纯算法秒抠)         (AI换底 + 本地解算)
```


这种双轨策略分流带来了显著的工程优势与逻辑闭环：
1. **防范降效 (CV路径)**：将电商白底、强对比边缘物体分配给纯 CV 计算，可实现毫秒级提取并直接绕过大模型网络延迟。
2. **拒绝“蒙版幻觉” (强化Hybrid路径)**：大模型本质是生成模型，而非严格的像素级分割模型。直接让AI去画边缘（输出蒙版）会导致极其不可靠和模糊的边缘。因此我们**彻底废弃了原有的AI直接出Alpha的路径**，在面对复杂混叠颜色场景时（如狗腿与地毯同色），不是放弃Hybrid，而是**更坚定地依赖Hybrid生成式绿幕替换能力**，通过AI换底强行分离物理语义。
3. **保留超解空间与多色防缠绕**：剥离抠像和重构逻辑，AI只负责改变环境背景像素甚至提升对象清晰度，将边缘计算交给精准高效的数学解算进程。

---

## 摘要：核心 API 与模型调用归属

目前系统综合使用了多个模型和本地算法实现提取，并非单一依赖某个特定抠图接口。

### 1. Gemini 多模态大模型 (Google GenAI API)
大模型主要负责理解和生成式重绘（Generative Reconstruction）。
* **位置：** `src/ai-services/skills-engine.js` -> `analyzeImageLayers`
  * **作用：** **层级结构理解**。输入整图，让 AI 不仅做目标检测，还能基于室内设计或通用场景输出每个物体的层级、遮挡关系、类别以及 Bounding Box（边界框）。
* **位置：** `src/ui/modals.js` -> `performLayerAction`
  * **作用：** **生成式纯色背景替换（抠图核心步骤1）**。在提取单图层时，让 Gemini 将物体的背景完美替换为本地计算出的高对比度纯色（如纯绿、部分特定对象的纯黑等），并在必要时通过 AI 脑补还原被遮挡的边缘。
* **位置：** `src/ui/workbench/layer-assets.js` -> `cleanBackground`
  * **作用：** **底板净化（Inpainting）**。提取物体后，调用 Gemini 将该物体从原图中擦除，并根据该物体周围环境完美补全缺失的背景（如补全地板花纹、墙面光影等）。
* **位置：** `src/ui/layer-manager.js` -> `triggerLayerExplosion` (批量拆解)
  * **作用：** 批量图层提取的核心控制器。并发控制大模型处理所有图块，并且目前已完全同步使用 V5 Hybrid Matte Pipeline（与单图层独立提取管线保持物理一致性），进行动态底板计算、像素还原投影等功能。

### 2. In-house Matte Task System (本地 Worker 线程绿幕解算)
系统内置的纯本地、无网络调用的前端数学解算引擎。
* **位置：** `src/graphics/matte-task-system.js` & `src/graphics/matte.worker.js`
  * **作用：** **高保真边缘提取（抠图核心步骤2）**。接收 Gemini 吐出的“带背景纯色图”，利用 Web Worker 通过三维欧几里得色彩距离算法和边缘 Gamma 软羽化曲线，进行抗锯齿去底操作，保留图片高频细节。

---

## 全链路状态流转：从【语义分析】到【单图层独立提取】

以下详细记录一张平面图片（如室内场景）进入“语义图层分析”，并触发“提取圆凳”这一动作背后的完整技术数据流转。

### Phase 1: 语义结构分析 (Semantic Analysis)
**目标：将具有透视视角的单张平面图，结构化解析为带深度信息的 2D 树形层级。**

1. **图片预处理**：将原始主图转换为 Base64，同步获取图片的宽高数值。
2. **构建 Prompt 上下文**：向大模型下发具有高度约束的伪参数化指令格式，强制要求其构建类 PSD 结构的设计层结构图，限制找 3-8 个最有重组/拆解价值的对象。
3. **AI 推理执行**：大模型结构化提取以下对象并标记：
   * `bbox`（边界框）: 物体包围盒的归一化坐标数组 `[ymin, xmin, ymax, xmax]`
   * `layerType`（图层基础类）: 解析并归还 `foreground_asset`（可复用资产）或 `background_plate`（环境背景盘）类型。
   * `zIndex`: 根据空间远近透视规则估算的视觉遮挡深度优先级。
4. **场景实例化 (Scene Instantiation)**：收到 JSON 后解析并在工作台中绑定建立图层对象 `SceneDocument`，UI 开始映射对应边界框 `bbox` 并显示悬浮挂载组件。

### Phase 2: 独立图层提取重计算 (Independent Layer Extraction)
**目标：无损剥离对象，并使用周围自然属性修补原视图。**
*(业务动作钩子位于：层双击 -> `src/ui/modals.js` -> `performLayerAction`)*

**Step 2.1: 局部裁切与上下文扩充 (Crop & Local-Context Padding)**
* 读取第 1 阶段中的 `bbox` 物理坐标，反向映射至高清图片原像素 `img.naturalWidth / naturalHeight` 上。
* **扩展边距（Pad Context）**：向周围额外切出 `15%` 的填充边距（Padding），主要为了保留周围的接缝光影并帮助大模型理解该物体位于何种地面或墙面上。

**Step 2.2: AI 超分辨率放大补偿 (AI Super-Resolution Upscaling)**
* **原理**：直接将低重比（如 100x100px）的内容切图提交给大模型重构会导致模型由于处理尺度过小而直接“马赛克化”涂抹掉局部固有细节。
* **操作**：通过判断物理尺寸 `maxDim`，若截图最大边长低于 `1024px`，将计算上采样缩放比例 `scaleFactor`，在原 Canvas 上调用最高质量的原生 `imageSmoothingQuality = high`，将其扩缩容至 `1024x1024` 基准边界池中进行超分辨脑补发送（`hrW, hrH` 参数化锁定）。

**Step 2.3: 动态色键策略计算 (Dynamic Backdrop Decision Algorithm)**
* 于刚才生成好的高清缓冲池内做一次截面 60% 边界内缩小像素采用（防止干扰和抗边缘杂色）。
* **加权特征计算**：得到物体的总体基色 `Avg RGB` （排除极白极黑色）。
* **Euclidean Color Matrix（欧几里德色彩矩阵计算）**：携带主体基色，在预置色彩字典（如纯绿 `#00FF00`、洋红 `#FF00FF`、蓝 `#0000FF` 等）中遍历比较，通过公式：`Distance = sqrt(ΔR² + ΔG² + ΔB²)` 选出在三原色色相环中偏差最大的互补作为背景填充色。

**Step 2.4: 生成模型定向重塑输出 (Generative Build Request)**
* 以“严格禁止改变主体自身”为强前提下达 Prompt: *“ZERO HALLUCINATION: You MUST NOT reconstruct, redraw, or alter the object... Your ONLY task is to change the background to mathematically solid X.”*
* *例外拦截：* 若检测为云、雾状粒子流类名称，不注入纯背景提示，改要求直接返回 Alpha Mask 深度通道掩码黑白图。
* 在返回被处理图后依然锁死前面放大时用的长宽度比例（`hrW, hrH`），绘制进入缓冲区 `reconstructedImg` 以避免二次 AI 生成拉伸造成的像素破裂。

**Step 2.5: Worker 高能态解算与无损印刻投射 (Matte Worker Math Process & Mask Projection)**
* **解算层 Worker**：将包含绿幕的像素 Uint8ClampedArray 发送到子线程，经过硬击杀（Hard Kill 斩断特定RGB容差）、软边缘 Gamma 渗透和 Spill 去环境色污染后，返还一套只包含处理好透明度的 `Alpha` 数组。
* **物理像素高保真锁 (Mask Projection)**：对于边界清晰的物理实态物体，代码并不直接采用 Worker 计算后的图片像素（因为这组 RGB 来源于 Gemini 重构结果），而是进行**蒙版反向穿透映射**：
  * `originalImageDataObj.data[i + 3] = processedAlphaData[i + 3];` （强行将计算好的干净 Alpha 贴在用户未经任何压缩裁剪下来的极致原像素 `original` 通道上）。
* 确保截取输出 `transparentDataUrl` 无比保真，完全还原物理原分辨率毛孔和材质。

**Step 2.6: 底板镜像净化 (Background Plate Inpainting & Recomposition)**
* 获取独立资产结束后，后台系统紧接着用原始图发送到另一条管道 `cleanBackground(layer-assets.js)` 进行“魔术橡皮擦”。
* AI 将填平挖走的大型物体。
* 保存新的大图赋值给 `cleanPlateDataUrl` 成为新底板，该目标 Layer 组件自动销毁原边界框并转换为一个漂浮的、带 `cutoutUrl` Src 指针的独立可移动图片。
* 拆卸、重塑循环完美闭环。
