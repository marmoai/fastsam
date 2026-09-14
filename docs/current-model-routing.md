# 当前模型与推理路由说明

> 更新时间：2026-08-26
>
> 本文以当前代码为准，不以旧的模型能力文档、历史实验记录或供应商宣传页为准。
> 模型名称、供应商接口和默认路由可能变化；修改模型后应同步更新本文。

## 1. 先看结论

MarmoAid 不是一个“选择一个模型后所有事情都由它完成”的结构，而是由几类模型协同工作：

1. **Gemini 文本/视觉理解层**：理解用户指令、读取 Workbench 状态、意图分类、流程规划、OCR、图层语义分析和视觉搜索。
2. **图片生成/编辑层**：根据当前选择执行生图、整图编辑、局部编辑或图层资产生成。
3. **分割与抠图层**：由本地或云端的 SAM、FastSAM、Matte/CV 负责生成透明图层和边界。
4. **视频层**：由 SiliconFlow Wan 或旧的 Gemini Veo 路径负责视频生成。

当前最容易混淆的一点是：

- 普通 ChatPanel 对话和大多数解释器/意图链路固定使用 `gemini-3.6-flash`。
- 选择 Qwen 图片模型时，最终图片仍由 Qwen 执行，但当前 `ENABLE_QWEN_LOCAL_ROUTING_BYPASS = false`，因此图片任务会先经过 Gemini 解释器和意图分类链路。
- 选择菜单中的 `flash`、`pro`、`lite`，主要影响图片模型；它们在 `MODEL_SUITES` 中虽然各自保存了 `text` 字段，但当前 `getTextModel()` 固定返回 `gemini-3.6-flash`，不会按套装的 `text` 字段切换普通聊天模型。

## 2. 用户可选模型

模型选择器位于 `index.html` 的 `#modelSelect`。以下是当前菜单中的选项，以及真实执行方式。

### 2.1 `gpt-image-2`

**模型用途**

- 高质量文生图。
- 基于原图的图片编辑。
- 有原图和蒙版时执行局部编辑或局部重绘。
- 可用于普通图片生成、精准修改、重光照等图片任务。

**当前路由**

- 前端通过 `src/ai-services/skills-engine.js` 请求后端。
- 后端 `index.js` 转发到 QuickRouter：
  - 生成：`/v1/images/generations`
  - 编辑：`/v1/images/edits`
- 当前域名是 `https://api.quickrouter.us`。
- 认证变量：`QUICKROUTER_API_TOKEN`。

**适合任务**

- 追求生成质量的整图创作。
- 需要参考图、蒙版或较复杂画面重绘的任务。
- 未拆分图片的整图局部重绘。

**边界**

- 它不负责识别 Magic Layers 图层，也不负责维护 Workbench 的版本、位置和层级。
- 图层原位替换仍由前端的资产版本逻辑、Matte/SAM 和 Workbench 完成。
- 模型返回的图像比例和透视并不等于严格的几何变换，必须通过图层回贴流程约束。

### 2.2 `qwen-image-3.0-pro`

**模型用途**

- Qwen Image 3.0 Pro 图片生成和图片编辑。
- 当前可以处理纯文本生图，也可以接收原图、参考图和蒙版。
- 可用于清理背景、局部修改、素材生成和图层资产编辑。

**当前路由**

- 前端函数：`callQwenImageRouter()`。
- 后端函数：`callDashscopeQwenImage()`。
- 供应商：阿里云 DashScope。
- 默认模型：`qwen-image-3.0-pro`。
- 默认接口：`DASHSCOPE_API_ENDPOINT`，默认值为 DashScope 多模态生成接口。
- 认证变量：`DASHSCOPE_API_KEY`。
- 模型别名 `qwen-image-3.0` 会被后端归一化到 `qwen-image-3.0-pro`。

**适合任务**

- 中文指令图片编辑。
- 室内设计场景改造和家具/材质生成。
- 需要参考图、原图或遮罩的视觉编辑。
- 纯文本生成室内图、广告图或概念图。

**当前解释链路**

- 当前 `src/ai-services/qwen-local-routing.js` 中：
  `ENABLE_QWEN_LOCAL_ROUTING_BYPASS = false`。
- 因此显式 Qwen 图片任务不会直接跳过解释器。
- ChatPanel 会先通过 Gemini 3.6 Flash 完成解释、意图分类和任务调度，最终图片请求再发给 Qwen Image 3.0 Pro。
- 这不意味着 Gemini 负责生图；Gemini 负责理解，Qwen 负责图片执行。

**边界**

- Qwen 返回结果的尺寸、透视和主体边界仍可能需要前端归一化。
- 透明图层替换不是模型单独完成的能力，仍需要 Matte、必要时 SAM，以及原位版本替换逻辑。
- DashScope 的请求格式和 QuickRouter/Gemini 格式不同，不能直接用 Gemini 的请求体测试。

### 2.3 `qwen-image-edit-2509`

**模型用途**

- Qwen Image Edit 2509 的图片编辑和图片生成兼容路由。
- 当前不再仅限于背景净化；选择该模型后，所有进入图片执行层的相关任务都可以使用它。

**当前路由**

- 前端函数：`callQwenImageEdit2509Router()`。
- 后端通过 `SILICONFLOW_IMAGE_API_ENDPOINT` 转发到 SiliconFlow 图片接口。
- 默认模型：`Qwen/Qwen-Image-Edit-2509`。
- 认证变量：`SILICONFLOW_IMAGE_EDIT_API_TOKEN`，如果未配置则兼容读取 `SILICONFLOW_API_TOKEN`。
- 默认接口：`https://api.siliconflow.cn/v1/images/generations`，具体可由 `SILICONFLOW_IMAGE_API_ENDPOINT` 覆盖。

**适合任务**

- 图片编辑、局部修改、材质替换和参考图编辑。
- 已拆分图层的独立资产生成。
- 未拆分图片的整图编辑。
- 背景净化和遮挡补全。

**边界**

- 该模型不会自动知道 Workbench 的图层身份、父子关系和目标层级。
- “同步到当前图层”由前端判断 `replaceCurrent`，然后将结果写入父图层版本。
- 结果是否干净，取决于生成结果、Matte/SAM 质量和回贴尺寸，而不是模型名称本身。

### 2.4 `flash`

**菜单显示**

- 当前 UI 文案仍是 `Gemini 3 Flash/Banana 2`，这是历史命名，不能当作精确的当前版本说明。

**当前图片模型**

- `gemini-3.1-flash-image-preview`。
- 通过 `gemini_lite_generate` 路由转发到 QuickRouter 的 Gemini `generateContent` 接口。

**当前文本模型**

- 普通对话实际固定为 `gemini-3.6-flash`。
- `MODEL_SUITES.flash.text` 中的 `gemini-3-flash-preview` 是旧套装元数据，目前不控制 `getTextModel()`。

**适合任务**

- 速度优先的普通对话和视觉理解。
- Gemini 图片生成/编辑能力可用时的快速图片处理。
- OCR、图像描述、建议生成等日常任务，具体后台任务通常由 Lite 固定模型承担。

### 2.5 `pro`

**当前图片模型**

- `gemini-3-pro-image-preview`。
- 通过 QuickRouter Gemini `generateContent` 路由执行。

**当前文本模型**

- 普通聊天实际仍为 `gemini-3.6-flash`，不是 `MODEL_SUITES.pro.text` 中记录的旧 `gemini-3.1-pro-preview`。

**适合任务**

- 需要更高图片质量或更复杂视觉编辑的 Gemini 图片任务。
- 复杂图片生成、风格控制和较高质量的场景编辑。

**边界**

- 不能把它理解成“所有任务都切到 Pro 文本推理”。当前普通 ChatPanel 的文本模型由全局常量控制。

### 2.6 `lite`

**当前图片模型**

- `gemini-2.5-flash-image`。

**当前后台文本/视觉模型**

- `backgroundModel = gemini-3.1-flash-lite`。
- 该模型主要用于成本敏感的后台任务，而不是普通 ChatPanel 的主对话。

**适合任务**

- 会话标题生成。
- 智能建议。
- 图片分类。
- 物体名称识别。
- 视觉搜索前的商品识别和搜索词生成。
- 一些代理分析和辅助识别任务。

**边界**

- Lite 不是当前普通聊天的默认文本模型。
- Lite 的视觉结果适合结构化辅助任务，不应直接等同于高精度 Magic Layers 语义拆层结果。

## 3. 当前固定的 Gemini 模型

### 3.1 `gemini-3.6-flash`

这是当前主文本/视觉理解模型，定义在 `src/ai-services/gemini-client.js`：

- `CHAT_TEXT_MODEL`
- `CONTINUITY_ANCHOR_MODEL`
- `getTextModel()` 的返回值
- `skills-engine.js` 非 Lite 场景下的语义模型回退

**主要任务**

- ChatPanel 普通对话。
- Workbench 状态理解。
- 对话解释器。
- 意图分类器。
- 图片任务的前置理解和调度。
- 流程图规划 `planGraph()`。
- 连续性视觉锚点。
- 部分 OCR 和语义分析。
- Magic Layers 图层语义识别的非 Lite 路径。

**当前供应商路径**

- 前端通过 `proxyGenerateContent()` 将 `gemini_lite_generate` 请求发送到后端。
- 后端 `index.js` 使用 QuickRouter：
  `https://api.quickrouter.us/v1beta/models/{model}:generateContent`。
- 认证变量：`QUICKROUTER_API_TOKEN`。

**重要说明**

- 当前模型切换不会改变图片执行模型。
- 例如选择 Qwen Image 3.0 Pro 时，Gemini 3.6 Flash 负责理解，Qwen Image 3.0 Pro 负责生成/编辑。
- 如果 Gemini 链路再次维护，不能只改后端域名；需要确认前端是否还要关闭 Qwen 本地直通开关。

### 3.2 `gemini-3.1-flash-lite`

定义为 `backgroundModel`，用于低成本后台视觉/文本任务。

**主要任务**

- 智能建议。
- 会话标题。
- 图片类别判断。
- 目标物体名称识别。
- 视觉搜索商品分析。
- 视觉搜索查询词和网页搜索辅助。
- 代理类轻量分析。

另外，`src/ui/predictive-prompt.js` 的输入补全也显式使用该模型。

### 3.3 `gemini-3.1-pro-preview` 和其他旧套装文本字段

这些名称仍存在于 `MODEL_SUITES` 的套装元数据中，主要用于兼容旧配置和显示/规划语义，**不是当前 `getTextModel()` 的实际返回值**。

不要仅凭以下字段判断运行时模型：

```js
MODEL_SUITES.pro.text
MODEL_SUITES.flash.text
MODEL_SUITES.qwen-image-3.0-pro.text
```

实际应检查：

```js
getTextModel()
CHAT_TEXT_MODEL
CONTINUITY_ANCHOR_MODEL
backgroundModel
```

## 4. Gemini 图片模型与图片任务

Gemini 图片模型通过 `gemini_lite_generate` 发送，当前主要包括：

| 模型 | 主要用途 | 当前入口 |
|---|---|---|
| `gemini-3.1-flash-image-preview` | 快速图片生成/编辑 | `flash` 套装 |
| `gemini-3-pro-image-preview` | 高质量图片生成/编辑 | `pro` 套装 |
| `gemini-2.5-flash-image` | 低成本/兼容图片任务 | `lite` 套装 |
| `imagen-4.0-generate-001` | 已知图片路由不匹配时的旧回退生图 | `generateImage()` 最后回退 |

当前 `generateImage()` 的优先级大致为：

1. Qwen Image Edit 2509。
2. Qwen Image 3.0 Pro。
3. GPT-image-2。
4. Gemini 图片模型。
5. Imagen 4 回退。

这不是所有图片操作的唯一顺序，因为精准编辑、重光照、图层编辑和工作流节点会根据任务走不同函数，但它说明了图片模型选择和文本模型选择是两套独立逻辑。

## 5. 图片分割、抠图和透明资产模型

这些不是 ChatPanel 的对话模型，也不是用户菜单中的生图模型。

### 5.1 SAM-B

- 权重：`sam_b.pt`。
- 后端：`fastsam-backend/main.py`，通过 Ultralytics `SAM` 加载。
- 默认高精度分割模型。
- 输入通常是图片加 bbox、点或蒙版。
- 输出透明 cutout、mask 和质量信息。

**适合任务**

- 从 Magic Layers 语义 bbox 中恢复真实物体轮廓。
- 硬边实体抠图。
- 对生成后的图层进行透明边界检查。
- 对本地 Matte 结果进行边界复核。

### 5.2 SAM-L

- 权重：`sam_l.pt`。
- 是 SAM-B 的高质量/高成本升级路径。
- 后端一次只保留一个高精度 SAM 变体，B/L 切换时会释放另一个模型。

**触发方式**

- SAM-B 结果内部空洞、填充困难、边界质量不足时升级。
- 大图、复杂家具、多部件实体或困难填充任务触发。
- 具体标准由 `fastsam-backend/main.py` 中的质量评分、困难填充和 B/L arbitration 逻辑决定。

**边界**

- SAM-L 不是语义识别模型。它不会可靠地理解“这是沙发还是底座”，主要依赖 bbox、点、mask 和前置语义提示。
- SAM-L 也不会主动决定 Workbench 图层名称或版本关系。

### 5.3 FastSAM-x

- 权重：`FastSAM-x.pt`。
- 使用 Ultralytics `FastSAM`。
- 后端仍保留 FastSAM 自动分割和测试接口。

**当前状态**

- FastSAM 是本地/云端分割后端能力的一部分。
- 但当前前端 `normalizeSegmentationEngine()` 会将可用分割引擎归一到 `sam`，主 Magic Layers 路径主要显示为高精度 SAM。
- 不应把 FastSAM 自动候选结果理解为已经具备大模型级物体语义。

### 5.4 本地 Matte/CV

这不是 SAM，也不是 YOLO。

- 代码位置：`src/graphics/matte-engine.js`、`matte.worker.js`、`matte-task-system.js`。
- 主要是浏览器 Worker 中执行的颜色距离、Alpha 推断、边缘处理、despill 等本地算法。
- `StrategyDispatcher` 会根据主体和背景特征选择纯 CV 快速路径或更复杂的处理路径。

**适合任务**

- 生成结果带纯色背景时快速去背。
- 已经有较可靠 Alpha 时直接透传或轻量清理。
- 低延迟的实体图层提取。

**边界**

- 主体颜色接近背景色时容易出现漏抠、误抠或色溢出。
- 半透明、毛发、羽毛、玻璃和复杂阴影不应只依赖 Matte。
- 当前流程会在 Alpha 不可靠时再请求 SAM 复核，但 SAM 是质量补救，不是本地 Matte 的组成部分。

### 5.5 YOLO

当前仓库中没有可作为主生产路由的独立 YOLO 检测/实例分割模型配置。Ultralytics 依赖和 `YOLO_CONFIG_DIR` 的日志主要来自 FastSAM/SAM 后端运行环境。

因此当前不能描述为：

```text
YOLO 识别语义 -> SAM 分割 -> Magic Layers
```

当前更准确的描述是：

```text
Gemini 语义识别/生成 bbox -> SAM 或 FastSAM 轮廓分割 -> Matte/质量门控 -> runtime asset
```

如果以后重新接入 YOLO，需要单独增加模型权重、类别映射、实例 mask 合并规则和失败处理，不能只在文档中把 FastSAM 改名为 YOLO。

## 6. 视频模型

### 6.1 SiliconFlow Wan

- 默认模型：`Wan-AI/Wan2.2-I2V-A14B`。
- 代码：`src/ai-services/siliconflow-video.js`。
- 后端提交接口：`SILICONFLOW_VIDEO_SUBMIT_ENDPOINT`。
- 后端状态接口：`SILICONFLOW_VIDEO_STATUS_ENDPOINT`。
- Token：`SILICONFLOW_VIDEO_EDIT_API_TOKEN`。

**主要任务**

- 静态图片转短视频。
- Magic Motion 预览。
- 根据图片比例选择 `1280x720`、`720x1280` 或 `960x960`。
- 生成任务采用提交任务、轮询状态、取得视频 URL 的异步流程。

**当前 Magic Layers 关系**

- 系统已经能把语义图层转换成 motion-ready layers，并在前端生成动效预览。
- 这主要是图层 CSS/Canvas 动效与 Wan 视频生成的组合。
- 还不是逐帧视频拆层、对象跟踪或视频局部替换系统。

### 6.2 Gemini Veo 3.1

- 代码仍存在 `generateVeoVideo()`。
- 模型：`veo-3.1-generate-preview`。
- 通过 Google GenAI SDK 直接调用。

**当前状态**

- 属于保留的旧/备用视频路径。
- 当前 Magic Motion 主路径使用 SiliconFlow Wan 的提交和轮询接口。
- 不应把 Veo 作为当前默认视频执行模型，除非调用方明确进入该函数。

## 7. 遗留 API_TOKEN 路由

后端 `index.js` 仍有两个旧接口：

- `mode: recognize`
- `mode: chat`

这两个接口使用：

- Token：`API_TOKEN`。
- 地址：`API_ENDPOINT`，默认 SiliconFlow Chat Completions。
- 硬编码模型：`Qwen/Qwen3-VL-32B-Instruct`。

**当前定位**

- 这是早期室内识别/物品问答接口的遗留路由。
- 当前 ChatPanel 的主对话不是通过这两个接口，而是通过 `gemini_lite_generate` 和 QuickRouter Gemini 链路。
- 该模型没有被当前主 ChatPanel 路由升级为 Gemini 3.6 Flash 的替代品。
- 如果后续要删除或改造，必须先确认是否还有外部客户端调用 `recognize` 或 `chat`，不能仅凭前端搜索结果删除。

## 8. 常见任务到底由谁完成

| 用户任务 | 理解/规划 | 最终执行 | 后处理 |
|---|---|---|---|
| 普通聊天 | Gemini 3.6 Flash | 无图片执行模型 | 无 |
| “生成一张客厅图” | Gemini 3.6 Flash | 当前选择的图片模型 | 结果进入 Workbench |
| 选择 Qwen 3.0 生成图片 | Gemini 3.6 Flash | Qwen Image 3.0 Pro | 尺寸/结果归一化 |
| 选择 Qwen Edit 修改图片 | Gemini 3.6 Flash | Qwen Image Edit 2509 | 可能经过 Matte/SAM |
| 选择 GPT-image-2 编辑图片 | Gemini 3.6 Flash | GPT-image-2 | 后端 QuickRouter 返回结果 |
| Magic Layers 语义拆层 | Gemini 3.6 Flash 或 Lite 场景模型 | 无图片生成模型 | 生成层定义和 bbox |
| 语义图层透明提取 | 已有语义 bbox | SAM-B/SAM-L 或 FastSAM | mask、cutout、质量门控 |
| 生成资产去背景 | 前置图层语义 | 本地 Matte/CV | Alpha 清理，必要时 SAM |
| Lens 搜索同款 | Lite 视觉模型 | Google Search grounding/网页链接 | Lens 面板展示 |
| 清除背景/文字 | Gemini 解释与任务构造 | 当前图片编辑模型 | clean plate、图层回写 |
| 静态图转视频 | 当前任务提示词 | SiliconFlow Wan | 异步轮询视频结果 |

## 9. 环境变量与接口速查

### Gemini/QuickRouter

- `QUICKROUTER_API_TOKEN`：QuickRouter Gemini、GPT-image-2 使用。
- `api.quickrouter.us`：当前 QuickRouter 域名。
- Gemini 文本接口：`/v1beta/models/{model}:generateContent`。
- GPT 图片生成：`/v1/images/generations`。
- GPT 图片编辑：`/v1/images/edits`。

### Qwen Image

- `DASHSCOPE_API_KEY`：Qwen Image 3.0 Pro。
- `DASHSCOPE_API_ENDPOINT`：DashScope 图片接口，可覆盖默认地址。
- `SILICONFLOW_IMAGE_EDIT_API_TOKEN`：Qwen Image Edit 2509。
- `SILICONFLOW_IMAGE_API_ENDPOINT`：Qwen Image Edit 2509 图片接口。
- `SILICONFLOW_IMAGE_EDIT_MODEL`：默认 `Qwen/Qwen-Image-Edit-2509`。

### 视频

- `SILICONFLOW_VIDEO_EDIT_API_TOKEN`：Wan 视频接口。
- `SILICONFLOW_VIDEO_SUBMIT_ENDPOINT`：Wan 提交接口。
- `SILICONFLOW_VIDEO_STATUS_ENDPOINT`：Wan 状态接口。
- `SILICONFLOW_VIDEO_MODEL`：默认 `Wan-AI/Wan2.2-I2V-A14B`。

### 旧 SiliconFlow 对话接口

- `API_TOKEN`：仅对应旧 `recognize`/`chat` 路由，默认模型为 `Qwen/Qwen3-VL-32B-Instruct`。
- 不要把 `API_TOKEN` 与 `QUICKROUTER_API_TOKEN` 混用。

### 分割后端

- `FASTSAM_CLOUD_ENDPOINT`：后端 FastSAM/SAM 云端代理地址。
- `FASTSAM_CLOUD_TOKEN`：云端分割代理 Token。
- `SAM_MODEL_PATH` / `SAM_B_MODEL_PATH`：SAM-B 权重。
- `SAM_L_MODEL_PATH`：SAM-L 权重。
- `FASTSAM_MODEL_PATH`：FastSAM-x 权重。
- `MODEL_BASE_URL`、`SAM_B_MODEL_URL`、`SAM_L_MODEL_URL`、`FASTSAM_MODEL_URL`：权重下载地址配置。

## 10. 测试时如何判断实际调用了哪个模型

不要只看前端菜单。建议同时看浏览器和后端日志：

### 普通聊天或解释器

应看到：

```text
正在转发 Gemini Lite 请求到 QuickRouter...
model=gemini-3.6-flash
```

### Qwen Image 3.0 Pro

应看到：

```text
正在转发 Qwen Image 请求到 DashScope: model=qwen-image-3.0-pro
```

### Qwen Image Edit 2509

应看到：

```text
正在转发图片任务到 SiliconFlow Qwen Image Edit
model=Qwen/Qwen-Image-Edit-2509
```

### GPT-image-2

应看到：

```text
正在转发图片生成请求到 GPT-image-2 (QuickRouter)...
```

或：

```text
正在转发图片编辑请求到 GPT-image-2 (QuickRouter)...
```

### SAM/SAM-L

分割后端应出现类似：

```text
Loading high precision SAM-B model
```

或：

```text
SAM route ... model=B -> L
Loading high precision SAM-L model
```

### Qwen 本地直通开关

当前应为：

```js
ENABLE_QWEN_LOCAL_ROUTING_BYPASS = false
```

这表示 Qwen 图片任务不会绕过 Gemini 解释器/意图分类器。

## 11. 当前已知边界和维护规则

1. 模型本身不维护图层版本、bbox、z-index、父子资产关系；这些属于前端 Workbench/runtime。
2. Gemini 负责理解不等于 Gemini 负责最终图片生成；图片执行由选择的图片模型决定。
3. SAM 负责轮廓，不负责高层语义；它需要 Gemini、YOLO 或用户提供的 bbox/点/mask。
4. Matte/CV 是速度优先的本地算法，不是 SAM，也不等于 AI 语义抠图。
5. 当前没有可作为主生产路由的独立 YOLO 自动检测模型。
6. `MODEL_SUITES` 中的历史 `text` 字段与 `getTextModel()` 可能不一致，改模型时必须检查真实调用函数。
7. QuickRouter 的 `.ai` 和 `.us` 是部署网络问题的一部分；当前后端默认使用 `.us`。
8. Qwen Image 3.0 Pro、Qwen Image Edit 2509 和 GPT-image-2 的结果格式不同，不能共用供应商响应解析逻辑。
9. 视频模型目前解决的是静态图转视频和动效预览，不代表已经实现视频对象逐帧编辑。

## 12. 相关代码入口

- 前端模型定义和供应商调用：[src/ai-services/gemini-client.js](../src/ai-services/gemini-client.js)
- 图片任务分流：[src/ai-services/skills-engine.js](../src/ai-services/skills-engine.js)
- Qwen 本地直通开关：[src/ai-services/qwen-local-routing.js](../src/ai-services/qwen-local-routing.js)
- 后端所有主要 API 路由：[index.js](../index.js)
- 分割服务客户端：[src/services/segmentation-service.js](../src/services/segmentation-service.js)
- SAM/FastSAM 服务：[fastsam-backend/main.py](../fastsam-backend/main.py)
- 本地 Matte：[src/graphics/matte-engine.js](../src/graphics/matte-engine.js)
- Wan 视频：[src/ai-services/siliconflow-video.js](../src/ai-services/siliconflow-video.js)
- 用户模型选择器：[index.html](../index.html)

