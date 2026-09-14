# Magic Layers 当前完整流程与责任边界

> 文档状态：基于当前仓库代码整理，更新时间：2026-09-10。  
> 适用范围：Magic Layers 全图拆层、手动画框单层提取、遮挡物自动补全、背景净化、工作台渲染与持久化，以及已实现但默认关闭的 Agent 事务入口。

## 1. 先看结论

Magic Layers 的最终目标不是单纯生成几张透明 PNG，而是把一张平面图变成一组可以独立编辑、移动、分层、回写和恢复的工作台资产：

```text
用户点击 Magic Layers / 发起提取请求
        ↓
全图语义分析：识别对象、文字、背景、层级和遮挡关系
        ↓
选择可提取图层：默认选中可见、未锁定、非背景、非语义组
        ↓
锁定本次任务的原始场景
        ↓
按图层类型分流
  ├─ 真实物体 / 产品：SAM-B 起步，必要时 SAM-L 仲裁
  ├─ 文字：OCR，恢复为可编辑文字层
  ├─ 平面设计图形：保留为运行时图形，当前不走光栅抠图
  └─ 语义组 / 背景：不作为普通独立透明资产提取
        ↓
生成初始透明层；低质量结果进入 held / manual_review
        ↓
验证遮挡关系
        ↓
被遮挡硬物进入延迟补全：场景局部补全 → 二次 SAM-L → 质量审计
        ↓
补全资产通过后，原位替换同一个工作台子图层
        ↓
根据已成功提取的对象生成 clean plate
        ↓
渲染：clean plate 作为底板，透明子层按 zIndex 叠加
        ↓
Runtime / IndexedDB / Session / OSS 持久化
```

当前默认入口是工作台工具箱的 Magic Layers 按钮，最终调用 `triggerMagicLayers()`。`magicLayersCommand` 默认值为 `false`，因此默认走直接执行链；Agent 入口、聊天触发和确认式事务链已经存在，但只有打开对应 feature flag 后才会接管入口。

## 2. 谁负责什么

### 2.1 责任总表

| 角色 / 模块 | 负责什么 | 不负责什么 | 主要入口 |
| --- | --- | --- | --- |
| 用户 | 选择源图；必要时锁定或取消图层；在 Agent 模式下确认提交；处理 manual review | 不需要手动编排 SAM、补全或 clean plate | 工作台、图层面板、Agent 任务卡 |
| 工具箱 UI | 找到当前工作台图片，响应按钮点击，显示进度和结果 | 不判断对象是否被遮挡，不直接选择 mask | `src/ui/toolboxes.js` |
| Magic Layers 编排器 | 串联分析、选层、锁场景、分割、提取、补全、净化、渲染、保存 | 不实现语义模型、SAM 推理或图像生成模型本身 | `src/ui/layer-manager.js` |
| 语义分析模型 | 从整图提出场景图：对象、bbox、zIndex、语义类型、文本和关系 | 不生成透明 cutout，不修改工作台 | `analyzeImageLayers()` |
| Semantic Layer Views | 将模型结果规范化为 editable layers、clean-plate layers、layer graph 和 completion tasks | 不执行 SAM 或生成式补全 | `src/services/semantic-layer-views.js` |
| Segmentation Service | 组织分割请求、选择本地 / 云端传输、校验响应、归一化质量结果 | 不决定业务上是否延迟补全 | `src/services/segmentation-service.js` |
| SAM 后端 | 根据 bbox、语义上下文和质量策略产生候选 mask / cutout；执行 B/L 路由、候选选择和质量评估 | 不创建工作台子资产，不保存 session | `fastsam-backend/api.py`、`segmentation_core.py`、`candidate_selection.py` |
| 本地后处理 | Canvas 裁切、alpha 处理、OCR 文字状态、matte 和旧版回退路径 | 不替代主 SAM 质量门 | `layer-manager.js`、`matte-task-system.js`、`strategy-dispatcher.js` |
| Completion Coordinator | 只调度已验证的被遮挡硬物，等待必要 mask，管理补全批次 | 不直接生成补全像素，不负责最终子层替换细节 | `runAutomaticCompletionForItem()` |
| Object Completion Executor | 准备场景补全输入、调用图像编辑模型、恢复尺寸、二次 SAM、执行审计、产出 canonical asset | 不把 canonical asset 直接挂到场景上 | `src/services/object-completion-executor.js` |
| 图像编辑 / 生成路由 | 对遮挡区域做场景补全，对 clean plate 做背景修复 | 不判定补全结果是否可发布 | `editOrQueryImageWithGemini()`、QuickRouter / Qwen 路由 |
| Clean Plate 模块 | 根据待移除对象生成联合 mask，调用选定图像模型修复底板 | 不处理透明前景层，不负责对象分割 | `src/ui/workbench/layer-assets.js` |
| Workbench / Renderer | 注册父图和子图、保留 bbox / transform / zIndex、更新画布和图层面板 | 不重新推断语义，不重新选择模型 | `workbench/items.js`、`workbench/renderer.js` |
| Core Runtime | 保存 Workspace 内的资产、场景图和元数据，提供 Agent 事务边界 | 不持有 Magic Layers 的像素算法 | `src/runtime/CoreRuntime.ts`、`WorkspaceRuntime.ts` |
| Session / OSS 持久化 | 上传 durable 图片 URL，保存 session 和 runtime workspace，支持刷新恢复和跨设备同步 | 不决定何时接受某个 cutout | `src/core/session.js`、`src/services/ossService.js` |
| Agent Runtime（可选） | 将提取包装成可验证、可撤销、可确认、可提交的事务 | 不复制像素管线；像素工作仍由 Magic Layers 编排器完成 | `src/runtime/AgentRuntime.ts`、`AgentCommands.ts` |

### 2.2 责任边界原则

有三条边界必须保持：

1. 语义模型只提出“是什么、在哪里、前后关系是什么”；它不等于抠图结果。
2. SAM 只负责“目标像素边界是什么”；它不负责决定是否应该生成工作台资产。
3. Magic Layers 编排器负责业务状态和生命周期；它不应该把某个模型的具体响应格式泄漏给工作台或 Runtime。

## 3. 入口和当前开关

### 3.1 工作台按钮入口（当前默认路径）

1. `src/ui/toolboxes.js` 监听 `.magic-layers`。
2. 读取 `state.currentActiveWorkbenchItemId`。
3. 调用 `triggerMagicLayers(itemId)`。
4. 若没有完整语义分析，则先执行全图语义分析。
5. 自动选择可提取图层。
6. 因为当前 `magicLayersCommand=false`，直接调用 `triggerLayerExplosion(itemId)`。
7. 完成后打开图层管理面板并刷新底部操作区。

### 3.2 聊天 / Agent 入口（已实现，默认关闭）

只有 `magicLayersCommand=true` 时，聊天中的“提取 / 拆解 / 分离某个对象或全部图层”等文本才会进入 `handleLayerExtractionRequest()`。该入口会：

- 从当前工作区解析目标图层；
- 创建 `extract_layers` Agent Job；
- 默认设置 `requiresConfirmation=true`；
- 通过任务卡显示运行、验证、确认、取消和重试；
- 最终仍调用原有 `triggerLayerExplosion()`，不会另起一套像素算法。

### 3.3 相关 feature flag

| flag | 当前默认值 | 作用 |
| --- | --- | --- |
| `magicLayersCommand` | `false` | 开启 Agent / 聊天式 Magic Layers 提取入口 |
| `capabilityCommands` | `false` | 开启 Agent 能力事务入口 |
| `objectEditing` | `true` | 开启对象编辑 Agent 能力，但不等于开启 Magic Layers 提取命令 |
| `verificationAutoRepair` | `false` | Agent 验证失败时是否允许自动修复 |
| `agentTelemetry` | `false` | 是否输出 Agent 事件遥测 |

## 4. 全图语义分析

### 4.1 触发条件

当图片没有 `item.hasFullSemanticAnalysis`，并且现有图层中没有已经识别出的语义层时，`triggerMagicLayers()` 会调用 `analyzeImageLayers()`。如果 `triggerLayerExplosion()` 被单独调用，它也会在缺少图层时补做同样的分析。

若已有语义图层，Magic Layers 会复用它们，不会每次点击都重新分析。

### 4.2 分析输出

语义分析模型返回 JSON 图层数组，主要字段包括：

- `id`、`name`、`layerType`；
- `semanticType`、`designRole`、`renderMode`；
- `bbox`，格式为 `[ymin, xmin, ymax, xmax]`，范围 `0..1000`；
- `zIndex`，数值越大越靠前；
- `editable`、`promptHint`；
- 文字层的 `textContent`、`fontStyle`；
- 可选 `children`，用于组合对象和真实层级。

正常照片通常提出 4–14 个顶层语义单元；平面广告、海报、菜单等设计图会按可编辑设计单元拆分，文字、面板、价格牌、产品图片和装饰图形分别提出。

当前语义模型由 `getTextModel()` 返回，代码中为 `CHAT_TEXT_MODEL = 'gemini-3.8-flash'`，通过 `proxyGenerateContent()` 发送到后端代理。它不是当前 UI 下拉框选择的图片生成模型。

### 4.3 Semantic Layer Views 二次规范化

`buildSemanticLayerViews()` 和 `applySemanticLayerViewsToItem()` 将原始结果整理为：

- `editableSceneLayers`：可编辑对象视图；
- `cleanPlateLayers`：用于背景净化的对象视图；
- `layerGraph`：对象之间的父子、前后、遮挡和实体关系；
- `completionTasks`：潜在的“目标物被前景硬物遮挡”任务；
- `completionAssets`：补全契约和 observed / canonical 资产状态；
- `motionReadyLayers`：后续动画或 Magic Motion 使用的图层视图。

这里还会展开文字、补充 OCR 相关信息、规范化 composite entity。一个实体可能由父图层、子图层或多个带同一 `entityId` 的原子层组成；后续补全阶段必须通过同一套实体解析规则处理，不能只凭名称或 bbox 猜测归属。

### 4.4 手动画框图层

用户也可以先用框选工具创建 `box-layer-*` 图层。再次进行 Magic Layers 分析时：

- 手动画框层会被保留；
- 与手动画框重叠超过阈值的自动语义层会被过滤；
- 手动画框层插入场景并提高 zIndex；
- 后续通过同一分割服务执行单层提取。

## 5. 选层策略

### 5.1 全图按钮的默认选择

`selectAllMagicLayers()` 会遍历当前 `scene.layers` 或 `item.layers`，只自动选择满足以下条件的图层：

- 不是语义组 / composite group；
- 不是背景、底板或 background master；
- `visible !== false`；
- `locked !== true`。

因此，“自动选中”不等于“所有语义层都做透明抠图”。文字、平面设计图形和语义组会在后面的运行时分流阶段被分别处理。

### 5.2 Agent / 单层调用

当调用方传入 `layerIds` 时，`triggerLayerExplosion()` 只处理这些 ID。未锁定是硬条件；没有选中可执行图层时任务中止，不会调用 SAM。

## 6. 锁定任务原场景

正式分割前，`createMagicLayersSceneLock()` 从 `item.originalDataUrl || item.dataUrl` 读取图片，按原始像素尺寸复制为任务内 PNG，并保存到内存属性 `item.__magicLayersSceneLock`。

锁场景负责保证：

- 首次 SAM、场景补全和二次 SAM 在同一坐标系、同一组像素上运行；
- 不会把已经生成的 clean plate 当成首次分割源；
- 生成模型返回不同尺寸时可以恢复到任务原始尺寸；
- 所有延迟补全任务引用同一个任务锁，而不是每个对象临时读取当前图片。

锁场景是事务内存数据，不写进 session JSON。任务结束或异常时由 `releaseMagicLayersSceneLock()` 清除，避免重复持有整张大图造成浏览器内存压力。

## 7. 图层分流与首次提取

### 7.1 分流责任

`triggerLayerExplosion()` 会把选中的图层分为以下几类：

| 图层类型 | 当前处理方式 | 是否创建普通透明子资产 |
| --- | --- | --- |
| 真实物体、产品、可分割 raster layer | 批量提交 SAM | 是，质量通过时 |
| 被遮挡且满足延迟补全条件的目标 | 首次 SAM 只保存 observed reference | 否，等待补全 |
| 前景遮挡物 | 正常 SAM，同时作为 completion occluder 审计对象 | 是 |
| 文字层 | 缓存 OCR 或调用 strict OCR，恢复文本容器 / CSS 文字 | 是文字工作台资产，不是普通 raster cutout |
| `vector_shape` / 平面广告图形 | 标记 runtime vector pending / hold | 当前不生成普通 raster 子资产 |
| 语义组、背景层 | 不作为普通独立抠图对象 | 否 |

### 7.2 SAM 请求

`segmentLayers()` 负责：

1. 选择图片源。正式 Magic Layers 使用任务锁定场景，清空 `cleanPlateDataUrl`，避免误用旧底板。
2. 规范化引擎为 `sam`。
3. 解析分割传输方式：明确配置的后端 URL 优先；显式 cloud / local 配置其次；默认优先探测本地 `127.0.0.1:8000`，不可用时使用云端代理。
4. 发送图片、bbox、layerIds、语义上下文、实体层级和 `qualityProfile`。
5. 校验 HTTP 和业务成功标志。
6. 校验每个请求图层都有返回 cutout；缺少任一图层时，整批请求视为失败。
7. 把后端质量结果统一为 `accept`、`hold`、`needsHigherPrecision` 等前端字段。

前端的批量提取质量档位是 `completion`，普通单层提取也可以从同一服务进入该档位。后端仓库名仍是 `fastsam-backend`，但当前 `/segment` 契约实际管理 SAM-B / SAM-L 生命周期和候选仲裁：

- 首次普通分割通常先走 SAM-B；
- 候选低质量、碎片化、填充异常或语义质量不足时，后端可升级到 SAM-L；
- 带 `completionSegmentation` 标记的二次补全分割强制使用 SAM-L；
- 后端将最终 mask 映射回输入场景的原始尺寸。

### 7.3 首次结果质量处理

每个分割结果都会经过质量门。通过时，编排器把返回的透明图转成 File 并创建工作台子资产；子资产至少保留：

- `parentId`；
- `sourceLayerId`；
- `originalBbox`；
- `layerName`；
- `zIndex`；
- `extractEngine`；
- `quality`；
- `type: 'layer-explode'`。

低质量结果通常写入语义层 metadata，并标记 `held` / `manual_review`。对于必须让用户看到并继续处理的特定 raster 对象，代码可强制创建一个待高精度处理的子资产；这不等于质量门已经通过。

当前 `DISABLE_NON_SEMANTIC_GEMINI_FOR_FASTSAM_TEST=true`，所以批量 SAM 失败或手动画框单层 SAM 失败时，会阻止进入旧的 Gemini / StrategyDispatcher 兜底路径并直接失败。旧的 crop + matte + 生成式抠图代码仍保留，但不是当前 SAM 失败时的默认恢复机制。

## 8. 文字与平面设计层

文字层不应该被当成普通照片对象抠出。当前流程是：

1. 优先读取文本缓存；
2. 没有缓存时，调用 strict OCR；
3. 将 OCR 行限制在语义 bbox 内；
4. 通过 `prepareTextContainerCandidates()` 和 `restoreTextContainerShapes()` 恢复文字容器；
5. 通过 `buildExtractedTextState()` 创建可编辑文字状态和工作台文字项；
6. 后续 clean plate 仍需要移除原图中的文字像素。

平面广告中的 panel、badge、decor graphic 等图形会被标记为 `vector_shape` 或 `runtime_vector_pending`，当前 Magic Layers 不承诺为其生成高质量 raster cutout。它们的职责归工作台运行时图形层，而不是 SAM。

## 9. 遮挡物自动补全

这是 Magic Layers 与普通“按 bbox 抠图”的关键区别。

### 9.1 候选提出

`buildLayerGraph()` 根据语义层、bbox、zIndex 和实体关系提出 `completionTasks`。`getCompletionEligibility()` 会排除以下对象：

- 背景；
- 全局文字、文字层；
- soft edge、语义组；
- vector shape、panel、badge、button、logo 等平面设计元素。

家具、灯具、包装、食物、饮料、花瓶、雕塑以及其他通过筛选的 opaque hard entity 可以进入自动补全候选。

### 9.2 为什么要延迟

如果目标桌子被前景凳子挡住，首次 SAM 只能看到桌子的可见部分。该结果会保存为：

- `observedCutout`：只作为可见像素参考；
- `completionObservationMask`：用于二次质量审计；
- completion contract：保存目标、遮挡物、bbox、zIndex、任务签名。

被遮挡目标不会直接作为普通完整图层发布，也不会立即放入第一次 clean plate 清理清单。前景遮挡物先完成自己的 mask，目标才具备可靠的补全输入。

### 9.3 mask 验证

`runAutomaticCompletionForItem()` 调用 `validateCompletionCandidatesWithMasks()`，结合语义关系和实际 mask 验证：

- 目标 mask 是否存在；
- 前景遮挡物 mask 是否存在且属于目标前方；
- 两个 mask 是否真的相邻 / 接触；
- bbox 是否提供遮挡证据；
- 前景遮挡物是否通过 SAM-B / SAM-L 审计。

结果分为：

| 结果 | 后续 |
| --- | --- |
| `verified + auto` | `ready_for_completion`，进入自动补全 |
| `manual_review` / `unavailable` | 不自动调用生成模型，保留待人工或高精度处理 |
| `rejected` / `excluded` | 不生成补全资产 |

验证采用 mask 接触作为主要证据；仅 bbox 相邻不足以证明遮挡，避免把“墙上挂画在桌子上方”这类普通布局关系误判成缺失几何。

### 9.4 场景补全

`executeObjectCompletion()` 负责一次补全任务，步骤如下：

1. 读取任务锁定场景，而不是把残缺 cutout 当成主图。
2. 根据目标 bbox 和前景遮挡物 mask 生成局部编辑画布。
3. 编辑 mask 只覆盖已验证的前景遮挡区域，并保留未编辑区域。
4. 调用当前选定的图片编辑模型，在原场景中恢复被挡住的目标部分。
5. 检查生成结果是否没有残留遮挡物；检测到残留时最多再重试一次。
6. 将生成结果恢复到任务锁定场景尺寸。

图片编辑模型不是固定写死的 GPT-image-2。它由 `getImageModel()` 根据当前模型套装决定：默认状态为 `lite`，对应 `gemini-2.5-flash-image`；用户也可以选择 GPT-image-2、Qwen Image 或其他 Gemini 图片模型。选择 GPT-image-2 时才走 QuickRouter 的 GPT-image-2 编辑路径；选择 Qwen 时走相应 Qwen 路由。

### 9.5 二次 SAM 与 canonical 质量门

补全后的场景不能直接作为透明资产。系统再次调用 `segmentSingleLayer()`：

- 输入是补全后的原尺寸场景；
- bbox 优先根据 observed alpha 实际边界和相邻遮挡物推导；
- 发送 observation mask、occlusion mask 和 foreground context mask；
- 使用 `canonical_completion` 质量档位；
- 后端强制 SAM-L；
- 检查候选是否保留首次可见像素、是否越界、是否吸收其他前景对象、是否出现 scene-like 大 mask；
- 必要时使用 observed mask 做保守 reconciliation；
- 最终通过 original-observation audit 和 foreground-occluder audit。

只有通过质量门，才会生成 `canonicalAsset`。HTTP 200、返回图片或返回非空 data URL 都不能单独视为成功。

### 9.6 原位替换规则

`replaceSplitChildWithCanonicalAsset()` 的责任是把 canonical 资产提交到工作台：

- 优先按 `sourceLayerId` / entity identity 找原子子图；
- 不按相似名称或相交 bbox 随意替换另一个对象；
- 找到原子子图时原位替换，不新建重叠桌子；
- 如果目标之前只有 observation、没有发布子图，则创建它的第一份工作台表示；
- 保留原来的 transform、placement、父场景和 zIndex；
- 保留 provenance、observed asset、generation attempt 和 canonical version；
- 清理重复匹配子图。

最终组合关系例如：

```text
clean plate
  ├─ 桌子 canonical layer，zIndex 6
  ├─ 凳子 layer，zIndex 8
  └─ 花瓶 layer，zIndex 9
```

桌子可以是完整透明对象，但凳子仍通过更高 zIndex 覆盖在桌子前面；凳子像素不能被带入桌子的 canonical cutout。

## 10. Clean Plate 背景净化

### 10.1 普通对象

提取成功的对象会进入 cleanup 清单。`cleanMultipleBackgrounds()`：

1. 以原始场景作为底图；
2. 按对象 bbox 生成联合 mask；
3. 按空间距离把相近对象聚合成修复区域；
4. 对对象区域做适度膨胀和羽化；
5. 根据对象语义和当前图片模型生成 clean plate；
6. 将返回的 full image 设为 `item.cleanPlateDataUrl` 和 `item.dataUrl`。

Clean plate 使用的是原始场景，不是 GPT 补全后的场景。这样背景和未拆分对象仍尽量保持原始像素。

### 10.2 延迟补全对象

首次提取阶段会先跳过被延迟补全目标的清理，避免底板在目标仍残缺时就被净化。

补全批次完成后，当前代码会把已成功 canonical 的目标加入最终 cleanup 清单，再次调用 clean plate。需要注意当前实现的真实行为：

- 如果所有延迟目标成功，生成完整清理结果；
- 如果只有部分目标成功，但已有普通 cleanup layer 或至少一个目标成功，代码可能按“已成功对象 + 普通对象”生成部分 clean plate，并记录 `partial`；
- 如果没有可清理对象，则跳过最终 clean plate；
- 某个补全失败不会自动把错误的 canonical 资产发布到场景，但可能造成底板与预期图层集合不完全一致。

这也是当前最需要产品 / QA 关注的状态：`completionResult.completed < deferredCompletionTargetIds.size` 时，结果应被视为部分完成，而不是完整成功。当前日志会记录 `partial`，但用户提示文案不保证显式显示“部分完成”，所以这是现状风险，不应在验收时按全量成功处理。

### 10.3 Clean plate 模型

Clean plate 同样由 `getImageModel()` 动态决定：

- Gemini 图片模型通过 `editOrQueryImageWithGemini()`；
- GPT-image-2 通过 GPT-image-2 编辑路由；
- Qwen Image 家族在启用实验 clean-plate router 且当前模型为 Qwen 时，走 QuickRouter 的 Qwen clean-plate 模式；
- 默认模型套装为 `lite`，因此默认图片模型是 `gemini-2.5-flash-image`，除非用户或 session 改变模型选择。

## 11. 渲染和工作台状态

当父图得到 clean plate 后：

- 父图片元素显示 clean plate；
- 被提取成功的原图语义层通常被设置为不可见；
- 透明子图作为独立工作台项挂到 `parentId`；
- 子图使用语义 bbox 换算出的 `left / top / width / height`；
- 子图保留语义 zIndex；
- `renderLayerList()` 刷新图层面板；
- `renderCanvasLayers()` 刷新画布叠加结果；
- Runtime 收到新的 layers、scene、semanticViews 和 clean plate metadata。

批量提取使用有界队列，当前默认并发为 1，并在 job 之间让出浏览器线程。这是为了避免同一时间保留多个大 canvas、原图和生成 PNG，导致 renderer 内存峰值过高。

### 11.1 提取后的对象编辑与回写

Magic Layers 子图不是一次性导出结果，而是后续对象编辑的输入：

- 对象编辑器默认针对独立子资产工作，不把整张场景重新发给模型；
- 生成新版本时，编辑结果作为新的 isolated asset / version 保存；
- 选择“更新当前图层”时，只有具备 `parentId`、split placement 和可匹配父语义层的 Magic Layers 子图允许原位替换；
- 原位替换保留原来的位置、尺寸、透视、zIndex 和父子关系；
- 修改后可通过 `handleQuickFusionSync()` 按 `originalBbox` 把结果同步回父场景；
- `handleIsolatedAssetEdit()` 负责透明资产编辑、必要的再次分割 / matte、版本生成和持久化；
- Agent 对象编辑器只负责发起事务，实际图像编辑仍复用这条既有资产编辑链。

因此，Magic Layers 的“谁负责什么”在后续编辑阶段仍然不变：编辑器负责用户意图，独立资产编辑器负责子图版本，回写模块负责父图定位，Runtime / Session 负责保存版本和关系。

## 12. 持久化、恢复和 durable 资产

### 12.1 普通直接路径

每次关键阶段完成后，`persistLayerStateToRuntime()` 会：

1. 将父资产的 `layers`、`scene`、`semanticViews`、语义分析状态和 clean plate 状态写入 Runtime Asset Registry；
2. 如当前有 session，调用 `dbHelper.saveSession()`；
3. Session 同时保存 legacy workbench state 和 `runtimeWorkspace`；
4. 图片和 cutout 可以先用 data URL / blob URL 立即渲染，再由后台上传到 OSS；
5. 云端 session 同步阶段会把 session 中可上传的 inline media 替换为 OSS URL。

### 12.2 Agent 事务路径

Agent 模式会额外建立 operation snapshot：

1. 执行前记录父资产 metadata、旧图层状态和已有工作台 ID；
2. `triggerLayerExplosion()` 执行期间抑制普通 session 持久化，避免半成品被提前提交；
3. Agent command 验证输出资产是否存在、是否注册到 Runtime、是否在 scene graph 中；
4. 用户确认后，`commitAgentLayerExtraction()` 等待子资产上传完成；
5. 若仍有 data URL / blob URL，先确保有 durable OSS image reference；
6. clean plate 如仍为 data URL，则上传 OSS 并替换；
7. 提交 Runtime workspace 和 session；
8. 提交成功后释放 operation；失败或取消则按 snapshot undo。

### 12.3 刷新恢复

刷新时工作台应从 session / runtime workspace 中恢复：

- 父图的 clean plate URL；
- 子资产的 OSS source URL；
- `parentId`、source layer identity、bbox、transform、zIndex；
- active version；
- semantic layer graph；
- completion contract、observed asset 和 canonical asset 状态。

任务锁 `__magicLayersSceneLock` 不属于可恢复数据；刷新后不应依赖它。可恢复的 canonical URL 和 clean plate URL 才是正式结果。

## 13. Agent 模式的完整生命周期（可选）

当 `magicLayersCommand=true` 时，提取任务走以下状态：

```text
created / planned
    ↓
running：执行 extract_layers
    ↓
command verify：检查产出资产、Runtime 注册和质量
    ↓
verifying：执行 Workspace 结构验证和宿主检查
    ├─ 失败 → undo / 可重试
    └─ 通过 → waiting_confirmation
                    ↓ 用户确认
                 committing
                    ↓
                completed
```

Agent Runtime 只保存 ID、operation reference 和 output reference，不复制图片像素。真正的 Magic Layers 像素工作仍由 `layer-manager.js`、分割后端和图像服务完成。

如果 command 验证失败，Agent 会撤销本次 operation；如果用户取消或任务提交失败，也会按操作顺序反向 undo。当前 `verificationAutoRepair=false`，所以不应默认承诺会自动修复 DOM / Runtime 不一致。

## 14. 失败状态和处理责任

| 失败点 | 当前行为 | 责任方 / 下一步 |
| --- | --- | --- |
| 找不到工作台源图 | 弹出 / 返回错误，流程不启动 | UI / 用户确认当前选中图片 |
| 语义分析失败 | 拆解中止，保留原图 | 语义模型路由和用户重试 |
| 没有可编辑层 | 流程结束，不调用 SAM | 语义分析或用户补充手动画框 |
| 所有图层均背景 / 锁定 | 选择阶段中止 | 用户解锁或选择正确图层 |
| SAM 批量请求失败 | 当前开关下直接失败，不走 Gemini 兜底 | 分割服务 / SAM 后端检查本地或云端后端 |
| 单图层质量不足 | 写入 held / manual_review；特定对象可能创建待处理子图 | QA / 高精度重新处理 |
| OCR 失败 | 文字层仍完成流程，但可能没有可编辑文字行 | OCR / 用户人工重建文字 |
| 遮挡关系不可验证 | 标记 manual_review 或 rejected，不调用生成模型 | 语义图 / mask 质量检查 |
| 场景补全失败 | 不发布 canonical；completion asset 标为 manual_review | 图像模型 / 用户重试 |
| 补全场景残留遮挡物 | 最多重试一次，仍失败则任务失败 | Object Completion Executor |
| 二次 SAM 没有 cutout | 视为失败，不能把空响应当成功 | SAM 后端和质量门 |
| 二次 SAM 未通过 observation audit | 拒绝 canonical，保留 observed provenance | Completion Executor / QA |
| canonical OSS 上传失败 | 当前可短期保留本地 canonical，但不能当成 durable 云端结果 | OSS / 持久化层必须补传 |
| clean plate 失败 | 父图不替换为错误结果，透明子层仍可保留 | Clean Plate 模块 / 用户重试 |
| 延迟补全部分成功 | 可能生成 `partial` clean plate；当前日志会记录部分状态，但 UI 文案不保证显式标注 | 编排器提供状态，产品 / QA 不能按全量成功验收 |
| session 保存失败 | IndexedDB 可能仍有当前状态，跨设备 / 清缓存恢复可能落后 | Session / OSS 持久化层 |
| Agent 验证失败 | 事务 undo，进入失败或可重试状态 | Agent Runtime / 宿主 Runtime |

## 15. 关键状态对象

| 状态对象 | 所有者 | 作用 |
| --- | --- | --- |
| `item.originalDataUrl` | Workbench / Session | 原始源图，作为正式场景来源之一 |
| `item.dataUrl` | Workbench | 当前父图显示源，成功净化后通常指向 clean plate |
| `item.cleanPlateDataUrl` | Clean Plate + Workbench | 父场景净化底板 |
| `item.cleanPlateStatus` | Workbench / Runtime | `idle`、`ready`、`failed` 等底板状态 |
| `item.scene.layers` | Semantic Layer Views / Workbench | 语义图层定义、bbox、zIndex 和运行时属性 |
| `item.semanticViews.layerGraph` | Semantic Layer Views | 关系、遮挡、completion task 和统计 |
| `item.semanticViews.completionAssets` | Completion pipeline | observed / canonical / attempt / quality 状态 |
| `item.__magicLayersSceneLock` | Magic Layers transaction | 任务期间固定的原始像素，仅存在于内存 |
| 子工作台项 | Workbench / Runtime | 独立透明资产、parentId、transform、version |
| `operationId` | Agent adapter | Agent 事务的 undo / commit / release 引用 |

## 16. 关键日志

| 日志 | 说明 |
| --- | --- |
| `[Magic Layers] locked scene source` | 任务锁定了哪张原始场景和什么尺寸 |
| `[Explosion] SAM routing` | 当前选中层、提交 SAM 层和被跳过层 |
| `[segmentation-service] resolved transport` | 分割请求走本地还是云端，以及 endpoint |
| `[Completion Candidates] validated auto=...` | 自动、人工复核、拒绝候选的统计 |
| `[Object Completion] scene inpaint input/generated` | 场景补全输入和生成尺寸 |
| `[Object Completion] scene inpaint edit audit` | 补全场景是否残留遮挡物 |
| `二次SAM原始cutout - ...` | canonical 前的二次分割结果和审计指标 |
| `[Object Completion] canonical ready` | canonical 已完成质量门，等待替换 |
| `[Completion] replacement committed` | canonical 已原位替换并更新状态 |
| `[cleanMultipleBackgrounds] Starting for layers...` | clean plate 开始清理哪些对象 |
| `final clean plate after deferred completion` | 延迟补全后生成最终 / 部分 clean plate |
| `[Explosion] clean plate deferred...` | 仍有延迟目标，暂不做第一次底板净化 |
| `[Magic Layers] fatal_error` | 全链路异常，需结合前面 stage 日志定位 |

## 17. 代码地图

| 流程职责 | 文件 / 函数 |
| --- | --- |
| 按钮入口 | `src/ui/toolboxes.js`：`.magic-layers` handler |
| 总入口 | `src/ui/layer-manager.js`：`triggerMagicLayers()` |
| 直接拆解编排 | `src/ui/layer-manager.js`：`triggerLayerExplosion()` |
| 场景锁 | `src/ui/layer-manager.js`：`createMagicLayersSceneLock()`、`releaseMagicLayersSceneLock()` |
| 自动选择 | `isMagicLayersAutoSelectable()`、`selectAllMagicLayers()` |
| 语义分析 | `src/ai-services/skills-engine.js`：`analyzeImageLayers()` |
| 语义视图和关系图 | `src/services/semantic-layer-views.js`：`buildSemanticLayerViews()`、`buildLayerGraph()` |
| 遮挡候选验证 | `validateCompletionCandidatesWithMasks()` |
| 分割请求 | `src/services/segmentation-service.js`：`segmentLayers()`、`segmentSingleLayer()` |
| 分割后端 | `fastsam-backend/api.py`、`segmentation_core.py`、`candidate_selection.py` |
| 自动补全调度 | `runAutomaticCompletionForItem()`、`autoCompleteVerifiedOccludedAssets()` |
| 场景补全和二次 SAM | `src/services/object-completion-executor.js`：`executeObjectCompletion()` |
| canonical 原位替换 | `replaceSplitChildWithCanonicalAsset()` |
| 背景净化 | `src/ui/workbench/layer-assets.js`：`cleanMultipleBackgrounds()` |
| 工作台子资产 | `src/ui/workbench/items.js`：`addImageToWorkbench()` |
| 渲染 | `src/ui/workbench/renderer.js`、`src/ui/workbench/layers.js` |
| Runtime 同步 | `persistLayerStateToRuntime()`、`src/runtime/CoreRuntime.ts` |
| Session / OSS | `src/core/session.js`、`src/services/ossService.js` |
| Agent 入口 | `src/ui/agent-task-controller.js`、`src/runtime/AgentRuntime.ts` |
| Agent Magic Layers adapter | `executeAgentLayerExtraction()`、`commitAgentLayerExtraction()`、`undoAgentLayerExtraction()` |

## 18. 当前验收口径

一次完整成功的 Magic Layers 任务至少应满足：

- 语义层、bbox、zIndex 和图层关系可追溯；
- 普通对象的透明 cutout 通过质量门；
- 文字不是被错误当作普通照片 cutout；
- 被遮挡硬物的 observed 与 canonical 状态区分清楚；
- canonical 通过二次 SAM 和 observation / occluder 审计；
- canonical 原位替换，没有新旧对象重叠；
- 父图显示 clean plate，前景子层按 zIndex 正确叠加；
- 所有成功子资产和 clean plate 都有 durable OSS 引用，或明确标记为尚未持久化；
- 刷新后仍能恢复相同的父子关系、位置、层级、active version 和 completion 状态；
- 如果补全只成功一部分，当前至少应从 `completed / skipped` 和日志中的 `partial` 判断为部分完成；若要求界面和业务结果显式显示 partial，需要在编排器结果协议和 UI 文案中补齐，不能把当前成功提示理解为全量完成。

## 19. 稳定性改进建议：从线性补全改为受约束闭环

本节记录针对当前遮挡物补全链路的改进建议。它是后续改造方案，不代表当前代码已经全部实现。

### 19.1 当前稳定性风险

当前补全主链路是：

```text
原图
  ↓
首次 SAM
  ↓
Observed Cutout
  ↓
场景局部补全
  ↓
完整补全场景
  ↓
二次 SAM-L
  ↓
Canonical Asset
```

这条链路的隐含假设是：生成模型补出的场景足够准确，因此二次 SAM-L 可以重新识别完整目标。

实际运行中，生成模型可能同时改变：

- 原有目标的边缘和比例；
- 材质纹理、扶手、支撑结构等局部形状；
- 目标与地面的接触关系；
- 阴影、反射和周围像素；
- 目标与其他对象之间的边界。

因此，二次 SAM-L 可能准确地分割出“生成模型改变后的错误内容”。SAM-L 提升的是分割能力，并不能自动解决输入场景本身的歧义。

### 19.2 改造目标：受约束闭环

补全不应被视为“GPT 补全后再重新认识整个对象”，而应被视为一个带证据约束的闭环：

```text
Original Scene
      ↓
First SAM
      ↓
Observed Mask ───────────────┐
      │                      │
      ├─ Occluder Mask       │
      └─ Target Geometry     │
               ↓             │
          Hidden Region      │
               ↓             │
        GPT Scene Completion │
               ↓             │
       Completion Evidence   │
               ↓             │
          SAM-L Candidate    │
               ↓             │
    Observed Audit + Hidden Audit
               ↓
          Reconciliation
          ├─ PASS → Canonical
          └─ FAIL → Retry / Hold / Manual Review
```

### 19.3 Pixel Authority Hierarchy

后续实现应明确规定不同来源像素的权威级别：

1. 原图中首次 SAM 确认的目标可见像素：最高权威；
2. 已确认的前景遮挡物 ownership mask：用于识别前景对象归属和遮挡关系，但不禁止目标在其后方空间重叠；
3. 生成模型补出的隐藏目标像素：只有通过审计后才能使用；
4. 生成模型对未遮挡区域的重新解释：默认不接受。

这里的“遮挡物像素最高权威”应理解为 ownership prior：遮挡物必须保留为前景对象的独立资产，不能把遮挡物自身的视觉内容误识别成目标；但目标 canonical 可以在空间上与遮挡物 mask 重叠，因为目标应当存在于遮挡物后方，最终由更高 zIndex 的前景遮挡物覆盖它。

### 19.4 Hidden-only Canonical 规则

最终 canonical 不应直接等于二次 SAM 的完整结果。推荐使用以下模型：

```text
O = sanitized observed mask
F = verified foreground occluder ownership mask
G = target geometry
H = G ∩ dilated(F)
S2 = second SAM candidate

HiddenCandidate = S2 - O
ValidatedHidden = HiddenCandidate ∩ H
CanonicalMask = O ∪ ValidatedHidden
```

对应的资产合成规则是：

```text
Canonical = OriginalObserved + ValidatedHidden
Occluder   = 独立前景资产，由更高 zIndex 覆盖
```

必须满足：

- observed 区域的颜色、alpha 和边界来自原始可见 cutout；
- 二次 SAM 只提供隐藏区域的新像素；
- 二次候选不得覆盖或重新着色原始可见区域；
- 候选必须落在目标几何和允许的隐藏区域内；
- 遮挡物自身的视觉内容必须通过 ownership / leakage 规则排除，但目标在遮挡物后方的合法空间重叠不能被简单删除；
- 允许保留窄的 transition band，避免隐藏区域和可见边界之间出现断裂；
- 不能简单地把所有 `S2 - O` 都接受为隐藏区域。

当前 `reconcileCompletionSegmentation()` 已经具备一部分相关能力，包括观察区域清理、遮挡物 mask 膨胀、连通域筛选和生成区域统计。但当前逻辑仍可能在候选覆盖 observed 区域时使用二次候选像素；后续应把“observed 永远优先”提升为最终像素合成规则，并让 `full_scene_sam` 路径也必须经过同一套 reconciliation。

### 19.5 OBSERVED / OCCLUDED / UNKNOWN 三分区

建议在 completion contract 或 reconciliation 中明确保存三类区域：

| 区域 | 含义 | 生成策略 |
| --- | --- | --- |
| `OBSERVED` | 原图中真实可见且已被首次 SAM 确认的目标区域 | 固定保留，不允许生成模型覆盖 |
| `OCCLUDED` | 目标几何与已验证前景遮挡物相交的隐藏区域 | 允许生成，但必须审计 |
| `UNKNOWN` | 没有足够证据证明属于目标的区域 | 默认不自动生成，进入 hold / manual review |

`OCCLUDED` 不应直接等于整个 occluder bbox。真正的编辑区域应优先使用：

```text
Target Geometry ∩ Occluder Mask
```

再加一个受限的边缘过渡带。这样可以避免把“遮挡物覆盖的所有位置”误当成目标应该存在的位置。

### 19.6 Negative Mask、Ownership Leakage 与 Shape Prior

二次 SAM 除了接收 bbox、observation mask、occlusion mask 和 foreground context mask，还应明确接收或派生两类约束：

#### Negative Mask 与合法空间重叠

```text
AllowedHidden = Target Geometry ∩ dilated(Occluder Mask)
                + 必要的窄 transition band
```

候选的空间约束应满足：

```text
Candidate outside AllowedHidden ≈ 0
Candidate outside Target Geometry ≈ 0
```

这里不能把 `Candidate ∩ OccluderMask` 直接当成泄漏。桌腿、桌面或柜体在凳子、椅子等前景遮挡物后方的空间重叠是合法完成结果，最终由前景对象更高的 zIndex 覆盖。真正需要阻断的是遮挡物自身被识别成目标的 ownership leakage，尤其是发生在允许隐藏区域之外、目标几何之外，或表现为独立遮挡物形状的候选像素。

因此，后续建议使用以下更精确的判定：

```text
Candidate outside allowed hidden region ≈ 0
Candidate outside target geometry ≈ 0
Occluder-like leakage outside hidden ownership region ≈ 0
```

`auditCanonicalSegmentationAgainstOccluders()` 当前已经遵循这一原则：目标与遮挡物的空间重叠本身是预期的完成证据，不应直接被报告为泄漏。审核重点应放在遮挡物视觉内容是否被目标吸收，以及候选是否越出允许的隐藏区域。

#### Shape Prior

首次 SAM 的 observed silhouette 应成为目标形状先验。二次候选可以沿可解释的遮挡方向扩展，但不应在没有证据的方向大幅膨胀。建议检查：

- 可见边界是否保持；
- 隐藏区域是否从可见边界自然延伸；
- 面积和 bbox 是否异常增长；
- 是否出现新的孤立组件；
- 是否出现无法由目标类别解释的凹陷、凸起或细长结构。

桌子、家具等低 fill ratio 对象需要使用对象类型专属阈值，不能套用普通实体的面积阈值。

### 19.7 Completion Difficulty Score

不是所有遮挡都适合自动补全。建议在候选验证后增加遮挡难度评分，至少包含：

- 遮挡面积占目标 bbox 的比例；
- 目标可见轮廓长度；
- 目标语义类型和几何复杂度；
- 纹理 / 材质复杂度；
- 前景遮挡物数量；
- 目标剩余可见面积。

初始分级可以是：

| 等级 | 建议策略 |
| --- | --- |
| `easy` | 自动补全、标准审计 |
| `medium` | 自动补全、严格审计，必要时执行一次针对性重试 |
| `hard` | 直接 manual review / hold，不强行生成完整资产 |

遮挡面积达到约 40%–50% 或目标可见轮廓过少时，应倾向于 `hard`。这属于系统能力边界，不应被当成普通异常反复重试。

### 19.8 Failure-aware Completion

生成失败后不应统一执行“再次调用一次 GPT”。应先根据失败原因选择动作：

| 失败原因 | 建议动作 |
| --- | --- |
| `occluder_residual` | 扩大严格受限的 occluder edit mask，重新补全 |
| `observed_boundary_drift` | 缩小编辑区域，强化原始像素保护 |
| `hidden_shape_discontinuity` | 加强 visible-edge continuation 约束 |
| `hidden_region_over_generation` | 拒绝候选，禁止扩大重试范围 |
| `unrelated_object_leakage` | 直接 reject 或进入 manual review |
| `empty_or_missing_cutout` | 视为分割失败，不进入 canonical 提交 |

当前代码已经会在生成结果 unchanged 或残留遮挡物时重试一次。后续应把失败原因结构化，并把 `retryStrategy` 一并写入 completion evidence，避免无目的增加重试次数。

### 19.9 Completion Reconciliation Engine

长期可以把当前 `object-completion-executor.js` 中的 reconciliation、审计和质量指标抽成独立的 `Completion Reconciliation Engine`。它的职责应保持单一：

输入：

- Original Scene；
- Observed Mask；
- Occluder Mask；
- Target Geometry；
- Completion Scene / Evidence；
- SAM Candidate；
- Other Known Object Masks。

输出：

- `canonicalMask` / canonical pixels；
- `confidence`；
- `failureReason`；
- `retryStrategy`；
- `observedRecall`；
- `hiddenCoverage`；
- `boundaryDrift`；
- `occluderLeakage`；
- `bboxExpansion`；
- `status: accepted | hold | manual_review | rejected`。

推荐的 evidence 结构示例：

```json
{
  "observedRecall": 0.98,
  "hiddenCoverage": 0.91,
  "boundaryDrift": 0.04,
  "occluderLeakage": 0.00,
  "bboxExpansion": 0.03,
  "confidence": 0.94,
  "status": "accepted",
  "failureReason": null,
  "retryStrategy": null
}
```

建议先在现有函数中固定像素权威规则和指标，再抽模块。不要同时进行大规模重构和算法变更，否则难以判断稳定性变化来自哪里。

### 19.10 黄金案例回归库

建议建立固定的遮挡补全黄金案例，而不是只依靠随机图片和人工肉眼验收。第一阶段可以先建立 12–20 个案例，覆盖：

- 沙发 ← 茶几；
- 桌子 ← 椅子 / 凳子；
- 床 ← 床头柜；
- 柜子 ← 植物；
- 餐桌 ← 椅子；
- 电视柜 ← 装饰物；
- 低、中、高遮挡比例；
- 单遮挡物和多遮挡物；
- 硬边对象和软边对象。

每个案例至少保存：

- Original Scene；
- Observed Mask；
- Occluder Mask；
- Expected Hidden Region；
- GPT Completion Result；
- SAM-L Candidate；
- Final Canonical；
- reconciliation evidence。

回归重点不应要求生成图片逐像素完全一致，而应比较指标范围和最终状态：

- observed recall；
- visible boundary drift；
- hidden coverage；
- occluder leakage；
- bbox expansion；
- disconnected component 数量；
- accepted / hold / manual_review / rejected 是否正确。

### 19.11 推荐实施顺序

| 优先级 | 改造内容 | 目标 |
| --- | --- | --- |
| P0 | 固定 Pixel Authority；observed 永远优先；所有路径统一经过 reconciliation | 防止 GPT / SAM 覆盖真实可见像素 |
| P0 | 显式计算 `Target Geometry ∩ Occluder Mask`，增加 ownership / leakage 约束 | 限制生成区域和对象归属泄漏，同时保留合法空间重叠 |
| P0 | 记录 hidden-only 合成指标 | 能判断错误来自观察区还是补全区 |
| P1 | 增加 Shape Continuity Score 和对象类型专属阈值 | 拒绝形状漂移和异常扩张 |
| P1 | 增加 Completion Difficulty Score | 对高遮挡案例主动 hold，而不是硬生成 |
| P1 | Failure-aware retry 和结构化 failure reason | 减少无效重试，提升可诊断性 |
| P2 | 抽取 Completion Reconciliation Engine | 固化模块边界，便于维护和测试 |
| P2 | 建立黄金案例自动回归 | 防止后续优化破坏已验证案例 |

### 19.12 稳定性验收口径

改造后，一次自动补全只有同时满足以下条件才允许进入 canonical：

- 原始 observed 区域被完整保留；
- observed 区域没有被生成像素覆盖；
- hidden candidate 只位于允许的隐藏区域或窄 transition band；
- candidate 与 occluder、其他已知对象的泄漏低于阈值；
- 形状从可见边界自然延伸，没有异常 bbox 扩张；
- 生成的孤立组件满足目标类别的结构约束；
- hidden coverage 足够，且没有大面积 UNKNOWN 被擅自生成；
- evidence 中记录了完整指标和接受 / 拒绝原因；
- 高难度案例能够进入 hold / manual_review，而不是被强制发布；
- partial completion 不被误报为全量完成。

这套规则的核心不是让模型“更聪明”，而是明确规定模型最多可以改变哪些像素，并让所有不确定区域可被审计、暂停和恢复。

### 19.13 当前实现进度

本轮已落地的 P0/P1 基础能力：

- completion 的所有输出模式统一经过 reconciliation；`full_scene_sam` 不再绕过像素权威规则；
- 清理后的 observed 像素优先级高于二次 SAM，二次候选只允许在验证过的隐藏区域提供新增像素；
- 记录 `hiddenCoverage`、`unknownGeneratedPixels`、`bboxExpansion`、`observedPriorityPixels` 等 evidence；
- 生成失败记录 `failureReason` 与 `retryStrategy`；
- 组合遮挡物按结构和候选组件证据处理，不按“一个/两个”名称分支；
- 最终工作台替换优先依据完整 canonical evidence，避免旧的低 fill 状态覆盖已通过最终审计的结果。
- 隐藏候选恢复已抽成独立的约束函数，并加入无模型合成回归：只允许进入目标 bbox 内、已验证遮挡区内、且不与前景 ownership context 重叠的候选增量；不会复制备用候选的完整场景轮廓。
- 后端会输出 `Completion hidden candidate rescue audit`，分别记录候选原始增量、遮挡区/ownership/几何支持拒绝量和最终接受量，用于区分“候选没有隐藏证据”和“证据被约束拒绝”。
- canonical evidence 的构造与工作台替换放行共用 `completion-evidence.js`，避免执行器和 UI 各自解释 `accepted/hold` 造成状态分叉。
- 修复了“辅助点恢复未命中但主 SAM-L 候选已验证”分支中的作用域错误：调用方不再重复拼装审计，也不会引用不存在的恢复像素变量。

仍需用固定真实案例验收的内容：

- hidden candidate 的覆盖率是否足够恢复不同透视下的桌腿/支撑；
- GPT 场景补全是否在单体和组合遮挡物下都移除全部遮挡物残留；
- 5 个黄金案例的 accepted / hold / manual_review 分布是否符合预期。
