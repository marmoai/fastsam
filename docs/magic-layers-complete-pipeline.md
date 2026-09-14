# Magic Layers 完整处理流程

## 目的与范围

Magic Layers 将一张场景图拆成可独立编辑、拖动和持久化的透明图层，并生成移除这些图层后的背景净化底板。对于被前景物体遮挡的硬质对象，例如被圆凳挡住的端景桌，系统不会把残缺的首次抠图直接当作最终结果，而是走延迟补全路径。

本文描述当前实现的完整运行时流程。模型选择会受当前路由和配置影响；本文只列出当前职责与默认路径，不把可配置模型名称当作永久协议。

## 总览

```text
原始场景图
  |
  +-- 语义分析：对象、bbox、zIndex、遮挡关系
  |
  +-- 锁定本次任务的原场景 PNG
  |
  +-- 首次 SAM 分割
  |     |
  |     +-- 无需补全的对象 -> 初始透明图层
  |     |
  |     +-- 需要补全的对象 -> observedCutout / observation mask（不发布）
  |
  +-- 前景遮挡物已提取后，验证补全候选
  |     |
  |     +-- 不通过 -> 手动审核或不处理
  |     |
  |     +-- 通过 -> GPT-image-2 场景局部补全
  |                    |
  |                    +-- 二次 SAM 分割完整对象
  |                    +-- 质量审计与 canonical 透明图层
  |                    +-- 原位替换同一 split child
  |
  +-- 全部延迟补全成功后 -> 联合背景净化 -> clean plate
  |
  +-- Runtime / IndexedDB / OSS session 持久化
```

## 模型与本地算法

| 阶段 | 模型或算法 | 职责 | 输出 |
| --- | --- | --- | --- |
| 语义分析 | 当前配置的多模态文本模型，通常经 Gemini/代理路由 | 识别对象、bbox、语义类型、层级和遮挡关系 | `scene.layers`、`layerGraph` |
| 首次分割 | 高精 SAM 后端，SAM-B 先行，质量不足时升级 SAM-L | 从锁定原图提取可见对象，建立首次观察 mask | cutout、alpha/mask、质量指标 |
| 场景补全 | 当前补全路径为 `GPT-image-2` | 仅在前景遮挡区域恢复被挡住的目标，并保持场景其余像素 | 完整但不透明的补全场景 |
| 二次分割 | 高精 SAM 后端，SAM-B 先行，必要时 SAM-L | 从补全场景重新抠出完整目标 | canonical 候选透明图层 |
| 背景净化 | `cleanMultipleBackgrounds()` 的当前 clean-plate 模型路由 | 从原场景移除已发布图层，补全墙面、地面等背景 | `cleanPlateDataUrl` |
| 本地后处理 | Canvas、mask 审计、候选筛选、alpha matte、形态学约束 | 尺寸恢复、遮挡校验、边缘保护、重复层清理 | 受验证的运行时资产 |

SAM-B 与 SAM-L 不是固定的主从模型。每次请求先尝试 B；后端根据 fill、碎片化、内部空洞、观察 mask 召回等质量信号，必要时切换到 L。SAM 内部可使用 `imgsz` 推理尺寸，但返回 mask 会映射回输入场景的原始像素尺寸。

## 1. 语义分析

入口为 `analyzeImageLayers()`，由 `triggerMagicLayers()` 发起。语义分析模型读取整张场景图，生成可编辑对象的结构化描述：

- `layerId`、名称与语义类型，例如 `table`、`furniture`、`wall_art`。
- 归一化 bbox，格式为 `[ymin, xmin, ymax, xmax]`，坐标范围为 `0..1000`。
- `zIndex`，数值越大表示越靠前。
- 图层关系和可能的遮挡关系。
- 哪些对象适合拆解，哪些应保留为背景、文本或装饰性元素。

这一步只做场景理解，不输出透明图层，也不修改原图。

## 2. 锁定原始场景

`createMagicLayersSceneLock()` 会从 `item.originalDataUrl || item.dataUrl` 读取原图，并按图片固有宽高写成任务内 PNG。该场景锁在本次 Magic Layers 任务中固定使用，例如日志：

```text
[Magic Layers] locked scene source
dimensions: 798x628
source: originalDataUrl
```

场景锁的目的：

- 首次 SAM、GPT 补全和二次 SAM 都基于同一份原场景坐标系。
- 不会误用已经背景净化、缩放或被先前编辑过的底板。
- GPT 返回不同尺寸时，可恢复回锁定场景的原始 canvas 尺寸。

场景锁只在任务运行期保存在内存属性 `__magicLayersSceneLock` 中，不重复嵌入 session JSON，避免会话膨胀。

## 3. 首次 SAM 分割

`triggerLayerExplosion()` 使用锁定场景调用 `segmentLayers()`，质量配置为 `completion`。请求会携带目标 bbox、语义策略、上下文图层和原始尺寸图片。

后端 `/segment`：

1. 以 bbox prompt 调用 SAM-B。
2. 根据策略选择候选 mask；桌子、家具等硬实体会采用对应候选筛选和边缘保护。
3. 当 B 的结果不符合质量门槛时，升级至 SAM-L 再做仲裁。
4. 将 mask 和透明 cutout 返回到输入场景同尺寸坐标系。

日志中的 `Image received, size: (628, 798, 3)` 是后端收到的实际场景像素尺寸，不表示前端先把图片缩小到模型输出尺寸。后端可能以更大的 `imgsz` 推理，但最终 mask 会重采样回这个原始尺寸。

### 未遮挡对象

没有可信前景遮挡的对象，首次 SAM cutout 可以直接成为普通拆解图层。它会被加入工作台资产、保留 bbox/zIndex，并加入之后的背景净化清单。

### 被遮挡对象

目标被更高 `zIndex` 的硬质前景对象遮住时，首次 SAM 得到的只是原图中实际可见的部分。此时系统保存：

- `observedCutout`：原图中可见像素的透明提取。
- `completionObservationMask`：首次 mask，用于后续质量审计和定位锚定。
- 完成契约：目标、前景遮挡物、原始 bbox、zIndex 和任务签名。

首次结果不会作为最终 canonical 图层发布，避免残缺桌子先显示在工作台，再与补全结果重叠。

## 4. 遮挡候选验证与延迟调度

`validateCompletionCandidatesWithMasks()` 将语义关系与实际 mask 结合验证。目标需要是可信硬实体，遮挡物必须位于目标前方，并且其可用 mask 与目标 bbox 支持真实遮挡关系。

常见结果：

| 状态 | 含义 | 后续行为 |
| --- | --- | --- |
| `auto` / `pending` | 遮挡关系和 mask 均可信 | 进入自动补全 |
| `manual_review` | mask 缺失、关系模糊或对象不适合自动补全 | 不自动调用补全 |
| `rejected` | mask 不相邻、对象类型不匹配或不构成遮挡 | 不生成补全资产 |

对桌子被凳子遮挡的案例，凳子会先被分割成透明前景图层。只有需要的前景 mask 已就绪后，`runAutomaticCompletionForItem()` 才调用桌子的补全任务。这样 GPT 的编辑区域可准确限定在凳子原先挡住桌子的区域。

## 5. GPT 场景局部补全

`executeObjectCompletion()` 使用锁定原场景、前景遮挡物 mask 和目标 bbox 构建局部编辑请求。当前补全路径会调用 `GPT-image-2`。

输入原则：

- 输入是完整的锁定场景，而不是首次抠出的残缺桌子小图。
- 编辑 mask 主要覆盖已知前景遮挡物，带有窄过渡区域。
- prompt 要求只恢复目标被遮挡部分，保持目标的透视、尺寸、材质、光照和其余场景内容。
- 未编辑区域会从原场景恢复，避免模型改写墙面、地面或其他对象。

GPT 返回的是完整、不透明的场景图，不能直接当作最终透明桌子图层。返回尺寸如果与输入不同，会经 `restoreMaskedSceneToSource()` 恢复到场景锁尺寸。

示例日志：

```text
[Object Completion] scene inpaint generated
sourceSize: 798x628
generatedSize: 800x624
restoredSize: 798x628
samPromptBbox: [537.6, 440.6, 785.4, 1000]
```

## 6. 二次 SAM 和补全质量门槛

补全场景恢复到原尺寸后，系统通过 `segmentSingleLayer()` 仅对补全目标再做一次 SAM 分割，使用 `canonical_completion` / completion 专用策略。

二次 SAM 的输入包括：

- GPT 补全后的原尺寸场景。
- 目标 bbox。
- 首次 `completionObservationMask`。
- 已提取前景遮挡物 mask 与 bbox。

后端补全模式不会把第一次残缺 mask 当成最终形状，而是用它保证原来可见的桌子部分仍被保留。候选必须满足 containment、bbox overlap、合理 fill 等条件，并避免选择整个场景或 bbox 填满的背景候选。

前端还会审计二次结果：

- `observationRecall` 必须保留足够多的首次可见目标像素，当前最低阈值为 `0.72`。
- 二次 mask 不能在非遮挡区域相对原观察结果发生过大漂移。
- 与已知前景遮挡物的异常重叠会被拒绝。
- 初始 observation 只作为来源记录和缺失像素兜底；有效的二次 SAM 结果优先作为最终形状。

成功的参考日志：

```text
二次SAM原始cutout - console_table
observationRecall: 0.985
observationIoU: 0.67
completionAnchor: true
qualityReason: semantic_table_primary_mask_with_attachments
```

这意味着二次 SAM 是补全对象的最终分割来源；首次 SAM 的 `observedCutout` 不是最终 canonical 图层，也不应与它叠加显示。

## 7. Canonical 资产与原位替换

二次 SAM 通过质量门槛后，前端将结果写成 canonical 透明资产并上传。`replaceSplitChildWithCanonicalAsset()` 在同一个 split child 上替换内容，而非创建第二个桌子图层。

替换时保留：

- 目标图层的 zIndex，确保凳子仍在桌子前方。
- 工作台 transform、旋转、父场景关系和已有交互状态。
- 原始 extraction geometry，避免补全资产因 GPT 返回尺寸差异而偏移。

更新的内容包括 canonical URL、完成状态、active version、bbox/placement 元数据。原始 `observedCutout` 仍作为 provenance 留存，不再作为可发布的运行时图层。重复匹配 child 会被清理，避免新旧资产重叠。

## 8. 最终背景净化

`cleanMultipleBackgrounds()` 在原场景上联合移除已成功发布的对象，生成 clean plate。它会按空间相邻关系把多个对象合并成清理区域，构建羽化 mask，并调用当前配置的 clean-plate 图像编辑模型。

普通拆解对象完成后可以进入净化。存在延迟补全目标时，系统必须等待所有预期 completion 成功，才会执行最终净化：

```text
[Explosion] final clean plate skipped because deferred completion did not fully succeed
```

该日志表示至少一个补全目标没有生成 canonical 图层，例如 GPT 失败、二次 SAM 未返回 cutout 或质量审计拒绝。此时系统保留现有图层状态，不用不完整的清单净化底板。

最终 clean plate 使用原始场景作为输入，而不是 GPT 补全场景，以便背景和未拆解对象保持原始像素。清理清单包含普通已提取对象，以及已经成功 canonical replacement 的补全对象。

## 9. 渲染和遮挡关系

工作台渲染时，场景底板显示 clean plate，拆出的透明对象按 zIndex 叠加：

```text
背景净化底板
  -> 桌子 canonical 透明层，zIndex 6
  -> 凳子透明层，zIndex 8
  -> 花瓶透明层，zIndex 9
```

因此，桌子可以在自身被遮挡的区域恢复完整形态，同时圆凳仍正确覆盖在桌子前方。补全的桌子不需要把凳子像素带入透明图层。

## 10. 持久化与刷新恢复

每次拆解、canonical 替换和背景净化完成后，状态会写入 Runtime，并由 MVR 保存到 IndexedDB；持久化资源上传后，session JSON 同步到 OSS。

| 数据 | 持久化方式 |
| --- | --- |
| 原场景锁 | 仅当前任务内存，不写入 session JSON |
| 普通和 canonical 图层图像 | 上传 OSS，session 保存 URL 和元数据 |
| 图层状态、zIndex、transform、active version | Runtime、IndexedDB、session JSON |
| clean plate | OSS URL / session 元数据 |
| `observedCutout` 和 completion contract | session 元数据，供追踪和恢复 |

浏览器可以短暂使用 Blob URL 立即渲染，但可恢复版本必须写入 OSS URL 和 session 状态。刷新时，工作台按 session 中的 active canonical asset 与 clean plate 恢复，不应退回到首次残缺观察图层。

## 11. 失败处理

| 失败点 | 系统行为 |
| --- | --- |
| 语义分析失败 | 不启动 Magic Layers 拆解 |
| 首次 SAM 无可接受 mask | 图层不发布，进入 held/manual 状态 |
| 遮挡关系不可验证 | 标记 `manual_review` 或 `rejected`，不调用 GPT |
| GPT 场景补全失败 | 不替换现有图层，记录 completion 失败 |
| 二次 SAM HTTP 200 但没有目标 cutout | 视为失败，不能把空结果当成功 |
| 二次 SAM 未通过 observation/漂移审计 | 拒绝 canonical，保留来源结果，不做错误替换 |
| canonical 上传失败 | 允许短期本地显示，但需标记持久化风险，不能把它当云端已保存 |
| 任一延迟补全失败 | 跳过最终 clean plate，避免底板与图层集合不一致 |
| session 同步失败 | IndexedDB 可暂存当前状态；刷新跨设备或清缓存前云端恢复可能落后 |

## 12. 关键日志对照

| 日志 | 含义 |
| --- | --- |
| `[Magic Layers] locked scene source` | 已锁定本次使用的原始场景与尺寸 |
| `[Completion Candidates] validated auto=...` | 遮挡补全候选校验结果 |
| `[Object Completion] scene inpaint input/generated` | GPT 场景补全输入与尺寸恢复状态 |
| `二次SAM原始cutout` | 二次 SAM 的原始透明候选，尚未完成最终替换 |
| `[Object Completion] canonical ready` | 二次 cutout 已通过审计并成为 canonical 候选 |
| `[Completion] replacing split child in place` | 正在原位替换现有图层，不新增重叠图层 |
| `[Completion] replacement committed` | canonical 图层、zIndex 和持久化状态已提交 |
| `[Explosion] final clean plate skipped...` | 至少一个延迟补全失败，因此未做最终背景净化 |

## 代码入口

| 职责 | 主要文件 / 函数 |
| --- | --- |
| Magic Layers 入口、场景锁、批量拆解、替换 | `src/ui/layer-manager.js`: `triggerMagicLayers()`、`createMagicLayersSceneLock()`、`triggerLayerExplosion()`、`runAutomaticCompletionForItem()`、`replaceSplitChildWithCanonicalAsset()` |
| 语义图层与候选验证 | `src/services/semantic-layer-views.js`: `buildSemanticLayerViews()`、`validateCompletionCandidatesWithMasks()` |
| 分割请求与响应校验 | `src/services/segmentation-service.js`: `segmentLayers()`、`segmentSingleLayer()` |
| GPT 场景补全、二次 SAM 和 canonical 处理 | `src/services/object-completion-executor.js`: `executeObjectCompletion()`、`reconcileCompletionSegmentation()` |
| SAM B/L 路由、候选选择、质量门槛 | `fastsam-backend/main.py`: `/segment`、`should_escalate_sam_to_l()`、`choose_completion_recovery_mask()` |
| 联合背景净化 | `src/ui/workbench/layer-assets.js`: `cleanMultipleBackgrounds()` |

## 当前验证基线

当前已验证的桌子被圆凳遮挡案例应具备以下结果：

- GPT 补全场景中不保留凳子轮廓。
- 二次 SAM 输出完整桌子透明图层，且保留首次可见部分。
- canonical 桌子替换同一图层，不产生新旧重叠。
- zIndex 保持桌子在凳子后方。
- 所有延迟补全成功后才执行背景净化。
- 刷新后从持久化的 canonical 图层和 clean plate 恢复相同工作台状态。

