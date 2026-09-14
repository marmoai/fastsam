# Result Feedback Loop Analysis

本文档用于定义 MarmoAid 下一阶段最关键的一层能力：

**结果级反馈闭环（Result Feedback Loop）**

它的目标不是继续记录“用户做了什么”，而是开始回答下面这些更关键的问题：

- 哪个推荐最终真的被用户采纳了？
- 哪次编辑最后被保留下来了？
- 哪个结果进入了导出、资产库或后续复用？
- 哪条路径被撤销、删除或放弃了？

只有当系统开始稳定记录“结果”，推荐系统、补全系统和 `Magic Layers` 工作流才会真正变得越来越聪明。

---

## 一、为什么这一层现在必须做

当前系统已经具备：

- `Dispatcher -> CreativeMemoryLayer` 的行为记录链
- `ImplicitMemoryEngine` 的 Prompt 轨收口逻辑
- `SemanticRecommender` 的图层级推荐
- `Fusion Editor / Inspiration Capsule` 的推荐 UI
- `Magic Layers` 的语义拆层与后续编辑链路

这些已经足够形成一个“会记录行为”的 Runtime。

但现在仍有一个根本问题：

> 系统更擅长记录“做过什么”，还不够擅长记录“什么真的有效”。

这会带来几个明显限制：

1. 推荐系统容易学到“常见动作”，却不一定学到“好结果”
2. Chat 补全容易学到“用户常写什么”，却不一定学到“用户最终接受什么表达”
3. `Magic Layers` 可以越来越懂你怎么改图层，但还不够懂哪些拆层结果真的值得沉淀

所以从 Runtime 演化顺序上看：

- 第一阶段：行为记录
- 第二阶段：结果记录
- 第三阶段：基于结果的推荐与编排

当前最值得做的，就是第二阶段。

---

## 二、当前系统已经有哪些“接近结果”的信号

目前代码里已经存在一些零散的结果信号，只是还没有被统一成一套 Result Layer。

### 1. 显式正反馈：选用此图

当用户在聊天区点击“选用此图”时，当前逻辑会触发：

- `implicitMemoryEngine.registerImplicitSatisfaction(...)`

这说明系统已经承认：

> “选用此图”是一种高置信度的用户认可动作。

这是当前最可靠的显式正反馈之一。

### 2. 隐式正反馈：停留 + 互动

`ImplicitMemoryEngine` 里当前存在一套遥测收口逻辑：

- 资产被加入工作台后开始追踪
- 如果用户做过微调、裁切、拆解、移动或缩放
- 并且该结果停留超过 2 分钟
- 系统会自动收录为一条 `satisfaction rule`

这套逻辑有价值，但它更适合作为弱正反馈，而不是高价值结果依据。

### 3. 隐式收口：主题切换

当用户在同一会话里从一个主题切到明显不相关的下一个主题时，当前系统会尝试把上一条 prompt chain 自动收口。

这个逻辑说明：

> 当前系统已经具备“上一轮大概率结束了”的意识。

但它更像“会话结束信号”，不是稳定的结果信号。

### 4. 导出行为

图层方案现在已经支持导出：

- `导出方案`

这是非常强的结果信号，因为导出通常意味着：

- 当前版本足够成熟
- 已接近交付
- 用户认为结果可带离系统

但现在导出动作还没有被写回 Runtime 结果层。

### 5. 存入独立资产库

当前右键菜单里已经支持：

- `已存入独立资产库`

这个动作非常重要，因为它说明：

> 用户认为这个结果不是一次性中间态，而是值得未来复用的资产。

这是一类非常强的长期价值信号。

### 6. 删除惩罚

`ImplicitMemoryEngine` 当前已有删除后的惩罚逻辑。

这说明系统已经有“失败路径”意识，只是它现在主要作用在 Prompt Rule 层，还没有升级成统一的结果负反馈。

### 7. 版本演化

当前每次 `UPDATE_FUSION` 并带有 `intent` 时，会把当前状态记为一个 variant。

这是很好的过程信号，但不等于“这个 variant 最终被采纳”。

---

## 三、当前缺的不是信号，而是统一抽象

现在的问题不是“系统没有结果相关动作”，而是这些动作分散在多个系统里：

- Prompt 收口归 `ImplicitMemoryEngine`
- 图层编辑归 `CreativeMemoryLayer`
- 导出只是 UI 行为
- 资产入库只是本地保存
- 删除只是惩罚规则
- 推荐应用没有统一 source id

这会导致一个很大的问题：

> 系统知道很多动作发生了，但不知道这些动作最后如何构成“一个被接受的结果”。

所以第一步不是继续增加更多复杂逻辑，而是要先建立一个薄的结果事件层。

---

## 四、应该如何定义“结果事件”

结果事件（Result Event）不是普通行为事件，它表示：

> 用户已经对某个结果给出了更强的接受、保留、复用或拒绝信号。

建议第一阶段先定义一个最小模型：

```ts
interface ResultEvent {
  id: string;
  timestamp: number;
  type:
    | 'result_selected'
    | 'result_exported'
    | 'result_saved_to_asset_library'
    | 'result_reused'
    | 'result_deleted'
    | 'result_reverted';

  assetUid?: string;
  layerId?: string;
  schemeId?: string;
  sessionId?: string;
  projectId?: string;

  sourceType?: 'manual' | 'recommendation' | 'prompt_completion' | 'magic_layers' | 'workflow_action';
  sourceId?: string;

  context?: {
    taskType?: string;
    semanticType?: string;
    designRole?: string;
    workflowStage?: string;
  };
}
```

这个模型的重点不是一步到位，而是先保证：

1. 有统一事件入口
2. 能归因给某个对象
3. 能归因给某次推荐或某条工作流

---

## 五、第一阶段应该先接哪些正反馈

第一阶段不应该贪多，只接高置信度正反馈。

### 1. `result_selected`

对应动作：

- 用户点击“选用此图”

意义：

- 这是最强、最明确的“我认可这版结果”信号之一

优先级：

- 最高

### 2. `result_exported`

对应动作：

- 用户导出方案

意义：

- 说明结果至少达到了用户愿意带离系统的程度

优先级：

- 很高

### 3. `result_saved_to_asset_library`

对应动作：

- 用户把结果存入独立资产库

意义：

- 说明结果具备未来复用价值

优先级：

- 很高

### 4. `result_reused`

对应动作：

- 某个资产被再次放入新项目、被再次调用、再次成为编辑对象

意义：

- 这是最强的长期价值信号之一

优先级：

- 第一阶段末尾或第二阶段初期接入

---

## 六、第一阶段应该先接哪些负反馈

同样不建议一开始就铺开太多，只补最关键的两个。

### 1. `result_deleted`

对应动作：

- 结果很快被删除

意义：

- 说明该结果不被接受，或者至少没有保留价值

### 2. `result_reverted`

对应动作：

- 用户撤销到前一状态

意义：

- 说明当前这次尝试没有被接受

这两个负反馈比“用户没点推荐”更有价值，因为它们更明确。

---

## 七、为什么不建议一开始就做复杂满意度评分

当前系统已经有一些隐式判定逻辑，例如：

- 停留 2 分钟
- 发生主题切换
- 发生过裁切、微调、拆解

这些都不是没用，但它们的置信度不够高。

如果一开始就把这些弱信号混进高价值推荐，会有几个风险：

1. 系统会学到“用户经常做什么”，却不一定学到“什么对用户真正有价值”
2. 某些只是中间态的结果会被误判为成功
3. 用户停下来思考或暂时放置的结果，会被错误当成满意结果

所以更稳的做法是：

- 第一阶段：只接高置信度结果事件
- 第二阶段：再让弱信号参与评分
- 第三阶段：再建立置信度模型

---

## 八、结果事件要如何归因

结果级闭环真正难的地方，不是“记一个事件”，而是“这个结果到底归因给谁”。

建议每个结果事件至少能回答下面三个问题。

### 1. 归因给哪个对象

例如：

- 哪个 asset
- 哪个 layer
- 哪个 scheme
- 哪个 workbench item

### 2. 归因给哪次动作

例如：

- 来自用户手动微调
- 来自 `Magic Layers`
- 来自 `isolated edit`
- 来自 `fusion sync`
- 来自某个推荐卡片
- 来自某个 prompt completion

### 3. 归因给哪个任务上下文

例如：

- 商品海报
- 餐饮海报
- 空间设计
- 电商主图
- 宣传页排版

没有这三层归因，结果事件就只能做统计，不能真正反哺推荐。

---

## 九、第一阶段最适合接入的代码入口

当前项目里已经有几个非常适合挂结果事件的入口。

### A. 选用此图

当前已经触发：

- `registerImplicitSatisfaction(...)`

建议额外补：

- `recordResultEvent({ type: 'result_selected', ... })`

### B. 导出方案

当前已经有按钮和导出行为。

建议补：

- `recordResultEvent({ type: 'result_exported', ... })`

### C. 存入独立资产库

当前已经有资产注册、本地资产库写入和提示文案。

建议补：

- `recordResultEvent({ type: 'result_saved_to_asset_library', ... })`

### D. 删除

当前已有删除 penalty 逻辑。

建议升级为：

- `recordResultEvent({ type: 'result_deleted', ... })`

### E. 撤销

当前已有 `undo / redo` 机制，但还没有“本次结果被撤销”的统一事件。

建议补：

- `recordResultEvent({ type: 'result_reverted', ... })`

### F. 推荐应用

当前推荐应用和工作流按钮操作还缺少统一 `sourceId`。

建议后续补：

- recommendation id
- workflow action id
- prompt completion id

否则未来即便记录了结果，也无法知道是哪个推荐带来的。

---

## 十、第一阶段完成后，系统会立刻获得什么

一旦把最小结果事件层搭起来，哪怕只接了少量事件，也会立即带来几个质变：

### 1. 区分“做过”和“被接受”

系统终于不再只是知道：

- 用户改过什么

而开始知道：

- 用户接受了什么

### 2. 区分“推荐被展示”和“推荐有效”

系统终于不再只统计：

- 推荐被点了多少次

而能开始统计：

- 推荐是否带来了选用、导出、入库、复用

### 3. 给 Chat 补全建立更可靠的反馈基础

未来补全不应该只看：

- 用户是否按了 `Tab`

还可以进一步看：

- 这个补全最终有没有导向有效结果

### 4. 给 `Magic Layers` 建立工作流级学习基础

系统可以开始知道：

- 哪类拆层结果最终被保留
- 哪类拆层结果经常被入库
- 哪类图层工作流经常通向导出或复用

---

## 十一、建议的第一阶段范围（MVP）

为了避免范围失控，建议第一阶段只做：

### 记录以下 5 类事件

- `result_selected`
- `result_exported`
- `result_saved_to_asset_library`
- `result_deleted`
- `result_reverted`

### 暂时不做

- 不做复杂满意度分数
- 不做多层级自动权重体系
- 不做团队共享学习
- 不做自动行业策略归纳
- 不做太重的可视化分析面板

### 第一阶段目标

只回答一个问题：

> 哪些结果，已经被用户明确地接受、保留或放弃了？

只要这一步成立，后续所有推荐系统都会更稳。

---

## 十二、建议的后续演化路径

### 第二阶段

引入：

- `result_reused`
- recommendation source id
- prompt completion source id
- workflow action source id

目标：

- 知道“哪个推荐/补全/工作流带来了结果”

### 第三阶段

引入：

- 弱信号参与评分
- 置信度模型
- 项目级与用户级结果聚合

目标：

- 让 Runtime 从“记录结果”走向“基于结果进行预测和编排”

---

## 十三、当前阶段的定义

当前阶段可以这样定义这项工作：

> Result Feedback Loop 不是一个统计层，而是 Runtime 从“会记住行为”升级到“会记住哪些行为最终有效”的关键桥梁。

这一步一旦做成，后续不管是：

- ChatPanel 的 thought completion
- 灵感仓的推荐排序
- `Magic Layers` 的工作流编排
- 资产复用与模板化

都会开始真正具备“越用越聪明”的基础。

---

## 十四、下一步建议

本文档之后，建议继续补两份配套文档：

1. `chat-completion-signals.md`
   `ChatPanel` 里如何记录补全展示、接受、发送与放弃

2. `workflow-completion-for-magic-layers.md`
   `Magic Layers` 如何从 feature 走向可学习的工作流入口

这样可以把结果反馈层继续向补全系统和工作流系统接上。
