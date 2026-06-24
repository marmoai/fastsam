# Visual Minimal Viable Runtime (MVR) Spec

## 核心定位
此文档定义了 **Visual Asset OS** 底层运行时的最小可用版本（MVR）。
避免“过度 OS 化”，我们现阶段不搞高深的图形引擎或知识图谱，只做“状态管理与留存”。

## MVR 核心构造 (The 2+2 Pillars)

通过在 `src/runtime/` 中构建新核心结构，我们分离出了以下引擎：

### 【绝对核心 - 第一优先级】
### 1. Asset Runtime (资产生命周期)
- **目标**: 将视觉内容“实体化”(Objectification)，让一切对象拥有 UID。这是世界的第一步。
- **核心数据结构**: `AssetEntity`
- **说明**: 放弃单纯的“图层叠加”，提取出的元素必须转为 Asset。它拥有修改历史 (`variants`)、语义属性和图片本体，作为一切后续操作的基础。

### 2. Workspace Runtime (空间持久化)
- **目标**: 让用户“能回来，能积累”。
- **核心数据结构**: `ProjectWorkspace`, `WorldState`
- **说明**: 以 Project 为单位保存状态。只包含画板配置 (`CanvasState`)、被引入画板的图册资源 (`AssetRegistry`) 和本地留存逻辑。

---

### 【克制发展 - 第二优先级】
### 3. Graph Engine (极简图谱)
- **目标**: 维护画布内图片彼此最基本的 `parent_of` 或 `z-index` 上下关系。
- **限制**: 当前坚决不做反光、深度投影等 3D 关联。
- **核心数据结构**: `SceneGraph`, 基础 `ObjectRelation`

### 4. Decision Runtime (意图日志记录)
- **目标**: 记录最基本的上下文：对什么对象，做了什么操作，传入了什么提示词。
- **限制**: 当前坚决不做复杂的 Reasoning Chain 推断分析，也不管用户的转换率。只做一个带 Context 的日志栈 (`DecisionLog`)。

## 架构依赖关系 (Hierarchy)

```text
Global -> CoreRuntime (单例 Orchestrator)
           └── ALL: Map<ProjectId, ProjectWorkspace>
                 └── ProjectWorkspace
                      ├── WorldState
                      │    ├── AssetRegistry (项目使用的实体)
                      │    ├── SceneGraph (简单的层级关系)
                      │    └── CanvasState (尺寸/背景等配置)
                      └── DecisionGraph (基础事件和意图记录日志)
```

这套极简宪法，确保我们在避免陷入“架构性死亡”的同时，打通向真正视觉操作系统演化的桥梁。
