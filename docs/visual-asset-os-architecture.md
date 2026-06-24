# Visual Asset OS Architecture Spec (Minimal Viable Runtime)

## 0. 核心定位 (Core Positioning)
- **拒绝过度OS化**：不盲目构建庞大复杂的图形和物理引擎。当前系统必须回归“最小可用运行时 (MVR)”原则。
- **阶段性核心任务**：“让视觉对象活下来”，即建立最基础的对象 UID、状态和画布空间。

---

## 1. Minimal Asset Runtime (第一优先级)
系统的基石是让对象摆脱“一次性像素”。先不做复杂的 PBR、法线、深度等机制。

### 1.1 资产实体 (Asset Entity) MVR 结构
```typescript
interface AssetEntity {
    uid: string;                 // 灵魂：全局唯一身份标识
    type: SemanticType;          // 'product' | 'human' | 'background' | 'element'
    
    // 最小必要数据
    sourceImage: string;         // 对象主体图片
    masks: string[];             // 相关遮罩
    
    variants: AssetVariant[];    // 基于此对象的修改变体历史
    
    metadata: Metadata;          // 创建与更新基础信息
}
```

### 1.2 View Layer 降级
从现在起，逻辑核心操作 `AssetEntity`，图层 `Layer` 只是 UI 的展示映射。这是向对象化转变的最关键一步。

---

## 2. Minimal Workspace Runtime (第二优先级)
解决当前 Demo 中编辑完成后即焚（无法沉淀）的问题，让创作空间化。

### 2.1 Workspace 基础结构
不搞多人协同、不在早期上云端分布式同步，仅保证工程状态的长期保留。
```text
Project (工程)
│
├── CanvasState          // 核心画布的基础尺寸、颜色配置
├── AssetRegistry        // 当前项目中被引用的带有 UID 的实体表
└── History/Config       // 基础配置记录
```

---

## 3. 需克制的基础引擎 (早期保持简单)

### 3.1 简化的关系图谱 (Simple Scene Graph)
**不做**复杂物理关系、光影传播和空间拓扑。
目前只需要：
- `parent_of`
- `above` / `below` (层级关系)
- `mask_dependency`

### 3.2 简化的决策日志 (Simple Decision Log)
**不做**高阶的 CTR 预测链和复杂的知识推理库。
目前只需要一个 Action 搭配 Context 的 Logger。
```json
{
  "action": "change_style",
  "targetAssetId": "shoe_123",
  "prompt": "red leather",
  "timestamp": 169999999
}
```

---

## 4. 执行路线 (Execution Path)

**阶段一：Asset Runtime (当前)**  
赋予现有图像提取、生成流水线产生的对象唯一的 UID。

**阶段二：Workspace Runtime (随后)**  
让一切对象能在 Project 里留存和读取。

**阶段三：Relation Runtime (未来)**  
等到出现大量的跨项目拖拽、自定义图库复用时，再引入真正的 Graph 引擎。

**阶段四：Decision Runtime (远期)**  
积累大量 Prompt、Target 的上下文后，再提炼设计知识网络。
