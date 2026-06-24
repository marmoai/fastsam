# 智能图层提取与抠图分流策略 (Semantic Matting Routing Strategy)

## 1. 核心思想与流水线概述
在复杂的图像编辑与图层拆解中，使用单一的抠图算法（如通用的背景去除）往往无法应对所有材质。例如，具有大量极细边缘的“毛发”或具有半透明特性的“云朵”与硬边缘的“沙发”在抠图（Matting & Segmentation）处理上有着截然不同的需求。

**流水线 (Pipeline)：**
1. **语义解析 (Semantic Parsing)**: 获取图层名称 (LAYERNAME)，并可能结合大语言模型或关键词网络提取核心材质与物体类别特征。
2. **路由分发 (Routing)**: 根据类别映射字典，将当前目标派发给最合适的抠图算法或模型分支。
3. **针对性抠图 (Specialized Matting)**: 执行选定的专属算法（例如生成精确的 Alpha 通道或直接输出二值 Mask）。
4. **后处理 (Post-Processing)**: 边缘羽化、色彩去溢出 (Color Decontamination) 或半透明融合。

---
## 2. 图层材质分类及分流策略详解

我们根据物体的**边缘特征**和**透明度特性**，将常见语义 (LAYERNAME) 划分为以下几个主要大类，并设定专属的处理策略。

### 📌 类目 A: 高频复杂边缘类 (High-Frequency / Complex Edges)
*   **常见关键词**: 毛发、猫、狗、宠物、羽毛、绒毛、头发、树叶。
*   **特性**: 边缘极其破碎，包含大量的细长结构结构，无法用简单的二值（黑白）遮罩表示，必须依赖精细的渐变 Alpha 通道。
*   **分流逻辑 (Routing Logic)**:
    *   **算法选择**: 必须走 **深度学习 Alpha Matting** 流程（如 ViTMatte, MODNet, 或特定的人像/动物精细抠图模型）。
    *   **处理步骤**: 
        1. 先使用粗略分割模型（如 SAM）生成目标的粗略 Mask。
        2. 根据粗略 Mask 自动生成 Trimap（三元图：确定的前景、确定的背景、未知区域）。
        3. 将 Trimap 输入给高精度 Matting 模型，专门针对“未知区域”（细丝边缘）预测精确的 Alpha 通道。
*   **注意**: 绝对避免采用基于多边形或轮廓点 (Polygon) 的粗略分割，否则会有严重的“剪切感”。

### 📌 类目 B: 半透明与柔和渐变类 (Translucent / Soft Gradients)
*   **常见关键词**: 云朵、烟雾、窗纱、水花、水滴、火焰、光效、透明、玻璃。
*   **特性**: 物体中心或大面积区域也可能存在半透明度（Alpha < 1.0），背景经常会透过物体显示出来。传统的抠像如果强制变不透明，会显得极其虚假。
*   **分流逻辑 (Routing Logic)**:
    *   **算法选择**: 需要具备色彩和透明度解耦能力的模型（如 Background Matting V2，或专用的透明材质分割算法）。
    *   **处理步骤**: 启动物理通道分离 (PS Channel Matte)，生成精准光影轮廓与 Alpha。对于流体或无边界形体进行较大的平滑。

### 📌 类目 B+: 半透明织物类 (Translucent Fabric) - **核心优化点**
*   **常见关键词**: 窗帘、轻纱、窗纱、蕾丝、丝绸、婚纱。
*   **特性**: 同时具备“高频纹理（网孔、丝线）”与“全局可变半透明（Alpha随褶皱厚度急剧变化）”两种特征。绝不能与纯“云雾”类合并处理。
*   **分流逻辑 (Routing Logic)**:
    *   **代码标识**: `channel_fabric` / `TRANSLUCENT_FABRIC`
    *   **算法选择**: 增强型物理通道分离 (Enhanced Channel Matting)。保留所有网格高频细节（不重度羽化核心边界），同时维持全局基础色彩。
    *   **处理步骤**: 必须基于语义（deepMask）边界设定特定的 compressive 曲线（Choke Matte），防止背景色带渗透，同时保留透光褶皱。不再直接走二值分割或者普通毛发流程。

### 📌 类目 C: 几何硬边缘与工业类 (Hard Edges / Industrial & Solid)
*   **常见关键词**: 沙发、汽车、手机、桌子、椅子、建筑、鞋子。
*   **特性**: 轮廓清晰、边缘锐利，主要由硬材质构成，基本不包含半透明区域。
*   **分流逻辑 (Routing Logic)**:
    *   **算法选择**: 通用实例分割模型，如 **Segment Anything (SAM)** 或常用的通用去除背景 API (如 `imgly/background-removal`)。
    *   **处理步骤**:
        1. 利用语义直接生成二值化 Mask。
        2. 进行微弱的抗锯齿 (Anti-aliasing) 处理，即在边缘像素上做 1px 或 0.5px 的羽化 (Feathering)，避免图像拼接时出现像素狗牙。

### 📌 类目 D: 柔性织物与复杂折痕类 (Soft Fabrics / Folds)
*   **常见关键词**: 衣服、裙子、厚窗帘、床单、抱枕。
*   **特性**: 边缘总体清晰，但是表面存在大量高对比度的阴影、褶皱。简单的颜色或阈值分割容易把阴影当成背景抠除。
*   **分流逻辑 (Routing Logic)**:
    *   **算法选择**: U²-Net, 基于显著性目标检测 (SOD) 的模型，或 SAM。
    *   **处理步骤**: 重点在于保持主体的完整性，提取后如果边缘相对柔软，可以进行适度的边缘模糊以模拟景深。

---

## 3. 分流执行核心逻辑框架 (代码参考思路)

在具体代码实现时，可以构建一个分类映射工厂（Factory Pattern）：

```javascript
// 关键词到类别的映射网络
const CATEGORY_MAP = {
  COMPLEX_EDGE: ['头发', '毛发', '羽毛', '绒毛', '猫', '狗', '鸟'],
  TRANSLUCENT_FABRIC: ['窗帘', '窗纱', '轻纱', '蕾丝', '丝绸', '纱'],
  TRANSLUCENT: ['云', '云朵', '烟雾', '光效', '水花', '水', '玻璃', '冰'],
  FABRIC: ['衣服', '裙子', '床单', '织物'],
  HARD_EDGE: ['沙发', '桌子', '汽车', '手机', '杯子']
};

export function determineMattingStrategy(layerName) {
  // 1. 标准化语义 (去除前后空格和特殊符号，转小写等)
  const semanticName = layerName.trim().toLowerCase();
  
  // 2. 特征匹配 (判断输入名称属于哪个分类)
  for (const [category, keywords] of Object.entries(CATEGORY_MAP)) {
    if (keywords.some(keyword => semanticName.includes(keyword))) {
      return category;
    }
  }
  
  // 3. 回退默认选项
  return 'DEFAULT_FALLBACK';
}

export async function processLayerExtraction(layerName, imageBlob) {
  const strategy = determineMattingStrategy(layerName);
  
  console.log(`[Layer Routing] ${layerName} 命中策略: ${strategy}`);
  
  switch(strategy) {
    case 'COMPLEX_EDGE':
      // 启动 Trimap + Alpha Matting 引擎
      return await executeAlphaMatting(imageBlob);
    
    case 'TRANSLUCENT_FABRIC':
      // 启动半透高频物理通道解算引擎 (适度修边，保护纹理与透明度)
      return await executeChannelMatting(imageBlob, { type: 'fabric' });
      
    case 'TRANSLUCENT':
      // 启动半透明度特殊处理引擎 (如色彩通道提取法)
      return await executeChannelMatting(imageBlob, { type: 'cloud' });
      
    case 'FABRIC':
    case 'HARD_EDGE':
      // 使用精确的二值分割模型 (如 SAM / imgly API)
      return await executeStandardSegmentation(imageBlob, { feather: 1.0 });
      
    default:
      // 通用兜底处理
      return await runGeneralBackgroundRemoval(imageBlob);
  }
}
```

## 4. 后处理与去溢出 (Color Decontamination)
对于复杂边缘（羽毛）和半透明区域（窗帘、云朵），从原图中抠出后，边缘往往会携带原图的背景色带（比如抠蓝天背景下的白云，云的边缘发蓝）。
属于 **A** 和 **B** 类的图层分流后，还应当经过一个后处理步骤：
* **Foreground Estimation (前景颜色估计)**：强迫像素颜色只使用主体颜色，消除背景反光和色彩渗透，才能在后期组合编辑时完美融入任意的新背景。

## 5. 总结
建立基于 `LAYERNAME` 的语义路由机制，是实现“专业级”自动图像分层的核心护城河。通过把“大一统”的抠图操作拆分成**高频边缘、半透明态、固体硬边**三大专精流水线，可以彻底解决诸如“云朵被抠成了实心白块”、“宠物毛发变成锯齿多边形”等传统自动抠图最典型的痛点。
