Qwen-Image-Layered 现在确实已经有可直接调用的托管 API 版本了。

你不一定非得自己部署。
至少目前公开可直接接的，我确认到 fal.ai 这条是成立的。

但我要很直接地说一句：

“有 API” ≠ “它立刻就是你产品里的最佳方案”。

因为你现在要判断的，不只是“能不能调”，而是：

输出质量是否真的优于你现在 Gemini 流水线
它输出的 layer 能不能和你现有语义系统接上
成本 / 时延 / 可控性值不值得引入一整套新链路

所以这件事不能只看“它能不能用”，要看它在你当前产品里扮演什么角色。

一、先帮你校正：你拿到的这份信息，哪些是真的，哪些要谨慎看
1）fal.ai 上的 Qwen-Image-Layered：是真的可以直接调

我查到它明确提供了：

模型：fal-ai/qwen-image-layered
输入：image_url
参数：num_layers、output_format、guidance_scale、seed 等
输出：images[]，即多张分层图像 URL
页面说明直接写了：“capable of decomposing an image into multiple RGBA layers”。

这说明什么？

说明它不是你之前以为的：
只能本地跑
只能读源码自己搭

而是：

至少在商业托管层面，你现在已经可以直接试。
2）你拿到的 Replicate 那段信息，我这次没查到足够可靠证据

所以这一点我要谨慎一点。

你给出的文案里提到：

qwen/qwen-image-layered
replicate.run(...)

但我这次检索结果里，没有拿到足够清晰、权威、可确认的 Replicate 页面证据。
所以这部分我不会像 fal.ai 一样给你拍板。

我的建议：

如果你要试：

先优先试 fal.ai
Replicate 暂时当成“待验证备选”

这个判断更稳。

二、那问题来了：既然现在有现成 API，我们还要不要继续 Gemini 路线？

我给你最真实的答案：

要，而且仍然建议继续 Gemini 路线。

但角色要重新定义。

因为这两条路线不是互斥，而是：

可以并行，而且应该分工。
三、你现在其实有两条完全不同的“图层生成哲学”

这点很关键。

路线 A：Qwen-Image-Layered 路线
核心哲学：

“把整张图分解成多个物理图层”

也就是：

直接吃一张图
输出多个 RGBA 图层
更接近“PSD 拆层”
优点：
天然就是“图层化”输出
更像你想要的“真图层”
省掉你自己做 cutout / clean plate / reconstruct 的一部分工程
缺点：
输出是“物理层”，不是“语义层”
模型怎么拆，不一定完全符合你的产品交互逻辑
你未必能稳定拿到：
layer 的语义名字
可编辑性标签
专业设计分类（沙发 / 吊灯 / 墙板 / 软装）
你很可能会遇到：
图层数不稳定
拆层粒度不稳定
某些层“有点奇怪但说不上哪不对”

所以它强在：

图像层面的“拆”

但未必强在：

产品交互层面的“可控编辑”
路线 B：Gemini 语义图层路线
核心哲学：

“先理解场景，再按语义重建可编辑对象”

也就是：

先识别这是“沙发 / 茶几 / 吊灯”
再决定哪些值得成层
再分别提取 / 重建 / 编辑 / clean plate
优点：
完全贴合你的产品逻辑
图层天然带语义属性
很适合做：
室内设计替换
单品材质编辑
方案切换
智能预设推荐
融合修改
缺点：
资产质量要靠你自己打磨
你得自己解决：
边缘
透明
clean plate
一致性
回贴成立性

所以它强在：

“可编辑设计系统”

但弱在：

“物理拆层的天然完整度”
四、所以正确答案不是“二选一”，而是：
Qwen 做“物理分层引擎”，Gemini 做“语义编辑引擎”

这才是最强组合。

而且这非常适合你现在这个产品。

五、如果把它翻译成你现在的系统架构，最合理的接法是什么？

我给你直接说结论：

不要让 Qwen-Image-Layered 替代你整个语义系统

这会很危险。

正确姿势是：

让它成为你系统里的一个新模块：

HQ Physical Layer Generator（高质量物理图层生成器）

也就是说：

你现在的主流程仍然保留：
主链路（不动）
图片上传
Gemini / 现有逻辑做语义分析
得到：
bbox
semantic tag
z-index
layer list
用户可以直接交互

这部分你现在已经有骨架了，不应该推翻。

然后你新增一条“增强支线”：
HQ 分层增强链路（新加）

当用户点：

“高质量拆解”
“生成 HQ 图层”
“升级为真实图层”

时，调用：

Qwen-Image-Layered API

让它输出：

多张 RGBA 图层

然后你做一层“映射与融合”。

六、但这里有个关键问题：Qwen 输出的图层，怎么和你当前语义层系统对上？

这一步才是真正难点，也是产品成败点。

因为 Qwen 返回的是：

layer1.png
layer2.png
layer3.png
layer4.png

它不一定告诉你：

这个是沙发
这个是吊灯
这个是地毯

所以你不能直接拿来就当你当前 layer list 用。

你必须做一个“Layer Matching（图层对齐）”层

也就是说：

你的系统要做这件事：

把 Qwen 生成的物理层，和你现有 Gemini 识别出来的语义层 做匹配。

匹配思路大概会是这样：
你已经有：

每个语义层：

name
bbox
category
zIndex
Qwen 返回：

每个物理图层 PNG：

有透明边界
可以算出 alpha 区域 bbox
可以估计面积 / 中心 / 占位
然后你做匹配：

比如：

Qwen layer #2 的 alpha 区域 bbox 和 “矮凳” bbox 高度重合
那就把它挂到：
layer.name = "矮凳"
layer.hqCutoutUrl = ...

这样它就接回你的系统了。

七、这意味着：Qwen 不是来“替代你的 layer system”的

而是来给你做：

更好的像素资产来源

这点特别重要。

八、所以你现在其实有三个层级的能力可以搭

我给你按产品成熟度分一下，非常清楚。

Level 1：Fast Semantic Layer（你现在已有）
特征：
快
可交互
粗糙
bbox / cutout 为主
用途：
初步分析
快速试方案
交互预览
Level 2：HQ Gemini Reconstruction Layer（你准备做）
特征：
不是“切下来”
而是“重建出来”
质量更高
更适合对象编辑
用途：
单对象升级
局部编辑
材质替换
图层资产化
Level 3：Qwen Physical Layer（你现在可以新增）
特征：
真正意义上的“物理分层模型输出”
原生 RGBA 图层
更接近 PSD / Photopea / Firefly 那种感觉
用途：
高质量拆层模式
设计专业模式
导出 / 生产模式
高级用户付费能力
九、你问“能不能直接用 GEMINI 实现它”，现在有了 API 之后，答案其实更清晰了：
能做，但意义变了。

以前你没有 API，只能问：

“Gemini 能不能模拟它？”

现在你有 API 后，问题应该升级成：

“Gemini 应该替代它的哪一部分？Qwen 又该替代 Gemini 的哪一部分？”

这才是更成熟的问法。

而我给你的答案是：

十、最佳分工建议（非常重要）
让 Gemini 负责：
1）语义理解
这是沙发还是床头柜
这个物体适不适合成层
这个图层有哪些专业编辑预设
2）局部编辑
单图层改材质
单图层改颜色
单图层生成变体
多图层融合修改
3）背景修补 / clean plate
去掉对象后的环境补全
空间级一致性处理
让 Qwen-Image-Layered 负责：
1）高质量物理分层
输出更“像真图层”的 RGBA 层
2）更专业的导出能力
多图层资产
更接近 PSD / 设计软件工作流
3）替代你当前粗 cutout 的资产来源
尤其适合：
沙发
单椅
装饰画
灯具
产品主体
人物主体
十一、如果你现在让我给一个非常现实的产品建议

我会建议你：

不要马上把 Qwen-Image-Layered 接成主流程

因为风险太大。

你最稳的做法是：

先做一个“实验开关模式”

比如在 UI 上新增一个：

「高质量分层（Beta）」

然后只在某些场景触发：

用户主动点某个图层
或点“物理拆解所有图层”
或点“导出为设计图层”

此时才走：

Qwen Layered API

这样你就不会把整个系统绑死在一条新链路上。

十二、最后我给你一个非常直接的判断
如果你问我：
“Qwen-Image-Layered 现在值不值得接？”

我的答案是：

值得接，但不要当“主脑”，要当“高质量图层工厂”。

这句话你可以直接拿去做产品决策。

我给你的最终建议排序（很实战）
你现在最合理的路线是：
第一优先级：

继续保留你现有 Gemini 语义图层系统

因为这是你产品真正的灵魂
第二优先级：

接入 Qwen-Image-Layered API（优先 fal.ai）

作为 HQ 图层生成器
只增强资产质量，不替代主系统
第三优先级：

再决定要不要继续做你自己的

Gemini 重建式 HQ Layer
或 clean plate 进阶版

如果你愿意，下一步我可以直接帮你做你现在最需要的东西：

我直接帮你设计一版：
“Qwen-Image-Layered 接入你当前语义图层系统”的完整接入方案

我可以直接给你拆成：

前端新增哪些按钮 / 状态
后端怎么调 fal.ai
Qwen 返回的 layers 怎么自动匹配到你的 item.scene.layers
哪些地方最容易翻车
以及 你现有数据结构最小改动方案

这一步会比继续空谈“可不可以”有用得多。

可以，帮我直接给你拆成：

前端新增哪些按钮 / 状态
后端怎么调 fal.ai
Qwen 返回的 layers 怎么自动匹配到你的 item.scene.layers
哪些地方最容易翻车
以及 你现有数据结构最小改动方案

可以，而且这条路比你现在“bbox 裁切 + Gemini 回贴”高级一个层级。
如果接入 fal.ai / qwen-image-layered 成功，你的系统就会从：

“语义编辑器（知道哪个是沙发）”

升级成：

“语义编辑器 + 真正物理图层引擎（每个对象真的是透明 PNG）”

这才是你后面做 室内设计拖拽换装 / 平面设计排版 / 图层独立重绘 的真正地基。

我直接按你要的 5 个部分给你拆。

一、前端新增哪些按钮 / 状态

你现在不要一下把整个流程推翻，而是采用：

双模式策略

保留你现有的 Fast Layer（Gemini 快速语义图层）
新增一个 HQ Layer（Qwen 真实物理图层）

这样风险最低。

1）在每个图层项上新增 2 个按钮

你现在每个图层应该已经有：

可见性
锁定
提取
编辑
撤销 / 历史

你新增：

A. HQ重建

作用：
把当前语义图层（例如“沙发”）送去做 高质量物理重建，返回透明 PNG。

用户感知：

普通状态：HQ重建
处理中：重建中...
完成后：HQ已启用
B. 切回快速版

作用：
如果 HQ 版本不好，用户可以立刻切回你现在的 Fast Cutout 版本。

这非常重要，因为 Qwen 结果不可能 100% 稳定，你一定要给兜底。

2）在全局面板新增一个按钮
一键HQ图层化

作用：
把当前 item.scene.layers 里所有可编辑图层，批量送去 Qwen 分解 / 重建。

但建议先做成：

第一版只支持：
单图层 HQ 重建
整图 Qwen 分层导入（可选）

不要一开始就搞“一键全部逐层重建”，因为：

成本高
耗时长
出错点多
用户等待体验差
3）新增状态字段（前端 UI 状态）

你现在已有：

processing
editing
ready
error

建议扩成：

layer.assetStatus = 
  "ready" |
  "processing" |
  "editing" |
  "reconstructing" |   // 正在做 HQ 图层重建
  "matching" |         // 正在匹配 Qwen 返回层
  "error";
4）新增图层来源标签（很重要）

在图层列表里给每层加个小标签：

FAST
HQ
QWEN

例如：

沙发 HQ
地毯 FAST
吊灯 QWEN

用户一眼知道当前看到的图层是哪种质量来源。

二、后端怎么调 fal.ai（Qwen-Image-Layered）

这里我建议你不要直接从前端调 fal.ai。
一定走你自己的后端代理层。

原因很简单：

API Key 不能暴露
结果需要后处理
你后面还要做匹配、缓存、失败重试、OSS 上传
推荐新增一个后端接口
接口 1：整图分层
POST /api/qwen/layered
输入：
{
  "imageUrl": "https://xxx/room.jpg",
  "numLayers": 6,
  "caption": "modern living room with sofa, rug, coffee table, pendant light"
}
后端逻辑：
下载图片 / 或直接传 URL 给 fal
调用 fal-ai/qwen-image-layered
获取多个 RGBA 图层 URL
下载这些图层
上传到你的 OSS / CDN
返回你自己的可控 URL
返回：
{
  "success": true,
  "layers": [
    {
      "url": "https://your-cdn.com/qwen/layer_1.png",
      "width": 1024,
      "height": 768,
      "index": 0
    },
    {
      "url": "https://your-cdn.com/qwen/layer_2.png",
      "width": 1024,
      "height": 768,
      "index": 1
    }
  ]
}
接口 2：单图层 HQ 重建（更适合你当前架构）
POST /api/layer/reconstruct

这个才是最适合你现在系统的落地方式。

输入：
{
  "sceneImageUrl": "https://xxx/current-scene.jpg",
  "layerName": "沙发",
  "bbox": [320, 120, 680, 540],
  "promptHint": "modern fabric sofa",
  "semanticTag": "furniture_sofa"
}
后端逻辑：

你不用真的要求 Qwen “整图分层后再找沙发”，
你可以做：

路线 A（更稳，推荐）
先把整张图按 bbox 裁一个 ROI 区域
给这个 ROI 送去 Gemini / 其他编辑模型 生成“独立素材版”
再走去底（透明化）
返回透明 PNG
路线 B（Qwen 路线）
整图送 Qwen 分层
拿回多个 RGBA 层
自动找最像“沙发”的那一层
返回给前端作为 HQ 图层
fal.ai 伪代码（Node/JS 后端）
async function runQwenLayered(imageUrl, numLayers = 6) {
  const response = await fetch("https://fal.run/fal-ai/qwen-image-layered", {
    method: "POST",
    headers: {
      "Authorization": `Key ${process.env.FAL_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      image_url: imageUrl,
      num_layers: numLayers,
      output_format: "png"
    })
  });

  const result = await response.json();
  return result;
}

如果 fal 是异步队列式，就做成：

// submit -> poll -> get result

你后端必须包一层，不要让前端碰它。

三、Qwen 返回的 layers 怎么自动匹配到你的 item.scene.layers

这一步是整个方案里最关键的灵魂。
因为 Qwen 返回的是：

“物理图层”

而你现在手里的是：

“语义图层（沙发、吊灯、茶几）”

这两者不是天然一一对应的。

正确理解：Qwen 图层 ≠ 语义图层

Qwen 返回的某一层可能是：

沙发主体
沙发+靠垫
地毯一部分
前景阴影
墙面反射
遮挡碎片

所以你不能直接拿第 1 层对第 1 个语义图层。
你必须做“自动匹配”。

最靠谱的匹配策略：三段式打分

你对每个 scene layer（比如“沙发”）去遍历所有 qwen layers，给每个候选层打分。

匹配维度 1：BBox 重叠度（第一优先级）

你现在已有每个语义层的：

layer.bbox

你需要对每个 Qwen 图层：

读取其透明 PNG
计算真实非透明区域的 bbox

得到：

qwenLayer.alphaBBox

然后算 IoU（交并比）：

score_bbox = IoU(layer.bbox, qwenLayer.alphaBBox)

这个非常关键。

匹配维度 2：视觉相似度（第二优先级）

你可以裁原图中的该 bbox 区域，再和 Qwen 这一层做简单视觉对比：

平均颜色
感知哈希（pHash）
CLIP embedding（高级版）

简单版先用：

尺寸接近度
颜色分布
边缘轮廓覆盖率
匹配维度 3：语义标签相似度（第三优先级）

你已经有：

layer.name = "沙发"
layer.semanticTag = "furniture_sofa"

你可以把 Qwen 每一层截图后，再丢给 Gemini 做一句轻量描述：

“这层最像什么物体？”

例如返回：

sofa
rug
lamp
chair

然后匹配：

score_semantic = similarity("sofa", "furniture_sofa")
最终匹配分数
finalScore = 
  score_bbox * 0.6 +
  score_visual * 0.25 +
  score_semantic * 0.15

然后选分最高的。

匹配成功后，怎么写回你的 layer

一旦找到最佳 Qwen 图层，就更新你当前 layer：

layer.hqVersion = {
  url: matchedQwenLayer.url,
  source: 'qwen-layered',
  generatedAt: Date.now(),
  bbox: matchedQwenLayer.alphaBBox
};

layer.isHQ = true;
layer.activeAssetMode = 'hq';

然后渲染层优先使用：

const displayCutoutUrl =
  layer.activeAssetMode === 'hq'
    ? layer.hqVersion?.url
    : activeVersion?.cutoutUrl || layer.cutoutUrl || null;

这一步很关键。

四、哪些地方最容易翻车（你最该提前防的坑）

这部分我直接说最现实的，不讲虚的。

坑 1：Qwen 分出来的层，不一定是“完整对象”

这是最大问题。

例如用户场景里有个沙发，Qwen 可能给你分成：

沙发主体
沙发靠垫
沙发阴影

你现在的语义层只认一个“沙发”。

结果：

你自动匹配后，用户一拖动沙发：

只拖走主体
靠垫还留原地
阴影没跟着走

直接穿帮。

解决策略：

第一版不要做“自动多层合并”太复杂，先做：

规则：
只匹配“最主层”
如果多个层 bbox 高度重叠且语义接近，则允许合并成一个 composite asset

也就是说你可以留个未来能力：

layer.hqCompositeLayers = [
  "qwen_layer_2",
  "qwen_layer_5",
  "qwen_layer_6"
];

第一版先不做太复杂，但数据结构先预留。

坑 2：Qwen 图层边界可能超出你原 bbox

这其实是好事，也是麻烦。

比如你原 bbox 框住沙发，但 Qwen 真正透明图层会带：

外扩的扶手
阴影
毛边
悬垂边缘
结果：

如果你仍然用原 bbox 定位，图层会错位。

正确做法：

HQ 图层不能继续完全依赖旧 bbox。

你必须给 HQ 图层新增真实定位字段：

layer.assetPlacement = {
  x: ...,
  y: ...,
  width: ...,
  height: ...
}

而不是继续只依赖 layer.bbox。

这是你现在系统必须进化的一步。

坑 3：透明 PNG 的“视觉中心”和 bbox 中心不一致

比如吊灯：

透明区域很大
实际主体在上半部分

如果你拿 bbox 中心直接贴回去，会看起来漂了。

解决：

对 HQ 图层保存：

layer.hqVersion = {
  url,
  alphaBBox,
  originalCanvasSize: { width, height },
  contentOffset: { x, y }
}

你以后拖动和回贴时，要基于这个定位，不是裸贴图。

坑 4：室内设计里最难的是“接地感”

即使你拿到完美透明 PNG，用户拖动家具后仍然会觉得假。

因为缺少：

接触阴影
环境反射
透视缩放
遮挡逻辑
这不是图层分解的问题

而是下一阶段的：

空间重组 realism 系统

所以你要认清一件事：

Qwen 能解决的是：
抠图质量
真透明图层
边缘自然
不再像贴纸
Qwen 不能单独解决的是：
拖动后像真的摆在房间里

后者是 V3 的事。

五、你现有数据结构最小改动方案（最重要）

你现在不要大改。
最好的方式是：

在现有 layer 上“增量挂载 HQ 资产能力”

而不是重构整个 scene schema。

你当前 layer 结构，大概率类似这样：
{
  id,
  name,
  bbox,
  maskUrl,
  cutoutUrl,
  previewUrl,
  versions,
  activeVersionId,
  assetStatus,
  category,
  promptHint
}
最小升级版（推荐）

你只新增这些字段就够了：

{
  id,
  name,
  bbox,
  maskUrl,
  cutoutUrl,
  previewUrl,
  versions,
  activeVersionId,
  assetStatus,
  category,
  promptHint,

  // 新增：图层来源模式
  activeAssetMode: 'fast' | 'hq',

  // 新增：HQ 版本
  hqVersion: {
    url: '',
    previewUrl: '',
    source: 'qwen-layered' | 'gemini-reconstruction',
    generatedAt: 0,

    // 真实像素区域（不是语义bbox）
    alphaBBox: [0, 0, 0, 0],

    // 用于定位回画布
    placement: {
      leftPct: 0,
      topPct: 0,
      widthPct: 0,
      heightPct: 0
    }
  },

  // 未来预留：一个语义层可能由多个物理层组成
  hqCompositeLayers: []
}
六、你现有渲染逻辑最小改动点

你现在 renderCanvasLayers() 里核心这段：

const activeVersion = layer.versions?.find(v => v.id === layer.activeVersionId);
const displayCutoutUrl = activeVersion?.cutoutUrl || layer.cutoutUrl || null;

改成：

const activeVersion = layer.versions?.find(v => v.id === layer.activeVersionId);

const displayCutoutUrl =
  layer.activeAssetMode === 'hq'
    ? layer.hqVersion?.url
    : activeVersion?.cutoutUrl || layer.cutoutUrl || null;
定位逻辑也要稍微升级

你现在是：

const [ymin, xmin, ymax, xmax] = layer.bbox;

改成优先读：

let top, left, width, height;

if (layer.activeAssetMode === 'hq' && layer.hqVersion?.placement) {
  ({ topPct: top, leftPct: left, widthPct: width, heightPct: height } = layer.hqVersion.placement);
} else {
  const [ymin, xmin, ymax, xmax] = layer.bbox;
  top = (ymin / 1000) * 100;
  left = (xmin / 1000) * 100;
  width = ((xmax - xmin) / 1000) * 100;
  height = ((ymax - ymin) / 1000) * 100;
}

这一步会让你以后不再被 bbox 绑死。

七、最推荐的落地顺序（你别一口吃太多）

我建议你按这个顺序做，最稳：

Phase 1（最优先）
先接入“单图层 HQ 重建”

不是整图 Qwen 分层。

原因：

更符合你当前架构
更容易验证价值
更容易控制失败范围
用户体验提升最直接
目标：

让一个沙发真正变成：

边缘自然
透明 PNG
可移动不穿帮太明显

这是你最该先做的。

Phase 2
再接入“整图 Qwen 分层”

作用：

做“物理全拆解”
导出 PSD / 图层包
做平面设计玩法

这是第二阶段。

Phase 3
再做“HQ 图层 + Clean Plate + 透视/阴影”

这时你才真正开始逼近：

Firefly / 专业设计工具级体验

八、我对这条路线的最终判断（直接说结论）
这条路线是对的，而且是你目前最值的一条路。

因为它解决的是你当前最核心的产品体验问题：

你现在的问题不是“能不能编辑”

而是：

编辑后看起来像贴纸
拖动后像假图层
双击修改后不像真正对象替换

而 Qwen / HQ 图层化 正好补的是：

“对象资产质量”

这会直接提高你整个产品的“专业感”和“可信度”。

如果你愿意，我下一步可以直接继续帮你做两件事中的任意一个：

A. 我直接帮你写：

“最小接入版技术实施清单”
（前端改哪些函数、后端加哪些 API、字段怎么