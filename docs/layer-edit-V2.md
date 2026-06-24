你现在的 V1 是“状态层正确了”

这非常重要，因为这一步很多团队都会偷懒跳过。
但你现在要做的 V2，本质上已经不是“再加一点 AI 功能”，而是：

把“语义图层”升级成“可操作像素图层”

也就是从：

scene.layers[] 是一组“知道自己是谁的对象”

升级到：

scene.layers[] 是一组“真的可以被拿起来、移动、局部编辑、再融合回去的资产”
先说一句最核心的产品定义

你最终想做的不是普通 AI 修图器，而是：

AI 驱动的 PSD 化编辑器


而 V2 的任务就是：

让每个 layer 真正“拥有身体”
你现在 V2 的目标，可以拆成 4 个系统

如果你不按系统拆，会很容易越做越乱。
我建议你把 V2 明确拆成这四层：

V2-A：图层像素资产化（Pixel Assets）

这是最重要的一层。

你现在 layer 只有：

id
name
bbox
category
zIndex
editable

这还只是“描述对象”。

V2 必须给每个 layer 增加真正的像素资产：

type LayerPixelAssets = {
  maskUrl?: string;        // 像素级蒙版
  cutoutUrl?: string;      // 抠出的透明PNG
  cleanPlateUrl?: string;  // 去掉该对象后的背景补图
  previewUrl?: string;     // 图层缩略图（给图层面板）
};
这一步的意义

只有有了这些，你才能真正做：

鼠标悬浮高亮某个对象
单独拖动它
单独删掉它
单独替换它
单独局部重生成它

否则你现在所有交互本质还是：

“在一张图上做智能框选”

而不是“图层编辑”。

V2-B：编辑器交互层（Canvas Interaction）

这层负责你说的这些体验：

悬浮高亮图层
鼠标变成箭头 / move cursor
选中图层
拖动图层
双击图层
文本图层进入文字编辑模式

这层和 AI 没关系，它是 编辑器能力。

很多 AI 产品死在这里，不是模型不行，而是交互不像工具。

V2-C：对象级 AI 编辑层（Per-Layer AI Editing）

这是你说的：

双击某个图层，弹出指令输入，只改这个区域

这一步不是“重新生成整张图”，而是：

只对当前图层做定向重生成 / 替换 / 修复

比如：

把这盏灯改成现代极简吊灯
把这张沙发换成奶油风
把这个 logo 换成蓝色玻璃材质

这一步要基于：

当前图层 mask
当前图层 cutout
当前图层 promptHint
当前图层 bbox / transform

来做局部编辑。

V2-D：融合层（Harmonization / Recomposition）

这是最终“像不像真图”的关键。

因为即使你把图层单独改好了，如果最后：

光影不对
透视不对
接缝不对
字体边缘不对
阴影不连续

用户还是会觉得假。

所以 V2 不能只做“能改”，还必须做：

改完以后自动融合回原图

这层决定产品质感。

所以 V2 的一句话定义应该是：
“让 SceneDocument 里的每个 layer，从 metadata 变成真实可操作的视觉对象。”
我建议你 V2 的开发顺序，不要按“功能酷炫程度”排

而要按“能不能让产品稳定起来”排：

第一优先级：先做 可拖拽的真图层

因为这是你整个产品从“AI 改图”变成“AI 编辑器”的分水岭。

也就是说：

先别急着做“高级 prompt 改图”

先把这三件事做出来：

1）图层 hover 高亮
2）图层 click 选中
3）图层 drag move

只要这三件事成了，你的产品立刻“像回事”。

你现在 V2 最正确的路线，不是“让模型更聪明”

而是先把这条链打通：

Analyze → Mask → Cutout → Canvas Selection → Drag → Recompose

只要这条链通了，后面所有高级 AI 编辑都能挂上去。

我下面直接按你目标，帮你拆 V2 应该怎么做
一、V2 最先要补的：Mask Pipeline

这是 V2 的起点。

你现在有 bbox，但要实现：

hover 高亮
精准选中
拖动对象
局部编辑对象

bbox 已经不够了。

你需要把每个 layer 从：

{
  id,
  bbox
}

升级成：

{
  id,
  bbox,
  maskUrl,
  cutoutUrl,
  cleanPlateUrl
}
V2.1 你应该新增一个步骤：
generateLayerAssets(scene)

它的职责是：

对每个 editable layer：
根据 bbox 裁切局部区域
对该区域生成精确 mask
用 mask 抠出透明 PNG（cutout）
删除该对象并补背景（clean plate）
生成 preview 缩略图
这一步的输出结果应该是：
scene.layers[i] = {
  ...
  maskUrl: ".../layer_sofa_01_mask.png",
  cutoutUrl: ".../layer_sofa_01_cutout.png",
  cleanPlateUrl: ".../layer_sofa_01_cleanplate.png",
  previewUrl: ".../layer_sofa_01_preview.png"
}
为什么 cleanPlateUrl 这么重要？

因为你说的“单独拖动图层”不是只拖前景这么简单。

比如：

你把沙发拖走了
原位置后面原本被挡住的地板和墙面必须自动补出来

所以你要么：

预先生成 cleanPlate
要么移动时实时做局部补图
V2 最稳的方案：先预生成

因为实时补图会卡、会抖、会不稳定。

二、V2 的交互核心：把图层从“UI 列表项”变成“画布对象”

这是你整个产品体验最关键的一步。

你现在 V1 大概率是：

左边图层面板
右边一张图
点击图层面板项，做一些操作

这还不够。

V2 要让右侧画布真正变成“编辑器画布”。

你应该给每个 layer 增加画布实体

也就是在前端渲染时，每个图层不只是 SceneDocument 里的一条数据，而是：

一个真实 Canvas / DOM / WebGL 对象

比如：

type CanvasLayerNode = {
  layerId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  visible: boolean;
  selected: boolean;
  hovered: boolean;
  image: HTMLImageElement | null;
  mask: HTMLImageElement | null;
};
然后右侧画布必须支持这 5 个事件：
1）mousemove

判断鼠标当前悬浮的是哪个 layer

2）mouseenter / hover

高亮该 layer

3）click

选中 layer

4）dragstart / dragmove / dragend

拖动图层

5）dblclick

进入编辑模式

这时候 hover 判定不要再用 bbox 粗判

而应该：

先 bbox 粗筛，再用 mask 做 alpha hit test

这是专业编辑器感的关键。

也就是说：
用户鼠标移动时：
先看鼠标是否落在某个 layer 的 bbox 里
如果在，再检查对应 mask 该点 alpha 是否 > 阈值
如果是，说明鼠标真的在对象上

这样你 hover 高亮和 click 选中就会非常“准”。

这一步做完后，你就能实现：
悬浮沙发只高亮沙发轮廓
鼠标变成 move cursor
点击精确选中对象
不会点到背景空白区

这体验会一下子非常像“真编辑器”。

三、V2 的拖动逻辑：先做“视觉移动”，再做“背景修复”

这是一个非常关键的工程策略。

很多人一上来就想：

“拖动的时候就实时生成新图”

这基本必炸。

正确做法是：
拖动时只做这两件事：
1）把对象 layer 的 cutout 在画布上移动
2）把原位置用 clean plate 补上

也就是：

原图不再直接显示为一整张
而是：
背景层（可能是 clean plate / base plate）
前景对象层（cutout）
拖动流程应该是：
用户开始拖动 layer_sofa_01

系统做：

隐藏原位置该 layer
显示 cleanPlate 补原位
拖动 cutout 到新位置
用户拖动结束

更新 scene：

layer.transform.x += dx
layer.transform.y += dy

然后重新合成当前视图。

这样你 V2 就能实现真正的：
对象级位置编辑

而不是“重新 AI 生成一张差不多的图”。

这一步非常值钱。

四、V2 的“双击编辑”要分成两条路

你这里目标其实有两种完全不同的图层：

路线 A：普通对象图层

比如：

沙发
茶几
瓶子
人物
灯具

双击之后应该：

弹出“对象编辑命令框”

用户输入：

“改成奶油白”
“换成北欧木腿风格”
“改成玻璃材质”
“换成黑色皮革”

然后系统只编辑当前 layer。

这一步你的执行逻辑应该是：
输入：
当前 layer 的 cutout
当前 layer 的 mask
当前 layer 的 promptHint
用户新指令
输出：
新版 cutoutUrl
（必要时）新版 maskUrl

然后替换该 layer 的视觉内容。

关键原则：
不要把整张图送去 edit

只改当前 layer。

路线 B：文字图层

这是另一条完全不同的系统，不能和对象图层混做。

你目标里最值钱的一句其实是：

如果该图层是文字，双击之后可以直接选中文字或字母修改

这非常对，但它意味着：

文本图层必须“对象化”，而不是“图片的一部分”
所以文字图层 V2 必须额外新增：
type TextLayerData = {
  textContent: string;
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: string;
  color?: string;
  lineHeight?: number;
  letterSpacing?: number;
  perspective?: {
    skewX?: number;
    skewY?: number;
    rotate?: number;
    warp?: string;
  };
  textBoxes?: Array<{
    char: string;
    bbox: [number, number, number, number];
  }>;
};
也就是说：

文字图层的最终目标不是“重新生一张带字的图”，而是：

把文字从 raster 恢复成可编辑 text object

这就是你最终能打穿 Canva / 传统 AI 修图器差异的地方之一。

所以 V2 对文字图层应该做什么？
1）先 OCR

识别文字内容

2）文本框重建

识别：

文本框位置
对齐方式
行数
字距 / 行距
大致字体风格
3）建立 text layer

用户双击时，不是弹 prompt，而是：

直接进入 text edit mode

就像设计工具一样。

用户修改后怎么“完美融入原图”？

这就涉及你说的最难但最值钱的一层：

透视 / 光影 / 字体融合

这一步不是单纯 DOM 文本覆盖就能搞定。

你要做的是：

V2 初版：

先做到 80% 产品体验

文本可编辑
大致位置和字体接近
保持图层级独立
V3 再做：
perspective warp
光影贴合
表面材质附着（比如文字在包装盒上、墙面上）

这个别在 V2 一口吃掉，不然你会陷入黑洞。

五、V2 真正的“融合系统”应该怎么做

这是决定“修改完像不像原图”的关键。

你目标里最重要的一句是：

修改完完美融入原图，不管透视还是光影，还是字体

这个目标非常对，但它不是一个单功能，是三个子系统：

1）几何融合（Geometry Fit）

负责：

位置
缩放
透视
旋转
V2 先做什么？

先做：

平移
缩放
旋转

先别急着上复杂透视网格。

为什么？

因为 80% 图层编辑先靠这三项就能成立。

2）视觉融合（Visual Harmonization）

负责：

色温一致
光照一致
边缘一致
材质一致
V2 怎么做最现实？

先做成一个后处理按钮 / 自动步骤：

“融合当前修改”

用户修改完 layer 后，系统对该 layer 区域做一次：

feather edge
color match
relight / harmonize（如果你接 AI）

不要在每次拖动都跑这个，会很卡。

3）阴影与遮挡融合（Occlusion / Shadow）

这是高级感来源。

比如：

沙发换了位置后，阴影要跟着走
文本改了后，背景反光不能假
新物体不能像“贴纸”
V2 建议：

先做成简化版

阴影先不完全自动重建
优先保证对象边缘干净、背景不穿帮
为什么？

因为真正自动阴影系统是大坑。
V2 先别掉进去。

六、V2 的最优开发顺序（非常重要）

如果你现在让我当 PM / 技术负责人，我会强制你按这个顺序做：

V2.1（必须先做）
图层像素资产化

目标：

生成 maskUrl
生成 cutoutUrl
生成 previewUrl
（最好）生成 cleanPlateUrl
成功标准：

你能在前端真正把图层“抠出来单独显示”。

V2.2（第二步）
画布选中 / hover / 拖动

目标：

鼠标悬浮高亮对象轮廓
点击精确选中对象
鼠标变 move cursor
拖动对象位置
成功标准：

你能像在简化版 Photoshop 一样拖动物体。

V2.3（第三步）
双击对象 → 弹 prompt → 单独编辑该图层

目标：

不改整图
只改该 layer 的视觉内容
成功标准：

用户可以双击“沙发”，输入“换成奶油风”，只改沙发。

V2.4（第四步）
文字图层特殊模式

目标：

OCR + 文本框恢复
双击直接编辑文字内容
成功标准：

用户能直接改 banner / 海报里的字，而不是“重新生成一张看起来差不多的图”。

V2.5（第五步）
融合优化

目标：

改完不假
边缘不脏
局部更自然
成功标准：

不是“能改”，而是“改完看不出来”。

七、如果你问我一句最现实的话：
V2 真正的核心不是“更强 AI”

而是：

“把图层当作一等公民（first-class objects）来管理。”

只要你做到了：

每个 layer 有自己的像素资产
每个 layer 有自己的交互实体
每个 layer 有自己的编辑入口
每个 layer 有自己的融合逻辑

那你这个产品就已经不是普通 AI 修图器了。

最后我直接给你一句非常明确的 V2 任务定义，你可以拿去当你们内部开发目标：
V2 的目标不是“让 AI 再聪明一点”

而是：

“让每个识别出的 layer，真的能被用户像拿起一张透明贴纸一样操作，并且改完还能自然回到原图里。”

这就是你接下来最该做的版本。