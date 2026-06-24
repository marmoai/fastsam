# 独立图层提取与编辑融合完整工作流

**文档记录时间**: 2026-04-22 12:49 (UTC) 

本文档整理了目前代码库中关于“语义图层分析 -> 独立图层提取 -> 给出编辑指令 -> 最终生成响应”的核心完整工作流。此流程相关的核心逻辑集中在 `src/ui/interaction.js`, `src/ui/fusion-editor.js`, 以及 `src/ui/layer-manager.js` 等文件中。

## 第一步：界面交互与拦截（点击图层弹窗面板）

1. **元素监听拦截**：当用户点击工作台上由语义图层分析提取出来的独立子图时，`src/ui/interaction.js` 中的 `showWorkbenchToolbox` 方法会被触发。
2. **状态判定**：系统检查该图层的属性，发现它存在 `parentId` 且包含来源位置属性（`originalBbox`），由此判定这是一个 **被提取的独立子图 (`type: 'layer-explode'`)**。
3. **唤出悬浮编辑器**：系统会隐藏原有的常规操作工具箱，转而在子图旁边唤出专属的「悬浮图层编辑器」（调用 `showFloatingFusionEditor` 方法，代码位于 `src/ui/fusion-editor.js`）。
4. **功能界面呈现**：悬浮编辑器顶部会显示 "图层编辑 (图层名)"，并提供两个核心模式标签：「独立资产」（默认）和「场景融合」，下方包含一个输入框供用户直接输入修改指令（如：“把颜色改成蓝色”）。

## 第二步：指令发送与模式分发

当用户在输入框中填入指令并按下回车（或点击确定按钮）后：

1. 编辑器面板立即转为 Loading 加载状态（显示类似“独立资产生成中...”或“同步至原图...”的提示）。
2. `fusion-editor.js` 中的处理函数会读取当前选中的模式组件状态，进行分支派发：
   - 若处于 **「独立资产」** 模式（默认），调用图层管理方法：`handleIsolatedAssetEdit(itemId, prompt)`
   - 若处于 **「场景融合」** 模式，调用图层管理方法：`handleQuickFusionSync(itemId, prompt)`

---

## 第三步 (分支A)：「独立资产」模式 (Isolated Asset Mode)

**核心目标**：仅修改当前抠出的透明物体图层本身，将其作为一个纯透明的独立素材继续使用，绝对不附带原图背景。
**执行路线**：进入 `src/ui/layer-manager.js` 中的 `handleIsolatedAssetEdit` 函数。

1. **预处理 - 垫底色 (Image Padding)**：由于 AI 图像大模型（如 Gemini）在处理带有透明通道 (Alpha) 的图片时易出现边缘伪影或识别错误。系统首先会将抠出来的物体图层置于一张纯白色的 Canvas 背景 (`#FFFFFF`) 正中心。
2. **发送带约束的局部重绘指令 (Prompt)**：系统自动封装组合一个极度严格的内部 Prompt，与添加了白底的图像一起发送给大模型。
   > **Prompt示例指令**：
   > "Act as a precise image asset editor. Your task is to modify the object in the image according to this instruction: '{用户指令}'.
   > CRITICAL REQUIREMENTS: 
   > 1. BACKGROUND LOCK: The background MUST remain pure, solid white...
   > 2. OBJECT ISOLATION: Only modify the object itself..."
3. **大模型重绘**：模型理解诉求，接收【白底图 + 指令】，最终返回符合指令修改要求的 **带有实体白底的图片**（如：变为蓝色，同时保留或修改了逼真的质感光影）。
4. **二次抠图 (Background Removal)**：系统自动调用前端 AI 图像库 `@imgly/background-removal` 对大模型生成的这部分新白底图片执行二次去背处理。
5. **边缘像素净化 (Edge Cleanup)**：为防止二次去背后物体边缘残留因白底融合产生的“白边”，前端利用 Canvas 像素级算法对 Alpha 透明通道像素进行一次形态学膨胀操作和临界值阈值过滤，将瑕疵白边剔除。
6. **产出与放置**：将新生成的、没有任何背景颜色的崭新透明子图资产安置到当前工作台中（位于旧图层下方附近位置），以便用户直接用于排版。

---

## 第三步 (分支B)：「场景融合」模式 (Quick Fusion Sync Mode)

**核心目标**：在原图（主场景）中应用修改指令，使物体发生改变的同时，与其周围环境的光影、关联透视完美融合。
**执行路线**：进入 `src/ui/layer-manager.js` 中的 `handleQuickFusionSync` 函数。

1. **定位原图溯源**：查阅该子图记录的 `parentId`，在工作台中准确查找到当初提取出该子图的“原始主场景层”。
2. **计算实时精准蒙版 (Mask Registration)**：
   - **空间几何映射**：系统精确计算此子图此时此刻在工作台上的坐标、相对主场景的平移距离 (px)、缩放大小 (Scale)、旋转角度 (Rotation)。
   - **绘制反向蒙版**：在内存中绘制一张与主场景等大尺寸的黑白位图（Mask）。默认背景为纯黑，然后根据刚刚解算出的映射关系逆推回去，将子图投影到这张黑图的对应区域并画成纯白（遮住该区域以指定为编辑工作区）。
3. **主场景快照截取**：通过渲染函数 `renderSceneToCanvas` 获取此时当前主场景的视觉合并图像，作为基底图。
4. **提交 Inpainting（局部重绘）指令至大模型**：
   - 需要携带：主场景图快照 + 精确计算的反向黑白蒙版 + 文本Prompt。
   - **Prompt示例指令**: 
     > "{用户指令}. Change only the visual look of the object within the masked area. Keep the background and perspective of the rest of the scene perfect."
5. **大模型执行场景级光影融合**：大模型被指令与所携带的黑白蒙板约束，完全锁定（Lock）蒙版以外的像素保持不变；而在白色蒙版的内部完成物品的“换色”等逻辑，此过程会自动兼顾原场景的光照、阴影投射和纹理自然感。
6. **产出与放置**：系统获取到带有局部修改的全新一帧“大场景完整图片”，将其作为一个全新的原图对象排布进工作台，并将选取焦点（外发光特效）转移至新图。

---

## 第四步：行为记录与结束

无论进入哪一种模式，AI 生成任务周期跑完后都会执行下列结尾操作：

1. **界面回收**：关闭并销毁针对触发子图的专属悬浮编辑器控制台。
2. **操作记录存档**：主动向页面右侧的全局聊天/会话窗口派发一个交互记录事件日志。
   - 包含图文状态，如：`✅ 独立编辑 [xxx] 已为您生成新的透明资产...` 或 `✅ 同步融合 [xxx] 已完成更新...`
   - 将这幅新产生的图片渲染出缩略图，一并在聊天记录中列出。此设计也作为可视化的“返回指针”，供用户后续点击重新使用和对比效果。
