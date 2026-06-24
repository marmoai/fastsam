基于对当前应用代码（特别是 src/ui/layer-manager.js 和 src/ui/interaction.js）的深度分析，我已经完全掌握了目前的运行机制。
💡 现状深度剖析
目前系统在 handleQuickFusionSync 函数中，采用的是**“严格坐标映射”**逻辑：
依赖历史数据：系统高度依赖拆解时保存的 originalBbox、spawnLeft 和 spawnTop。
相对位移计算：通过计算图层在工作台上的当前位置与初始位置的差值（relDx, relDy），再按比例缩放回原图坐标系。
局部重绘（Inpainting）：生成一个精确的黑白蒙版（Mask），只让大模型修改蒙版区域。
这种模式的局限性：它本质上是在“修补原图”，而不是“全新创作”。一旦用户大幅度改变图层大小、旋转角度，或者想把图层放到一张全新的背景图上，现有的坐标映射逻辑就会彻底崩溃。
🚀 “自由构图模式”全新流程规划
既然图层的拖拽、缩放、旋转在前端已经完美实现，我们完全可以抛弃 originalBbox，转向**“所见即所得（WYSIWYG）”**的草图合成流。
以下是具体的实施规划：
阶段一：底层数据与层级关系的重构 (Data & Hierarchy)
目标：打破“父子图层”的强绑定，建立“底板（Canvas）”与“悬浮图层（Floating Layer）”的新关系。
确立“纯净底板”身份：
在 state.workbenchItems 中，为作为背景的图片增加属性 isBasePlate: true。
底板在工作台中应固定在最底层（z-index: 0）。
动态图层归属：
废弃强依赖的 parentId。当用户将一个拆解出的图层拖拽到“纯净底板”上方松手时，更新该图层的 targetBaseId = 底板ID。
Z轴排序（Z-Index）：
确保 workbenchItems 中记录每个图层的 zIndex，以便在合成时知道谁遮挡了谁。
阶段二：构建“所见即所得”的草图渲染引擎 (Sketch Rendering Engine)
目标：彻底废弃 handleQuickFusionSync 中的坐标逆向映射，改为正向的 Canvas 绘制。
我们需要在 src/graphics/ 下新增一个 composition-utils.js，实现 generateCompositionSketch(baseId) 函数：
创建离屏 Canvas：尺寸与“纯净底板”的真实像素尺寸（naturalWidth/Height）完全一致。
绘制底板：将底板图片 1:1 绘制到 Canvas 上。
遍历并绘制图层：
查找所有 targetBaseId === baseId 的图层，按 zIndex 升序排列。
DOM 到 Canvas 的坐标系转换（核心）：读取图层在工作台上的真实 DOM 属性（offsetLeft, offsetTop, offsetWidth, offsetHeight, transform: rotate）。
计算这些 DOM 坐标相对于底板 DOM 的比例，然后映射到底板的真实像素坐标上。
使用 ctx.translate 和 ctx.rotate 将图层精准绘制到 Canvas 上。
输出草图：生成一张包含底板和所有摆放好图层的 sketchDataUrl。
阶段三：大模型融合生成 (AI Fusion Generation)
目标：将粗糙的拼接草图，转化为光影和谐的最终大片。
修改 API 调用逻辑：
不再需要生成黑白 Mask（因为不再是局部重绘）。
直接将阶段二生成的 sketchDataUrl 作为唯一输入图发给 Gemini（或对应的生图模型）。
重写 Prompt 策略：
之前的 Prompt 是：“在蒙版区域内重绘”。
新的 Prompt 策略：“这是一张粗略的构图草图（Composition Sketch）。请将其转化为一张高度逼真、光影和谐的最终图像。必须严格遵循草图中物体
的空间位置、大小比例、层级遮挡关系以及材质特征。你需要统一全局的光照、阴影、透视关系，并完美融合物体边缘。”
结果替换：将 AI 返回的最终高清图，作为一个新的 isBasePlate: true 的项目添加到工作台，完成闭环。
阶段四：UI/UX 优化升级 (UI/UX Enhancements)
目标：让用户的操作更符合直觉。
图层面板升级 (src/ui/layer-manager.js)：
激活现有的图层面板，支持拖拽列表项来改变图层的 zIndex。
合成触发按钮：
在底板的工具箱（workbenchToolbox）中，新增一个 “✨ 融合生成 (Synthesize)” 按钮。只有当底板上存在关联图层时才亮起。
视觉反馈：
当拖动图层经过底板上方时，底板边缘发光，提示用户“松手即可将图层放置于此底板”。
总结
这个规划完全抛弃了历史包袱（BBox映射）。前端只负责一件事：把用户在屏幕上拼好的画面，原封不动地“截图（Canvas渲染）”下来。剩下的
光影融合、透视修正、边缘处理，全部交给大模型去解决。这不仅大大降低了前端坐标计算的复杂度，还能实现真正的“自由构图”。