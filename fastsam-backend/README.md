# FastSAM 极速拆层本地验证服务

## 后端模块边界

`main.py` 只负责进程启动和自检；请求与分割逻辑按职责拆分，避免模型生命周期、语义策略和 HTTP 状态互相污染：

- `api.py`：FastAPI 应用、`/healthz`、`/segment` 请求/响应适配。
- `sam_runtime.py`：模型路径、下载缓存、SAM-B/SAM-L 互斥生命周期、推理入口和 CUDA 内存回收。
- `segmentation_policy.py`：语义元数据到策略 profile 的解析；不执行推理。
- `mask_ops.py`：几何、连通域、孔洞修复、bbox 约束等无模型操作。
- `segmentation_primitives.py`：跨模块共享的提示点、边界负点和 mask 栅格化原语。
- `matte_ops.py`：提示点、alpha/matte 和局部细化。
- `candidate_selection.py`：候选评分、排除关系和语义仲裁。
- `segmentation_core.py`：B/L 路由、completion 二次 SAM（强制 SAM-L）和 cutout 组装。
- `diagnostics.py`：不依赖模型权重的确定性回归自检。

所有分类仍通过同一个 `/segment` 契约进入原有策略；只有带 completion 标记的二次分割请求强制走 SAM-L，首次提取和其他分类的路由不变。

正如您在 `implicit_memory_and_telemetry.md` 中提出的“方案3——最快验证”思路，这里是一个完整的 Python 本地微服务，它使用 Ultralytics 的 FastSAM 对图像进行全自动语义拆层。

为了在工作台中验证这一方案，我已经修改了前端代码，在“语义图层”面板（Layer Manager）中增加了一个绿色的 **“FastSAM 测试”** 按钮。当您选中画布上的一张图片并打开图层管理器时，点击该按钮，它就会将当前图片以 Base64 发送到您本地运行的此 Python 服务进行拆分。

## 启动方法

请在您的**本地物理机**（具有 Python 环境）上执行以下步骤：

1. **安装依赖:**
   ```bash
   cd fastsam-backend
   pip install -r requirements.txt
   ```

2. **准备模型权重:**
   当前验证硬物体质量默认使用 `FastSAM-x.pt`。请把权重文件放到 `fastsam-backend/FastSAM-x.pt`。

   如果要临时回到小模型，可以使用环境变量：
   ```bash
   FASTSAM_MODEL_PATH=FastSAM-s.pt python main.py
   ```

3. **启动服务:**
   ```bash
   python main.py
   ```
   *服务将运行在 `http://127.0.0.1:8000`。*

4. **测试验证:**
   回到 MarmoAid 网页版工作台：
   - 选中任意一张带有物体的图片。
   - 点击工具箱里的 **语义图层分析**（<i class="fas fa-layer-group"></i> 按钮）。
   - 在弹出的语义图层面板右上方，点击绿色的 **“<i class="fas fa-bolt"></i> FastSAM 测试”** 按钮。
   - 前端会将图片发给本地 8000 端口的模型，无需请求云端，瞬间完成各对象的 Mask 裁切、图层生成、Bbox 归一化。
   - 识别分离出来的透明层会自动排列并进入图层树（Layer Panel）。
