# FastSAM 图层提取基线与增强原则

## 适用范围

本文档是 2026-09-16 09:54–12:43 期间完成的 FastSAM 修改摘要，也是后续不同图片、不同图层问题的统一基线。后续修正必须在当前基线上增量加强，不得破坏已经验收的人物路径。

## 已完成的基线能力

### 性能与稳定性

- 请求内按模型变体和 `imgsz` 复用 image embedding。
- 多图层共享 image encoder；每个 BBOX 仍独立执行 prompt decoder。
- `SAM_SHARED_EMBEDDING=0/1` 控制共享能力。
- embedding 编码使用 inference mode，避免缓存计算图造成内存增长。
- 记录 image encoder 次数、prompt decoder 次数和各阶段耗时。
- 请求结束清理 embedding。
- 输出阶段先裁剪目标区域，再进行 BGR/BGRA 转换，减少整图复制。

### 通用主体恢复

恢复逻辑使用“能力 + 证据”，不按人物、食品、家具复制流程：

- `boundary_truncated`：主体是否触碰并可能超出语义 BBOX；
- `candidate_disagreement`：SAM 多候选是否存在结构差异；
- `context_conflict`：扩展区域是否撞到显式语义上下文；
- `preserve_core`：恢复是否保留原主体核心；
- `recovery_growth`：新增区域是否连通且增长受限；
- `edge_type`：硬边、软边、文字等只提供参数。

人物路径保留 SAM-B 首次结果、positive probe、必要时 SAM-L、候选验收和失败回退策略。

### 诊断与前端

- `SAM_DEBUG_DUMP=1` 保存 source、候选 mask、selected mask、alpha 和 summary JSON。
- 日志记录 encoder/decoder 次数、耗时、候选面积、bbox 和路由原因。
- 前端修复已提取/隐藏/重排图层后的 BBOX 选择错位，并有对应测试。

## 后续修改的硬性原则

1. 人物现有 boundary recovery、positive probe、B→L 路由和验收阈值不得改变。
2. 普通产品、家具、软边、文字路径不得因组合主体修正而改变。
3. 新逻辑必须由明确能力开关触发；证据不足时回退当前 mask。
4. 不使用“白色/低饱和”直接判定背景，避免误删白盘、白碗和浅色主体。
5. 每次修改必须做人物回归、单主体回归和组合主体回归，并比较 mask 面积、bbox、IoU 和最终 alpha。

## 当前待实现能力：compound component discovery

组合主体可能由主对象、器皿、小菜、装饰等多个视觉组件组成。组件发现能力需要：

- 用现有主体 mask 作为不可删除的 core；
- 复用当前请求 embedding，增加受控 prompt decoder 搜索扩展组件；
- 对新增组件逐个审核连接性、边界证据、上下文冲突和背景包络特征；
- 恢复顶部小碗、小菜等真实组件；
- 移除右侧、底部等没有物体边界支撑的米色背景包络；
- 组件发现失败时保留现有 B 结果，不影响人物路径。
