# 方法与代码结构

## 流水线

```
SOP 文本 + 动作锚点 + 初始物料
        │
        ▼
  编译器 (LoRA 微调 Qwen3-14B-FP8 / Qwen2.5-3B，或少样本)   sop2program/compiler.py, fp8.py, constrained.py
        │  每个锚点生成一个带类型的操作；逐步传入预测状态
        ▼
  验证器 verify()                                           sop2program/verify.py
        │  模式/证据/前置条件/生命周期/归纳规则 → 结构化违规
        ▼
  确定性修复 minimal_repair()  (D)                          sop2program/verify.py
        │  受源文本约束的局部补丁 + 束搜索，按 repair_objective 选择
        ▼
  学习型提议器 (P：生产者反馈)                              scripts/producer_repair.py, sop2program/repair.py
        │  把"缺失物体"反馈发回应当产出它的步骤；候选仍由验证器裁决
        ▼
  身份重放（只用于评估）                                    sop2program/annotation.py,
        标注实体链状态重放 → "确认"                          scripts/recheck_annotation_identity.py
```

## 程序表示（`sop2program/ir.py`）

- `Workflow`：`id`、`source`（原文）、`initial`（初始物料）、`steps`。
- `Step`：一个 `Operator`（动作、参数 `Argument`、状态变化 `Change`）加上 `Evidence`（原文字符跨度）。
- 所有模型均为 pydantic 严格模式（`Strict`），未知字段直接报错；`ground()` 把文本参数定位回原文偏移。

## 验证（`sop2program/verify.py`）

`verify(workflow, rules, state_checks, evidence_checks, invariants)` 返回违规列表，每条违规带步骤号、类型和可读反馈。检查包括：

- 模式与证据：参数必须能在原文中定位；
- 状态：投影状态模型（存在、属性）下的前置条件与生命周期；物体身份按**表面名称**解析（这是身份缺口的来源）；
- 归纳规则：由 `sop2program/induce.py`（文档级 Beta 准入）从训练折挖掘的时序/前置规则。

## 修复

- **D（确定性）**：`minimal_repair()` 在违规步骤附近生成补丁（`_patches`：替换动作、删除参数、插入生产者等），先应用无歧义编辑，再做最佳优先束搜索（束宽 4、最多 16 轮、512 个已验证候选，每轮展开前两个失败步骤），返回编辑代价最低的通过候选，否则标记 REVIEW。不调用任何模型。
- **P（生产者反馈）**：提议器 LoRA（`build_repair_training.py --feedback-only --node-feedback` 构造训练行）接收验证器反馈，为应当产出缺失物体的步骤生成新节点；`controlled_repair.py` 中的对照臂：
  - N：同一 adapter，改为报错节点反馈（failing-node feedback）；
  - O：同一 adapter，省略整个反馈输入块（候选操作与违规都去掉，但保留验证器引导的目标步骤选择）；
  - C：同一 adapter，只省略违规字段；
  - U：生产者反馈，但禁用 adapter（未微调基座）；
  - 所有臂共享规则、锚点、物料和最终搜索，生成上限（ceiling）相同，论文主比较为上限 8。
- **外部基线**（`scripts/external_baselines.py`，后端 `sop2program/llm.py`）：Self-Refine、CRITIC、CLAIRify 式修复与路由；同一验证器裁决，同一预算。

## 评估端点

- **接受**：完整字符串验证器通过。
- **确认**：接受 **且** 在标注实体链（X-WLP 共指；ChEMU 为 ChEMU-Ref 共指+桥接推导）下逐步重放状态仍然通过。
- **身份缺口** = 被接受但未通过重放的比例。
- 统计：精确 McNemar（成对文档）、Holm 校正族、Wilson 区间；见 `scripts/cv_report.py` 与 `paper/applied_intelligence/tools/audit_*.py`。

## ChEMU 推导（`sop2program/chemu.py`）

把 ChEMU 2020 事件抽取标注（反应步骤触发词与论元）与 ChEMU-Ref（共指、反应关联/工作流程桥接）合并，生成参考程序和标注实体链；只保留两种重放都能通过的参考。规则、提示、超参数沿用 X-WLP，不再调整（`scripts/chemu_pipeline.py` 冻结设计）。

## 交叉验证与路径（`sop2program/paths.py`）

五折实验通过环境变量切换数据与知识库：

- `SOP_DATA`：例如 `data/cv/fold0`（含 `train/dev/test.jsonl` 与每折训练行）；
- `SOP_KNOWLEDGE`：例如 `results/cv/fold0/knowledge`（每折规则与词表）。

所有挖掘（规则、修复词表、设定词表、提议器训练行）都只使用当折训练集。

## 断点续跑与一致性

- `evaluate.py` 从已有 `predictions.jsonl` 续跑；`run_cv.py`、`seed_replication.py`、`chemu_pipeline.py` 按步骤完成标记跳过已完成步骤。
- `external_baselines.py`、`seed_replication.py` 在开始时对一组源文件计算 SHA256 指纹，续跑时指纹不一致会拒绝运行。这就是"不要修改已有 `.py` 文件"的原因。
- `sop2program/numerics.py` 在保存前拒绝含 NaN/Inf 的模型输出或分数。
