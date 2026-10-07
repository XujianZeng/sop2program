# 代码索引

各文件开头的文档字符串有更详细的说明；带 `argparse` 的脚本可用 `--help` 查看参数。

## 核心库 `sop2program/`

| 模块 | 内容 |
|---|---|
| `ir.py` | 程序表示（`Workflow`、`Step`、`Operator`、`Argument`、`Evidence`），源文本定位（`ground`），动作模板到前置条件/效果的投影（`semantic_template`、`make_step`） |
| `verify.py` | 符号验证器 `verify`（状态、证据、不变式三类检查），确定性修复 `minimal_repair`（束宽 4、16 轮、512 个候选）与 `settle_unavailable`，编辑代价与修复目标 |
| `induce.py` | 算法 B：从训练集归纳规则，按文档级 Beta 准则准入 |
| `compiler.py` | 提示构造、少样本示例选择、`LocalCompiler`（逐步带状态的批量受约束解码） |
| `constrained.py` | LM Format Enforcer 的 JSON 模式约束适配层 |
| `fp8.py` | 冻结的块 FP8 基座权重 + BF16 LoRA（Qwen3-14B-FP8 训练与推理） |
| `repair.py` | 算法 C 的可选 LLM 节点提议，只由外部验证器裁决 |
| `pipeline.py` | 对外 API：锚定 SOP → 编译 → 重放 → 有界修复 |
| `annotation.py` | 由 X-WLP 标注构造实体链，用于身份重放（"确认"判定） |
| `identity.py` | 不用标注的预测身份（探索性分析） |
| `chemu.py` | 由 ChEMU 2020 + ChEMU-Ref 推导参考程序与实体链 |
| `baselines.py` | H1 步骤图基线、H2 Declare 基线 |
| `llm.py` | 外部基线的对话后端（DeepSeek API 或本地模型）；密钥只从 `.env` 读取，不写入日志 |
| `metrics.py` | 锚定编译指标（缺失/无效步骤保留在分母中） |
| `numerics.py` | 拦截非有限值的模型输出与分数 |
| `paths.py` | 数据与知识库路径，可用 `SOP_DATA`、`SOP_KNOWLEDGE` 按折覆盖 |

## 数据准备 `scripts/`

| 脚本 | 用途 |
|---|---|
| `prepare_data.py` | X-WLP 原始 PEG → `data/processed/{split}.jsonl`、`{split}_workflows.json`、`manifest.json` |
| `audit_data.py` | 检查泄漏、源文本偏移与弱监督的局限（无模型） |
| `derive_annotation_states.py` | 比较文本身份与标注身份下的状态标签 |
| `make_cv_folds.py` | 文档级五折划分（同源文档同折，少样本示例固定在训练部分）并为每折挖掘规则 |
| `build_repair_lexicon.py` | 训练集触发词 → 动作清单（修复时只能改为语料支持的动作） |
| `build_setting_lexicon.py` | 训练集中被标为"设置"的提及 |
| `build_identity_lexicon.py` | 训练集共指中的中心词转移（预测身份用） |
| `build_repair_training.py` | 生产者反馈与节点反馈训练行（`--feedback-only`、`--node-feedback`、`--no-product-repeats`） |
| `prepare_chemu.py` | ChEMU 片段与 ChEMU-Ref 连接，生成第二语料（不得再分发） |
| `download_model.py` | 从 ModelScope 或 HF 镜像断点续传并校验 SHA256 |

## 训练、编译与修复

| 脚本 | 用途 |
|---|---|
| `train.py` | LoRA 监督微调，含六个辅助语义损失和反事实损失 |
| `train_supervisor.py` | GPU 驱动重置后自动续跑 `train.py`；输出目录放 `STOP` 文件可停止 |
| `evaluate.py` | 编译器评测：同一 SOP 内逐步带状态，跨 SOP 批量；可断点续跑并核对配置 |
| `evaluate_supervisor.py` | 带自动续跑的多方法评测 |
| `score.py` | 对保存的预测打分（无 GPU） |
| `score_first_pass.py` | 首遍质量，含文本身份与标注身份下的生命周期通过率 |
| `deterministic_repair.py` | 对保存的评测结果做不调用模型的确定性修复（D） |
| `producer_repair.py` | 生产者反馈修复与级联（T），只有通过有界修复才被接受 |
| `recheck_annotation_identity.py` | 用标注实体链重放最终程序，得到"确认"结果 |
| `compile_sop.py` | 编译用户自己的 SOP（给定动作跨度与初始物料） |

## 实验队列

| 脚本 | 论文位置 |
|---|---|
| `run_experiments.py` | 原始划分流水线（审计 → 符号实验 → 训练 → 评测 → 报告） |
| `symbolic_experiments.py` | 规则归纳与数值突变检测（补充表 S9） |
| `hypothesis_tests.py`、`hypothesis_extensions.py`、`h5_strong_judge.py` | 原始划分的 H1–H5 探索性检验（补充材料 S2） |
| `identity_pilot.py`、`check_oracle_identity.py` | 预测身份试点、特权链身份一致性检查（6.3） |
| `run_cv.py` | 五折主实验（表 1） |
| `cv_report.py`、`cv_first_pass_consistent.py`、`cv_lost_documents.py` | 五折汇总与配对检验、统一口径的首遍结果、修复后丢失的文档 |
| `bootstrap_ci.py` | 首遍指标与最终通过率的文档级 bootstrap 区间 |
| `controlled_repair.py` | 受控反馈对比 P/N/O/U（表 2、图 2） |
| `refinement_extension.py` | 后加对照 C：只删违规字段 |
| `seed_replication.py` | 训练种子复现（表 3） |
| `external_baselines.py` | Self-Refine、CRITIC、CLAIRify 风格基线与路由臂（表 4） |
| `chemu_pipeline.py` | ChEMU 冻结设计迁移（表 5） |
| `report.py` | 由已保存指标生成本地 HTML 报告 |

## 论文复核与归档 `paper/applied_intelligence/tools/`

| 脚本 | 用途 |
|---|---|
| `run_audits.py` | 统一入口：固定 `PYTHONHASHSEED=1`，依次运行下列全部 CPU 复核 |
| `audit_submission.py` | 五折主结果 CPU 重放 |
| `audit_original.py` | 原始划分结果 CPU 重放 |
| `audit_strengthening.py` | `--existing` 首选/级联分解；`--controlled` 受控实验；`--controller-replay` 控制器重放 |
| `audit_robustness.py`、`audit_robustness_v2.py` | 评分敏感性、分层与 C 臂审计（v1 冻结存档，v2 为实际使用版本） |
| `audit_extensions.py` | 外部基线、种子、ChEMU 的计数、区间、McNemar 与 Holm 重算 |
| `finish_cv_audit.py` | 五折完成后的就绪检查与证据刷新 |
| `publication_figures.py`、`controlled_figure.py` | 图 1、图 2 |
| `render_supplement.py` | 补充材料 LaTeX 片段 → 离线 HTML |
| `build_submission.py` | 生成投稿包与 ESM 复现归档 |
| `validate_submission_archive.py` | 解压归档后校验哈希、许可限制与密钥，并重跑全部复核 |

## 测试 `tests/`

`python -X utf8 -m pytest tests -q`。需要真实数据或模型的测试在缺少文件时自动跳过（`tests/conftest.py`）。
