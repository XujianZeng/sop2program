# 论文结果 → 脚本 → 结果文件 → 复核

"复核"列是不加载模型的 CPU 脚本（位于 `paper/applied_intelligence/tools/`），从保存的输出重算论文中的数字，结果写入 `paper/applied_intelligence/evidence/`。命令见 [reproduce.md](reproduce.md) 第 7 节。

| 论文位置 | 生成脚本 | 结果文件（`results/` 下） | 复核脚本 → evidence 文件 |
|---|---|---|---|
| 表 1 五折主结果（F/D/T，276 篇） | `run_cv.py`、`cv_report.py` | `cv/foldK/eval_*`、`cv/foldK/repair_*`、`cv/report.json` | `audit_submission.py` → `audit.json`、`document_outcomes.csv`、`fold_results.csv`、`paired_comparisons.csv` |
| 正文 5.1 首选/级联分解 | 同上 | 同上 | `audit_strengthening.py --existing` → `strengthening_existing.json` |
| 表 2、图 2 受控反馈（P/N/O/U） | `controlled_repair.py` | `controlled_repair/protocol.json`、`controlled_repair/foldK/{producer_feedback,node_feedback,producer_no_feedback,untuned_producer}.json` | `audit_strengthening.py --controlled` → `controlled_audit.json`、`controlled_document_outcomes.csv`；`--controller-replay` → `controlled_controller_replay.json`；图：`controlled_figure.py` → `figures/Fig2.*` |
| 表 2 后加对照 C | `refinement_extension.py` | `robustness_extension/protocol.json`、`robustness_extension/foldK/producer_candidate_only.json` | `audit_robustness_v2.py --omission` → `violation_omission_audit.json`、`violation_omission_outcomes.csv` |
| 表 3 训练种子 42/43/44 | `seed_replication.py` | `seeds/summary.json`、`seeds/document_outcomes.csv`、`seeds/seed4X/foldK/*`（种子 42 即 `cv/` 与 `controlled_repair/`） | `audit_extensions.py` → `extensions_audit.json`（`seeds`） |
| 表 4 外部基线（API 与同骨干） | `external_baselines.py` | `external_baselines/{deepseek-flash,qwen14-base}/{summary.json,document_outcomes.csv,foldK/<arm>.json}` | `audit_extensions.py` → `extensions_audit.json`（`external`、`self_refine_feedback`、`external_generation`） |
| 表 5 ChEMU 第二语料 | `prepare_chemu.py`、`chemu_pipeline.py` | `chemu/summary.json`、`chemu/document_outcomes.csv`（其余 ChEMU 派生文件受许可限制，不对外发布） | `audit_extensions.py` → `extensions_audit.json`（`chemu`） |
| 图 1 流程图 | — | — | `publication_figures.py` → `figures/Fig1.*` |
| 正文 6.2 评分敏感性、补充表 S8/S11/S17 | — | 受控与主结果输出 | `audit_robustness_v2.py --analyze` → `robustness_audit.json`、`robustness_details.json` |
| 补充表 S18 长度/歧义分层 | — | 同上 | 同上 → `robustness_strata.csv` |
| 补充材料 S2 原始划分（195/42/42）、表 S6 | `run_experiments.py`、`producer_repair.py`、`deterministic_repair.py` | `evaluation*/`、`repair_*`（原始划分） | `audit_original.py` → `original_replayed.json`、`original_first_metrics.json` |
| 补充表 S9 符号检查器、合成故障 | `symbolic_experiments.py` | `symbolic/` | — |
| 正文 6.3 探索性分析（特权链身份、启发式解析器、H5 判别器） | `check_oracle_identity.py`、`identity_pilot.py`、`hypothesis_tests.py`、`h5_strong_judge.py` | `hypotheses_v3*.json`、`h5_qwen3_14b_judge.json`、`annotation_states/identity_pilot.json` | — |
| 补充表 S19–S21（各上限的外部基线、种子、ChEMU） | 同表 3–5 | 同表 3–5 | `audit_extensions.py` |

补充材料表号与 LaTeX 标签的对应见 `paper/applied_intelligence/supplement/table_mapping.json`（论文仓库内）。

## 关键数字速查

| 结果 | 数值 |
|---|---|
| 微调 14B：F / D / T 确认（276 篇） | 97 / 142 / 188 |
| 受控上限 8：P / N / O / U / C 确认 | 180 / 149 / 146 / 144 / 142 |
| 种子 42 / 43 / 44 的 P | 180 / 185 / 170 |
| 同骨干 CRITIC / CLAIRify 风格 | 158 / 160 |
| ChEMU：F / D 确认（225 篇） | 71 / 150 |
