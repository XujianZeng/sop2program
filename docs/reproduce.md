# 复现论文实验

所有命令在仓库根目录、PowerShell 下运行，并预先设置：

```powershell
$env:PYTHONUTF8 = '1'
$py = 'python'          # 或虚拟环境中的 python.exe
```

GPU 步骤都可断点续跑：中断后重跑同一条命令即可，已完成的步骤会被跳过。论文实验全部在单张 RTX 5090 D v2（24 GB）上串行完成；同一时间只运行一个 GPU 任务。

## 0. 准备

```powershell
# 数据与模型见 docs/data_and_models.md
& $py -X utf8 scripts/prepare_data.py
& $py -X utf8 scripts/audit_data.py
& $py -X utf8 scripts/download_model.py --repo Qwen/Qwen3-14B-FP8
& $py -X utf8 scripts/download_model.py --repo Qwen/Qwen2.5-3B-Instruct
& $py -X utf8 -m pytest tests -q
```

## 1. 原始划分（195/42/42）与规则归纳

```powershell
& $py -X utf8 scripts/symbolic_experiments.py          # → results/symbolic（规则、词表、突变检测）
& $py -X utf8 scripts/build_repair_lexicon.py
& $py -X utf8 scripts/build_setting_lexicon.py
& $py -X utf8 scripts/build_repair_training.py --output train_repair_fb --no-product-repeats
& $py -X utf8 scripts/build_repair_training.py --output train_feedback_only --feedback-only --node-feedback --no-product-repeats
```

原始划分上的少样本/零样本 14B 首遍输出是五折实验的共享输入（`run_cv.py` 中的 `SHARED`）：

```powershell
& $py -X utf8 scripts/evaluate.py --method base --model models/Qwen3-14B-FP8 --split test --few-shot 3 `
    --batch-size 32 --attention sdpa --pin-trigger --output results/evaluation_qwen14_fewshot3_test/base
& $py -X utf8 scripts/evaluate.py --method base --model models/Qwen3-14B-FP8 --split dev `
    --batch-size 32 --attention sdpa --pin-trigger --output results/evaluation_qwen14_zeroshot_dev/base
& $py -X utf8 scripts/evaluate.py --method base --model models/Qwen3-14B-FP8 --split test `
    --batch-size 32 --attention sdpa --pin-trigger --output results/evaluation_qwen14_zeroshot_test/base
```

`run_experiments.py`、`hypothesis_tests.py`、`hypothesis_extensions.py`、`h5_strong_judge.py`、`identity_pilot.py` 产生补充材料 S2 中的原始划分与探索性结果。

## 2. 五折交叉验证（论文表 1）

```powershell
& $py -X utf8 scripts/make_cv_folds.py --folds 5 --seed 42 --output data/cv --knowledge results/cv
foreach ($k in 0..4) {
  $env:SOP_DATA = "data/cv/fold$k"; $env:SOP_KNOWLEDGE = "results/cv/fold$k/knowledge"
  & $py -X utf8 scripts/build_repair_lexicon.py
  & $py -X utf8 scripts/build_setting_lexicon.py
  & $py -X utf8 scripts/build_repair_training.py --output train_repair_fb --no-product-repeats
  & $py -X utf8 scripts/build_repair_training.py --output train_feedback_only --feedback-only --node-feedback --no-product-repeats
}
Remove-Item Env:SOP_DATA, Env:SOP_KNOWLEDGE
& $py -X utf8 scripts/run_cv.py --phases ABC     # 训练 14B/3B/3B 反馈提议器、评测、三种修复
& $py -X utf8 scripts/cv_first_pass_consistent.py
& $py -X utf8 scripts/cv_report.py               # → results/cv/report.json（汇总、McNemar、Holm）
```

`make_cv_folds.py` 同时为每折挖掘规则（`results/cv/foldK/knowledge/full_rules.json`）。核对用的行数：第 0 折 `train.jsonl` 2,720 行，`train_repair_fb.jsonl` 2,979 行（多出 259 条生产者反馈行），`train_feedback_only.jsonl` 764 行；五折的反馈行数为 691–775。

## 3. 受控反馈对比（表 2、图 2）

```powershell
& $py -X utf8 scripts/controlled_repair.py --prepare     # 冻结协议 → results/controlled_repair/protocol.json
& $py -X utf8 scripts/controlled_repair.py --run         # P/N/O/U 四臂 × 五折
& $py -X utf8 scripts/refinement_extension.py --prepare  # 后加对照 C（只删违规字段）
& $py -X utf8 scripts/refinement_extension.py --run
```

## 4. 训练种子复现（表 3）

```powershell
& $py -X utf8 scripts/seed_replication.py --seeds 43 44          # 每个种子每折重训 14B adapter 并重新编译、修复
& $py -X utf8 scripts/seed_replication.py --seeds 42 43 44 --score   # → results/seeds/summary.json
```

## 5. 外部修复基线（表 4）

```powershell
Copy-Item .env.example .env   # 填入 DEEPSEEK_API_KEY
& $py -X utf8 scripts/external_baselines.py --backend deepseek-flash --arms self_refine critic clairify routing
& $py -X utf8 scripts/external_baselines.py --backend deepseek-flash --score
& $py -X utf8 scripts/external_baselines.py --backend qwen14-base --arms critic clairify   # 本地 GPU，未微调的 14B
& $py -X utf8 scripts/external_baselines.py --backend qwen14-base --score
```

可先用 `--folds 0 --max-docs 5` 试跑（写入单独的 pilot 目录）。API 模型可能被服务方更新，重跑不保证逐字一致；论文使用的原始输出全部保存在 `results/external_baselines/`。

## 6. ChEMU 第二语料（表 5）

```powershell
& $py -X utf8 scripts/prepare_chemu.py --output data/chemu --knowledge results/chemu/knowledge --dev-share .15
& $py -X utf8 scripts/chemu_pipeline.py --queue                                   # 训练、编译、D、P、U
& $py -X utf8 scripts/chemu_pipeline.py --external deepseek-flash --arms critic clairify routing
& $py -X utf8 scripts/chemu_pipeline.py --score                                   # → results/chemu/summary.json
```

## 7. CPU 复核（不加载模型）

```powershell
$t = 'paper/applied_intelligence/tools'
& $py -X utf8 "$t/run_audits.py"        # 固定 PYTHONHASHSEED=1，依次运行下列全部复核
& $py -X utf8 "$t/publication_figures.py"; & $py -X utf8 "$t/controlled_figure.py"
& $py -X utf8 "$t/render_supplement.py"
```

`run_audits.py` 的运行顺序：

| 脚本 | 复核内容 |
|---|---|
| `audit_submission.py` | 五折主结果（表 1） |
| `audit_original.py` | 原始划分结果（补充材料 S2） |
| `audit_strengthening.py --existing / --controlled / --controller-replay` | 首选/级联分解、受控实验、控制器重放 |
| `audit_robustness_v2.py --omission / --analyze` | C 臂、评分敏感性与分层 |
| `audit_extensions.py` | 外部基线、种子、ChEMU |

审计结果写入 `paper/applied_intelligence/evidence/`。Python 字符串哈希顺序只影响 `replayed_details.json` 中一个未在论文报告的描述性字段（第 5 条记录的 `fidelity_on_accepted.precondition_exact_chain`，变化 0.3–0.5 个百分点）；`run_audits.py` 固定 `PYTHONHASHSEED=1`，结果与归档值逐字节一致。单独运行某个复核脚本时，请先设置 `$env:PYTHONHASHSEED='1'`。这些复核脚本的哈希已被冻结协议记录，所以种子在入口脚本里固定，而不是写进脚本本身。

## 运行注意事项（Windows + 24 GB 显卡）

- 14B 评测（批大小 32）约需 23 GB 显存。Windows WDDM 在显存不足时会把张量溢出到共享内存而不报错：每批从约 45 秒变为 95 秒，严重时超过 1,000 秒。远程桌面串流（如 `GameViewerServer.exe`）和 `dwm.exe` 合计可占约 2.5 GB；长时间运行时断开远程桌面、关闭占显存的桌面程序。
- 查看各进程显存：`Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage'`。
- 设置 `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256` 实测没有缓解溢出（共享内存反而更高），论文运行未使用；`expandable_segments` 在 Windows 上不受支持。
- 不要为了省显存改批大小或注意力实现：续跑会核对配置，改动后已保存的预测不能接续，数值也可能不同。
- `train_supervisor.py`、`evaluate_supervisor.py` 可在 GPU 驱动重置后自动续跑；在输出目录放置 `STOP` 文件可让受监督的训练停止。
