# 数据、模型与许可

本仓库不包含任何语料、派生数据或模型权重。以下路径均相对仓库根目录。

## X-WLP（主语料）

- 来源：Tamari et al., *Process-level representation of scientific protocols with interactive annotation*, EACL 2021。
- 仓库：<https://github.com/ronentk/textlabs-xwlp-data>，固定提交 `db06ed1f27f9e9ef86a7324df5a0ee3607956db1`，MIT 许可。

```powershell
git clone https://github.com/ronentk/textlabs-xwlp-data data/raw/xwlp
git -C data/raw/xwlp checkout db06ed1f27f9e9ef86a7324df5a0ee3607956db1
python -X utf8 scripts/prepare_data.py      # → data/processed/{train,dev,test}.jsonl, *_workflows.json, manifest.json
python -X utf8 scripts/audit_data.py        # 泄漏、偏移与弱监督检查
```

`prepare_data.py` 读取 `data/raw/xwlp/data/*/*.peg` 及同名 `.txt`。原始划分为按源文本去重后的文档级 70/15/15（种子 42，**不是**官方划分），之后的五折交叉验证由 `make_cv_folds.py` 生成（276 篇文档）。

## ChEMU 2020 + ChEMU-Ref（第二语料）

| 组成 | 来源 | 放置位置 |
|---|---|---|
| ChEMU 2020 事件抽取（EE）训练/开发集 | Mendeley Data，数据集 `wy6745bjfj`：<https://data.mendeley.com/datasets/wy6745bjfj>（下载需接受 Elsevier 有限数据许可） | 解压为 `data/raw/chemu/ee_train/`、`data/raw/chemu/ee_dev/`（`*.txt` + `*.ann`） |
| ChEMU-Ref 共指与桥接标注 | <https://github.com/biaoyanf/ChEMU-Ref>，`data/train.english.jsonlines`、`data/dev.english.jsonlines` | 重命名为 `data/raw/chemu/ref_train.jsonl`、`ref_dev.jsonl` |

```powershell
foreach ($s in 'train','dev') {
  Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/biaoyanf/ChEMU-Ref/main/data/$s.english.jsonlines" `
    -OutFile "data/raw/chemu/ref_$s.jsonl"
}
python -X utf8 scripts/prepare_chemu.py --output data/chemu --knowledge results/chemu/knowledge --dev-share .15
```

论文使用的推导结果：1,125 个参考程序通过两种重放，其中 633 个含同名实体链；划分为 766/134/225 篇（5,802/1,065/1,605 个操作），从训练部分挖掘 20 条规则，测试集可达上限 224。

**许可限制**：ChEMU 数据受 Elsevier 有限数据许可约束，仅限研究使用（以随数据提供的许可文本为准，建议将其保存为 `data/raw/chemu/ELSEVIER_LIMITED_DATA_LICENCE.pdf`）。原始文本、标注以及由其派生的任何文件（`data/chemu/`、`results/chemu/knowledge/`、含程序或文本的预测文件）都**不得再分发**。论文补充材料只发布代码、片段 ID 和汇总结果（`results/chemu/summary.json`、`document_outcomes.csv`）。

## 模型

| 用途 | 模型 | 目录 |
|---|---|---|
| 主编译器与提议器（FP8 冻结基座 + BF16 LoRA） | `Qwen/Qwen3-14B-FP8` | `models/Qwen3-14B-FP8` |
| 3B 编译器 | `Qwen/Qwen2.5-3B-Instruct` | `models/Qwen2.5-3B-Instruct` |

```powershell
python -X utf8 scripts/download_model.py --repo Qwen/Qwen3-14B-FP8 --source modelscope
python -X utf8 scripts/download_model.py --repo Qwen/Qwen2.5-3B-Instruct --source modelscope
# 或 --source huggingface --endpoint https://huggingface.co
```

下载支持断点续传并校验 SHA256，保存到 `models/<仓库名>`。`tests/test_fp8.py` 中有一个测试需要本地 Qwen2.5-3B 分词器，模型不存在时由 `tests/conftest.py` 自动跳过。

## DeepSeek API（仅外部 API 基线）

`sop2program/llm.py` 从仓库根目录的 `.env` 读取：

```
DEEPSEEK_API_KEY=...
DEEPSEEK_URL=https://api.deepseek.com   # 可选，默认即此；代码自动追加 /chat/completions
```

复制 `.env.example` 为 `.env` 并填入密钥。密钥不会写入日志或结果文件；`.env` 已列入 `.gitignore`，不要提交或分发。
