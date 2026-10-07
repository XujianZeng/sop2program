# SOP2Program

把实验室标准操作规程（SOP）编译成带类型的操作程序，用符号验证器检查，再做受源文本约束的修复；最后用标注的实体链重放已接受的程序，检查"验证器接受"是否真的对应正确的实体身份。

本仓库是论文 *Identity-sensitive verification and learned repair for language-model compilation of laboratory protocols*（投稿 Applied Intelligence）的全部实验代码。代码文件与产生论文结果时**逐字节一致**（见 `CODE_SHA256.json`）；部分脚本会对源码做指纹校验以保证断点续跑的一致性，因此**不要修改已有 `.py` 文件**，需要改动时请新建脚本或输出目录。

## 主要结论（论文数字）

| 实验 | 结果 |
|---|---|
| X-WLP 五折，276 篇，微调 14B 编译器 | 初次编译确认 97 篇 → 确定性修复 142 → 训练式级联 188 |
| 身份缺口 | 修复后被字符串验证器接受的程序中，15.5–22.2% 未通过实体链重放 |
| 受控对比（生成上限 8，无备用模型） | 生产者反馈 P 确认 180；N/O/U/C 对照 142–149 |
| 三个训练种子 | P 为 180/185/170（均值 178.3，SD 7.6）；D 均值 147.3；每个种子 P 对 D 都没有损失 |
| 同基座外部基线（base Qwen3-14B） | CRITIC 式 158、CLAIRify 式 160；P 显著更好（Holm 校正 p=0.0025） |
| API 模型外部基线（deepseek-flash） | Self-Refine 167、CRITIC 186、CLAIRify 189；与 P 差异不显著，CLAIRify 输出 token 为 P 的 11 倍 |
| ChEMU 第二语料（225 篇测试片段） | 确定性修复 71 → 150（79:0）；所有修复方法的接受中 28–29% 未通过重放；学习型与 API 修复最多再加 8 篇 |

"确认"（confirmed）= 完整字符串验证器通过 **且** 标注实体链状态重放通过。详细定义见 `docs/architecture.md`。

## 目录结构

```
sop2program/          核心库：IR、编译器、验证器、修复、规则归纳、身份重放、ChEMU 推导、LLM 后端
scripts/              数据准备、训练、评测、修复、各实验队列（扁平目录，脚本之间互相 import）
tests/                单元测试（pytest）
paper/applied_intelligence/tools/
                      论文复核脚本：只读已保存的输出，CPU 重算全部报告数字，生成图表与投稿归档
examples/             自定义 SOP 编译示例
docs/                 文档
CODE_SHA256.json      每个代码文件的 SHA256
```

运行时会在仓库根目录下使用（不随仓库分发）：

```
data/raw/xwlp/        X-WLP 原始语料（git clone，见 docs/data_and_models.md）
data/raw/chemu/       ChEMU 2020 + ChEMU-Ref 原始数据（需自行下载，受 Elsevier 许可限制）
data/processed/       由 prepare_data.py 生成的 X-WLP 程序与提示行
data/cv/              五折划分及每折训练行
data/chemu/           由 prepare_chemu.py 生成（不得再分发）
models/               Qwen3-14B-FP8、Qwen2.5-3B-Instruct 权重
results/              全部实验输出
.env                  DEEPSEEK_API_KEY（仅外部 API 基线需要）
```

## 环境

- Python 3.13；论文实验使用 Windows 原生 CUDA 12.8 环境，PyTorch 2.8.0、Transformers 4.57.6、PEFT 0.18.1。
- GPU：一张约 24 GB 显存的卡（论文使用 RTX 5090 D v2）。14B 编译评测在批大小 32 时需要约 23 GB；Windows 上显存不足会回退到系统内存，速度下降数倍乃至一个数量级以上（见 `docs/reproduce.md` 的"运行注意事项"）。
- 只做 CPU 复核和单元测试时，只需 `requirements-cpu.txt`。

```powershell
# CPU：单元测试与复核（先安装 CPU 版 torch==2.8.0）
python -m pip install -r requirements-cpu.txt
python -X utf8 -m pytest tests -q        # 无数据、无模型时：92 通过，9 跳过

# GPU：先按 CUDA 版本安装 torch==2.8.0，再安装其余依赖
python -m pip install -r requirements-gpu.txt
```

Windows 下所有命令建议加 `$env:PYTHONUTF8='1'` 和 `-X utf8`。

## 快速开始：编译自己的 SOP

```powershell
python -X utf8 scripts/compile_sop.py --input examples/lifecycle.txt `
    --anchors examples/lifecycle_anchors.json --inventory examples/empty_inventory.json `
    --adapter results/lora3b_full/adapter --rules results/symbolic/full_rules.json `
    --output results/lifecycle_workflow.json
```

输入为 UTF-8 文本、按执行顺序排列的动作跨度 `{start,end}`（end 不含）和初始物料清单。输出包含程序、逐步状态、原始模型输出、结构化违规和 PASS/REPAIRED/REVIEW。需要先训练 adapter 并归纳规则（见 `docs/reproduce.md`）。

## 文档

| 文档 | 内容 |
|---|---|
| `docs/architecture.md` | 方法与代码模块的对应关系、程序表示、验证、修复、身份重放 |
| `docs/data_and_models.md` | X-WLP、ChEMU、模型的获取方式、许可与目录约定 |
| `docs/reproduce.md` | 论文每组实验的完整复现命令与运行注意事项 |
| `docs/results_map.md` | 论文每张表/图 → 产生脚本 → 结果文件 → 复核脚本 |
| `docs/scripts.md` | 每个脚本的用途 |

## 范围与限制

- 动作锚点和初始物料清单由标注提供；这是组件级编译任务，不是端到端 SOP 理解。
- "确认"检查的是与投影状态模型的一致性，不代表实验可在真实实验室执行；本仓库不连接任何实验设备。
- 实体链解析是启发式的，重放失败是诊断性估计。
- 论文的局限与解释边界以论文正文为准。

## 许可与数据

- 作者自有代码随论文审稿与复现提供，未另行声明开放许可证。
- X-WLP 语料为 MIT 许可（Tamari et al., EACL 2021）。
- ChEMU 2020 / ChEMU-Ref 受 Elsevier 有限数据许可约束，仅限研究使用，**原始数据及任何派生数据（包括 `data/chemu/` 与 `results/chemu/` 中含文本的文件）均不得再分发**。
- 模型权重来自 Qwen 官方发布，不随仓库分发。
