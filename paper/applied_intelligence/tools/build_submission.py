"""Build the local submission and reproducibility archives from audited artifacts."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'paper/applied_intelligence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(name,text):(P/name).write_text(text.strip()+'\n',encoding='utf-8')
A=read(P/'evidence/audit.json')
assert A['completed_folds']==list(range(5)) and A['documents']==276
C=read(P/'evidence/controlled_audit.json')
assert C['documents']==276 and len(C['pooled'])==16 and len(C['comparisons'])==3
assert read(P/'evidence/controlled_controller_replay.json')['all_saved_fields_equal']
V=read(P/'evidence/violation_omission_audit.json')
R=read(P/'evidence/robustness_audit.json')
assert V['endpoints']==1104 and V['controller_replay_all_fields_equal']
assert R['includes_candidate_only'] and R['documents']==276
V8=next(r for r in V['pooled'] if r['budget']==8)
COMMON=next(r for r in R['common_subsets'] if r['family']=='controlled')
TITLE='Identity-sensitive verification and learned repair for language-model compilation of laboratory protocols'

write('cover_letter.txt',f'''Dear Editor,

Please consider our manuscript, "{TITLE}", for publication as a research article in Applied Intelligence.

The study examines a limitation of language-model compilation of laboratory protocols: symbolic verification can accept a program whose object mentions denote unintended entities. We combine induced constraints, bounded source-constrained repair and trained proposals, and evaluate accepted programs using existing X-WLP entity annotations.

In retrospective five-fold evaluation of 276 protocols, 15.5-22.2% of repaired programs accepted by the string verifier fail annotation-derived replay. For the fine-tuned 14B compiler, deterministic repair increases jointly confirmed programs from 97 to 142; a trained cascade reaches 188. A controlled study fixes the architecture, compiled inputs and generation ceilings without fallback. At the eight-sequence ceiling, producer feedback confirms 180 programs, versus 149, 146 and 144 for the original controls. A later candidate-preserving control omitting only explicit violations confirms {V8['confirmed']}, with 38:0 paired gains:losses. Scoring sensitivity preserves the aggregate contrasts while exposing incomplete strict-resolution coverage. Actual inference use differs, and individual regressions are reported.

The paper connects neuro-symbolic procedural reasoning to the validity of its evaluation oracle. The concise main article presents the central methods and comparisons; Online Resource 1 retains complete fold-level and statistical tables, resource accounting, automatic diagnostics and saved-output CPU replay. Limitations include retrospective development choices, one seed, supplied anchors and inventories, heuristic entity resolution and projected state semantics. We claim neither physical laboratory execution nor independent human validation. Generative-AI assistance in manuscript preparation is disclosed.

Thank you for considering this work.

Sincerely,
Xujian Zeng
Corresponding author
Changsha IMADEK Intelligent Technology Co., Ltd.
Changsha 410100, Hunan, China
Email: zengxujian5@gmail.com
''')

write('author_information.txt',f'''Manuscript: {TITLE}
Author order and contact information transcribed from the author reference supplied by the requester. Authors must confirm that affiliations, contacts and contributions apply to this study.

1. Xujian Zeng — corresponding author
Email: zengxujian5@gmail.com
ORCID: 0009-0005-9627-1327
Affiliation: Changsha IMADEK Intelligent Technology Co., Ltd.
Full address: Rooms 601-602, Building 1, Phase II, Kaiyang Intelligent Manufacturing Industrial Park, No. 1306 Kaiyuan East Road, Xingsha Industrial Base, Changsha Economic and Technological Development Zone, Changsha 410100, Hunan, China.

2. Xin Lu
Email: xinlu5417@126.com
Affiliation: School of Information Engineering, Guilin Institute of Information Technology.
Full address: No. 9 Yantu Road, Lingui District, Guilin 541004, Guangxi, China.

3. Shuaikang Wu
Email: shwaikang@gmail.com
Affiliation: Laibin Survey Team, National Bureau of Statistics of China.
Full address: 16th Floor, Investment and Development Building, No. 44 Renmin Road, Laibin 546100, Guangxi, China.

The manuscript uses concise institutional addresses. These full addresses may be used in the submission system after author verification.
''')

write('submission_notes_zh.txt',f'''Applied Intelligence 投稿说明（2026-10-05）

稿件：{TITLE}
目前为已完成五折主实验、补充对照及技术复核的本地投稿稿，尚未向期刊提交，也不能保证录用。投稿前仍需作者确认声明和最终稿。

一、文件
本轮按用户要求压缩至14页（包含参考文献），保留3张核心表、2幅图；字体、字号、页边距和模板不变。原完整18张表、扩展方法及2幅补充图移入ESM中的supplementary_details.html及可编辑.tex片段；实验结果不变。
manuscript.tex / manuscript.pdf：当前编辑器中的同一篇论文源文件及PDF。
refs.bib、sn-jnl.cls、sn-basic.bst、Fig1.pdf、Fig2.pdf（本地位于figures/，源文件投稿包中置于根目录）：编译所需文件。
figures/Fig1-2.eps：期刊可用的矢量图。
ESM_1.zip：Online Resource 1，包含实验数据划分、保存的预测/修复、规则、配置、复核代码、逐文档结果和原始语料许可证。
source_files.zip：可编辑LaTeX源文件及图。
cover_letter.txt：英文投稿附信。
author_information.txt：作者顺序、完整地址和联系方式。
Applied_Intelligence_submission_package.zip：以上投稿文件的本地汇总包。

二、五折结果
测试协议总数276，操作总数3869；固定的3个few-shot示例始终留在训练集。
Few-shot 14B：首遍4；确定性修复45；未训练提议器49；训练式级联77。
Fine-tuned 3B：首遍71；确定性修复134；训练式级联167。
Fine-tuned 14B：首遍97；确定性修复142；训练式级联188。
以上都使用“完整字符串验证器通过 + 注释实体身份状态回放通过”的统一条件。
六项确定性/训练式改进的Holm校正p值均不超过9.32e-10；未训练提议器的比较p=0.125。
统计检验条件于已保存的模型，不代表多训练种子独立复现。
原开发集42篇全部进入五折测试，稿件已明确说明这是设计后进行的回顾性重采样，不是嵌套交叉验证或外部独立测试。

新增固定架构与生成上限的对照：同一批276篇fine-tuned14B编译输出，无备用模型和示例，比较四组单14B提议器。在每篇最多8次生成的上限下，生产步骤反馈180篇、失败节点反馈149篇、删去反馈输入块146篇、关闭微调adapter144篇；P相对三对照的增益/损失分别31/0、37/3、37/1，三项独立成组的Holm校正p分别1.86e-9、1.95e-8、8.51e-10。
实际生成序列数270/166/192/201并不相同，不能称为计算量完全相等。删去的反馈块同时包含当前候选算子和违规信息，不能解释为只去掉违规文字。微调训练目标仍未分别消融。
四组在1/2/4/8次上限的所有结果、推理token/时间、逐折结果、Wilson区间及反例均完整报告。20组新运行产生829条序列，复核了4416个预算终点，并用保存的模型输出重放控制流程；所有保存字段一致。原主实验和原42篇测试结果未更改。
去掉原开发集42篇后，剩余234篇仍保留改进方向；这是事后敏感性分析，不能把它称为独立验证。原14B级联的46篇净改进中，39篇已由主模型完成，备用模型再增加7篇。

本次进一步补强：新增C组只删除违规字段，保留候选算子和原有提示指令、生产步骤选择及提议筛选，不能称为完全没有验证器信息。五折全部完成；8次生成上限下确认{V8['confirmed']}/276篇、字符串验证接受{V8['accepted']}篇、实际生成{V8['calls']}条序列。P相对C的配对增益/损失为{V['comparison']['gain']}/{V['comparison']['loss']}，双侧精确p={V['comparison']['p']:.6g}；这是独立的一项事后补充比较，不是原三项预定检验之一，也不声称预注册。
新增5组运行的1104个预算终点、原始生成输出及全部控制器字段已用CPU复核。旧主实验和旧四组对照均未改动或重新生成。
评分敏感性保持参考实体库存固定。去掉存活实体偏好后，原P/N/O/U四组分别为178/147/144/142篇。严格唯一精确匹配将无法解析的输出记为不可判定，同时报告覆盖率；五个对照组共同可解析子集为{COMMON['documents']}篇，属于输出相关的选择子集，不能外推为总体准确率或人工验证。全部五折的流程长度、实体数、依赖距离和同名异物分层，以及自动首个失败原因均已归档，不用分层结果挑选有利实验。
证据：evidence/violation_omission_audit.json、robustness_audit.json、robustness_details.json及相应CSV。实验方案在生成新结果前写入results/robustness_extension/protocol.json；CPU读取旧档案时处理缺失的不完整REVIEW输出的修正记录于analysis_implementation_amendment.json，原代码和协议保留。

三、投稿前必须由作者核实的事实
作者姓名、顺序和单位已按指定参考PDF填写。
经费已于2026-10-05由作者明确确认：本研究没有经费来源。论文Funding声明为“No funding was received for conducting this study.”，此项已完成确认。
利益冲突和CRediT贡献仍是从参考PDF暂拟移入，尚待作者确认适用于本研究；经费确认不代表利益冲突、贡献分配或全体作者批准也已确认。当前利益冲突文字为“无相关利益冲突”，贡献分配见稿件；若实际情况不同，须据实修改。
请所有署名作者通读、核实并同意最终稿；在投稿系统中如实确认原创性、是否同时投往其他期刊和作者同意情况。附信没有替作者预先作出这些未经确认的声明。
方法与声明部分已披露ChatGPT Codex用于写作、复核脚本和作图辅助；AI没有被列为作者，也没有生成实验观测值。

四、投稿定位和解释范围
亮点是验证/修复系统的实体身份敏感性及其评估方法，而不是端到端自动实验室执行。
论文保留金标动作锚点和初始库存、模板投影状态、启发式共指解析、单语料单种子等限制。新增对照固定架构、输入和最大生成上限，原级联仍属于完整系统比较；并未证明通用效率最优或解决全部混杂因素。
本地代码与保存输出支持CPU重新计分；归档不含预训练模型和adapter权重，不把该归档称为无需模型下载即可重训。
作者表示目前无法开展人工盲审，因此不列为本轮待完成任务。已有manual_audit/reviewer_packet.zip只是40个匿名案例、说明和空白表单，没有人工评分，不能声称人工核验、专家一致性或Kappa已完成。选择键不随审阅包或ESM分发。新增自动稳健性检查也不能代替人工语义评估。

五、格式与技术检查
篇幅参考：抽样Applied Intelligence 54(24)、Neural Computing and Applications 37(1)、Soft Computing 29(1)，排除勘误、撤稿说明和综述后共74篇研究文章，出版页数中位数18.5；45篇为12–20页。这不是全部SCI的统计，也不将出版页数等同于投稿页数。来源和逐篇页码保存在evidence/length_benchmark.json。
采用Springer sn-jnl及数字编号文献，摘要在150-250词范围，6个关键词，含作者声明、数据与代码可得性和补充材料说明。
官方指南：https://link.springer.com/journal/10489/submission-guidelines
投稿入口：https://www.editorialmanager.com/apin/
内置编辑器编译器曾报告平台标准目录错误；项目已有Tectonic可编译同一源文件为同名PDF，不需要安装TeX。最终技术检查见evidence/quality_check.json（若已生成）。
''')

files=set()
def add(p):
    p=ROOT/p if isinstance(p,str) else p
    assert p.is_file(),p
    files.add(p)
def tree(folder, suffixes=('.json','.jsonl')):
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts:add(p)
for folder in ('sop2program','scripts','tests'):tree(folder,('.py',))
for folder in ('data/processed','data/cv','results/symbolic','results/annotation_states'):tree(folder)
add('results/controlled_repair/protocol.json')
add('results/robustness_extension/protocol.json')
add('results/robustness_extension/analysis_implementation_amendment.json')
for k in range(5):
    path=f'results/robustness_extension/fold{k}/producer_candidate_only.json'
    assert read(ROOT/path)['complete'],path
    add(path)
for k in range(5):
    for arm in ('producer_feedback','node_feedback','producer_no_feedback','untuned_producer'):
        path=f'results/controlled_repair/fold{k}/{arm}.json'
        assert read(ROOT/path)['complete'],path
        add(path)
tree('data/raw/xwlp/data',('.peg','.txt'))
for name in ('LICENSE','README.md','call-graph-format.md'):add('data/raw/xwlp/'+name)
for name in ('requirements.txt',):add(name)
for k in range(5):
    base=f'results/cv/fold{k}'
    tree(base+'/knowledge')
    for model in ('qwen14','lora3b','lora3b_repair'):
        for name in ('config.json','training.jsonl','adapter/adapter_config.json'):
            p=ROOT/base/model/name
            if p.exists():add(p)
    for c in ('fewshot3','lora3b','qwen14'):
        tree(base+'/eval_'+c)
        for kind in ('deterministic','trained',*(['untuned'] if c=='fewshot3' else [])):
            tree(base+f'/repair_{kind}_{c}')
            add(base+f'/recheck_{kind}_{c}.json')
add('results/cv/first_pass_consistent.json')
# All original-split replay dependencies, including the privileged-identity study.
for r in read(P/'evidence/original_replayed.json'):
    tree(r['repair'])
    summary=read(ROOT/r['repair']/'summary.json')
    tree(summary['workflows'])
    cfg=ROOT/summary['workflows']/'run_config.json'
    if cfg.exists():
        adapter=read(cfg).get('adapter')
        if adapter and adapter!='none':
            parent=(ROOT/adapter).parent
            for name in ('config.json','training.jsonl'):
                if (parent/name).exists():add(parent/name)
for name in ('results/hypothesis_opt/summary.json','results/hypotheses_v4.json','results/h5_qwen3_14b_judge.json'):
    if (ROOT/name).exists():add(name)
for model in ('Qwen3-14B-FP8','Qwen2.5-3B-Instruct'):
    for name in ('config.json','download_manifest.json'):
        add(f'models/{model}/{name}')
for name in ('audit_submission.py','audit_original.py','publication_figures.py','finish_cv_audit.py','audit_strengthening.py','controlled_figure.py','audit_robustness.py','audit_robustness_v2.py','render_supplement.py'):
    add(P/'tools'/name)
for p in (P/'evidence').iterdir():
    if p.suffix in ('.json','.csv') and '_interim' not in p.name and p.name not in ('package_validation.json','manual_audit_selection_key.json'):add(p)
for p in (P/'figures').glob('Fig[12].pdf'):add(p)
for p in (P/'supplement').iterdir():
    if p.is_file():add(p)

readme=f'''Online Resource 1 for:
{TITLE}
Authors: Xujian Zeng; Xin Lu; Shuaikang Wu.
Correspondence: zengxujian5@gmail.com
Affiliations: Changsha IMADEK Intelligent Technology Co., Ltd., China;
School of Information Engineering, Guilin Institute of Information Technology, China;
Laibin Survey Team, National Bureau of Statistics of China, China.

READING GUIDE
The main article is shortened to 14 pages, including references, with 3 tables
and 2 figures. Full numerical detail is retained in
paper/applied_intelligence/supplement/supplementary_details.html (open locally
in a browser; retain its adjacent PNG files). Sections S1-S6 contain extended
methods, original-split evidence, diagnostics, all 18 tables and 2 extra plots.
The adjacent .tex file is an editable source fragment, not a second manuscript.
Table labels in table_mapping.json map the previous full version to S1-S18.
No experiment was changed or rerun during this editorial shortening.

SCOPE
Five completed folds: 276 distinct protocols and 3869 operators. Three fixed
demonstration protocols are always in training. There are 35 completed
compiler/repair arms and 20 archived original-split rechecks.
The supplementary controlled study adds 20 fold/arm runs, 829 generated
sequences and 4416 document/arm/budget endpoints (four arms, four ceilings).
All four arms, including individual regressions, are retained. Equal maximum
allowances do not imply equal actual token or generation use.
The later candidate-only control adds 5 fold runs and 1104 budget endpoints;
all 5 controlled arms are scored under three pre-specified resolution policies.
Strict coverage and common-subset outcomes, complete complexity strata and
automatic first-failure diagnostics are supplied without human-label claims.
This archive supports CPU replay of saved outputs, not physical lab execution.
Pretrained model and adapter weights are excluded. Configuration records and
training/inference source are supplied for researchers with the required models.

REPLAY (run from the extracted archive root; Python 3.13 was used)
  python -m pip install -r requirements-cpu.txt
  python -X utf8 paper/applied_intelligence/tools/audit_submission.py
  python -X utf8 paper/applied_intelligence/tools/audit_original.py
  python -X utf8 paper/applied_intelligence/tools/audit_strengthening.py --existing
  python -X utf8 paper/applied_intelligence/tools/audit_strengthening.py --controlled
  python -X utf8 paper/applied_intelligence/tools/audit_strengthening.py --controller-replay
  python -X utf8 paper/applied_intelligence/tools/audit_robustness_v2.py --omission
  python -X utf8 paper/applied_intelligence/tools/audit_robustness_v2.py --analyze
These regenerate evidence/audit.json, replayed_details.json, CSV tables,
original_replayed.json and original_first_metrics.json. No model is loaded.
The checks verify full final acceptance, source/inventory preservation,
entity-chain replay, cached outcomes, consistent first-pass scoring and splits.
The supplementary audit recomputes all endpoints, the three paired tests and
their separate Holm correction. Controller replay uses the saved model outputs
to reconstruct every prompt, decision, workflow and counter, without a model.
The frozen protocol also records hashes of large weight files not distributed
here; checking those weight hashes or rerunning GPU inference requires the
original models/adapters. They are not needed for any CPU command above.
The v1 audit source remains archived for provenance; use v2 for execution.
Its implementation amendment only handles absent final records for incomplete
REVIEW outputs, retaining original incomplete workflows. New GPU code and all
pre-specified scoring policies, bins and comparisons remained unchanged.
For figure generation, also install matplotlib and run
  python -X utf8 paper/applied_intelligence/tools/publication_figures.py
  python -X utf8 paper/applied_intelligence/tools/controlled_figure.py
The manuscript source is supplied separately; plotting also works without it.
The historical GPU requirements.txt is NOT required for CPU replay.
Do not regenerate data before auditing: the stored projections are those used
by the experiments. Retrospective design choices and annotation-derived inputs
are described in the article.

SCORING CAUTION
The old recheck field first_pass_lifecycle.chain_identity uses state-only replay.
Use evidence/audit.json and results/cv/first_pass_consistent.json for the joint
criterion in the article: full string verifier AND annotation-derived replay.
The second replay also reconstructs inventory and dependency links; it is not
a pure ID substitution or a physical execution oracle. Historical source
comments using 'upper bound' or 'identity alone' should be read with this
qualification and the manuscript's definition.
The supplementary feedback-block omission removes both the current candidate
operator and violation text; it is not an isolated removal of violation text.
The later candidate-only arm removes just the explicit violation field. It
retains the candidate, shared instruction, routing, and verifier-guided proposal
admission; it is not independent of verifier information. Its single additional
paired test is post hoc, separate from the prior three-test family.
The strict scoring policy abstains on unresolved/ambiguous references. Its
common subset is output-dependent and not a representative external test.
Removing original development documents is descriptive sensitivity, not
independent validation. No human semantic ratings or physical runs are claimed.

CONTENTS AND PROVENANCE
Project-relative paths are preserved. SHA256SUMS.json covers archived files.
Corpus: https://github.com/ronentk/textlabs-xwlp-data
Commit: db06ed1f27f9e9ef86a7324df5a0ee3607956db1
Tamari et al. (2021), doi:10.18653/v1/2021.eacl-main.187.
The upstream MIT license and README are under data/raw/xwlp/.
Derived artifacts and project code are supplied for review and reproducibility;
no new blanket license is asserted for author-owned files. Third-party model
weights are not redistributed. Model manifests record their upstream IDs.
'''
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
esm=P/'ESM_1.zip'
with zipfile.ZipFile(esm,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('README.txt',readme)
    z.writestr('requirements-cpu.txt','pydantic==2.13.4\nscipy==1.18.1\n')
    z.writestr('SHA256SUMS.json',json.dumps(hashes,indent=2))
with zipfile.ZipFile(esm) as z:assert z.testzip() is None

source=[P/name for name in ('manuscript.tex','refs.bib','sn-jnl.cls','sn-basic.bst')]
source+=list((P/'figures').glob('Fig[12].pdf'))
with zipfile.ZipFile(P/'source_files.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in source:z.write(p,p.name)
package=[P/name for name in ('manuscript.pdf','ESM_1.zip','source_files.zip','cover_letter.txt','author_information.txt','submission_notes_zh.txt')]+source
package+=list((P/'figures').glob('Fig[12].eps'))
with zipfile.ZipFile(P/'Applied_Intelligence_submission_package.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in package:z.write(p,p.name)
with zipfile.ZipFile(P/'Applied_Intelligence_submission_package.zip') as z:assert z.testzip() is None
print(json.dumps({'supplement_files':len(files),'ESM_bytes':esm.stat().st_size,
                  'submission_bytes':(P/'Applied_Intelligence_submission_package.zip').stat().st_size},indent=2))
