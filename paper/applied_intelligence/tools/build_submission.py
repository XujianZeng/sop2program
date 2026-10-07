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
X=read(P/'evidence/extensions_audit.json')
assert not X['problems'] and X['chemu']['arms_at_cap8']['D_deterministic']['confirmed']==150
assert X['seeds']['across_seeds']['producer_feedback']['values']==[180,185,170]
TITLE='Identity-sensitive verification and learned repair for language-model compilation of laboratory protocols'

write('cover_letter.txt',f'''Dear Editor,

Please consider our manuscript, "{TITLE}", for publication as a research article in Applied Intelligence.

The study examines a limitation of language-model compilation of laboratory protocols: symbolic verification can accept a program whose object mentions denote unintended entities. We combine induced constraints, bounded source-constrained repair and trained proposals, and evaluate accepted programs using existing X-WLP entity annotations.

In retrospective five-fold evaluation of 276 protocols, 15.5-22.2% of repaired programs accepted by the string verifier fail annotation-derived replay. For the fine-tuned 14B compiler, deterministic repair increases jointly confirmed programs from 97 to 142; a trained cascade reaches 188. A controlled study fixes the architecture, compiled inputs and generation ceilings without fallback: at the eight-sequence ceiling, producer feedback confirms 180 programs (mean 178.3 over three training seeds, with no losses against deterministic repair in any seed), versus 142-149 for four feedback controls. Reimplemented CRITIC- and CLAIRify-style refinement on the same 14B backbone confirms 158 and 160 (Holm-adjusted p=0.0025); driven by a commercial API model, these strategies confirm 186-189, not significantly different from the fitted proposer, at up to 11 times its output tokens. On a second corpus derived from ChEMU and ChEMU-Ref, evaluated with the design frozen, deterministic repair raises confirmations from {X['chemu']['arms_at_cap8']['F_first_pass']['confirmed']} to {X['chemu']['arms_at_cap8']['D_deterministic']['confirmed']}, 28-29% of acceptances fail replay under every repair method, and refinement adds at most eight confirmations because few verifier-detectable errors remain.

The paper connects neuro-symbolic procedural reasoning to the validity of its evaluation oracle and shows that verifier-guided refinement, learned or prompted, is bounded by what the verifier can see. The main article presents the central methods and comparisons; Online Resource 1 retains complete fold-level, seed, external-baseline and statistical tables, resource accounting, automatic diagnostics and saved-output CPU replay. Limitations include retrospective development choices, supplied anchors and inventories, rule-derived ChEMU chains, heuristic entity resolution and projected state semantics. We claim neither physical laboratory execution nor independent human validation. The analysis code is provided in Online Resource 1 and at https://github.com/XujianZeng/sop2program.

We confirm that the manuscript is original, has not been published previously, and is not under consideration for publication elsewhere; it will not be submitted to another journal while it is under review with you. All authors have read and approved the submitted version and agree with its submission. The authors declare no competing interests, and the research received no external funding. ChatGPT Codex was used to assist with code generation and manuscript checking; the authors reviewed the results against the code and the saved experimental outputs, and take full responsibility for the content. This use is disclosed in the manuscript.

Thank you for considering this work.

Sincerely,
Xujian Zeng
Corresponding author
Changsha IMADEK Intelligent Technology Co., Ltd.
Changsha 410100, Hunan, China
Email: zengxujian5@gmail.com
''')

write('author_information.txt',f'''Manuscript: {TITLE}
Author order, contacts and ORCID iDs follow the authors' earlier submission record (ESWA-D-26-40712).

1. Xujian Zeng — corresponding author
Email: zengxujian5@gmail.com
ORCID: 0009-0005-9627-1327
Affiliation: Changsha IMADEK Intelligent Technology Co., Ltd.
Full address: Rooms 601-602, Building 1, Phase II, Kaiyang Intelligent Manufacturing Industrial Park, No. 1306 Kaiyuan East Road, Xingsha Industrial Base, Changsha Economic and Technological Development Zone, Changsha 410100, Hunan, China.

2. Xin Lu
Email: xinlu5417@126.com
ORCID: 0009-0006-4060-7773
Affiliation: School of Information Engineering, Guilin Institute of Information Technology.
Full address: No. 9 Yantu Road, Lingui District, Guilin 541004, Guangxi, China.

3. Shuaikang Wu
Email: shwaikang@gmail.com
ORCID: 0009-0001-7803-7714
Affiliation: Laibin Survey Team, National Bureau of Statistics of China.
Full address: 16th Floor, Investment and Development Building, No. 44 Renmin Road, Laibin 546100, Guangxi, China.

The manuscript uses concise institutional addresses; the full addresses above can be entered in the submission system.

CRediT authorship contribution statement (as in the manuscript):
Xujian Zeng: Conceptualization, Methodology, Validation, Formal analysis, Investigation, Visualization, Supervision, Project administration, Writing - review & editing. Xin Lu: Writing - original draft, Writing - review & editing. Shuaikang Wu: Software, Validation, Data curation.
''')

write('submission_notes_zh.txt',f'''Applied Intelligence 投稿说明（2026-10-07）

稿件：{TITLE}
本地投稿稿，尚未向期刊提交，不能保证录用。

一、上传文件（见同目录“上传清单.txt”）
manuscript.pdf：正文PDF，18页（含参考文献），5张表、2幅图；Springer sn-jnl模板，字体、字号、页边距未改。
manuscript.tex、manuscript.bbl、refs.bib、sn-jnl.cls、sn-basic.bst、Fig1.pdf、Fig2.pdf：LaTeX源文件（source_files.zip为同一组文件的压缩包）。
Fig1.eps、Fig2.eps：矢量图。
ESM_1.zip：Online Resource 1（补充材料S1–S7、21张补充表、全部代码、保存的预测与修复、外部基线与种子复现的原始输出、逐文档结果、CPU复核脚本）。
cover_letter.txt：英文附信。author_information.txt：作者信息（填系统用，不上传）。

二、主要结果（全部为“完整字符串验证器通过 + 注释实体链状态回放通过”）
五折276篇：fine-tuned 14B首遍97、确定性修复142、训练式级联188；3B为71/134/167；few-shot为4/45/77。修复后被字符串验证器接受的程序中15.5–22.2%未通过实体链回放。
受控对比（同一批编译输出、单14B提议器、无备用模型、每篇最多8次生成）：生产步骤反馈P 180篇；N/O/U/C对照149/146/144/142篇。
三个训练种子（42/43/44）：P为180/185/170（均值178.3，SD 7.6），确定性修复均值147.3，未微调提议器均值148.7；每个种子中P相对确定性修复都没有损失。
同基座外部基线（base Qwen3-14B，与P同一底座）：CRITIC式158、CLAIRify式160；P的配对增益/损失33:11、30:10，Holm校正p均为0.0025。
商用API模型（deepseek-flash）外部基线：CLAIRify式189、CRITIC式186、Self-Refine式167；与P差异不显著（Holm p=0.30/0.43/0.16），CLAIRify式输出token为P的11.4倍；用API模型替换P的提议器降至163（Holm p=8.9e-4）。
第二语料ChEMU（225篇测试片段，设计冻结）：确定性修复由71升至150（79:0）；所有修复方法的接受中28–29%未通过回放；学习型与API修复最多再增加8篇，均不显著。
复核：evidence/extensions_audit.json由逐文档结果独立重算上述外部基线、种子和ChEMU数字，与各自summary完全一致；原有主实验、受控对比、稳健性审计均已重跑通过。

三、数据许可
X-WLP为MIT许可，原许可证和README随ESM提供。
ChEMU 2020与ChEMU-Ref受Elsevier有限数据许可约束，仅限研究使用。ESM只包含推导与评测代码、片段ID和汇总结果（results/chemu/summary.json、document_outcomes.csv），不含任何原始或派生的ChEMU文本、标注、程序、规则或含提示词的输出。

四、声明（2026-10-07按作者要求定稿，与ESWA-D-26-40712一致）
1. 经费：无经费（2026-10-05确认）。
2. 利益冲突：The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
3. CRediT：Xujian Zeng负责构思、方法、验证、形式分析、调查、可视化、指导、项目管理、审阅与修改；Xin Lu负责初稿撰写、审阅与修改；Shuaikang Wu负责软件、验证、数据整理。英文原文见正文和author_information.txt。
4. 生成式AI声明：ChatGPT Codex用于辅助代码生成和稿件检查（正文5.5节与声明部分两处）。
5. 代码：Online Resource 1，以及公开仓库https://github.com/XujianZeng/sop2program。
6. 投稿系统中如实确认原创性和未一稿多投；稿件未声称任何人工评分、专家一致性或物理实验执行。

五、解释范围
亮点是验证/修复系统的实体身份敏感性及其评估方法，而不是端到端实验室执行。保留的限制：金标动作锚点和初始库存、模板投影状态、启发式共指解析、ChEMU实体链由规则推导、回顾性折划分与开发集选择、主级联和外部基线只有一次训练。
补充审计中一项仅在补充数据文件出现的描述性指标（replayed_details.json第5条记录的precondition_exact_chain）随Python字符串哈希顺序变化约0.3–0.5个百分点；该指标未在正文或补充表中报告。复核入口tools/run_audits.py已固定PYTHONHASHSEED=1，按顺序运行全部CPU复核，可逐字节复现归档值。

六、格式与入口
摘要210词（要求150–250），6个关键词，含作者声明、数据与代码可得性及补充材料说明。
官方指南：https://link.springer.com/journal/10489/submission-guidelines
投稿入口：https://www.editorialmanager.com/apin/
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
for backend in ('deepseek-flash','qwen14-base'):
    tree(f'results/external_baselines/{backend}',('.json','.csv'))
add('results/seeds/summary.json')
add('results/seeds/document_outcomes.csv')
for seed in (43,44):
    for k in range(5):
        base=f'results/seeds/seed{seed}/fold{k}'
        for name in ('producer_feedback.json','untuned_producer.json','recheck_deterministic_qwen14.json',
                     'qwen14/config.json','qwen14/training.jsonl'):
            add(f'{base}/{name}')
        tree(base+'/eval_qwen14')
        tree(base+'/repair_deterministic_qwen14')
# ChEMU: summary outcomes by snippet id only; raw and derived data are under an Elsevier licence.
add('results/chemu/summary.json')
add('results/chemu/document_outcomes.csv')
for name in ('results/hypothesis_opt/summary.json','results/hypotheses_v4.json','results/h5_qwen3_14b_judge.json'):
    if (ROOT/name).exists():add(name)
for model in ('Qwen3-14B-FP8','Qwen2.5-3B-Instruct'):
    for name in ('config.json','download_manifest.json'):
        add(f'models/{model}/{name}')
for name in ('audit_submission.py','audit_original.py','publication_figures.py','finish_cv_audit.py','audit_strengthening.py','controlled_figure.py','audit_robustness.py','audit_robustness_v2.py','audit_extensions.py','run_audits.py','render_supplement.py'):
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
The main article has 18 pages, including references, with 5 tables and
2 figures. Full numerical detail is retained in
paper/applied_intelligence/supplement/supplementary_details.html (open locally
in a browser; retain its adjacent PNG files). Sections S1-S7 contain extended
methods, original-split evidence, diagnostics, external refinement baselines,
training-seed replication, the ChEMU second corpus, 21 tables and 2 plots.
The adjacent .tex file is an editable source fragment, not a second manuscript.
table_mapping.json maps the earlier full-version table labels to S1-S21.

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
External refinement baselines (results/external_baselines): Self-Refine-,
CRITIC- and CLAIRify-style refinement and producer routing driven by the
deepseek-flash API, and CRITIC/CLAIRify on the base Qwen3-14B backbone; every
raw prompt, reply, token count and document outcome is retained.
Training-seed replication (results/seeds): seeds 43 and 44 retrain the 14B
adapter in every fold; compiled outputs, repairs and outcomes are retained
(adapter weights and checkpoints are excluded). Seed 42 is the saved CV run.
ChEMU second corpus (results/chemu): only summary.json and
document_outcomes.csv (snippet ids and binary outcomes). ChEMU 2020 and
ChEMU-Ref are licensed by Elsevier for research use only; raw text,
annotations, derived programs, rules and prompt-bearing outputs are NOT
distributed. scripts/prepare_chemu.py and scripts/chemu_pipeline.py rebuild
them from the providers' releases.
This archive supports CPU replay of saved outputs, not physical lab execution.
Pretrained model and adapter weights are excluded. Configuration records and
training/inference source are supplied for researchers with the required models.

REPLAY (run from the extracted archive root; Python 3.13 was used)
  python -m pip install -r requirements-cpu.txt
  python -X utf8 paper/applied_intelligence/tools/run_audits.py
run_audits.py sets PYTHONHASHSEED=1 and runs, in order:
  audit_submission.py; audit_original.py;
  audit_strengthening.py --existing / --controlled / --controller-replay;
  audit_robustness_v2.py --omission / --analyze; audit_extensions.py.
These regenerate evidence/audit.json, replayed_details.json, CSV tables,
original_replayed.json, original_first_metrics.json and extensions_audit.json.
No model is loaded. The fixed hash seed reproduces replayed_details.json byte
for byte: one descriptive field that is not reported in the article or
supplement (record 5, fidelity_on_accepted.precondition_exact_chain) depends on
string-hash ordering by 0.3-0.5 percentage points. Run individual audits with
PYTHONHASHSEED=1 set in the environment for the same result.
The code is also available at https://github.com/XujianZeng/sop2program. In the extracted archive
extensions_audit.json marks the ChEMU API generation check as not distributed,
because those raw outputs contain licensed text; every other field matches.
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
CHEMU_OK={'results/chemu/summary.json','results/chemu/document_outcomes.csv'}
assert all(p.suffix=='.py' or p.relative_to(ROOT).as_posix() in CHEMU_OK for p in files
           if 'chemu' in p.relative_to(ROOT).as_posix().lower()), 'ChEMU-derived data must not be archived'
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
esm=P/'ESM_1.zip'
with zipfile.ZipFile(esm,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('README.txt',readme)
    z.writestr('requirements-cpu.txt','pydantic==2.13.4\nscipy==1.18.1\nmatplotlib==3.11.2\n')
    z.writestr('SHA256SUMS.json',json.dumps(hashes,indent=2))
with zipfile.ZipFile(esm) as z:assert z.testzip() is None

source=[P/name for name in ('manuscript.tex','manuscript.bbl','refs.bib','sn-jnl.cls','sn-basic.bst')]
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
