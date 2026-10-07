"""Generate a local HTML report from actual saved metrics, with explicit scope limits."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def percent(value):
    return '—' if value is None else f'{value*100:.2f}%'


def number(value,digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def reference_ceiling(rules,split):
    """How far the weak-label reference itself gets: the attainable workflow-level maximum."""
    import sys
    sys.path.insert(0,str(ROOT))
    from sop2program.ir import Workflow
    from sop2program.verify import verify
    docs=[Workflow.model_validate(w) for w in read(ROOT/f'data/processed/{split}_workflows.json')]
    basic=sum(verify(w,invariants=False)['pass'] for w in docs)
    full=sum(verify(w,rules)['pass'] for w in docs)
    return {'documents':len(docs),'basic_pass':basic,'full_pass':full,
            'basic_pass_rate':basic/len(docs) if docs else None,
            'full_pass_rate':full/len(docs) if docs else None,
            'scope':'Reference template projection replayed through the same verifier.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--training-dir',default='results/lora3b_full')
    parser.add_argument('--hypotheses',help='Explicit hypothesis file for this training/evaluation run')
    parser.add_argument('--training-log',help='Supervisor log used to count attempts for this training run')
    parser.add_argument('--runtime-audit',help='Explicit numerical recovery and runtime benchmark audit for this run')
    args=parser.parse_args()
    out=ROOT/'results'
    metrics=[read(ROOT/run/'metrics.json') for run in args.runs]
    configs=[read(ROOT/run/'run_config.json') for run in args.runs]
    training=read(ROOT/args.training_dir/'config.json')
    audit=read(out/'data_audit.json')
    symbolic=read(out/'symbolic/summary.json')
    efficiency=[read(ROOT/run/'efficiency.json') for run in args.runs]
    cache_recovery={config['method']:read(ROOT/run/'cache_recovery.json')
                    for run,config in zip(args.runs,configs) if (ROOT/run/'cache_recovery.json').exists()}
    hypotheses=read(ROOT/args.hypotheses) if args.hypotheses else None
    runtime_audit=read(ROOT/args.runtime_audit) if args.runtime_audit else None
    if runtime_audit:
        expected_adapter=hashlib.sha256((ROOT/args.training_dir/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()
        if runtime_audit['adapter_sha256']!=expected_adapter or any(
                config['data_sha256']!=runtime_audit['data_sha256'] for config in configs):
            raise ValueError('Runtime recovery audit does not match the reported adapter and test data')
    if hypotheses:
        adapter_hash=hashlib.sha256((ROOT/args.training_dir/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()
        if hypotheses.get('adapter_sha256')!=adapter_hash:
            raise ValueError('Hypothesis results do not match the reported training adapter')
        matching=[run for run,config in zip(args.runs,configs) if config['method']=='lora']
        if len(matching)!=1 or (ROOT/hypotheses['run']).resolve()!=(ROOT/matching[0]).resolve():
            raise ValueError('Hypothesis results do not match the reported LoRA evaluation')
        if hypotheses.get('rules_sha256')!=metrics[0]['rules_sha256']:
            raise ValueError('Hypothesis and scored evaluation used different rule libraries')
    training_log=None
    if args.training_log:
        log=(ROOT/args.training_log).read_text(encoding='utf-8')
        attempts=sum(line.startswith('=== ATTEMPT ') for line in log.splitlines())
        training_log={'path':args.training_log,'attempts':attempts,'restarts':max(0,attempts-1)}
    ceiling=reference_ceiling(read(out/'symbolic/full_rules.json'),configs[0]['split'])
    complete=training['status']=='complete' and all(m['evaluation_complete'] for m in metrics)
    if hypotheses:
        complete=complete and hypotheses.get('status')=='complete' and all(key in hypotheses for key in ('H4','H5'))
    runtime=training.get('runtime',{})
    recovery_note=(f'看护日志记录 {training_log["attempts"]} 次启动、{training_log["restarts"]} 次恢复。'
                   if training_log else '恢复位置与环境记录在训练配置中。')
    runtime_note=(f'本轮记录的运行环境为 {escape(runtime.get("platform","未记录"))}，'
                  f'PyTorch {escape(training.get("torch","未记录"))} / CUDA {escape(str(runtime.get("cuda","未记录")))}。'
                  f'{recovery_note}训练完成不表示未发生中断；失败与恢复须对照原始日志。')
    # Rollback/replay of a checkpoint can repeat log entries. Last value per step wins.
    records={}
    for line in (ROOT/args.training_dir/'training.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);records[(row['epoch'],row['example'])]=row
    history=[records[key] for key in sorted(records)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    values=[r['lm_loss'] for r in history]
    smooth=[sum(values[max(0,i-19):i+1])/len(values[max(0,i-19):i+1]) for i in range(len(values))]
    axes[0].plot([r['example'] for r in history],values,color='#9fc0d9',alpha=.45,lw=.7,label='batch loss')
    axes[0].plot([r['example'] for r in history],smooth,color='#176488',lw=2,label='20-batch average')
    axes[0].set(xlabel='Training examples (one epoch)',ylabel='Target-token cross entropy',title='Training trace')
    axes[0].legend(frameon=False)
    names=[config['method'] for config in configs]
    for i,(key,label,color) in enumerate([('action_accuracy','Action accuracy','#176488'),('schema_valid_rate','Schema valid','#ee9d50')]):
        axes[1].bar([x+(i-.5)*.35 for x in range(len(names))],[m[key] for m in metrics],width=.35,label=label,color=color)
    axes[1].set_xticks(range(len(names)),names,rotation=15)
    axes[1].set(ylim=(0,1.05),ylabel='Rate',title='Held-out anchored compilation')
    axes[1].legend(frameon=False)
    for axis in axes:axis.spines[['top','right']].set_visible(False)
    fig.savefig(out/'experiment_summary.png',dpi=170)
    plt.close(fig)
    snapshot={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'complete':complete,
              'training':training,'data_audit':audit,'metrics':metrics,'efficiency':efficiency,
              'runs':args.runs,'run_configs':configs,'training_log':training_log,
              'hypotheses_path':args.hypotheses,'cache_recovery':cache_recovery,
              'runtime_audit':runtime_audit,'runtime_audit_path':args.runtime_audit,
              'reference_ceiling':ceiling,
              'symbolic':{k:(v if k!='repair' else {a:b for a,b in v.items() if a!='details'})
                          for k,v in symbolic.items()},
              'hypotheses':hypotheses}
    (out/'report.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    metric_rows=''.join('<tr><td>'+escape(config['method'])+'</td>'+''.join('<td>'+percent(value)+'</td>' for value in [
        m['schema_valid_rate'],m['action_accuracy'],m['argument_tuple']['f1'],m['weak_precondition_exact_rate'],
        m['weak_effect_exact_rate'],m['weak_transition_exact_rate'],m['basic_workflow_pass_rate'],m['full_workflow_pass_rate']])+'</tr>'
        for config,m in zip(configs,metrics))
    graph_rows=''.join('<tr><td>'+escape(config['method'])+'</td>'+''.join(f'<td>{cell}</td>' for cell in [
        percent(m['process_graph']['node_f1']['f1']),percent(m['process_graph']['edge_f1']['f1']),
        m['process_graph']['graph_edit_distance'],number(m['process_graph']['normalized_graph_edit_distance'],3),
        percent(m['process_graph']['execution_order_accuracy'])])+'</tr>'
        for config,m in zip(configs,metrics))
    repair_rows=''.join('<tr><td>'+escape(config['method'])+'</td>'+''.join(f'<td>{cell}</td>' for cell in [
        m['repair']['eligible_failed_documents'],m['repair']['repaired'],percent(m['repair']['success_rate']),
        number(m['repair']['mean_field_edit_cost']),number(m['repair'].get('mean_candidates_verified'),1),
        escape(', '.join(f'{k}×{v}' for k,v in list(m['repair'].get('operations_used',{}).items())[:4]) or '—')])+'</tr>'
        for config,m in zip(configs,metrics))
    settings=[(name,s) for name,s in symbolic.items() if 'detection_rate' in s]
    symbolic_rows=''.join(f'<tr><td>{escape(name)}</td><td>{s.get("rules","—")}</td><td>{percent(s["detection_rate"])}</td><td>{percent(s["false_positive_rate"])}</td></tr>'
                          for name,s in settings)
    kind_header=''.join(f'<th>{escape(name)}</th>' for name,_ in settings)
    kinds=list(dict.fromkeys(k for _,s in settings for k in s['by_kind']))
    kind_rows=''.join('<tr>'+f'<td>{escape(kind)}</td>'+''.join(
        f'<td>{percent(s["by_kind"][kind]) if kind in s["by_kind"] else "—"}</td>' for _,s in settings)+'</tr>'
        for kind in kinds)
    families=symbolic.get('candidate_families') or {}
    empty=[(name,f) for name,f in families.items() if not f['admitted']]
    family_note=''.join(
        f'{name} 族生成 {f["candidates"]} 个候选但一条未准入，最强候选（{escape(f["best_candidate"])}）'
        f'的后验下界仅 {f["best_posterior_lower_95"]:.2f}，低于准入门槛——该族在本语料上不被支持，而不是未实现。'
        for name,f in empty)
    coverage=symbolic.get('parameter_coverage')
    parameter_note=('' if not coverage else
        f'数量扰动共 {coverage["mutations"]} 例，其中 {coverage["with_applicable_invariant"]} 例落在已准入不变式的（动作，单位）组合上，'
        f'另外 {coverage["without_applicable_invariant"]} 例该组合在训练集中观测不足、根本没有可用不变式——'
        f'漏检主要来自规则覆盖，而不是区间过宽。')
    hypothesis_html=''
    if hypotheses:
        h4=hypotheses.get('H4');h5=hypotheses.get('H5')
        blocks=[]
        if hypotheses.get('status')!='complete':
            progress=hypotheses.get('progress',{})
            blocks.append(f'<p class="note">H4/H5 尚未完成。H4 已完成 {progress.get("h4_completed","—")}/'
                f'{progress.get("h4_total","—")} 份文档，H5 已保存 {progress.get("h5_cases",0)} 个判断。'
                f'{escape(hypotheses.get("failure_reason","计算尚未结束"))}。未生成最终假设检验结论。</p>')
        if h4:
            arms=[('bounded_local_repair','有界局部修复'),('local_repair_with_model_patches','局部修复 + 模型补丁'),
                  ('full_regeneration','已有节点重生成')]
            rows=''.join(f'<tr><td>{escape(label)}</td><td>{h4[key]["repaired"]}/{h4[key]["documents"]}</td>'
                         f'<td>{percent(h4[key]["success_rate"])}</td><td>{number(h4[key]["mean_edit_cost"])}</td>'
                         f'<td>{number(h4[key]["mean_source_deviation"])}</td></tr>' for key,label in arms if key in h4)
            blocks.append('<h3>H4：最小修复与已有节点重生成</h3><table><thead><tr><th>方式</th><th>修复成功</th>'
                          '<th>成功率</th><th>平均字段编辑代价</th><th>平均源偏离</th></tr></thead><tbody>'+rows+'</tbody></table>'
                          '<p>有界局部修复使用源证据、模板与训练集触发词词典；两个模型分支使用同一个微调模型。局部模型接收节点反馈，重生成接收文档反馈。'
                          '重生成仅重写已有动作节点，无法补回编译时缺失的节点；这类输入在两个模型分支先行判为 REVIEW，避免生成必定无效的候选。'
                          '该对照范围有限。源偏离统计不在源文本中逐字出现的算子文本字段。</p>')
            paired=h4.get('paired_success_test');common=h4.get('common_success_comparison',{})
            if paired:
                blocks.append(f'<p>修复成功的文档配对精确 McNemar 检验（双侧）：p={paired["p_value"]:.4g}；'
                              f'仅局部修复成功 {paired["left_only"]} 份，仅重生成成功 {paired["right_only"]} 份。'
                              f'已有输入缺失节点的文档 {h4.get("incomplete_input_documents",0)} 份。'
                              f'编辑代价与源偏离的表格均值各自在成功文档上计算；共同成功 {common.get("documents",0)} 份，'
                              f'局部减重生成的平均编辑代价差 {number(common.get("mean_local_minus_regeneration_cost"))}，'
                              f'源偏离差 {number(common.get("mean_local_minus_regeneration_deviation"))}。不能把不同成功子集的均值当作配对证据。</p>')
        if h5:
            rows=''.join(f'<tr><td>{escape(label)}</td><td>{percent(h5[key]["detection_rate"])}</td>'
                         f'<td>{percent(h5[key]["localization_rate"])}</td><td>{percent(h5[key]["false_positive_rate"])}</td>'
                         f'<td>{number(h5[key]["seconds_total"],1)}</td></tr>'
                         for key,label in [('verifier','确定性验证器'),('llm_judge','纯文本模型裁判')])
            blocks.append(f'<h3>H5：验证器与模型裁判</h3><table><thead><tr><th>判定方</th><th>注入错误检出</th>'
                          f'<th>定位到注入步骤</th><th>干净流程误判</th><th>总耗时/秒</th></tr></thead><tbody>{rows}</tbody></table>'
                          f'<p>共 {h5["injected_cases"]} 个单因素近似错误和 {h5["clean_controls"]} 份干净对照。'
                          f'模型裁判输出可解析率 {percent(h5["llm_judge"]["parse_rate"])}；无法解析一律计为未检出，不计为通过。'
                          f'输入被字符或 token 预算截断 {h5["llm_judge"].get("truncated_inputs",0)} 例。模型裁判接收源文本、库存、算子与依赖；'
                          '没有接收证据跨度或归纳数量规则，且使用编译器微调模型的零样本审计提示，结论只适用于该配置。</p>')
            paired=h5.get('paired_detection_test');localized=h5.get('paired_localization_test')
            if paired:
                blocks.append(f'<p>按 {paired["documents"]} 份文档整体翻转配对差的精确双侧检验：检出率差 p={paired["p_value"]:.4g}，'
                              f'定位率差 p={localized["p_value"]:.4g}。同一文档的多个注入错误作为一组，不视作独立样本。'
                              'p 值未做多重比较校正，定位率为次要指标；这些结果属于单一种子的弱标签实验。</p>')
        hypothesis_html=('<section><h2>论文假设的直接对照</h2>'+''.join(blocks)+
                         '<p class="note">这两项只覆盖 H4 和 H5，且都在弱标签投影上测量；H1 至 H3 需要跨语料和人工裁定子集，本轮未做。</p></section>')
    efficiency_rows=''.join(f'<tr><td>{escape(config["method"])}</td><td>{m["operators"]}</td><td>{eff["total_generation_s"]:.1f}</td><td>{eff["peak_vram_gb"]:.2f}</td><td>{m["repair"]["repaired"]}</td><td>{m["repair"]["review"]}</td></tr>'
                            for config,m,eff in zip(configs,metrics,efficiency))
    links=''.join(f'<li><b>{escape(config["method"])}</b>：<a href="{escape(str(Path(run).relative_to("results")))}'+
                  '/metrics.json">指标</a> · <a href="'+escape(str(Path(run).relative_to('results')))+
                  '/predictions.jsonl">逐步预测</a> · <a href="'+escape(str(Path(run).relative_to('results')))+
                  '/audit.json">验证和修复记录</a></li>' for config,run in zip(configs,args.runs))
    evaluation_recovery_note=''.join(
        f'{escape(method)} 曾有两份看护并发写入；{record["duplicate_records"]} 条重复记录已核对并处理，'
        f'冲突预测 {len(record["conflicting_ids"])} 条，恢复时保留 {record["kept_records"]} 条有效记录。'
        '原文件已备份，恢复审计保存在对应评测目录的 cache_recovery.json。随后加入操作系统互斥锁防止重复看护。'
        '并发期间的延迟记录仍保留，因此该时长不适合当作独占 GPU 的吞吐基准。'
        for method,record in cache_recovery.items())
    runtime_audit_note=''
    if runtime_audit:
        b=runtime_audit['wsl_benchmark']
        runtime_audit_note=(f"<p class='note'>旧 Windows no_state 评测出现异常数值，432 条输出中只有 64 条可解析，"
                           "其中 360 条为重复感叹号；该缓存已隔离并重新生成。已完成的 base/lora 预测保留，"
                           "剩余评测和 H4/H5 改用 WSL 独立 CUDA 12.8 环境，增加权重及每步生成分数的有限值检查。"
                           f"同一组 8 个样本的 WSL 普通批量解码为 {b['ordinary_batched_seconds']:.2f} 秒，"
                           f"编译后为 {b['compiled_batched_seconds']:.2f} 秒；四轮输出逐 token 相同，也与旧缓存中对应有效输出相同。"
                           "这是小样本控制测试，不是整轮独占 GPU 吞吐。平台和解码执行方式发生了变化，"
                           "因此本轮表格不属于同一运行环境下的严格数值复现。逐项审计见 "
                           f"<a href='{escape(str(Path(args.runtime_audit).relative_to('results')))}'>运行恢复记录</a>。</p>")
        if runtime_audit.get('h4_long_prompt_benchmark'):
            long=runtime_audit['h4_long_prompt_benchmark'];old,new=long['trials']
            runtime_audit_note+=(f"<p class='note'>H4/H5 最终使用 SDPA（批量上限 4，普通 KV 缓存）。"
                f"同一组 4 个完整长提示的 eager 单条执行为 {old['seconds_including_first_compile']:.2f} 秒，"
                f"SDPA 批量执行为 {new['seconds_including_first_compile']:.2f} 秒，逐 token 输出相同。"
                "单次小样本计时包含首个编译调用，不能代表整轮稳定加速比。WSL 运行中也曾出现 NaN，"
                "有限值检查阻止异常批次写入，重启后恢复；GPU 不稳定的根因尚未确定。</p>")
    if runtime_audit and runtime_audit.get('gpu_corruption_diagnosis',{}).get('post_boot'):
        diagnosis=runtime_audit['gpu_corruption_diagnosis']
        runtime_audit_note+=(f"<p class='note'>GPU 完整性复测：重启并变更至 {escape(diagnosis['driver'])} 驱动后，"
            "独立 CUDA 内核仍出现按字节校验的数据损坏。最终责任组件尚未确定；WSL 共用宿主 Windows 驱动。"
            "有限值检查不能排除仍为有限值的数据损坏，本轮已保存指标需在稳定环境独立复核。"
            "<a href='gpu_corruption_diagnosis.json'>GPU 诊断记录</a>。</p>")
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SOP2Program 本地实验报告</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#183143;background:#eef3f7;margin:0;line-height:1.7}}
main{{max-width:1240px;margin:auto;padding:40px 28px}}section{{background:white;padding:24px 28px;margin:20px 0;border-radius:12px}}
h1{{font-size:32px;margin-bottom:8px}}h2{{font-size:21px;margin-top:0}}.tag{{color:#176488;font-weight:700}}p{{max-width:1050px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #dce5ec}}th:first-child,td:first-child{{text-align:left}}
th{{background:#eef5f8}}.scroll{{overflow-x:auto}}img{{width:100%;height:auto}}a{{color:#176488}}code{{background:#eef3f7;padding:2px 5px}}
.note{{border-left:4px solid #ee9d50;padding-left:16px}}li{{margin:6px 0}}
</style><main><div class="tag">X-WLP · Qwen2.5-3B-Instruct · LoRA · 单张 GPU</div>
<h1>SOP2Program 本地实验报告</h1><p>运行状态：{'本轮训练和所列评测已完成' if complete else '部分完成，请检查逐项状态'}。本报告读取实际训练日志、预测和验证记录生成。</p>
<section><h2>实验范围</h2><p>训练集 195 份 SOP / 2,716 个操作，验证集 42 份 / 615 个操作，测试集 42 份 / 584 个操作。采用自定义按文档划分，随机种子 42；本轮评测实际覆盖 {metrics[0]['documents']} 份文档 / {metrics[0]['operators']} 个操作。</p>
<p class="note">这是给定动作位置和初始材料清单的编译实验。状态、效果和依赖的参考答案来自保守模板，不是人工裁定的独立金标。结果不能直接解释为全文 SOP 自动理解精度，也不能证明论文全部假设。</p>
<p>硬件：{escape(training['gpu'])}。训练 {training['epochs']} 轮，LoRA rank={training['lora_rank']}，microbatch={training['batch_size']}、梯度累积={training['gradient_accumulation']}，BF16，学习率 {training['learning_rate']}。优化器更新 {number(training.get('updates'),0)} 次，记录训练时间 {number(training.get('elapsed_s'),1)} 秒，有记录的显存峰值 {number(training.get('peak_vram_gb'),2)} GiB。{runtime_note}这一轮用于受控的本地方法实验，尚不是完整论文复现。</p></section>
<section><h2>训练与评测概览</h2><img src="experiment_summary.png" alt="训练损失和测试指标"></section>
<section><h2>编译指标</h2><div class="scroll"><table><thead><tr><th>方法</th><th>Schema 合法</th><th>动作准确率</th><th>论元元组 F1</th><th>弱前置条件精确率</th><th>弱效果精确率</th><th>弱状态迁移准确率</th><th>基础流程 PASS</th><th>含规则 PASS</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<p>base 为原始模型；lora 为微调模型；no_state 仅在推理时移除状态反馈；unconstrained 移除 JSON Schema 解码限制。论元指标匹配角色/文本/类型元组，不是独立跨度 F1。缺失或非法输出计入失败；状态迁移只在参考流程可执行的步骤上计算。</p>
<p class="note">流程级 PASS 的可达上限由弱标签参考本身决定：同一验证器回放参考投影时，{ceiling['documents']} 份文档中有 {ceiling['basic_pass']} 份（{percent(ceiling['basic_pass_rate'])}）通过基础状态与证据检查，{ceiling['full_pass']} 份（{percent(ceiling['full_pass_rate'])}）同时通过归纳规则库。模型指标应结合这一参考可达上限解读；参考流程可执行不表示已有独立人工裁定。</p></section>
<section><h2>流程图指标</h2><div class="scroll"><table><thead><tr><th>方法</th><th>节点 F1</th><th>边 F1</th><th>图编辑距离</th><th>归一化图编辑距离</th><th>执行顺序准确率</th></tr></thead><tbody>{graph_rows}</tbody></table></div>
<p>动作锚点给定，节点按步骤下标对齐，因此图编辑距离是精确值，不需要近似图匹配：它统计算子标签替换、缺失或多余节点，以及依赖边的对称差。执行顺序准确率只在参考回放中已提交且存在先后关系的依赖对上计算。</p></section>
<section><h2>局部修复</h2><div class="scroll"><table><thead><tr><th>方法</th><th>未通过文档</th><th>修复成功</th><th>成功率</th><th>平均字段编辑代价</th><th>平均验证候选数</th><th>用到的修复操作</th></tr></thead><tbody>{repair_rows}</tbody></table></div>
<p>候选按式 (10) 排序：字段编辑代价、剩余违规、无源支持字段、以及相对源文本的额外偏离。修复操作限于论文 3.3 节允许的六种，全部以源文本为准；动作改标只允许训练集为该触发词见证过的标签，且不得凭空产生原动作不会产生的对象。搜索有预算上限，报告的是候选集合内的最小值，不是全局最优。</p></section>
<section><h2>符号验证与规则归纳</h2><p>从训练文档归纳 {symbolic['full']['rules']} 条规则，覆盖六类候选模板。反事实测试包含 {symbolic['corpus']['mutation_cases']} 个注入错误，来源为 {symbolic['corpus']['weak_test_clean']} 份通过基础验证的弱标签流程。</p>
<table><thead><tr><th>验证设置</th><th>规则数</th><th>注入错误检出率</th><th>干净弱标签流程误拒率</th></tr></thead><tbody>{symbolic_rows}</tbody></table>
<table><thead><tr><th>注入错误类型</th>{kind_header}</tr></thead><tbody>{kind_rows}</tbody></table>
<p class="note">规则候选在训练集上按 Beta 后验下界、覆盖率和分组稳定性准入，再用开发集确认：任何拒绝干净开发文档的规则都记为实测假阳性并剔除，测试集不参与准入。<code>no_holdout</code> 是移除这道确认的消融。规则族的适用范围与其断言一致——效应规则不追究本身无法产生该效应的算子，依赖规则要求前驱作用在同一实体上，而不是文档中恰好更早出现过。数量约束取对数尺度上的容差区间（内容 99.9%、置信 95%），而非训练集经验包络：n 次观测的最小—最大区间会以约 2/(n+1) 的比例拒绝合法取值。{parameter_note}{family_note}</p></section>
{hypothesis_html}
<section><h2>效率与局部修复</h2><table><thead><tr><th>方法</th><th>操作数</th><th>生成总时长/秒</th><th>峰值显存/GiB</th><th>修复成功文档</th><th>需复核文档</th></tr></thead><tbody>{efficiency_rows}</tbody></table>
<p>推理批次包含不同文档，各文档内部顺序执行。生成总时长累加批次耗时，不含模型载入；逐条记录另含批次延迟和分摊延迟。{evaluation_recovery_note}修复仅搜索源证据恢复和模板效果补丁，未使用 LLM 提议补丁；最小代价仅在已枚举候选中成立。</p></section>
{runtime_audit_note}
<section><h2>论文步骤与实现边界</h2><ul>
<li>算法 A：已实现有状态、受约束的逐步编译与 LoRA 训练；辅助头是动作/类型/角色/效果摘要及触发词指针，尚不是完整实体对关系解码器。</li>
<li>算法 B：已实现文档级 Beta 后验、覆盖率、分组稳定性、本体泛化、去冗余与开发集确认；数量约束为对数尺度容差区间，依赖约束要求作用于同一实体。当前归纳输入为训练集弱标签投影，尚未扩展到模型编译的大规模新语料。</li>
<li>算法 C：已实现确定性验证、失败步骤不提交状态、有限候选修复和 REVIEW；另提供可选模型局部节点补丁，经外部验证器裁定，尚未纳入本报告的主评测，也不是无限制补丁搜索。</li>
<li>仍待完成：自动动作/库存识别、实体别名与共指、独立人工状态金标、完整参数槽与单位类型、WLP-MSTG 等额外语料、跨来源测试、多个种子及 7B/14B 对比。</li>
</ul></section><section><h2>可追溯文件</h2><ul>{links}<li><a href="data_audit.json">数据与划分审计</a> · <a href="symbolic/summary.json">符号实验汇总</a> · <a href="report.json">报告数据</a></li></ul></section></main></html>'''
    (out/'reproduction_report.html').write_text(html,encoding='utf-8')
    print(str(out/'reproduction_report.html'))


if __name__=='__main__':main()
