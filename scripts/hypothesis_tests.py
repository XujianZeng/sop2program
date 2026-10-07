"""H4 and H5 from Section 5.4, run against the trained compiler.

H4: bounded local repair versus full regeneration by the same model on the same feedback.
H5: the deterministic verifier versus a text-only LLM judge on single-factor near-misses.
"""
import argparse,hashlib,json,math,os,sys,time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from transformers import AutoTokenizer,set_seed
from sop2program.compiler import LocalCompiler
from sop2program.ir import Workflow
from sop2program.verify import verify,minimal_repair,edit_cost,source_deviation
from sop2program.repair import (regenerate_with_compiler,repair_with_compiler,
                                JUDGE,judge_rows,parse_judgement)
from scripts.symbolic_experiments import mutations
from scripts.evaluate_supervisor import acquire_lock
from sop2program.numerics import load_checked_model,FiniteLogitsProcessor


def atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


class CaseCache:
    """Persist each finished arm/batch and reject results from another experiment."""
    def __init__(self,path,signature):
        self.path=path;path.mkdir(parents=True,exist_ok=True)
        config=path/'config.json'
        if config.exists() and json.loads(config.read_text(encoding='utf-8'))!=signature:
            raise ValueError('Hypothesis checkpoint configuration differs; use a new --checkpoint-dir')
        atomic_json(config,signature)

    def get(self,key):
        path=self.path/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        if not path.exists():return None
        record=json.loads(path.read_text(encoding='utf-8'))
        if record['key']!=key:raise ValueError('Hypothesis checkpoint key mismatch')
        return record['value']

    def put(self,key,value):
        atomic_json(self.path/(hashlib.sha256(key.encode()).hexdigest()+'.json'),
                    {'key':key,'value':value})

    def compute(self,key,fn):
        value=self.get(key)
        if value is None:
            value=fn();self.put(key,value)
        return value


def exact_paired_binary(left,right):
    """Two-sided exact McNemar test on paired binary outcomes."""
    wins=sum(bool(a) and not bool(b) for a,b in zip(left,right))
    losses=sum(bool(b) and not bool(a) for a,b in zip(left,right))
    n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return {'test':'exact McNemar, two-sided','pairs':len(left),'left_only':wins,
            'right_only':losses,'p_value':p}


def document_permutation(rows,left,right):
    """Flip all paired differences in a document together; enumerate sums exactly."""
    differences=Counter()
    for row in rows:differences[row['doc']]+=int(row[left])-int(row[right])
    distribution=Counter({0:1})
    for difference in differences.values():
        next_distribution=Counter()
        for total,count in distribution.items():
            next_distribution[total+difference]+=count
            next_distribution[total-difference]+=count
        distribution=next_distribution
    observed=sum(differences.values())
    extreme=sum(count for total,count in distribution.items() if abs(total)>=abs(observed))
    return {'test':'exact paired document-block sign-flip, two-sided',
            'documents':len(differences),'cases':len(rows),'left_minus_right':observed,
            'p_value':extreme/2**len(differences) if differences else 1.,
            'document_differences':dict(differences)}


def load_model(adapter,base=False,compile_cache=False,attention='eager'):
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-3B-Instruct',local_files_only=True)
    model=load_checked_model(ROOT/'models/Qwen2.5-3B-Instruct',None if base else ROOT/adapter,compile_cache=compile_cache,attention=attention)
    return model,tokenizer


def workflow_complete(workflow):
    return bool(workflow.metadata.get('schema_complete',True)) and len(workflow.steps)==workflow.metadata.get('expected_steps',len(workflow.steps))


def local_repair_result(workflow,rules,lexicon):
    result=minimal_repair(workflow,rules,lexicon=lexicon)
    complete=workflow_complete(workflow)
    return {'status':result['status'] if complete else 'REVIEW',
            'cost':result['cost'] if complete else None,'tested':result['tested'],
            'deviation':source_deviation(result['workflow'])}


def judge_batch(model,tokenizer,payloads,max_new_tokens=48,batch_size=4):
    """Text-only verdicts, one generation per candidate workflow."""
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    out=[]
    for start in range(0,len(payloads),batch_size):
        chunk=payloads[start:start+batch_size]
        serialized=[json.dumps(p,ensure_ascii=False,separators=(',',':')) for p in chunk]
        texts=[tokenizer.apply_chat_template(
            [{'role':'system','content':JUDGE},
             {'role':'user','content':payload[:12000]}],
            tokenize=False,add_generation_prompt=True) for payload in serialized]
        lengths=[len(ids) for ids in tokenizer(texts,padding=False,truncation=False)['input_ids']]
        inputs=tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=3584).to(model.device)
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id,logits_processor=[FiniteLogitsProcessor()])
        for i in range(len(chunk)):
            out.append({'raw':tokenizer.decode(generated[i,inputs.input_ids.shape[1]:],skip_special_tokens=True),
                        'payload_characters':len(serialized[i]),'prompt_tokens_before_truncation':lengths[i],
                        'input_truncated':len(serialized[i])>12000 or lengths[i]>3584})
    return out


def run_h4(model,tokenizer,rules,lexicon,run,limit,cache=None):
    compiler=LocalCompiler(model,tokenizer)
    workflows=[Workflow.model_validate(w) for w in json.loads((ROOT/run/'workflows.json').read_text(encoding='utf-8'))]
    failing=[w for w in workflows if not workflow_complete(w) or not verify(w,rules)['pass']][:limit or None]
    rows=[]
    for w in failing:
        def local_arm():
            return local_repair_result(w,rules,lexicon)
        def assisted_arm():
            if not workflow_complete(w):
                return {'status':'REVIEW','cost':None,'tested':0,'proposals':0,'deviation':None,
                        'skip_reason':'Missing anchored nodes cannot be restored by this repair arm'}
            result=repair_with_compiler(w,compiler,rules,lexicon=lexicon)
            return {'status':result['status'] if workflow_complete(w) else 'REVIEW',
                    'cost':result['cost'] if workflow_complete(w) else None,'tested':result['tested'],
                    'proposals':len(result['model_proposals']),'deviation':source_deviation(result['workflow'])}
        def regeneration_arm():
            if not workflow_complete(w):
                return {'status':'REVIEW','cost':None,'rewritten':0,'deviation':None,
                        'skip_reason':'Existing-node regeneration cannot restore missing anchored nodes'}
            result=regenerate_with_compiler(w,compiler,rules)
            return {'status':result['status'] if workflow_complete(w) else 'REVIEW',
                    'cost':result['cost'] if workflow_complete(w) else None,'rewritten':result['rewritten'],
                    'deviation':result.get('source_deviation')}
        row={'doc':w.id,'steps':len(w.steps),'expected_steps':w.metadata.get('expected_steps',len(w.steps))}
        for arm,fn in [('local',local_arm),('local_llm',assisted_arm),('regeneration',regeneration_arm)]:
            row[arm]=cache.compute(f'H4/{w.id}/{arm}',fn) if cache else fn()
            print('H4_ARM',w.id,arm,row[arm]['status'],flush=True)
        rows.append(row)
        print('H4',w.id,rows[-1]['local']['status'],rows[-1]['regeneration']['status'],flush=True)
    def summarize(key):
        ok=[r[key] for r in rows if r[key]['status']=='REPAIRED']
        costs=[r['cost'] for r in ok if r['cost'] is not None]
        deviations=[r['deviation'] for r in ok if r.get('deviation') is not None]
        return {'documents':len(rows),'repaired':len(ok),
                'success_rate':len(ok)/len(rows) if rows else None,
                'mean_edit_cost':sum(costs)/len(costs) if costs else None,
                'mean_source_deviation':sum(deviations)/len(deviations) if deviations else None}
    paired=exact_paired_binary([r['local']['status']=='REPAIRED' for r in rows],
                              [r['regeneration']['status']=='REPAIRED' for r in rows])
    common=[r for r in rows if r['local']['status']==r['regeneration']['status']=='REPAIRED']
    differences={field:[r['local'][field]-r['regeneration'][field] for r in common]
                 for field in ('cost','deviation')}
    return {'scope':'Failing documents of the scored run. Local arm uses source/templates and train lexicon; '
                    'both model arms use the same trained compiler. Local feedback is node-specific; '
                    'regeneration feedback is document-wide. Regeneration rewrites existing anchored nodes '
                    'using original replay-state snapshots and cannot recover a missing node.',
            'bounded_local_repair':summarize('local'),
            'local_repair_with_model_patches':summarize('local_llm'),
            'full_regeneration':summarize('regeneration'),'documents':rows,
            'paired_success_test':paired,
            'common_success_comparison':{'documents':len(common),
                **{f'mean_local_minus_regeneration_{field}':sum(values)/len(values) if values else None
                   for field,values in differences.items()}},
            'incomplete_input_documents':sum(r['steps']!=r['expected_steps'] for r in rows),
            'structural_rejections':sum(not workflow_complete(w) for w in failing),
            'cost_summary_scope':'Mean costs/deviations use each arm successful documents; paired comparisons use shared successes.'}


def run_h5(model,tokenizer,rules,limit,cache=None):
    test=[Workflow.model_validate(w) for w in json.loads(
        (ROOT/'data/processed/test_workflows.json').read_text(encoding='utf-8'))]
    clean=[w for w in test if verify(w)['pass']]
    cases=[(w.id,kind,step,mutated) for w in clean for kind,step,mutated in mutations(w)][:limit or None]
    controls=[(w.id,'clean',None,w) for w in clean][:limit or None]
    everything=cases+controls
    verdicts=[];judge_s=0.
    for start in range(0,len(everything),4):
        chunk=everything[start:start+4]
        key='H5/'+json.dumps([(doc,kind,step) for doc,kind,step,_ in chunk])
        def generate():
            begin=time.perf_counter()
            results=judge_batch(model,tokenizer,[judge_rows(w) for _,_,_,w in chunk])
            return {'verdicts':results,'seconds':time.perf_counter()-begin}
        result=cache.compute(key,generate) if cache else generate()
        verdicts.extend(result['verdicts']);judge_s+=result['seconds']
        print('H5',len(verdicts),'/',len(everything),flush=True)
    rows=[];verifier_s=0.
    for (doc,kind,step,w),verdict in zip(everything,verdicts):
        raw=verdict['raw']
        begin=time.perf_counter();outcome=verify(w,rules);verifier_s+=time.perf_counter()-begin
        parsed=parse_judgement(raw)
        rows.append({'doc':doc,'kind':kind,'injected_step':step,
                     'verifier_flagged':not outcome['pass'],
                     'verifier_localized':any(v['step']==step for v in outcome['violations']),
                     'judge_parsed':parsed is not None,
                     'judge_flagged':bool(parsed and parsed['error']),
                     'judge_localized':bool(parsed and parsed['error'] and parsed['step']==step),
                     'judge_raw':raw,'input_truncated':verdict['input_truncated'],
                     'payload_characters':verdict['payload_characters'],
                     'prompt_tokens_before_truncation':verdict['prompt_tokens_before_truncation']})
    injected=[r for r in rows if r['kind']!='clean'];controls=[r for r in rows if r['kind']=='clean']
    def rate(items,key):return sum(r[key] for r in items)/len(items) if items else None
    kinds=sorted({r['kind'] for r in injected})
    return {'scope':'Single-factor near-misses over verifier-clean reference projections. Text judge uses '
                    'the trained compiler with a zero-shot audit prompt, source, initial inventory, operators '
                    'and dependency ids; evidence spans and learned numeric rules are not supplied. '
                    'Character/token truncation follows the original judge budget and is explicitly counted.',
            'injected_cases':len(injected),'clean_controls':len(controls),
            'verifier':{'detection_rate':rate(injected,'verifier_flagged'),
                        'localization_rate':rate(injected,'verifier_localized'),
                        'false_positive_rate':rate(controls,'verifier_flagged'),
                        'seconds_total':verifier_s},
            'llm_judge':{'detection_rate':rate(injected,'judge_flagged'),
                         'localization_rate':rate(injected,'judge_localized'),
                         'false_positive_rate':rate(controls,'judge_flagged'),
                         'parse_rate':rate(rows,'judge_parsed'),'seconds_total':judge_s,
                         'truncated_inputs':sum(r['input_truncated'] for r in rows)},
            'paired_detection_test':document_permutation(injected,'verifier_flagged','judge_flagged'),
            'paired_localization_test':document_permutation(injected,'verifier_localized','judge_localized'),
            'by_kind':{kind:{'cases':len(group),'verifier_detection':rate(group,'verifier_flagged'),
                            'judge_detection':rate(group,'judge_flagged'),
                            'verifier_localization':rate(group,'verifier_localized'),
                            'judge_localization':rate(group,'judge_localized')}
                       for kind in kinds for group in [[r for r in injected if r['kind']==kind]]},
            'cases':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',default='results/lora3b_v3/adapter')
    p.add_argument('--run',default='results/evaluation_v3/lora')
    p.add_argument('--rules',default='results/symbolic/full_rules.json')
    p.add_argument('--lexicon',default='results/symbolic/trigger_lexicon.json')
    p.add_argument('--output',default='results/hypotheses_v3.json')
    p.add_argument('--checkpoint-dir')
    p.add_argument('--prepare-only',action='store_true',help='Cache deterministic H4 local repairs on CPU, without loading a GPU model')
    p.add_argument('--compile-cache',action='store_true')
    p.add_argument('--attention',choices=['eager','sdpa'],default='eager')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--skip',nargs='*',default=[],choices=['h4','h5'])
    args=p.parse_args()
    args.adapter=Path(args.adapter).as_posix();args.run=Path(args.run).as_posix()
    set_seed(42);torch.set_num_threads(6)
    rules=json.loads((ROOT/args.rules).read_text(encoding='utf-8'))
    lexicon=json.loads((ROOT/args.lexicon).read_text(encoding='utf-8'))['lexicon']
    def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    report={'adapter':args.adapter,'run':args.run,
            'rules_sha256':digest(args.rules),'lexicon_sha256':digest(args.lexicon),
            'adapter_sha256':digest(Path(args.adapter)/'adapter_model.safetensors'),
            'workflow_sha256':digest(Path(args.run)/'workflows.json'),
            'test_workflows_sha256':digest('data/processed/test_workflows.json'),
            'model_manifest_sha256':digest('models/Qwen2.5-3B-Instruct/download_manifest.json'),
            'repair_code_sha256':digest('sop2program/repair.py'),
            'verifier_code_sha256':digest('sop2program/verify.py'),
            'compiler_code_sha256':digest('sop2program/compiler.py'),
            'constrained_code_sha256':digest('sop2program/constrained.py'),
            'ir_code_sha256':digest('sop2program/ir.py'),
            'mutation_code_sha256':digest('scripts/symbolic_experiments.py'),
            'hypothesis_code_sha256':digest('scripts/hypothesis_tests.py'),
            'supervisor_code_sha256':digest('scripts/evaluate_supervisor.py'),
            'numerics_code_sha256':digest('sop2program/numerics.py'),
            'software_versions':{name:version(name) for name in ['torch','transformers','peft','lm-format-enforcer']},
            'seed':42,'limit':args.limit,'judge_batch_size':4,'judge_max_new_tokens':48,
            'compile_cache':args.compile_cache,'attention_implementation':args.attention,
            'sdpa_compiler_max_batch':4,'sdpa_compiler_padded_token_budget':16_384,
            'matmul_reduced_precision':False,
            'judge_max_input_tokens':3584,'judge_max_payload_characters':12000}
    cache=CaseCache(ROOT/(args.checkpoint_dir or args.output+'.checkpoints'),report)
    if 'h4' not in args.skip:
        workflows=[Workflow.model_validate(w) for w in json.loads((ROOT/args.run/'workflows.json').read_text(encoding='utf-8'))]
        failing=[w for w in workflows if not workflow_complete(w) or not verify(w,rules)['pass']][:args.limit or None]
        for workflow in failing:
            cache.compute(f'H4/{workflow.id}/local',lambda:local_repair_result(workflow,rules,lexicon))
        print('H4_LOCAL_PREPARED',len(failing),flush=True)
    if args.prepare_only:return
    # Use the evaluation's OS lock so another launcher cannot overlap GPU stages.
    lock=acquire_lock((ROOT/args.run).parent)
    model,tokenizer=load_model(args.adapter,compile_cache=args.compile_cache,attention=args.attention)
    report['status']='running'
    if 'h4' not in args.skip:
        report['H4']=run_h4(model,tokenizer,rules,lexicon,args.run,args.limit,cache)
        atomic_json(ROOT/args.output,report)
    if 'h5' not in args.skip:
        if args.compile_cache:
            import gc
            torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
        report['H5']=run_h5(model,tokenizer,rules,args.limit,cache)
        atomic_json(ROOT/args.output,report)
    report['status']='complete';report['completed_hypotheses']=[key for key in ('H4','H5') if key in report]
    report['statistical_scope']='Two-sided paired tests; H5 permutes documents as blocks. '
    report['statistical_scope']+='P-values are unadjusted; localization is secondary. This is a single-seed weak-label study.'
    atomic_json(ROOT/args.output,report)
    lock.close()
    print('HYPOTHESES_COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
