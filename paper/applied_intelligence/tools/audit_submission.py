"""Read-only audit of saved experiments; writes only this manuscript's evidence directory.

Run with the project Python (pydantic and scipy are required). No model is loaded.
The worker replays both original and repaired workflows and recomputes metrics.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'paper/applied_intelligence/evidence'
FOLDS = tuple(range(5))
sys.path.insert(0, str(ROOT))

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))

def worker(fold, compiler):
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows, evaluate_predictions
    from sop2program.verify import verify
    from scripts.recheck_annotation_identity import recheck
    data = f'data/cv/fold{fold}'
    base = f'results/cv/fold{fold}'
    run = f'{base}/eval_{compiler}'
    docs = {x['id']: Workflow.model_validate(x) for x in read(f'{data}/test_workflows.json')}
    rows = [json.loads(x) for x in (ROOT / data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    pred = [json.loads(x) for x in (ROOT / run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(pred), (fold, compiler, 'incomplete predictions')
    rules = read(f'{base}/knowledge/full_rules.json')
    metrics, _ = evaluate_predictions(rows, pred, docs, rules)
    arms = {'deterministic': 'deterministic_repair', 'trained': 'cascade'}
    if compiler == 'fewshot3':
        arms['untuned'] = 'producer_feedback'
    checks = {a: recheck(f'{base}/repair_{a}_{compiler}', arm) for a, arm in arms.items()}
    chain = checks['deterministic']['chain_identity_by_document']
    first = {w.id: bool(w.metadata['schema_complete'] and verify(w, rules)['pass'] and chain[w.id][0])
             for w in assemble_workflows(rows, pred, docs)}
    # Independently recheck full acceptance, source and inventory preservation for saved final programs.
    for a, arm in arms.items():
        final = read(f'{base}/repair_{a}_{compiler}/{arm}_workflows.json')
        summary = read(f'{base}/repair_{a}_{compiler}/summary.json')
        for d in summary[arm]['details']:
            if d['status'] not in ('PASS', 'REPAIRED'):
                continue
            w = Workflow.model_validate(final[d['doc']])
            assert verify(w, rules)['pass'], (fold, compiler, a, d['doc'], 'stale acceptance')
            assert w.source == docs[w.id].source
            assert w.initial == docs[w.id].initial
        cached = read(f'{base}/recheck_{a}_{compiler}.json')[0]
        assert checks[a]['chain_identity_by_document'] == cached['chain_identity_by_document']
    return {'fold': fold, 'compiler': compiler, 'metrics': metrics, 'first': first, 'checks': checks}

def paired(a, b):
    assert set(a) == set(b)
    gain = sum(b[d] and not a[d] for d in a)
    loss = sum(a[d] and not b[d] for d in a)
    n = gain + loss
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gain, loss) + 1)) / 2**n) if n else 1.0
    return {'n': len(a), 'gain': gain, 'loss': loss, 'difference_pp': 100 * (gain-loss)/len(a), 'p': p}

def wilson(k, n):
    z=1.959963984540054
    den=1+z*z/n
    mid=(k/n+z*z/(2*n))/den
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return [100*(mid-half),100*(mid+half)]

def main():
    OUT.mkdir(exist_ok=True)
    jobs=[(k,c) for k in FOLDS for c in ('fewshot3','lora3b','qwen14')]
    def run(job):
        k,c=job
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',SOP_DATA=f'data/cv/fold{k}',SOP_KNOWLEDGE=f'results/cv/fold{k}/knowledge')
        p=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)), '--worker',str(k),c],env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        if p.returncode:
            raise RuntimeError(p.stderr)
        obj=json.loads(p.stdout)
        print('Audited',k,c,flush=True)
        return obj
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(run,jobs))
    manifests=[read(f'data/cv/fold{k}/manifest.json') for k in range(5)]
    expected_documents=sum(manifests[k]['counts']['test']['documents'] for k in FOLDS)
    seen=set()
    for m in manifests:
        parts={s:set(v) for s,v in m['documents'].items()}
        assert not (parts['train']&parts['test'] or parts['train']&parts['dev'] or parts['dev']&parts['test'])
        assert not seen&parts['test']
        seen |= parts['test']
        assert {'xwlp_91','xwlp_212','xwlp_216'} <= parts['train']
    saved=read('results/cv/first_pass_consistent.json')
    pooled={}; records=[]; comparisons=[]; folds=[]
    for c in ('fewshot3','lora3b','qwen14'):
        first={}; final={a:{} for a in ('deterministic','trained','untuned')}; accepted={a:0 for a in final}
        for r in [x for x in results if x['compiler']==c]:
            first.update(r['first'])
            f={'fold':r['fold'],'compiler':c,'documents':len(r['first']),'first':sum(r['first'].values())}
            for a,check in r['checks'].items():
                final[a].update({d:v[1] for d,v in check['chain_identity_by_document'].items()})
                accepted[a]+=check['accepted_text_identity']
                f[a]=check['confirmed_chain_identity']
                f[a+'_accepted']=check['accepted_text_identity']
                f[a+'_argument_f1_before']=check['fidelity_on_accepted']['before_repair']['argument_f1']
                f[a+'_argument_f1_after']=check['fidelity_on_accepted']['after_repair']['argument_f1']
            folds.append(f)
        assert first == saved[c], ('first-pass mismatch',c)
        assert len(first)==expected_documents
        pooled[c]={'documents':expected_documents,'first':sum(first.values()),'first_wilson95':wilson(sum(first.values()),expected_documents)}
        for a,vals in final.items():
            if not vals:continue
            assert set(vals)==set(first)
            count=sum(vals.values())
            pooled[c][a]={'accepted':accepted[a],'confirmed':count,'confirmed_wilson95':wilson(count,expected_documents),
                          'unconfirmed_share':(accepted[a]-count)/accepted[a]}
            base=first if a=='deterministic' else final['deterministic']
            comparisons.append({'compiler':c,'comparison':a+'_vs_'+('first' if a=='deterministic' else 'deterministic'),**paired(base,vals)})
        for d,v in first.items():
            records.append({'compiler':c,'document':d,'first':int(v),**{a:int(vals[d]) for a,vals in final.items() if vals}})
    # Seven pooled comparisons; adjusted values are descriptive, conditional on saved fitted models.
    last=0.0
    for rank,x in enumerate(sorted(comparisons,key=lambda x:x['p'])):
        last=max(last,min(1.0,(len(comparisons)-rank)*x['p']))
        x['holm_p']=last
    paths=[]
    for k,c in jobs:
        base=ROOT/f'results/cv/fold{k}'
        paths.extend([base/f'eval_{c}/predictions.jsonl',base/f'eval_{c}/run_config.json',base/'knowledge/full_rules.json'])
        for a in ('deterministic','trained','untuned'):
            p=base/f'repair_{a}_{c}'
            if p.exists():paths.extend(p.glob('*workflows.json'));paths.append(p/'summary.json')
    paths.extend(ROOT/'sop2program'/p for p in ('annotation.py','metrics.py','ir.py','verify.py','repair.py','induce.py'))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    audit={'completed_folds':list(FOLDS),'planned_folds':5,'documents':expected_documents,
           'operators':sum(manifests[k]['counts']['test']['operators'] for k in FOLDS),
           'pooled':pooled,'comparisons':comparisons,'folds':folds,
           'original_dev_overlap':len(set(read('data/processed/dev_workflows.json')[i]['id'] for i in range(42)) & set(first)),
           'assertions':'fresh state/identity replay, full final verifier acceptance, source/inventory preservation, cached outcome equality, disjoint test IDs, first-pass equality all passed',
           'source_sha256':hashes}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (OUT/'replayed_details.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for filename,rows in [('document_outcomes.csv',records),('fold_results.csv',folds),('paired_comparisons.csv',comparisons)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps({'pooled':pooled,'comparisons':comparisons,'assertions':audit['assertions']},indent=2))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        print(json.dumps(worker(int(sys.argv[2]),sys.argv[3])))
    else:
        main()
