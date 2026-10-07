"""Recheck saved original-split outputs without model inference."""
import json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'paper/applied_intelligence/evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
if len(sys.argv)>1:
    from scripts.recheck_annotation_identity import recheck
    print(json.dumps(recheck(sys.argv[1],sys.argv[2])))
else:
    jobs={}
    for p in (ROOT/'results/annotation_states').glob('*.json'):
        d=read(p)
        if isinstance(d,list):
            for r in d:
                if 'repair' in r:jobs[(r['repair'],r['arm'])]=r
    def job(pair):
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        env.pop('SOP_DATA',None);env.pop('SOP_KNOWLEDGE',None)
        proc=subprocess.run([sys.executable,'-X','utf8',__file__,*pair],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        if proc.returncode:raise RuntimeError(proc.stderr)
        r=json.loads(proc.stdout)
        assert r['chain_identity_by_document']==jobs[pair]['chain_identity_by_document'],pair
        print('Verified',pair,flush=True)
        return r
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(job,jobs))
    (OUT/'original_replayed.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows,evaluate_predictions
    from sop2program.verify import verify
    data=ROOT/'data/processed'
    docs={x['id']:Workflow.model_validate(x) for x in read(data/'test_workflows.json')}
    rows=[json.loads(x) for x in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    rules=read(ROOT/'results/symbolic/full_rules.json')
    consistent=read(ROOT/'results/annotation_states/first_pass_consistent_test.json')
    metrics={}
    for c,repair in [('zeroshot','deterministic_test_qwen14_zeroshot'),('fewshot3','deterministic_test_qwen14_fewshot3'),('lora3b','deterministic_test_3b_lora'),('qwen14','deterministic_test_qwen14_lora')]:
        path='results/'+repair
        run=read(ROOT/path/'summary.json')['workflows']
        pred=[json.loads(x) for x in (ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        r=next(x for x in results if x['repair']==path)
        outcomes={w.id:bool(w.metadata['schema_complete'] and verify(w,rules)['pass'] and r['chain_identity_by_document'][w.id][0]) for w in assemble_workflows(rows,pred,docs)}
        assert outcomes==consistent[c],c
        metrics[c]=evaluate_predictions(rows,pred,docs,rules)[0]
    (OUT/'original_first_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('All',len(results),'original-split rechecks agree with saved outcomes.')
