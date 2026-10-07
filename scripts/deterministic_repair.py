"""Model-free repair of a saved evaluation run, in the producer_repair output layout.

Only the bounded local repair search and the settled-unavailable pass are used, so the
result is independent of which compiler produced the run. Output is readable by
recheck_annotation_identity.py with --arm deterministic_repair.
"""
import argparse,json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.ir import Workflow
from sop2program.metrics import assemble_workflows
from sop2program.paths import DATA,KNOWLEDGE
from sop2program.verify import minimal_repair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['test','dev'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--settings-lexicon',default=str(KNOWLEDGE/'setting_lexicon.json'))
    args=p.parse_args()
    rules=json.loads((KNOWLEDGE/'full_rules.json').read_text(encoding='utf-8'))
    lexicon=json.loads((KNOWLEDGE/'trigger_lexicon.json').read_text(encoding='utf-8'))['lexicon']
    payload=json.loads((ROOT/args.settings_lexicon).read_text(encoding='utf-8'))
    mentions={'settings':set(payload['settings']),'products':set(payload['products'])}
    data=DATA
    docs={w['id']:Workflow.model_validate(w) for w in json.loads((data/f'{args.split}_workflows.json').read_text(encoding='utf-8'))}
    rows=[json.loads(l) for l in (data/f'{args.split}.jsonl').read_text(encoding='utf-8').splitlines()]
    preds=[json.loads(l) for l in (ROOT/args.run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    details=[];final={};started=time.perf_counter()
    for w in assemble_workflows(rows,preds,docs):
        if not w.metadata.get('schema_complete'):
            details.append({'doc':w.id,'status':'REVIEW'});continue
        r=minimal_repair(w,rules,lexicon=lexicon,mentions=mentions)
        details.append({'doc':w.id,'status':r['status'],'cost':r['cost'],'operations':r.get('operations',[])})
        final[w.id]=r['workflow'].model_dump()
    summary={'workflows':args.run,'split':args.split,'adapter':None,
             'deterministic_repair':{'accepted':sum(d['status'] in ('PASS','REPAIRED') for d in details),
                                     'documents':len(details),'seconds':time.perf_counter()-started,'details':details}}
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    (out/'deterministic_repair_workflows.json').write_text(json.dumps(final,ensure_ascii=False),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(args.run,'deterministic accepted',summary['deterministic_repair']['accepted'],'/',len(details))


if __name__=='__main__':main()
