"""Oracle identity inside verify must agree with renaming to gold chains and replaying."""
import json,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.annotation import gold_identity
from sop2program.ir import Workflow
from sop2program.metrics import assemble_workflows
from sop2program.verify import verify

def oracle(w):
    return w.model_copy(update={'initial':gold_identity(w.id).gold().initial,'metadata':dict(w.metadata,identity='oracle')})

gold_fail=[]
for split in ('train','dev','test'):
    for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text(encoding='utf-8')):
        w=Workflow.model_validate(w)
        if not verify(oracle(w),evidence_checks=False,invariants=False)['pass']:gold_fail.append(w.id)
print('gold failing under oracle identity',len(gold_fail),gold_fail[:10])
for run,split in [('results/evaluation_qwen14_dev/lora','dev'),('results/evaluation_qwen14_test/lora','test')]:
    docs={w['id']:Workflow.model_validate(w) for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text(encoding='utf-8'))}
    rows=[json.loads(l) for l in (ROOT/f'data/processed/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    preds=[json.loads(l) for l in (ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    c=Counter()
    for w in assemble_workflows(rows,preds,docs):
        if not w.metadata['schema_complete']:continue
        g=gold_identity(w.id)
        a=verify(g.rename(w,g.gold().initial),evidence_checks=False,invariants=False)['pass']
        b=verify(oracle(w),evidence_checks=False,invariants=False)['pass']
        c[(a,b)]+=1
    print(run,'(rename pass, oracle pass):',dict(c))
