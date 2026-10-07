"""First-pass quality of evaluation runs, with lifecycle pass under text and annotated identity."""
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.annotation import gold_identity
from sop2program.ir import Workflow
from sop2program.metrics import assemble_workflows, evaluate_predictions
from sop2program.paths import DATA, KNOWLEDGE
from sop2program.verify import verify

rules=json.loads((KNOWLEDGE/'full_rules.json').read_text(encoding='utf-8'))
for run in sys.argv[1:]:
    split=json.loads((ROOT/run/'run_config.json').read_text(encoding='utf-8'))['split']
    docs={w['id']:Workflow.model_validate(w) for w in json.loads((DATA/f'{split}_workflows.json').read_text(encoding='utf-8'))}
    rows=[json.loads(l) for l in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    preds=[json.loads(l) for l in (ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    s,_=evaluate_predictions(rows,preds,docs,rules)
    chain=text=0
    for w in assemble_workflows(rows,preds,docs):
        if not w.metadata['schema_complete']:continue
        g=gold_identity(w.id)
        text+=verify(w,evidence_checks=False,invariants=False)['pass']
        chain+=verify(g.rename(w,g.gold().initial),evidence_checks=False,invariants=False)['pass']
    print(run,split,{'schema':round(s['schema_valid_rate'],3),'action':round(s['action_accuracy'],3),
                     'arg_f1':round(s['argument_tuple']['f1'],3),'pre':round(s['weak_precondition_exact_rate'],3),
                     'op_exact':round(s['operator_exact_rate'],3),'complete':s['complete_documents'],
                     'violations':sum(s['violation_counts'].values()),'lifecycle_text':text,'lifecycle_chain':chain})
