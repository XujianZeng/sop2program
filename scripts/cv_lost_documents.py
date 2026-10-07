"""Documents executable at first pass under annotated identity but not after deterministic repair.

For each, record whether the text-identity verifier had flagged the first pass (so repair was
triggered by a text-only violation) and which operations the repair applied.
"""
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.metrics import assemble_workflows
from sop2program.ir import Workflow
from sop2program.paths import DATA,KNOWLEDGE
from sop2program.verify import verify

compiler=sys.argv[1];k=int(sys.argv[2])
rules=json.loads((KNOWLEDGE/'full_rules.json').read_text(encoding='utf-8'))
recheck=json.loads((ROOT/f'results/cv/fold{k}/recheck_deterministic_{compiler}.json').read_text(encoding='utf-8'))[0]
summary=json.loads((ROOT/f'results/cv/fold{k}/repair_deterministic_{compiler}/summary.json').read_text(encoding='utf-8'))
ops={d['doc']:d.get('operations',[]) for d in summary['deterministic_repair']['details']}
docs={w['id']:Workflow.model_validate(w) for w in json.loads((DATA/'test_workflows.json').read_text(encoding='utf-8'))}
rows=[json.loads(l) for l in (DATA/'test.jsonl').read_text(encoding='utf-8').splitlines()]
preds=[json.loads(l) for l in (ROOT/f'results/cv/fold{k}/eval_{compiler}/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
first={w.id:w for w in assemble_workflows(rows,preds,docs)}
for doc,(before,after) in recheck['chain_identity_by_document'].items():
    if before and not after:
        full=verify(first[doc],rules)
        state=verify(first[doc],evidence_checks=False,invariants=False)
        codes=sorted({v['code'] for v in full['violations']})
        print(json.dumps({'fold':k,'doc':doc,'text_state_pass':state['pass'],'full_text_pass':full['pass'],
                          'violation_codes':codes,'operations':ops.get(doc)}))
