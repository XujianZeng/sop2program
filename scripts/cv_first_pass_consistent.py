"""First-pass counts under the same criterion as repaired outputs.

A repaired document counts when the full text-identity verifier (state, evidence, invariants)
accepts it and annotated identity confirms its state replay. The recheck's first-pass column
used state checks only; this recomputes first pass with full acceptance as well, per fold.
Prints one JSON line per compiler with the consistent first-pass outcome by document.
"""
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.ir import Workflow
from sop2program.metrics import assemble_workflows
from sop2program.paths import DATA,KNOWLEDGE
from sop2program.verify import verify

run=sys.argv[1];recheck=sys.argv[2]
rules=json.loads((KNOWLEDGE/'full_rules.json').read_text(encoding='utf-8'))
chain=json.loads((ROOT/recheck).read_text(encoding='utf-8'))[0]['chain_identity_by_document']
split=json.loads((ROOT/run/'run_config.json').read_text(encoding='utf-8'))['split']
docs={w['id']:Workflow.model_validate(w) for w in json.loads((DATA/f'{split}_workflows.json').read_text(encoding='utf-8'))}
rows=[json.loads(l) for l in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
preds=[json.loads(l) for l in (ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
out={}
for w in assemble_workflows(rows,preds,docs):
    full=w.metadata['schema_complete'] and verify(w,rules)['pass']
    out[w.id]=bool(full and chain[w.id][0])
print(json.dumps(out))
