"""Audit leakage, source offsets and weak-supervision limitations without a model."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.ir import Workflow
from sop2program.verify import verify


def main():
    summary={};hashes={};ids={}
    for split in ('train','dev','test'):
        path=ROOT/f'data/processed/{split}.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        docs=[Workflow.model_validate(w) for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text(encoding='utf-8'))]
        hashes[split]={hashlib.sha256(w.source.encode()).hexdigest() for w in docs}
        ids[split]={w.id for w in docs}
        assert len({r['id'] for r in rows})==len(rows),'duplicate example IDs'
        assert all(r['source'][r['offset']:r['offset']+len(r['trigger'])]==r['trigger'] for r in rows),'bad trigger offset'
        assert len(rows)==sum(len(w.steps) for w in docs),'lost operators'
        arguments=[a for r in rows for a in r['target']['arguments']]
        missing_context=sum(a['text'].casefold() not in r['context'].casefold() for r in rows for a in r['target']['arguments'])
        replays=[verify(w,invariants=False) for w in docs]
        summary[split]={'documents':len(docs),'operators':len(rows),'arguments':len(arguments),
                        'arguments_outside_prompt_context':missing_context,
                        'verifier_clean_weak_documents':sum(r['pass'] for r in replays),
                        'weak_uncommitted_steps':sum(not t['committed'] for r in replays for t in r['trace']),
                        'jsonl_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not (hashes[a]&hashes[b]),f'source leakage {a}/{b}'
        assert not (ids[a]&ids[b]),f'document leakage {a}/{b}'
    summary['split_audit']='PASS: disjoint document IDs and exact source hashes; valid trigger offsets'
    summary['limitations']=['Custom 70/15/15 document split, not official X-WLP split.',
                           'Gold action mentions and argument-derived inventory are supplied.',
                           'Effects, preconditions and dependencies are template projections, not human gold.',
                           'Some gold arguments are absent from bounded source context; reported explicitly.',
                           'No external-source corpus or multi-seed significance test is included.']
    (ROOT/'results/data_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
