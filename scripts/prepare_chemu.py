"""Second corpus: ChEMU 2020 snippets joined with ChEMU-Ref, as a frozen external test.

The official ChEMU-Ref training snippets are split by source-text group into training and
rule-confirmation (dev) parts; the official development snippets are the test set, never
consulted while the method was designed on X-WLP. Prompt rows mirror scripts/prepare_data.py,
except that the local unit is the annotated sentence (a snippet is one paragraph), and the
annotated chains go to {split}_identity.json, outside every prompt.

The raw corpora are licensed for research use only and are not redistributed; this script
and the snippet ids reproduce every derived file from the official downloads.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sop2program.chemu import derive, read_ann, sentence_bounds
from sop2program.induce import mine
from sop2program.ir import Workflow
from sop2program.verify import verify

RAW = ROOT / 'data/raw/chemu'


def load(split):
    out = []
    for line in (RAW / f'ref_{split}.jsonl').open(encoding='utf-8'):
        ref = json.loads(line)
        key = ref['doc_key']
        text = (RAW / f'ee_{split}/{key}.txt').read_text(encoding='utf-8')
        entities, relations = read_ann(RAW / f'ee_{split}/{key}.ann')
        w, chains, names = derive(f'chemu_{split}_{key}', text, entities, relations, ref)
        w.metadata['source_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        out.append((w, {'chains': chains, 'names': {str(c): n for c, n in names.items()}},
                    sentence_bounds(text, ref['sentences'])))
    return out


def rows_for(w, bounds):
    trace = verify(w, evidence_checks=False, invariants=False)['trace']
    rows = []
    for i, (s, t) in enumerate(zip(w.steps, trace)):
        ev = s.evidence['trigger']
        start, end = next((a, b) for a, b in bounds if ev.start < b or (a, b) == bounds[-1])
        rows.append({'id': w.id + ':' + s.id, 'doc': w.id, 'step': i, 'source': w.source, 'local': w.source[start:end],
                     'context': w.source[max(0, start - 400):end], 'trigger': s.operator.trigger, 'state': t['before'],
                     'target': s.operator.model_dump(), 'initial': w.initial, 'offset': ev.start})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/chemu')
    p.add_argument('--knowledge', default='results/chemu/knowledge')
    p.add_argument('--dev-share', type=float, default=.15)
    args = p.parse_args()
    official = {'train': load('train'), 'test': load('dev')}
    groups = sorted({w.metadata['source_sha256'] for w, _, _ in official['train']})
    test_groups = {w.metadata['source_sha256'] for w, _, _ in official['test']}
    random.Random(42).shuffle(groups)
    dev_groups = set(groups[:int(args.dev_share * len(groups))])
    parts = {'train': [x for x in official['train'] if x[0].metadata['source_sha256'] not in dev_groups | test_groups],
             'dev': [x for x in official['train'] if x[0].metadata['source_sha256'] in dev_groups - test_groups],
             'test': official['test']}
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'corpus': 'ChEMU 2020 event extraction + ChEMU-Ref (train/dev releases)',
                'split': 'test = official ChEMU-Ref dev snippets; train/dev = official train split by source group (seed 42); '
                         'training snippets whose text equals a test snippet are dropped',
                'licence': 'Elsevier limited data licence: research use; raw and derived data not redistributed',
                'documents': {}, 'counts': {}}
    for split, items in parts.items():
        rows = [r for w, _, bounds in items for r in rows_for(w, bounds)]
        (out / f'{split}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
        (out / f'{split}_workflows.json').write_text(json.dumps([w.model_dump() for w, _, _ in items], ensure_ascii=False),
                                                    encoding='utf-8')
        (out / f'{split}_identity.json').write_text(json.dumps({w.id: ident for w, ident, _ in items}, ensure_ascii=False),
                                                   encoding='utf-8')
        manifest['documents'][split] = [w.id for w, _, _ in items]
        manifest['counts'][split] = {'documents': len(items), 'operators': len(rows)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    to_workflows = lambda split: [Workflow.model_validate(w.model_dump()) for w, _, _ in parts[split]]
    rules, audit = mine(to_workflows('train'), holdout=to_workflows('dev'))
    knowledge = ROOT / args.knowledge
    knowledge.mkdir(parents=True, exist_ok=True)
    (knowledge / 'full_rules.json').write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding='utf-8')
    (knowledge / 'full_candidates.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    families = {f: sum(r['family'] == f for r in rules) for f in sorted({r['family'] for r in rules})}
    print(json.dumps(manifest['counts']), len(rules), families)


if __name__ == '__main__':
    main()
