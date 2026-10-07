"""CPU-only audit of supplementary controls and archived-system decomposition.

--existing recomputes primary/cascade outcomes and development-overlap sensitivity.
--controlled checks all new saved endpoints and their generation accounting.
This script does not generate predictions, select favorable arms, or tune a model.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'paper/applied_intelligence/evidence'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def write(path, data):
    (OUT / path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def adjust(rows):
    last = 0
    for rank, row in enumerate(sorted(rows, key=lambda r: r['p'])):
        last = max(last, min(1., (len(rows) - rank) * row['p']))
        row['holm_p'] = last
    return rows


def existing_worker(fold):
    from scripts.recheck_annotation_identity import recheck
    from sop2program.ir import Workflow
    from sop2program.verify import verify
    rows = []
    base = f'results/cv/fold{fold}'
    rules = read(f'{base}/knowledge/full_rules.json')
    for compiler in ('fewshot3', 'lora3b', 'qwen14'):
        repair = f'{base}/repair_trained_{compiler}'
        primary = recheck(repair, 'producer_feedback')
        summary = read(f'{repair}/summary.json')
        final = read(f'{repair}/producer_feedback_workflows.json')
        cascade = read(f'{base}/recheck_trained_{compiler}.json')[0]
        for d in summary['producer_feedback']['details']:
            if d['status'] in ('PASS', 'REPAIRED'):
                assert verify(Workflow.model_validate(final[d['doc']]), rules)['pass']
        for doc, values in primary['chain_identity_by_document'].items():
            rows.append({'fold': fold, 'compiler': compiler, 'document': doc,
                         'primary': int(values[1]),
                         'cascade': int(cascade['chain_identity_by_document'][doc][1])})
    return rows


def existing():
    from paper.applied_intelligence.tools.audit_submission import paired
    rows = []
    for fold in range(5):
        env = dict(os.environ, SOP_DATA=f'data/cv/fold{fold}',
                   SOP_KNOWLEDGE=f'results/cv/fold{fold}/knowledge', PYTHONUTF8='1')
        r = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__)), '--existing-worker', str(fold)],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', check=True)
        rows.extend(json.loads(r.stdout))
    dev = {w['id'] for w in read('data/processed/dev_workflows.json')}
    audit = read('paper/applied_intelligence/evidence/audit.json')
    import io
    records = list(csv.DictReader(io.StringIO((OUT / 'document_outcomes.csv').read_text(encoding='utf-8'))))
    decomposed = []
    sensitivity = []
    for c in ('fewshot3', 'lora3b', 'qwen14'):
        subset = [r for r in rows if r['compiler'] == c]
        first = {r['document']: bool(r['primary']) for r in subset}
        final = {r['document']: bool(r['cascade']) for r in subset}
        assert len(first) == 276 and sum(final.values()) == audit['pooled'][c]['trained']['confirmed']
        decomposed.append({'compiler': c, 'primary_confirmed': sum(first.values()),
                           'cascade_confirmed': sum(final.values()), **paired(first, final)})
        for excluded in (False, True):
            take = [r for r in records if r['compiler'] == c and (not excluded or r['document'] not in dev)]
            columns = ('first', 'deterministic', 'trained', 'untuned') if c == 'fewshot3' else ('first', 'deterministic', 'trained')
            sensitivity.append({'compiler': c, 'subset': 'excluding_original_dev' if excluded else 'all',
                                'documents': len(take),
                                **{a: sum(int(r[a]) for r in take) for a in columns},
                                'per_fold': [{'fold': k, 'documents': sum(r['document'] in {x['document'] for x in subset if x['fold'] == k} for r in take),
                                              **{a: sum(int(r[a]) for r in take if r['document'] in {x['document'] for x in subset if x['fold'] == k}) for a in columns}}
                                             for k in range(5)]})
    result = {'scope': 'Post hoc descriptive decomposition and sensitivity, not independent validation or a causal ablation.',
              'original_dev_overlap': len(dev), 'decomposition': decomposed, 'sensitivity': sensitivity,
              'documents': rows}
    write('strengthening_existing.json', result)
    print(json.dumps({k: result[k] for k in ('decomposition', 'sensitivity')}, indent=2))


def controlled():
    from scripts.controlled_repair import ARMS, BUDGETS
    from sop2program.annotation import GoldIdentity
    from sop2program.compiler import user_content
    from sop2program.ir import Workflow, Operator
    from sop2program.metrics import assemble_workflows
    from sop2program.verify import verify
    from paper.applied_intelligence.tools.audit_submission import paired, wilson
    protocol_path = ROOT / 'results/controlled_repair/protocol.json'
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    data_rows = []
    dev = {w['id'] for w in read('data/processed/dev_workflows.json')}
    for fold in range(5):
        gold = {w['id']: Workflow.model_validate(w) for w in read(f'data/cv/fold{fold}/test_workflows.json')}
        rules = read(f'results/cv/fold{fold}/knowledge/full_rules.json')
        identities = {d: GoldIdentity(w, ROOT / w.metadata['peg_file']) for d, w in gold.items()}
        for arm in ARMS:
            path = f'results/controlled_repair/fold{fold}/{arm}.json'
            saved = read(path)
            assert saved['complete'] and saved['protocol_sha256'] == protocol_hash
            assert set(saved['documents']) == set(gold)
            events = {}
            for e in saved['events']:
                events.setdefault(e['doc'], []).append(e)
                assert e['info']['finite_logits_checked']
                assert 0 < e['info']['output_tokens'] <= 384
                assert ('repair_feedback' in e['prompt_row']) == (arm != 'producer_no_feedback')
                try:
                    op = Operator.model_validate_json(e['info']['raw'])
                except Exception:
                    op = None
                if op is not None:
                    op = op.model_copy(update={'trigger': e['prompt_row']['trigger']})
                assert (op.model_dump() if op is not None else None) == e['operator'], 'Raw generation mismatch'
            for doc, s in saved['documents'].items():
                ev = events.get(doc, [])
                assert [e['sequence'] for e in ev] == list(range(1, s['calls'] + 1))
                assert s['calls'] <= max(BUDGETS)
                assert s['output_tokens'] == sum(e['info']['output_tokens'] for e in ev)
                for budget in BUDGETS:
                    row = s['budgets'][str(budget)]
                    assert row['calls'] <= budget
                    prefix = ev[:row['calls']]
                    assert row['output_tokens'] == sum(e['info']['output_tokens'] for e in prefix)
                    assert row['input_tokens'] == sum(e['info']['input_tokens'] for e in prefix)
                    final = Workflow.model_validate(row['workflow'])
                    assert final.source == gold[doc].source and final.initial == gold[doc].initial
                    accepted = row['status'] in ('PASS', 'REPAIRED')
                    if accepted:
                        assert final.metadata.get('schema_complete') and verify(final, rules)['pass']
                    identity = identities[doc]
                    confirmed = bool(accepted and verify(identity.rename(final, identity.gold().initial),
                                                         evidence_checks=False, invariants=False)['pass'])
                    data_rows.append({'fold': fold, 'document': doc, 'arm': arm, 'budget': budget,
                                      'accepted': int(accepted), 'confirmed': int(confirmed),
                                      'original_dev': int(doc in dev),
                                      **{k: row[k] for k in ('calls', 'input_tokens', 'output_tokens', 'generation_seconds')}})
    pooled = []
    for arm in ARMS:
        for budget in BUDGETS:
            rows = [r for r in data_rows if r['arm'] == arm and r['budget'] == budget]
            assert len(rows) == 276 and len({r['document'] for r in rows}) == 276
            confirmed = sum(r['confirmed'] for r in rows)
            accepted = sum(r['accepted'] for r in rows)
            pooled.append({'arm': arm, 'budget': budget, 'documents': len(rows), 'confirmed': confirmed,
                           'confirmed_wilson95': wilson(confirmed, len(rows)), 'accepted': accepted,
                           'unconfirmed_share': (accepted - confirmed) / accepted if accepted else None,
                           **{k: sum(r[k] for r in rows) for k in ('calls', 'input_tokens', 'output_tokens', 'generation_seconds')},
                           'fold_confirmed': [sum(r['confirmed'] for r in rows if r['fold'] == k) for k in range(5)],
                           'excluding_original_dev_confirmed': sum(r['confirmed'] for r in rows if not r['original_dev'])})
    chosen = {r['document']: bool(r['confirmed']) for r in data_rows if r['arm'] == 'producer_feedback' and r['budget'] == 8}
    comparisons = []
    for arm in ARMS[1:]:
        baseline = {r['document']: bool(r['confirmed']) for r in data_rows if r['arm'] == arm and r['budget'] == 8}
        comparisons.append({'comparison': f'producer_feedback vs {arm}', **paired(baseline, chosen)})
    result = {'protocol_sha256': protocol_hash, 'documents': 276,
              'pooled': pooled, 'comparisons': adjust(comparisons),
              'assertions': 'All 20 arms complete; 4416 endpoints checked; full accepted verifier, annotation replay, source/inventory, raw generations, caps and token accounting passed.'}
    write('controlled_audit.json', result)
    with (OUT / 'controlled_document_outcomes.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data_rows[0]))
        writer.writeheader(); writer.writerows(data_rows)
    print(json.dumps(result, indent=2))


def replay_worker(fold):
    """Regenerate every controller decision from saved model outputs, with no model."""
    from contextlib import redirect_stdout
    import tempfile
    from scripts.controlled_repair import ARMS, run_arm
    from sop2program.ir import Workflow, Operator
    from sop2program.metrics import assemble_workflows
    data = ROOT / f'data/cv/fold{fold}'
    base = ROOT / f'results/cv/fold{fold}'
    docs = {w['id']: Workflow.model_validate(w) for w in read(data / 'test_workflows.json')}
    rows = [json.loads(l) for l in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    pred = [json.loads(l) for l in (base / 'eval_qwen14/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    workflows = assemble_workflows(rows, pred, docs)
    rules = read(base / 'knowledge/full_rules.json')
    lexicon = read(base / 'knowledge/trigger_lexicon.json')['lexicon']
    payload = read(base / 'knowledge/setting_lexicon.json')
    mentions = {k: set(payload[k]) for k in ('settings', 'products')}

    class RecordedCompiler:
        def __init__(self, events):
            self.events, self.cursor = events, 0

        def compile_batch(self, requests):
            outputs = []
            for request in requests:
                event = self.events[self.cursor]
                self.cursor += 1
                assert json.loads(json.dumps(request)) == event['prompt_row'], 'Reconstructed prompt differs'
                outputs.append((Operator.model_validate(event['operator']) if event['operator'] else None,
                                event['info']))
            return outputs

    result = []
    # A fresh, uniquely allocated directory ensures no completed checkpoint can bypass replay.
    (ROOT / 'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='controlled_replay_', dir=ROOT / 'tmp') as temp:
        for arm in ARMS:
            saved = read(f'results/controlled_repair/fold{fold}/{arm}.json')
            compiler = RecordedCompiler(saved['events'])
            with redirect_stdout(sys.stderr):
                rebuilt = run_arm(arm, workflows, compiler, rules, lexicon, mentions,
                                  Path(temp) / f'{arm}.json', saved['protocol_sha256'])
            assert compiler.cursor == len(saved['events'])
            # JSON normalization also checks every workflow, status, cost, cap,
            # prompt, progression decision, stop decision and token/time counter.
            assert json.loads(json.dumps(rebuilt)) == saved, (fold, arm, 'Controller replay mismatch')
            result.append({'fold': fold, 'arm': arm, 'documents': len(docs),
                           'sequences': compiler.cursor, 'all_saved_fields_equal': True})
    return result


def controller_replay():
    results = []
    for fold in range(5):
        env = dict(os.environ, SOP_DATA=f'data/cv/fold{fold}',
                   SOP_KNOWLEDGE=f'results/cv/fold{fold}/knowledge', PYTHONUTF8='1')
        r = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__)), '--replay-worker', str(fold)],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        if r.returncode:
            raise RuntimeError(r.stderr)
        results.extend(json.loads(r.stdout))
        print(f'Controller replay passed fold {fold}', flush=True)
    result = {'scope': 'CPU controller reconstruction using saved model outputs; no model loaded',
              'arms': results, 'all_saved_fields_equal': True}
    write('controlled_controller_replay.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--existing', action='store_true')
    group.add_argument('--existing-worker', type=int)
    group.add_argument('--controlled', action='store_true')
    group.add_argument('--controller-replay', action='store_true')
    group.add_argument('--replay-worker', type=int)
    args = parser.parse_args()
    if args.existing_worker is not None:
        print(json.dumps(existing_worker(args.existing_worker)))
    elif args.existing:
        existing()
    elif args.replay_worker is not None:
        print(json.dumps(replay_worker(args.replay_worker)))
    elif args.controller_replay:
        controller_replay()
    else:
        controlled()
