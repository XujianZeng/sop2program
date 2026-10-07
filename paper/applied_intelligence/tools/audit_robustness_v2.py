"""CPU audit v2: handles absent saved finals for schema-incomplete REVIEW records.

The v1 source remains frozen and archived. This input-format repair does not alter
the predeclared scoring policies, GPU experiment, strata or statistical comparisons.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.controlled_repair import read, digest, atomic, ARMS, BUDGETS, run_arm
from scripts.refinement_extension import ARM, OUT as RUN, CandidateOnlyCompiler, load_fold
from sop2program.annotation import GoldIdentity
from sop2program.ir import norm
from paper.applied_intelligence.tools.audit_submission import paired, wilson

OUT = ROOT / 'paper/applied_intelligence/evidence'


class StrictIdentity(GoldIdentity):
    """No span expansion or ambiguous/state-dependent tie-breaking; abstain explicitly."""
    def resolve(self, text, index, alive=None):
        key = norm(text)
        local = {c for a, c in zip(self.workflow.steps[index].operator.arguments, self.mentions[index])
                 if c is not None and norm(a.text) == key} if index < len(self.mentions) else set()
        candidates = local or self.by_text.get(key, set())
        matched = len(candidates) == 1
        self.coverage[(index, key)] = 'local' if matched and local else 'document' if matched else 'ambiguous' if candidates else 'unmatched'
        return self.name[next(iter(candidates))] if matched else key


def features(gold, identity, gold_chain):
    distances = [i - int(d[1:]) for i, s in enumerate(gold_chain.steps) for d in s.depends_on]
    return {'operators': len(gold.steps), 'annotated_entities': len(identity.name),
            'max_reference_dependency_distance': max(distances, default=0),
            'same_name_distinct_chain_names': sum(len(cs) > 1 for cs in identity.by_text.values())}


def score(workflow, accepted, identity, strict, initial):
    from sop2program.verify import verify
    published = identity.rename(workflow, initial)
    p = verify(published, evidence_checks=False, invariants=False)
    independent = identity.rename(workflow, initial, charitable=False)
    q = verify(independent, evidence_checks=False, invariants=False)
    strict.coverage = {}
    renamed = strict.rename(workflow, initial, charitable=False)
    counts = Counter(strict.coverage.values())
    covered = not (counts['ambiguous'] or counts['unmatched'])
    s = verify(renamed, evidence_checks=False, invariants=False) if covered else None
    return {'published': bool(accepted and p['pass']),
            'state_independent': bool(accepted and q['pass']),
            'strict_unique': bool(accepted and s['pass']) if covered else None,
            'fully_assessable': covered, 'matched_references': counts['local'] + counts['document'],
            'total_references': sum(counts.values()), 'resolution_counts': dict(counts),
            'published_state_pass': p['pass'], 'published_violations': p['violations'][:4]}, p


def worker(fold, include_new=True):
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows
    from sop2program.verify import verify
    from scripts.recheck_annotation_identity import diagnose
    gold, _, rules, _, _ = load_fold(fold)
    data = ROOT / f'data/cv/fold{fold}'
    base = ROOT / f'results/cv/fold{fold}'
    input_rows = [json.loads(l) for l in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    configurations = {}
    for compiler in ('fewshot3', 'lora3b', 'qwen14'):
        preds = [json.loads(l) for l in (base / f'eval_{compiler}/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        first = assemble_workflows(input_rows, preds, gold)
        configurations[(f'main_{compiler}', 'first')] = {w.id: (w, bool(w.metadata.get('schema_complete') and verify(w, rules)['pass'])) for w in first}
        for kind, arm in [('deterministic', 'deterministic_repair'), ('trained', 'cascade')] + ([('untuned', 'producer_feedback')] if compiler == 'fewshot3' else []):
            folder = base / f'repair_{kind}_{compiler}'
            summary = read(folder / 'summary.json')[arm]['details']
            finals = read(folder / f'{arm}_workflows.json')
            configs = {}
            first_map = {w.id: w for w in first}
            for d in summary:
                accepted = d['status'] in ('PASS', 'REPAIRED')
                if d['doc'] in finals:
                    final = Workflow.model_validate(finals[d['doc']])
                else:
                    final = first_map[d['doc']]
                    assert not accepted and not final.metadata.get('schema_complete'), (compiler, d)
                configs[d['doc']] = (final, accepted)
            configurations[(f'main_{compiler}', kind)] = configs
    for arm in (*ARMS, *([ARM] if include_new else [])):
        folder = RUN if arm == ARM else ROOT / 'results/controlled_repair'
        saved = read(folder / f'fold{fold}/{arm}.json')
        assert saved['complete']
        configurations[('controlled', arm)] = {doc: (Workflow.model_validate(s['budgets']['8']['workflow']), s['budgets']['8']['status'] in ('PASS', 'REPAIRED')) for doc, s in saved['documents'].items()}
    result = []
    for doc, reference in gold.items():
        identity = GoldIdentity(reference, ROOT / reference.metadata['peg_file'])
        strict = StrictIdentity(reference, ROOT / reference.metadata['peg_file'])
        gold_chain = identity.gold()
        f = features(reference, identity, gold_chain)
        for (family, arm), workflows in configurations.items():
            w, accepted = workflows[doc]
            assert w.source == reference.source and w.initial == reference.initial
            if accepted:
                assert w.metadata.get('schema_complete') and verify(w, rules)['pass']
            values, p = score(w, accepted, identity, strict, gold_chain.initial)
            cause = diagnose(p, gold_chain, w) if accepted and not p['pass'] else None
            result.append({'fold': fold, 'document': doc, 'family': family, 'arm': arm,
                           'accepted': accepted, **f, **values, 'automatic_cause': cause})
    return result


def run_workers(flag, existing=False):
    def one(fold):
        env = dict(os.environ, SOP_DATA=f'data/cv/fold{fold}', SOP_KNOWLEDGE=f'results/cv/fold{fold}/knowledge', PYTHONUTF8='1')
        cmd = [sys.executable, '-X', 'utf8', str(Path(__file__)), flag, str(fold)]
        if existing: cmd.append('--existing-only')
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        if p.returncode: raise RuntimeError(p.stderr)
        print(f'{flag} fold {fold} complete', flush=True)
        return json.loads(p.stdout)
    with ThreadPoolExecutor(max_workers=3) as pool:
        return [item for values in pool.map(one, range(5)) for item in values]


def counts(rows, mode):
    eligible = [r for r in rows if r[mode] is not None]
    return {'documents': len(rows), 'assessed': len(eligible),
            'confirmed': sum(r[mode] for r in eligible), 'accepted': sum(r['accepted'] for r in rows)}


def analyze(existing=False):
    protocol = read(RUN / 'protocol.json')
    amendment = read(RUN / 'analysis_implementation_amendment.json')
    assert digest(Path(__file__)) == amendment['analysis_sha256']
    # CPU archives omit weights; validate only distributed analytical dependencies here.
    excluded = ('models/', '/adapter/')
    for rel, expected in protocol['source_sha256'].items():
        if rel.startswith(excluded[0]) or excluded[1] in rel: continue
        assert digest(ROOT / rel) == expected, f'Frozen analytical input changed: {rel}'
    records = run_workers('--worker', existing)
    groups = sorted({(r['family'], r['arm']) for r in records})
    original = read(OUT / 'audit.json')
    with (OUT / 'controlled_document_outcomes.csv').open(encoding='utf-8') as f:
        expected_controls = {(r['document'], r['arm']): int(r['confirmed']) for r in csv.DictReader(f) if r['budget'] == '8'}
    pooled, common, strata, causes = [], [], [], []
    for family, arm in groups:
        rows = [r for r in records if (r['family'], r['arm']) == (family, arm)]
        assert len(rows) == 276 and len({r['document'] for r in rows}) == 276
        n = sum(r['published'] for r in rows)
        if family.startswith('main_'):
            c = family[5:]
            expected = original['pooled'][c]['first'] if arm == 'first' else original['pooled'][c][arm]['confirmed']
            assert n == expected, (family, arm, n, expected)
        elif arm in ARMS:
            assert all(r['published'] == expected_controls[r['document'], arm] for r in rows)
        pooled.append({'family': family, 'arm': arm, 'modes': {m: counts(rows, m) for m in ('published', 'state_independent', 'strict_unique')},
                       'matched_references': sum(r['matched_references'] for r in rows), 'total_references': sum(r['total_references'] for r in rows),
                       'changed_by_state_preference': sum(r['published'] != r['state_independent'] for r in rows),
                       'resolution_counts': dict(sum((Counter(r['resolution_counts']) for r in rows), Counter()))})
        causes.append({'family': family, 'arm': arm, 'accepted': sum(r['accepted'] for r in rows),
                       'unconfirmed': sum(r['accepted'] and not r['published'] for r in rows),
                       'causes': dict(Counter(r['automatic_cause'] for r in rows if r['automatic_cause']))})
        for feature, cuts in protocol['strata'].items():
            for j, label in enumerate((f'<= {cuts[0]}', f'{cuts[0]+1}..{cuts[1]}', f'> {cuts[1]}')):
                take = [r for r in rows if sum(r[feature] > cut for cut in cuts) == j]
                strata.append({'family': family, 'arm': arm, 'feature': feature, 'bin': label, **counts(take, 'published')})
    for family in sorted({f for f, _ in groups}):
        arms = [a for f, a in groups if f == family]
        by_doc = {}
        for r in records:
            if r['family'] == family: by_doc.setdefault(r['document'], {})[r['arm']] = r
        keep = {d: rows for d, rows in by_doc.items() if all(rows[a]['fully_assessable'] for a in arms)}
        common.append({'family': family, 'documents': len(keep), 'selection': 'All compared arms must be fully strictly resolvable; descriptive selected subset, not representative or independent.',
                       'arms': {a: {m: sum(rows[a][m] for rows in keep.values()) for m in ('published', 'state_independent', 'strict_unique')} for a in arms},
                       'document_ids': sorted(keep)})
    contrasts = []
    indexed = {(r['family'], r['arm'], r['document']): r for r in records}
    for family in sorted({f for f, _ in groups}):
        arms = [a for f, a in groups if f == family]
        pairs = [(a, 'producer_feedback') for a in arms if a != 'producer_feedback'] if family == 'controlled' else [('first', 'deterministic'), ('deterministic', 'trained')]
        for a, b in pairs:
            ids = [r['document'] for r in records if r['family'] == family and r['arm'] == a]
            for mode in ('published', 'state_independent'):
                av = {d: indexed[family, a, d][mode] for d in ids}
                bv = {d: indexed[family, b, d][mode] for d in ids}
                # Descriptive counts only: avoid an unplanned collection of significance tests.
                contrasts.append({'family': family, 'baseline': a, 'candidate': b, 'mode': mode, 'documents': len(ids),
                                  'gain': sum(bv[d] and not av[d] for d in ids), 'loss': sum(av[d] and not bv[d] for d in ids),
                                  'candidate_only_ids': [d for d in ids if bv[d] and not av[d]],
                                  'baseline_only_ids': [d for d in ids if av[d] and not bv[d]]})
    report = {'protocol_sha256': digest(RUN / 'protocol.json'), 'scope': 'Post hoc descriptive automatic sensitivity and diagnostics; no human validation.',
              'includes_candidate_only': not existing, 'documents': 276, 'records': len(records), 'pooled': pooled,
              'common_subsets': common, 'strata': strata, 'causes': causes, 'contrasts': contrasts,
              'assertions': 'Original published main totals and all archived controlled per-document outcomes reproduced; original sources/inventories preserved.'}
    suffix = '_interim' if existing else ''
    atomic(OUT / f'robustness_audit{suffix}.json', report)
    atomic(OUT / f'robustness_details{suffix}.json', records)
    with (OUT / f'robustness_strata{suffix}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(strata[0])); writer.writeheader(); writer.writerows(strata)
    print(json.dumps({'pooled': pooled, 'common': [{k:v for k,v in s.items() if k != 'document_ids'} for s in common]}, indent=2))


def omission_worker(fold):
    from sop2program.ir import Workflow, Operator
    from sop2program.verify import verify
    gold, workflows, rules, lexicon, mentions = load_fold(fold)
    saved = read(RUN / f'fold{fold}/{ARM}.json')
    assert saved['complete'] and saved['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert set(saved['documents']) == set(gold)
    class Recorded:
        def __init__(self): self.cursor = 0
        def compile_batch(self, rows):
            out = []
            for row in rows:
                e = saved['events'][self.cursor]; self.cursor += 1
                assert row == e['prompt_row'] and set(row['repair_feedback']) == {'candidate'}
                info = e['info']
                assert info['finite_logits_checked'] and 0 < info['output_tokens'] <= 384
                try: op = Operator.model_validate_json(info['raw'])
                except Exception: op = None
                if op is not None: op = op.model_copy(update={'trigger': row['trigger']})
                assert (op.model_dump() if op else None) == e['operator']
                out.append((op, info))
            return out
    compiler = Recorded()
    (ROOT / 'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='omission_replay_', dir=ROOT / 'tmp') as tmp:
        with redirect_stdout(sys.stderr):
            rebuilt = run_arm(ARM, workflows, CandidateOnlyCompiler(compiler), rules, lexicon, mentions,
                              Path(tmp) / 'replayed.json', saved['protocol_sha256'])
        assert compiler.cursor == len(saved['events']) and json.loads(json.dumps(rebuilt)) == saved
    rows = []
    for doc, s in saved['documents'].items():
        identity = GoldIdentity(gold[doc], ROOT / gold[doc].metadata['peg_file'])
        ev = [e for e in saved['events'] if e['doc'] == doc]
        assert [e['sequence'] for e in ev] == list(range(1, s['calls'] + 1))
        for budget in BUDGETS:
            row = s['budgets'][str(budget)]
            assert row['calls'] <= budget
            for name in ('input_tokens', 'output_tokens'):
                assert row[name] == sum(e['info'][name] for e in ev[:row['calls']])
            w = Workflow.model_validate(row['workflow'])
            assert w.source == gold[doc].source and w.initial == gold[doc].initial
            accepted = row['status'] in ('PASS', 'REPAIRED')
            if accepted: assert w.metadata.get('schema_complete') and verify(w, rules)['pass']
            confirmed = bool(accepted and verify(identity.rename(w, identity.gold().initial), evidence_checks=False, invariants=False)['pass'])
            rows.append({'fold': fold, 'document': doc, 'budget': budget, 'accepted': accepted, 'confirmed': confirmed,
                         **{k: row[k] for k in ('calls', 'input_tokens', 'output_tokens', 'generation_seconds')}})
    return rows


def omission():
    rows = run_workers('--omission-worker')
    assert len(rows) == 1104
    pooled = []
    for cap in BUDGETS:
        subset = [r for r in rows if r['budget'] == cap]
        n = sum(r['confirmed'] for r in subset)
        pooled.append({'budget': cap, 'documents': 276, 'confirmed': n, 'wilson95': wilson(n, 276),
                       'accepted': sum(r['accepted'] for r in subset),
                       'fold_confirmed': [sum(r['confirmed'] for r in subset if r['fold'] == k) for k in range(5)],
                       **{k: sum(r[k] for r in subset) for k in ('calls', 'input_tokens', 'output_tokens', 'generation_seconds')}})
    with (OUT / 'controlled_document_outcomes.csv').open(encoding='utf-8') as f:
        p = {r['document']: bool(int(r['confirmed'])) for r in csv.DictReader(f) if r['budget'] == '8' and r['arm'] == 'producer_feedback'}
    c = {r['document']: r['confirmed'] for r in rows if r['budget'] == 8}
    report = {'protocol_sha256': digest(RUN / 'protocol.json'), 'documents': 276, 'endpoints': len(rows),
              'pooled': pooled, 'comparison': {'baseline': ARM, 'candidate': 'producer_feedback', **paired(c, p)},
              'multiplicity': 'One pre-specified follow-up comparison, distinct from the earlier three-test family; no within-family adjustment required.',
              'controller_replay_all_fields_equal': True,
              'assertions': 'Five arms complete; all raw outputs, 1104 endpoints, full verifier, identity replay, source/inventory and token caps checked; entire controller replay equals saved fields.'}
    atomic(OUT / 'violation_omission_audit.json', report)
    with (OUT / 'violation_omission_outcomes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--analyze', action='store_true')
    g.add_argument('--omission', action='store_true')
    g.add_argument('--worker', type=int)
    g.add_argument('--omission-worker', type=int)
    p.add_argument('--existing-only', action='store_true')
    args = p.parse_args()
    if args.worker is not None: print(json.dumps(worker(args.worker, not args.existing_only)))
    elif args.omission_worker is not None: print(json.dumps(omission_worker(args.omission_worker)))
    elif args.omission: omission()
    else: analyze(args.existing_only)
