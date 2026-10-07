"""Supplementary budget-controlled repair experiment; never changes the CV runs.

Prepare a frozen protocol before --run. The queue is single-GPU and resumes only
completed rounds. Every output sequence counts against the document budget,
including invalid JSON and proposals rejected by the verifier. No annotation-
identity score is available to the search or used to choose an arm or checkpoint.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/controlled_repair'
ARMS = ('producer_feedback', 'node_feedback', 'producer_no_feedback', 'untuned_producer')
BUDGETS = (1, 2, 4, 8)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def prepare():
    if (OUT / 'protocol.json').exists():
        validate_protocol()
        print('Existing frozen protocol verified; not overwritten.', flush=True)
        return
    paths = [Path(__file__), ROOT / 'paper/applied_intelligence/evidence/audit.json']
    paths += sorted((ROOT / 'sop2program').glob('*.py'))
    paths += [ROOT / 'models/Qwen3-14B-FP8/config.json',
              ROOT / 'models/Qwen3-14B-FP8/model.safetensors.index.json']
    for k in range(5):
        base = ROOT / f'results/cv/fold{k}'
        paths += [ROOT / f'data/cv/fold{k}/{name}' for name in
                  ('manifest.json', 'test.jsonl', 'test_workflows.json')]
        paths += [base / 'eval_qwen14' / name for name in ('predictions.jsonl', 'run_config.json')]
        paths += list((base / 'knowledge').glob('*.json'))
        paths += list((base / 'qwen14/adapter').glob('*.json'))
        paths += [base / 'qwen14/adapter/adapter_model.safetensors']
    assert all(p.exists() for p in paths)
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'design': 'Post hoc supplementary controlled experiment, fixed before these new outcomes; not preregistered.',
        'folds': list(range(5)), 'documents': 276,
        'compiler': 'Saved fold-specific fine-tuned Qwen3-14B-FP8 compiler predictions',
        'proposer': 'Same Qwen3-14B-FP8 architecture, with fold-specific 14B adapter or disabled adapter',
        'arms': {
            'producer_feedback': 'Fine-tuned proposer; nearest producing steps receive feedback; max_producers=2',
            'node_feedback': 'Same fine-tuned proposer; feedback only to first failing node; max_producers=0',
            'producer_no_feedback': 'Same fine-tuned proposer and producer-target selection; omit repair_feedback prompt block',
            'untuned_producer': 'Same producer routing and feedback; disable all LoRA adapters; no demonstrations'
        },
        'generation_caps_per_document': list(BUDGETS), 'primary_cap': 8,
        'primary_comparisons': ['producer_feedback vs node_feedback',
                                'producer_feedback vs producer_no_feedback',
                                'producer_feedback vs untuned_producer'],
        'statistics': 'Three two-sided exact paired tests at cap 8, Holm adjustment within this supplementary family; other caps descriptive; conditional on fitted models.',
        'max_rounds': 12, 'max_new_tokens_per_sequence': 384, 'seed': 42,
        'decoding': 'Greedy schema-constrained JSON, pinned given trigger, thinking disabled, SDPA',
        'retype_producers': False, 'fallback': False, 'demonstrations': 0,
        'search': 'Existing source-constrained repair operators and fold rules; same supplied inventory and anchors; no gold identity inside search',
        'cap_semantics': 'Count every generated sequence; truncate requests in deterministic existing order; stop on no improving request or full verifier pass. At a cap within a round, finalize the best proposal from its prefix. Lower caps are snapshots of the same request trajectory, not separately tuned runs.',
        'resource_reporting': 'Equal maximum sequence allowance and output-token cap, not equal realized compute; record actual sequences, prompt/output tokens and amortized generation seconds.',
        'endpoint': 'Full verifier acceptance AND annotation-derived state replay, with full acceptance and identity check reported separately.',
        'sensitivity': 'Exclude the 42 original development documents descriptively; this is not independent validation and does not undo development-informed design.',
        'limitations': 'Single corpus/seed, retrospective non-nested CV, supplied gold anchors/inventory, heuristic annotation mapping, no physical execution. Untuned proposer comparison does not ablate individual training objectives.',
        'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}
    }
    atomic(OUT / 'protocol.json', protocol)
    snapshot = ROOT / 'paper/applied_intelligence/evidence/pre_controlled_snapshot'
    snapshot.mkdir(exist_ok=True)
    import shutil
    for name in ('manuscript.tex', 'manuscript.pdf', 'cover_letter.txt', 'submission_notes_zh.txt'):
        target = snapshot / name
        if not target.exists():
            shutil.copy2(ROOT / 'paper/applied_intelligence' / name, target)
    print('Protocol frozen:', OUT / 'protocol.json', flush=True)


def validate_protocol():
    protocol = read(OUT / 'protocol.json')
    for rel, expected in protocol['source_sha256'].items():
        assert digest(ROOT / rel) == expected, f'Frozen input changed: {rel}'
    assert protocol['generation_caps_per_document'] == list(BUDGETS)
    assert tuple(protocol['arms']) == ARMS
    return digest(OUT / 'protocol.json')


def cap_rows(state, original, rules, lexicon, mentions, candidate, budgets, final=False):
    """Finalize endpoints without feeding any final-search result back into the model loop."""
    from sop2program.verify import minimal_repair, verify
    targets = [b for b in budgets if str(b) not in state['budgets'] and
               (state['calls'] == b or final and state['calls'] <= b)]
    if not targets:
        return
    if not original.metadata.get('schema_complete'):
        result = {'status': 'REVIEW', 'cost': None, 'workflow': original, 'operations': []}
    else:
        changed = candidate.model_dump_json() != original.model_dump_json()
        result = minimal_repair(original, rules, proposals=[candidate] if changed else [],
                                lexicon=lexicon, mentions=mentions, drop_arguments=True)
    w = result['workflow']
    assert w.source == original.source and w.initial == original.initial
    if result['status'] in ('PASS', 'REPAIRED'):
        assert verify(w, rules)['pass']
    row = {'status': result['status'], 'cost': result['cost'],
           'operations': result.get('operations', []), 'workflow': w.model_dump(),
           **{key: state[key] for key in ('calls', 'input_tokens', 'output_tokens', 'generation_seconds')}}
    for b in targets:
        state['budgets'][str(b)] = row


def run_arm(arm, workflows, compiler, rules, lexicon, mentions, checkpoint, protocol_hash,
            budgets=BUDGETS):
    """Batch independent requests; save all state atomically after each complete round."""
    from sop2program.ir import Workflow
    from sop2program.repair import feedback_requests, apply_proposal, replay_progress
    from sop2program.verify import settle_unavailable, verify
    checkpoint = Path(checkpoint)
    originals = {w.id: w for w in workflows}
    if checkpoint.exists():
        saved = read(checkpoint)
        assert saved['protocol_sha256'] == protocol_hash and saved['arm'] == arm
        assert set(saved['documents']) == set(originals)
        if saved['complete']:
            print(f'{arm}: already complete', flush=True)
            return saved
    else:
        saved = {'protocol_sha256': protocol_hash, 'arm': arm, 'complete': False,
                 'next_round': 0, 'events': [], 'documents': {}}
        for doc, w in originals.items():
            working = settle_unavailable(w, rules, mentions) if w.metadata.get('schema_complete') else w
            s = {'working': working.model_dump(), 'tried': [], 'calls': 0,
                 'input_tokens': 0, 'output_tokens': 0, 'generation_seconds': 0.0,
                 'budgets': {}, 'active': bool(w.metadata.get('schema_complete') and not verify(working, rules)['pass'])}
            saved['documents'][doc] = s
            if not s['active']:
                cap_rows(s, w, rules, lexicon, mentions, working, budgets, final=True)
        atomic(checkpoint, saved)
    for round_index in range(saved['next_round'], 12):
        batch = []
        for doc, s in sorted(saved['documents'].items()):
            if not s['active'] or s['calls'] >= max(budgets):
                continue
            working = Workflow.model_validate(s['working'])
            outcome, requests = feedback_requests(working, rules, 0 if arm == 'node_feedback' else 2, False)
            fresh = []
            for request in requests:
                key = json.dumps([request['index'], request['object'],
                                  working.steps[request['index']].operator.model_dump()], sort_keys=True)
                if key not in s['tried']:
                    s['tried'].append(key)
                    fresh.append(request)
                if len(fresh) >= max(budgets) - s['calls']:
                    break
            if not fresh:
                s['active'] = False
                cap_rows(s, originals[doc], rules, lexicon, mentions, working, budgets, final=True)
            else:
                batch.append((doc, working, outcome, fresh))
        if not batch:
            break
        rows = [dict(r['row']) for _, _, _, requests in batch for r in requests]
        if arm == 'producer_no_feedback':
            for row in rows:
                row.pop('repair_feedback', None)
        outputs = compiler.compile_batch(rows)
        assert len(outputs) == len(rows)
        cursor = 0
        for doc, working, outcome, requests in batch:
            s = saved['documents'][doc]
            best, best_score = None, replay_progress(outcome)
            for request in requests:
                op, info = outputs[cursor]
                prompt_row = rows[cursor]
                cursor += 1
                candidate = apply_proposal(working, request, op)
                score = replay_progress(verify(candidate, rules)) if candidate is not None else None
                selected = score is not None and score > best_score
                if selected:
                    best, best_score = candidate, score
                s['calls'] += 1
                s['input_tokens'] += info['input_tokens']
                s['output_tokens'] += info['output_tokens']
                s['generation_seconds'] += info['amortized_latency_s']
                saved['events'].append({'doc': doc, 'round': round_index, 'sequence': s['calls'],
                                        'index': request['index'], 'object': request['object'],
                                        'prompt_row': prompt_row, 'operator': op.model_dump() if op else None,
                                        'info': info, 'candidate_valid': candidate is not None,
                                        'progress': list(score) if score is not None else None, 'selected': selected})
                cap_rows(s, originals[doc], rules, lexicon, mentions, best or working, budgets)
            if best is not None:
                s['working'] = best.model_dump()
            final_working = best or working
            s['active'] = bool(best is not None and not verify(final_working, rules)['pass'] and s['calls'] < max(budgets))
            if not s['active']:
                cap_rows(s, originals[doc], rules, lexicon, mentions, final_working, budgets, final=True)
        saved['next_round'] = round_index + 1
        atomic(checkpoint, saved)
        print(f'{arm} round={round_index} sequences={len(rows)} total={len(saved["events"])} active={sum(s["active"] for s in saved["documents"].values())}', flush=True)
    for doc, s in saved['documents'].items():
        cap_rows(s, originals[doc], rules, lexicon, mentions, Workflow.model_validate(s['working']), budgets, final=True)
        assert set(s['budgets']) == set(map(str, budgets)), (doc, s['calls'])
        assert s['calls'] <= max(budgets)
        s['active'] = False
    saved['complete'] = True
    atomic(checkpoint, saved)
    return saved


def worker(fold):
    protocol_hash = validate_protocol()
    os.environ['SOP_DATA'] = f'data/cv/fold{fold}'
    os.environ['SOP_KNOWLEDGE'] = f'results/cv/fold{fold}/knowledge'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer, set_seed
    from sop2program.compiler import LocalCompiler
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows
    from sop2program.numerics import load_checked_model
    data = ROOT / os.environ['SOP_DATA']
    base = ROOT / f'results/cv/fold{fold}'
    docs = {w['id']: Workflow.model_validate(w) for w in read(data / 'test_workflows.json')}
    rows = [json.loads(l) for l in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    predictions = [json.loads(l) for l in (base / 'eval_qwen14/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(predictions)
    workflows = assemble_workflows(rows, predictions, docs)
    rules = read(base / 'knowledge/full_rules.json')
    lexicon = read(base / 'knowledge/trigger_lexicon.json')['lexicon']
    settings = read(base / 'knowledge/setting_lexicon.json')
    mentions = {key: set(settings[key]) for key in ('settings', 'products')}
    pending = [a for a in ARMS if not (OUT / f'fold{fold}/{a}.json').exists() or not read(OUT / f'fold{fold}/{a}.json')['complete']]
    if not pending:
        print(f'Fold {fold} already complete', flush=True)
        return
    set_seed(42)
    model_path = ROOT / 'models/Qwen3-14B-FP8'
    model = load_checked_model(model_path, base / 'qwen14/adapter', attention='sdpa')
    assert hasattr(model, 'disable_adapter'), 'Untuned control requires an unmerged LoRA adapter'
    compiler = LocalCompiler(model, AutoTokenizer.from_pretrained(model_path, local_files_only=True),
                             max_new_tokens=384, pin_trigger=True, demonstrations=[])
    for arm in pending:
        print(f'BEGIN fold={fold} arm={arm}', flush=True)
        with model.disable_adapter() if arm == 'untuned_producer' else nullcontext():
            run_arm(arm, workflows, compiler, rules, lexicon, mentions,
                    OUT / f'fold{fold}/{arm}.json', protocol_hash)
        print(f'END fold={fold} arm={arm}', flush=True)


def queue():
    validate_protocol()
    # OS-held byte lock is released automatically on a crash; never delete a live lock.
    import msvcrt
    OUT.mkdir(exist_ok=True)
    with (OUT / 'queue.lock').open('a+b') as lock:
        lock.seek(0)
        if lock.read(1) == b'':
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('A controlled repair queue is already running') from exc
        atomic(OUT / 'status.json', {'status': 'running', 'pid': os.getpid(),
                                   'started_utc': datetime.now(timezone.utc).isoformat()})
        for fold in range(5):
            print(f'BEGIN_FOLD {fold} {datetime.now(timezone.utc).isoformat()}', flush=True)
            result = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(Path(__file__)), '--worker', str(fold)], cwd=ROOT)
            if result.returncode:
                atomic(OUT / 'status.json', {'status': 'failed', 'fold': fold, 'returncode': result.returncode})
                raise SystemExit(result.returncode)
            print(f'END_FOLD {fold} {datetime.now(timezone.utc).isoformat()}', flush=True)
        atomic(OUT / 'status.json', {'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat()})
        print('CONTROLLED_REPAIR_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--worker', type=int, choices=range(5))
    args = p.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        queue()
    else:
        worker(args.worker)
