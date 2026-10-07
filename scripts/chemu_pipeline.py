"""ChEMU external-corpus run with the X-WLP design frozen: same hyperparameters, prompts,
search, ceilings and endpoint. Prepare data first with scripts/prepare_chemu.py.

    python scripts/chemu_pipeline.py --queue                         # GPU: train, compile, D, P, U
    python scripts/chemu_pipeline.py --external deepseek-flash --arms critic clairify routing
    python scripts/chemu_pipeline.py --score
"""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['SOP_DATA'] = 'data/chemu'
os.environ['SOP_KNOWLEDGE'] = 'results/chemu/knowledge'
import run_cv
from controlled_repair import BUDGETS, atomic, read, run_arm

DATA = ROOT / 'data/chemu'
OUT = ROOT / 'results/chemu'
KNOWLEDGE = OUT / 'knowledge'
ARMS = ('producer_feedback', 'untuned_producer')
run_cv.LOGS = OUT / 'logs'


def fingerprint(*extra):
    h = hashlib.sha256()
    for p in [Path(__file__), ROOT / 'scripts/controlled_repair.py', ROOT / 'scripts/external_baselines.py',
              *(ROOT / f'sop2program/{m}.py' for m in ('chemu', 'compiler', 'constrained', 'ir', 'llm', 'metrics', 'repair', 'verify'))]:
        h.update(p.read_bytes())
    h.update(json.dumps(extra).encode())
    return h.hexdigest()


def load():
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows
    docs = {w['id']: Workflow.model_validate(w) for w in read(DATA / 'test_workflows.json')}
    rows = [json.loads(l) for l in (DATA / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    predictions = [json.loads(l) for l in (OUT / 'eval_qwen14/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(predictions)
    workflows = assemble_workflows(rows, predictions, docs)
    rules = read(KNOWLEDGE / 'full_rules.json')
    lexicon = read(KNOWLEDGE / 'trigger_lexicon.json')['lexicon']
    settings = read(KNOWLEDGE / 'setting_lexicon.json')
    mentions = {k: set(settings[k]) for k in ('settings', 'products')}
    return docs, workflows, rules, lexicon, mentions


def worker():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer, set_seed
    from sop2program.compiler import LocalCompiler
    from sop2program.numerics import load_checked_model
    _, workflows, rules, lexicon, mentions = load()
    pending = [a for a in ARMS if not (OUT / f'{a}.json').exists() or not read(OUT / f'{a}.json')['complete']]
    if pending:
        set_seed(42)
        model_path = ROOT / 'models/Qwen3-14B-FP8'
        model = load_checked_model(model_path, OUT / 'qwen14/adapter', attention='sdpa')
        assert hasattr(model, 'disable_adapter')
        compiler = LocalCompiler(model, AutoTokenizer.from_pretrained(model_path, local_files_only=True),
                                 max_new_tokens=384, pin_trigger=True, demonstrations=[])
        for arm in pending:
            print(f'BEGIN chemu arm={arm}', flush=True)
            with model.disable_adapter() if arm == 'untuned_producer' else nullcontext():
                run_arm(arm, workflows, compiler, rules, lexicon, mentions, OUT / f'{arm}.json', fingerprint(arm))
            print(f'END chemu arm={arm}', flush=True)
    (OUT / 'controlled.done').write_text('complete\n', encoding='utf-8')


def queue():
    base = OUT.relative_to(ROOT).as_posix()
    run_cv.train('chemu_train_qwen14', f'{base}/qwen14', 'data/chemu/train_repair_fb.jsonl', None,
                 ['--model', run_cv.Q14, '--empty-cache-every', '1'])
    run_cv.step('chemu_eval_qwen14', ['scripts/evaluate.py', '--method', 'lora', '--model', run_cv.Q14,
                '--adapter', f'{base}/qwen14/adapter', '--split', 'test', *run_cv.EVAL, '--output', f'{base}/eval_qwen14'],
                f'{base}/eval_qwen14/efficiency.json')
    run_cv.step('chemu_deterministic', ['scripts/deterministic_repair.py', '--run', f'{base}/eval_qwen14', '--split', 'test',
                '--output', f'{base}/repair_deterministic_qwen14'], f'{base}/repair_deterministic_qwen14/summary.json')
    run_cv.step('chemu_controlled', [str(Path(__file__).relative_to(ROOT)), '--worker'], f'{base}/controlled.done')
    print('CHEMU_QUEUE_DONE', flush=True)


def external(backend, arms):
    from external_baselines import make_chat, run_baseline
    from sop2program.llm import ChatOperatorCompiler
    _, workflows, rules, lexicon, mentions = load()
    chat = make_chat(backend)
    for arm in arms:
        path = OUT / f'external/{backend}/{arm}.json'
        print(f'BEGIN chemu backend={backend} arm={arm} documents={len(workflows)}', flush=True)
        if arm == 'routing':
            run_arm('producer_feedback', workflows, ChatOperatorCompiler(chat), rules, lexicon, mentions,
                    path, fingerprint(backend, arm))
        else:
            run_baseline(arm, workflows, chat, rules, lexicon, mentions, path, fingerprint(backend, arm))
        print(f'END chemu backend={backend} arm={arm}', flush=True)


def score():
    from external_baselines import confirmed_rows
    from recheck_annotation_identity import mcnemar
    from sop2program.chemu import ChemuIdentity
    from sop2program.verify import verify
    gold, first, rules, _, _ = load()
    first = {w.id: w for w in first}
    identity_file = read(DATA / 'test_identity.json')
    identities = {d: ChemuIdentity(w, identity_file[d]) for d, w in gold.items()}
    from sop2program.ir import Workflow
    repaired = {d: Workflow.model_validate(w) for d, w in
                read(OUT / 'repair_deterministic_qwen14/deterministic_repair_workflows.json').items()}
    status = {r['doc']: r['status'] for r in read(OUT / 'repair_deterministic_qwen14/summary.json')['deterministic_repair']['details']}
    records = []

    def confirm(w, accepted, d):
        g = identities[d]
        return bool(accepted and verify(g.rename(w, g.gold().initial), evidence_checks=False, invariants=False)['pass'])
    for d, w in first.items():
        ok = bool(w.metadata['schema_complete'] and verify(w, rules)['pass'])
        records.append({'arm': 'F_first_pass', 'budget': 8, 'document': d, 'accepted': ok, 'confirmed': confirm(w, ok, d),
                        'calls': 0, 'input_tokens': 0, 'output_tokens': 0})
        ok = status[d] in ('PASS', 'REPAIRED')
        records.append({'arm': 'D_deterministic', 'budget': 8, 'document': d, 'accepted': ok,
                        'confirmed': confirm(repaired.get(d, w), ok, d), 'calls': 0, 'input_tokens': 0, 'output_tokens': 0})
    sources = {f'{a}': OUT / f'{a}.json' for a in ARMS}
    for backend_dir in sorted((OUT / 'external').glob('*')) if (OUT / 'external').exists() else []:
        for path in sorted(backend_dir.glob('*.json')):
            sources[f'{backend_dir.name}:{path.stem}'] = path
    for name, path in sources.items():
        if not path.exists() or not read(path)['complete']:
            continue
        saved = read(path)
        for cap in BUDGETS:
            for d, r in confirmed_rows(saved, gold, rules, identities, cap).items():
                records.append({'arm': name, 'budget': cap, 'document': d, 'accepted': r['accepted'], 'confirmed': r['confirmed'],
                                'calls': r['calls'], 'input_tokens': r['input_tokens'], 'output_tokens': r['output_tokens']})
    names = sorted({r['arm'] for r in records})
    pooled = []
    for name in names:
        for cap in sorted({r['budget'] for r in records if r['arm'] == name}):
            take = [r for r in records if r['arm'] == name and r['budget'] == cap]
            pooled.append({'arm': name, 'budget': cap, 'documents': len(take), 'accepted': sum(r['accepted'] for r in take),
                           'confirmed': sum(r['confirmed'] for r in take),
                           **{k: sum(r[k] for r in take) for k in ('calls', 'input_tokens', 'output_tokens')}})
    at8 = {n: {r['document']: r['confirmed'] for r in records if r['arm'] == n and r['budget'] == 8} for n in names}
    comparisons = []
    for ref in ('producer_feedback', 'D_deterministic'):
        if ref not in at8:
            continue
        for n in names:
            if n != ref:
                b = sum(at8[ref][d] and not at8[n][d] for d in at8[ref])
                c = sum(at8[n][d] and not at8[ref][d] for d in at8[ref])
                comparisons.append({'reference': ref, 'arm': n, 'reference_only': b, 'arm_only': c, 'exact_mcnemar_p': mcnemar(b, c)})
    atomic(OUT / 'summary.json', {'endpoint': 'Full verifier acceptance AND annotation-chain state replay',
                                  'documents': len(gold), 'pooled': pooled, 'comparisons_at_cap8': comparisons})
    with (OUT / 'document_outcomes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for r in pooled:
        if r['budget'] == 8:
            print(f"{r['arm']:32s} confirmed {r['confirmed']}/{r['documents']} accepted {r['accepted']} seq {r['calls']}")
    for c in comparisons:
        print(c)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--queue', action='store_true')
    g.add_argument('--worker', action='store_true')
    g.add_argument('--external')
    g.add_argument('--score', action='store_true')
    p.add_argument('--arms', nargs='+', default=['critic', 'clairify', 'routing'])
    args = p.parse_args()
    if args.queue:
        queue()
    elif args.worker:
        worker()
    elif args.external:
        external(args.external, args.arms)
    else:
        score()
