"""Training-seed replication of the fine-tuned 14B compiler and producer-feedback repair.

For each extra seed and fold, retrain the fold's 14B adapter with the original settings
(only --seed changes), recompile the fold's test documents, and score first pass (F),
deterministic repair (D), producer feedback (P) and the untuned producer (U) under the
controlled ceilings. Seed 42 is the saved CV run and is scored from its existing files.
Every step is resumable: a step whose completion marker exists is skipped.

    python scripts/seed_replication.py --seeds 43 44        # GPU queue
    python scripts/seed_replication.py --seeds 42 43 44 --score
"""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_cv
from controlled_repair import BUDGETS, atomic, read, run_arm

OUT = ROOT / 'results/seeds'
ARMS = ('producer_feedback', 'untuned_producer')
run_cv.LOGS = OUT / 'logs'


def base_dir(seed, fold):
    return ROOT / f'results/cv/fold{fold}' if seed == 42 else OUT / f'seed{seed}/fold{fold}'


def arm_path(seed, fold, arm):
    if seed == 42:
        return ROOT / f'results/controlled_repair/fold{fold}/{arm}.json'
    return OUT / f'seed{seed}/fold{fold}/{arm}.json'


def fingerprint():
    h = hashlib.sha256()
    for p in [Path(__file__), ROOT / 'scripts/controlled_repair.py',
              *(ROOT / f'sop2program/{m}.py' for m in ('compiler', 'constrained', 'ir', 'metrics', 'numerics', 'repair', 'verify'))]:
        h.update(p.read_bytes())
    return h.hexdigest()


def worker(seed, fold):
    os.environ['SOP_DATA'] = f'data/cv/fold{fold}'
    os.environ['SOP_KNOWLEDGE'] = f'results/cv/fold{fold}/knowledge'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer, set_seed
    from sop2program.compiler import LocalCompiler
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows
    from sop2program.numerics import load_checked_model
    data = ROOT / os.environ['SOP_DATA']
    knowledge = ROOT / os.environ['SOP_KNOWLEDGE']
    base = base_dir(seed, fold)
    docs = {w['id']: Workflow.model_validate(w) for w in read(data / 'test_workflows.json')}
    rows = [json.loads(l) for l in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    predictions = [json.loads(l) for l in (base / 'eval_qwen14/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(predictions)
    workflows = assemble_workflows(rows, predictions, docs)
    rules = read(knowledge / 'full_rules.json')
    lexicon = read(knowledge / 'trigger_lexicon.json')['lexicon']
    settings = read(knowledge / 'setting_lexicon.json')
    mentions = {key: set(settings[key]) for key in ('settings', 'products')}
    pending = [a for a in ARMS if not arm_path(seed, fold, a).exists() or not read(arm_path(seed, fold, a))['complete']]
    done = base / 'controlled.done'
    if not pending:
        done.write_text('complete\n', encoding='utf-8')
        return
    set_seed(42)
    model_path = ROOT / 'models/Qwen3-14B-FP8'
    model = load_checked_model(model_path, base / 'qwen14/adapter', attention='sdpa')
    assert hasattr(model, 'disable_adapter'), 'Untuned control requires an unmerged LoRA adapter'
    compiler = LocalCompiler(model, AutoTokenizer.from_pretrained(model_path, local_files_only=True),
                             max_new_tokens=384, pin_trigger=True, demonstrations=[])
    protocol_hash = fingerprint()
    for arm in pending:
        print(f'BEGIN seed={seed} fold={fold} arm={arm}', flush=True)
        with model.disable_adapter() if arm == 'untuned_producer' else nullcontext():
            run_arm(arm, workflows, compiler, rules, lexicon, mentions, arm_path(seed, fold, arm), protocol_hash)
        print(f'END seed={seed} fold={fold} arm={arm}', flush=True)
    done.write_text('complete\n', encoding='utf-8')


def queue(seeds):
    for seed in seeds:
        assert seed != 42, 'Seed 42 is the saved CV run'
        for k in range(5):
            base = base_dir(seed, k).relative_to(ROOT).as_posix()
            data = f'data/cv/fold{k}'
            run_cv.train(f's{seed}_f{k}_train_qwen14', f'{base}/qwen14', f'{data}/train_repair_fb.jsonl', k,
                         ['--model', run_cv.Q14, '--empty-cache-every', '1', '--seed', str(seed)])
            run_cv.step(f's{seed}_f{k}_eval_qwen14', ['scripts/evaluate.py', '--method', 'lora', '--model', run_cv.Q14,
                        '--adapter', f'{base}/qwen14/adapter', '--split', 'test', *run_cv.EVAL,
                        '--output', f'{base}/eval_qwen14'], f'{base}/eval_qwen14/efficiency.json', k)
            run_cv.step(f's{seed}_f{k}_deterministic', ['scripts/deterministic_repair.py', '--run', f'{base}/eval_qwen14',
                        '--split', 'test', '--output', f'{base}/repair_deterministic_qwen14'],
                        f'{base}/repair_deterministic_qwen14/summary.json', k)
            run_cv.step(f's{seed}_f{k}_recheck_deterministic', ['scripts/recheck_annotation_identity.py', '--repairs',
                        f'{base}/repair_deterministic_qwen14', '--arm', 'deterministic_repair',
                        '--output', f'{base}/recheck_deterministic_qwen14.json'], f'{base}/recheck_deterministic_qwen14.json', k)
            run_cv.step(f's{seed}_f{k}_controlled', [str(Path(__file__).relative_to(ROOT)), '--worker', str(seed), str(k)],
                        f'{base}/controlled.done', k)
    print('SEED_QUEUE_DONE', flush=True)


def score(seeds):
    from external_baselines import confirmed_rows
    from recheck_annotation_identity import mcnemar
    from sop2program.annotation import GoldIdentity
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows
    from sop2program.verify import verify
    records = []
    for seed in seeds:
        for fold in range(5):
            base = base_dir(seed, fold)
            data = ROOT / f'data/cv/fold{fold}'
            gold = {w['id']: Workflow.model_validate(w) for w in read(data / 'test_workflows.json')}
            rules = read(ROOT / f'results/cv/fold{fold}/knowledge/full_rules.json')
            identities = {d: GoldIdentity(w, ROOT / w.metadata['peg_file']) for d, w in gold.items()}
            rows = [json.loads(l) for l in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
            predictions = [json.loads(l) for l in (base / 'eval_qwen14/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            first = {w.id: w for w in assemble_workflows(rows, predictions, gold)}
            recheck = read(base / 'recheck_deterministic_qwen14.json')
            recheck = recheck[0] if isinstance(recheck, list) else recheck
            for d, w in first.items():
                ok = bool(w.metadata['schema_complete'] and verify(w, rules)['pass'])
                g = identities[d]
                confirmed = bool(ok and verify(g.rename(w, g.gold().initial), evidence_checks=False, invariants=False)['pass'])
                records.append({'seed': seed, 'fold': fold, 'document': d, 'arm': 'F_first_pass', 'budget': 8,
                                'accepted': ok, 'confirmed': confirmed})
                records.append({'seed': seed, 'fold': fold, 'document': d, 'arm': 'D_deterministic', 'budget': 8,
                                'accepted': None, 'confirmed': bool(recheck['chain_identity_by_document'][d][1])})
            for arm in ARMS:
                saved = read(arm_path(seed, fold, arm))
                assert saved['complete'], (seed, fold, arm)
                for cap in BUDGETS:
                    for d, r in confirmed_rows(saved, gold, rules, identities, cap).items():
                        records.append({'seed': seed, 'fold': fold, 'document': d, 'arm': arm, 'budget': cap,
                                        'accepted': r['accepted'], 'confirmed': r['confirmed']})
    summary = {'endpoint': 'Full verifier acceptance AND annotation-chain state replay', 'seeds': {}}
    for seed in seeds:
        take = [r for r in records if r['seed'] == seed]
        at8 = {}
        for r in take:
            if r['budget'] == 8:
                at8.setdefault(r['arm'], {})[r['document']] = r['confirmed']
        counts = {arm: sum(v.values()) for arm, v in at8.items()}
        counts.update({f'{arm}@{cap}': sum(r['confirmed'] for r in take if r['arm'] == arm and r['budget'] == cap)
                       for arm in ARMS for cap in BUDGETS})
        tests = {}
        for ref, other in (('producer_feedback', 'D_deterministic'), ('producer_feedback', 'untuned_producer'),
                           ('D_deterministic', 'F_first_pass')):
            gain = sum(at8[ref][d] and not at8[other][d] for d in at8[ref])
            loss = sum(at8[other][d] and not at8[ref][d] for d in at8[ref])
            tests[f'{ref} vs {other}'] = {'gains': gain, 'losses': loss, 'exact_mcnemar_p': mcnemar(gain, loss)}
        summary['seeds'][str(seed)] = {'documents': len(at8['F_first_pass']), 'confirmed_at_cap8': counts, 'paired_at_cap8': tests}
    from statistics import mean, stdev
    if len(seeds) > 1:
        summary['across_seeds'] = {arm: {'mean': mean(summary['seeds'][str(s)]['confirmed_at_cap8'][arm] for s in seeds),
                                         'sd': stdev(summary['seeds'][str(s)]['confirmed_at_cap8'][arm] for s in seeds),
                                         'values': [summary['seeds'][str(s)]['confirmed_at_cap8'][arm] for s in seeds]}
                                   for arm in ('F_first_pass', 'D_deterministic', *ARMS)}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic(OUT / 'summary.json', summary)
    with (OUT / 'document_outcomes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', nargs='+', type=int, default=[43, 44])
    g = p.add_mutually_exclusive_group()
    g.add_argument('--score', action='store_true')
    g.add_argument('--worker', nargs=2, type=int, metavar=('SEED', 'FOLD'))
    args = p.parse_args()
    if args.worker:
        worker(*args.worker)
    elif args.score:
        score(args.seeds)
    else:
        queue(args.seeds)
