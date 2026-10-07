"""Frozen follow-up: candidate-preserving violation omission; original runs stay intact."""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.controlled_repair import read, atomic, digest, run_arm, BUDGETS

OUT = ROOT / 'results/robustness_extension'
ARM = 'producer_candidate_only'


class CandidateOnlyCompiler:
    """Change exactly one prompt field; keep candidate, routing and controller intact."""
    def __init__(self, compiler):
        self.compiler = compiler

    def compile_batch(self, rows):
        for row in rows:
            before = copy.deepcopy(row)
            block = copy.deepcopy(row['repair_feedback'])
            assert set(block) == {'candidate', 'violations'} and block['violations']
            del block['violations']
            row['repair_feedback'] = block
            expected = copy.deepcopy(before)
            del expected['repair_feedback']['violations']
            assert row == expected and row['repair_feedback']['candidate'] == before['repair_feedback']['candidate']
        # Mutate only the independent prompt row, so saved events contain the actual prompt.
        return self.compiler.compile_batch(rows)


def prepare():
    if (OUT / 'protocol.json').exists():
        validate()
        print('Existing extension protocol verified; not overwritten.')
        return
    from scripts.controlled_repair import validate_protocol
    validate_protocol()
    old = read(ROOT / 'results/controlled_repair/protocol.json')
    paths = {ROOT / p for p in old['source_sha256']}
    paths |= {ROOT / p for p in read(ROOT / 'paper/applied_intelligence/evidence/audit.json')['source_sha256']}
    paths |= {Path(__file__), ROOT / 'paper/applied_intelligence/tools/audit_robustness.py',
              ROOT / 'tests/test_robustness_extension.py', ROOT / 'results/controlled_repair/protocol.json',
              ROOT / 'paper/applied_intelligence/evidence/controlled_audit.json',
              ROOT / 'paper/applied_intelligence/evidence/controlled_document_outcomes.csv',
              ROOT / 'paper/applied_intelligence/tools/audit_submission.py',
              ROOT / 'scripts/recheck_annotation_identity.py'}
    paths |= set((ROOT / 'data/processed').glob('*workflows.json'))
    paths |= set((ROOT / 'data/raw/xwlp/data').rglob('*.peg'))
    paths |= set((ROOT / 'results/controlled_repair').glob('fold*/*.json'))
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'design': 'Post hoc follow-up specified before new generations and sensitivity outcomes; not preregistered.',
        'documents': 276, 'folds': list(range(5)), 'new_arm': ARM,
        'intervention': 'Delete ONLY repair_feedback.violations; retain candidate operator, shared instruction, source, state, target selection and verifier-guided proposal admission.',
        'comparison': 'Archived producer_feedback vs new producer_candidate_only, same fitted models and frozen controller; no original GPU experiment rerun.',
        'generation_caps': list(BUDGETS), 'primary_cap': 8, 'seed': 42,
        'decoding': old['decoding'], 'max_new_tokens': 384, 'fallback': False,
        'statistics': 'One new two-sided exact paired test at cap8, separate one-test family; all lower caps and scoring/subgroup analyses descriptive. No selection by outcome.',
        'scoring_modes': {
            'published': 'Original GoldIdentity.rename(charitable=True), unchanged gold-chain inventory.',
            'state_independent': 'Same resolver and inventory but charitable=False: remove preference for predicted-live chains; all other choices unchanged.',
            'strict_unique': 'Unique exact same-step chain, otherwise unique exact document chain only; no span overlap, ambiguous tie-break or predicted-state preference. Mark workflow unassessable if any resolved physical reference lacks a unique match. Compare systems also on a common fully assessable document subset per compiler family and across all five controlled arms.'},
        'coverage': 'Unique (step index, normalized physical reference) resolution calls; disclose matched references and fully assessable documents. Strict unresolved cases are unassessable, not evidence of semantic failure.',
        'strata': {'operators': [10, 20], 'annotated_entities': [10, 20],
                   'max_reference_dependency_distance': [1, 5], 'same_name_distinct_chain_names': [0, 2]},
        'error_analysis': 'All folds; automatic first-failure producer diagnosis on accepted but unconfirmed outputs; all P-only/control-only documents. No human labels or prevalence claim beyond this corpus.',
        'limits': 'One corpus/seed, retrospective CV, supplied anchors/inventory, heuristic replay. Omission retains verifier information in target selection and proposal admission; not a feedback-free system. No independent human or physical validation.',
        'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}
    }
    atomic(OUT / 'protocol.json', protocol)
    import shutil
    snap = ROOT / 'paper/applied_intelligence/evidence/pre_robustness_snapshot'
    snap.mkdir(exist_ok=True)
    for name in ('manuscript.tex', 'manuscript.pdf', 'cover_letter.txt', 'submission_notes_zh.txt', 'evidence/quality_check.json'):
        dest = snap / Path(name).name
        if not dest.exists():
            shutil.copy2(ROOT / 'paper/applied_intelligence' / name, dest)
    print('Extension protocol frozen', flush=True)


def validate():
    p = read(OUT / 'protocol.json')
    for rel, expected in p['source_sha256'].items():
        assert digest(ROOT / rel) == expected, f'Frozen input changed: {rel}'
    return digest(OUT / 'protocol.json')


def load_fold(fold):
    from sop2program.ir import Workflow
    from sop2program.metrics import assemble_workflows
    data = ROOT / f'data/cv/fold{fold}'
    base = ROOT / f'results/cv/fold{fold}'
    docs = {w['id']: Workflow.model_validate(w) for w in read(data / 'test_workflows.json')}
    import json
    rows = [json.loads(l) for l in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    predictions = [json.loads(l) for l in (base / 'eval_qwen14/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    workflows = assemble_workflows(rows, predictions, docs)
    rules = read(base / 'knowledge/full_rules.json')
    lexicon = read(base / 'knowledge/trigger_lexicon.json')['lexicon']
    settings = read(base / 'knowledge/setting_lexicon.json')
    mentions = {k: set(settings[k]) for k in ('settings', 'products')}
    return docs, workflows, rules, lexicon, mentions


def worker(fold):
    fingerprint = validate()
    os.environ['SOP_DATA'] = f'data/cv/fold{fold}'
    os.environ['SOP_KNOWLEDGE'] = f'results/cv/fold{fold}/knowledge'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    path = OUT / f'fold{fold}/{ARM}.json'
    if path.exists() and read(path)['complete']:
        print(f'Fold {fold} already complete', flush=True)
        return
    from transformers import AutoTokenizer, set_seed
    from sop2program.compiler import LocalCompiler
    from sop2program.numerics import load_checked_model
    _, workflows, rules, lexicon, mentions = load_fold(fold)
    set_seed(42)
    model_path = ROOT / 'models/Qwen3-14B-FP8'
    model = load_checked_model(model_path, ROOT / f'results/cv/fold{fold}/qwen14/adapter', attention='sdpa')
    compiler = LocalCompiler(model, AutoTokenizer.from_pretrained(model_path, local_files_only=True),
                             max_new_tokens=384, pin_trigger=True, demonstrations=[])
    run_arm(ARM, workflows, CandidateOnlyCompiler(compiler), rules, lexicon, mentions, path, fingerprint)


def queue():
    validate()
    import msvcrt
    # Share the ORIGINAL queue's OS lock; this also prevents accidental concurrent old queue use.
    with (ROOT / 'results/controlled_repair/queue.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        atomic(OUT / 'status.json', {'status': 'running', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
        for fold in range(5):
            print(f'BEGIN_FOLD {fold}', flush=True)
            r = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(Path(__file__)), '--worker', str(fold)], cwd=ROOT)
            if r.returncode:
                atomic(OUT / 'status.json', {'status': 'failed', 'fold': fold, 'returncode': r.returncode})
                raise SystemExit(r.returncode)
            print(f'END_FOLD {fold}', flush=True)
        atomic(OUT / 'status.json', {'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat()})
        print('ROBUSTNESS_EXTENSION_GPU_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--worker', type=int, choices=range(5))
    a = p.parse_args()
    if a.prepare: prepare()
    elif a.run: queue()
    else: worker(a.worker)
