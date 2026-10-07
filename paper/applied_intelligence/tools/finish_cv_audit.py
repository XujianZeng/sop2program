"""Validate five completed folds and refresh CPU-replayed publication evidence.

Does not launch training, change experimental settings, or edit the manuscript.
Run from any directory with the project's Python after run_cv.py --phases B.
--check-ready performs only readiness checks (exit 2 means incomplete).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / 'paper/applied_intelligence'
COMPILERS = ('fewshot3', 'lora3b', 'qwen14')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def readiness():
    missing = []
    for k in range(5):
        base = ROOT / f'results/cv/fold{k}'
        for model in ('qwen14', 'lora3b', 'lora3b_repair'):
            p = base / model / 'config.json'
            if not p.exists() or read(p).get('status') != 'complete':
                missing.append(str(p.relative_to(ROOT)))
        for compiler in COMPILERS:
            run = base / f'eval_{compiler}'
            if not (run / 'predictions.jsonl').exists():
                missing.append(str((run / 'predictions.jsonl').relative_to(ROOT)))
            if compiler != 'fewshot3':
                p = run / 'efficiency.json'
                if not p.exists() or read(p).get('status') != 'complete':
                    missing.append(str(p.relative_to(ROOT)))
            for kind in ('deterministic', 'trained', *(['untuned'] if compiler == 'fewshot3' else [])):
                p = base / f'repair_{kind}_{compiler}/summary.json'
                if not p.exists():
                    missing.append(str(p.relative_to(ROOT)))
    return missing

def run(args, env=None):
    return subprocess.run([sys.executable, '-X', 'utf8', *args], cwd=ROOT,
                          env=env, check=True, text=True, encoding='utf-8',
                          capture_output=True)

def validate_matching_settings():
    """Reject changes to the frozen training and evaluation protocol across folds."""
    train_keys = ('epochs', 'seed', 'no_state', 'max_length', 'gradient_accumulation',
                  'batch_size', 'attention_backend', 'gradient_checkpointing',
                  'cuda_launch_blocking', 'empty_cache_every', 'cf_every', 'cf_margin',
                  'cf_weight', 'learning_rate', 'model', 'model_manifest_sha256',
                  'lora_rank', 'lora_alpha', 'precision', 'quantization', 'aux_weight',
                  'source_sha256', 'software_versions')
    eval_keys = ('method', 'max_docs', 'split', 'batch_size', 'max_new_tokens',
                 'attention', 'model', 'pin_trigger', 'few_shot',
                 'model_manifest_sha256', 'software_versions', 'compiler_sha256',
                 'constrained_sha256', 'verifier_sha256', 'ir_sha256',
                 'evaluation_sha256', 'numerics_sha256', 'attention_backend',
                 'precision', 'matmul_reduced_precision')
    for relative, keys in [(f'{m}/config.json', train_keys) for m in ('qwen14', 'lora3b', 'lora3b_repair')] + [
            (f'eval_{m}/run_config.json', eval_keys) for m in ('qwen14', 'lora3b')]:
        reference = read(ROOT / 'results/cv/fold0' / relative)
        for k in range(1, 5):
            current = read(ROOT / f'results/cv/fold{k}' / relative)
            differences = {key: [reference.get(key), current.get(key)] for key in keys
                           if reference.get(key) != current.get(key)}
            assert not differences, (k, relative, differences)

def main():
    missing = readiness()
    if '--check-ready' in sys.argv or missing:
        print(json.dumps({'ready': not missing, 'missing': missing}, indent=2))
        return 2 if missing else 0
    validate_matching_settings()

    # Preserve the reviewed four-fold manuscript and calculations before replacing evidence.
    snapshot = PAPER / 'evidence/four_fold_snapshot'
    snapshot.mkdir(exist_ok=True)
    sources = [PAPER/'manuscript.tex', PAPER/'manuscript.pdf',
               ROOT/'results/cv/report.json', ROOT/'results/cv/first_pass_consistent.json']
    sources += [PAPER/'evidence'/name for name in ('audit.json', 'replayed_details.json',
                'document_outcomes.csv', 'fold_results.csv', 'paired_comparisons.csv')]
    sources += list((PAPER/'figures').glob('Fig[123].*'))
    for source in sources:
        dest = snapshot / source.name
        if source.exists() and not dest.exists():
            shutil.copy2(source, dest)

    # Generate per-fold repaired-program chain rechecks. Its legacy pooled first-pass
    # column is state-only; the publication must use audit.json produced below.
    print(run(['scripts/cv_report.py']).stdout, flush=True)
    first = {c: {} for c in COMPILERS}
    for k in range(5):
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                   SOP_DATA=f'data/cv/fold{k}', SOP_KNOWLEDGE=f'results/cv/fold{k}/knowledge')
        for c in COMPILERS:
            out = run(['scripts/cv_first_pass_consistent.py',
                       f'results/cv/fold{k}/eval_{c}',
                       f'results/cv/fold{k}/recheck_deterministic_{c}.json'], env=env)
            outcomes = json.loads(out.stdout)
            assert not (set(first[c]) & set(outcomes)), (k, c, 'duplicate test document')
            first[c].update(outcomes)
    assert all(len(v) == 276 for v in first.values())
    (ROOT/'results/cv/first_pass_consistent.json').write_text(json.dumps(first, indent=2), encoding='utf-8')
    print(run(['paper/applied_intelligence/tools/audit_submission.py']).stdout, flush=True)
    audit = read(PAPER/'evidence/audit.json')
    assert audit['completed_folds'] == list(range(5))
    assert audit['documents'] == 276 and audit['operators'] == 3869
    print('FIVE_FOLD_AUDIT_COMPLETE: 276 protocols, 3869 operators, 35 repair arms.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
