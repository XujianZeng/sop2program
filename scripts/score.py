"""Score saved predictions without loading a GPU model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sop2program.ir import Workflow
from sop2program.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--rules', default='results/symbolic/full_rules.json')
    parser.add_argument('--lexicon', default='results/symbolic/trigger_lexicon.json')
    parser.add_argument('--no-action-correction', action='store_true')
    args = parser.parse_args()
    rules = json.loads((ROOT / args.rules).read_text(encoding='utf-8'))
    lexicon = None if args.no_action_correction else json.loads(
        (ROOT / args.lexicon).read_text(encoding='utf-8'))['lexicon']
    summaries = {}
    for run in args.runs:
        directory = ROOT / run
        config = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
        split = config['split']
        docs = {w['id']: Workflow.model_validate(w) for w in json.loads(
            (ROOT / f'data/processed/{split}_workflows.json').read_text(encoding='utf-8'))}
        if config['max_docs']:
            keys = sorted(docs)[:config['max_docs']]
            docs = {key: docs[key] for key in keys}
        rows = [r for r in map(json.loads, (ROOT / f'data/processed/{split}.jsonl').read_text(encoding='utf-8').splitlines()) if r['doc'] in docs]
        predictions = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        summary, details = evaluate_predictions(rows, predictions, docs, rules, lexicon)
        summary['run'] = run
        summary['evaluation_complete'] = len(predictions) == len(rows)
        summary['rules_sha256'] = hashlib.sha256((ROOT / args.rules).read_bytes()).hexdigest()
        summary['lexicon_sha256'] = (hashlib.sha256((ROOT / args.lexicon).read_bytes()).hexdigest()
                                     if lexicon is not None else None)
        summary['metrics_code_sha256'] = hashlib.sha256((ROOT / 'sop2program/metrics.py').read_bytes()).hexdigest()
        summary['verifier_code_sha256'] = hashlib.sha256((ROOT / 'sop2program/verify.py').read_bytes()).hexdigest()
        (directory / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (directory / 'audit.json').write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding='utf-8')
        summaries[run] = summary
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
