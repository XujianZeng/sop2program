"""Sequential state feedback within each SOP, batched across independent SOPs."""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import os
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from transformers import AutoTokenizer, set_seed
from sop2program.compiler import LocalCompiler
from sop2program.ir import Operator, Workflow, make_step, triggers_match
from sop2program.verify import verify
from sop2program.numerics import load_checked_model
from sop2program.paths import CORPUS, DATA


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(path):
    """Only a torn final line may be discarded on resume; other corruption is fatal."""
    if not path.exists():
        return {}
    records = {}
    good_end = 0
    final_line_valid = False
    with path.open('rb') as stream:
        lines = stream.readlines()
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise
            with path.open('r+b') as stream:
                stream.truncate(good_end)
            break
        if record['id'] in records:
            raise ValueError(f"Duplicate cached prediction {record['id']}")
        records[record['id']] = record
        good_end += len(line)
        final_line_valid = i == len(lines) - 1
    if final_line_valid and not lines[-1].endswith(b'\n'):
        with path.open('ab') as stream:
            stream.write(b'\n')
    return records


def run(model, tokenizer, rows, docs, out, method, constrained=True, with_state=True,
        batch_size=8, max_new_tokens=384, pin_trigger=False, demonstrations=()):
    old = read_predictions(out / 'predictions.jsonl')
    compiler = LocalCompiler(model, tokenizer, constrained, max_new_tokens, pin_trigger=pin_trigger,
                             demonstrations=demonstrations)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['doc']].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row['step'])
    workflows = {key: Workflow(id=doc.id, source=doc.source, initial=doc.initial, steps=[],
                              metadata={'expected_steps': len(grouped[key]), 'method': method})
                 for key, doc in docs.items()}
    cursors = {key: 0 for key in grouped}
    schedule = sorted(cursors, key=lambda key: (-len(grouped[key]), key))
    predictions = {}

    def commit(row, op, record):
        workflow = workflows[row['doc']]
        if op is not None and triggers_match(op.trigger, row['trigger']):
            workflow.steps.append(make_step(op, workflow.source, row['step'], row['offset'], workflow.steps))
        predictions[row['id']] = record
        cursors[row['doc']] += 1

    with (out / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
        while any(cursors[key] < len(grouped[key]) for key in cursors):
            pending = []
            for key in schedule:
                while cursors[key] < len(grouped[key]):
                    row = grouped[key][cursors[key]]
                    if row['id'] not in old:
                        break
                    record = old[row['id']]
                    op = Operator.model_validate(record['prediction']) if record['prediction'] is not None else None
                    commit(row, op, record)
                if cursors[key] < len(grouped[key]) and len(pending) < batch_size:
                    row = grouped[key][cursors[key]]
                    pending.append(dict(row, state=verify(workflows[key], invariants=False)['final_state']))
            if not pending:
                continue
            outputs = compiler.compile_batch(pending, with_state)
            for row, (op, info) in zip(pending, outputs):
                record = {'id': row['id'], 'doc': row['doc'], 'step': row['step'],
                          'prediction': op.model_dump() if op is not None else None,
                          'gold': row['target'], 'method': method,
                          'anchor_match': op is not None and triggers_match(op.trigger, row['trigger']), **info}
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                commit(row, op, record)
            print(method, len(predictions), '/', len(rows), 'batch_s',
                  round(outputs[0][1]['latency_s'], 2), flush=True)
    for key, workflow in workflows.items():
        workflow.metadata['schema_complete'] = len(workflow.steps) == len(grouped[key])
    (out / 'workflows.json').write_text(
        json.dumps([w.model_dump() for w in workflows.values()], ensure_ascii=False), encoding='utf-8')
    return [predictions[row['id']] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['base', 'lora', 'no_state', 'unconstrained'], required=True)
    parser.add_argument('--max-docs', type=int, default=0)
    parser.add_argument('--adapter', default='results/lora3b_full/adapter')
    parser.add_argument('--output')
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--compile-cache',action='store_true')
    parser.add_argument('--attention', choices=['eager','sdpa'], default='eager')
    parser.add_argument('--model', default='models/Qwen2.5-3B-Instruct', help='Local base model directory')
    parser.add_argument('--pin-trigger', action='store_true',
                        help='Constrain the trigger field to the supplied marked span')
    parser.add_argument('--few-shot', type=int, default=0,
                        help='In-context demonstrations drawn from the training split')
    args = parser.parse_args()
    args.adapter=Path(args.adapter).as_posix()
    if args.output:args.output=Path(args.output).as_posix()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError('Batch and token limits must be positive')
    set_seed(42)
    torch.set_num_threads(6)
    data = DATA
    docs = {w['id']: Workflow.model_validate(w)
            for w in json.loads((data / f'{args.split}_workflows.json').read_text(encoding='utf-8'))}
    rows = [json.loads(line) for line in (data / f'{args.split}.jsonl').read_text(encoding='utf-8').splitlines()]
    if args.max_docs:
        keys = sorted(docs)[:args.max_docs]
        docs = {key: doc for key, doc in docs.items() if key in keys}
        rows = [row for row in rows if row['doc'] in docs]
    out = ROOT / (args.output or f'results/{args.method}')
    out.mkdir(parents=True, exist_ok=True)
    demonstrations = []
    if args.few_shot:
        from sop2program.compiler import select_demonstrations
        # Demonstrations stay fixed across folds; their documents are pinned to every fold's training part.
        train_rows = [json.loads(line) for line in (CORPUS / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
        demonstrations = select_demonstrations(train_rows, args.few_shot)
    signature = vars(args) | {'demonstration_ids': [d['id'] for d in demonstrations],'data_sha256': file_hash(data / f'{args.split}.jsonl'),
                             'workflows_sha256': file_hash(data / f'{args.split}_workflows.json'),
                             'model_manifest_sha256': file_hash(ROOT / args.model / 'download_manifest.json'),
                             'software_versions': {name: version(name) for name in ['torch','transformers','peft','lm-format-enforcer']},
                             'compiler_sha256': file_hash(ROOT / 'sop2program/compiler.py'),
                             'constrained_sha256': file_hash(ROOT / 'sop2program/constrained.py'),
                             'verifier_sha256': file_hash(ROOT / 'sop2program/verify.py'),
                             'ir_sha256': file_hash(ROOT / 'sop2program/ir.py'),
                             'evaluation_sha256': file_hash(Path(__file__)),
                             'numerics_sha256': file_hash(ROOT / 'sop2program/numerics.py'),
                             'model_loading': 'CPU safe adapter merge, checked CUDA transfer',
                             'attention_backend':args.attention,'precision':'bfloat16',
                             'matmul_reduced_precision':False,
                             'platform':platform.platform(),'python_version':platform.python_version(),
                             'cuda_launch_blocking':os.environ.get('CUDA_LAUNCH_BLOCKING','0'),
                             'adapter_sha256': file_hash(ROOT / args.adapter / 'adapter_model.safetensors')
                             if args.method != 'base' else None}
    config_path = out / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != signature:
        raise ValueError('Cached run configuration differs; use a new --output directory')
    if (out / 'predictions.jsonl').exists() and not config_path.exists():
        raise ValueError('Cannot resume an unversioned prediction cache; use a new --output')
    config_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model, local_files_only=True)
    model = load_checked_model(ROOT / args.model,
                               ROOT / args.adapter if args.method != 'base' else None,compile_cache=args.compile_cache,attention=args.attention)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        predictions = run(model, tokenizer, rows, docs, out, args.method,
                          constrained=args.method != 'unconstrained', with_state=args.method != 'no_state',
                          batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                          pin_trigger=args.pin_trigger, demonstrations=demonstrations)
    except (FloatingPointError,RuntimeError) as error:
        (out/'failure.json').write_text(json.dumps({'error':str(error),'time':time.time(),
            'saved_predictions':len(read_predictions(out/'predictions.jsonl')),'run_config':signature},
            ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    elapsed = time.time() - start
    efficiency = {'elapsed_s_this_session': elapsed, 'peak_vram_gb': torch.cuda.max_memory_allocated() / 2**30,
                  'n': len(rows), 'documents': len(docs), 'device': torch.cuda.get_device_name(),
                  'gold_mentions_provided': True, 'gold_inventory_provided': True,
                  'state_at_test': 'predicted replay; no gold intermediate states',
                  'no_state_ablation': 'inference only' if args.method == 'no_state' else None,
                  'latency_definition': 'request latency per concurrent batch; amortized latency also saved',
                  'total_generation_s': sum(p['amortized_latency_s'] for p in predictions),
                  'status': 'complete'}
    (out / 'efficiency.json').write_text(json.dumps(efficiency, indent=2), encoding='utf-8')
    print('EVALUATION_COMPLETE', args.method, flush=True)


if __name__ == '__main__':
    main()
