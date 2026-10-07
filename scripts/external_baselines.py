"""External repair baselines on the controlled-study inputs; earlier runs stay untouched.

All arms start from the same saved fold-specific fine-tuned 14B compiler outputs as
the controlled producer-feedback arm (P), share its per-document allowance of 1/2/4/8
generated sequences (every sequence counts, including unparsable ones) and end with
the same source-constrained deterministic search. The LLM only chooses actions and
arguments; effects and dependencies are re-projected exactly as for P.

  self_refine  Self-Refine: the LLM critiques the whole program, then rewrites it from
               its own feedback. No verifier inside the loop; stops when it reports done.
  critic       CRITIC-style: the verifier is the tool. Its errors at the first failing step
               go back to the LLM, which revises that step; revisions are accepted as is.
  clairify     CLAIRify-style: all verifier errors go back with the whole program, which
               the LLM regenerates; loop until the verifier passes.
  routing      Our producer routing and progress-gated admission with the external LLM
               as an untrained proposer (the controlled controller, unchanged).

Decoding is greedy (temperature 0), so a parsed reply that changes no step ends that
document's loop: the next prompt would be identical.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.controlled_repair import read, atomic, cap_rows, run_arm, BUDGETS

ARMS = ('self_refine', 'critic', 'clairify', 'routing')
BACKENDS = {'deepseek-flash': ('deepseek', 'deepseek-flash'),
            'deepseek-v4-pro': ('deepseek', 'deepseek-v4-pro'),
            'qwen14-base': ('local', 'models/Qwen3-14B-FP8')}
MAX_TOKENS = {'program': 4096, 'step': 512, 'feedback': 3072}
MAX_ERRORS = 25

GUIDE = '''A laboratory protocol has been compiled into a program with one step per marked action. Each step has an id, the marked trigger (fixed), an action and typed arguments.
Actions: CREATE TRANSFER DESTROY CONVERT TEMP_TREAT SPIN MEASURE WASH SEAL REMOVE WAIT MIX OTHER.
Argument roles: a = operated input; b or c = an explicit product of SPIN or CONVERT (for example supernatant or pellet); site = destination; setting = time, temperature or speed; usage = tool or instrument.
Argument types: material container device seal setting measurement modifier method unknown. Physical types are material, container, device and seal.
Copy argument text verbatim from the protocol and never invent objects. Preconditions and effects are derived from the action and arguments: every physical argument except b/c products of SPIN or CONVERT must already exist, either in the initial inventory or produced by an earlier step; CREATE creates its physical arguments; SPIN and CONVERT produce their b/c arguments; DESTROY removes its a argument.'''

PROGRAM_FORMAT = 'Return JSON {"steps":[{"id":"s0","action":"...","arguments":[{"role":"...","text":"...","type":"..."}]}, ...]} with every step, in order, keeping each id.'


def schemas():
    from sop2program.ir import ACTIONS, Argument
    argument = Argument.model_json_schema()
    step = {'type': 'object', 'properties': {'id': {'type': 'string'}, 'action': {'type': 'string', 'enum': ACTIONS},
                                             'arguments': {'type': 'array', 'items': argument, 'maxItems': 12}},
            'required': ['id', 'action', 'arguments']}
    return {'program': {'type': 'object', 'properties': {'steps': {'type': 'array', 'items': step, 'maxItems': 64}},
                        'required': ['steps']},
            'step': step,
            'feedback': {'type': 'object', 'properties': {
                'issues': {'type': 'array', 'maxItems': 10, 'items': {'type': 'object', 'properties': {
                    'step': {'type': 'string'}, 'problem': {'type': 'string', 'maxLength': 300}},
                    'required': ['step', 'problem']}},
                'done': {'type': 'boolean'}}, 'required': ['issues', 'done']}}


def program_view(workflow):
    return [{'id': s.id, 'trigger': s.operator.trigger, 'action': s.operator.action,
             'arguments': [a.model_dump() for a in s.operator.arguments]} for s in workflow.steps]


def errors_view(violations):
    return [{k: v.get(k) for k in ('step', 'code', 'field', 'required', 'actual')} for v in violations[:MAX_ERRORS]]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def context(workflow):
    return (f'Protocol:\n{workflow.source}\n\nInitial inventory: {compact(workflow.initial)}\n\n'
            f'Current program:\n{compact(program_view(workflow))}')


def request(kind, user):
    return {'kind': kind, 'messages': [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': user}],
            'schema': schemas()[kind], 'max_tokens': MAX_TOKENS[kind]}


def clairify_request(workflow, outcome):
    return request('program', context(workflow) + '\n\nA verifier rejected the program with these errors:\n'
                   + compact(errors_view(outcome['violations']))
                   + '\n\nRewrite the program so that it satisfies the verifier. ' + PROGRAM_FORMAT)


def critic_request(workflow, outcome, step_id):
    failures = [v for v in outcome['violations'] if v['step'] == step_id]
    return request('step', context(workflow) + f'\n\nA verifier reports these errors at step {step_id}:\n'
                   + compact(errors_view(failures))
                   + f'\n\nRevise step {step_id} so that the program satisfies the verifier. '
                     f'Return JSON {{"id":"{step_id}","action":"...","arguments":[{{"role":"...","text":"...","type":"..."}}]}}.')


def feedback_request(workflow):
    return request('feedback', context(workflow) + '\n\nReview the program step by step against the protocol. Look for '
                   'wrong action types, arguments not copied verbatim, wrong roles or types, objects used before they '
                   'exist, products that a later step uses but no step produces, and arguments that refer to the wrong '
                   'object. Report at most 10 issues, most important first, each in one sentence of under 30 words. '
                   'Return JSON {"issues":[{"step":"sN","problem":"..."}],"done":false}; '
                   'if the program is correct return {"issues":[],"done":true}.')


def refine_request(workflow, issues):
    return request('program', context(workflow) + '\n\nFeedback on the program:\n' + compact(issues)
                   + '\n\nRewrite the program to address the feedback. ' + PROGRAM_FORMAT)


def parse_steps(raw, kind):
    """Valid step edits by id; malformed items are ignored, never guessed."""
    from sop2program.ir import ACTIONS, Argument
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    items = payload.get('steps') if kind == 'program' and isinstance(payload, dict) else [payload]
    if not isinstance(items, list):
        return None
    edits = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or item.get('action') not in ACTIONS:
            continue
        try:
            arguments = [Argument.model_validate(a) for a in item.get('arguments') or []][:12]
        except Exception:
            continue
        edits.setdefault(item['id'], (item['action'], arguments))
    return edits


def apply_edits(workflow, edits, only=None):
    """Rebuild edited steps with template effects; triggers, ids, source and inventory are fixed."""
    from sop2program.ir import Operator, make_step, semantic_template
    from sop2program.repair import _relink_dependencies
    candidate = workflow.model_copy(deep=True)
    changed = []
    for i, step in enumerate(candidate.steps):
        if step.id not in edits or only is not None and step.id != only:
            continue
        action, arguments = edits[step.id]
        op = Operator(action=action, trigger=step.operator.trigger, arguments=arguments,
                      **semantic_template(action, arguments))
        if op == step.operator:
            continue
        rebuilt = make_step(op, workflow.source, i, max(step.evidence['trigger'].start, 0), candidate.steps[:i])
        rebuilt.id = step.id
        candidate.steps[i] = rebuilt
        changed.append(step.id)
    _relink_dependencies(candidate)
    return candidate, changed


def fingerprint(backend, arm, max_docs):
    h = hashlib.sha256()
    for path in (Path(__file__), ROOT / 'sop2program/llm.py', ROOT / 'sop2program/repair.py',
                 ROOT / 'sop2program/verify.py', ROOT / 'sop2program/compiler.py', ROOT / 'scripts/controlled_repair.py'):
        h.update(path.read_bytes())
    h.update(compact([backend, arm, max_docs, list(BUDGETS)]).encode())
    return h.hexdigest()


def run_baseline(arm, workflows, chat, rules, lexicon, mentions, checkpoint, protocol_hash, budgets=BUDGETS):
    """One generated sequence per active document per round; state saved after each round."""
    from sop2program.ir import Workflow
    from sop2program.verify import settle_unavailable, verify
    checkpoint = Path(checkpoint)
    originals = {w.id: w for w in workflows}
    if checkpoint.exists():
        saved = read(checkpoint)
        assert saved['protocol_sha256'] == protocol_hash and saved['arm'] == arm
        assert set(saved['documents']) == set(originals)
        if saved['complete']:
            return saved
    else:
        saved = {'protocol_sha256': protocol_hash, 'arm': arm, 'backend': chat.name, 'complete': False,
                 'next_round': 0, 'events': [], 'documents': {}}
        for doc, w in originals.items():
            working = settle_unavailable(w, rules, mentions) if w.metadata.get('schema_complete') else w
            s = {'working': working.model_dump(), 'calls': 0, 'input_tokens': 0, 'output_tokens': 0,
                 'cache_hit_tokens': 0, 'generation_seconds': 0.0, 'budgets': {}, 'phase': 'feedback', 'issues': None,
                 'active': bool(w.metadata.get('schema_complete') and not verify(working, rules)['pass'])}
            saved['documents'][doc] = s
            if not s['active']:
                cap_rows(s, w, rules, lexicon, mentions, working, budgets, final=True)
        atomic(checkpoint, saved)
    for round_index in range(saved['next_round'], max(budgets)):
        batch = []
        for doc, s in sorted(saved['documents'].items()):
            if not s['active']:
                continue
            working = Workflow.model_validate(s['working'])
            if arm == 'self_refine':
                req = feedback_request(working) if s['phase'] == 'feedback' else refine_request(working, s['issues'])
                batch.append((doc, req, None))
                continue
            outcome = verify(working, rules)
            target = next((v['step'] for v in outcome['violations'] if v['step'] is not None), None)
            if outcome['pass'] or target is None:
                s['active'] = False
                cap_rows(s, originals[doc], rules, lexicon, mentions, working, budgets, final=True)
            elif arm == 'clairify':
                batch.append((doc, clairify_request(working, outcome), None))
            else:
                batch.append((doc, critic_request(working, outcome, target), target))
        if not batch:
            break
        outputs = chat.chat_batch([req for _, req, _ in batch])
        assert len(outputs) == len(batch)
        for (doc, req, target), (raw, info) in zip(batch, outputs):
            s = saved['documents'][doc]
            working = Workflow.model_validate(s['working'])
            s['calls'] += 1
            for key in ('input_tokens', 'output_tokens', 'cache_hit_tokens'):
                s[key] += info.get(key, 0)
            s['generation_seconds'] += info['amortized_latency_s']
            event = {'doc': doc, 'round': round_index, 'sequence': s['calls'], 'kind': req['kind'],
                     'target': target, 'prompt': req['messages'][-1]['content'], 'raw': raw, 'info': info}
            if req['kind'] == 'feedback':
                try:
                    payload = json.loads(raw)
                    issues = [i for i in payload.get('issues', []) if isinstance(i, dict)]
                    done = bool(payload.get('done')) or not issues
                except (ValueError, AttributeError):
                    issues, done = None, False
                event.update(parsed=issues is not None, done=done, issues=issues)
                if done:
                    s['active'] = False
                elif issues is not None:
                    s['issues'], s['phase'] = issues, 'refine'
            else:
                edits = parse_steps(raw, req['kind'])
                if req['kind'] == 'step' and edits:
                    # The request names one step; a reply that mislabels its id still answers it.
                    edits = {target: next(iter(edits.values()))}
                changed = []
                if edits:
                    working, changed = apply_edits(working, edits, only=target)
                    s['working'] = working.model_dump()
                if arm == 'self_refine':
                    s['phase'], s['issues'] = 'feedback', None
                # A parsed reply that changes nothing makes the next prompt identical; greedy decoding
                # would only repeat it, so the loop ends instead of spending the remaining allowance.
                if edits is not None and not changed:
                    s['active'] = False
                event.update(parsed=edits is not None, changed_steps=changed,
                             verifier_pass_after=verify(working, rules)['pass'])
            saved['events'].append(event)
            cap_rows(s, originals[doc], rules, lexicon, mentions, working, budgets)
            if s['calls'] >= max(budgets):
                s['active'] = False
            if not s['active']:
                cap_rows(s, originals[doc], rules, lexicon, mentions, working, budgets, final=True)
        saved['next_round'] = round_index + 1
        atomic(checkpoint, saved)
        print(f'{arm} round={round_index} sequences={len(batch)} active={sum(s["active"] for s in saved["documents"].values())}', flush=True)
    for doc, s in saved['documents'].items():
        cap_rows(s, originals[doc], rules, lexicon, mentions, Workflow.model_validate(s['working']), budgets, final=True)
        assert set(s['budgets']) == set(map(str, budgets)) and s['calls'] <= max(budgets), doc
        s['active'] = False
    saved['complete'] = True
    atomic(checkpoint, saved)
    return saved


def make_chat(backend):
    kind, model = BACKENDS[backend]
    if kind == 'deepseek':
        from sop2program.llm import DeepSeekChat
        return DeepSeekChat(model)
    from transformers import AutoTokenizer, set_seed
    from sop2program.llm import LocalChat
    from sop2program.numerics import load_checked_model
    set_seed(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    model_path = ROOT / model
    llm = load_checked_model(model_path, None, attention='sdpa')
    return LocalChat(llm, AutoTokenizer.from_pretrained(model_path, local_files_only=True), f'local:{model}')


def out_dir(backend, max_docs):
    return ROOT / ('results/external_baselines_pilot' if max_docs else 'results/external_baselines') / backend


def run(backend, arms, folds, max_docs):
    from scripts.refinement_extension import load_fold
    from sop2program.llm import ChatOperatorCompiler
    chat = make_chat(backend)
    for fold in folds:
        os.environ['SOP_DATA'] = f'data/cv/fold{fold}'
        os.environ['SOP_KNOWLEDGE'] = f'results/cv/fold{fold}/knowledge'
        _, workflows, rules, lexicon, mentions = load_fold(fold)
        if max_docs:
            workflows = sorted(workflows, key=lambda w: w.id)[:max_docs]
        for arm in arms:
            path = out_dir(backend, max_docs) / f'fold{fold}/{arm}.json'
            protocol = fingerprint(backend, arm, max_docs)
            print(f'BEGIN backend={backend} fold={fold} arm={arm} documents={len(workflows)}', flush=True)
            if arm == 'routing':
                run_arm('producer_feedback', workflows, ChatOperatorCompiler(chat), rules, lexicon, mentions,
                        path, protocol)
            else:
                run_baseline(arm, workflows, chat, rules, lexicon, mentions, path, protocol)
            print(f'END backend={backend} fold={fold} arm={arm}', flush=True)


def confirmed_rows(saved, gold, rules, identities, cap):
    from sop2program.ir import Workflow
    from sop2program.verify import verify
    rows = {}
    for doc, s in saved['documents'].items():
        row = s['budgets'][str(cap)]
        final = Workflow.model_validate(row['workflow'])
        assert final.source == gold[doc].source and final.initial == gold[doc].initial
        accepted = row['status'] in ('PASS', 'REPAIRED')
        if accepted:
            assert verify(final, rules)['pass']
        identity = identities[doc]
        confirmed = bool(accepted and verify(identity.rename(final, identity.gold().initial),
                                             evidence_checks=False, invariants=False)['pass'])
        rows[doc] = {'accepted': accepted, 'confirmed': confirmed, 'calls': row['calls'],
                     'input_tokens': row['input_tokens'], 'output_tokens': row['output_tokens'],
                     'generation_seconds': row['generation_seconds']}
    return rows


def score(backend, max_docs):
    from scripts.recheck_annotation_identity import mcnemar
    from sop2program.annotation import GoldIdentity
    from sop2program.ir import Workflow
    base = out_dir(backend, max_docs)
    records = []
    for fold in range(5):
        arms = [a for a in ARMS if (base / f'fold{fold}/{a}.json').exists()]
        if not arms:
            continue
        gold = {w['id']: Workflow.model_validate(w) for w in read(ROOT / f'data/cv/fold{fold}/test_workflows.json')}
        rules = read(ROOT / f'results/cv/fold{fold}/knowledge/full_rules.json')
        identities = {d: GoldIdentity(w, ROOT / w.metadata['peg_file']) for d, w in gold.items()}
        sources = {a: base / f'fold{fold}/{a}.json' for a in arms}
        sources['P_producer_feedback'] = ROOT / f'results/controlled_repair/fold{fold}/producer_feedback.json'
        sources['U_untuned_producer'] = ROOT / f'results/controlled_repair/fold{fold}/untuned_producer.json'
        sources['C_candidate_only'] = ROOT / f'results/robustness_extension/fold{fold}/producer_candidate_only.json'
        docs = None
        for name, path in sources.items():
            saved = read(path)
            if name in ARMS and not saved['complete']:
                continue
            if docs is None:
                docs = set(saved['documents'])
            for cap in BUDGETS:
                rows = confirmed_rows({'documents': {d: saved['documents'][d] for d in docs}}, gold, rules, identities, cap)
                records += [{'fold': fold, 'document': d, 'arm': name, 'budget': cap, **r} for d, r in rows.items()]
        recheck = read(ROOT / f'results/cv/fold{fold}/recheck_deterministic_qwen14.json')
        recheck = recheck[0] if isinstance(recheck, list) else recheck
        for d in docs:
            for cap in BUDGETS:
                records.append({'fold': fold, 'document': d, 'arm': 'D_deterministic', 'budget': cap, 'accepted': None,
                                'confirmed': bool(recheck['chain_identity_by_document'][d][1]), 'calls': 0,
                                'input_tokens': 0, 'output_tokens': 0, 'generation_seconds': 0.0})
    names = sorted({r['arm'] for r in records})
    pooled = []
    for name in names:
        for cap in BUDGETS:
            take = [r for r in records if r['arm'] == name and r['budget'] == cap]
            pooled.append({'arm': name, 'budget': cap, 'documents': len(take),
                           'confirmed': sum(r['confirmed'] for r in take),
                           'accepted': None if name == 'D_deterministic' else sum(r['accepted'] for r in take),
                           **{k: sum(r[k] for r in take) for k in ('calls', 'input_tokens', 'output_tokens', 'generation_seconds')}})
    at8 = {name: {r['document']: r['confirmed'] for r in records if r['arm'] == name and r['budget'] == 8} for name in names}
    comparisons = []
    for reference in ('P_producer_feedback', 'D_deterministic'):
        for name in names:
            if name == reference or set(at8[name]) != set(at8[reference]):
                continue
            gain = sum(at8[reference][d] and not at8[name][d] for d in at8[name])
            loss = sum(at8[name][d] and not at8[reference][d] for d in at8[name])
            comparisons.append({'reference': reference, 'arm': name, 'reference_only': gain, 'arm_only': loss,
                                'exact_mcnemar_p': mcnemar(gain, loss)})
    summary = {'backend': backend, 'pilot_max_docs': max_docs, 'endpoint': 'Full verifier acceptance AND annotation-chain state replay',
               'pooled': pooled, 'comparisons_at_cap8': comparisons}
    atomic(base / 'summary.json', summary)
    with (base / 'document_outcomes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in pooled:
        if row['budget'] == 8:
            print(f"{row['arm']:22s} cap8 confirmed {row['confirmed']}/{row['documents']} accepted {row['accepted']} "
                  f"seq {row['calls']} in {row['input_tokens']} out {row['output_tokens']}")
    for c in comparisons:
        print(c)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=BACKENDS, required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--max-docs', type=int, default=0, help='Pilot: first N documents per fold, separate output tree')
    p.add_argument('--score', action='store_true')
    a = p.parse_args()
    if a.score:
        score(a.backend, a.max_docs)
    else:
        run(a.backend, a.arms, a.folds, a.max_docs)
