"""Honest anchored-compiler metrics. Missing/invalid steps remain in denominators.

Arguments are role/text/type tuples, NOT independently adjudicated exact spans.
States, effects and dependencies use weak template projections, NOT human gold.
"""
from collections import Counter, defaultdict
from .ir import Operator, Workflow, make_step, norm, triggers_match
from .verify import verify, minimal_repair


def prf(tp, predicted, gold):
    return {'tp': tp, 'predicted': predicted, 'gold': gold,
            'precision': tp / predicted if predicted else 0.0,
            'recall': tp / gold if gold else 0.0,
            'f1': 2 * tp / (predicted + gold) if predicted + gold else 0.0}


def arguments(op):
    return Counter((a.role, norm(a.text), a.type) for a in op.arguments) if op else Counter()


def effects(op):
    if op is None:
        return None
    return (set(map(norm, op.add)), set(map(norm, op.delete)),
            {(norm(m.object), m.attribute, norm(m.value)) for m in op.modify})


def assemble_workflows(rows, predictions, docs):
    """Workflows whose steps survived whitespace-tolerant anchor matching."""
    expected={row['id']:row for row in rows}
    if len(expected)!=len(rows):
        raise ValueError('Duplicate evaluation rows')
    by_id={}
    for record in predictions:
        if record['id'] not in expected or record['id'] in by_id:
            raise ValueError('Unexpected or duplicate prediction ID')
        by_id[record['id']]=record
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['doc']].append(row)
    workflows=[]
    for key,group in sorted(grouped.items()):
        gold=docs[key]
        workflow=Workflow(id=key,source=gold.source,initial=gold.initial,steps=[],metadata=dict(gold.metadata))
        for row in sorted(group,key=lambda row:row['step']):
            record=by_id.get(row['id'])
            op=None
            if record and record.get('prediction') is not None:
                try:
                    op=Operator.model_validate(record['prediction'])
                except ValueError:
                    op=None
            target=Operator.model_validate(row['target'])
            if op is not None and triggers_match(op.trigger,target.trigger):
                workflow.steps.append(make_step(op,workflow.source,row['step'],row['offset'],workflow.steps))
        workflow.metadata['expected_steps']=len(group)
        workflow.metadata['schema_complete']=len(workflow.steps)==len(group)
        workflows.append(workflow)
    return workflows


def evaluate_predictions(rows, predictions, docs, rules=(), lexicon=None):
    expected = {row['id']: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate evaluation rows')
    by_id = {}
    for record in predictions:
        if record['id'] not in expected or record['id'] in by_id:
            raise ValueError('Unexpected or duplicate prediction ID')
        by_id[record['id']] = record
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['doc']].append(row)
    n = len(rows)
    counts = Counter()
    codes = Counter()
    per_doc = []
    all_violations = []
    repair_details = []
    for key, group in sorted(grouped.items()):
        gold = docs[key]
        workflow = Workflow(id=key, source=gold.source, initial=gold.initial, steps=[])
        gold_steps = {step.id: step for step in gold.steps}
        complete = True
        local_counts = Counter()
        for row in sorted(group, key=lambda row: row['step']):
            record = by_id.get(row['id'])
            op = None
            if record and record.get('prediction') is not None:
                try:
                    op = Operator.model_validate(record['prediction'])
                except ValueError:
                    pass
            target = Operator.model_validate(row['target'])
            counts['predictions_present'] += record is not None
            counts['schema_valid'] += op is not None
            anchor_match=op is not None and triggers_match(op.trigger,target.trigger)
            counts['anchor_match']+=anchor_match
            complete &= anchor_match
            if op:
                counts['action_correct'] += op.action == target.action
                counts['operator_exact'] += op == target
                counts['precondition_exact'] += set(map(norm, op.pre)) == set(map(norm, target.pre))
                counts['effect_exact'] += effects(op) == effects(target)
                step = make_step(op, workflow.source, row['step'], row['offset'], workflow.steps)
                if anchor_match:workflow.steps.append(step)
                fields = {'trigger': op.trigger, **{f'arguments.{j}': a.text for j, a in enumerate(op.arguments)}}
                for field, value in fields.items():
                    evidence = step.evidence[field]
                    supported = (0 <= evidence.start < evidence.end <= len(gold.source)
                                 and norm(gold.source[evidence.start:evidence.end]) == norm(value))
                    counts['evidence_fields'] += 1
                    counts['supported_evidence_fields'] += supported
            predicted_args = arguments(op)
            gold_args = arguments(target)
            correct = sum((predicted_args & gold_args).values())
            counts['arg_tp'] += correct
            counts['arg_pred'] += sum(predicted_args.values())
            counts['arg_gold'] += sum(gold_args.values())
            local_counts['arg_tp'] += correct
            local_counts['arg_pred'] += sum(predicted_args.values())
            local_counts['arg_gold'] += sum(gold_args.values())
        basic = verify(workflow, invariants=False)
        full = verify(workflow, rules)
        gold_trace = {entry['step']: entry for entry in verify(gold, invariants=False)['trace']}
        trace = {entry['step']: entry for entry in basic['trace']}
        for row in group:
            step_id = f"s{row['step']}"
            reference = gold_trace[step_id]
            if reference['committed']:
                counts['reference_executable_steps'] += 1
                counts['transition_exact'] += (step_id in trace and trace[step_id]['committed']
                                               and trace[step_id]['after'] == reference['after'])
        relevant_ids = {f"s{row['step']}" for row in group}
        gold_edges = {(dependency, step_id) for step_id, step in gold_steps.items() if step_id in relevant_ids
                      for dependency in step.depends_on}
        predicted_edges = {(dependency, step.id) for step in workflow.steps for dependency in step.depends_on}
        counts['dep_tp'] += len(gold_edges & predicted_edges)
        counts['dep_pred'] += len(predicted_edges)
        counts['dep_gold'] += len(gold_edges)
        # Section 5.2 process-graph scores. Anchors align nodes by step index, so the
        # graph edit distance is exact and needs no approximate node matching.
        predicted_nodes = {step.id: step.operator for step in workflow.steps}
        gold_nodes = {step_id: step.operator for step_id, step in gold_steps.items() if step_id in relevant_ids}
        shared = set(predicted_nodes) & set(gold_nodes)
        counts['node_tp'] += sum(predicted_nodes[k].action == gold_nodes[k].action for k in shared)
        counts['node_pred'] += len(predicted_nodes)
        counts['node_gold'] += len(gold_nodes)
        substitutions = sum(predicted_nodes[k] != gold_nodes[k] for k in shared)
        counts['ged'] += (substitutions + len(set(gold_nodes) ^ set(predicted_nodes))
                          + len(gold_edges ^ predicted_edges))
        counts['ged_norm'] += len(gold_nodes) + len(gold_edges)
        gold_commit = {entry['step']: i for i, entry in enumerate(verify(gold, invariants=False)['trace'])
                       if entry['committed']}
        predicted_commit = {entry['step']: i for i, entry in enumerate(basic['trace']) if entry['committed']}
        for before, after in gold_edges:
            if not (before in gold_commit and after in gold_commit and gold_commit[before] < gold_commit[after]):
                continue
            counts['order_gold'] += 1
            counts['order_correct'] += (before in predicted_commit and after in predicted_commit
                                        and predicted_commit[before] < predicted_commit[after])
        counts['complete_documents'] += complete
        counts['basic_pass'] += complete and basic['pass']
        counts['full_pass'] += complete and full['pass']
        failing_steps = {v['step'] for v in full['violations']}
        missing_steps = relevant_ids - {step.id for step in workflow.steps}
        counts['failed_or_missing_steps'] += len(failing_steps | missing_steps)
        for violation in full['violations']:
            codes[violation['code']] += 1
            all_violations.append({'doc': key, **violation})
        if missing_steps:
            codes['SCHEMA_MISSING_STEP'] += len(missing_steps)
        if complete:
            repair = minimal_repair(workflow, rules, lexicon=lexicon)
            detail = {'doc': key, 'status': repair['status'], 'cost': repair['cost'],
                      'tested': repair['tested'], 'operations': repair.get('operations', [])}
        else:
            detail = {'doc': key, 'status': 'REVIEW', 'cost': None, 'tested': 0,
                      'reason': 'Missing, schema-invalid or mismatched action anchor; cannot certify incomplete SOP'}
        repair_details.append(detail)
        per_doc.append({'doc': key, 'steps': len(group), 'schema_complete': complete,
                        'basic_pass': complete and basic['pass'], 'full_pass': complete and full['pass'],
                        'argument_tuple': prf(local_counts['arg_tp'], local_counts['arg_pred'], local_counts['arg_gold'])})
    documents = len(grouped)
    repaired = [r for r in repair_details if r['status'] == 'REPAIRED']
    failures = documents - counts['full_pass']
    summary = {
        'scope': 'Gold action anchors and initial inventory; weak projected state/dependency references; one seed.',
        'operators': n, 'documents': documents, 'predictions_present': counts['predictions_present'],
        'schema_valid_rate': counts['schema_valid'] / n if n else None,
        'anchor_match_rate': counts['anchor_match'] / n if n else None,
        'action_accuracy': counts['action_correct'] / n if n else None,
        'argument_tuple': prf(counts['arg_tp'], counts['arg_pred'], counts['arg_gold']),
        'operator_exact_rate': counts['operator_exact'] / n if n else None,
        'weak_precondition_exact_rate': counts['precondition_exact'] / n if n else None,
        'weak_effect_exact_rate': counts['effect_exact'] / n if n else None,
        'weak_dependency': prf(counts['dep_tp'], counts['dep_pred'], counts['dep_gold']),
        'process_graph': {
            'node_f1': prf(counts['node_tp'], counts['node_pred'], counts['node_gold']),
            'edge_f1': prf(counts['dep_tp'], counts['dep_pred'], counts['dep_gold']),
            'graph_edit_distance': counts['ged'],
            'normalized_graph_edit_distance': counts['ged'] / counts['ged_norm'] if counts['ged_norm'] else None,
            'execution_order_accuracy': counts['order_correct'] / counts['order_gold'] if counts['order_gold'] else None,
            'ordered_reference_pairs': counts['order_gold'],
            'scope': 'Nodes aligned by supplied anchors; edges are weak template dependencies.'},
        'weak_transition_exact_rate': counts['transition_exact'] / counts['reference_executable_steps']
                                      if counts['reference_executable_steps'] else None,
        'reference_executable_steps': counts['reference_executable_steps'],
        'predicted_evidence_support_rate': counts['supported_evidence_fields'] / counts['evidence_fields']
                                           if counts['evidence_fields'] else None,
        'basic_workflow_pass_rate': counts['basic_pass'] / documents if documents else None,
        'full_workflow_pass_rate': counts['full_pass'] / documents if documents else None,
        'complete_documents': counts['complete_documents'],
        'failed_or_missing_step_rate': counts['failed_or_missing_steps'] / n if n else None,
        'violation_counts': dict(codes),
        'repair': {'eligible_failed_documents': failures, 'repaired': len(repaired),
                   'success_rate': len(repaired) / failures if failures else None,
                   'mean_field_edit_cost': sum(r['cost'] for r in repaired) / len(repaired) if repaired else None,
                   'review': sum(r['status'] == 'REVIEW' for r in repair_details),
                   'mean_candidates_verified': sum(r['tested'] for r in repair_details) / len(repair_details)
                                               if repair_details else None,
                   'operations_used': dict(Counter(op.split(':')[-1] for r in repaired
                                                   for op in r.get('operations', ())).most_common()),
                   'action_correction_enabled': lexicon is not None,
                   'scope': 'bounded source/template patches; no LLM patch proposals'},
    }
    return summary, {'documents': per_doc, 'violations': all_violations, 'repairs': repair_details}
