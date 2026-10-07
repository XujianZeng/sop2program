import json
import pytest

from scripts.controlled_repair import run_arm
from sop2program.ir import Argument, Operator, Workflow, make_step, semantic_template


def missing_product():
    source = 'Centrifuge tube.\nCentrifuge tube again.\nRemove supernatant.'
    specs = [('SPIN', 'Centrifuge', 'tube', 'container', 0),
             ('SPIN', 'Centrifuge', 'tube', 'container', source.index('Centrifuge tube again')),
             ('DESTROY', 'Remove', 'supernatant', 'material', source.index('Remove'))]
    steps = []
    for i, (action, trigger, text, kind, offset) in enumerate(specs):
        args = [Argument(role='a', text=text, type=kind)]
        op = Operator(action=action, trigger=trigger, arguments=args, **semantic_template(action, args))
        steps.append(make_step(op, source, i, offset, steps))
    return Workflow(id='example', source=source, initial=['tube'], steps=steps,
                    metadata={'schema_complete': True})


class StubCompiler:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail

    def compile_batch(self, rows):
        self.rows += rows
        if self.fail:
            raise FloatingPointError('simulated interruption before completed round')
        outputs = []
        for i, row in enumerate(rows):
            args = [Argument(role='a', text='tube', type='container'),
                    Argument(role='c', text='supernatant', type='material')]
            good = Operator(action='SPIN', trigger=row['trigger'], arguments=args,
                            **semantic_template('SPIN', args))
            outputs.append((None if i == 0 else good,
                            {'input_tokens': 100, 'output_tokens': 20,
                             'amortized_latency_s': 0.1, 'raw': 'stub'}))
        return outputs


def execute(tmp_path, compiler, budgets=(1, 2, 4, 8), arm='producer_feedback', fingerprint='frozen'):
    return run_arm(arm, [missing_product()], compiler, [], {},
                   {'settings': set(), 'products': set()}, tmp_path / 'state.json', fingerprint, budgets)


def test_caps_count_invalid_outputs_and_finalize_request_prefix(tmp_path):
    c = StubCompiler()
    result = execute(tmp_path, c)
    state = result['documents']['example']
    assert len(c.rows) == 2
    assert state['budgets']['1']['calls'] == 1
    assert state['budgets']['1']['output_tokens'] == 20
    assert state['budgets']['2']['calls'] == 2
    assert state['budgets']['2']['status'] == 'REPAIRED'
    assert state['budgets']['1']['workflow'] != state['budgets']['2']['workflow']
    assert state['budgets']['8'] == state['budgets']['2']
    assert all(r['workflow']['initial'] == ['tube'] for r in state['budgets'].values())


def test_hard_cap_truncates_batch_and_no_feedback_omits_only_block(tmp_path):
    c = StubCompiler()
    result = execute(tmp_path, c, budgets=(1,), arm='producer_no_feedback')
    assert len(c.rows) == 1
    assert 'repair_feedback' not in c.rows[0]
    assert 'context' in c.rows[0] and 'state' in c.rows[0]
    assert result['documents']['example']['calls'] == 1


def test_resume_does_not_consume_a_failed_round_or_repeat_completed_work(tmp_path):
    with pytest.raises(FloatingPointError):
        execute(tmp_path, StubCompiler(fail=True))
    saved = json.loads((tmp_path / 'state.json').read_text(encoding='utf-8'))
    assert saved['documents']['example']['calls'] == 0
    assert saved['documents']['example']['tried'] == []
    done = execute(tmp_path, StubCompiler())
    skip = StubCompiler(fail=True)
    assert execute(tmp_path, skip) == done
    assert not skip.rows
    with pytest.raises(AssertionError):
        execute(tmp_path, skip, fingerprint='changed')


def test_node_arm_targets_failing_consumer(tmp_path):
    c = StubCompiler()
    execute(tmp_path, c, arm='node_feedback')
    assert len(c.rows) == 1
    assert c.rows[0]['trigger'] == 'Remove'
    assert c.rows[0]['repair_feedback']['violations']
