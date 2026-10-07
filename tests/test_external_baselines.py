import json

import pytest

from scripts.external_baselines import run_baseline
from tests.test_controlled_repair import missing_product

GOOD_SPIN = {'action': 'SPIN', 'arguments': [{'role': 'a', 'text': 'tube', 'type': 'container'},
                                             {'role': 'c', 'text': 'supernatant', 'type': 'material'}]}


class StubChat:
    name = 'stub'

    def __init__(self, replies, fail=False):
        self.replies, self.fail, self.requests = list(replies), fail, []

    def chat_batch(self, requests):
        if self.fail:
            raise RuntimeError('simulated outage before the round completed')
        self.requests += requests
        return [(self.replies.pop(0), {'input_tokens': 50, 'output_tokens': 10, 'amortized_latency_s': 0.1})
                for _ in requests]


def program(fix):
    steps = [{'id': 's0', 'action': 'SPIN', 'arguments': [{'role': 'a', 'text': 'tube', 'type': 'container'}]},
             {'id': 's1', **(GOOD_SPIN if fix else {'action': 'SPIN', 'arguments': [
                 {'role': 'a', 'text': 'tube', 'type': 'container'}]})},
             {'id': 's2', 'action': 'DESTROY', 'arguments': [{'role': 'a', 'text': 'supernatant', 'type': 'material'}]}]
    return json.dumps({'steps': steps})


def execute(tmp_path, arm, chat, fingerprint='frozen'):
    return run_baseline(arm, [missing_product()], chat, [], {}, {'settings': set(), 'products': set()},
                        tmp_path / f'{arm}.json', fingerprint)


def test_clairify_counts_unparsable_output_and_stops_once_verifier_passes(tmp_path):
    chat = StubChat(['not json', program(fix=True)])
    state = execute(tmp_path, 'clairify', chat)['documents']['example']
    assert state['calls'] == 2
    assert 'errors' in chat.requests[0]['messages'][-1]['content']
    assert state['budgets']['1']['calls'] == 1
    assert state['budgets']['2']['status'] in ('PASS', 'REPAIRED')
    assert state['budgets']['8'] == state['budgets']['2']


def test_reply_that_changes_nothing_ends_the_loop(tmp_path):
    chat = StubChat([program(fix=False)] * 8)
    state = execute(tmp_path, 'clairify', chat)['documents']['example']
    assert state['calls'] == 1
    assert state['budgets']['8'] == state['budgets']['1']


def test_critic_revises_only_the_failing_step(tmp_path):
    reply = json.dumps({'id': 'wrong-id', 'action': 'DESTROY',
                        'arguments': [{'role': 'a', 'text': 'tube', 'type': 'container'}]})
    chat = StubChat([reply] * 8)
    saved = execute(tmp_path, 'critic', chat)
    first = saved['events'][0]
    assert first['target'] == 's2' and first['changed_steps'] == ['s2']
    final = saved['documents']['example']['working']['steps']
    assert [s['operator']['action'] for s in final] == ['SPIN', 'SPIN', 'DESTROY']
    assert final[1]['operator']['arguments'] == [{'role': 'a', 'text': 'tube', 'type': 'container'}]


def test_self_refine_uses_no_verifier_and_stops_when_model_reports_done(tmp_path):
    chat = StubChat([json.dumps({'issues': [{'step': 's1', 'problem': 'supernatant is never produced'}], 'done': False}),
                     program(fix=True), json.dumps({'issues': [], 'done': True})])
    state = execute(tmp_path, 'self_refine', chat)['documents']['example']
    assert state['calls'] == 3
    assert all('verifier' not in r['messages'][-1]['content'] for r in chat.requests)
    assert 'supernatant is never produced' in chat.requests[1]['messages'][-1]['content']
    assert state['budgets']['8']['status'] in ('PASS', 'REPAIRED')


def test_resume_does_not_consume_a_failed_round(tmp_path):
    with pytest.raises(RuntimeError):
        execute(tmp_path, 'clairify', StubChat([], fail=True))
    saved = json.loads((tmp_path / 'clairify.json').read_text(encoding='utf-8'))
    assert saved['documents']['example']['calls'] == 0 and saved['next_round'] == 0
    done = execute(tmp_path, 'clairify', StubChat([program(fix=True)]))
    assert done['documents']['example']['calls'] == 1
    assert execute(tmp_path, 'clairify', StubChat([], fail=True)) == done
    with pytest.raises(AssertionError):
        execute(tmp_path, 'clairify', StubChat([]), fingerprint='changed')
