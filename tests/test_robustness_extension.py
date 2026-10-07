import copy
from types import SimpleNamespace
from collections import defaultdict

from scripts.refinement_extension import CandidateOnlyCompiler, ARM
from scripts.controlled_repair import run_arm
from paper.applied_intelligence.tools.audit_robustness import StrictIdentity
from sop2program.ir import Argument
from test_controlled_repair import StubCompiler, missing_product


def test_candidate_preserved_and_only_violation_field_removed():
    row = {'context': 'source', 'state': {'exists': ['tube']}, 'trigger': 'Spin',
           'repair_feedback': {'candidate': {'action': 'SPIN'}, 'violations': [{'object': 'pellet'}]}}
    before = copy.deepcopy(row)
    class Check:
        def compile_batch(self, rows):
            expected = copy.deepcopy(before); del expected['repair_feedback']['violations']
            assert rows == [expected]
            return []
    CandidateOnlyCompiler(Check()).compile_batch([row])
    assert before['repair_feedback']['violations']


def test_candidate_only_uses_producer_routing_and_records_actual_prompt(tmp_path):
    c = StubCompiler()
    saved = run_arm(ARM, [missing_product()], CandidateOnlyCompiler(c), [], {},
                    {'settings': set(), 'products': set()}, tmp_path/'arm.json', 'test')
    assert len(c.rows) == 2
    assert all(row['trigger'] == 'Centrifuge' for row in c.rows)
    assert all(set(e['prompt_row']['repair_feedback']) == {'candidate'} for e in saved['events'])
    assert saved['documents']['example']['budgets']['1']['calls'] == 1
    assert saved['documents']['example']['budgets']['8']['status'] == 'REPAIRED'
    assert saved['complete']


def strict_example():
    g = object.__new__(StrictIdentity)
    a = lambda text: Argument(role='a', text=text, type='material')
    g.workflow = SimpleNamespace(steps=[SimpleNamespace(operator=SimpleNamespace(arguments=[a('cells'), a('buffer')])),
                                        SimpleNamespace(operator=SimpleNamespace(arguments=[a('pellet')]))])
    g.mentions = [['c1', 'b1'], ['c2']]
    g.name = {'c1': 'cells #1', 'c2': 'pellet #2', 'b1': 'buffer #3'}
    g.by_text = defaultdict(set, {'cells': {'c1', 'c2'}, 'pellet': {'c2'}, 'buffer': {'b1'}})
    g.coverage = {}
    return g


def test_strict_resolver_abstains_on_ambiguity_and_substrings():
    g = strict_example()
    assert g.resolve('cells', 0) == 'cells #1'
    assert g.resolve('buffer', 1) == 'buffer #3'
    assert g.resolve('cells', 1, {'cells #1'}) == 'cells'
    assert g.coverage[1, 'cells'] == 'ambiguous'
    assert g.resolve('cell', 0) == 'cell'
    assert g.coverage[0, 'cell'] == 'unmatched'
    assert g.resolve('cells', 1, {'pellet #2'}) == 'cells'
