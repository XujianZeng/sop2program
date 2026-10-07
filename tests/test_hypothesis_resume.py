import json

import pytest

from scripts.hypothesis_tests import CaseCache, document_permutation, exact_paired_binary, local_repair_result
from sop2program.ir import Argument,Operator,Workflow,make_step,semantic_template


def test_completed_arm_is_reused_after_process_restart(tmp_path):
    cache = CaseCache(tmp_path, {'adapter': 'one'})
    assert cache.compute('H4/doc/local', lambda: {'status': 'REPAIRED'})['status'] == 'REPAIRED'
    restarted = CaseCache(tmp_path, {'adapter': 'one'})
    def should_not_run():
        raise AssertionError('completed arm was recomputed')
    assert restarted.compute('H4/doc/local', should_not_run) == {'status': 'REPAIRED'}


def test_failed_arm_is_not_cached(tmp_path):
    cache = CaseCache(tmp_path, {'adapter': 'one'})
    def crash():
        raise RuntimeError('GPU reset')
    with pytest.raises(RuntimeError):
        cache.compute('H4/doc/regeneration', crash)
    assert cache.get('H4/doc/regeneration') is None
    assert cache.compute('H4/doc/regeneration', lambda: {'status': 'REVIEW'}) == {'status': 'REVIEW'}


def test_cache_refuses_changed_provenance(tmp_path):
    CaseCache(tmp_path, {'adapter': 'one'})
    with pytest.raises(ValueError, match='configuration differs'):
        CaseCache(tmp_path, {'adapter': 'two'})
    assert json.loads((tmp_path / 'config.json').read_text()) == {'adapter': 'one'}


def test_exact_binary_test_handles_ties_and_small_sample():
    assert exact_paired_binary([True, False], [True, False])['p_value'] == 1
    result = exact_paired_binary([True] * 4, [False] * 4)
    assert result['left_only'] == 4
    assert result['p_value'] == .125


def test_document_test_keeps_correlated_mutations_in_one_block():
    rows = [{'doc': 'a', 'left': True, 'right': False}] * 10
    result = document_permutation(rows, 'left', 'right')
    assert result['cases'] == 10
    assert result['documents'] == 1
    assert result['p_value'] == 1  # Ten variants of one document are not ten independent trials.
    rows += [{'doc': 'b', 'left': True, 'right': False}] * 10
    assert document_permutation(rows, 'left', 'right')['p_value'] == .5


def test_document_test_cancels_within_document_differences():
    rows = [{'doc': 'a', 'left': True, 'right': False},
            {'doc': 'a', 'left': False, 'right': True}]
    assert document_permutation(rows, 'left', 'right')['p_value'] == 1


def test_incomplete_workflow_cannot_be_counted_as_repaired_or_passed():
    arguments=[Argument(role='a',text='pellet',type='material')]
    operator=Operator(action='WASH',trigger='Wash',arguments=arguments,**semantic_template('WASH',arguments))
    workflow=Workflow(id='doc',source='Wash pellet.',initial=['pellet'],
                      steps=[make_step(operator,'Wash pellet.',0)],
                      metadata={'expected_steps':2,'schema_complete':False})
    result=local_repair_result(workflow,[],None)
    assert result['status']=='REVIEW'
    assert result['cost'] is None
def test_incomplete_h4_inputs_reject_without_wasting_model_calls(tmp_path,monkeypatch):
    import json
    import scripts.hypothesis_tests as hypotheses
    from sop2program.ir import Workflow
    workflow=Workflow(id='missing',source='Wait.',initial=[],steps=[],
                      metadata={'expected_steps':1,'schema_complete':False})
    directory=tmp_path/'lora';directory.mkdir()
    (directory/'workflows.json').write_text(json.dumps([workflow.model_dump()]),encoding='utf-8')
    monkeypatch.setattr(hypotheses,'ROOT',tmp_path)
    class Compiler:
        def __init__(self,*args):pass
        def compile(self,*args):raise AssertionError('Missing-node arm must reject before generation')
        def compile_batch(self,*args):raise AssertionError('Missing-node arm must reject before generation')
    monkeypatch.setattr(hypotheses,'LocalCompiler',Compiler)
    result=hypotheses.run_h4(None,None,[],{},'lora',0)
    assert result['structural_rejections']==1
    assert result['documents'][0]['local_llm']['status']=='REVIEW'
    assert result['documents'][0]['regeneration']['status']=='REVIEW'
    assert result['documents'][0]['regeneration']['rewritten']==0

