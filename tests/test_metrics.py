from sop2program.ir import Argument, Operator, Workflow, make_step, semantic_template
from sop2program.metrics import evaluate_predictions


def example():
    source='Create pellet. Wash pellet.'
    args=[Argument(role='a',text='pellet',type='material')]
    ops=[Operator(action=a,trigger=t,arguments=args,**semantic_template(a,args))
         for a,t in [('CREATE','Create'),('WASH','Wash')]]
    steps=[]
    rows=[]
    for i,op in enumerate(ops):
        steps.append(make_step(op,source,i,source.index(op.trigger),steps))
        rows.append({'id':f'd:s{i}','doc':'d','step':i,'offset':source.index(op.trigger),'target':op.model_dump()})
    workflow=Workflow(id='d',source=source,initial=[],steps=steps)
    predictions=[{'id':r['id'],'prediction':r['target']} for r in rows]
    return rows,predictions,{'d':workflow}


def test_perfect_projection_scores():
    rows,predictions,docs=example()
    scores,_=evaluate_predictions(rows,predictions,docs)
    assert scores['argument_tuple']['f1']==1
    assert scores['weak_dependency']['f1']==1
    assert scores['weak_transition_exact_rate']==1
    assert scores['full_workflow_pass_rate']==1
    graph=scores['process_graph']
    assert graph['node_f1']['f1']==1 and graph['edge_f1']['f1']==1
    assert graph['graph_edit_distance']==0 and graph['normalized_graph_edit_distance']==0
    assert graph['execution_order_accuracy']==1


def test_graph_scores_charge_a_wrong_action_and_a_missing_node():
    from copy import deepcopy
    rows,predictions,docs=example();predictions=deepcopy(predictions)
    predictions[1]['prediction']['action']='SPIN'
    scores,_=evaluate_predictions(rows,predictions,docs)
    graph=scores['process_graph']
    assert graph['node_f1']['tp']==1 and graph['node_f1']['f1']==.5
    # One label substitution plus the dependency edge the wrong effects no longer support.
    assert graph['graph_edit_distance']>0
    rows,predictions,docs=example();predictions=deepcopy(predictions)
    predictions[1]['prediction']=None
    missing,_=evaluate_predictions(rows,predictions,docs)
    assert missing['process_graph']['node_f1']['recall']==.5
    assert missing['process_graph']['execution_order_accuracy']==0


def test_missing_step_cannot_improve_pass_rate_or_recall():
    rows,predictions,docs=example()
    predictions[1]['prediction']=None
    scores,details= evaluate_predictions(rows,predictions,docs)
    assert scores['schema_valid_rate']==.5
    assert scores['argument_tuple']['recall']==.5
    assert scores['full_workflow_pass_rate']==0
    assert scores['repair']['review']==1
    assert details['repairs'][0]['reason'].startswith('Missing')


def test_no_predictions_is_not_a_valid_empty_workflow():
    rows,_,docs=example()
    scores,_=evaluate_predictions(rows,[],docs)
    assert scores['schema_valid_rate']==0
    assert scores['full_workflow_pass_rate']==0
    assert scores['failed_or_missing_step_rate']==1


def test_duplicate_argument_counts_as_false_positive():
    rows,predictions,docs=example()
    from copy import deepcopy
    predictions=deepcopy(predictions)
    predictions[0]['prediction']['arguments']*=2
    scores,_=evaluate_predictions(rows,predictions,docs)
    assert scores['argument_tuple']['precision']==2/3


def test_unknown_prediction_id_rejected():
    import pytest
    rows,predictions,docs=example()
    predictions[0]['id']='not-a-test-row'
    with pytest.raises(ValueError):evaluate_predictions(rows,predictions,docs)


def test_trigger_newline_still_anchors_the_marked_span():
    from copy import deepcopy
    rows,predictions,docs=example();predictions=deepcopy(predictions)
    predictions[1]['prediction']['trigger']='Wash\n'
    scores,_=evaluate_predictions(rows,predictions,docs)
    assert scores['anchor_match_rate']==1
    assert scores['complete_documents']==1


def test_action_from_different_source_span_is_not_a_complete_workflow():
    from copy import deepcopy
    rows,predictions,docs=example();predictions=deepcopy(predictions)
    predictions[1]['prediction']['trigger']='Create'
    scores,_=evaluate_predictions(rows,predictions,docs)
    assert scores['schema_valid_rate']==1
    assert scores['anchor_match_rate']==.5
    assert scores['full_workflow_pass_rate']==0
