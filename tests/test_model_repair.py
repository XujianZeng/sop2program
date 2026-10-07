from sop2program.ir import Argument,Operator,Workflow,make_step,semantic_template
from sop2program.repair import (apply_proposal,feedback_requests,repair_with_compiler,
                                parse_judgement,replay_progress)
from sop2program.verify import verify


def two_spins(second_products):
    source='Centrifuge tube.\nDiscard supernatant.\nCentrifuge tube again.\nRemove supernatant.'
    tube=Argument(role='a',text='tube',type='container')
    spec=[('SPIN','Centrifuge',[tube,Argument(role='c',text='supernatant',type='material')],0),
          ('DESTROY','Discard',[Argument(role='a',text='supernatant',type='material')],source.index('Discard')),
          ('SPIN','Centrifuge',[tube,*second_products],source.index('Centrifuge tube again')),
          ('DESTROY','Remove',[Argument(role='a',text='supernatant',type='material')],source.index('Remove'))]
    steps=[]
    for i,(action,trigger,args,near) in enumerate(spec):
        op=Operator(action=action,trigger=trigger,arguments=args,**semantic_template(action,args))
        steps.append(make_step(op,source,i,near,steps))
    return Workflow(id='d',source=source,initial=['tube'],steps=steps)


def test_missing_product_is_sent_to_the_producer_after_the_last_discard():
    workflow=two_spins([])
    outcome,requests=feedback_requests(workflow)
    assert not outcome['pass']
    assert [r['index'] for r in requests]==[2]
    assert requests[0]['row']['repair_feedback']['violations'][0]['code']=='MISSING_PRODUCT'
    product=[Argument(role='a',text='tube',type='container'),Argument(role='c',text='supernatant',type='material')]
    proposal=Operator(action='SPIN',trigger='Centrifuge\n',arguments=product,pre=[],add=[],delete=[],modify=[])
    candidate=apply_proposal(workflow,requests[0],proposal)
    assert verify(candidate)['pass']
    assert replay_progress(verify(candidate))>replay_progress(outcome)


def test_producer_proposal_without_the_object_is_rejected():
    workflow=two_spins([])
    _,requests=feedback_requests(workflow)
    same=workflow.steps[2].operator
    assert apply_proposal(workflow,requests[0],same) is None


def test_node_feedback_only_when_producers_are_disabled():
    workflow=two_spins([])
    _,requests=feedback_requests(workflow,max_producers=0)
    assert [r['index'] for r in requests]==[3]


def mistyped_spin():
    workflow=two_spins([])
    tube=[Argument(role='a',text='tube',type='container')]
    op=Operator(action='OTHER',trigger='Centrifuge',arguments=tube,**semantic_template('OTHER',tube))
    step=make_step(op,workflow.source,2,workflow.source.index('Centrifuge tube again'),workflow.steps[:2])
    step.id=workflow.steps[2].id;workflow.steps[2]=step
    return workflow


def test_mistyped_producer_is_asked_only_when_retyping_is_enabled():
    workflow=mistyped_spin()
    _,requests=feedback_requests(workflow)
    assert [(r['index'],r['object']) for r in requests]==[(3,None)]
    _,requests=feedback_requests(workflow,retype_producers=True)
    assert [(r['index'],r['object']) for r in requests]==[(2,'supernatant'),(3,None)]
    product=[Argument(role='a',text='tube',type='container'),Argument(role='c',text='supernatant',type='material')]
    proposal=Operator(action='SPIN',trigger='Centrifuge\n',arguments=product,pre=[],add=[],delete=[],modify=[])
    assert verify(apply_proposal(workflow,requests[0],proposal))['pass']


def test_typed_producer_is_still_preferred_when_retyping():
    _,requests=feedback_requests(two_spins([]),retype_producers=True)
    assert [r['index'] for r in requests]==[2]


def fixture():
    source='Wash pellet.'
    args=[Argument(role='a',text='pellet',type='setting')]
    op=Operator(action='WASH',trigger='Wash',arguments=args,**semantic_template('WASH',args))
    return Workflow(id='d',source=source,initial=['pellet'],steps=[make_step(op,source,0)])


class CorrectingCompiler:
    def compile(self,row):
        assert row['repair_feedback']['violations']
        args=[Argument(role='a',text='pellet',type='material')]
        return Operator(action='WASH',trigger='Wash',arguments=args,**semantic_template('WASH',args)),{'raw':'candidate'}


def test_source_grounded_type_patch_is_externally_accepted():
    workflow=fixture()
    result=repair_with_compiler(workflow,CorrectingCompiler())
    assert result['status']=='REPAIRED'
    assert workflow.steps[0].operator.arguments[0].type=='setting'
    assert len(result['model_proposals'])==1


def test_model_patch_cannot_invent_missing_inventory():
    workflow=fixture();workflow.initial=[]
    result=repair_with_compiler(workflow,CorrectingCompiler())
    assert result['status']=='REVIEW'
    assert result['workflow'].initial==[]


def test_judge_string_boolean_is_not_counted_as_detection():
    assert parse_judgement('{"error":"false","step":null}') is None
    assert parse_judgement('{"error":false,"step":null}') == {'error':False,'step':None}
    assert parse_judgement('```json\n{"error":true,"step":"s2"}\n```') == {'error':True,'step':'s2'}
