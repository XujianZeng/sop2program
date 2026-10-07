from sop2program.baselines import mine_declare,precondition_recovery,state_error_caught,step_graph_violations
from sop2program.ir import Argument,Operator,Workflow,make_step,semantic_template


def workflow():
    source='Create pellet. Wash pellet. Discard pellet.'
    steps=[]
    for i,(action,trigger) in enumerate([('CREATE','Create'),('WASH','Wash'),('DESTROY','Discard')]):
        args=[Argument(role='a',text='pellet',type='material')]
        op=Operator(action=action,trigger=trigger,arguments=args,**semantic_template(action,args))
        steps.append(make_step(op,source,i,source.index(trigger),steps))
    return Workflow(id='d',source=source,initial=[],steps=steps)


def test_step_graph_misses_a_use_after_destroy():
    source='Create pellet. Discard pellet. Wash pellet.'
    steps=[]
    for i,(action,trigger) in enumerate([('CREATE','Create'),('DESTROY','Discard'),('WASH','Wash')]):
        args=[Argument(role='a',text='pellet',type='material')]
        op=Operator(action=action,trigger=trigger,arguments=args,**semantic_template(action,args))
        steps.append(make_step(op,source,i,source.index(trigger),steps))
    broken=Workflow(id='d',source=source,initial=[],steps=steps)
    caught=state_error_caught(broken)
    assert caught['tst_ir_detected']
    assert not caught['step_graph_detected']


def test_tst_ir_stores_preconditions_the_step_graph_only_implies():
    scores=precondition_recovery(workflow())
    assert scores['tst_ir_recall']==1
    assert scores['step_graph_recovered']<=scores['tst_ir_recovered']


def test_declare_admits_a_stable_precedence_and_no_state_rule():
    def chain(name):
        source='Create pellet. Wash pellet.'
        steps=[]
        for i,(action,trigger) in enumerate([('CREATE','Create'),('WASH','Wash')]):
            args=[Argument(role='a',text='pellet',type='material')]
            op=Operator(action=action,trigger=trigger,arguments=args,**semantic_template(action,args))
            steps.append(make_step(op,source,i,source.index(trigger),steps))
        return Workflow(id=name,source=source,initial=[],steps=steps)
    train=[chain(f'train{i}') for i in range(20)]
    rules,_=mine_declare(train,min_docs=8,min_coverage=.5,stability=.8)
    assert rules and all(r['family']=='temporal' for r in rules)
    assert any(r['relation']=='precedence' and r['earlier']=='CREATE' and r['later']=='WASH' for r in rules)
