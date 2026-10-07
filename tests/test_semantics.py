from copy import deepcopy
from sop2program.ir import *
from sop2program.verify import verify,minimal_repair,State
from sop2program.induce import mine,rule_applies,rule_holds

def fixture():
    source='Create pellet. Wash pellet. Discard pellet.'
    ops=[]
    for action,trigger in [('CREATE','Create'),('WASH','Wash'),('DESTROY','Discard')]:
        args=[Argument(role='a',text='pellet',type='material')]
        ops.append(Operator(action=action,trigger=trigger,arguments=args,**semantic_template(action,args)))
    steps=[make_step(op,source,i,source.index(op.trigger)) for i,op in enumerate(ops)]
    return Workflow(id='fixture',source=source,initial=[],steps=steps)

def spin_with_invented_product(product_role='c'):
    source='Spin tube briefly.'
    args=[Argument(role='a',text='tube',type='container'),
          Argument(role=product_role,text='supernatant',type='material')]
    op=Operator(action='SPIN',trigger='Spin',arguments=args,**semantic_template('SPIN',args))
    return Workflow(id='spin',source=source,initial=['tube'],steps=[make_step(op,source,0)])

def test_product_absent_from_the_source_is_dropped_not_kept():
    w=spin_with_invented_product()
    assert any(v['code']=='UNSUPPORTED_HALLUCINATION' for v in verify(w)['violations'])
    result=minimal_repair(w)
    assert result['status']=='REPAIRED'
    assert 's0:drop_unsupported_argument.1' in result['operations']
    assert [a.text for a in result['workflow'].steps[0].operator.arguments]==['tube']

def test_operated_input_is_never_dropped_as_unsupported():
    source='Spin briefly.'
    args=[Argument(role='a',text='tube',type='container')]
    op=Operator(action='SPIN',trigger='Spin',arguments=args,**semantic_template('SPIN',args))
    w=Workflow(id='spin',source=source,initial=['tube'],steps=[make_step(op,source,0)])
    assert not any(o.startswith('drop_unsupported_argument') for o in minimal_repair(w).get('operations',[]))

MENTIONS={'settings':{'ice'},'products':{'pellet'}}

def chill_on(site_text,initial=('tube',)):
    source=f'Chill tube on {site_text}.'
    args=[Argument(role='a',text='tube',type='container'),Argument(role='site',text=site_text,type='container')]
    op=Operator(action='TEMP_TREAT',trigger='Chill',arguments=args,**semantic_template('TEMP_TREAT',args))
    return Workflow(id='chill',source=source,initial=list(initial),steps=[make_step(op,source,0)])

def test_unavailable_site_attested_as_setting_is_relabelled():
    result=minimal_repair(chill_on('ice'),mentions=MENTIONS)
    assert result['status']=='REPAIRED'
    assert 's0:setting_unavailable_argument.1' in result['operations']
    assert [(a.role,a.text) for a in result['workflow'].steps[0].operator.arguments]==[('a','tube'),('setting','ice')]

def test_unavailable_tool_is_dropped_but_commonly_produced_object_is_not():
    assert 's0:drop_unavailable_argument.1' in minimal_repair(chill_on('rack'),mentions=MENTIONS)['operations']
    assert minimal_repair(chill_on('pellet'),mentions=MENTIONS)['status']=='REVIEW'

def test_unavailable_ops_are_off_without_mentions_and_never_touch_inventory():
    assert minimal_repair(chill_on('ice'))['status']=='REVIEW'
    assert minimal_repair(chill_on('ice',initial=('tube','ice')),mentions=MENTIONS)['status']=='PASS'

def test_invented_product_is_renamed_to_the_object_a_later_step_needs():
    source='Spin tube briefly. Discard solution.'
    spin_args=[Argument(role='a',text='tube',type='container'),Argument(role='c',text='supernatant',type='material')]
    drop_args=[Argument(role='a',text='solution',type='material')]
    ops=[Operator(action='SPIN',trigger='Spin',arguments=spin_args,**semantic_template('SPIN',spin_args)),
         Operator(action='DESTROY',trigger='Discard',arguments=drop_args,**semantic_template('DESTROY',drop_args))]
    steps=[make_step(op,source,i,source.index(op.trigger)) for i,op in enumerate(ops)]
    w=Workflow(id='rename',source=source,initial=['tube'],steps=steps)
    assert minimal_repair(w)['status']=='REVIEW'
    result=minimal_repair(w,mentions=MENTIONS)
    assert result['status']=='REPAIRED'
    assert 's0:rename_unsupported_product.1' in result['operations']
    assert result['workflow'].steps[0].operator.add==['solution']

def test_lifecycle_and_transaction_rollback():
    w=fixture();assert verify(w)['pass'];assert verify(w)['final_state']['exists']==[]
    w.steps=[w.steps[0],w.steps[2],w.steps[1]]
    result=verify(w);assert not result['pass']
    assert any(v['code']=='STATE_VIOLATION' and v['step']=='s1' for v in result['violations'])
    assert result['trace'][-1]['before']==result['trace'][-1]['after']

def test_empty_pre_cannot_bypass_object_lifecycle():
    w=fixture();w.steps=[w.steps[1]];w.steps[0].operator.pre=[]
    assert not verify(w)['pass']

def test_source_span_is_bound_to_actual_field():
    w=fixture();w.steps[1].operator.arguments[0].text='invented_reagent'
    assert any(v['code']=='UNSUPPORTED_HALLUCINATION' for v in verify(w)['violations'])

def test_failed_dependency_not_satisfied_by_failed_node():
    w=fixture();w.steps[0].operator.arguments[0].text='unattested'
    w.steps[1].depends_on=['s0']
    assert any(v['code']=='DEPENDENCY_VIOLATION' for v in verify(w)['violations'])

def test_no_repair_by_inventing_initial_state_or_dropping_supported_step():
    w=fixture();w.steps=w.steps[1:]
    candidate=w.model_copy(deep=True);candidate.initial=['pellet']
    empty=w.model_copy(deep=True);empty.steps=[]
    assert minimal_repair(w,proposals=[candidate,empty])['status']=='REVIEW'

def test_repair_evidence_typo_without_gold_labels():
    w=fixture();w.steps[1].operator.arguments[0].text='pelet'
    repaired=minimal_repair(w)
    assert repaired['status']=='REPAIRED'
    assert verify(repaired['workflow'])['pass']

def test_inventory_stocks_first_use_but_never_restocks_a_destroyed_object():
    w=fixture()
    assert required_inventory(w.steps)==[]
    reordered=fixture();reordered.steps=[reordered.steps[1],reordered.steps[0]]
    assert required_inventory(reordered.steps)==['pellet']
    after_discard=fixture();after_discard.steps=[after_discard.steps[0],after_discard.steps[2],after_discard.steps[1]]
    assert required_inventory(after_discard.steps)==[]
    assert not verify(after_discard.model_copy(update={'initial':required_inventory(after_discard.steps)}))['pass']


def test_repair_composes_patches_across_several_failing_steps():
    w=fixture()
    w.steps[1].operator.modify=[];w.steps[2].operator.delete=[]
    assert len({v['step'] for v in verify(w)['violations']})==2
    repaired=minimal_repair(w)
    assert repaired['status']=='REPAIRED'
    assert verify(repaired['workflow'])['pass']


def test_action_relabelling_requires_corpus_evidence():
    from sop2program.verify import _patches
    w=fixture();w.steps[1].operator.modify=[]
    result=verify(w)
    unattested={label for label,_ in _patches(w,'s1',result,None)}
    attested={label for label,_ in _patches(w,'s1',result,{'wash':['WASH','MIX']})}
    assert not any(label.startswith('correct_action') for label in unattested)
    assert 'correct_action.MIX' in attested


def test_action_relabelling_cannot_create_a_missing_object():
    w=fixture();w.steps=w.steps[1:]
    assert minimal_repair(w,lexicon={'wash':['WASH','CREATE']})['status']=='REVIEW'


def device_fixture(device,doc):
    source='Create pellet. Wash pellet with brush. Discard pellet.'
    steps=[]
    for action,trigger,args in [
            ('CREATE','Create',[Argument(role='a',text='pellet',type='material')]),
            ('WASH','Wash',[Argument(role='a',text='pellet',type='material')]
             +([Argument(role='usage',text='brush',type='device')] if device else [])),
            ('DESTROY','Discard',[Argument(role='a',text='pellet',type='material')])]:
        op=Operator(action=action,trigger=trigger,arguments=args,**semantic_template(action,args))
        steps.append(make_step(op,source,len(steps),source.index(trigger),steps))
    return Workflow(id=doc,source=source,initial=[],steps=steps)


def test_holdout_confirmation_drops_a_rule_the_dev_split_contradicts():
    train=[device_fixture(True,f'train{i}') for i in range(30)]
    resource=lambda rules:[r for r in rules if r['family']=='resource' and r['action']=='WASH']
    mined,_=mine(train,min_docs=4,min_coverage=.1)
    assert resource(mined)
    holdout=[device_fixture(False,'dev0')]
    assert verify(holdout[0])['pass']
    confirmed,_=mine(train,min_docs=4,min_coverage=.1,holdout=holdout)
    assert not resource(confirmed)
    assert all(c['holdout_rejected']==0 for c in confirmed)


def test_grounding_matches_the_whitespace_the_verifier_ignores():
    source='Add NEB 5-alpha\u00a0Competent cells to the tube.'
    text='NEB 5-alpha Competent cells'
    ev=ground(text,source)
    assert ev.start>=0 and norm(ev.text)==norm(text)
    args=[Argument(role='a',text=text,type='material')]
    op=Operator(action='CREATE',trigger='Add',arguments=args,**semantic_template('CREATE',args))
    w=Workflow(id='nbsp',source=source,initial=[],steps=[make_step(op,source,0,source.index('Add'))])
    assert verify(w)['pass']


def spin_fixture(duration,doc):
    source=f'Create pellet. Spin pellet for {duration} min. Discard pellet.'
    trigger=f'Spin'
    steps=[]
    for action,trig,args in [
            ('CREATE','Create',[Argument(role='a',text='pellet',type='material')]),
            ('SPIN',trigger,[Argument(role='a',text='pellet',type='material'),
                             Argument(role='setting',text=f'{duration} min',type='setting')]),
            ('DESTROY','Discard',[Argument(role='a',text='pellet',type='material')])]:
        op=Operator(action=action,trigger=trig,arguments=args,**semantic_template(action,args))
        steps.append(make_step(op,source,len(steps),source.index(trig),steps))
    return Workflow(id=doc,source=source,initial=[],steps=steps)


def test_parameter_bound_admits_values_beyond_the_observed_envelope():
    """The training envelope is a sample, not the admissible range."""
    train=[spin_fixture(d,f'train{i}') for i,d in enumerate(range(5,25))]
    rules,_=mine(train,min_docs=4,min_coverage=.1)
    duration=[r for r in rules if r['family']=='parameter' and r['value'][1]=='min']
    assert duration,'a duration invariant should be admitted'
    low,high=duration[0]['value'][2],duration[0]['value'][3]
    assert duration[0]['observed']==[5.0,24.0]
    assert high>24 and low<5
    assert verify(spin_fixture(40,'unseen'),rules)['pass']
    assert not verify(spin_fixture(40000,'absurd'),rules)['pass']


def test_effect_invariant_spares_an_operator_that_cannot_produce_the_effect():
    """TRANSFER only relocates when a destination is given, so a themed-only
    TRANSFER is out of scope rather than in violation."""
    rule={'action':'TRANSFER','type':'*','family':'effect','value':'modify','id':'r','code':'INVARIANT_VIOLATION'}
    source='Create pellet. Transfer pellet to tube.'
    args=[Argument(role='a',text='pellet',type='material')]
    op=Operator(action='TRANSFER',trigger='Transfer',arguments=args,**semantic_template('TRANSFER',args))
    assert not rule_applies(rule,op)
    sited=args+[Argument(role='site',text='tube',type='container')]
    placed=Operator(action='TRANSFER',trigger='Transfer',arguments=sited,**semantic_template('TRANSFER',sited))
    assert rule_applies(rule,placed)
    placed.modify=[]
    assert not rule_holds(rule,placed,State(set()),[])
    placed.modify=[Change(object='pellet',attribute='location',value='elsewhere')]
    assert not rule_holds(rule,placed,State(set()),[])
    placed.modify=[Change(object='Pellet',attribute='location',value='Tube')]
    assert rule_holds(rule,placed,State(set()),[])


def test_temporal_invariant_needs_a_predecessor_on_the_same_object():
    """Otherwise any action that merely happened earlier counts as a dependency."""
    rule={'action':'SEAL','type':'*','family':'temporal','value':'WASH','id':'r','code':'DEPENDENCY_VIOLATION'}
    args=[Argument(role='a',text='pellet',type='material')]
    op=Operator(action='SEAL',trigger='Seal',arguments=args,**semantic_template('SEAL',args))
    def history(text):
        other=[Argument(role='a',text=text,type='material')]
        washed=Operator(action='WASH',trigger='Wash',arguments=other,**semantic_template('WASH',other))
        return [{'committed':True,'operator':washed.model_dump()}]
    assert rule_holds(rule,op,State({'pellet'}),history('pellet'))
    assert not rule_holds(rule,op,State({'pellet'}),history('filter'))


def test_low_count_rules_not_admitted():
    rules,audit=mine([fixture(),fixture().model_copy(update={'id':'second'})])
    assert not rules and audit

def test_induction_counts_documents_not_repeated_events():
    w=fixture();w.steps=w.steps[:2]
    rules,audit=mine([w]*100)
    assert not rules
    assert all(c['applicable_docs']==1 for c in audit)

def test_deterministic_replay():
    w=fixture();assert verify(w)==verify(w)


def test_destroy_cannot_omit_deletion():
    w=fixture();w.steps[2].operator.delete=[]
    assert any(v['field']=='delete' for v in verify(w)['violations'])


def test_effect_cannot_delete_unrelated_inventory():
    w=fixture();w.initial=['tube'];w.steps[2].operator.delete.append('tube')
    result=verify(w)
    assert not result['pass']
    assert 'tube' in result['final_state']['exists']


def test_unsupported_location_attribute_is_rejected():
    w=fixture();w.steps[1].operator.modify=[Change(object='pellet',attribute='location',value='unmentioned destination')]
    assert not verify(w)['pass']


def test_signed_parameters_and_speed_unit_identity():
    from sop2program.induce import numeric_parameters
    assert numeric_parameters('-20 °C, .5 mL, 12,000 × g, 800 rpm')==[(-20.,'°c'),(.5,'ml'),(12000.,'xg'),(800.,'rpm')]


def test_parameter_rule_uses_each_units_kind():
    from sop2program.induce import rule_holds
    w=fixture();op=w.steps[1].operator
    op.arguments.append(Argument(role='setting',type='setting',text='37°C for 500 minutes'))
    assert not rule_holds({'family':'parameter','value':['duration','min',1,60]},op,State(),[])
