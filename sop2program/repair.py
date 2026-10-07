"""Algorithm C's optional LLM node proposals, judged only by the external verifier."""
import json
from .ir import make_step, Operator, ground, norm, physical, semantic_template, triggers_match
from .verify import verify, minimal_repair, edit_cost, source_deviation

PRODUCERS={'SPIN','CONVERT','CREATE'}


def _source_trigger(workflow,index):
    ev=workflow.steps[index].evidence.get('trigger')
    if ev is None or not (0<=ev.start<ev.end<=len(workflow.source)):return None
    span=workflow.source[ev.start:ev.end]
    return span if triggers_match(workflow.steps[index].operator.trigger,span) else None


def _row(workflow,index,state,feedback):
    """Same context window as training rows: 400 characters before the trigger line."""
    ev=workflow.steps[index].evidence['trigger']
    start=workflow.source.rfind('\n',0,ev.start)+1
    end=workflow.source.find('\n',ev.end)
    if end<0:end=len(workflow.source)
    return {'source':workflow.source,'offset':ev.start,'trigger':workflow.source[ev.start:ev.end],
            'context':workflow.source[max(0,start-400):end],'state':state,'repair_feedback':feedback}


def _relink_dependencies(workflow):
    for i,step in enumerate(workflow.steps):
        pre=set(map(norm,step.operator.pre))
        step.depends_on=[old.id for old in workflow.steps[:i] if set(map(norm,old.operator.add))&pre]


def replay_progress(outcome):
    """Committed prefix length, then fewer violations. A pass ranks above everything."""
    if outcome['pass']:return (len(outcome['trace'])+1,0)
    prefix=next((i for i,t in enumerate(outcome['trace']) if not t['committed']),len(outcome['trace']))
    return (prefix,-len(outcome['violations']))


def missing_product_feedback(candidate,consumer_id,obj):
    """Shared by inference requests and training rows, so the model sees one format."""
    return {'candidate':candidate.model_dump(),
            'violations':[{'step':consumer_id,'code':'MISSING_PRODUCT','object':obj,
                           'detail':'A later step uses this object, but no earlier step produces it. '
                                    'If the source shows this step produces it, list it as a b or c argument.'}]}


def feedback_requests(workflow,rules=(),max_producers=2,retype_producers=False):
    """One round of model requests for the first replay failure.

    A missing object is sent back to the nearest earlier producer, after the last
    step that removed it, when the source mentions the object at or after that
    producer. With retype_producers, a step of any action type is asked when no
    producer qualifies, since its action may be the mislabelled part. Any other
    failure is sent back to the failing node itself.
    """
    outcome=verify(workflow,rules)
    if outcome['pass']:return outcome,[]
    first=outcome['violations'][0]
    index=next((i for i,s in enumerate(workflow.steps) if s.id==first['step']),None)
    if index is None:return outcome,[]
    requests=[];typed_producer=False
    if max_producers and first['code']=='STATE_VIOLATION' and first['field']=='pre' and first.get('required')=='exists':
        missing=norm(first['actual']);mention=first.get('mention',first['actual'])
        removed=max((i for i,t in enumerate(outcome['trace'][:index])
                     if missing in t['before']['exists'] and missing not in t['after']['exists']),default=-1)
        for actions in [PRODUCERS,None] if retype_producers else [PRODUCERS]:
            for j in range(index-1,removed,-1):
                if len(requests)>=max_producers:break
                step=workflow.steps[j]
                if actions is not None and step.operator.action not in actions:continue
                if _source_trigger(workflow,j) is None:continue
                if ground(mention,workflow.source[step.evidence['trigger'].start:]).start<0:continue
                requests.append({'index':j,'object':mention,
                                 'row':_row(workflow,j,outcome['trace'][j]['before'],
                                            missing_product_feedback(step.operator,workflow.steps[index].id,mention))})
            if requests:
                typed_producer=actions is not None
                break
    # A retyped producer is a guess, so the failing node is asked as well.
    if not typed_producer and _source_trigger(workflow,index) is not None:
        failures=[v for v in outcome['violations'] if v['step']==first['step']]
        requests.append({'index':index,'object':None,
                         'row':_row(workflow,index,outcome['trace'][index]['before'],
                                    {'candidate':workflow.steps[index].operator.model_dump(),'violations':failures})})
    return outcome,requests


def apply_proposal(workflow,request,op):
    """Workflow with one regenerated step, effects re-projected from its action and arguments."""
    if op is None or not triggers_match(op.trigger,request['row']['trigger']):return None
    if request['object'] is not None and norm(request['object']) not in {norm(a.text) for a in op.arguments if physical(a)}:
        return None
    op=op.model_copy(deep=True)
    for field,value in semantic_template(op.action,op.arguments).items():setattr(op,field,value)
    index=request['index'];candidate=workflow.model_copy(deep=True)
    step=make_step(op,workflow.source,index,request['row']['offset'],candidate.steps[:index])
    step.id=workflow.steps[index].id
    candidate.steps[index]=step
    _relink_dependencies(candidate)
    return candidate


def regenerate_with_compiler(workflow, compiler, rules=(), batch_size=8):
    """H4 contrast: rewrite every step from document-level feedback, not just failing nodes.

    Same model and same violation evidence as the local patcher; only the scope differs.
    """
    original=verify(workflow,rules)
    if original['pass']:
        return {'status':'PASS','workflow':workflow,'cost':0,'tested':1,'scope':'regeneration','rewritten':0}
    feedback={'violations':original['violations'],
              'workflow':[s.operator.model_dump() for s in workflow.steps]}
    rows=[]
    for i,step in enumerate(workflow.steps):
        ev=step.evidence.get('trigger')
        if ev is None or not (0<=ev.start<ev.end<=len(workflow.source)):continue
        if not triggers_match(step.operator.trigger,workflow.source[ev.start:ev.end]):continue
        line_start=workflow.source.rfind('\n',0,ev.start)+1
        line_end=workflow.source.find('\n',ev.end)
        if line_end<0:line_end=len(workflow.source)
        rows.append((i,{'source':workflow.source,'offset':ev.start,'trigger':step.operator.trigger,
                        'context':workflow.source[max(0,line_start-400):line_end],
                        'state':original['trace'][i]['before'],'repair_feedback':feedback}))
    candidate=workflow.model_copy(deep=True);rewritten=0
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size]
        for (i,row),(op,_) in zip(chunk,compiler.compile_batch([r for _,r in chunk])):
            if op is None or not triggers_match(op.trigger,workflow.steps[i].operator.trigger):continue
            step=make_step(op,workflow.source,i,row['offset'],candidate.steps[:i])
            step.id=workflow.steps[i].id
            rewritten+=step.operator!=workflow.steps[i].operator
            candidate.steps[i]=step
    outcome=verify(candidate,rules)
    return {'status':'REPAIRED' if outcome['pass'] else 'REVIEW',
            'workflow':candidate if outcome['pass'] else workflow,
            'cost':edit_cost(workflow,candidate) if outcome['pass'] else None,
            'tested':1,'scope':'regeneration','rewritten':rewritten,
            'source_deviation':source_deviation(candidate)}


JUDGE='''You are auditing a compiled laboratory workflow against its source protocol.
Decide whether any step contains an error: a missing object, a wrong type, a wrong effect, a broken dependency, an out-of-range parameter, or an unsupported mention.
Answer with only a JSON object {"error":true|false,"step":"sN"|null}. Use the step ids shown. No other text.'''


def judge_rows(workflow):
    """Prompt payload giving a text-only judge the same evidence the verifier replays."""
    return {'source':workflow.source,'initial':workflow.initial,
            'steps':[{'id':s.id,'operator':s.operator.model_dump(),'depends_on':s.depends_on}
                     for s in workflow.steps]}


def parse_judgement(raw):
    """Lenient parse; an unparsable answer counts as no detection, never as a pass."""
    start=raw.find('{');end=raw.rfind('}')
    if start<0 or end<start:return None
    try:payload=json.loads(raw[start:end+1])
    except ValueError:return None
    if not isinstance(payload,dict) or not isinstance(payload.get('error'),bool):return None
    step=payload.get('step')
    return {'error':bool(payload['error']),'step':step if isinstance(step,str) else None}


def repair_with_compiler(workflow, compiler, rules=(), max_proposals=4, lexicon=None):
    """Generate bounded local candidates; never accept a model's self-assessment."""
    original=verify(workflow,rules)
    if original['pass']:
        return {'status':'PASS','workflow':workflow,'cost':0,'tested':1,'model_proposals':[]}
    candidates=[];attempts=[];working=workflow.model_copy(deep=True)
    bad_ids={v['step'] for v in original['violations']}
    for i,step in enumerate(workflow.steps):
        if step.id not in bad_ids or len(attempts)>=max_proposals:continue
        ev=step.evidence.get('trigger')
        if ev is None or not (0<=ev.start<ev.end<=len(workflow.source)):continue
        if not triggers_match(step.operator.trigger,workflow.source[ev.start:ev.end]):continue
        checked=verify(working,rules)
        failures=[v for v in checked['violations'] if v['step']==step.id]
        if not failures:continue
        line_start=workflow.source.rfind('\n',0,ev.start)+1
        line_end=workflow.source.find('\n',ev.end)
        if line_end<0:line_end=len(workflow.source)
        row={'source':workflow.source,'offset':ev.start,'trigger':step.operator.trigger,
             'context':workflow.source[max(0,line_start-400):line_end],
             'state':checked['trace'][i]['before'],
             'repair_feedback':{'candidate':working.steps[i].operator.model_dump(),
                                'violations':failures}}
        op,info=compiler.compile(row)
        attempts.append({'step':step.id,'prediction':op.model_dump() if op else None,**info})
        if op is None or not triggers_match(op.trigger,step.operator.trigger):continue
        candidate=working.model_copy(deep=True)
        candidate.steps[i]=make_step(op,workflow.source,i,ev.start,candidate.steps[:i])
        candidate.steps[i].id=step.id
        candidates.append(candidate)
        working=candidate
    result=minimal_repair(workflow,rules,proposals=candidates,lexicon=lexicon)
    result['model_proposals']=attempts
    return result
