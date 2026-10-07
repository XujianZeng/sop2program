from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
import re
from .ir import Workflow, Step, norm, physical, semantic_template

@dataclass
class State:
    exists: set = field(default_factory=set)
    attributes: dict = field(default_factory=dict)

    def serialize(self):
        return {'exists':sorted(self.exists),'attributes':dict(sorted(self.attributes.items()))}

def parameter_kind(text):
    text=text.lower().replace('渭','u').replace('碌','u')
    if re.search(r'(掳\s*[cf]|degrees?\s*[cf]|temperature)',text): return 'temperature'
    if re.search(r'\b(rpm|rcf)\b|[x脳]\s*g\b',text): return 'speed'
    if re.search(r'\b(sec(onds?)?|s|min(utes?)?|h(ours?)?|hrs?)\b',text): return 'duration'
    if re.search(r'\b(ul|ml|liters?|litres?)\b',text): return 'volume'
    return 'unknown'

class _Aliased(set):
    """Membership through predicted identity, for invariants written against mention text."""
    def __init__(self,keys,resolve):
        super().__init__(keys);self._resolve=resolve
    def __contains__(self,item):
        return set.__contains__(self,item) or set.__contains__(self,self._resolve(item))

def verify(workflow:Workflow,rules=(),*,state_checks=True,evidence_checks=True,invariants=True):
    from .identity import identity_resolver
    resolver=identity_resolver(workflow)
    state=State(set(resolver.order) if resolver else set(map(norm,workflow.initial)))
    violations=[];trace=[];committed=set()
    seen_ids=set()
    for index,step in enumerate(workflow.steps):
        op=step.operator;local=[]
        if resolver is None:key=norm
        else:
            alive=frozenset(state.exists);position=int(step.id[1:]) if step.id[1:].isdigit() else index
            key=lambda text,alive=alive,position=position:resolver.resolve(text,alive,position)
        def fail(code,field,required,actual):
            local.append({'step':step.id,'code':code,'field':field,'required':required,'actual':actual,
                          'evidence':{k:v.model_dump() for k,v in step.evidence.items()}})
        if step.id in seen_ids: fail('SCHEMA_VIOLATION','id','unique',step.id)
        seen_ids.add(step.id)
        if any(d not in committed for d in step.depends_on):
            fail('DEPENDENCY_VIOLATION','depends_on','previously committed',step.depends_on)
        if evidence_checks:
            fields={'trigger':op.trigger,**{f'arguments.{i}':a.text for i,a in enumerate(op.arguments)}}
            for name,text in fields.items():
                ev=step.evidence.get(name)
                if ev is None or not (0<=ev.start<ev.end<=len(workflow.source)) or norm(workflow.source[ev.start:ev.end])!=norm(text) or norm(ev.text)!=norm(text):
                    fail('UNSUPPORTED_HALLUCINATION',name,text,ev.model_dump() if ev else None)
            mentioned={norm(a.text) for a in op.arguments if physical(a)}
            for name,objects in [('pre',op.pre),('add',op.add),('delete',op.delete),('modify',[m.object for m in op.modify])]:
                for obj in objects:
                    if norm(obj) not in mentioned and key(obj) not in state.exists:
                        fail('UNSUPPORTED_HALLUCINATION',name,'source-bound object',obj)
        for a in op.arguments:
            if a.role in ('a','b','c','site') and a.type in ('setting','measurement'):
                fail('TYPE_VIOLATION','arguments','physical object',a.model_dump())
        expected=semantic_template(op.action,op.arguments)
        reads={t:key(t) for t in [*op.pre,*expected['pre'],*op.delete,*(m.object for m in op.modify)]}
        if state_checks:
            required={reads[t] for t in [*op.pre,*expected['pre']]}
            # Mandatory lifecycle check cannot be bypassed by deleting model preconditions.
            for missing in sorted(required-state.exists):
                fail('STATE_VIOLATION','pre','exists',missing)
                if resolver is not None:
                    local[-1]['mention']=next(t for t in [*op.pre,*expected['pre']] if reads[t]==missing)
            for m in op.modify:
                if reads[m.object] not in state.exists and norm(m.object) not in set(map(norm,op.add)):
                    fail('STATE_VIOLATION','modify','existing object',m.object)
            if set(map(norm,op.add)) & set(map(norm,op.delete)):
                fail('STATE_VIOLATION','effects','disjoint add/delete','overlap')
            # Effect legality is part of this conservative executable subset. A model
            # cannot discard unrelated inventory or omit DESTROY to avoid a later failure.
            for name in ('add','delete'):
                actual=set(map(norm,getattr(op,name)))
                expected_objects=set(map(norm,expected[name]))
                if actual!=expected_objects:
                    fail('STATE_VIOLATION',name,sorted(expected_objects),sorted(actual))
            actual_changes={(norm(m.object),m.attribute,norm(m.value)) for m in op.modify}
            expected_changes={(norm(m.object),m.attribute,norm(m.value)) for m in expected['modify']}
            if actual_changes!=expected_changes:
                fail('STATE_VIOLATION','modify',sorted(expected_changes),sorted(actual_changes))
        if invariants:
            from .induce import rule_applies,rule_holds
            seen=state if resolver is None else State(_Aliased(state.exists,key),state.attributes)
            for rule in rules:
                if rule_applies(rule,op) and not rule_holds(rule,op,seen,trace):
                    fail(rule.get('code','INVARIANT_VIOLATION'),rule['family'],rule['value'],rule['id'])
        before=state.serialize()
        if not local:
            for obj in op.delete:
                state.exists.discard(reads[obj]);state.attributes.pop(reads[obj],None)
            if resolver is None:
                products={norm(t):norm(t) for t in op.add}
            else:
                products=resolver.commit(position,[(t,reads[t]) for t in [*op.pre,*expected['pre']]],op.add)
            state.exists.update(products.values())
            for m in op.modify:
                target=products.get(norm(m.object),reads[m.object])
                state.attributes.setdefault(target,{})[m.attribute]=m.value
            committed.add(step.id)
        violations.extend(local)
        trace.append({'step':step.id,'operator':op.model_dump(),'before':before,'after':deepcopy(state.serialize()),'committed':not local})
    return {'pass':not violations,'violations':violations,'trace':trace,'final_state':state.serialize()}

def edit_cost(a:Workflow,b:Workflow):
    """Field edits plus insertion/deletion cost, with stable step IDs."""
    aa={s.id:s.model_dump() for s in a.steps};bb={s.id:s.model_dump() for s in b.steps}
    cost=len(set(aa)^set(bb))
    for key in set(aa)&set(bb):
        for field in aa[key]['operator']:
            cost+=aa[key]['operator'][field]!=bb[key]['operator'][field]
        cost+=aa[key]['depends_on']!=bb[key]['depends_on']
        cost+=aa[key]['evidence']!=bb[key]['evidence']
    return cost

def unsupported_fields(workflow:Workflow):
    """Eq. (10) Unsup: mention fields whose evidence does not quote the source."""
    total=0
    for step in workflow.steps:
        op=step.operator
        fields={'trigger':op.trigger,**{f'arguments.{i}':a.text for i,a in enumerate(op.arguments)}}
        for name,text in fields.items():
            ev=step.evidence.get(name)
            if ev is None or not (0<=ev.start<ev.end<=len(workflow.source)) or norm(workflow.source[ev.start:ev.end])!=norm(text):
                total+=1
    return total

def source_deviation(workflow:Workflow):
    """Operator mentions that do not occur verbatim in the SOP text."""
    from .ir import ground
    return sum(ground(text,workflow.source).start<0
               for step in workflow.steps
               for text in [step.operator.trigger,*(a.text for a in step.operator.arguments)])

REPAIR_WEIGHTS={'edit':1.,'violation':4.,'unsupported':2.,'deviation':2.}

def repair_objective(original:Workflow,candidate:Workflow,violations,weights=None):
    """Eq. (10) with Dev measured as extra distance from the source, not from the candidate."""
    w=weights or REPAIR_WEIGHTS
    drift=max(0,source_deviation(candidate)-source_deviation(original))
    return (w['edit']*edit_cost(original,candidate)+w['violation']*len(violations)
            +w['unsupported']*unsupported_fields(candidate)+w['deviation']*drift)

def _admissible(original:Workflow,candidate:Workflow):
    """A repair may not rewrite the SOP, invent inventory, or drop attested operations."""
    from collections import Counter
    from .ir import ground
    if candidate.initial!=original.initial or candidate.source!=original.source:return False
    attested=Counter(norm(s.operator.trigger) for s in original.steps
                     if ground(s.operator.trigger,original.source).start>=0)
    return not attested-Counter(norm(s.operator.trigger) for s in candidate.steps)

def _rebuild(workflow:Workflow,index:int):
    """Recompute evidence and dependencies after a step's mentions changed."""
    from .ir import make_step
    step=workflow.steps[index];near=step.evidence['trigger'].start
    rebuilt=make_step(step.operator,workflow.source,index,max(near,0),workflow.steps[:index])
    rebuilt.id=step.id;workflow.steps[index]=rebuilt

def _patches(workflow:Workflow,step_id,result,lexicon=None,drop_arguments=True,mentions=None):
    """The six repair operations of Section 3.3, each deterministic and source-bound,
    plus dropping a non-input argument the source never mentions. With training-split
    setting/product mentions, also demote unavailable site/usage objects and rename invented products."""
    from .ir import ground,Argument
    index=next(i for i,s in enumerate(workflow.steps) if s.id==step_id)
    source=workflow.source;step=workflow.steps[index]
    before=next(t['before'] for t in result['trace'] if t['step']==step_id)
    codes={v['code'] for v in result['violations'] if v['step']==step_id}
    out=[]
    def variant(label,mutate):
        w=workflow.model_copy(deep=True)
        if mutate(w):out.append((label,w))

    def reground(w):
        """Re-link an argument to the span its own evidence already points at."""
        changed=False
        for key,ev in w.steps[index].evidence.items():
            if key.startswith('arguments.') and 0<=ev.start<ev.end<=len(source):
                j=int(key.split('.')[1])
                if w.steps[index].operator.arguments[j].text!=source[ev.start:ev.end]:
                    w.steps[index].operator.arguments[j].text=source[ev.start:ev.end]
                    ev.text=source[ev.start:ev.end];ev.provenance='source';changed=True
        return changed
    variant('reground_arguments',reground)

    def retemplate(w):
        """Restore the conservative effect projection for the declared action."""
        op=w.steps[index].operator;fixed=semantic_template(op.action,op.arguments)
        if all(getattr(op,k)==v for k,v in fixed.items()):return False
        for k,v in fixed.items():setattr(op,k,v)
        return True
    variant('template_effects',retemplate)

    def redepend(w):
        """Drop dependency edges on steps the simulator never committed."""
        committed={t['step'] for t in result['trace'] if t['committed']}
        kept=[d for d in w.steps[index].depends_on if d in committed]
        if kept==w.steps[index].depends_on:return False
        w.steps[index].depends_on=kept;return True
    variant('fix_dependencies',redepend)

    # Re-link a physical argument that denotes no live object onto one that does exist.
    from .identity import surface
    live=sorted(surface(o) for o in before['exists'] if ground(surface(o),source).start>=0)
    for j,arg in enumerate(step.operator.arguments):
        if not physical(arg) or norm(arg.text) in before['exists']:continue
        words={w for w in norm(arg.text).split() if len(w)>2}
        for target in live:
            if not (words&set(norm(target).split()) or norm(target) in norm(arg.text) or norm(arg.text) in norm(target)):continue
            def relink(w,j=j,target=target):
                span=ground(target,source,step.evidence['trigger'].start)
                w.steps[index].operator.arguments[j].text=source[span.start:span.end]
                _rebuild(w,index);return True
            variant(f'relink_argument.{j}',relink)

    # Re-link a parameter mention onto a numeric span actually written on this line.
    if 'PARAMETER_VIOLATION' in codes:
        anchor=step.evidence['trigger'].start
        line_start=source.rfind('\n',0,max(anchor,0))+1
        line_end=source.find('\n',max(anchor,0))
        line=source[line_start:line_end if line_end>=0 else len(source)]
        for j,arg in enumerate(step.operator.arguments):
            if arg.type not in ('setting','measurement'):continue
            for match in sorted({m.group(0).strip() for m in re.finditer(
                    r'[-+]?[\d.,]+\s*(?:rpm|rcf|[x脳]\s*g|min(?:utes?)?|sec(?:onds?)?|s|h(?:ours?)?|hrs?|[u碌渭]l|ml|掳\s*[cf])',line,re.I)}):
                if norm(match)==norm(arg.text):continue
                def reparam(w,j=j,match=match):
                    w.steps[index].operator.arguments[j].text=match;_rebuild(w,index);return True
                variant(f'relink_parameter.{j}',reparam)

    # Correct the action type, but only to one the training corpus attests for this trigger,
    # and never to one whose projection would create objects the original did not create.
    for action in sorted((lexicon or {}).get(norm(step.operator.trigger),())):
        if action==step.operator.action:continue
        current=semantic_template(step.operator.action,step.operator.arguments)
        proposed=semantic_template(action,step.operator.arguments)
        if not set(map(norm,proposed['add']))<=set(map(norm,current['add'])):continue
        def reaction(w,action=action):
            op=w.steps[index].operator;op.action=action
            for k,v in semantic_template(action,op.arguments).items():setattr(op,k,v)
            return True
        variant(f'correct_action.{action}',reaction)

    # Drop a non-input argument whose text is attested nowhere in the source. Removing an
    # invented product never adds inventory; the operated input (role a) is never dropped.
    if drop_arguments and 'UNSUPPORTED_HALLUCINATION' in codes:
        for j,arg in enumerate(step.operator.arguments):
            if arg.role=='a' or ground(arg.text,source).start>=0:continue
            def unclaim(w,j=j):
                op=w.steps[index].operator;op.arguments.pop(j)
                for k,v in semantic_template(op.action,op.arguments).items():setattr(op,k,v)
                _rebuild(w,index);return True
            variant(f'drop_unsupported_argument.{j}',unclaim)

    if mentions is not None:
        settings,products=mentions['settings'],mentions['products']
        initial={norm(surface(x)) for x in workflow.initial}
        produced={norm(x) for s in workflow.steps for x in s.operator.add}
        missing={norm(v.get('mention',v['actual'])) for v in result['violations'] if v['step']==step_id
                 and v['code']=='STATE_VIOLATION' and v['field']=='pre' and isinstance(v['actual'],str)}
        # A site/usage object that no inventory entry or step can supply never occurs in the
        # training gold. Read it as a setting where the corpus attests that, else drop it unless
        # the corpus commonly produces it, which points to a missing producer instead.
        for j,arg in enumerate(step.operator.arguments):
            text=norm(arg.text)
            if not physical(arg) or arg.role not in ('site','usage') or text not in missing or text in initial|produced:continue
            if text not in settings and text in products:continue
            def demote(w,j=j,as_setting=text in settings):
                op=w.steps[index].operator
                if as_setting:op.arguments[j]=Argument(role='setting',text=op.arguments[j].text,type='setting')
                else:op.arguments.pop(j)
                for k,v in semantic_template(op.action,op.arguments).items():setattr(op,k,v)
                _rebuild(w,index);return True
            variant(f'{"setting" if text in settings else "drop"}_unavailable_argument.{j}',demote)
        # Rename an invented product to an object a later step needs and the source names.
        after={s.id for s in workflow.steps[index+1:]}
        later={norm(v.get('mention',v['actual'])) for v in result['violations'] if v['code']=='STATE_VIOLATION' and v['field']=='pre'
               and isinstance(v['actual'],str) and v['step'] in after}
        needed=sorted(o for o in later if o not in initial and ground(o,source).start>=0)[:3]
        for j,arg in enumerate(step.operator.arguments):
            if arg.role not in ('b','c') or not physical(arg) or ground(arg.text,source).start>=0:continue
            for target in needed:
                def rename(w,j=j,target=target):
                    span=ground(target,source,step.evidence['trigger'].start)
                    op=w.steps[index].operator;op.arguments[j].text=source[span.start:span.end]
                    for k,v in semantic_template(op.action,op.arguments).items():setattr(op,k,v)
                    _rebuild(w,index);return True
                variant(f'rename_unsupported_product.{j}',rename)

    # Delete a node only when its trigger is not attested anywhere in the source.
    if ground(step.operator.trigger,source).start<0:
        def drop(w):
            w.steps.pop(index)
            for s in w.steps:s.depends_on=[d for d in s.depends_on if d!=step_id]
            return True
        variant('drop_unsupported_node',drop)
    return out

def settle_unavailable(workflow,rules=(),mentions=None):
    """Apply the unavailable site/usage operation wherever it fires, so producer feedback is
    never requested for an object no step could supply."""
    if mentions is None:return workflow
    for _ in range(4*len(workflow.steps)):
        result=verify(workflow,rules)
        failing=sorted({v['step'] for v in result['violations'] if v['step'] is not None},key=lambda s:(len(s),s))
        patched=next((w for step_id in failing for label,w in _patches(workflow,step_id,result,mentions=mentions)
                      if '_unavailable_argument.' in label),None)
        if patched is None:return workflow
        workflow=patched
    return workflow

def minimal_repair(workflow,rules=(),proposals=(),max_candidates=512,beam_width=4,
                   max_rounds=16,lexicon=None,weights=None,drop_arguments=True,mentions=None):
    """Bounded best-first search over Section 3.3 patches; minimum within the tested set.

    drop_arguments=False restricts the search to the six Section 3.3 operations.
    mentions=None leaves out the unavailable-object and invented-product operations.
    """
    original=verify(workflow,rules)
    if original['pass']:return {'status':'PASS','workflow':workflow,'cost':0,'tested':1,'operations':[]}
    tested=1;feasible=[];applied={}

    def consider(candidate,history):
        nonlocal tested
        if tested>=max_candidates or not _admissible(workflow,candidate):return None
        tested+=1
        outcome=verify(candidate,rules)
        if outcome['pass']:
            feasible.append((edit_cost(workflow,candidate),candidate.model_dump_json(),candidate,history))
            return None
        return (repair_objective(workflow,candidate,outcome['violations'],weights),
                candidate.model_dump_json(),candidate,outcome,history)

    seen={workflow.model_dump_json()}
    frontier=[]
    for proposal in proposals:
        key=proposal.model_dump_json()
        if key in seen:continue
        seen.add(key)
        node=consider(proposal,['model_proposal'])
        if node:frontier.append(node)

    # One composite pass first: the unambiguous patches applied to every failing step at once.
    composite=workflow.model_copy(deep=True);labels=[]
    for step_id in sorted({v['step'] for v in original['violations']} ,key=lambda s:(len(s),s)):
        if step_id is None or not any(s.id==step_id for s in composite.steps):continue
        current=verify(composite,rules)
        for label,patched in _patches(composite,step_id,current,lexicon,drop_arguments,mentions):
            if label in ('reground_arguments','template_effects','fix_dependencies'):
                composite=patched;labels.append(f'{step_id}:{label}')
    key=composite.model_dump_json()
    if key not in seen:
        seen.add(key)
        node=consider(composite,labels)
        if node:frontier.append(node)
    frontier.append((repair_objective(workflow,workflow,original['violations'],weights),
                     workflow.model_dump_json(),workflow,original,[]))

    for _ in range(max_rounds):
        if not frontier or tested>=max_candidates:break
        frontier.sort(key=lambda node:(node[0],node[1]))
        expanded=[]
        for _,_,candidate,outcome,history in frontier[:beam_width]:
            failing=sorted({v['step'] for v in outcome['violations'] if v['step'] is not None},key=lambda s:(len(s),s))
            for step_id in failing[:2]:
                if not any(s.id==step_id for s in candidate.steps):continue
                for label,patched in _patches(candidate,step_id,outcome,lexicon,drop_arguments,mentions):
                    key=patched.model_dump_json()
                    if key in seen:continue
                    seen.add(key)
                    node=consider(patched,[*history,f'{step_id}:{label}'])
                    if node:expanded.append(node)
                    if tested>=max_candidates:break
        frontier=expanded
    for _,_,candidate,history in feasible:applied[candidate.model_dump_json()]=history
    if not feasible:
        return {'status':'REVIEW','workflow':workflow,'cost':None,'tested':tested,
                'violations':original['violations'],'operations':[]}
    cost,key,best,history=min(feasible,key=lambda item:(item[0],item[1]))
    return {'status':'REPAIRED','workflow':best,'cost':cost,'tested':tested,
            'operations':history,
            'objective':repair_objective(workflow,best,[],weights)}
