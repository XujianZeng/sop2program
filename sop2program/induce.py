"""Algorithm B: document-level Beta admission with explicit weak-label provenance."""
from collections import defaultdict
from hashlib import sha256
import re
from .ir import Operator,norm

PARENT={'material':'physical','container':'physical','device':'physical','seal':'physical','physical':'entity',
        'setting':'abstract','measurement':'abstract','modifier':'abstract','method':'abstract','abstract':'entity'}

def ancestors(t):
    chain=[t]
    while t in PARENT:t=PARENT[t];chain.append(t)
    return chain

def target_type(op):
    return next((a.type for a in op.arguments if a.role=='a'),'unknown')

def physical_objects(op):
    from .ir import physical
    return {norm(a.text) for a in op.arguments if physical(a)}

def in_scope(rule,op):
    """Does this operator carry what the rule talks about?

    Without this the verifier charges an operator for an effect its own argument roles
    cannot produce, or for a unit it never mentions.
    """
    from .ir import semantic_template
    family,value=rule['family'],rule['value']
    if family=='pre':return any(a.role==value for a in op.arguments)
    if family=='effect':return bool(semantic_template(op.action,op.arguments)[value])
    if family=='parameter':return any(u==value[1] for a in op.arguments for _,u in numeric_parameters(a.text))
    if family=='temporal':return bool(physical_objects(op))
    return True

def rule_applies(rule,op):
    return (op.action==rule['action'] and (rule['type']=='*' or rule['type'] in ancestors(target_type(op)))
            and in_scope(rule,op))

def rule_holds(rule,op,state,history):
    family,value=rule['family'],rule['value']
    if family=='pre':return all(norm(a.text) in state.exists for a in op.arguments if a.role==value)
    if family=='effect':
        from .ir import semantic_template
        expected=semantic_template(op.action,op.arguments)[value]
        actual=getattr(op,value)
        # Admission saw template-built lists. Order and case are not a different effect.
        if value=='modify':
            def changes(items):
                return {(norm(item.object),item.attribute,norm(item.value)) for item in items}
            return changes(actual)==changes(expected)
        return set(map(norm,actual))==set(map(norm,expected))
    if family=='resource':return any(a.type=='device' for a in op.arguments)
    if family=='type':return value in ancestors(target_type(op))
    if family=='temporal':
        # A dependency invariant is about this operator's own material, not about
        # anything that happened to be done earlier in the document.
        objects=physical_objects(op)
        return any(t['committed'] and t['operator']['action']==value and
                   objects&operator_objects(t['operator']) for t in history)
    if family=='parameter':
        kind,unit,lo,hi=value
        for a in op.arguments:
            for val,u in numeric_parameters(a.text):
                if unit_kind(u)==kind and u==unit and not lo<=val<=hi:return False
        return True
    raise ValueError(family)

def operator_objects(payload):
    from .ir import Operator,physical
    op=payload if isinstance(payload,Operator) else Operator.model_validate(payload)
    return physical_objects(op)

def numeric_parameters(text):
    # Keep signs, decimals and unit identity; rpm and relative g are not interchangeable.
    pattern=r'(?<![\w.])([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*(rpm|rcf|[x×]\s*g|minutes?|min|seconds?|sec|s|hours?|hrs?|h|µl|μl|ul|ml|°\s*[cf])\b'
    aliases={'minutes':'min','minute':'min','seconds':'s','second':'s','sec':'s',
             'hours':'h','hour':'h','hrs':'h','hr':'h','µl':'ul','μl':'ul','×g':'xg'}
    values=[]
    for number,unit in re.findall(pattern,text,re.I):
        unit=''.join(unit.lower().split())
        values.append((float(number.replace(',','')),aliases.get(unit,unit)))
    return values


def tolerance_interval(values,content=.999,confidence=.95):
    """Admissible range for a parameter, not the range the training split happened to show.

    The observed min-max of n draws is a biased-inward estimate of the support: used as a
    hard constraint it rejects legal values at a rate of about 2/(n+1), so a 60 min
    incubation seen in training makes a 90 min one a violation. This returns a two-sided
    normal tolerance interval (Howe's factor) taken in log space for strictly positive
    quantities, widened to always contain the observed values. The default content is set
    so the invariant objects to implausible quantities rather than to the distribution tail.
    """
    from math import exp,log,sqrt
    from scipy.stats import chi2,norm as gaussian
    lo,hi=min(values),max(values)
    n=len(values)
    if n<3:return lo,hi
    logscale=lo>0
    sample=[log(v) for v in values] if logscale else list(values)
    mean=sum(sample)/n
    sd=sqrt(sum((v-mean)**2 for v in sample)/(n-1))
    k=gaussian.ppf((1+content)/2)*sqrt((n-1)*(1+1/n)/chi2.ppf(1-confidence,n-1))
    low,high=mean-k*sd,mean+k*sd
    if logscale:low,high=exp(low),exp(high)
    return min(lo,low),max(hi,high)

def unit_kind(unit):
    if unit in ('rpm','rcf','xg'):return 'speed'
    if unit in ('min','s','h'):return 'duration'
    if unit in ('ul','ml'):return 'volume'
    if unit in ('°c','°f'):return 'temperature'
    return 'unknown'

def replay_records(workflows,clean_only=False):
    """(document, operator, state before, committed history) triples for rule statistics."""
    from .verify import verify,State
    out=[]
    for w in workflows:
        replay=verify(w,evidence_checks=False,invariants=False)
        if clean_only and not replay['pass']:continue
        history=[]
        for t in replay['trace']:
            op=Operator.model_validate(t['operator']);state=State(set(t['before']['exists']),t['before']['attributes'])
            out.append((w.id,op,state,list(history)));history.append(t)
    return out

def holdout_rejections(rule,records):
    """Held-out documents this rule would reject, i.e. its measured false positives."""
    bad=set()
    for doc,op,state,history in records:
        if rule_applies(rule,op) and not rule_holds(rule,op,state,history):bad.add(doc)
    return sorted(bad)

def mine(workflows,*,ontology=True,temporal_only=False,lower=0.8,min_docs=8,stability=0.8,min_coverage=0.05,
         holdout=()):
    from scipy.stats import beta
    from .verify import verify,State
    records=[];candidates={};ranges=defaultdict(list)
    for w in workflows:
        replay=verify(w,evidence_checks=False,invariants=False)
        history=[]
        for t in replay['trace']:
            op=Operator.model_validate(t['operator']);state=State(set(t['before']['exists']),t['before']['attributes'])
            records.append((w.id,op,state,list(history)))
            group_types=ancestors(target_type(op)) if ontology else [target_type(op)]
            if not temporal_only:
                # Action => target type, with an untyped scope, is testable.
                # Scoping the rule by that same type would be a tautology.
                for typ in group_types:
                    if typ not in ('entity','unknown'):
                        key=(op.action,'*','type',typ)
                        candidates[key]={'action':op.action,'type':'*','family':'type','value':typ}
            # Lift to observed ancestors and re-count all matching documents before admission.
            for typ in group_types:
                objects=physical_objects(op)
                templates=[('temporal',a) for a in sorted({h['operator']['action'] for h in history
                                                           if h['committed'] and objects&operator_objects(h['operator'])})]
                if not temporal_only:
                    templates += [('pre',a.role) for a in op.arguments if norm(a.text) in state.exists]
                    templates += [('effect',field) for field in ('add','delete','modify') if getattr(op,field)]
                    templates += [('type',typ)]
                    if any(a.type=='device' for a in op.arguments):templates.append(('resource','device'))
                    # A quantity bound belongs to the action and the unit. Scoping it by the
                    # target's ontology class only splits the sample and leaves operators
                    # without an 'a' argument outside every typed rule.
                    for a in op.arguments:
                        for v,u in numeric_parameters(a.text):ranges[(op.action,'*',unit_kind(u),u)].append(v)
                for family,value in templates:
                    if family=='type':continue # Do not mine tautological scope=>scope rules.
                    key=(op.action,typ,family,str(value))
                    candidates[key]={'action':op.action,'type':typ,'family':family,'value':value}
            history.append(t)
    if not temporal_only:
        for (action,typ,kind,unit),values in ranges.items():
            # Empirical train envelope is exploratory, not a physical/safety limit.
            if len(values)>=min_docs:
                key=(action,typ,'parameter',kind+unit)
                low,high=tolerance_interval(values)
                candidates[key]={'action':action,'type':typ,'family':'parameter','value':[kind,unit,low,high],
                                 'observed':[min(values),max(values)],'observations':len(values)}
    admitted=[];audit=[]
    for key,c in sorted(candidates.items()):
        bydoc=defaultdict(list)
        for doc,op,state,history in records:
            if rule_applies(c,op):bydoc[doc].append(rule_holds(c,op,state,history))
        pos=sum(all(v) for v in bydoc.values());neg=len(bydoc)-pos
        bound=float(beta.ppf(.05,1+pos,1+neg))
        folds=defaultdict(list)
        for doc,v in bydoc.items():folds[int(sha256(doc.encode()).hexdigest()[:8],16)%3].append(all(v))
        stab=min((sum(v)/len(v) for v in folds.values()),default=0)
        coverage=len(bydoc)/max(1,len({w.id for w in workflows}))
        c.update(support_docs=pos,contradiction_docs=neg,applicable_docs=len(bydoc),coverage=coverage,posterior_lower_95=bound,stability=stab,
                 provenance='X-WLP gold arguments + template-derived weak states',id=sha256(str(key).encode()).hexdigest()[:12])
        c['code']={'pre':'STATE_VIOLATION','effect':'INVARIANT_VIOLATION','resource':'RESOURCE_VIOLATION','parameter':'PARAMETER_VIOLATION','temporal':'DEPENDENCY_VIOLATION'}.get(c['family'],'TYPE_VIOLATION')
        c['admitted']=len(bydoc)>=min_docs and coverage>=min_coverage and bound>=lower and len(folds)==3 and stab>=stability
        audit.append(c)
        if c['admitted']:admitted.append(c)
    pruned=[]
    for c in admitted:
        redundant=any(d is not c and d['action']==c['action'] and d['family']==c['family'] and d['value']==c['value'] and
                      d['type'] in ancestors(c['type'])[1:] and d['posterior_lower_95']>=c['posterior_lower_95']-.02 for d in admitted)
        if c['family']=='type':
            redundant=any(d is not c and d['action']==c['action'] and d['family']=='type' and
                          c['value'] in ancestors(d['value'])[1:] and
                          d['posterior_lower_95']>=c['posterior_lower_95']-.02 for d in admitted)
        if not redundant:pruned.append(c)
    # Admission is confirmed on a held-out split: a rule that rejects a verifier-clean
    # development document is a measured false positive, not an invariant.
    if holdout:
        checks=replay_records(holdout,clean_only=True)
        documents=len({doc for doc,_,_,_ in checks})
        kept=[]
        for c in pruned:
            rejected=holdout_rejections(c,checks)
            c['holdout_documents']=documents;c['holdout_rejected']=len(rejected)
            c['holdout_examples']=rejected[:5]
            if rejected:c['admitted']=False
            else:kept.append(c)
        pruned=kept
    return pruned,audit
