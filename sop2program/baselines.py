"""H1 step-graph baseline and H2 Declare baseline.

The step graph keeps execution order and dependency edges. It does not store
preconditions, effects, or a typed inventory. Declare constraints are
action-name precedence and response, with the same document-level admission
thresholds as ontology-guided induction.
"""
from collections import defaultdict
from hashlib import sha256

from .ir import norm, semantic_template
from .verify import verify


def step_graph_violations(workflow):
    """Order and edges only. A destroyed object is still a legal next step."""
    seen=[]
    violations=[]
    for step in workflow.steps:
        if step.id in seen:
            violations.append({'step':step.id,'code':'SCHEMA_VIOLATION','field':'id'})
        missing=[dep for dep in step.depends_on if dep not in seen]
        if missing:
            violations.append({'step':step.id,'code':'DEPENDENCY_VIOLATION','field':'depends_on','actual':missing})
        seen.append(step.id)
    return violations


def precondition_recovery(workflow):
    """Template preconditions stored on the operator, or only implied by earlier products."""
    produced=set(map(norm,workflow.initial))
    total=tst=graph=0
    for step in workflow.steps:
        pre=set(map(norm,semantic_template(step.operator.action,step.operator.arguments)['pre']))
        total+=len(pre)
        tst+=len(pre & set(map(norm,step.operator.pre)))
        graph+=len(pre & produced)
        produced|=set(map(norm,step.operator.add))
    return {'template_preconditions':total,'tst_ir_recovered':tst,'step_graph_recovered':graph,
            'tst_ir_recall':tst/total if total else None,
            'step_graph_recall':graph/total if total else None}


def state_error_caught(workflow):
    outcome=verify(workflow,state_checks=True,evidence_checks=False,invariants=False)
    tst=[v for v in outcome['violations'] if v['code']=='STATE_VIOLATION']
    graph=step_graph_violations(workflow)
    return {'tst_ir':tst,'step_graph':graph,'tst_ir_detected':bool(tst),'step_graph_detected':bool(graph)}


def _applicable_hold(actions,relation,left,right):
    if relation=='precedence':
        indexes=[i for i,action in enumerate(actions) if action==right]
        if not indexes:return None
        return all(any(actions[j]==left for j in range(i)) for i in indexes)
    if relation=='response':
        indexes=[i for i,action in enumerate(actions) if action==left]
        if not indexes:return None
        return all(any(actions[j]==right for j in range(i+1,len(actions))) for i in indexes)
    raise ValueError(relation)


def mine_declare(workflows,*,holdout=(),lower=0.8,min_docs=8,stability=0.8,min_coverage=0.05):
    """Action-name Declare constraints. No objects, states, or parameters."""
    from scipy.stats import beta
    sequences=[(w.id,[step.operator.action for step in w.steps]) for w in workflows]
    pairs=set()
    for _,actions in sequences:
        seen=set(actions)
        for left in seen:
            for right in seen:
                if left!=right:
                    pairs.add(('precedence',left,right))
                    pairs.add(('response',left,right))
    admitted=[];audit=[]
    documents=max(1,len(sequences))
    for relation,left,right in sorted(pairs):
        bydoc={}
        for doc,actions in sequences:
            held=_applicable_hold(actions,relation,left,right)
            if held is not None:bydoc[doc]=held
        if not bydoc:continue
        pos=sum(bydoc.values());neg=len(bydoc)-pos
        bound=float(beta.ppf(.05,1+pos,1+neg))
        folds=defaultdict(list)
        for doc,held in bydoc.items():
            folds[int(sha256(doc.encode()).hexdigest()[:8],16)%3].append(held)
        stab=min((sum(v)/len(v) for v in folds.values()),default=0)
        coverage=len(bydoc)/documents
        row={'relation':relation,'earlier':left,'later':right,'family':'temporal',
             'support_docs':pos,'contradiction_docs':neg,'applicable_docs':len(bydoc),
             'coverage':coverage,'posterior_lower_95':bound,'stability':stab,
             'admitted':len(bydoc)>=min_docs and coverage>=min_coverage and bound>=lower and len(folds)==3 and stab>=stability}
        audit.append(row)
        if row['admitted']:admitted.append(row)
    if holdout:
        checks=[(w.id,[step.operator.action for step in w.steps]) for w in holdout]
        kept=[]
        for row in admitted:
            rejected=[doc for doc,actions in checks
                      if _applicable_hold(actions,row['relation'],row['earlier'],row['later']) is False]
            row['holdout_rejected']=len(rejected)
            if not rejected:kept.append(row)
            else:row['admitted']=False
        admitted=kept
    return admitted,audit
