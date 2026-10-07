import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sop2program.ir import *
from sop2program.induce import mine,numeric_parameters,rule_applies
from sop2program.verify import verify,minimal_repair,edit_cost

def load(split):return [Workflow.model_validate(w) for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text())]

def rescale(text,factor=100):
    """Push the first recognised quantity two orders of magnitude out, keeping its unit."""
    for value,unit in numeric_parameters(text):
        for token in (f'{value:g}',f'{value:,.0f}',str(int(value)) if value==int(value) else ''):
            if token and token in text:return text.replace(token,f'{value*factor:g}',1)
    return None

def mutations(w):
    result=[];seen=set()
    def append(kind,step,workflow):
        if kind not in seen:result.append((kind,step,workflow));seen.add(kind)
    for i,s in enumerate(w.steps):
        if s.operator.arguments:
            x=w.model_copy(deep=True);x.steps[i].operator.arguments[0].text='UNSUPPORTED_OBJECT_73'
            append('argument_hallucination',s.id,x)
            x=w.model_copy(deep=True);x.steps[i].evidence['trigger'].start=-1
            append('invalid_evidence',s.id,x)
            phys=next((j for j,a in enumerate(s.operator.arguments) if physical(a) and a.role=='a'),None)
            if phys is not None and s.operator.action!='CREATE':
                x=w.model_copy(deep=True);x.steps[i].operator.arguments[phys].type='setting'
                append('wrong_type',s.id,x)
                obj=norm(s.operator.arguments[phys].text)
                if obj in w.initial and not any(obj in map(norm,t.operator.add) for t in w.steps[:i]):
                    x=w.model_copy(deep=True);x.initial=[a for a in x.initial if norm(a)!=obj]
                    append('missing_initial_object',s.id,x)
            if i:
                x=w.model_copy(deep=True);x.steps[i].depends_on=['nonexistent_step']
                append('missing_dependency',s.id,x)
            for field in ('add','delete','modify'):
                if getattr(s.operator,field):
                    x=w.model_copy(deep=True);setattr(x.steps[i].operator,field,[])
                    append('omitted_'+field,s.id,x)
            if s.operator.modify:
                x=w.model_copy(deep=True);x.steps[i].operator.modify[0].value='UNSUPPORTED_EFFECT_73'
                append('unsupported_effect',s.id,x)
            # Sensitivity of quantity invariants is reported against perturbation magnitude,
            # so a single arbitrary factor cannot stand in for the whole family.
            for factor in (10,100,1000):
                for j,a in enumerate(s.operator.arguments):
                    scaled=rescale(a.text,factor)
                    if scaled is None:continue
                    # Attested in the source, so only the parameter invariant can object.
                    x=w.model_copy(deep=True);start=len(x.source)+1;x.source=f'{x.source} {scaled}'
                    x.steps[i].operator.arguments[j].text=scaled
                    x.steps[i].evidence[f'arguments.{j}']=Evidence(start=start,end=start+len(scaled),text=scaled)
                    append(f'out_of_range_parameter_x{factor}',s.id,x)
                    break
    return result

def main():
    train=load('train');dev=load('dev');test=load('test');out=ROOT/'results/symbolic';out.mkdir(parents=True,exist_ok=True)
    libraries={};summary={}
    # Candidates are mined on train and confirmed on dev; test is never used for admission.
    for name,kwargs in [('full',{}),('no_ontology',{'ontology':False}),('temporal_only',{'temporal_only':True}),
                        ('no_holdout',{'holdout':()})]:
        rules,audit=mine(train,**{'holdout':dev}|kwargs);libraries[name]=rules
        (out/f'{name}_rules.json').write_text(json.dumps(rules,indent=2,ensure_ascii=False));(out/f'{name}_candidates.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
        summary[name]={'rules':len(rules),'families':{f:sum(r['family']==f for r in rules) for f in sorted({r['family'] for r in rules})},
                       'admission':'train posterior + dev confirmation' if kwargs.get('holdout',dev) else 'train posterior only'}
    # An empty family is a result, not an omission: record how close its best candidate came.
    audit=json.loads((out/'full_candidates.json').read_text(encoding='utf-8'))
    summary['candidate_families']={}
    for family in sorted({c['family'] for c in audit}):
        group=[c for c in audit if c['family']==family]
        best=max(group,key=lambda c:c['posterior_lower_95'])
        summary['candidate_families'][family]={
            'candidates':len(group),'admitted':sum(c['admitted'] for c in group),
            'best_posterior_lower_95':best['posterior_lower_95'],
            'best_candidate':f"{best['action']} / {best['type']} / {best['value']}"}
    clean=[w for w in test if verify(w)['pass']]
    cases=[(w.id,*m) for w in clean for m in mutations(w)]
    summary['corpus']={'train_documents':len(train),'test_documents':len(test),'weak_test_clean':len(clean),'weak_test_unresolved':len(test)-len(clean),'mutation_cases':len(cases),'scope':'Mutations of verifier-clean template projections, not independent human state gold.'}
    details=[]
    for name,rules,state_checks,evidence_checks in [('schema_only',[],False,False),('state_evidence',[],True,True),('full',libraries['full'],True,True),('temporal_only',libraries['temporal_only'],False,False),('no_ontology',libraries['no_ontology'],True,True),('no_holdout',libraries['no_holdout'],True,True)]:
        clean_reject=sum(not verify(w,rules,state_checks=state_checks,evidence_checks=evidence_checks)['pass'] for w in clean)
        detected=localized=0;bykind={};start=time.perf_counter()
        for doc,kind,step,w in cases:
            r=verify(w,rules,state_checks=state_checks,evidence_checks=evidence_checks)
            hit=not r['pass'];loc=any(v['step']==step for v in r['violations'])
            detected+=hit;localized+=loc;bykind.setdefault(kind,[]).append(hit)
            details.append({'method':name,'doc':doc,'kind':kind,'injected_step':step,'detected':hit,'localized':loc,'violations':r['violations']})
        summary[name]=summary.get(name,{})|{'detected':detected,'total':len(cases),'detection_rate':detected/len(cases) if cases else None,'localized_rate':localized/len(cases) if cases else None,'clean_rejected':clean_reject,'clean_total':len(clean),'false_positive_rate':clean_reject/len(clean) if clean else None,'latency_ms_per_case':1000*(time.perf_counter()-start)/max(1,len(cases)),'by_kind':{k:sum(v)/len(v) for k,v in bykind.items()}}
    # A missed quantity mutation is either an interval that is too wide or, more often,
    # an (action, unit) pair the library never covers. Reporting one number hides which.
    covered=uncovered=0
    for doc,kind,step,w in cases:
        if not kind.startswith('out_of_range_parameter'):continue
        op=[t for t in w.steps if t.id==step][0].operator
        if any(rule_applies(r,op) for r in libraries['full'] if r['family']=='parameter'):covered+=1
        else:uncovered+=1
    summary['parameter_coverage']={'mutations':covered+uncovered,'with_applicable_invariant':covered,
                                   'without_applicable_invariant':uncovered,
                                   'note':'An (action, unit) pair needs enough training observations to admit an invariant at all.'}
    # Repair faces the same library the verifier used, otherwise a rule-only violation
    # is scored as if the workflow had been clean all along.
    repairs=[]
    for doc,kind,step,w in cases:
        r=minimal_repair(w,libraries['full'])
        repairs.append({'doc':doc,'kind':kind,'status':r['status'],'cost':r['cost'],'tested':r['tested']})
    summary['repair']={'cases':len(repairs),'repaired':sum(r['status']=='REPAIRED' for r in repairs),'review':sum(r['status']=='REVIEW' for r in repairs),'details':repairs,'candidate_scope':'source-span restoration and weak effect template patches; no LLM proposals in this symbolic experiment'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'mutation_predictions.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in details))
    print(json.dumps({k:v for k,v in summary.items() if k!='repair'},indent=2))

if __name__=='__main__':main()
