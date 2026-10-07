"""H1–H4 measurements that do not retrain the 3B compiler.

H1 compares TST-IR with a PEG-style step graph. H2 compares ontology-guided
induction with Declare action-name constraints. H3 applies the induced library
to the saved v4 compiler outputs. H4 reports edit cost only on documents where
regeneration actually ran. Saved v4 predictions are not overwritten.
"""
import json,sys
from collections import Counter,defaultdict
from pathlib import Path
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.baselines import (mine_declare,precondition_recovery,state_error_caught,
                                   step_graph_violations)
from sop2program.induce import mine
from sop2program.ir import Workflow
from sop2program.metrics import assemble_workflows,evaluate_predictions
from sop2program.verify import minimal_repair,source_deviation,verify
from scripts.symbolic_experiments import load,mutations

STATE_KINDS={'omitted_add','omitted_delete','omitted_modify','missing_initial_object','unsupported_effect'}
SEMANTIC={'STATE_VIOLATION','INVARIANT_VIOLATION','PARAMETER_VIOLATION','DEPENDENCY_VIOLATION','RESOURCE_VIOLATION','TYPE_VIOLATION'}


def mcnemar(left,right):
    only_left=sum(a and not b for a,b in zip(left,right))
    only_right=sum(b and not a for a,b in zip(left,right))
    discordant=only_left+only_right
    return {'left_only':only_left,'right_only':only_right,'pairs':len(left),
            'p_value':float(binomtest(only_left,discordant,.5).pvalue) if discordant else None}


def h1(test):
    recovery=Counter();consistency={'tst_ir_pass':0,'step_graph_pass':0,'documents':len(test)}
    for workflow in test:
        scores=precondition_recovery(workflow)
        for key in ('template_preconditions','tst_ir_recovered','step_graph_recovered'):
            recovery[key]+=scores[key]
        outcome=state_error_caught(workflow)
        consistency['tst_ir_pass']+=not outcome['tst_ir']
        consistency['step_graph_pass']+=not outcome['step_graph']
    clean=[w for w in test if verify(w)['pass']]
    cases=[(w.id,*item) for w in clean for item in mutations(w)]
    rows=[]
    for doc,kind,step,workflow in cases:
        caught=state_error_caught(workflow)
        rows.append({'doc':doc,'kind':kind,'step':step,
                     'tst_ir':caught['tst_ir_detected'] and any(v['step']==step for v in caught['tst_ir']),
                     'step_graph':caught['step_graph_detected'] and any(v['step']==step for v in caught['step_graph']),
                     'tst_ir_detected':caught['tst_ir_detected'],
                     'step_graph_detected':caught['step_graph_detected']})
    state=[r for r in rows if r['kind'] in STATE_KINDS]
    return {
        'scope':'Test-set weak projections. The step graph checks order and dependency ids only. '
                'Precondition recall is explicit operator.pre versus objects already produced. '
                'Localization uses the same single-factor mutations as the symbolic verifier.',
        'precondition_recovery':dict(recovery)|{
            'tst_ir_recall':recovery['tst_ir_recovered']/recovery['template_preconditions'],
            'step_graph_recall':recovery['step_graph_recovered']/recovery['template_preconditions']},
        'reference_consistency':consistency,
        'state_error_localization':{
            'cases':len(state),
            'tst_ir_localized':sum(r['tst_ir'] for r in state),
            'step_graph_localized':sum(r['step_graph'] for r in state),
            'paired_test':mcnemar([r['tst_ir'] for r in state],[r['step_graph'] for r in state])},
        'all_mutation_detection':{
            'cases':len(rows),
            'tst_ir':sum(r['tst_ir_detected'] for r in rows),
            'step_graph':sum(r['step_graph_detected'] for r in rows)}}


def h2(train,dev):
    ontology,_=mine(train,holdout=dev)
    declare,_=mine_declare(train,holdout=dev)
    def families(rules):
        return dict(Counter(r['family'] for r in rules))
    return {
        'scope':'Same training traces and development confirmation. Declare uses action names only. '
                'Ontology induction may admit state, resource, parameter, effect, and type constraints.',
        'ontology_rules':len(ontology),'ontology_families':families(ontology),
        'declare_rules':len(declare),'declare_families':families(declare),
        'declare_relations':dict(Counter(r['relation'] for r in declare)),
        'state_resource_parameter':{
            'ontology':sum(r['family'] in {'pre','resource','parameter'} for r in ontology),
            'declare':sum(r['family'] in {'pre','resource','parameter'} for r in declare)}}


def h3(workflows,rules):
    modes={
        'schema':dict(state_checks=False,evidence_checks=True,invariants=False),
        'induced_library':dict(state_checks=False,evidence_checks=True,invariants=True),
        'state_checks':dict(state_checks=True,evidence_checks=True,invariants=False),
        'state_and_library':dict(state_checks=True,evidence_checks=True,invariants=True)}
    summary={}
    for name,flags in modes.items():
        outcomes=[verify(w,rules,**flags) for w in workflows]
        codes=Counter(v['code'] for outcome in outcomes for v in outcome['violations'])
        summary[name]={
            'documents':len(workflows),
            'accepted':sum(outcome['pass'] for outcome in outcomes),
            'semantic_violations':sum(codes[code] for code in SEMANTIC),
            'violation_counts':dict(codes)}
    schema=summary['schema']['semantic_violations']
    library=summary['induced_library']['semantic_violations']
    summary['library_minus_schema_semantic_violations']=library-schema
    summary['larger_unconstrained_models']='not run; 7B and 14B weights are not in this workspace'
    return summary


def h4(workflows,rules,lexicon,saved):
    fresh=[]
    for workflow in workflows:
        if workflow.metadata.get('schema_complete') and not verify(workflow,rules)['pass']:
            result=minimal_repair(workflow,rules,lexicon=lexicon)
            fresh.append({'doc':workflow.id,'status':result['status'],'cost':result['cost'],
                          'deviation':source_deviation(result['workflow'])})
        elif not workflow.metadata.get('schema_complete'):
            fresh.append({'doc':workflow.id,'status':'REVIEW','cost':None,'deviation':None,
                          'reason':'anchor still missing after whitespace-tolerant matching'})
    executed=[row for row in saved if 'skip_reason' not in row['regeneration']]
    common=[row for row in executed if row['local']['status']==row['regeneration']['status']=='REPAIRED']
    costs=[row['local']['cost']-row['regeneration']['cost'] for row in common]
    deviations=[row['local']['deviation']-row['regeneration']['deviation'] for row in common]
    return {
        'scope':'Regeneration results are the saved v4 run. Cost and source deviation use only documents '
                'where that run executed both arms. Documents regeneration skipped stay out of the cost test. '
                'The local-repair column below is recomputed on whitespace-tolerant workflows.',
        'saved_regeneration_executed':len(executed),
        'saved_structural_skips':sum('skip_reason' in row['regeneration'] for row in saved),
        'shared_successes':len(common),
        'mean_local_minus_regeneration_cost':sum(costs)/len(costs) if costs else None,
        'mean_local_minus_regeneration_deviation':sum(deviations)/len(deviations) if deviations else None,
        'rescored_local_repair':{
            'failing_documents':len(fresh),
            'repaired':sum(row['status']=='REPAIRED' for row in fresh),
            'review':sum(row['status']=='REVIEW' for row in fresh)},
        'rescored_failures':fresh}


def main():
    train,dev,test=load('train'),load('dev'),load('test')
    rules=json.loads((ROOT/'results/symbolic/full_rules.json').read_text(encoding='utf-8'))
    lexicon=json.loads((ROOT/'results/symbolic/trigger_lexicon.json').read_text(encoding='utf-8'))['lexicon']
    docs={w.id:w for w in test}
    rows=[json.loads(line) for line in (ROOT/'data/processed/test.jsonl').read_text(encoding='utf-8').splitlines()]
    out=ROOT/'results/hypothesis_opt'
    out.mkdir(parents=True,exist_ok=True)
    scores={}
    lora_workflows=None
    for method in ('base','lora','no_state','unconstrained'):
        source=ROOT/'results/evaluation_v4'/method
        predictions=[json.loads(line) for line in (source/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        directory=out/method
        directory.mkdir(parents=True,exist_ok=True)
        workflows=assemble_workflows(rows,predictions,docs)
        if method=='lora':lora_workflows=workflows
        (directory/'workflows.json').write_text(
            json.dumps([w.model_dump() for w in workflows],ensure_ascii=False),encoding='utf-8')
        summary,details=evaluate_predictions(rows,predictions,docs,rules,lexicon)
        summary['run']=f'results/hypothesis_opt/{method}'
        summary['anchor_rule']='whitespace-tolerant trigger match'
        (directory/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (directory/'audit.json').write_text(json.dumps(details,ensure_ascii=False),encoding='utf-8')
        scores[method]={'full_workflow_pass_rate':summary['full_workflow_pass_rate'],
                        'complete_documents':summary['complete_documents'],
                        'anchor_match_rate':summary['anchor_match_rate'],
                        'weak_transition_exact_rate':summary['weak_transition_exact_rate'],
                        'failed_or_missing_step_rate':summary['failed_or_missing_step_rate']}
        print(method,scores[method],flush=True)
    saved=json.loads((ROOT/'results/hypotheses_v4.json').read_text(encoding='utf-8'))['H4']['documents']
    payload={'scores':scores,'H1':h1(test),'H2':h2(train,dev),'H3':h3(lora_workflows,rules),
             'H4':h4(lora_workflows,rules,lexicon,saved)}
    (out/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({key:payload[key] for key in ('H1','H2','H3','H4') if key!='H4'},indent=2)[:4000])


if __name__=='__main__':main()
