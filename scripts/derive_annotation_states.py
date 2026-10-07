"""Compare text-keyed weak state labels with labels on X-WLP annotated object identity.

Gold: identity agreement between same-text and same-chain mention pairs, initial
inventory size, and whether the gold workflow still executes on chains.
Predictions: lifecycle verdicts (state checks only; evidence and invariants off on
both sides) and state-label scores under text identity vs chain identity.
"""
import argparse,json,sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.annotation import GoldIdentity
from sop2program.ir import Workflow, norm, physical
from sop2program.metrics import assemble_workflows, effects, prf
from sop2program.verify import verify

RUNS=['results/evaluation_v6_pinned_dev/lora','results/evaluation_qwen14_dev/lora',
      'results/evaluation_v6_pinned/lora','results/evaluation_qwen14_test/lora']


def lifecycle(workflow,missing=None):
    result=verify(workflow,evidence_checks=False,invariants=False)
    if missing is not None:
        for v in result['violations']:
            if v['code']=='STATE_VIOLATION' and v['field']=='pre':missing.setdefault(v['step'],set()).add(v['actual'])
    return result['pass'],{(v['step'],v['field']) for v in result['violations'] if v['code']=='STATE_VIOLATION'}


def gold_report(split):
    docs=[Workflow.model_validate(w) for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text(encoding='utf-8'))]
    pairs=Counter();inventory=Counter();spurious=[];failing=[];deps=Counter();identities={}
    for w in docs:
        g=GoldIdentity(w,ROOT/w.metadata['peg_file']);identities[w.id]=g
        mentions=[(norm(a.text),c) for step,local in zip(w.steps,g.mentions) for a,c in zip(step.operator.arguments,local) if c is not None]
        for (t1,c1),(t2,c2) in combinations(mentions,2):
            pairs[(t1==t2,c1==c2)]+=1
        chained=g.gold()
        inventory['text']+=len(w.initial);inventory['chain']+=len(chained.initial)
        # A stocked text whose chain was already present under another name is a split identity.
        for t in w.initial:
            i=next(k for k,s in enumerate(w.steps) if any(norm(a.text)==norm(t) for a in s.operator.arguments if physical(a)))
            c=g.resolve(t,i)
            chain=next(k for k,v in g.name.items() if v==c)
            if g.first_step[chain]<i:spurious.append({'doc':w.id,'stocked':t,'at':f's{i}','chain':c})
        ok,bad=lifecycle(chained)
        if not ok:failing.append({'doc':w.id,'violations':sorted(bad)})
        text_edges={(d,s.id) for s in w.steps for d in s.depends_on};chain_edges={(d,s.id) for s in chained.steps for d in s.depends_on}
        deps['text']+=len(text_edges);deps['chain']+=len(chain_edges);deps['shared']+=len(text_edges&chain_edges)
    same_text=pairs[(True,True)]+pairs[(True,False)];same_chain=pairs[(True,True)]+pairs[(False,True)]
    return identities,{
        'documents':len(docs),
        'mention_pairs':{'same_text_and_chain':pairs[(True,True)],'same_text_different_chain':pairs[(True,False)],
                         'different_text_same_chain':pairs[(False,True)],
                         'text_identity_precision':pairs[(True,True)]/same_text if same_text else None,
                         'text_identity_recall':pairs[(True,True)]/same_chain if same_chain else None},
        'initial_inventory':{'text_items':inventory['text'],'chain_items':inventory['chain'],
                             'stocked_continuations':len(spurious),'examples':spurious[:15]},
        'dependency_edges':dict(deps),
        'gold_not_executable_on_chains':{'documents':len(failing),'details':failing[:15]}}


def prediction_report(run,identities):
    config=json.loads((ROOT/run/'run_config.json').read_text(encoding='utf-8'));split=config['split']
    docs={w['id']:Workflow.model_validate(w) for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text(encoding='utf-8'))}
    rows=[json.loads(l) for l in (ROOT/f'data/processed/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    preds=[json.loads(l) for l in (ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    workflows=assemble_workflows(rows,preds,docs)
    verdict=Counter();steps=Counter();examples=[];scores=Counter();resolution=Counter();causes=Counter()
    for w in workflows:
        if not w.metadata['schema_complete']:continue
        g=identities[w.id];gold_chain=g.gold()
        renamed=g.rename(w,gold_chain.initial);resolution.update(g.resolved)
        missing={};a,va=lifecycle(w);b,vb=lifecycle(renamed,missing)
        verdict[(a,b)]+=1
        produced={norm(t) for s in gold_chain.steps for t in s.operator.add}
        for step_id,field in vb-va:
            if field!='pre':continue
            for obj in missing.get(step_id,()):
                causes[('gold_product' if norm(obj) in produced else 'gold_stock' if norm(obj) in map(norm,gold_chain.initial)
                        else 'unattested')]+=1
        steps['both']+=len(va&vb);steps['text_only']+=len(va-vb);steps['chain_only']+=len(vb-va)
        for kind,diff in (('text_only',va-vb),('chain_only',vb-va)):
            for step_id,field in sorted(diff)[:2]:
                i=int(step_id[1:])
                examples.append({'doc':w.id,'step':step_id,'field':field,'kind':kind,
                                 'predicted':w.steps[i].operator.model_dump(exclude={'trigger'}),
                                 'gold':docs[w.id].steps[i].operator.model_dump(exclude={'trigger','modify'})})
        for i,(p,q) in enumerate(zip(w.steps,renamed.steps)):
            t=docs[w.id].steps[i].operator;tc=gold_chain.steps[i].operator
            scores['steps']+=1
            scores['pre_text']+=set(map(norm,p.operator.pre))==set(map(norm,t.pre))
            scores['pre_chain']+=set(map(norm,q.operator.pre))==set(map(norm,tc.pre))
            scores['effect_text']+=effects(p.operator)==effects(t)
            scores['effect_chain']+=effects(q.operator)==effects(tc)
        for x,gold_w,key in ((w,docs[w.id],'text'),(renamed,gold_chain,'chain')):
            pe={(d,s.id) for s in x.steps for d in s.depends_on};ge={(d,s.id) for s in gold_w.steps for d in s.depends_on}
            scores[f'dep_tp_{key}']+=len(pe&ge);scores[f'dep_pred_{key}']+=len(pe);scores[f'dep_gold_{key}']+=len(ge)
    n=scores['steps']
    return {'run':run,'split':split,'complete_documents':sum(verdict.values()),
            'lifecycle_pass':{'text':sum(v for (a,_),v in verdict.items() if a),'chain':sum(v for (_,b),v in verdict.items() if b),
                              'both':verdict[(True,True)],'text_only':verdict[(True,False)],'chain_only':verdict[(False,True)],
                              'neither':verdict[(False,False)]},
            'flagged_step_fields':dict(steps),
            'predicted_object_strings':dict(resolution),
            'chain_only_missing_preconditions':dict(causes),
            'state_label_scores_on_complete_documents':{
                'steps':n,'precondition_exact_text':scores['pre_text']/n,'precondition_exact_chain':scores['pre_chain']/n,
                'effect_exact_text':scores['effect_text']/n,'effect_exact_chain':scores['effect_chain']/n,
                'dependency_f1_text':prf(scores['dep_tp_text'],scores['dep_pred_text'],scores['dep_gold_text'])['f1'],
                'dependency_f1_chain':prf(scores['dep_tp_chain'],scores['dep_pred_chain'],scores['dep_gold_chain'])['f1']},
            'disagreement_examples':examples[:20]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/annotation_states/report.json')
    args=p.parse_args()
    report={'identity':'X-WLP entity ids joined by co_ref_of events','gold':{},'predictions':[]}
    identities={}
    for split in ('train','dev','test'):
        ids,report['gold'][split]=gold_report(split);identities.update(ids)
        print(split,json.dumps({k:v for k,v in report['gold'][split].items() if k!='initial_inventory'} |
                               {'inventory':{k:v for k,v in report['gold'][split]['initial_inventory'].items() if k!='examples'}},ensure_ascii=False)[:900])
    for run in RUNS:
        r=prediction_report(run,identities);report['predictions'].append(r)
        print(run,json.dumps({k:r[k] for k in ('complete_documents','lifecycle_pass','flagged_step_fields','predicted_object_strings','chain_only_missing_preconditions','state_label_scores_on_complete_documents')}))
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
