"""Final repair outcomes rechecked on X-WLP annotated object identity.

A document accepted by the text-identity verifier is counted as confirmed only if its
final workflow also executes when every object string is resolved to its gold entity
chain (state checks; evidence and invariants do not depend on identity). Fidelity is
argument-tuple F1 and chain-identity precondition agreement before and after repair.
"""
import argparse,json,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.annotation import GoldIdentity
from sop2program.ir import Workflow, norm
from sop2program.metrics import arguments, assemble_workflows, prf
from sop2program.paths import DATA
from sop2program.verify import verify


def fidelity(workflow,gold,identity,gold_chain):
    c=Counter();renamed=identity.rename(workflow,gold_chain.initial)
    for step,chained in zip(workflow.steps,renamed.steps):
        i=int(step.id[1:]);target=gold.steps[i].operator
        p,g=arguments(step.operator),arguments(target)
        c['tp']+=sum((p&g).values());c['pred']+=sum(p.values());c['gold']+=sum(g.values())
        c['pre']+=set(map(norm,chained.operator.pre))==set(map(norm,gold_chain.steps[i].operator.pre))
        c['exact']+=step.operator==target
    c['steps']=len(gold.steps)
    return c,renamed


def diagnose(result,gold_chain,final):
    """Why the first missing chain is missing, judged at the gold step that produces it."""
    missing=next((v['actual'] for v in result['violations'] if v['code']=='STATE_VIOLATION' and v['field']=='pre'
                  and isinstance(v['actual'],str)),None)
    if missing is None:return 'effect_or_modify_only'
    producer=next((i for i,s in enumerate(gold_chain.steps) if missing in map(norm,s.operator.add)),None)
    if producer is None:return 'unattested_or_stock_object'
    predicted=next((s.operator for s in final.steps if s.id==f's{producer}'),None)
    if predicted is None:return 'producer_step_missing'
    gold_op=gold_chain.steps[producer].operator
    if predicted.action!=gold_op.action:return 'producer_action_wrong'
    return 'producer_arguments_wrong'


def mcnemar(b,c):
    """Two-sided exact McNemar p-value on discordant counts."""
    from math import comb
    n=b+c
    return min(1.,2*sum(comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.


def recheck(repair_dir,arm):
    summary=json.loads((ROOT/repair_dir/'summary.json').read_text(encoding='utf-8'))
    split=summary['split'];run=summary['workflows']
    docs={w['id']:Workflow.model_validate(w) for w in json.loads((DATA/f'{split}_workflows.json').read_text(encoding='utf-8'))}
    rows=[json.loads(l) for l in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    preds=[json.loads(l) for l in (ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    first={w.id:w for w in assemble_workflows(rows,preds,docs)}
    final={k:Workflow.model_validate(v) for k,v in json.loads((ROOT/repair_dir/f'{arm}_workflows.json').read_text(encoding='utf-8')).items()}
    counts=Counter();before=Counter();after=Counter();unconfirmed=[];outcome={}
    for d in summary[arm]['details']:
        doc=d['doc'];counts['documents']+=1
        before_chain=False
        if first[doc].metadata['schema_complete']:
            g=GoldIdentity(docs[doc],ROOT/docs[doc].metadata['peg_file'])
            counts['first_pass_text']+=verify(first[doc],evidence_checks=False,invariants=False)['pass']
            before_chain=verify(g.rename(first[doc],g.gold().initial),evidence_checks=False,invariants=False)['pass']
            counts['first_pass_chain']+=before_chain
        outcome[doc]=[before_chain,False]
        if d['status'] not in ('PASS','REPAIRED'):continue
        counts['accepted_text']+=1
        g=GoldIdentity(docs[doc],ROOT/docs[doc].metadata['peg_file']);gc=g.gold()
        b,_=fidelity(first[doc],docs[doc],g,gc);a,renamed=fidelity(final[doc],docs[doc],g,gc)
        before.update(b);after.update(a)
        result=verify(renamed,evidence_checks=False,invariants=False)
        outcome[doc][1]=result['pass']
        if result['pass']:counts['accepted_chain']+=1
        else:
            cause=diagnose(result,gc,final[doc])
            counts[f'cause:{cause}']+=1
            unconfirmed.append({'doc':doc,'status':d['status'],'cause':cause,'operations':d.get('operations',[]),
                                'violations':[(v['step'],v['field'],v['actual']) for v in result['violations']][:4]})
    def side(c):
        if not c['steps']:return None
        return {'argument_f1':prf(c['tp'],c['pred'],c['gold'])['f1'],'precondition_exact_chain':c['pre']/c['steps'],
                'operator_exact':c['exact']/c['steps'],'steps':c['steps']}
    return {'repair':repair_dir,'arm':arm,'split':split,'documents':counts['documents'],
            'first_pass_lifecycle':{'text_identity':counts['first_pass_text'],'chain_identity':counts['first_pass_chain']},
            'chain_identity_by_document':outcome,
            'repair_effect_chain_identity':{'gained':sum(not b and a for b,a in outcome.values()),
                                            'lost':sum(b and not a for b,a in outcome.values()),
                                            'exact_mcnemar_p':mcnemar(sum(not b and a for b,a in outcome.values()),
                                                                      sum(b and not a for b,a in outcome.values()))},
            'accepted_text_identity':counts['accepted_text'],'confirmed_chain_identity':counts['accepted_chain'],
            'fidelity_on_accepted':{'before_repair':side(before),'after_repair':side(after)},
            'unconfirmed':unconfirmed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repairs',nargs='+',required=True)
    p.add_argument('--arm',default='cascade')
    p.add_argument('--output',default='results/annotation_states/repair_recheck.json')
    args=p.parse_args()
    report=[recheck(r,args.arm) for r in args.repairs]
    if len(report)==2 and report[0]['split']==report[1]['split']:
        x,y=report[0]['chain_identity_by_document'],report[1]['chain_identity_by_document']
        b=sum(y[d][1] and not x[d][1] for d in x);c=sum(x[d][1] and not y[d][1] for d in x)
        report.append({'paired_final_chain_identity':{'baseline':report[0]['repair'],'candidate':report[1]['repair'],
                       'candidate_only':b,'baseline_only':c,'exact_mcnemar_p':mcnemar(b,c)}})
        print(report[-1])
    for r in report:
        if 'repair' not in r:continue
        print(r['repair'],r['repair_effect_chain_identity'],'first pass',r['first_pass_lifecycle'],f"text {r['accepted_text_identity']}/{r['documents']}",f"chain {r['confirmed_chain_identity']}/{r['documents']}",
              json.dumps(r['fidelity_on_accepted']))
        for u in r['unconfirmed']:print('   ',u['doc'],u['status'],u['violations'][:2])
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
