"""Document-level bootstrap 95% intervals for first-pass metrics and final pass rates, plus paired differences."""
import argparse,json,random,sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.ir import Workflow
from sop2program.metrics import evaluate_predictions

RATES={'action_accuracy':'operators','operator_exact_rate':'operators','weak_precondition_exact_rate':'operators',
       'weak_effect_exact_rate':'operators'}
PRFS={'argument_f1':('argument_tuple',),'dependency_f1':('weak_dependency',),'edge_f1':('process_graph','edge_f1')}


def per_document(run,rules,lexicon):
    config=json.loads((ROOT/run/'run_config.json').read_text(encoding='utf-8'));split=config['split']
    docs={w['id']:Workflow.model_validate(w) for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text(encoding='utf-8'))}
    rows=[json.loads(l) for l in (ROOT/f'data/processed/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    preds={p['id']:p for p in map(json.loads,(ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    out={}
    for doc,w in docs.items():
        r=[x for x in rows if x['doc']==doc]
        s,_=evaluate_predictions(r,[preds[x['id']] for x in r],{doc:w},rules,lexicon)
        n=s['operators'];c={}
        for key in RATES:c[key]=(s[key]*n,n)
        for name,path in PRFS.items():
            v=s
            for k in path:v=v[k]
            c[name]=(v['tp'],v['predicted'],v['gold'])
        g=s['process_graph'];c['execution_order_accuracy']=((g['execution_order_accuracy'] or 0)*g['ordered_reference_pairs'],g['ordered_reference_pairs'])
        out[doc]=c
    return out


def value(counts,docs,name):
    if name in PRFS:
        tp=sum(counts[d][name][0] for d in docs);p=sum(counts[d][name][1] for d in docs);g=sum(counts[d][name][2] for d in docs)
        return 2*tp/(p+g) if p+g else 0.
    num=sum(counts[d][name][0] for d in docs);den=sum(counts[d][name][1] for d in docs)
    return num/den if den else 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs=2,required=True,help='Baseline and candidate evaluation directories')
    p.add_argument('--repairs',nargs=2,required=True,help='Matching producer_repair output directories')
    p.add_argument('--samples',type=int,default=10000)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    rules=json.loads((ROOT/'results/symbolic/full_rules.json').read_text(encoding='utf-8'))
    lexicon=json.loads((ROOT/'results/symbolic/trigger_lexicon.json').read_text(encoding='utf-8'))['lexicon']
    counts=[per_document(r,rules,lexicon) for r in args.runs]
    for c,rep in zip(counts,args.repairs):
        s=json.loads((ROOT/rep/'summary.json').read_text(encoding='utf-8'))
        for d in s.get('cascade',s['producer_feedback'])['details']:
            c[d['doc']]['final_pass']=(int(d['status'] in ('PASS','REPAIRED')),1)
    docs=sorted(counts[0]);assert docs==sorted(counts[1])
    names=[*RATES,*PRFS,'execution_order_accuracy','final_pass']
    rng=random.Random(0);draws=defaultdict(list)
    for _ in range(args.samples):
        sample=[rng.choice(docs) for _ in docs]
        for name in names:
            a,b=value(counts[0],sample,name),value(counts[1],sample,name)
            draws[(name,0)].append(a);draws[(name,1)].append(b);draws[(name,'diff')].append(b-a)
    def ci(xs):xs=sorted(xs);return [xs[int(.025*len(xs))],xs[int(.975*len(xs))-1]]
    report={'runs':args.runs,'repairs':args.repairs,'documents':len(docs),'samples':args.samples,'seed':0,
            'scope':'Documents resampled with replacement; paired differences are candidate minus baseline on the same resample.',
            'metrics':{}}
    print('metric'.ljust(30),'baseline [95% CI]'.ljust(26),'candidate [95% CI]'.ljust(26),'difference [95% CI]')
    for name in names:
        point=[value(c,docs,name) for c in counts]
        entry={'baseline':point[0],'baseline_ci':ci(draws[(name,0)]),'candidate':point[1],'candidate_ci':ci(draws[(name,1)]),
               'difference':point[1]-point[0],'difference_ci':ci(draws[(name,'diff')]),
               'difference_excludes_zero':not (ci(draws[(name,'diff')])[0]<=0<=ci(draws[(name,'diff')])[1])}
        report['metrics'][name]=entry
        f=lambda v,c:f'{v:.3f} [{c[0]:.3f},{c[1]:.3f}]'
        print(name.ljust(30),f(point[0],entry['baseline_ci']).ljust(26),f(point[1],entry['candidate_ci']).ljust(26),
              f(entry['difference'],entry['difference_ci']),'*' if entry['difference_excludes_zero'] else '')
    (ROOT/args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
