"""Pilot: lifecycle verdicts under predicted identity, scored against annotated identity.

Three verifiers see the same workflow: text identity (current), predicted identity
(sop2program/identity.py with the training-split transition lexicon), and annotated
identity (gold chains; the reference). Predicted identity receives the entity-level
inventory, the same oracle level as the text inventory it replaces.
"""
import argparse,json,re,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.annotation import GoldIdentity
from sop2program.ir import Workflow
from sop2program.metrics import assemble_workflows
from sop2program.verify import verify

LEXICON='results/symbolic/identity_lexicon.json'


def entity_inventory(gold_chain):
    return [re.sub(r' #\d+$','',name) for name in gold_chain.initial]


def predicted_identity(workflow,inventory,lexicon=LEXICON):
    meta=dict(workflow.metadata,identity='resolved')
    if lexicon:meta['identity_lexicon']=lexicon
    return workflow.model_copy(update={'initial':inventory,'metadata':meta})


def lifecycle(workflow):
    return verify(workflow,evidence_checks=False,invariants=False)


def load(split):
    return {w['id']:Workflow.model_validate(w) for w in json.loads((ROOT/f'data/processed/{split}_workflows.json').read_text(encoding='utf-8'))}


def gold_false_alarms(split,lexicon):
    docs=load(split);fails=[];steps=Counter()
    for w in docs.values():
        g=GoldIdentity(w,ROOT/w.metadata['peg_file']);inv=entity_inventory(g.gold())
        r=lifecycle(predicted_identity(w,inv,lexicon))
        steps['steps']+=len(w.steps);steps['flagged']+=len({v['step'] for v in r['violations']})
        if not r['pass']:fails.append({'doc':w.id,'violations':[(v['step'],v['field'],v['actual']) for v in r['violations']][:3]})
    return {'documents':len(docs),'failing':len(fails),'flagged_steps':steps['flagged'],'steps':steps['steps'],'examples':fails[:8]}


def predictions(run,lexicon):
    config=json.loads((ROOT/run/'run_config.json').read_text(encoding='utf-8'));split=config['split']
    docs=load(split)
    rows=[json.loads(l) for l in (ROOT/f'data/processed/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    preds=[json.loads(l) for l in (ROOT/run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    table=Counter()
    for w in assemble_workflows(rows,preds,docs):
        if not w.metadata['schema_complete']:continue
        g=GoldIdentity(docs[w.id],ROOT/docs[w.id].metadata['peg_file']);gc=g.gold()
        reference=lifecycle(g.rename(w,gc.initial))['pass']
        text=lifecycle(w)['pass']
        predicted=lifecycle(predicted_identity(w,entity_inventory(gc),lexicon))['pass']
        table[('reference_pass' if reference else 'reference_fail','text_pass' if text else 'text_fail',
               'predicted_pass' if predicted else 'predicted_fail')]+=1
    def agree(name):
        tp=sum(v for k,v in table.items() if k[0]=='reference_fail' and k[1 if name=='text' else 2].endswith('fail'))
        fp=sum(v for k,v in table.items() if k[0]=='reference_pass' and k[1 if name=='text' else 2].endswith('fail'))
        fn=sum(v for k,v in table.items() if k[0]=='reference_fail' and k[1 if name=='text' else 2].endswith('pass'))
        tn=sum(v for k,v in table.items() if k[0]=='reference_pass' and k[1 if name=='text' else 2].endswith('pass'))
        return {'missed_failures':fn,'false_alarms':fp,'caught_failures':tp,'correct_passes':tn,
                'agreement':(tp+tn)/max(1,tp+tn+fp+fn)}
    return {'run':run,'split':split,'documents':sum(table.values()),'text_identity':agree('text'),
            'predicted_identity':agree('predicted'),'table':{'|'.join(k):v for k,v in sorted(table.items())}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',default=['results/evaluation_v6_pinned_dev/lora','results/evaluation_qwen14_dev/lora'])
    p.add_argument('--splits',nargs='+',default=['train','dev'])
    p.add_argument('--no-lexicon',action='store_true')
    p.add_argument('--output',default='results/annotation_states/identity_pilot.json')
    args=p.parse_args()
    lexicon=None if args.no_lexicon else LEXICON
    report={'lexicon':lexicon,'gold':{s:gold_false_alarms(s,lexicon) for s in args.splits},
            'predictions':[predictions(r,lexicon) for r in args.runs]}
    for s,g in report['gold'].items():print('gold',s,{k:v for k,v in g.items() if k!='examples'});[print('   ',e) for e in g['examples'][:5]]
    for r in report['predictions']:print(r['run'],'text',r['text_identity'],'\n    predicted',r['predicted_identity'])
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
