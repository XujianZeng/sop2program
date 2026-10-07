"""Training rows for producer feedback, built from the training split only.

Each producer step (SPIN, CONVERT, CREATE) whose product is used by a later step
yields one feedback row: the candidate omits that product, the feedback names the
later consumer, and the target is the unchanged reference operator. Producer rows
are also repeated once without feedback, as extra product supervision.

With --node-feedback, steps of training workflows that replay cleanly are corrupted
one at a time and paired with the verifier's own feedback for that node.
"""
import argparse,hashlib,json,random,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.ir import ACTIONS,Operator,Workflow,ground,make_step,norm,physical,semantic_template
from sop2program.paths import DATA,KNOWLEDGE
from sop2program.repair import PRODUCERS,_relink_dependencies,feedback_requests,missing_product_feedback
from sop2program.verify import verify


def products(op):
    return [i for i,a in enumerate(op.arguments) if physical(a) and
            (op.action=='CREATE' or a.role in ('b','c'))]


STOP={'the','a','an','of','to','in','into','on','with','and','or','for','from','each','all','at','by','per','then'}


def corruptions(row,target,rng,swap_rate):
    """Plausible one-node errors: a wrong action, or an argument span widened past its inventory name."""
    out=[]
    if target.action=='CREATE':
        out+=[('action',target.model_copy(update={'action':a})) for a in ('TRANSFER','MIX')]
    elif rng.random()<swap_rate:
        out.append(('action',target.model_copy(update={'action':rng.choice([a for a in ACTIONS if a!=target.action])})))
    exists=set(row['state'].get('exists',[]))
    source=row['source']
    for j,arg in enumerate(target.arguments):
        if not physical(arg) or norm(arg.text) not in exists:continue
        ev=ground(arg.text,source,row['offset'])
        if ev.start<0:continue
        before=re.search(r'([A-Za-z][A-Za-z-]+)\s+$',source[max(0,ev.start-30):ev.start])
        if not before or before.group(1).casefold() in STOP|set(norm(target.trigger).split()):continue
        wider=f'{before.group(1)} {arg.text}'
        if norm(wider) in exists:continue
        args=[a.model_copy(update={'text':wider}) if k==j else a for k,a in enumerate(target.arguments)]
        out.append(('span',target.model_copy(update={'arguments':args})))
    return out


def node_feedback_rows(rows,workflows,rules,rng,swap_rate=.15,max_span=250):
    """Feedback exactly as feedback_requests sends it for a failing node, on a corrupted reference.

    Span rows are subsampled so they do not outnumber the plain extraction rows they share a step with.
    """
    passing={doc for doc,w in workflows.items() if verify(w,rules)['pass']}
    out=[]
    for row in rows:
        if row['doc'] not in passing:continue
        target=Operator.model_validate(row['target'])
        for kind,bad in corruptions(row,target,rng,swap_rate):
            bad=bad.model_copy(update=semantic_template(bad.action,bad.arguments))
            if bad==target:continue
            w=workflows[row['doc']].model_copy(deep=True);i=row['step']
            step=make_step(bad,w.source,i,row['offset'],w.steps[:i]);step.id=w.steps[i].id
            w.steps[i]=step;_relink_dependencies(w)
            _,requests=feedback_requests(w,rules,max_producers=0)
            if len(requests)!=1 or requests[0]['index']!=i:continue
            out.append(dict(row,id=f"{row['id']}:node:{kind}:{len(out)}",
                            repair_feedback=requests[0]['row']['repair_feedback']))
    spans=[r for r in out if ':node:span:' in r['id']]
    keep={r['id'] for r in rng.sample(spans,min(max_span,len(spans)))}
    return [r for r in out if ':node:span:' not in r['id'] or r['id'] in keep]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='train_repair')
    parser.add_argument('--no-product-repeats',action='store_true')
    parser.add_argument('--node-feedback',action='store_true',help='Add corrupted-node rows with verifier feedback')
    parser.add_argument('--feedback-only',action='store_true',help='Write only feedback rows, for a repair adapter continued from a trained one')
    parser.add_argument('--mistyped-producers',action='store_true',
                        help='Also send each producer row with a non-producer action, so the target restores the action too')
    opts=parser.parse_args()
    data=DATA
    rows=[json.loads(line) for line in (data/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    workflows={w['id']:Workflow.model_validate(w) for w in json.loads((data/'train_workflows.json').read_text(encoding='utf-8'))}
    extra=[];duplicates=[];mistyped=[];mistype=random.Random(7)
    for row in rows:
        target=Operator.model_validate(row['target'])
        if target.action not in PRODUCERS:continue
        indexes=products(target)
        if not indexes:continue
        steps=workflows[row['doc']].steps
        used=False
        for index in indexes:
            obj=norm(target.arguments[index].text)
            consumer=next((s.id for s in steps[row['step']+1:] if obj in set(map(norm,s.operator.pre))),None)
            if consumer is None:continue
            args=[a for j,a in enumerate(target.arguments) if j!=index]
            candidate=Operator(action=target.action,trigger=target.trigger,arguments=args,
                               **semantic_template(target.action,args))
            extra.append(dict(row,id=f"{row['id']}:repair:{index}",
                              repair_feedback=missing_product_feedback(candidate,consumer,target.arguments[index].text)))
            if opts.mistyped_producers:
                action=mistype.choice(sorted(set(ACTIONS)-PRODUCERS))
                wrong=Operator(action=action,trigger=target.trigger,arguments=args,**semantic_template(action,args))
                mistyped.append(dict(row,id=f"{row['id']}:repair_mistyped:{index}",
                                     repair_feedback=missing_product_feedback(wrong,consumer,target.arguments[index].text)))
            used=True
        if used:duplicates.append(dict(row,id=row['id']+':product'))
    if opts.no_product_repeats:duplicates=[]
    node=[]
    if opts.node_feedback:
        rules=json.loads((KNOWLEDGE/'full_rules.json').read_text(encoding='utf-8'))
        node=node_feedback_rows(rows,workflows,rules,random.Random(42))
    out=data/f'{opts.output}.jsonl'
    written=([] if opts.feedback_only else rows)+extra+duplicates+node+mistyped
    out.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in written)+'\n',encoding='utf-8')
    manifest={'source':(data/'train.jsonl').relative_to(ROOT).as_posix(),
              'source_sha256':hashlib.sha256((data/'train.jsonl').read_bytes()).hexdigest(),
              'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
              'base_rows':0 if opts.feedback_only else len(rows),'feedback_rows':len(extra),'product_repeats':len(duplicates),
              'node_feedback_rows':len(node),'mistyped_producer_rows':len(mistyped),
              'node_feedback_kinds':{k:sum(f':node:{k}:' in r['id'] for r in node) for k in ('action','span')},
              'scope':'Training split only. Feedback rows drop one product that a later reference step consumes. '
                      'Node rows corrupt one reference step and carry the verifier feedback for that step.'}
    (data/f'{opts.output}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
