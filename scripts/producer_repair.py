"""Verifier feedback sent back to the step that should have produced a missing object.

Runs the v4 compiler on the whitespace-tolerant lora workflows. Two arms share the
model, rules, and round budget: feedback only on the failing node, and feedback
also routed to the nearest earlier producer. Final acceptance is the same bounded
local repair search, so neither arm can pass without the external verifier.
"""
import argparse,json,os,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from transformers import AutoTokenizer,set_seed
from sop2program.compiler import LocalCompiler
from sop2program.ir import Workflow
from sop2program.numerics import load_checked_model
from sop2program.paths import CORPUS,DATA,KNOWLEDGE
from sop2program.repair import apply_proposal,feedback_requests,replay_progress
from sop2program.verify import minimal_repair,settle_unavailable,source_deviation,verify


def run_arm(name,workflows,compiler,rules,lexicon,max_producers,max_rounds,log,drop_arguments=True,mentions=None,
            retype_producers=False):
    working={w.id:settle_unavailable(w,rules,mentions) if w.metadata.get('schema_complete') else w for w in workflows}
    active={w.id for w in workflows if w.metadata.get('schema_complete') and not verify(working[w.id],rules)['pass']}
    tried={key:set() for key in working}
    generations=0;started=time.perf_counter()
    for round_index in range(max_rounds):
        batch=[]
        for doc in sorted(active):
            outcome,requests=feedback_requests(working[doc],rules,max_producers,retype_producers)
            if outcome['pass']:continue
            fresh=[]
            for request in requests:
                key=(request['index'],request['object'],
                     working[doc].steps[request['index']].operator.model_dump_json())
                if key in tried[doc]:continue
                tried[doc].add(key);fresh.append(request)
            if fresh:batch.append((doc,outcome,fresh))
        if not batch:break
        rows=[request['row'] for _,_,requests in batch for request in requests]
        outputs=compiler.compile_batch(rows)
        generations+=len(rows)
        cursor=0;accepted=0
        for doc,outcome,requests in batch:
            best=None;best_score=replay_progress(outcome)
            for request in requests:
                op,info=outputs[cursor];cursor+=1
                candidate=apply_proposal(working[doc],request,op)
                score=replay_progress(verify(candidate,rules)) if candidate is not None else None
                log.write(json.dumps({'arm':name,'round':round_index,'doc':doc,'index':request['index'],
                                      'object':request['object'],'raw':info.get('raw'),
                                      'accepted_candidate':candidate is not None,
                                      'progress':score,'before':best_score},ensure_ascii=False)+'\n')
                if score is not None and score>best_score:best,best_score=candidate,score
            if best is None:active.discard(doc)
            else:working[doc]=best;accepted+=1
        log.flush()
        print(f'{name} round {round_index}: requests {len(rows)}, accepted {accepted}, active {len(active)}',flush=True)
    rows=[]
    for w in workflows:
        if not w.metadata.get('schema_complete'):
            rows.append({'doc':w.id,'status':'REVIEW','reason':'missing anchored node'});continue
        changed=working[w.id].model_dump_json()!=w.model_dump_json()
        result=minimal_repair(w,rules,proposals=[working[w.id]] if changed else [],lexicon=lexicon,
                              drop_arguments=drop_arguments,mentions=mentions)
        row={'doc':w.id,'status':result['status'],'cost':result['cost'],
             'model_rewrites':changed,'operations':result.get('operations',[]),
             'source_deviation':source_deviation(result['workflow']),
             'workflow':result['workflow'].model_dump()}
        if result['status']=='REVIEW':
            row['remaining_violations']=verify(working[w.id],rules)['violations'][:3]
        rows.append(row)
    counted=[r for r in rows if r['status']!='PASS']
    return {'documents':len(rows),
            'accepted':sum(r['status'] in ('PASS','REPAIRED') for r in rows),
            'already_pass':sum(r['status']=='PASS' for r in rows),
            'repaired':sum(r['status']=='REPAIRED' for r in rows),
            'review':sum(r['status']=='REVIEW' for r in rows),
            'failing_documents':len(counted),
            'generations':generations,'seconds':time.perf_counter()-started,
            'mean_repair_cost':(lambda c:sum(c)/len(c) if c else None)(
                [r['cost'] for r in rows if r['status']=='REPAIRED']),
            'details':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflows',default='results/hypothesis_opt/lora/workflows.json')
    parser.add_argument('--run',help='Evaluation directory to assemble with whitespace-tolerant anchors instead of --workflows')
    parser.add_argument('--adapter',default='results/lora3b_v4/adapter',help="'none' proposes with the untuned base model")
    parser.add_argument('--output',default='results/producer_repair')
    parser.add_argument('--max-rounds',type=int,default=12)
    parser.add_argument('--max-producers',type=int,default=2)
    parser.add_argument('--pin-trigger',action='store_true',help='Constrain the trigger field to the supplied marked span')
    parser.add_argument('--fallback-adapter',help='Second proposer, tried only on documents the first left in review')
    parser.add_argument('--no-drop-arguments',dest='drop_arguments',action='store_false',
                        help='Use only the six Section 3.3 repair operations')
    parser.add_argument('--split',choices=['test','dev'],default='test',help='Split that --run was evaluated on')
    parser.add_argument('--model',default='models/Qwen2.5-3B-Instruct',help='Local base model for --adapter')
    parser.add_argument('--fallback-model',help='Base model for --fallback-adapter; defaults to --model')
    parser.add_argument('--no-unavailable-ops',dest='unavailable_ops',action='store_false',
                        help='Leave out demoting unavailable site/usage objects and renaming invented products')
    parser.add_argument('--settings-lexicon',default=str(KNOWLEDGE/'setting_lexicon.json'))
    parser.add_argument('--no-retype-producers',dest='retype_producers',action='store_false',
                        help='Send missing-object feedback only to steps already typed SPIN/CONVERT/CREATE')
    parser.add_argument('--few-shot',type=int,default=0,help='In-context demonstrations for the proposer')
    parser.add_argument('--identity',choices=['text','oracle','resolved'],default='text',
                        help='Object identity inside the verifier; oracle uses annotated chains (upper bound only)')
    parser.add_argument('--skip-node-feedback',action='store_true',
                        help='Run only the producer-feedback arm (and the cascade built on it)')
    args=parser.parse_args()
    set_seed(42)
    rules=json.loads((KNOWLEDGE/'full_rules.json').read_text(encoding='utf-8'))
    lexicon=json.loads((KNOWLEDGE/'trigger_lexicon.json').read_text(encoding='utf-8'))['lexicon']
    mentions=None
    if args.unavailable_ops:
        payload=json.loads((ROOT/args.settings_lexicon).read_text(encoding='utf-8'))
        mentions={'settings':set(payload['settings']),'products':set(payload['products'])}
    if args.run:
        from sop2program.metrics import assemble_workflows
        data=DATA
        docs={w['id']:Workflow.model_validate(w) for w in json.loads((data/f'{args.split}_workflows.json').read_text(encoding='utf-8'))}
        rows=[json.loads(line) for line in (data/f'{args.split}.jsonl').read_text(encoding='utf-8').splitlines()]
        predictions=[json.loads(line) for line in (ROOT/args.run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        workflows=assemble_workflows(rows,predictions,docs)
        args.workflows=args.run
    else:
        workflows=[Workflow.model_validate(w) for w in json.loads((ROOT/args.workflows).read_text(encoding='utf-8'))]
    if args.identity!='text':
        import re
        from sop2program.annotation import gold_identity
        changed=[]
        for w in workflows:
            entities=gold_identity(w.id).gold().initial
            meta=dict(w.metadata,identity=args.identity)
            if args.identity=='resolved':
                entities=[re.sub(r' #\d+$','',e) for e in entities];meta['identity_lexicon']='results/symbolic/identity_lexicon.json'
            changed.append(w.model_copy(update={'initial':entities,'metadata':meta}))
        workflows=changed
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    baseline=[];deterministic=[]
    for w in workflows:
        if not w.metadata.get('schema_complete'):
            baseline.append('REVIEW');deterministic.append({'doc':w.id,'status':'REVIEW'});continue
        result=minimal_repair(w,rules,lexicon=lexicon,drop_arguments=args.drop_arguments,mentions=mentions)
        baseline.append(result['status'])
        deterministic.append({'doc':w.id,'status':result['status'],'cost':result['cost'],
                              'operations':result.get('operations',[]),'workflow':result['workflow'].model_dump()})
    demonstrations=[]
    if args.few_shot:
        from sop2program.compiler import select_demonstrations
        train_rows=[json.loads(l) for l in (CORPUS/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        demonstrations=select_demonstrations(train_rows,args.few_shot)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/args.model,local_files_only=True)
    model=load_checked_model(ROOT/args.model,None if args.adapter=='none' else ROOT/args.adapter,attention='sdpa')
    compiler=LocalCompiler(model,tokenizer,pin_trigger=args.pin_trigger,demonstrations=demonstrations)
    summary={'workflows':args.workflows,'adapter':args.adapter,'max_rounds':args.max_rounds,
             'pin_trigger':args.pin_trigger,'split':args.split,'drop_arguments':args.drop_arguments,
             'unavailable_ops':args.unavailable_ops,'retype_producers':args.retype_producers,
             'scope':'Saved v4 lora outputs. The model proposes action and arguments; effects are '
                     're-projected by template and every candidate is judged by verifier replay. '
                     'No gold labels or inventory are used.',
             'few_shot':args.few_shot,'demonstration_ids':[d['id'] for d in demonstrations],'identity':args.identity,
             'deterministic_repair':{'accepted':sum(s in ('PASS','REPAIRED') for s in baseline),
                                     'documents':len(baseline),'details':deterministic}}
    arms=['producer_feedback'] if args.skip_node_feedback else ['node_feedback','producer_feedback']
    with (out/'generations.jsonl').open('w',encoding='utf-8') as log:
        if not args.skip_node_feedback:
            summary['node_feedback']=run_arm('node_feedback',workflows,compiler,rules,lexicon,0,args.max_rounds,log,
                                             args.drop_arguments,mentions)
        summary['producer_feedback']=run_arm('producer_feedback',workflows,compiler,rules,lexicon,
                                             args.max_producers,args.max_rounds,log,args.drop_arguments,mentions,
                                             args.retype_producers)
    if args.fallback_adapter:
        import gc,torch
        del compiler,model;gc.collect();torch.cuda.empty_cache()
        fallback_model=ROOT/(args.fallback_model or args.model)
        model=load_checked_model(fallback_model,ROOT/args.fallback_adapter,attention='sdpa')
        compiler=LocalCompiler(model,AutoTokenizer.from_pretrained(fallback_model,local_files_only=True),
                               pin_trigger=args.pin_trigger)
        first={d['doc']:d for d in summary['producer_feedback']['details']}
        retry=[w for w in workflows if first[w.id]['status']=='REVIEW' and w.metadata.get('schema_complete')]
        with (out/'generations.jsonl').open('a',encoding='utf-8') as log:
            second=run_arm('fallback_producer_feedback',retry,compiler,rules,lexicon,args.max_producers,args.max_rounds,log,
                           args.drop_arguments,mentions,args.retype_producers)
        details=[{**d,'proposer':'fallback'} if d['status']!='REVIEW' else first[d['doc']]
                 for d in second['details']]
        merged={**first,**{d['doc']:d for d in details}}
        rows=[merged[w.id] for w in workflows]
        summary['fallback_adapter']=args.fallback_adapter
        summary['cascade']={'documents':len(rows),
                            'accepted':sum(r['status'] in ('PASS','REPAIRED') for r in rows),
                            'repaired':sum(r['status']=='REPAIRED' for r in rows),
                            'repaired_by_fallback':sum(r.get('proposer')=='fallback' for r in rows),
                            'review':sum(r['status']=='REVIEW' for r in rows),
                            'generations':summary['producer_feedback']['generations']+second['generations'],
                            'seconds':summary['producer_feedback']['seconds']+second['seconds'],
                            'details':rows}
        arms.append('cascade')
    # Final workflows go to their own file so the summary stays readable.
    for arm in ['deterministic_repair',*arms]:
        final={row['doc']:row['workflow'] for row in summary[arm]['details'] if 'workflow' in row}
        (out/f'{arm}_workflows.json').write_text(json.dumps(final,ensure_ascii=False),encoding='utf-8')
    for arm in ['deterministic_repair',*arms]:
        for row in summary[arm]['details']:row.pop('workflow',None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    for arm in arms:
        s=summary[arm]
        print(arm,f"accepted {s['accepted']}/{s['documents']}",f"repaired {s['repaired']}",
              f"generations {s['generations']}",f"{s['seconds']:.0f}s",flush=True)
    print('deterministic',summary['deterministic_repair'],flush=True)


if __name__=='__main__':main()
