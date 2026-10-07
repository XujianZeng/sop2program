import json,hashlib,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sop2program.ir import *
from sop2program.verify import verify

ROOT=Path(__file__).resolve().parents[1]
OP_MAP={'create_op':'CREATE','transfer_op':'TRANSFER','destroy_op':'DESTROY','convert_op':'CONVERT','temp_treat_op':'TEMP_TREAT','spin_op':'SPIN','measure_op':'MEASURE','wash_op':'WASH','seal_op':'SEAL','remove_op':'REMOVE','time_op':'WAIT','mix_op':'MIX','default_op':'OTHER'}
TYPE_MAP={'rg':'material','loc':'container','d':'device','sl':'seal','s':'setting','m':'measurement','mod':'modifier','mth':'method'}

def convert(path):
    raw=json.loads(path.read_text());source=path.with_suffix('.txt').read_text()
    nodes={x['id']:x['data'] for x in raw['nodes']}
    ops=sorted([n for n in nodes.values() if n['parent_type']=='op' and n.get('spans')],key=lambda n:n['step'])
    steps=[];created=set()
    for i,n in enumerate(ops):
        args=[]
        for role,key in n.get('inputs',{}).items():
            if key not in nodes:continue
            a=nodes[key];r=role if role in ROLES else ('setting' if 'setting' in role else 'usage')
            args.append(Argument(role=r,text=a['display_name'],type=TYPE_MAP.get(a['type'],'unknown')))
        # All setting edges, including values omitted by the single-valued inputs dict.
        for edge in raw['links']:
            if edge['target']!=n['name']:continue
            for key,v in edge.items():
                if not isinstance(v,dict) or v.get('slot') not in ('setting_of','usage_of'):continue
                a=nodes.get(v.get('name',''))
                if a:
                    arg=Argument(role='setting' if v['slot']=='setting_of' else 'usage',text=a['display_name'],type=TYPE_MAP.get(a['type'],'unknown'))
                    if arg not in args:args.append(arg)
        action=OP_MAP[n['type']]
        op=Operator(action=action,trigger=source[n['spans'][0]:n['spans'][1]],arguments=args,**semantic_template(action,args))
        step=make_step(op,source,i,n['spans'][0],steps)
        steps.append(step);created.update(map(norm,op.add))
    # Annotated physical inventory is supplied S0 for this controlled benchmark.
    # It is not claimed to be an independently extracted inventory. Stocking follows
    # first use: an object produced only by a later step is not available earlier.
    initial=required_inventory(steps)
    return Workflow(id='xwlp_'+str(raw['graph']['proto_idx']),source=source,initial=initial,steps=steps,
        metadata={'state_labels':'weak_templates_v1','inventory':'gold_argument_inventory','triggers':'gold_mentions',
                  'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'peg_file':str(path.relative_to(ROOT))})

def main():
    out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    workflows=[convert(p) for p in sorted((ROOT/'data/raw/xwlp/data').glob('*/*.peg'))]
    # Split before supervision generation; identical source documents cannot cross splits.
    groups=sorted({w.metadata['source_sha256'] for w in workflows})
    random.Random(42).shuffle(groups);n=len(groups)
    mapping={g:'train' if i<int(.7*n) else 'dev' if i<int(.85*n) else 'test' for i,g in enumerate(groups)}
    manifest={'seed':42,'split_kind':'custom document-group 70/15/15, NOT official X-WLP split','xwlp_commit':'db06ed1f27f9e9ef86a7324df5a0ee3607956db1','documents':{},'counts':{}}
    for split in ['train','dev','test']:
        docs=[w for w in workflows if mapping[w.metadata['source_sha256']]==split]
        manifest['documents'][split]=[w.id for w in docs]
        rows=[]
        for w in docs:
            trace=verify(w,evidence_checks=False,invariants=False)['trace']
            for i,(s,t) in enumerate(zip(w.steps,trace)):
                ev=s.evidence['trigger'];start=max(0,w.source.rfind('\n',0,ev.start)+1);end=w.source.find('\n',ev.end)
                end=len(w.source) if end<0 else end
                rows.append({'id':w.id+':'+s.id,'doc':w.id,'step':i,'source':w.source,'local':w.source[start:end],
                             'context':w.source[max(0,start-400):end],'trigger':s.operator.trigger,'state':t['before'],
                             'target':s.operator.model_dump(),'initial':w.initial,'offset':ev.start})
        (out/f'{split}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
        (out/f'{split}_workflows.json').write_text(json.dumps([w.model_dump() for w in docs],ensure_ascii=False),encoding='utf-8')
        manifest['counts'][split]={'documents':len(docs),'operators':len(rows)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['counts']))

if __name__=='__main__':main()
