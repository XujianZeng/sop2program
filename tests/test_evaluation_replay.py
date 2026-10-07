import scripts.evaluate as evaluation
import pytest
from sop2program.ir import Argument,Operator,Workflow,semantic_template


def test_batched_evaluation_and_resume_use_only_predicted_states(tmp_path,monkeypatch):
    seen=[]
    class Compiler:
        def __init__(self,*args,**kwargs):pass
        def compile_batch(self,rows,with_state):
            outputs=[]
            assert len({r['doc'] for r in rows})==len(rows)
            for row in rows:
                seen.append((row['id'],row['state']))
                if row['id']=='b:s0':
                    outputs.append((None,{'raw':'bad json','latency_s':.1}))
                    continue
                outputs.append((Operator.model_validate(row['target']),{'raw':'test','latency_s':.1}))
            return outputs
    monkeypatch.setattr(evaluation,'LocalCompiler',Compiler)
    source='Create pellet. Wash pellet.'
    docs={key:Workflow(id=key,source=source,initial=[],steps=[]) for key in ('a','b')}
    rows=[]
    for key in docs:
        for i,(action,trigger) in enumerate([('CREATE','Create'),('WASH','Wash')]):
            args=[Argument(role='a',text='pellet',type='material')]
            op=Operator(action=action,trigger=trigger,arguments=args,**semantic_template(action,args))
            rows.append({'id':f'{key}:s{i}','doc':key,'step':i,'trigger':trigger,'offset':source.index(trigger),
                         'target':op.model_dump(),'state':{'exists':['LEAKED_GOLD_STATE']}})
    records=evaluation.run(None,None,rows,docs,tmp_path,'test',batch_size=2)
    states=dict(seen)
    assert states['a:s1']['exists']==['pellet']
    assert states['b:s1']['exists']==[]
    assert all('LEAKED_GOLD_STATE' not in state['exists'] for _,state in seen)
    seen.clear()
    assert evaluation.run(None,None,rows,docs,tmp_path,'test',batch_size=2)==records
    assert seen==[]


def test_numerical_failure_does_not_commit_any_prediction(tmp_path,monkeypatch):
    class Compiler:
        def __init__(self,*args,**kwargs):pass
        def compile_batch(self,*args):raise FloatingPointError('batch discarded')
    monkeypatch.setattr(evaluation,'LocalCompiler',Compiler)
    doc=Workflow(id='a',source='Wait.',initial=[],steps=[])
    row={'id':'a:s0','doc':'a','step':0,'trigger':'Wait','offset':0,'target':{}}
    with pytest.raises(FloatingPointError):
        evaluation.run(None,None,[row],{'a':doc},tmp_path,'no_state')
    assert (tmp_path/'predictions.jsonl').read_text()==''
    assert not (tmp_path/'workflows.json').exists()
