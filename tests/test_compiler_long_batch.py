import json
from types import SimpleNamespace
import pytest
import torch
from sop2program.compiler import LocalCompiler


@pytest.mark.parametrize('attention,expected_batches',[('eager',[1]*7),('sdpa',[4,3])])
def test_long_feedback_batches_preserve_complete_prompts_and_result_order(attention,expected_batches):
    class Inputs(dict):
        def __getattr__(self,name):return self[name]
        def to(self,device):return self
    class Tokenizer:
        pad_token_id=99;eos_token_id=99
        def apply_chat_template(self,messages,**kwargs):return messages[-1]['content']
        def __call__(self,texts,**kwargs):
            # Each original prompt retains its identifying source content.
            ids=torch.ones((len(texts),3359),dtype=torch.long)
            for i,text in enumerate(texts):ids[i,0]=int(text.split('document-')[1][0])+1
            return Inputs(input_ids=ids,attention_mask=torch.ones_like(ids))
        def decode(self,tokens,**kwargs):
            return json.dumps({'action':'WAIT','trigger':f'Wait{tokens[0]-1}',
                               'arguments':[],'pre':[],'add':[],'delete':[],'modify':[]})
    class Model:
        device=torch.device('cpu')
        config=SimpleNamespace(_attn_implementation=attention)
        generation_config=SimpleNamespace(eos_token_id=99)
        calls=[]
        def generate(self,input_ids,**kwargs):
            self.calls.append(tuple(input_ids.shape))
            tail=torch.stack((input_ids[:,0],torch.full((len(input_ids),),99)),dim=1)
            return torch.cat((input_ids,tail),dim=1)
    model=Model();compiler=LocalCompiler(model,Tokenizer(),constrained=False)
    rows=[{'context':f'document-{i}','trigger':f'Wait{i}',
           'repair_feedback':{'workflow':['full feedback']}} for i in range(7)]
    outputs=compiler.compile_batch(rows)
    assert model.calls==[(size,3359) for size in expected_batches]
    assert [operator.trigger for operator,_ in outputs]==[f'Wait{i}' for i in range(7)]
    assert all(info['input_tokens']==3359 for _,info in outputs)
