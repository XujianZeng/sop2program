import json
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer
from sop2program.constrained import tokenizer_data,prefix_filter,BatchedSchemaLogitsProcessor
from sop2program.ir import Operator


@pytest.fixture(scope='module')
def tokenizer_and_grammar():
    path=Path(__file__).resolve().parents[1]/'models/Qwen2.5-3B-Instruct'
    if not (path/'tokenizer.json').exists():pytest.skip('Local tokenizer not downloaded')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    return tokenizer,tokenizer_data(tokenizer)


def accepted(tokenizer,data,action):
    raw=json.dumps({'action':action,'trigger':'Wait','arguments':[], 'pre':[], 'add':[], 'delete':[], 'modify':[]},separators=(',',':'))
    ids=tokenizer.encode('Prompt:',add_special_tokens=False)
    allowed=prefix_filter(data,Operator.model_json_schema())
    for token in tokenizer.encode(raw,add_special_tokens=False)+[tokenizer.eos_token_id]:
        if token not in allowed(0,torch.tensor(ids)):return False
        ids.append(token)
    return True


def test_schema_accepts_valid_operator(tokenizer_and_grammar):
    tokenizer,data=tokenizer_and_grammar
    assert accepted(tokenizer,data,'WAIT')


def test_schema_rejects_out_of_inventory_action(tokenizer_and_grammar):
    tokenizer,data=tokenizer_and_grammar
    assert not accepted(tokenizer,data,'NOT_AN_ACTION')


def test_batch_mask_matches_transformers_prefix_mask(tokenizer_and_grammar):
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
    tokenizer,data=tokenizer_and_grammar
    old=PrefixConstrainedLogitsProcessor(prefix_filter(data,Operator.model_json_schema()),1)
    new=BatchedSchemaLogitsProcessor(data,Operator.model_json_schema())
    prefixes=['{"action":"WAIT","trigger":"Wait","arguments":[],"pre":[],"add":[],"delete":[],"modify":[]}',
              '{"action":"MIX","trigger":"Mix","arguments":[{"role":"a","text":"pellet","type":"material"}],"pre":["pellet"],"add":[],"delete":[],"modify":[]}']
    start=tokenizer.encode('Prompt:',add_special_tokens=False)
    sequences=[start+tokenizer.encode(p,add_special_tokens=False) for p in prefixes]
    generator=torch.Generator().manual_seed(42)
    for length in range(len(start),min(map(len,sequences))+1):
        ids=torch.tensor([s[:length] for s in sequences])
        scores=torch.randn((2,len(tokenizer)),generator=generator)
        assert torch.equal(old(ids,scores.clone()),new(ids,scores.clone()))
