"""Thin adapter to LM Format Enforcer's core API for Transformers 5.

lm-format-enforcer 0.11.3's bundled Transformers integration imports a class
from its old Transformers 4 location. This adapter uses the public core API
without modifying either installed package. Token decoding follows LMF's
prefix-token convention for preserving leading whitespace.
"""
from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


def tokenizer_data(tokenizer):
    prefix=tokenizer.encode('0',add_special_tokens=False)[-1]
    prefix_text=tokenizer.decode([prefix],clean_up_tokenization_spaces=False)
    special=set(tokenizer.all_special_ids)
    regular=[]
    for token in range(len(tokenizer)):
        if token in special:continue
        text=tokenizer.decode([prefix,token],clean_up_tokenization_spaces=False)[len(prefix_text):]
        standalone=tokenizer.decode([token],clean_up_tokenization_spaces=False)
        regular.append((token,text,len(text)>len(standalone)))
    def decode(tokens):
        return tokenizer.decode(tokens,clean_up_tokenization_spaces=False).rstrip('\ufffd')
    return TokenEnforcerTokenizerData(regular,decode,tokenizer.eos_token_id,False,len(tokenizer))


def prefix_filter(data,schema):
    enforcer=TokenEnforcer(data,JsonSchemaParser(schema))
    def allowed(batch_id,sent):
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
    return allowed


class BatchedSchemaLogitsProcessor:
    """One token transfer and one mask transfer per batch instead of per row."""
    def __init__(self,data,schema):
        self.enforcer=TokenEnforcer(data,JsonSchemaParser(schema))

    def __call__(self,input_ids,scores):
        import torch
        import numpy as np
        sequences=input_ids.tolist()
        # Avoid launching PyTorch's CPU worker pool for every small token mask.
        mask=np.zeros(tuple(scores.shape),dtype=np.bool_)
        for i,tokens in enumerate(sequences):
            allowed=self.enforcer.get_allowed_tokens(tokens).allowed_tokens
            if not allowed:raise ValueError(f'Unsatisfiable schema constraint for batch row {i}')
            mask[i,allowed]=True
        return scores.masked_fill(~torch.from_numpy(mask).to(scores.device),float('-inf'))
