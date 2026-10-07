import torch

from sop2program.compiler import LocalCompiler


class Tokenizer:
    padding_side = 'right'
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, raw):
        self.raw = raw

    def apply_chat_template(self, messages, **kwargs):
        return 'prompt'

    def __call__(self, prompts, **kwargs):
        class Batch(dict):
            def to(self, device):
                return self
        ids = torch.ones((len(prompts), 2), dtype=torch.long)
        batch = Batch(input_ids=ids, attention_mask=ids)
        batch.input_ids = ids
        batch.attention_mask = ids
        return batch

    def decode(self, tokens, **kwargs):
        return self.raw


class Model:
    device = torch.device('cpu')
    config = type('Config', (), {'_attn_implementation': 'sdpa'})()
    generation_config = type('Generation', (), {'eos_token_id': 0})()

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[5, 0]] * len(input_ids))], dim=1)


RAW = ('{"action":"OTHER","trigger":"chexelled","arguments":[],'
       '"pre":[],"add":[],"delete":[],"modify":[]}')


def test_pinned_trigger_replaces_the_copy_and_keeps_it_for_audit():
    compiler = LocalCompiler(Model(), Tokenizer(RAW), constrained=False, pin_trigger=True)
    op, info = compiler.compile({'context': 'c', 'trigger': 'chelexed', 'state': {}})
    assert op.trigger == 'chelexed'
    assert info['model_trigger'] == 'chexelled' and info['trigger_from_input']


def test_default_keeps_the_model_copy():
    compiler = LocalCompiler(Model(), Tokenizer(RAW), constrained=False)
    op, info = compiler.compile({'context': 'c', 'trigger': 'chelexed', 'state': {}})
    assert op.trigger == 'chexelled' and 'model_trigger' not in info
