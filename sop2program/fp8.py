"""Frozen block-FP8 base weights with BF16 LoRA, for Qwen's official FP8 checkpoints.

Each quantized projection keeps its float8_e4m3fn weight and one float32 inverse scale
per 128x128 block exactly as shipped. A forward pass dequantizes that one weight to
bfloat16 and multiplies in bfloat16, so autograd reaches the adapters without FP8
kernels or Triton. Storage is FP8; arithmetic is BF16.
"""
import json
from pathlib import Path

import torch
from torch import nn


def is_fp8_checkpoint(path):
    config=json.loads((Path(path)/'config.json').read_text(encoding='utf-8'))
    return (config.get('quantization_config') or {}).get('quant_method')=='fp8'


class FP8Linear(nn.Linear):
    """nn.Linear subclass so PEFT wraps it with an ordinary LoRA layer."""
    def __init__(self,in_features,out_features,block):
        nn.Module.__init__(self)
        self.in_features=in_features;self.out_features=out_features;self.block=tuple(block)
        rows=-(-out_features//self.block[0]);cols=-(-in_features//self.block[1])
        self.weight=nn.Parameter(torch.empty(out_features,in_features,dtype=torch.float8_e4m3fn,device='meta'),
                                 requires_grad=False)
        self.register_buffer('weight_scale_inv',torch.empty(rows,cols,dtype=torch.float32,device='meta'))
        self.register_parameter('bias',None)

    def dequantized(self):
        # Every e4m3 value is exact in bfloat16, so only the block scale and the product
        # round. A float32 detour would double the memory traffic of every decode step.
        (br,bc),(out,inp)=self.block,self.weight.shape
        scale=self.weight_scale_inv.to(torch.bfloat16)
        if out%br==0 and inp%bc==0:
            w=self.weight.view(out//br,br,inp//bc,bc).to(torch.bfloat16)
            return w.mul_(scale[:,None,:,None]).view(out,inp)
        full=scale.repeat_interleave(br,0)[:out].repeat_interleave(bc,1)[:,:inp]
        return self.weight.to(torch.bfloat16)*full

    def forward(self,x):
        return nn.functional.linear(x,self.dequantized())


def load_fp8_model(path,attention='eager',device='cuda'):
    """Build the architecture on the meta device, then stream FP8 shards straight to the GPU."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig,AutoModelForCausalLM
    path=Path(path)
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    block=config.quantization_config['weight_block_size']
    del config.quantization_config
    with init_empty_weights():
        model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16,attn_implementation=attention)
    index=json.loads((path/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    for key in index:
        if not key.endswith('.weight_scale_inv'):continue
        name=key[:-len('.weight_scale_inv')]
        parent,_,child=name.rpartition('.')
        old=model.get_submodule(name)
        setattr(model.get_submodule(parent),child,FP8Linear(old.in_features,old.out_features,block))
    for shard in sorted(set(index.values())):
        with safe_open(path/shard,framework='pt',device=device) as f:
            for key in f.keys():
                set_module_tensor_to_device(model,key,device,value=f.get_tensor(key))
    model.to(device)
    leftover=[n for n,t in list(model.named_parameters())+list(model.named_buffers()) if t.device.type=='meta']
    if leftover:raise RuntimeError(f'Checkpoint left tensors unloaded: {leftover[:5]}')
    if config.tie_word_embeddings:model.tie_weights()
    return model


def bf16_adapters(model):
    """PEFT moves new adapters to the base weight dtype, which here would be float8.

    Recast every LoRA tensor to bfloat16 and re-initialize it there, so nothing about
    the adapter (initial values, saved weights, updates) passes through FP8.
    """
    from peft.tuners.lora import LoraLayer
    for module in model.modules():
        if isinstance(module,LoraLayer) and isinstance(module.get_base_layer(),FP8Linear):
            for name in module.active_adapters:
                module.lora_A[name].to(torch.bfloat16);module.lora_B[name].to(torch.bfloat16)
                module.reset_lora_parameters(name,True)
    return model


def fp8_weights_finite(model):
    """float8_e4m3fn has no infinities; NaN is the 0x7F/0xFF pattern."""
    bad=[]
    for name,parameter in model.named_parameters():
        if parameter.dtype==torch.float8_e4m3fn and ((parameter.view(torch.uint8)&0x7F)==0x7F).any():
            bad.append(name)
    return bad
