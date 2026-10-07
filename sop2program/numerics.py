"""Reject corrupt GPU models or scores before they can become saved predictions."""
import torch
from transformers import LogitsProcessor


def require_finite_parameters(model,stage):
    from .fp8 import fp8_weights_finite
    bad=fp8_weights_finite(model)
    if bad:raise FloatingPointError(f'NaN FP8 weights at {stage}: {bad[:8]}')
    names=[];checks=[]
    for name,parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype!=torch.float8_e4m3fn:
            names.append(name);checks.append(torch.isfinite(parameter).all())
    if not checks:return
    flags=torch.stack(checks).cpu().tolist()
    failed=[name for name,finite in zip(names,flags) if not finite]
    if failed:raise FloatingPointError(f'Nonfinite model parameters at {stage}: {failed[:8]}')


class FiniteLogitsProcessor(LogitsProcessor):
    def __call__(self,input_ids,scores):
        # Negative infinity may be a legitimate mask from another processor.
        valid=(~torch.isnan(scores).any() & ~torch.isposinf(scores).any()
               & torch.isfinite(scores).any(dim=-1).all())
        if not valid.item():
            snapshot=scores.detach().cpu()
            details={'nan_per_row':torch.isnan(snapshot).sum(-1).tolist(),
                     'positive_inf_per_row':torch.isposinf(snapshot).sum(-1).tolist(),
                     'finite_per_row':torch.isfinite(snapshot).sum(-1).tolist()}
            raise FloatingPointError(f'Nonfinite generation scores at token length {input_ids.shape[1]}; '
                                     f'batch discarded; CPU-verified counts={details}')
        return scores


def contiguous_head_input(module,inputs):
    # Qwen slices the final token from [batch, prompt, hidden]. The strided view
    # otherwise dispatches F.linear to a separate GEMM per batch row on CUDA.
    return (inputs[0].contiguous(),*inputs[1:])


def load_checked_model(model_path,adapter_path=None,compile_cache=False,attention='eager'):
    if attention not in ('eager','sdpa'):raise ValueError(f'Unsupported attention backend: {attention}')
    from .fp8 import is_fp8_checkpoint
    if is_fp8_checkpoint(model_path):
        # Merging would re-quantize the update into FP8, so the adapter stays a BF16 side branch.
        from .fp8 import bf16_adapters,load_fp8_model
        model=load_fp8_model(model_path,attention)
        if adapter_path is not None:
            from pathlib import Path
            from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
            from safetensors.torch import load_file
            model=bf16_adapters(get_peft_model(model,LoraConfig.from_pretrained(adapter_path)))
            result=set_peft_model_state_dict(model,load_file(Path(adapter_path)/'adapter_model.safetensors'))
            missing=[k for k in result.missing_keys if 'lora_' in k]
            if missing or result.unexpected_keys:
                raise RuntimeError(f'Adapter mismatch: missing {missing[:3]} unexpected {result.unexpected_keys[:3]}')
        require_finite_parameters(model,'CUDA FP8 load')
    else:
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
                                                device_map='cpu',attn_implementation=attention,local_files_only=True)
        if adapter_path is not None:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,adapter_path).merge_and_unload(safe_merge=True)
        require_finite_parameters(model,'CPU load/adapter merge')
        model=model.to('cuda')
        require_finite_parameters(model,'CUDA transfer')
    model.lm_head.register_forward_pre_hook(contiguous_head_input)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    model._sop_compile_cache=compile_cache
    if compile_cache:
        torch._dynamo.config.cache_size_limit=32
    model.eval();model.config.use_cache=True
    return model
