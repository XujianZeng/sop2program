import torch
from torch import nn

from sop2program.fp8 import FP8Linear, bf16_adapters, fp8_weights_finite


def quantized_layer(out=256, inp=384, block=(128, 128), seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(out, inp)
    layer = FP8Linear(inp, out, block)
    rows, cols = -(-out // block[0]), -(-inp // block[1])
    scales = torch.empty(rows, cols)
    q = torch.empty(out, inp, dtype=torch.float8_e4m3fn)
    for r in range(rows):
        for c in range(cols):
            tile = weight[r*block[0]:(r+1)*block[0], c*block[1]:(c+1)*block[1]]
            scales[r, c] = tile.abs().max() / 448
            q[r*block[0]:(r+1)*block[0], c*block[1]:(c+1)*block[1]] = (tile / scales[r, c]).to(torch.float8_e4m3fn)
    layer.weight = nn.Parameter(q, requires_grad=False)
    layer.weight_scale_inv = scales
    return layer, weight


def test_block_dequantization_matches_the_shipped_scales():
    layer, weight = quantized_layer()
    error = (layer.dequantized().float() - weight).norm() / weight.norm()
    assert error < .05


def test_ragged_blocks_use_the_same_scales_as_whole_blocks():
    layer, weight = quantized_layer(out=200, inp=300)
    error = (layer.dequantized().float() - weight).norm() / weight.norm()
    assert error < .05


def test_gradient_reaches_bf16_adapters_through_frozen_fp8():
    from peft import LoraConfig, get_peft_model
    layer, _ = quantized_layer()
    model = nn.Sequential()
    model.add_module('proj', layer)
    model = bf16_adapters(get_peft_model(model, LoraConfig(r=4, target_modules=['proj'])))
    lora = [p for n, p in model.named_parameters() if 'lora_' in n]
    assert lora and all(p.dtype == torch.bfloat16 for p in lora)
    x = torch.randn(3, 384, dtype=torch.bfloat16, requires_grad=True)
    model(x).float().pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()
    assert all(p.grad is not None for p in lora if p.requires_grad)
    assert not fp8_weights_finite(model)


def test_qwen25_prompt_is_unchanged_by_the_thinking_flag():
    from pathlib import Path
    from transformers import AutoTokenizer
    from sop2program.compiler import messages
    tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / 'models/Qwen2.5-3B-Instruct',
                                              local_files_only=True)
    row = {'context': 'Spin tube.', 'trigger': 'Spin', 'state': {'exists': ['tube'], 'attributes': {}}}
    plain = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
    flagged = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
    assert plain == flagged
