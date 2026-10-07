import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def test_logits_from_the_last_prompt_token_give_the_full_loss():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = Qwen2ForCausalLM(config).eval()
    prefix, target = 23, 7
    ids = torch.randint(0, 97, (1, prefix + target))
    labels = ids.clone()
    labels[:, :prefix] = -100
    full = model(input_ids=ids, labels=labels).loss
    keep = ids.shape[1] - prefix + 1
    sliced = model(input_ids=ids, labels=labels[:, -keep:], logits_to_keep=keep).loss
    assert torch.allclose(full, sliced, atol=1e-6)
