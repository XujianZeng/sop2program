"""Chat backends for the external-baseline repairers: DeepSeek API or a local model.

Both return (text, info) per request so every generated sequence is counted against
the same per-document allowance, whatever produced it. Keys are read from .env and
never written to logs or result files.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ir import Operator

ROOT = Path(__file__).resolve().parents[1]


def load_env(path=ROOT / '.env'):
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class DeepSeekChat:
    """OpenAI-format chat completions with thinking disabled and JSON output enforced."""
    def __init__(self, model='deepseek-flash', concurrency=16, max_retries=6, timeout=180):
        import requests
        load_env()
        self.session = requests.Session()
        self.url = os.environ.get('DEEPSEEK_URL', 'https://api.deepseek.com').rstrip('/') + '/chat/completions'
        self.key = os.environ['DEEPSEEK_API_KEY']
        self.model, self.concurrency, self.max_retries, self.timeout = model, concurrency, max_retries, timeout
        self.name = f'deepseek:{model}'

    def _one(self, request):
        body = {'model': self.model, 'messages': request['messages'], 'temperature': 0,
                'max_tokens': request['max_tokens'], 'response_format': {'type': 'json_object'},
                'thinking': {'type': 'disabled'}}
        delay = 2.0
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                r = self.session.post(self.url, json=body, timeout=self.timeout,
                                      headers={'Authorization': f'Bearer {self.key}'})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f'HTTP {r.status_code}')
                r.raise_for_status()
                payload = r.json()
                choice = payload['choices'][0]
                usage = payload.get('usage', {})
                return choice['message'].get('content') or '', {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'cache_hit_tokens': usage.get('prompt_cache_hit_tokens', 0),
                    'cache_miss_tokens': usage.get('prompt_cache_miss_tokens', 0),
                    'finish_reason': choice.get('finish_reason'),
                    'hit_token_limit': choice.get('finish_reason') == 'length',
                    'latency_s': time.perf_counter() - start, 'attempts': attempt + 1,
                    'model': payload.get('model', self.model)}
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f'DeepSeek request failed after {self.max_retries} attempts: {exc}') from exc
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def chat_batch(self, requests):
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(self._one, requests))
        for _, info in results:
            info['amortized_latency_s'] = info['latency_s']
        return results


class LocalChat:
    """Greedy local generation; each request's JSON schema constrains decoding."""
    def __init__(self, model, tokenizer, name):
        from .constrained import tokenizer_data
        self.model, self.tokenizer, self.name = model, tokenizer, name
        self.token_data = tokenizer_data(tokenizer)

    def chat_batch(self, requests):
        groups = {}
        for i, request in enumerate(requests):
            key = (json.dumps(request['schema'], sort_keys=True), request['max_tokens'])
            groups.setdefault(key, []).append(i)
        results = [None] * len(requests)
        for (schema, max_tokens), indexes in groups.items():
            for i, output in zip(indexes, self._generate([requests[i]['messages'] for i in indexes],
                                                         json.loads(schema), max_tokens)):
                results[i] = output
        return results

    def _generate(self, conversations, schema, max_tokens):
        import torch
        from .compiler import inference_batch_capacity
        from .constrained import BatchedSchemaLogitsProcessor
        from .numerics import FiniteLogitsProcessor
        tok = self.tokenizer
        tok.padding_side = 'left'
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for m in conversations]
        inputs = tok(prompts, return_tensors='pt', padding=True).to(self.model.device)
        capacity = inference_batch_capacity(self.model, inputs.input_ids.shape[1] + max_tokens)
        if len(prompts) > capacity:
            return [o for s in range(0, len(prompts), capacity)
                    for o in self._generate(conversations[s:s + capacity], schema, max_tokens)]
        processors = [FiniteLogitsProcessor(), BatchedSchemaLogitsProcessor(self.token_data, schema)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=tok.eos_token_id, logits_processor=processors)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        stops = self.model.generation_config.eos_token_id
        stops = set(stops if isinstance(stops, list) else [stops])
        results = []
        for i in range(len(prompts)):
            tokens = out[i, inputs.input_ids.shape[1]:].tolist()
            stop = next((j for j, t in enumerate(tokens) if t in stops), len(tokens))
            results.append((tok.decode(tokens[:stop], skip_special_tokens=True), {
                'input_tokens': int(inputs.attention_mask[i].sum()), 'output_tokens': min(stop + 1, len(tokens)),
                'hit_token_limit': stop == len(tokens) and len(tokens) >= max_tokens,
                'latency_s': elapsed, 'amortized_latency_s': elapsed / len(prompts), 'batch_size': len(prompts),
                'finite_logits_checked': True}))
        return results


class ChatOperatorCompiler:
    """compile_batch over a chat backend, so the controlled controller can use any LLM."""
    def __init__(self, chat, max_tokens=1024):
        self.chat, self.max_tokens = chat, max_tokens

    def compile_batch(self, rows):
        from .compiler import messages
        schema = Operator.model_json_schema()
        outputs = self.chat.chat_batch([{'messages': messages(row), 'schema': schema, 'max_tokens': self.max_tokens}
                                        for row in rows])
        results = []
        for row, (raw, info) in zip(rows, outputs):
            try:
                operator, error = Operator.model_validate_json(raw), None
            except Exception as exc:
                operator, error = None, str(exc)[:300]
            pinned = {}
            if operator is not None:
                # The marked span is an input, so the model's copy is replaced, not trusted.
                pinned = {'model_trigger': operator.trigger, 'trigger_from_input': True}
                operator = operator.model_copy(update={'trigger': row['trigger']})
            results.append((operator, {**info, **pinned, 'raw': raw, 'error': error}))
        return results
