from pathlib import Path

import pytest

TOKENIZER = Path(__file__).resolve().parents[1] / 'models/Qwen2.5-3B-Instruct'
NEEDS_TOKENIZER = {'test_qwen25_prompt_is_unchanged_by_the_thinking_flag'}


def pytest_collection_modifyitems(config, items):
    if (TOKENIZER / 'tokenizer_config.json').exists():
        return
    skip = pytest.mark.skip(reason='needs the local Qwen2.5-3B-Instruct tokenizer in models/')
    for item in items:
        if item.name in NEEDS_TOKENIZER:
            item.add_marker(skip)
