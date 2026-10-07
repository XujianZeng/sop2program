import json
import pytest
from scripts.evaluate import read_predictions


def test_torn_final_record_is_discarded_without_extra_blank_line(tmp_path):
    path=tmp_path/'predictions.jsonl'
    path.write_bytes(b'{"id":"one"}\n{"id":')
    assert list(read_predictions(path))==['one']
    assert path.read_bytes()==b'{"id":"one"}\n'
    assert list(read_predictions(path))==['one']


def test_complete_final_record_without_newline_is_preserved(tmp_path):
    path=tmp_path/'predictions.jsonl'
    path.write_text('{"id":"one"}')
    assert list(read_predictions(path))==['one']
    assert path.read_bytes().endswith(b'\n')


def test_corruption_in_middle_is_not_silently_ignored(tmp_path):
    path=tmp_path/'predictions.jsonl'
    path.write_bytes(b'{bad}\n{"id":"two"}\n')
    with pytest.raises(json.JSONDecodeError):read_predictions(path)
