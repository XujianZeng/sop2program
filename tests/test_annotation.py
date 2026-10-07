import json
from pathlib import Path

import pytest

from sop2program.annotation import GoldIdentity
from sop2program.ir import Workflow
from sop2program.verify import verify

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed'
pytestmark=pytest.mark.skipif(not (DATA/'train_workflows.json').exists(),reason='processed X-WLP data not built')


def workflows():
    for split in ('train','dev','test'):
        yield from (Workflow.model_validate(w) for w in json.loads((DATA/f'{split}_workflows.json').read_text(encoding='utf-8')))


def identity(doc):
    w=next(w for w in workflows() if w.id==doc)
    return GoldIdentity(w,ROOT/w.metadata['peg_file'])


def test_every_gold_workflow_executes_on_annotated_identity():
    failing=[w.id for w in workflows()
             if not verify(GoldIdentity(w,ROOT/w.metadata['peg_file']).gold(),evidence_checks=False,invariants=False)['pass']]
    assert failing==[]


def test_coreference_joins_renamed_mentions():
    g=identity('xwlp_0')
    assert g.resolve('cells',1)==g.resolve('E. coli LB culture*',0)


def test_same_text_distinct_entities_stay_apart():
    g=identity('xwlp_0')
    assert g.resolve('supernatant',2)!=g.resolve('supernatant',10)


def test_unanchored_string_names_an_entity_that_already_exists():
    g=identity('xwlp_0')
    assert g.resolve('supernatant',9)==g.resolve('supernatant',2)


def test_unanchored_string_prefers_an_alive_reading():
    g=identity('xwlp_0')
    later=g.resolve('supernatant',10)
    assert g.resolve('supernatant',9,alive={later})==later
