import pytest
import torch
from sop2program.numerics import FiniteLogitsProcessor,require_finite_parameters,contiguous_head_input


@pytest.mark.parametrize('scores',[
    [[0.,float('nan')]],[[0.,float('inf')]],
    [[float('-inf'),float('-inf')]],
    [[1.,0.],[float('-inf'),float('-inf')]],
])
def test_corrupt_generation_scores_stop_immediately(scores):
    values=torch.tensor(scores)
    with pytest.raises(FloatingPointError,match='batch discarded'):
        FiniteLogitsProcessor()(torch.ones((len(scores),5),dtype=torch.long),values)


def test_finite_scores_and_legitimate_masks_are_preserved():
    scores=torch.tensor([[1.,float('-inf')],[0.,2.]])
    assert FiniteLogitsProcessor()(torch.ones((2,5),dtype=torch.long),scores) is scores


def test_model_parameter_check_names_corrupt_weight():
    model=torch.nn.Linear(2,2)
    require_finite_parameters(model,'load')
    with torch.no_grad():model.weight[0,0]=float('nan')
    with pytest.raises(FloatingPointError,match='weight'):
        require_finite_parameters(model,'load')


def test_strided_last_token_uses_one_contiguous_head_input():
    head=torch.nn.Linear(4,9,bias=False)
    hidden=torch.randn(3,7,4)
    last=hidden[:,-1:,:]
    assert not last.is_contiguous()
    expected=head(last)
    seen=[]
    head.register_forward_pre_hook(contiguous_head_input)
    head.register_forward_pre_hook(lambda module,inputs:seen.append(inputs[0].is_contiguous()))
    actual=head(last)
    torch.testing.assert_close(actual,expected)
    assert seen==[True]
