import pytest
import torch

from torchsparsegradutils import (
    is_sparse_bsc,
    is_sparse_bsr,
    is_sparse_coo,
    is_sparse_csc,
    is_sparse_csr,
    require_sparse_bsc,
    require_sparse_bsr,
    require_sparse_coo,
    require_sparse_csc,
    require_sparse_csr,
)


def _layout_examples():
    dense = torch.eye(2)
    return [
        (torch.sparse_coo, dense.to_sparse_coo(), is_sparse_coo, require_sparse_coo),
        (torch.sparse_csr, dense.to_sparse_csr(), is_sparse_csr, require_sparse_csr),
        (torch.sparse_csc, dense.to_sparse_csc(), is_sparse_csc, require_sparse_csc),
        (torch.sparse_bsr, dense.to_sparse_bsr((1, 1)), is_sparse_bsr, require_sparse_bsr),
        (torch.sparse_bsc, dense.to_sparse_bsc((1, 1)), is_sparse_bsc, require_sparse_bsc),
    ]


@pytest.mark.parametrize(("layout", "tensor", "predicate", "validator"), _layout_examples())
def test_sparse_layout_guards_and_validators(layout, tensor, predicate, validator):
    assert predicate(tensor)
    assert validator(tensor) is tensor
    assert tensor.layout == layout


@pytest.mark.parametrize(("layout", "tensor", "predicate", "validator"), _layout_examples())
def test_sparse_layout_guards_reject_other_values(layout, tensor, predicate, validator):
    del layout, tensor
    dense = torch.eye(2)
    assert not predicate(dense)
    assert not predicate(object())
    with pytest.raises(TypeError, match="Expected a torch.Tensor with layout"):
        validator(dense)
    with pytest.raises(TypeError, match="Expected a torch.Tensor with layout"):
        validator(object())


def test_sparse_layout_guards_do_not_confuse_layouts():
    examples = _layout_examples()
    for _, tensor, expected_predicate, _ in examples:
        assert expected_predicate(tensor)
        for _, _, other_predicate, _ in examples:
            assert other_predicate(tensor) is (other_predicate is expected_predicate)
