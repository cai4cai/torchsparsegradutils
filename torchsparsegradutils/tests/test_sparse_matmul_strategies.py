"""Public coalescing contracts, including batched value gradients and empty items."""

import pytest
import torch
from test_config import DEVICES

from torchsparsegradutils import sparse_mm

STRATEGIES = ["error", "when_needed", "assume_unique", "forward"]


def make_coo(device, batch_size, duplicates=False):
    indices = torch.tensor([[1, 0, 1], [2, 0, 1]], device=device)
    values = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64, device=device)
    if duplicates:
        indices = torch.cat([indices, indices[:, :1]], dim=1)
        values = torch.cat([values, -values[:1]])  # Preserve the cancelled coordinate's gradient.
    shape = (2, 3)
    if batch_size is not None:
        # The middle item is empty; nnz may differ between COO batch items.
        batches = [i for i in range(batch_size) if i != 1]
        indices = torch.cat([torch.cat([torch.full_like(indices[:1], i), indices]) for i in batches], dim=1)
        values = torch.cat([values * (i + 1) for i in batches])
        # Interleave batch coordinates: unflagged inputs need not be batch-sorted.
        order = torch.arange(values.numel(), device=device).reshape(len(batches), -1).t().reshape(-1)
        indices, values = indices[:, order], values[order]
        shape = (batch_size, *shape)
    return indices, values, shape


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("need_a,need_b", [(False, False), (True, False), (False, True), (True, True)])
def test_strategy_unique_value_gradients(device, batch_size, strategy, need_a, need_b):
    indices, values, shape = make_coo(device, batch_size)
    values.requires_grad_(need_a)
    reference_values = values.detach().clone().requires_grad_(need_a)
    a = torch.sparse_coo_tensor(indices, values, shape)
    if strategy == "error":
        a = a.coalesce()
    dense = torch.zeros(shape, dtype=values.dtype, device=device).index_put(
        tuple(indices), reference_values, accumulate=True
    )
    bshape = (*shape[:-2], 3, 2)
    # Noncontiguous RHS and upstream gradient exercise per-item slicing too.
    b = (
        torch.arange(1, 2 * torch.tensor(bshape).prod().item() + 1, dtype=values.dtype, device=device)
        .reshape(*bshape[:-1], 4)[..., ::2]
        .detach()
        .requires_grad_(need_b)
    )
    reference_b = b.detach().clone().requires_grad_(need_b)
    out = sparse_mm(a, b, coalesce_strategy=strategy)
    expected = dense @ reference_b
    torch.testing.assert_close(out, expected)
    if need_a or need_b:
        upstream = torch.arange(out.numel() * 2, dtype=values.dtype, device=device).reshape(*out.shape[:-1], 4)[
            ..., ::2
        ]
        actual_inputs = tuple(t for t in (values, b) if t.requires_grad)
        reference_inputs = tuple(t for t in (reference_values, reference_b) if t.requires_grad)
        for actual, reference in zip(
            torch.autograd.grad(out, actual_inputs, upstream),
            torch.autograd.grad(expected, reference_inputs, upstream),
        ):
            torch.testing.assert_close(actual, reference)
    else:
        assert out.grad_fn is None


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("strategy", ["when_needed", "forward"])
def test_strategy_duplicate_value_gradients(device, batch_size, strategy):
    indices, values, shape = make_coo(device, batch_size, duplicates=True)
    values.requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    a = torch.sparse_coo_tensor(indices, values, shape)
    b = torch.arange(1, 7, dtype=values.dtype, device=device).reshape(3, 2)
    if batch_size is not None:
        b = b.expand(batch_size, -1, -1)
    b = b.detach().requires_grad_()
    reference_b = b.detach().clone().requires_grad_()
    reference_a = torch.zeros(shape, dtype=values.dtype, device=device).index_put(
        tuple(indices), reference_values, accumulate=True
    )
    out = sparse_mm(a, b, coalesce_strategy=strategy)
    expected = reference_a @ reference_b
    upstream = torch.arange(1, out.numel() + 1, dtype=values.dtype, device=device).reshape(out.shape)
    torch.testing.assert_close(out, expected)
    for actual, reference in zip(
        torch.autograd.grad(out, (values, b), upstream),
        torch.autograd.grad(expected, (reference_values, reference_b), upstream),
    ):
        torch.testing.assert_close(actual, reference)
    assert not a.is_coalesced()
    torch.testing.assert_close(a._indices(), indices)
    torch.testing.assert_close(a._values(), values.detach())


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("duplicates", [False, True])
def test_default_rejects_unflagged_coo(device, batch_size, duplicates):
    indices, values, shape = make_coo(device, batch_size, duplicates)
    a = torch.sparse_coo_tensor(indices, values, shape)
    b = torch.ones((*shape[:-2], 3, 2), dtype=values.dtype, device=device)
    with pytest.raises(ValueError, match="Uncoalesced COO.*A.coalesce"):
        sparse_mm(a, b)
    with pytest.raises(ValueError, match="Uncoalesced COO"):
        sparse_mm(a, b, coalesce_strategy="error")


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_csr_accepts_every_strategy(device, strategy):
    a = torch.eye(3, dtype=torch.float64, device=device).to_sparse_csr().requires_grad_()
    b = torch.ones(3, 2, dtype=a.dtype, device=device, requires_grad=True)
    out = sparse_mm(a, b, coalesce_strategy=strategy)
    ga, gb = torch.autograd.grad(out.sum(), (a, b))
    torch.testing.assert_close(out, b)
    torch.testing.assert_close(ga.to_dense(), 2 * torch.eye(3, dtype=a.dtype, device=device))
    torch.testing.assert_close(gb, torch.ones_like(b))


def test_invalid_strategy_and_keyword_only():
    a, b = torch.eye(2).to_sparse(), torch.ones(2, 1)
    with pytest.raises(ValueError, match="coalesce_strategy"):
        sparse_mm(a, b, coalesce_strategy="typo")
    with pytest.raises(TypeError):
        sparse_mm(a, b, "forward")
