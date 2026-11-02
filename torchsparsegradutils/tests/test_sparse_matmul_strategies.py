"""Public COO uniqueness contracts, including batched value gradients and empty items."""

import pytest
import torch
from test_config import DEVICES

from torchsparsegradutils import sparse_mm


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
@pytest.mark.parametrize("assume_unique", [False, True])
@pytest.mark.parametrize("need_a,need_b", [(False, False), (True, False), (False, True), (True, True)])
def test_unique_value_gradients(device, batch_size, assume_unique, need_a, need_b):
    indices, values, shape = make_coo(device, batch_size)
    values.requires_grad_(need_a)
    reference_values = values.detach().clone().requires_grad_(need_a)
    a = torch.sparse_coo_tensor(indices, values, shape)
    if not assume_unique:
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
    out = sparse_mm(a, b, assume_unique=assume_unique)
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
def test_external_coalesce_duplicate_value_gradients(device, batch_size):
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
    out = sparse_mm(a.coalesce(), b)
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
def test_duplicate_coo_matrix_gradient_and_value_parameter_gradients(device):
    # A[0, 0] = 2 + 3. For loss = 4 * A[0, 0], the matrix derivative is 4,
    # and each independently trainable constructor value also has derivative 4.
    indices = torch.tensor([[0, 0], [0, 0]], device=device)
    values = torch.tensor([2.0, 3.0], dtype=torch.float64, device=device, requires_grad=True)
    a = torch.sparse_coo_tensor(indices, values, (1, 1))
    sparse_leaf = a.detach().requires_grad_()
    b = torch.tensor([[4.0]], dtype=values.dtype, device=device)

    sparse_mm(sparse_leaf.coalesce(), b).sum().backward()
    sparse_mm(a.coalesce(), b).sum().backward()

    # The sparse leaf stores two entries, but its matrix gradient stores one.
    assert sparse_leaf._nnz() == 2
    assert sparse_leaf.grad.is_coalesced()
    assert sparse_leaf.grad._nnz() == 1
    torch.testing.assert_close(sparse_leaf.grad.indices(), indices[:, :1])
    torch.testing.assert_close(sparse_leaf.grad.values(), b.reshape(1))
    # Constructor backward distributes the derivative without doubling it.
    torch.testing.assert_close(values.grad, torch.tensor([4.0, 4.0], dtype=values.dtype, device=device))
    assert not a.is_coalesced()
    torch.testing.assert_close(a._indices(), indices)
    torch.testing.assert_close(a._values(), values.detach())


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.parametrize("leaf", [False, True])
def test_default_rejects_unflagged_coo(device, batch_size, duplicates, leaf):
    indices, values, shape = make_coo(device, batch_size, duplicates)
    a = torch.sparse_coo_tensor(indices, values.requires_grad_(), shape)
    if leaf:
        a = a.detach().requires_grad_()
    b = torch.ones((*shape[:-2], 3, 2), dtype=values.dtype, device=device)
    with pytest.raises(ValueError, match="Uncoalesced COO.*A.coalesce"):
        sparse_mm(a, b)
    with pytest.raises(ValueError, match="Uncoalesced COO"):
        sparse_mm(a, b, assume_unique=False)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("assume_unique", [False, True])
def test_csr_accepts_either_option(device, assume_unique):
    a = torch.eye(3, dtype=torch.float64, device=device).to_sparse_csr().requires_grad_()
    b = torch.ones(3, 2, dtype=a.dtype, device=device, requires_grad=True)
    out = sparse_mm(a, b, assume_unique=assume_unique)
    ga, gb = torch.autograd.grad(out.sum(), (a, b))
    torch.testing.assert_close(out, b)
    torch.testing.assert_close(ga.to_dense(), 2 * torch.eye(3, dtype=a.dtype, device=device))
    torch.testing.assert_close(gb, torch.ones_like(b))


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("assume_unique", [False, True])
def test_coo_gradient_flag_preserves_known_ordering(device, assume_unique):
    # Unique coordinates need not be sorted: assume_unique must not mark them coalesced.
    indices = torch.tensor([[1, 0, 1], [2, 0, 1]], device=device)
    values = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64, device=device)
    a = torch.sparse_coo_tensor(indices, values, (2, 3))
    if not assume_unique:
        a = a.coalesce()
    a.requires_grad_()
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=a.dtype, device=device)
    upstream = torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=a.dtype, device=device)

    # Also reject a false True flag at construction, before downstream ops trust it.
    with torch.sparse.check_sparse_tensor_invariants():
        out = sparse_mm(a, b, assume_unique=assume_unique)
        (grad_a,) = torch.autograd.grad(out, a, upstream)

    assert grad_a.is_coalesced() == a.is_coalesced()
    if assume_unique:
        torch.testing.assert_close(grad_a._indices(), indices)
    mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=a.dtype, device=device)
    torch.testing.assert_close(grad_a.to_dense(), (upstream @ b.t()) * mask)


@pytest.mark.parametrize("invalid", [None, 0, 1, "assume_unique"])
def test_invalid_option_and_keyword_only(invalid):
    a, b = torch.eye(2).to_sparse(), torch.ones(2, 1)
    with pytest.raises(ValueError, match="assume_unique"):
        sparse_mm(a, b, assume_unique=invalid)
    with pytest.raises(TypeError):
        sparse_mm(a, b, True)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize(
    "mode", ["inference", "rhs_gradient", "no_grad", "inference_mode", "inference_mode_grad_enabled"]
)
@pytest.mark.parametrize("assume_unique", [False, True])
def test_uncoalesced_without_a_gradient(device, batch_size, mode, assume_unique):
    indices, values, shape = make_coo(device, batch_size, duplicates=True)
    a = torch.sparse_coo_tensor(indices, values, shape)
    a.requires_grad_(mode in ("no_grad", "inference_mode", "inference_mode_grad_enabled"))
    b = torch.arange(1, 7, dtype=values.dtype, device=device).reshape(3, 2)
    if batch_size is not None:
        b = b.expand(batch_size, -1, -1)
    b = b.detach().requires_grad_(mode == "rhs_gradient")
    ref_b = b.detach().clone().requires_grad_(b.requires_grad)
    expected = a.detach().to_dense() @ ref_b
    if mode in ("no_grad", "inference_mode", "inference_mode_grad_enabled"):
        context = torch.no_grad if mode == "no_grad" else torch.inference_mode
        with context(), torch.set_grad_enabled(mode == "inference_mode_grad_enabled"):
            actual = sparse_mm(a, b, assume_unique=assume_unique)
    else:
        actual = sparse_mm(a, b, assume_unique=assume_unique)
    torch.testing.assert_close(actual, expected)
    if mode == "rhs_gradient":
        upstream = torch.arange(actual.numel(), dtype=values.dtype, device=device).reshape(actual.shape)
        torch.testing.assert_close(
            torch.autograd.grad(actual, b, upstream)[0], torch.autograd.grad(expected, ref_b, upstream)[0]
        )
    else:
        assert actual.grad_fn is None
    assert not a.is_coalesced()
    torch.testing.assert_close(a._indices(), indices)
    torch.testing.assert_close(a._values(), values)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("assume_unique", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_coo_gradient_preserves_exact_input_storage(device, batch_size, assume_unique, empty):
    indices, values, shape = make_coo(device, batch_size)
    values[0] = 0  # Explicit zero entries must still receive derivatives.
    if empty:
        indices, values = indices[:, :0], values[:0]
    a = torch.sparse_coo_tensor(indices, values, shape)
    if not assume_unique:
        a = a.coalesce()
    a = a.detach().requires_grad_()
    b = torch.arange(1, 7, dtype=values.dtype, device=device).reshape(3, 2)
    if batch_size is not None:
        b = torch.stack([b + batch for batch in range(batch_size)])
    dense_a = a.to_dense().detach().requires_grad_()
    with torch.sparse.check_sparse_tensor_invariants():
        out = sparse_mm(a, b, assume_unique=assume_unique)
        upstream = torch.arange(1, out.numel() + 1, dtype=values.dtype, device=device).reshape(out.shape)
        (grad_a,) = torch.autograd.grad(out, a, upstream)
    (dense_grad,) = torch.autograd.grad(dense_a @ b, dense_a, upstream)
    assert grad_a.shape == a.shape
    assert grad_a._nnz() == a._nnz()
    assert grad_a.is_coalesced() == a.is_coalesced()
    torch.testing.assert_close(grad_a._indices(), a._indices())
    torch.testing.assert_close(grad_a._values(), dense_grad[tuple(a._indices())])


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_csr_gradient_preserves_exact_input_storage(device, batch_size):
    a = torch.tensor([[0.0, 2.0, 3.0], [4.0, 0.0, 5.0]], dtype=torch.float64, device=device).to_sparse_csr()
    # Keep the stored coordinate even when its value is zero.
    a.values()[0] = 0
    if batch_size is not None:
        from torchsparsegradutils.utils import stack_csr

        a = stack_csr([a] * batch_size)
    a.requires_grad_()
    b = torch.arange(1, 7, dtype=a.dtype, device=device).reshape(3, 2)
    if batch_size is not None:
        b = b.expand(batch_size, -1, -1)
    (grad_a,) = torch.autograd.grad(sparse_mm(a, b).sum(), a)
    torch.testing.assert_close(grad_a.crow_indices(), a.crow_indices())
    torch.testing.assert_close(grad_a.col_indices(), a.col_indices())
    torch.testing.assert_close(grad_a.values(), b.sum(-1).gather(-1, a.col_indices()))


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_coalesced_trainable_input_needs_no_python_coalesce(device, monkeypatch):
    a = torch.eye(3, dtype=torch.float64, device=device).to_sparse().requires_grad_()
    b = torch.ones(3, 2, dtype=a.dtype, device=device, requires_grad=True)

    def unexpected_coalesce(*args, **kwargs):
        pytest.fail("Already-coalesced unbatched input needs no Python coalesce call")

    monkeypatch.setattr(torch.Tensor, "coalesce", unexpected_coalesce)
    ga, gb = torch.autograd.grad(sparse_mm(a, b).sum(), (a, b))
    torch.testing.assert_close(ga._values(), torch.full((3,), 2.0, dtype=a.dtype, device=device))
    torch.testing.assert_close(gb, torch.ones_like(b))


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_default_check_cannot_anticipate_requested_gradient_target(device):
    indices, values, shape = make_coo(device, None)
    a = torch.sparse_coo_tensor(indices, values, shape).requires_grad_()
    b = torch.ones(3, 2, dtype=values.dtype, device=device, requires_grad=True)
    # Even if the caller intends autograd.grad(out, b), A advertises a gradient.
    with pytest.raises(ValueError, match="Uncoalesced COO"):
        sparse_mm(a, b)
    torch.testing.assert_close(
        torch.autograd.grad(sparse_mm(a.detach(), b).sum(), b)[0], a.to_dense().sum(0)[:, None].expand_as(b)
    )


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_reentrant_checkpoint_validates_on_recomputation(device):
    from torch.utils.checkpoint import checkpoint

    indices, values, shape = make_coo(device, None)
    a = torch.sparse_coo_tensor(indices, values, shape).requires_grad_()
    b = torch.ones(3, 2, dtype=values.dtype, device=device, requires_grad=True)
    out = checkpoint(sparse_mm, a, b, use_reentrant=True)
    torch.testing.assert_close(out, a.to_dense() @ b)
    with pytest.raises(ValueError, match="Uncoalesced COO"):
        out.sum().backward()


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
@pytest.mark.parametrize("created_inside", [False, True])
def test_inference_mode_input_creation_context(device, batch_size, layout, created_inside):
    from torchsparsegradutils.utils import stack_csr

    dense_a = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, -4.0], [5.0, 6.0, 0.0]], dtype=torch.float64, device=device)
    b = torch.arange(1, 7, dtype=dense_a.dtype, device=device).reshape(3, 2)
    if batch_size is not None:
        dense_a = torch.stack([dense_a * (batch + 1) for batch in range(batch_size)])
        b = torch.stack([b + 7 * batch for batch in range(batch_size)])
        items = [item.to_sparse(layout=layout) for item in dense_a]
        a = stack_csr(items) if layout == torch.sparse_csr else torch.stack(items).coalesce()
    else:
        a = dense_a.to_sparse(layout=layout)
    expected = dense_a @ b
    a.requires_grad_()
    with torch.inference_mode():
        if created_inside:
            a, b = a.clone(), b.clone()
        out = sparse_mm(a, b)
        assert torch.is_inference_mode_enabled()
        assert out.is_inference()
    torch.testing.assert_close(out, expected)
    assert out.grad_fn is None


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_stacked_trainable_unique_coo_value_gradients(device, batch_size):
    indices, initial_values, shape = make_coo(device, None)
    values = [(initial_values * (batch + 1)).detach().requires_grad_() for batch in range(batch_size)]
    reference_values = [v.detach().clone().requires_grad_() for v in values]
    a = torch.stack([torch.sparse_coo_tensor(indices, v, shape) for v in values])
    dense_a = torch.stack(
        [
            torch.zeros(shape, dtype=v.dtype, device=device).index_put(tuple(indices), v, accumulate=True)
            for v in reference_values
        ]
    )
    b = torch.arange(1, batch_size * 6 + 1, dtype=initial_values.dtype, device=device).reshape(batch_size, 3, 2)
    actual = sparse_mm(a, b, assume_unique=True)
    expected = dense_a @ b
    upstream = torch.arange(1, actual.numel() + 1, dtype=initial_values.dtype, device=device).reshape(actual.shape)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, values, upstream),
        torch.autograd.grad(expected, reference_values, upstream),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)
