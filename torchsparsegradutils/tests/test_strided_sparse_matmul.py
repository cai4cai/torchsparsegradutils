import importlib

import pytest
import torch
from packaging.version import Version
from test_config import DEVICES, SPARSE_LAYOUTS, VALUE_DTYPES

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.utils import stack_csr


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("layout", SPARSE_LAYOUTS)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("strided_rhs,strided_gradient", [(True, False), (False, True), (True, True)])
def test_single_column_strides_dense_reference(device, dtype, layout, batched, strided_rhs, strided_gradient):
    indices = torch.tensor([[0, 0, 1, 2, 2], [0, 3, 1, 1, 2]], device=device)
    values = torch.tensor([2.0, 0.0, -3.0, 1.5, 4.0], dtype=dtype, device=device)
    a = torch.sparse_coo_tensor(indices, values, (3, 4)).coalesce()
    if layout == torch.sparse_csr:
        a = a.to_sparse_csr()
    if batched:
        a = stack_csr([a, a]) if layout == torch.sparse_csr else torch.stack([a, a]).coalesce()
    a = a.detach().requires_grad_()
    batch_shape = (2,) if batched else ()
    rhs_storage = torch.arange(1, 1 + (24 if batched else 12), dtype=dtype, device=device).reshape(*batch_shape, 4, 3)
    b = rhs_storage[..., 1:2]
    if not strided_rhs:
        b = b.contiguous()
    b.requires_grad_()
    gradient_storage = torch.arange(2, 2 + (18 if batched else 9), dtype=dtype, device=device).reshape(
        *batch_shape, 3, 3
    )
    gradient = gradient_storage[..., 2:3]
    if not strided_gradient:
        gradient = gradient.contiguous()
    assert b.is_contiguous() != strided_rhs
    assert gradient.is_contiguous() != strided_gradient
    original_strides = b.stride(), gradient.stride()

    dense_a = a.detach().to_dense().requires_grad_()
    actual = sparse_mm(a, b)
    expected = dense_a @ b
    torch.testing.assert_close(actual, expected)
    actual_a, actual_b = torch.autograd.grad(actual, (a, b), gradient)
    expected_a, expected_b = torch.autograd.grad(expected, (dense_a, b), gradient)
    mask = torch.zeros((3, 4), dtype=torch.bool, device=device)
    mask[indices[0], indices[1]] = True
    torch.testing.assert_close(actual_a.to_dense(), expected_a * mask)
    torch.testing.assert_close(actual_b, expected_b)
    assert (b.stride(), gradient.stride()) == original_strides


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("layout", SPARSE_LAYOUTS)
def test_contiguous_singleton_strides_dense_reference(device, dtype, layout):
    # A singleton column can have a non-unit stride while still being contiguous.
    dense_a = torch.tensor(
        [[2.0, 0.0, 0.0, 1.0], [0.0, -3.0, 0.0, 0.0], [0.0, 1.5, 4.0, 0.0]], device=device, dtype=dtype
    )
    a = dense_a.to_sparse_coo() if layout == torch.sparse_coo else dense_a.to_sparse_csr()
    a.requires_grad_()
    b = torch.arange(1, 21, device=device, dtype=dtype).reshape(5, 4).t()[:, :1].requires_grad_()
    gradient = torch.arange(1, 16, device=device, dtype=dtype).reshape(5, 3).t()[:, :1]
    assert b.is_contiguous() and b.stride() == (1, 4)
    assert gradient.is_contiguous() and gradient.stride() == (1, 3)
    assert b.contiguous().stride() == b.stride()
    assert gradient.contiguous().stride() == gradient.stride()
    dense_a.requires_grad_()
    actual = sparse_mm(a, b)
    expected = dense_a @ b
    torch.testing.assert_close(actual, expected)
    actual_a, actual_b = torch.autograd.grad(actual, (a, b), gradient)
    expected_a, expected_b = torch.autograd.grad(expected, (dense_a, b), gradient)
    torch.testing.assert_close(actual_a.to_dense(), expected_a * (dense_a != 0))
    torch.testing.assert_close(actual_b, expected_b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", SPARSE_LAYOUTS)
@pytest.mark.parametrize("operand_kind", ["column_slice", "multi_slice", "transpose", "contiguous", "singleton_stride"])
def test_stride_workaround_copies_only_affected_operands(device, layout, operand_kind, monkeypatch):
    dense_a = torch.tensor([[2.0, 0.0, 1.0], [0.0, -3.0, 4.0]], device=device, dtype=torch.float64)
    a = dense_a.to_sparse_coo() if layout == torch.sparse_coo else dense_a.to_sparse_csr()

    def operand(rows):
        data = torch.arange(1, 1 + rows * 4, device=device, dtype=dense_a.dtype)
        if operand_kind == "column_slice":
            return data.reshape(rows, 4)[:, 1:2]
        if operand_kind == "multi_slice":
            return data.reshape(rows, 4)[:, ::2]
        if operand_kind == "transpose":
            return data.reshape(4, rows).t()
        if operand_kind == "singleton_stride":
            return data.reshape(4, rows).t()[:, :1]
        return data.reshape(rows, 4)

    b = operand(3).requires_grad_()
    upstream = operand(2)
    calls = []
    native_mm = torch.sparse.mm

    def record_mm(sparse, dense):
        calls.append((sparse.layout, dense))
        return native_mm(sparse, dense)

    monkeypatch.setattr(torch.sparse, "mm", record_mm)
    actual = sparse_mm(a, b)
    (grad_b,) = torch.autograd.grad(actual, b, upstream)
    torch.testing.assert_close(actual, dense_a @ b)
    torch.testing.assert_close(grad_b, dense_a.t() @ upstream)
    assert len(calls) == 2
    assert calls[0][0] == layout
    assert calls[1][0] == (torch.sparse_csc if layout == torch.sparse_csr else torch.sparse_coo)
    for (_, passed), original in zip(calls, (b, upstream)):
        needs_copy = (
            torch.version.cuda is not None
            and Version(torch.version.cuda) < Version("12.6")
            and a.is_cuda
            and layout == torch.sparse_csr
            and operand_kind == "column_slice"
        )
        if needs_copy:
            assert passed.data_ptr() != original.data_ptr()
            assert passed.is_contiguous()
        else:
            # Observe the actual native-call boundary: unchanged storage and strides.
            assert passed.data_ptr() == original.data_ptr()
            assert passed.stride() == original.stride()


@pytest.mark.parametrize(
    "cuda_version,expected",
    [(None, False), ("11.8", True), ("12.4", True), ("12.5", True), ("12.6", False), ("13.2", False)],
)
def test_stride_guard_uses_cuda_build_not_torch_release(cuda_version, expected, monkeypatch):
    from torchsparsegradutils import _compat

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.version, "cuda", cuda_version)
            patch.setattr(torch, "__version__", "2.99.0")
            importlib.reload(_compat)
            assert _compat._SPARSE_MM_STRIDE_WORKAROUND is expected
    finally:
        importlib.reload(_compat)
