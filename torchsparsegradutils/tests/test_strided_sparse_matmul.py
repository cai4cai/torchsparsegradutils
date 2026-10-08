import pytest
import torch

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.tests.test_config import DEVICES, SPARSE_LAYOUTS, VALUE_DTYPES
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
