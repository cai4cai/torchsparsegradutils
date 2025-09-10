import itertools

import pytest
import torch

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.utils import rand_sparse, rand_sparse_tri

# NOTE: tests pass using torch.sparse.mm for unbatched sparse COO and CSR matrices
# from torch.sparse import mm as sparse_mm


# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

TEST_DATA = [
    # name  A_shape, B_shape, A_nnz
    ("unbat", (4, 6), (6, 2), 8),  # unbatched
    ("unbat", (8, 16), (16, 10), 32),  # -
    ("unbat", (7, 4), (4, 9), 14),  # -
    ("bat", (1, 4, 6), (1, 6, 2), 8),  # batched
    ("bat", (4, 8, 16), (4, 16, 10), 32),  # -
    ("bat", (11, 7, 4), (11, 4, 9), 14),  # -
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
# NOTE: torch.float16  - Only works for COO on CPU
# RuntimeError: "addmm_sparse_cuda" not implemented for 'Half'


RTOL = 1e-6


# Define Test Names:
def data_id(shapes):
    return shapes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def layout_id(layout):
    return "coo" if layout == torch.sparse_coo else "csr"


# Define Fixtures


@pytest.fixture(params=TEST_DATA, ids=[data_id(d) for d in TEST_DATA])
def shapes(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=[dtype_id(d) for d in VALUE_DTYPES])
def value_dtype(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=[dtype_id(d) for d in INDEX_DTYPES])
def index_dtype(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=LAYOUTS, ids=[layout_id(d) for d in LAYOUTS])
def layout(request):
    return request.param


################################## Forwards and  Backwards Tests: #####################################


def test_sparse_mm_forward(layout, device, value_dtype, index_dtype, shapes):
    _, A_shape, B_shape, A_nnz = shapes
    A = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Ad = A.to_dense()

    res_sparse = sparse_mm(A, B)  # both results are dense
    res_dense = torch.matmul(Ad, B)

    assert torch.allclose(res_sparse, res_dense, rtol=RTOL)


def test_sprase_mm_backward(layout, device, value_dtype, index_dtype, shapes, is_backward=False):
    _, A_shape, B_shape, A_nnz = shapes
    As1 = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    Ad2 = As1.detach().clone().to_dense()  # detach and clone to create seperate graph

    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd2 = Bd1.detach().clone()

    As1.requires_grad_()
    Ad2.requires_grad_()
    Bd1.requires_grad_()
    Bd2.requires_grad_()

    res1 = sparse_mm(As1, Bd1)  # both results are dense
    res2 = torch.matmul(Ad2, Bd2)

    # Generate random gradients for the backward pass
    grad_output = torch.rand_like(res1, dtype=value_dtype, device=device)

    res1.backward(grad_output)
    res2.backward(grad_output)

    nz_mask = As1.grad.to_dense() != 0.0

    # Check sparsity of gradientns
    assert As1.grad.layout == layout
    # NOTE: _nnz per batch element for CSR and for whole tensor in COO
    if layout is torch.sparse_csr or len(As1.shape) == 2:
        assert As1.grad._nnz() == A_nnz
    elif layout is torch.sparse_coo and len(As1.shape) == 3:  # ie batched
        assert As1.grad._nnz() == A_nnz * As1.shape[0]
    else:
        raise ValueError(f"Unsupported layout: {layout} or shape: {As1.shape}")

    # Check gradient values
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, rtol=RTOL)


################################## Conditional Gradient Tests: #####################################


def test_sparse_mm_conditional_gradients(layout, device, value_dtype, index_dtype):
    # NOTE: This test is not actually testing "if ctx.needs_input_grad" -- it passes regardless
    # simple small shape
    A = rand_sparse((5, 4), 6, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    # coalesce for COO so indices are valid
    if layout == torch.sparse_coo:
        A = A.coalesce()
    B = torch.randn(4, 3, dtype=value_dtype, device=device)

    # Case 1: only B requires grad → A.grad should be None, B.grad non-None
    A1 = A.detach().clone()
    B1 = B.detach().clone().requires_grad_()
    out1 = sparse_mm(A1, B1)
    out1.sum().backward()
    assert not hasattr(A1, "grad") or A1.grad is None
    assert B1.grad is not None

    # Case 2: only A requires grad → B.grad should be None, A.grad non-None
    A2 = A.detach().clone().requires_grad_()
    B2 = B.detach().clone()
    out2 = sparse_mm(A2, B2)
    out2.sum().backward()
    assert A2.grad is not None
    assert not hasattr(B2, "grad") or B2.grad is None


################################## Bad Input Tests: #####################################

BAD_TEST_DATA = [
    # name, A, B, expected_error, error_msg
    ("bad_tensor", 5, torch.rand(6, 2), ValueError, "Both A and B should be instances of torch.Tensor"),
    (
        "bad_dim_A",
        torch.tensor([0, 1]).to_sparse(),
        torch.rand(6, 2),
        ValueError,
        "Both A and B should be at least 2-dimensional tensors",
    ),
    (
        "bad_dim_B",
        torch.rand(4, 6).to_sparse(),
        torch.rand(6),
        ValueError,
        "Both A and B should be at least 2-dimensional tensors",
    ),
    (
        "bad_dim_mismatch",
        torch.rand(4, 6).to_sparse(),
        torch.rand(1, 6, 2),
        ValueError,
        "A and B must both be 2D or both be 3D tensors",
    ),
    (
        "bad_format",
        torch.rand(4, 6).to_sparse_csc(),
        torch.rand(6, 2),
        ValueError,
        "A should be in either COO or CSR sparse format",
    ),
    (
        "bad_batch",
        torch.stack([torch.rand(4, 6).to_sparse(), torch.rand(4, 6).to_sparse()]),
        torch.rand(1, 6, 2),
        ValueError,
        "If batched, A and B must have the same batch size",
    ),
]


@pytest.fixture(params=BAD_TEST_DATA, ids=[data_id(d) for d in BAD_TEST_DATA])
def bad_inputs(request):
    return request.param


def test_sparse_mm_error(bad_inputs):
    _, A, B, expected_error, error_msg = bad_inputs
    with pytest.raises(expected_error) as e:
        sparse_mm(A, B)
    assert str(e.value) == error_msg


################################## Memory Usage Tests: #####################################

MEM_USAGE_TEST_DATA = [
    # name,   A_shape,     B_shape,      A_nnz
    ("mem_small", (2000, 2000), (2000, 128), 4000),
    ("mem_big", (10000, 10000), (10000, 512), 20000),
]


@pytest.mark.parametrize(
    "mem_shapes,layout",
    list(itertools.product(MEM_USAGE_TEST_DATA, [torch.sparse_coo, torch.sparse_csr])),
    ids=[
        f"{data_id(d[0])}_{'coo' if d[1] == torch.sparse_coo else 'csr'}"
        for d in itertools.product(MEM_USAGE_TEST_DATA, [torch.sparse_coo, torch.sparse_csr])
    ],
)
def test_sparse_mm_memory_advantage(device, value_dtype, index_dtype, mem_shapes, layout):
    name, A_shape, B_shape, A_nnz = mem_shapes

    # only measure on CUDA
    if device.type != "cuda":
        pytest.skip("requires CUDA for peak memory measurement")

    if layout == torch.sparse_csr:
        pytest.skip("CSR layout fails this test due to crow unpacking")

    # build one sparse matrix + dense B
    A = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    if layout == torch.sparse_coo:
        A = A.coalesce()
    B = torch.randn(*B_shape, dtype=value_dtype, device=device)

    # -------------------------
    # 1) vanilla torch.sparse.mm
    # -------------------------
    torch.cuda.reset_peak_memory_stats(device)
    A1 = A.detach().clone().requires_grad_()
    B1 = B.clone().requires_grad_()
    out1 = torch.sparse.mm(A1, B1).sum()
    out1.backward()
    mem_torch_sparse_mm = torch.cuda.max_memory_allocated(device)

    # -------------------------
    # 2) your sparse_mm
    # -------------------------
    torch.cuda.reset_peak_memory_stats(device)
    A2 = A.detach().clone().requires_grad_()
    B2 = B.clone().requires_grad_()
    out2 = sparse_mm(A2, B2).sum()
    out2.backward()
    mem_tsgu_sparse_mm = torch.cuda.max_memory_allocated(device)

    # # -------------------------
    # # 3) Dense matul reference
    # # -------------------------
    # torch.cuda.reset_peak_memory_stats(device)
    # Ad, Bd = A.to_dense().detach().clone().requires_grad_(), B.clone().requires_grad_()
    # out3 = torch.matmul(Ad, Bd).sum()
    # out3.backward()
    # mem_torch_dense_mm = torch.cuda.max_memory_allocated(device)

    # # prints for debugging
    # print(
    #     f"\nLayout={layout}, "
    #     f"torch sparse.mm={mem_torch_sparse_mm / 1e6:.1f}MB, "
    #     f"tsgu sparse_mm={mem_tsgu_sparse_mm / 1e6:.1f}MB, "
    #     f"dense={mem_torch_dense_mm / 1e6:.1f}MB"
    # )

    # sanity: we still got sparse grads
    assert A2.grad.layout == layout

    # confirm memory saving
    assert mem_tsgu_sparse_mm < mem_torch_sparse_mm, (
        f"sparse_mm should use less GPU memory than torch.sparse.mm "
        f"({mem_tsgu_sparse_mm / 1e6:.1f}MB vs {mem_torch_sparse_mm / 1e6:.1f}MB)"
    )


def test_sparse_mm_optimize_A_multiple_steps(layout, device, value_dtype, index_dtype):
    # small problem
    N, M, NNZ = 30, 10, 50
    A = rand_sparse((N, N), NNZ, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    if layout == torch.sparse_coo:
        A = A.coalesce()
    B = torch.randn(N, M, dtype=value_dtype, device=device)

    # make A require gradients on its values
    A.requires_grad_()
    lr = 1e-2

    for step in range(3):
        # forward
        # out = torch.sparse.mm(A, B)
        out = sparse_mm(A, B)
        loss = out.sum()

        # backward
        loss.backward()
        assert A.grad is not None
        assert B.grad is None

        # grab the old and new value tensors
        if layout == torch.sparse_coo:
            vals = A._values()
            gvals = A.grad._values()
        else:  # CSR
            vals = A.values()
            gvals = A.grad.values()

        old = vals.clone()

        # gradient step on A.values()
        with torch.no_grad():
            vals.sub_(lr * gvals)

        # zero gradients for next step
        A.grad = None  # NOTE: only COO CUDA seems to care about this, both for torch.sparse.mm and sparse_mm
        # w/o this CUDA COO: RuntimeError: The size of tensor a (50) must match the size of tensor b (100) at non-singleton dimension 0

        new = vals
        # confirm that the values actually changed
        assert not torch.allclose(old, new), f"Step {step}: A.values did not update"


def test_sparse_mm_memory_stability(layout, device):
    if device.type != "cuda":
        pytest.skip("requires CUDA for this test")
    N, M, NNZ = 500, 50, 200
    A = rand_sparse((N, N), NNZ, layout, device=device)
    B = torch.randn(N, M, device=device)

    A.requires_grad_()
    torch.cuda.reset_peak_memory_stats(device)
    peaks = []
    for _ in range(100):
        # out = torch.sparse.mm(A, B).sum()
        out = sparse_mm(A, B).sum()
        out.backward()
        peaks.append(torch.cuda.max_memory_allocated(device))
        A.grad = None  # NOTE: sparse_mm fails for COO without it, CSR passes regardless
        # NOTE: torch.sparse.mm doesn't seem to care "as much" passes at < 1.05

    # We expect some initial variance but no monotonic increase
    assert max(peaks) / min(peaks) < 1.2, f"Memory is growing: {peaks!r}"


def test_sparse_mm_double_backward_error(layout, device):
    N, M, NNZ = 20, 5, 10
    A = rand_sparse((N, N), NNZ, layout, device=device)
    A.requires_grad_()
    B = torch.randn(N, M, device=device)

    out = sparse_mm(A, B)
    loss = out.sum()

    # first backward should succeed
    loss.backward()
    # second backward on the *same* loss should raise the usual PyTorch error
    with pytest.raises(RuntimeError, match=".*second time.*"):
        loss.backward()


def test_torch_sparse_mm_errors_on_batched(layout, device, value_dtype, index_dtype):
    # Batched sparse mm unsupported in native torch.sparse.mm
    batch, M, N, P, A_nnz = 2, 4, 6, 2, 8
    A = rand_sparse((batch, M, N), A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    if layout == torch.sparse_coo:
        A = A.coalesce()
    B = torch.randn(batch, N, P, dtype=value_dtype, device=device)

    # native torch.sparse.mm should error on batched inputs
    with pytest.raises(RuntimeError):
        torch.sparse.mm(A, B)
