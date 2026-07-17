import pytest
import torch
from test_config import DEVICES

from torchsparsegradutils import gather_mm, segment_mm
from torchsparsegradutils._dispatch import backend_available

# Identify Testing Parameters
TEST_DATA = [
    # name  N, R, D1, D2
    ("small", 100, 32, 7, 10),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

ATOL = 1e-6  # relaxed tolerance to allow for float32
RTOL = 1e-4


# Define Test Names:
def data_id(shapes):
    return shapes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


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


# spec/commit.md Phase 3 commit 18: segment_mm/gather_mm now dispatch to
# tsgu::grouped_gemm, which is CUDA-only (architecture.md §4: "CUDA-required
# at runtime" -- no CPU implementation ships). DEVICES still includes cpu
# (other suites use it), so the `device` fixture below skips cleanly on a
# machine without a compatible CUDA backend instead of hitting the op's
# NotImplementedError -- the expected degraded-mode outcome (spec/testing.md
# "CPU CI (no GPU)"), mirroring tests/test_sparse_matmul.py's identical
# regating from commit 15. The extra dispatch-key check keeps this suite
# skipping (not erroring) when the backend package is loaded but the cuda/
# tree has not yet been rebuilt with the grouped_gemm kernel (commit 18 is
# developed in two concurrent lanes).
_grouped_gemm_ready = backend_available() and torch._C._dispatch_has_kernel_for_dispatch_key(
    "tsgu::grouped_gemm", "CUDA"
)
_CUDA_DEVICES = [d for d in DEVICES if d.type == "cuda"] if _grouped_gemm_ready else []


@pytest.fixture(params=_CUDA_DEVICES or [None], ids=[device_id(d) for d in _CUDA_DEVICES] or ["cuda-unavailable"])
def device(request):
    if request.param is None:
        pytest.skip(
            "segment_mm/gather_mm require a CUDA device and a loaded tsgu backend "
            "(spec/commit.md Phase 3 commit 18: tsgu::grouped_gemm is CUDA-only) -- "
            "none available on this machine."
        )
    return request.param


# Define Tests
#
# The test bodies below are the frozen-behaviour arbiter: they compare
# against explicit per-row/per-segment reference loops, unchanged from the
# pre-rewrite suite except for the CUDA regating above.


def test_segment_mm(device, value_dtype, index_dtype, shapes):
    _, N, R, D1, D2 = shapes

    a = torch.randn((N, D1), device=device)
    b = torch.randn((R, D1, D2), device=device)
    seglen_a = torch.randint(low=1, high=int(N / R), size=(R,), device=device)
    seglen_a[-1] = N - seglen_a[:-1].sum()

    ab = segment_mm(a, b, seglen_a)

    k = 0
    for i in range(R):
        for j in range(seglen_a[i]):
            assert torch.allclose(ab[k, :].squeeze(), a[k, :].squeeze() @ b[i, :, :].squeeze(), atol=ATOL, rtol=RTOL)
            k += 1


def test_gather_mm(device, value_dtype, index_dtype, shapes):
    _, N, R, D1, D2 = shapes

    a = torch.randn((N, D1), device=device)
    b = torch.randn((R, D1, D2), device=device)
    idx_b = torch.randint(low=0, high=R, size=(N,), device=device)

    ab = gather_mm(a, b, idx_b)

    for i in range(N):
        assert torch.allclose(ab[i, :].squeeze(), a[i, :].squeeze() @ b[idx_b[i], :, :].squeeze(), atol=ATOL, rtol=RTOL)
