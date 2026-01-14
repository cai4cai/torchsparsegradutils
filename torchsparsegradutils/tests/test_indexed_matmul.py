import pytest
import torch
from test_config import DEVICES, INDEX_DTYPES, VALUE_DTYPES, Tolerances

from torchsparsegradutils import gather_mm, segment_mm

TEST_DATA = [
    # name  N, R, D1, D2
    ("small", 100, 32, 7, 10),
]


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


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


# Define Tests


def test_segment_mm(device, value_dtype, index_dtype, shapes):
    # NOTE: value_dtype fixture not used - gather_mm has a bug with float64
    # TODO: Fix gather_mm to support float64, then enable dtype testing here
    _, N, R, D1, D2 = shapes

    a = torch.randn((N, D1), device=device)
    b = torch.randn((R, D1, D2), device=device)
    seglen_a = torch.randint(low=1, high=int(N / R), size=(R,), device=device)
    seglen_a[-1] = N - seglen_a[:-1].sum()

    ab = segment_mm(a, b, seglen_a)

    atol, rtol = Tolerances.direct(torch.float32)  # default dtype
    k = 0
    for i in range(R):
        for j in range(seglen_a[i]):
            assert torch.allclose(ab[k, :].squeeze(), a[k, :].squeeze() @ b[i, :, :].squeeze(), atol=atol, rtol=rtol)
            k += 1


def test_gather_mm(device, value_dtype, index_dtype, shapes):
    # NOTE: value_dtype fixture not used - gather_mm has a bug with float64
    # TODO: Fix gather_mm to support float64, then enable dtype testing here
    _, N, R, D1, D2 = shapes

    a = torch.randn((N, D1), device=device)
    b = torch.randn((R, D1, D2), device=device)
    idx_b = torch.randint(low=0, high=R, size=(N,), device=device)

    ab = gather_mm(a, b, idx_b)

    atol, rtol = Tolerances.direct(torch.float32)  # default dtype
    for i in range(N):
        assert torch.allclose(ab[i, :].squeeze(), a[i, :].squeeze() @ b[idx_b[i], :, :].squeeze(), atol=atol, rtol=rtol)


def test_segment_mm_raises_on_old_pytorch():
    """Test that segment_mm raises NotImplementedError on PyTorch < 2.4."""
    from unittest.mock import patch

    from torchsparsegradutils import indexed_matmul

    a = torch.randn(10, 4)
    b = torch.randn(2, 4, 3)
    seglen_a = torch.tensor([5, 5])

    with patch.object(torch, "__version__", "2.3.0"):
        with pytest.raises(NotImplementedError, match="PyTorch version is too old"):
            indexed_matmul.segment_mm(a, b, seglen_a)


def test_gather_mm_raises_on_old_pytorch():
    """Test that gather_mm raises NotImplementedError on PyTorch < 2.4."""
    from unittest.mock import patch

    from torchsparsegradutils import indexed_matmul

    a = torch.randn(3, 4)
    b = torch.randn(2, 4, 5)
    idx_b = torch.tensor([0, 1, 0])

    with patch.object(torch, "__version__", "2.3.0"):
        with pytest.raises(NotImplementedError, match="PyTorch version is too old"):
            indexed_matmul.gather_mm(a, b, idx_b)
