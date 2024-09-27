import torch
import pytest

if torch.__version__ < (2, 4):
    pytest.skip(
        "Skipping test based on nested tensors since an old version of pytorch is used", allow_module_level=True
    )

from torchsparsegradutils import gather_mm, segment_mm

# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

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


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


# Define Tests


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
