import pytest
import torch

from torchsparsegradutils import gather_mm, indexed_matmul, segment_mm

# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))
if torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))

TEST_DATA = [
    # name  N, R, D1, D2
    ("small", 100, 32, 7, 10),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

STRATEGIES = ["auto", "pad", "expand"]
if indexed_matmul.dgl_installed:
    STRATEGIES.append("dgl")

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


@pytest.fixture(params=STRATEGIES)
def strategy(request):
    return request.param


def _skip_unsupported(device, value_dtype, strategy):
    if device.type == "mps" and value_dtype == torch.float64:
        pytest.skip("MPS does not support float64")
    if strategy == "dgl" and device.type == "mps":
        pytest.skip("DGL does not support MPS")


def _segment_mm_reference(a, b, seglen_a):
    idx_b = torch.repeat_interleave(torch.arange(b.shape[0], device=a.device), seglen_a)
    return _gather_mm_reference(a, b, idx_b)


def _gather_mm_reference(a, b, idx_b):
    return torch.stack([a[i] @ b[idx_b[i]] for i in range(a.shape[0])]).reshape(a.shape[0], b.shape[2])


# Define Tests


def test_segment_mm(device, value_dtype, index_dtype, shapes, strategy):
    _skip_unsupported(device, value_dtype, strategy)
    _, N, R, D1, D2 = shapes

    a = torch.randn((N, D1), dtype=value_dtype, device=device)
    b = torch.randn((R, D1, D2), dtype=value_dtype, device=device)
    seglen_a = torch.randint(low=1, high=int(N / R), size=(R,), dtype=index_dtype, device=device)
    seglen_a[-1] = N - seglen_a[:-1].sum()

    ab = segment_mm(a, b, seglen_a, strategy=strategy)

    assert ab.shape == (N, D2)
    assert ab.dtype == value_dtype
    assert torch.allclose(ab, _segment_mm_reference(a, b, seglen_a), atol=ATOL, rtol=RTOL)


def test_gather_mm(device, value_dtype, index_dtype, shapes, strategy):
    _skip_unsupported(device, value_dtype, strategy)
    _, N, R, D1, D2 = shapes

    a = torch.randn((N, D1), dtype=value_dtype, device=device)
    b = torch.randn((R, D1, D2), dtype=value_dtype, device=device)
    idx_b = torch.randint(low=0, high=R, size=(N,), dtype=index_dtype, device=device)

    ab = gather_mm(a, b, idx_b, strategy=strategy)

    assert ab.shape == (N, D2)
    assert ab.dtype == value_dtype
    assert torch.allclose(ab, _gather_mm_reference(a, b, idx_b), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("seglen", [[10, 5, 0, 3], [0, 18, 0, 0], [18]], ids=["zero_len", "one_segment", "R=1"])
def test_segment_mm_edge_cases(strategy, seglen):
    if strategy == "dgl":
        pytest.skip("Edge cases target the pure PyTorch strategies")
    seglen_a = torch.tensor(seglen)
    a = torch.randn(18, 4, dtype=torch.float64)
    b = torch.randn(len(seglen), 4, 2, dtype=torch.float64)

    assert torch.allclose(segment_mm(a, b, seglen_a, strategy=strategy), _segment_mm_reference(a, b, seglen_a))


def test_gather_mm_unused_matrices(strategy):
    if strategy == "dgl":
        pytest.skip("Edge cases target the pure PyTorch strategies")
    a = torch.randn(6, 3, dtype=torch.float64)
    b = torch.randn(5, 3, 2, dtype=torch.float64)
    idx_b = torch.tensor([4, 4, 1, 4, 1, 4])

    assert torch.allclose(gather_mm(a, b, idx_b, strategy=strategy), _gather_mm_reference(a, b, idx_b))


@pytest.mark.parametrize("strategy", ["pad", "expand"])
def test_segment_mm_gradcheck(strategy):
    a = torch.randn(18, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 4, 2, dtype=torch.float64, requires_grad=True)
    seglen_a = torch.tensor([10, 5, 0, 3])

    assert torch.autograd.gradcheck(lambda a, b: segment_mm(a, b, seglen_a, strategy=strategy), (a, b))


@pytest.mark.parametrize("strategy", ["pad", "expand"])
def test_gather_mm_gradcheck(strategy):
    a = torch.randn(18, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 4, 2, dtype=torch.float64, requires_grad=True)
    idx_b = torch.tensor([0, 1, 3, 3, 0, 1, 1, 0, 3, 0, 1, 3, 3, 0, 0, 1, 3, 1])

    assert torch.autograd.gradcheck(lambda a, b: gather_mm(a, b, idx_b, strategy=strategy), (a, b))


@pytest.mark.parametrize(
    "N, R, L, D1, D2, expected",
    [
        (100_000, 100, 1_000, 32, 32, True),  # balanced segments: padding is cheap
        (100_000, 100, 50_000, 32, 32, False),  # one dominant segment: padding is wasteful
        (100_000, 100, 1_000, 1, 1, False),  # tiny matrices: expanding b is cheap
    ],
)
def test_prefer_pad(N, R, L, D1, D2, expected):
    assert indexed_matmul._prefer_pad(N, R, L, D1, D2) == expected


@pytest.mark.parametrize("op", [segment_mm, gather_mm])
def test_auto_without_dgl_dispatches_to_pure_pytorch(monkeypatch, op):
    monkeypatch.setattr(indexed_matmul, "dgl_installed", False)
    calls = []
    monkeypatch.setattr(indexed_matmul, "_segment_mm_pad", lambda *args: calls.append("pad") or torch.zeros(4, 2))
    monkeypatch.setattr(indexed_matmul, "_gather_mm_expand", lambda *args: calls.append("expand") or torch.zeros(4, 2))
    monkeypatch.setattr(indexed_matmul, "_prefer_pad", lambda *args: True)

    a = torch.randn(4, 3)
    b = torch.randn(2, 3, 2)
    x = torch.tensor([2, 2]) if op is segment_mm else torch.tensor([0, 1, 1, 0])
    op(a, b, x)
    assert calls == ["pad"]

    monkeypatch.setattr(indexed_matmul, "_prefer_pad", lambda *args: False)
    op(a, b, x)
    assert calls == ["pad", "expand"]


@pytest.mark.parametrize("op", [segment_mm, gather_mm])
def test_auto_prefers_dgl_when_installed(monkeypatch, op):
    class FakeDGLOps:
        @staticmethod
        def segment_mm(a, b, x):
            return "dgl"

        gather_mm = segment_mm

    monkeypatch.setattr(indexed_matmul, "dgl_installed", True)
    monkeypatch.setattr(indexed_matmul, "dglops", FakeDGLOps, raising=False)

    a = torch.randn(4, 3)
    b = torch.randn(2, 3, 2)
    x = torch.tensor([2, 2]) if op is segment_mm else torch.tensor([0, 1, 1, 0])
    assert op(a, b, x) == "dgl"
    assert op(a, b, x, strategy="dgl") == "dgl"
    assert op(a, b, x, strategy="pad") != "dgl"
    assert op(a, b, x, strategy="expand") != "dgl"


@pytest.mark.parametrize(
    "strategy, device_type, expected",
    [
        ("auto", "cpu", True),
        ("auto", "cuda", True),
        ("auto", "mps", False),  # DGL does not support MPS: fall back to pure PyTorch
        ("dgl", "mps", True),  # explicit request is honoured
        ("pad", "cpu", False),
        ("expand", "cuda", False),
    ],
)
def test_use_dgl_is_device_aware(monkeypatch, strategy, device_type, expected):
    monkeypatch.setattr(indexed_matmul, "dgl_installed", True)
    assert indexed_matmul._use_dgl(strategy, torch.device(device_type)) == expected


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("op", [segment_mm, gather_mm])
def test_auto_on_mps_skips_dgl(monkeypatch, op):
    class FailingDGLOps:
        @staticmethod
        def segment_mm(a, b, x):
            raise AssertionError("DGL should not be called for MPS tensors")

        gather_mm = segment_mm

    monkeypatch.setattr(indexed_matmul, "dgl_installed", True)
    monkeypatch.setattr(indexed_matmul, "dglops", FailingDGLOps, raising=False)

    device = torch.device("mps")
    a = torch.randn(4, 3, device=device)
    b = torch.randn(2, 3, 2, device=device)
    x = torch.tensor([2, 2], device=device) if op is segment_mm else torch.tensor([0, 1, 1, 0], device=device)
    expected = op(a.cpu(), b.cpu(), x.cpu(), strategy="pad")
    assert torch.allclose(op(a, b, x).cpu(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("op", [segment_mm, gather_mm])
def test_dgl_strategy_requires_dgl(monkeypatch, op):
    monkeypatch.setattr(indexed_matmul, "dgl_installed", False)
    with pytest.raises(ImportError, match="requires DGL"):
        op(torch.randn(4, 3), torch.randn(2, 3, 2), torch.tensor([2, 2]), strategy="dgl")


@pytest.mark.parametrize("op", [segment_mm, gather_mm])
def test_unknown_strategy_raises(op):
    with pytest.raises(ValueError, match="Unknown strategy"):
        op(torch.randn(4, 3), torch.randn(2, 3, 2), torch.tensor([2, 2]), strategy="nested")
