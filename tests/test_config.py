import torch

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32, torch.int64]
SPARSE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr]


def devices_match(actual: torch.device, expected: torch.device) -> bool:
    actual = torch.device(actual)
    expected = torch.device(expected)
    if actual.type != expected.type:
        return False
    if expected.index is None:
        return True
    return actual.index == expected.index


class Tolerances:
    DIRECT_ATOL_F64 = 1e-6
    DIRECT_RTOL_F64 = 1e-6

    ITERATIVE_ATOL_F64 = 1e-3
    ITERATIVE_RTOL_F64 = 1e-4

    LSTSQ_RTOL_F64 = 1e-2

    F32_FACTOR = 100

    @classmethod
    def direct(cls, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float32:
            return (cls.DIRECT_ATOL_F64 * cls.F32_FACTOR, cls.DIRECT_RTOL_F64 * cls.F32_FACTOR)
        return (cls.DIRECT_ATOL_F64, cls.DIRECT_RTOL_F64)

    @classmethod
    def iterative(cls, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float32:
            return (cls.ITERATIVE_ATOL_F64 * cls.F32_FACTOR, cls.ITERATIVE_RTOL_F64 * cls.F32_FACTOR)
        return (cls.ITERATIVE_ATOL_F64, cls.ITERATIVE_RTOL_F64)

    @classmethod
    def lstsq(cls, dtype: torch.dtype) -> float:
        if dtype == torch.float32:
            return cls.LSTSQ_RTOL_F64 * 10
        return cls.LSTSQ_RTOL_F64


def get_confidence_level(
    device: torch.device,
    value_dtype: torch.dtype,
    is_batched: bool = False,
    is_covariance_test: bool = False,
) -> float:
    if device.type == "cuda" and value_dtype == torch.float32:
        return 0.999 if is_covariance_test else 0.99
    if value_dtype == torch.float32:
        return 0.99
    return 0.95 if is_batched else 0.90
