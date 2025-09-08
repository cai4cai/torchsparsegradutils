import torch
from torch.profiler import ProfilerActivity, profile, record_function

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.utils import rand_sparse


def profile_sparse_mm(
    A_shape,
    B_shape,
    A_nnz,
    layout=torch.sparse_coo,
    device=torch.device("cuda"),
    value_dtype=torch.float32,
    index_dtype=torch.int64,
    row_limit: int = 20,
):
    """
    Profiles forward+backward of sparse_mm on a random sparse matrix.

    Args:
        A_shape (tuple): shape of A, e.g. (N, M) or (batch, N, M)
        B_shape (tuple): shape of B, must be compatible with A: (M, P) or (batch, M, P)
        A_nnz (int): number of nonzeros in A
        layout: torch.sparse_coo or torch.sparse_csr
        device: torch.device, must be "cuda" to profile CUDA memory
        value_dtype: torch.float32 or torch.float64
        index_dtype: torch.int32 or torch.int64
        row_limit: how many rows to print in the profiler table
    """
    if device.type != "cuda":
        raise RuntimeError("Profiler memory breakdown requires CUDA device")

    # 1) build random sparse A and dense B
    A = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    if layout == torch.sparse_coo:
        A = A.coalesce()
    B = torch.randn(*B_shape, dtype=value_dtype, device=device)

    # make sure we track gradients
    A.requires_grad_()
    B.requires_grad_()

    # 2) profile forward+backward
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("sparse_mm_forward_and_backward"):
            out = sparse_mm(A, B)
            # you can sum or pick any scalar reduction
            out.sum().backward()

    # 3) print a table sorted by the CUDA memory usage of each op
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=row_limit))
