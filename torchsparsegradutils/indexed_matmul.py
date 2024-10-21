import torch

try:
    import dgl.ops as dglops

    dgl_installed = True
except ImportError:
    dgl_installed = False


def segment_mm(a, b, seglen_a):
    """
    Performs matrix multiplication according to segments.
    See https://docs.dgl.ai/generated/dgl.ops.segment_mm.html

    Suppose ``seglen_a == [10, 5, 0, 3]``, the operator will perform
    four matrix multiplications::

        a[0:10] @ b[0], a[10:15] @ b[1],
        a[15:15] @ b[2], a[15:18] @ b[3]

    Args:
        a (torch.Tensor): The left operand, 2-D tensor of shape ``(N, D1)``
        b (torch.Tensor): The right operand, 3-D tensor of shape ``(R, D1, D2)``
        seglen_a (torch.Tensor): An integer tensor of shape ``(R,)``. Each element is the length of segments of input ``a``. The summation of all elements must be equal to ``N``.

    Returns:
        torch.Tensor: The output dense matrix of shape ``(N, D2)``
    """
    if torch.__version__ < (2, 4):
        raise NotImplementedError("PyTorch version is too old for nested tesors")

    if dgl_installed:
        # DGL is probably more computationally efficient
        # See https://github.com/pytorch/pytorch/issues/136747
        return dglops.segment_mm(a, b, seglen_a)

    if not a.dim() == 2 or not b.dim() == 3 or not seglen_a.dim() == 1:
        raise ValueError("Input tensors have unexpected dimensions")

    N, _ = a.shape
    R, D1, D2 = b.shape

    # Sanity check sizes
    if not a.shape[1] == D1 or not seglen_a.shape[0] == R:
        raise ValueError("Incompatible size for inputs")

    segidx_a = torch.cumsum(seglen_a[:-1], dim=0).cpu()

    # Ideally the conversions below to nested tensor would be handled natively
    nested_a = torch.nested.as_nested_tensor(torch.tensor_split(a, segidx_a, dim=0))
    nested_b = torch.nested.as_nested_tensor(torch.split(b, 1, dim=0)).reshape((R, D1, D2))

    # The actual gather matmul computation
    nested_ab = torch.matmul(nested_a, nested_b)

    # Convert back to tensors, again ideally this would be handled natively
    ab = torch.cat(nested_ab.unbind(), dim=0)
    return ab


def gather_mm(a, b, idx_b):
    """
    Gather data according to the given indices and perform matrix multiplication.
    See https://docs.dgl.ai/generated/dgl.ops.gather_mm.html

    Let the result tensor be ``c``, the operator conducts the following computation:

      c[i] = a[i] @ b[idx_b[i]]
      , where len(c) == len(idx_b)

    Args:
        a (torch.Tensor): A 2-D tensor of shape ``(N, D1)``
        b (torch.Tensor): A 3-D tensor of shape ``(R, D1, D2)``
        idx_b (torch.Tensor): An 1-D integer tensor of shape ``(N,)``.

    Returns:
        torch.Tensor: The output dense matrix of shape ``(N, D2)``
    """
    if torch.__version__ < (2, 4):
        raise NotImplementedError("PyTorch version is too old for nested tesors")

    if dgl_installed:
        # DGL is more computationally efficient
        # See https://github.com/pytorch/pytorch/issues/136747
        return dglops.gather_mm(a, b, idx_b)

    # Dependency free fallback
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not isinstance(idx_b, torch.Tensor):
        raise ValueError("Inputs should be instances of torch.Tensor")

    if not a.dim() == 2 or not b.dim() == 3 or not idx_b.dim() == 1:
        raise ValueError("Input tensors have unexpected dimensions")

    N = idx_b.shape[0]
    R, D1, D2 = b.shape

    # Sanity check sizes
    if not a.shape[0] == N or not a.shape[1] == D1:
        raise ValueError("Incompatible size for inputs")

    torchdevice = a.device
    src_idx = torch.arange(N, device=torchdevice)

    # Ideally the conversions below to nested tensor would be handled without for looops and without copy
    nested_a = torch.nested.as_nested_tensor([a[idx_b == i, :] for i in range(R)])
    src_idx_reshuffled = torch.cat([src_idx[idx_b == i] for i in range(R)])
    nested_b = torch.nested.as_nested_tensor(torch.split(b, 1, dim=0)).reshape((R, D1, D2))

    # The actual gather matmul computation
    nested_ab = torch.matmul(nested_a, nested_b)

    # Convert back to tensors, again, ideally this would be handled natively with no copy
    ab_segmented = torch.cat(nested_ab.unbind(), dim=0)
    ab = torch.empty((N, D2), device=torchdevice)
    ab[src_idx_reshuffled] = ab_segmented
    return ab
