import importlib

cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec is None:
    have_cupy = False
    import warnings

    warnings.warn(
        "\n\nAttempting to import an optional module in torchsparsegradutils that depends on cupy but cupy couldn't be imported -> falling back to numpy.\n"
    )
else:
    have_cupy = True

from .cupy_bindings import c2t_coo, t2c_coo, c2t_csr, t2c_csr, _get_array_modules
from .cupy_sparse_solve import sparse_solve_c4t

__all__ = ["c2t_coo", "t2c_coo", "c2t_csr", "t2c_csr", "_get_array_modules", "sparse_solve_c4t"]
