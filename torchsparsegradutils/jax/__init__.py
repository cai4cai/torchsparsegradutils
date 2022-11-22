import importlib

jax_spec = importlib.util.find_spec("jax")
if jax_spec is None:
    have_jax = False
    import warnings

    warnings.warn(
        "\n\nAttempting to import an optional module in torchsparsegradutils that depends on jax but jax couldn't be imported.\n"
    )
else:
    have_jax = True
    from .jax_bindings import j2t, t2j, spmm_t4j, j2t_coo, t2j_coo, j2t_csr, t2j_csr

    from .jax_sparse_solve import sparse_solve_j4t

    __all__ = ["j2t", "t2j", "spmm_t4j", "j2t_coo", "t2j_coo", "j2t_csr", "t2j_csr", "sparse_solve_j4t"]
