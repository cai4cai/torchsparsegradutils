# Sparsity-preserving gradient utility tools for PyTorch
A collection of utility functions to work with PyTorch sparse tensors. This is work-in-progress, here be dragons.

Currenly available features with backprop include:
- Memory efficient sparse mm with batch support (workaround for https://github.com/pytorch/pytorch/issues/41128)
- Sparse triangular solver with batch support (see discussion in https://github.com/pytorch/pytorch/issues/87358)
- Generic sparse linear solver (requires a non-differentiable backbone sparse solver)
- Generic sparse linear least-squares solver (requires a non-differentiable backbone sparse linear least-squares solver)
- Wrappers around [cupy sparse solvers](https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html#solving-linear-problems) (see discussion in https://github.com/pytorch/pytorch/issues/69538)
- Wrappers around [jax sparse solvers](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.sparse.linalg)
- Sparse multivariate normal distribution with sparse covariance and precision parameterisation, with reparameterised sampling (rsample)

Additional backbone solvers implemented in pytorch with no additional dependencies include:
- BICGSTAB (ported from [pykrylov](https://github.com/PythonOptimizers/pykrylov))
- CG (ported from [cornellius-gp/linear_operator](https://github.com/cornellius-gp/linear_operator))
- LSMR (ported from [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize))
- MINRES (ported from [cornellius-gp/linear_operator](https://github.com/cornellius-gp/linear_operator))

Additional features:
- Pairwise voxel encoder for encoding local neighbourhood relationships in a 3D spatial volume with multiple channels, into a sparse COO or CSR matrix.

Things that are missing may be listed as [issues](https://github.com/cai4cai/torchsparsegradutils/issues).

## Installation
The provided package can be installed using:

`pip install torchsparsegradutils` (TODO)

or

`pip install git+https://github.com/cai4cai/torchsparsegradutils`

## Unit Tests
A number of unittests are provided, which can be run as:

```
python -m pytest
```
 (Note that this also runs the tests from `unittest`)
