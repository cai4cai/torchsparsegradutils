---
title: 'Sparsity-preserving gradient utility tools for PyTorch'
tags:
  - Python
  - PyTorch
  - sparse tensors
  - machine learning
  - deep learning
authors:
  - name: Theodore Barfoot
    orcid: 0009-0005-2309-5238
    affiliation: 1
  - name: Ben Glocker
    orcid: 0000-0002-4897-9356
    affiliation: 2
  - name: Tom Vercauteren
    orcid: 0000-0003-1794-0456
    affiliation: 1
affiliations:
 - name: King's College London, London, UK
   index: 1
 - name: Imperial College London, London, UK
   index: 2
date: 03 February 2026
bibliography: paper.bib
---

# Summary

The `torchsparsegradutils` package provides differentiable sparse linear-algebra utilities for PyTorch [@pytorch] that preserve sparsity in returned gradients during backpropagation. While PyTorch supports sparse tensors, its default dense-equivalent backward semantics can densify gradients and make it difficult to optimise models with fixed sparsity patterns, such as sparse covariance or precision parameterisations.

The package provides sparse-dense matrix multiplication with sparse-gradient preservation, sparse triangular and generic linear system solvers (including BICGSTAB, CG, LSMR, and MINRES backends), optional CuPy [@cupy] and JAX [@jax] solver wrappers, sparse multivariate normal distributions with $\boldsymbol{L}\boldsymbol{L}^T$ and $\boldsymbol{L}\boldsymbol{D}\boldsymbol{L}^T$ parameterisations, and specialised encoders for spatial neighbourhood relationships in N-dimensional data.

The source code is available on GitHub at [https://github.com/cai4cai/torchsparsegradutils](https://github.com/cai4cai/torchsparsegradutils), with full documentation hosted at [https://torchsparsegradutils.readthedocs.io](https://torchsparsegradutils.readthedocs.io).

# Statement of need

Many scientific machine learning models benefit from representing large linear operators (e.g., neighbourhood couplings, precision matrices, sparse Jacobians) using sparse tensors to reduce memory and compute complexity. In high-dimensional settings, such as volumetric medical imaging, dense covariance or precision parameterisations are typically intractable, motivating sparse end-to-end parameterisations.

However, learning these models with gradient-based optimisation requires backpropagation through sparse linear algebra (matrix products, triangular solves, and linear system solves). PyTorch's default sparse semantics are not designed to preserve user-imposed sparsity structure during differentiation ([PyTorch issue #87448](https://github.com/pytorch/pytorch/issues/87448)), which can lead to memory blow-ups and prevent end-to-end optimisation of sparse probabilistic models.

`torchsparsegradutils` addresses this gap by implementing custom autograd functions for key sparse operators that return gradients only for stored nonzeros, enabling practical optimisation of models that rely on fixed sparse structure.

# State of the field

PyTorch [@pytorch] exposes sparse layouts (COO, CSR, and related formats) and implements a growing set of sparse operations. However, PyTorch's design goal is *dense-equivalent semantics* for sparse layouts: a guiding invariant is that applying an operation in sparse form should match applying it in dense form after conversion, including the backward function ([PyTorch issue #87448](https://github.com/pytorch/pytorch/issues/87448)). This makes it difficult to learn parameters that are intended to remain structurally sparse, because gradients may be produced for implicit zeros, or intermediate computations may densify.

PyTorch also provides `MaskedTensor`, which distinguishes specified and unspecified elements and is conceptually closer to the constrained-subspace interpretation of sparsity. However, `MaskedTensor` remains at prototype stage with incomplete operator coverage, and storing a full boolean mask incurs a significant memory overhead, partially negating the memory benefits of sparse index-based representations for large-scale problems.

Other libraries provide efficient sparse kernels but do not directly solve "sparsity-preserving gradients in PyTorch": SciPy [@scipy] provides mature sparse linear algebra but no automatic differentiation; CuPy [@cupy] and JAX [@jax] provide sparse solvers in their respective ecosystems but are not drop-in components for PyTorch autograd/training loops. GPyTorch [@gpytorch] targets scalable Gaussian process inference via kernel structure and approximations (e.g., inducing/structured methods) rather than arbitrary user-specified sparse covariance/precision factors. PyTorch Geometric’s torch_sparse [@pytorch_geometric] focuses on graph message-passing primitives rather than sparse covariance/precision modelling and differentiable sparse solves for probabilistic models.

# Software design

`torchsparsegradutils` is built around `torch.autograd.Function` operators that wrap PyTorch's forward sparse kernels but override the backward pass to preserve sparsity for selected inputs. This keeps the API close to standard PyTorch code while making sparsity preservation an explicit, opt-in choice.

Two design trade-offs shaped the implementation. First, the package targets *structure-preserving learning* over maximal operator coverage, focusing on sparse matrix products and sparse solves that support sparse multivariate normal sampling and related models. Second, it combines native PyTorch implementations (CG, BiCGSTAB, LSMR, MINRES) with optional CuPy and JAX wrappers so users can trade off portability and performance.

**Build vs. contribute justification.** PyTorch's current sparse semantics prioritise dense-equivalent behaviour ([PyTorch issue #87448](https://github.com/pytorch/pytorch/issues/87448)). In contrast, this package intentionally provides structure-preserving backward passes for specific operators to enable learning with fixed sparsity patterns. Because that is a semantic choice rather than just an implementation detail, the functionality is better delivered as an opt-in external library than as a change to PyTorch defaults.

# Research impact statement

This software provides an opt-in path to sparsity-preserving gradients for sparse linear algebra in PyTorch, enabling research prototypes that would otherwise be limited by dense gradients or densification. The package is already being used in ongoing medical-image segmentation projects, and the public repository provides tests, documentation, benchmarks, and issue tracking to support reuse and extension.

# Mathematics

## Sparse Matrix Operations

### Sparse Matrix Multiplication
The package implements sparse-dense matrix multiplication $\mathbf{C} = \mathbf{A}\mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{m \times n}$ is sparse and $\mathbf{B} \in \mathbb{R}^{n \times p}$ is dense. The forward pass uses PyTorch's native `torch.sparse.mm`, while the backward pass is reimplemented to preserve sparsity patterns in the gradients.

Given upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{C}} \in \mathbb{R}^{m\times p}$ from some scalar objective function $\mathcal{L}$, the chain rule gives:

**Gradient wrt $\mathbf{B}$** (dense):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = \mathbf{A}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{C}}.$$

**Gradient wrt $\mathbf{A}$** (sparse):
For a nonzero entry $\mathbf{A}_{ij}$,
$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}_{ij}} = \sum_{k=1}^p \Big(\frac{\partial \mathcal{L}}{\partial \mathbf{C}_{ik}}\Big) \mathbf{B}_{jk}.$$


### Sparse Linear System Solvers

The package provides multiple approaches for solving sparse linear systems
$$\mathbf{A}\mathbf{x} = \mathbf{B},$$
where $\mathbf{A}\in\mathbb{R}^{n\times n}$ is sparse, $\mathbf{B}\in\mathbb{R}^{n\times p}$ is dense (with $p=1$ for a single right-hand side), and $\mathbf{x}\in\mathbb{R}^{n\times p}$ is the dense solution. We support both direct triangular solves and iterative solvers (CG, BiCGSTAB, LSMR, MINRES). All are differentiable via the implicit function theorem.

Given upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} \in \mathbb{R}^{n\times p}$:

**Gradient wrt $\mathbf{b}$** (dense):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \mathbf{A}^{-\top} \frac{\partial \mathcal{L}}{\partial \mathbf{x}}.$$

**Gradient wrt $\mathbf{A}$** (sparse): The dense form would be
$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = - \left(\mathbf{A}^{-\top} \frac{\partial \mathcal{L}}{\partial \mathbf{x}}\right) \mathbf{x}^\top.$$
For a nonzero entry $\mathbf{A}_{ij}$ this becomes
$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}_{ij}} = - \sum_{k=1}^p \left(\mathbf{A}^{-\top} \frac{\partial \mathcal{L}}{\partial \mathbf{x}}\right)_{ik} \mathbf{x}_{jk} $$

## Sparse Multivariate Normal Distributions

The package implements sparse multivariate normal distributions $\boldsymbol{\eta} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \equiv \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Omega}^{-1})$ with two parameterisations for efficient sampling. Both methods transform standard normal samples $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ into samples from the desired multivariate normal distribution:

**$\boldsymbol{L}\boldsymbol{L}^{\top}$ Parameterisation**: Sparse lower triangular matrices $\boldsymbol{L}$ with positive diagonals:
$$
\boldsymbol{\eta}_{\boldsymbol{\Sigma}} = \boldsymbol{\mu} + \boldsymbol{L}_{\Sigma}\boldsymbol{\epsilon}, \quad \boldsymbol{\eta}_{\boldsymbol{\Omega}} = \boldsymbol{\mu} + \boldsymbol{L}_{\Omega}^{-T}\boldsymbol{\epsilon}
$$

**$\boldsymbol{L}\boldsymbol{D}\boldsymbol{L}^{\top}$ Parameterisation**: Sparse unit lower triangular matrices $\boldsymbol{L}$ and diagonal $\boldsymbol{D}$:
$$
\boldsymbol{\eta}_{\boldsymbol{\Sigma}} = \boldsymbol{\mu} + \boldsymbol{L}\boldsymbol{D}^{1/2}\boldsymbol{\epsilon}, \quad \boldsymbol{\eta}_{\boldsymbol{\Omega}} = \boldsymbol{\mu} + \boldsymbol{L}_\Omega^{-\top}\boldsymbol{D}_\Omega^{-1/2}\boldsymbol{\epsilon}
$$

We store $\boldsymbol{L}$ without its diagonal (strictly lower), and treat it as unit lower-triangular at use time. The $\boldsymbol{L}\boldsymbol{D}\boldsymbol{L}^{\top}$ parameterisation provides superior numerical stability for precision matrices by avoiding strict positive definiteness constraints.

# Usage Examples

Short examples are shown below; fuller worked examples are available in the ReadTheDocs quickstart.

```python
import torch
from torchsparsegradutils import sparse_mm, sparse_generic_solve
from torchsparsegradutils.distributions import SparseMultivariateNormal
from torchsparsegradutils.utils import (
    linear_cg,
    make_spd_sparse,
    rand_sparse,
    rand_sparse_tri,
)

n = 100
A = rand_sparse((n, n), nnz=500).requires_grad_(True)
sparse_mm(A, torch.randn(n, 8, requires_grad=True)).sum().backward()

A_spd, _ = make_spd_sparse(n, torch.sparse_coo, torch.float32, torch.int64, "cpu")
sparse_generic_solve(
    A_spd.requires_grad_(True),
    torch.randn(n),
    solve=linear_cg,
).sum().backward()

L = rand_sparse_tri(
    (n, n), nnz=300, upper=False, strict=True
).requires_grad_(True)
SparseMultivariateNormal(
    torch.zeros(n), diagonal=torch.rand(n), scale_tril=L
).rsample((10,)).sum().backward()
```

# Benchmarks

On the SuiteSparse Rothberg/cfd2 matrix ($123{,}440 \times 123{,}440$, 3.1M non-zeros), dense baselines and PyTorch's native COO backward pass ran out of memory, whereas `torchsparsegradutils` completed sparse matrix-multiplication backward in about 75 ms using 5.1 GB on one tested RTX 4090 setup (results vary by hardware). On the same setup, native COO iterative solvers were up to about 40$\times$ faster than CuPy wrappers because they avoid sparse-format conversion overhead; full benchmark scripts and hardware-specific results are available in the repository and ReadTheDocs benchmark documentation.

# AI usage disclosure

Generative AI tools were used during development of this software and manuscript. Various large language models were used to assist with code generation, refactoring, and test scaffolding for portions of the codebase, and AI assistance was used to draft and edit parts of the documentation and this manuscript. The repository was initiated prior to widespread AI coding assistant adoption, with AI tools incorporated during later development phases. All AI-assisted outputs were reviewed, edited, and validated by the human authors, who take responsibility for the final software and paper.

# Acknowledgements

We thank the PyTorch development team for foundational sparse tensor support. We also acknowledge upstream solver implementations and references used as starting points for iterative methods ([pykrylov](https://github.com/PythonOptimizers/pykrylov), [cornellius-gp/linear_operator](https://github.com/cornellius-gp/linear_operator), [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize)) [@saad2003iterative]. We thank Floris Laporte for his excellent tutorial on implementing sparse linear system solvers in PyTorch [@flaport2020sparse], which provided valuable insights for gradient computation strategies. This work was supported by the Wellcome/EPSRC Centre for Interventional and Surgical Sciences and the School of Biomedical Engineering & Imaging Sciences, King's College London.

# References
