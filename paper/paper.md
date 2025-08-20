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
date: 15 August 2025
bibliography: paper.bib
---

# Summary

The `torchsparsegradutils` package provides gradient-preserving sparse tensor operations for PyTorch [@pytorch], addressing the critical limitation that PyTorch's native sparse operations do not support sparse gradients during backpropagation. This package enables memory-efficient optimization of high-dimensional models by maintaining sparsity patterns throughout the entire forward and backward pass computation.

Key features include: (1) memory-efficient sparse matrix multiplication with sparse gradient preservation, (2) sparse triangular and generic linear system solvers with multiple algorithmic backends (BICGSTAB, CG, LSMR, MINRES), (3) cross-platform sparse solver wrappers for CuPy [@cupy] and JAX [@jax], (4) sparse multivariate normal distributions with LL^T and LDL^T parameterizations supporting millions of dimensions, and (5) specialized encoders for spatial neighborhood relationships in volumetric data.

The package addresses PyTorch issue #41128 by implementing custom autograd functions that preserve sparsity in gradients, enabling practical training of models with sparse covariance structures on high-dimensional data where dense alternatives become computationally intractable.

# Statement of need

Sparse tensors are essential for computationally tractable machine learning on high-dimensional data, yet PyTorch's sparse tensor operations suffer from a critical limitation: gradients are computed in dense format even when the forward pass maintains sparsity. This results in prohibitive memory usage that scales quadratically with problem dimension rather than linearly with the number of non-zero elements.

For applications requiring sparse covariance modeling—such as medical imaging with millions of voxels, spatial statistics, and large-scale Gaussian processes—dense gradient computation renders optimization infeasible. A sparse covariance matrix with 1 million dimensions and 0.1% sparsity contains 1 billion non-zero elements, but dense gradients would require storing 1 trillion parameters.

This package solves this fundamental limitation by implementing custom autograd functions that preserve sparsity patterns throughout both forward and backward passes. Our sparse multivariate normal distribution enables optimization of million-dimensional Gaussian models with memory usage scaling as O(nnz) rather than O(n²), where nnz is the number of non-zero elements and n is the dimension.

Beyond memory efficiency, the package addresses the algorithmic challenge of sparse linear system solving by providing multiple iterative solver backends with automatic differentiation support. This enables end-to-end optimization of complex probabilistic models that would be computationally intractable with existing PyTorch sparse operations.

# Mathematics

## Sparse Matrix Operations

### Sparse Matrix Multiplication with Gradient Preservation

The package implements sparse matrix multiplication $\boldsymbol{C} = \boldsymbol{A}\boldsymbol{B}$ where $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ is sparse and $\boldsymbol{B} \in \mathbb{R}^{n \times p}$ is dense. The forward pass uses PyTorch's native `torch.sparse.mm`, but the backward pass is reimplemented to preserve sparsity patterns.

For sparse matrix $\boldsymbol{A}$ with indices $(\boldsymbol{A}_{\text{rows}}, \boldsymbol{A}_{\text{cols}})$ and values $\boldsymbol{A}_{\text{values}}$, the gradient with respect to $\boldsymbol{A}$'s values is computed as:

$$
\frac{\partial L}{\partial \boldsymbol{A}_{\text{values}}} = \left(\frac{\partial L}{\partial \boldsymbol{C}}[\boldsymbol{A}_{\text{rows}}, :] \odot \boldsymbol{B}[\boldsymbol{A}_{\text{cols}}, :]\right) \boldsymbol{1}
$$

where $\odot$ denotes element-wise multiplication, and $\boldsymbol{1}$ is a summation vector. This selective computation maintains O(nnz) memory complexity for gradients rather than O(mn).

### Sparse Linear System Solvers

The package provides multiple approaches for solving sparse linear systems $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$:

**Triangular Systems**: For triangular matrices $\boldsymbol{T} \in \mathbb{R}^{n \times n}$, we use forward/backward substitution. The gradient computation during backpropagation follows:

$$
\frac{\partial L}{\partial \boldsymbol{b}} = \boldsymbol{T}^{-T}\frac{\partial L}{\partial \boldsymbol{x}}, \quad \frac{\partial L}{\partial \boldsymbol{T}_{\text{values}}} = -\left(\boldsymbol{T}^{-T}\frac{\partial L}{\partial \boldsymbol{x}}\right) \otimes \boldsymbol{x}^T
$$

where $\otimes$ denotes the outer product restricted to non-zero indices of $\boldsymbol{T}$.

**Generic Iterative Solvers**: For general sparse systems, we implement iterative methods including Conjugate Gradient (CG), Biconjugate Gradient Stabilized (BICGSTAB), Least Squares Minimal Residual (LSMR), and Minimal Residual (MINRES). These methods solve $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ iteratively while maintaining differentiability through the implicit function theorem.

## Sparse Multivariate Normal Distributions

The package implements sparse multivariate normal distributions $\boldsymbol{\eta} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \equiv \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Omega}^{-1})$ with two parameterizations for efficient sampling from $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$:

**LDL^T Parameterization**: Using sparse unit lower triangular matrices $\boldsymbol{L}$ and diagonal $\boldsymbol{D}$:
$$
\boldsymbol{\eta}_{\boldsymbol{\Sigma}} = \boldsymbol{\mu} + \boldsymbol{L}\boldsymbol{D}^{1/2}\boldsymbol{\epsilon}, \quad \boldsymbol{\eta}_{\boldsymbol{\Omega}} = \boldsymbol{\mu} + \boldsymbol{L}_\Omega^{-T}\boldsymbol{D}_\Omega^{-1/2}\boldsymbol{\epsilon}
$$

**LL^T Parameterization**: Using sparse lower triangular matrices $\boldsymbol{L}$ with positive diagonals:
$$
\boldsymbol{\eta}_{\boldsymbol{\Sigma}} = \boldsymbol{\mu} + \boldsymbol{L}_{\Sigma}\boldsymbol{\epsilon}, \quad \boldsymbol{\eta}_{\boldsymbol{\Omega}} = \boldsymbol{\mu} + \boldsymbol{L}_{\Omega}^{-T}\boldsymbol{\epsilon}
$$

The LDL^T parameterization provides superior numerical stability for precision matrices by avoiding strict positive definiteness constraints.

# Acknowledgements

The authors acknowledge the PyTorch development team for providing the foundational sparse tensor infrastructure. We thank the SciPy [@scipy], CuPy [@cupy], and JAX [@jax] communities for high-performance sparse linear algebra implementations. Algorithm implementations adapt and extend methods from pykrylov (BICGSTAB), cornellius-gp/linear_operator (CG, MINRES), and pytorch-minimize (LSMR) [@saad2003iterative]. This work was supported by the Wellcome/EPSRC Centre for Interventional and Surgical Sciences and the School of Biomedical Engineering & Imaging Sciences, King's College London.

# References