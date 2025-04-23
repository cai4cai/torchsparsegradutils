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
    orchid: 0000-0002-4897-9356
    affiliation: 2
  - name: Tom Vercauteren
    orchid: 0000-0003-1794-0456
    affiliation: 1
affiliations:
 - name: King's College London, London, UK
   index: 1
 - name: Imperial College london, London, UK
date: 30 July 2024
bibliography: paper.bib
---

# Summary
The `torchsparsegradutils` package provides a collection of utility functions to work with PyTorch [@pytorch] sparse tensors, ensuring memory efficiency and supporting various sparsity preserving tensor operations with backpropagation. 
Current features include memory-efficient sparse matrix multiplication and sparse triangular solver with batch support, and generic sparse linear and least-squares solvers (BICGSTAB, CG, LSMR, and MINRES), as well as wrappers around CuPy [@cupy] and JAX [@jax] sparse solvers.
Building from this, the package provides a sparse multivariate normal distribution with sparse lower triangular covariance and precision parameterization (in COO or CSR format), supporting reparameterized sampling (`rsample`) with backpropagation and sparse gradient support.
Furthermore, the package offers a pairwise voxel encoder for encoding local neighborhood relationships in a 3D spatial volume with multiple channels into a sparse COO or CSR matrix.

# Statement of need

Sparse tensors are promising for many machine learning applications, especially with large-scale data and high-dimensional problems. However, using these operations in machine learning requires backpropagation, which PyTorch does not natively support for sparse gradients. This results in high memory usage for large data sets.

The torchsparsegradutils package provides sparse operations with sparse gradient support, allowing end-to-end optimization while maintaining a workable memory footprint. High-dimensional multivariate normal distributions are particularly useful in fields like medical imaging, where data can contain millions of points. Dense covariance models are computationally infeasible, making sparse representations crucial. This package enables the optimization of such distributions by preserving sparsity in gradients during backpropagation.

# Mathematics

This package implements a sparse multivariate normal distribution, parameterised by a mean $\bm{\mu}$ and either a sparse covariance matrix $\bm{\Sigma}$ or sparse precision matrix $\bm{\Omega}$, where $ \bm{\Omega} = \bm{\Sigma}^{-1} $:
$$
\bm{\eta} \sim \mathcal{N}(\bm{\mu}, \bm{\Sigma}) \sim \mathcal{N}(\bm{\mu}, \bm{\Omega})
$$
Sampling, via the standard normal $\bm{\epsilon} \sim \mathcal{N}(0, \bm{I}) $, then takes the typical form: 
$$
\bm{\eta} = \bm{\mu} + \bm{\Sigma}^{\frac{1}{2}}\bm{\epsilon} = \bm{\mu} + \bm{\Omega}^{-\frac{1}{2}}\bm{\epsilon}
$$
Using the $\bm{L} \bm{D} \bm{L}^T$ decomposition for both $ \bm{\Sigma} $ and  $ \bm{\Omega} $, the final sampling equations are:
$$
\bm{\eta}_{\bm{\Sigma}} = \bm{\mu} + \bm{L}\bm{D}^{\frac{1}{2}}\bm{\epsilon}, 
\quad
\bm{\eta}_{\bm{\Omega}}  = \bm{\mu} +  \bm{L}^{-T}\bm{D}^{-\frac{1}{2}}\bm{\epsilon}
$$

# Acknowledgements
# References