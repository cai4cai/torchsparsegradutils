Mathematical Background
=======================

This section provides the mathematical foundations underlying the algorithms and methods implemented in torchsparsegradutils.

Sparse Matrix Representations
------------------------------

Coordinate Format (COO)
~~~~~~~~~~~~~~~~~~~~~~~

The coordinate format stores sparse matrices using three arrays:

- **indices**: A 2×nnz tensor containing row and column indices
- **values**: A 1D tensor of length nnz containing the non-zero values
- **size**: The shape of the full matrix

For a sparse matrix :math:`A \in \mathbb{R}^{m \times n}` with nnz non-zero elements:

.. math::

   A = \sum_{k=1}^{\text{nnz}} v_k \cdot e_{i_k} e_{j_k}^T

where :math:`v_k` are the values, :math:`(i_k, j_k)` are the indices, and :math:`e_i` are standard basis vectors.

Compressed Sparse Row (CSR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSR format uses three arrays:

- **indices**: Column indices of non-zero elements
- **indptr**: Row pointers indicating start of each row
- **values**: Non-zero values

The CSR format allows efficient row-wise operations and is preferred for matrix-vector multiplication.

Sparse Matrix Operations
-------------------------

Sparse Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For sparse matrix :math:`A` and dense matrix :math:`B`, the multiplication :math:`C = AB` is computed as:

.. math::

   C_{ij} = \sum_{k} A_{ik} B_{kj}

The key insight in our implementation is preserving sparsity in gradients. When computing :math:`\frac{\partial L}{\partial A}` for some loss :math:`L`, we ensure the gradient maintains the same sparsity pattern as :math:`A`.

**Gradient Preservation**

For the backward pass, we compute:

.. math::

   \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T

The gradient :math:`\frac{\partial L}{\partial A}` has the same sparsity pattern as :math:`A`, which is crucial for memory efficiency in sparse learning scenarios.

Linear System Solvers
----------------------

Triangular Systems
~~~~~~~~~~~~~~~~~~

For lower triangular systems :math:`Lx = b`, we use forward substitution:

.. math::

   x_i = \frac{1}{L_{ii}} \left( b_i - \sum_{j=1}^{i-1} L_{ij} x_j \right)

For upper triangular systems :math:`Ux = b`, we use backward substitution:

.. math::

   x_i = \frac{1}{U_{ii}} \left( b_i - \sum_{j=i+1}^{n} U_{ij} x_j \right)

**Batch Processing**

For batched systems :math:`LX = B` where :math:`X, B \in \mathbb{R}^{n \times k}`, we solve :math:`k` systems simultaneously, leveraging vectorized operations for efficiency.

Generic Linear Systems
~~~~~~~~~~~~~~~~~~~~~~~

For general systems :math:`Ax = b`, we provide several iterative solvers:

Conjugate Gradient (CG)
^^^^^^^^^^^^^^^^^^^^^^^

For symmetric positive definite matrices, CG generates a sequence of approximations:

.. math::

   x_{k+1} = x_k + \alpha_k p_k

.. math::

   r_{k+1} = r_k - \alpha_k A p_k

.. math::

   p_{k+1} = r_{k+1} + \beta_k p_k

where:

.. math::

   \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}, \quad \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}

**Convergence**: CG converges in at most :math:`n` steps for exact arithmetic, but practical convergence depends on the condition number :math:`\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}`.

BiConjugate Gradient Stabilized (BiCGSTAB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For non-symmetric systems, BiCGSTAB combines BiCG with stabilization:

.. math::

   x_{k+1} = x_k + \alpha_k p_k + \omega_k s_k

where :math:`s_k = r_k - \alpha_k v_k` and the parameters are chosen to minimize residual norms.

Least Squares Minimal Residual (LSMR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LSMR solves the least squares problem :math:`\min_x \|Ax - b\|_2` using a variant of the conjugate gradient method applied to the normal equations:

.. math::

   A^T A x = A^T b

The algorithm maintains numerical stability better than forming :math:`A^T A` explicitly.

Minimal Residual (MINRES)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For symmetric but indefinite matrices, MINRES minimizes the residual norm:

.. math::

   \|r_k\|_2 = \|b - A x_k\|_2

MINRES is based on the three-term recurrence relation and is particularly effective for saddle-point systems.

Sparse Multivariate Normal Distributions
-----------------------------------------

Parameterizations
~~~~~~~~~~~~~~~~~

We support multiple parameterizations of multivariate normal distributions with sparse precision/covariance matrices.

**Covariance Parameterization**

.. math::

   \mathcal{N}(\mu, \Sigma), \quad \Sigma = LL^T

where :math:`L` is the Cholesky factor.

**Precision Parameterization**

.. math::

   \mathcal{N}(\mu, \Lambda^{-1}), \quad \Lambda = Q^T Q

where :math:`\Lambda` is the precision matrix and :math:`Q` is its Cholesky factor.

**LDL Parameterization**

For numerical stability without positive definiteness constraints:

.. math::

   \Lambda = LDL^T

where :math:`L` is unit lower triangular and :math:`D` is diagonal.

Sampling and Probability Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Reparameterized Sampling**

For gradient-based optimization, we use reparameterization:

.. math::

   x = \mu + L \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

where :math:`L` satisfies :math:`LL^T = \Sigma`.

**Log Probability**

The log probability density is:

.. math::

   \log p(x) = -\frac{1}{2}(x-\mu)^T \Lambda (x-\mu) - \frac{1}{2}\log |2\pi \Lambda^{-1}|

.. math::

   = -\frac{1}{2}(x-\mu)^T \Lambda (x-\mu) + \frac{1}{2}\log |\Lambda| - \frac{d}{2}\log(2\pi)

**Efficient Computation**

For sparse precision matrices, we avoid computing :math:`\Lambda^{-1}` explicitly. Instead, we:

1. Solve :math:`\Lambda z = (x - \mu)` for :math:`z`
2. Compute the quadratic form as :math:`(x-\mu)^T z`
3. Use sparse Cholesky factorization for log-determinant computation

Numerical Considerations
------------------------

Condition Numbers and Stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The condition number :math:`\kappa(A) = \|A\| \|A^{-1}\|` determines the sensitivity of linear systems to perturbations. For sparse matrices:

- **Well-conditioned**: :math:`\kappa(A) \approx 1`
- **Ill-conditioned**: :math:`\kappa(A) \gg 1`

Our iterative solvers include tolerance parameters and maximum iteration limits to handle ill-conditioned systems.

Preconditioning
~~~~~~~~~~~~~~~

For faster convergence, we support preconditioning :math:`M^{-1}A x = M^{-1}b` where :math:`M \approx A` is easier to invert. Common choices include:

- **Diagonal (Jacobi)**: :math:`M = \text{diag}(A)`
- **Incomplete LU**: :math:`M = \tilde{L}\tilde{U} \approx A`

Gradient Computation and Automatic Differentiation
---------------------------------------------------

Implicit Function Theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~

For differentiating through linear system solutions, we use the implicit function theorem. Given :math:`Ax = b` with solution :math:`x^*`, the gradient with respect to parameters :math:`\theta` is:

.. math::

   \frac{dx^*}{d\theta} = -A^{-1} \left( \frac{dA}{d\theta} x^* - \frac{db}{d\theta} \right)

**Efficient Implementation**

Rather than computing :math:`A^{-1}` explicitly, we solve:

.. math::

   A^T \lambda = \frac{dL}{dx^*}

and then compute:

.. math::

   \frac{dL}{d\theta} = -\lambda^T \frac{dA}{d\theta} x^* + \lambda^T \frac{db}{d\theta}

This approach preserves sparsity and avoids expensive matrix inversions.

Complexity Analysis
-------------------

Time Complexity
~~~~~~~~~~~~~~~

- **Sparse matrix-vector multiplication**: :math:`O(\text{nnz})`
- **Sparse triangular solve**: :math:`O(\text{nnz})`
- **CG convergence**: :math:`O(\sqrt{\kappa} \cdot \text{nnz})` iterations
- **Sparse Cholesky factorization**: :math:`O(n^{3/2})` for typical sparse patterns

Space Complexity
~~~~~~~~~~~~~~~~~

- **Sparse storage**: :math:`O(\text{nnz})` vs :math:`O(n^2)` for dense
- **Iterative solvers**: :math:`O(n)` additional storage
- **Gradient preservation**: Same sparsity pattern as forward pass

This mathematical foundation ensures that torchsparsegradutils provides both theoretically sound and computationally efficient implementations for sparse tensor operations.
