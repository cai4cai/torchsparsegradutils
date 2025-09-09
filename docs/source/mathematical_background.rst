Mathematical Background
=======================

This section provides the mathematical foundations underlying the algorithms and methods implemented in torchsparsegradutils.

Notation and Conventions
------------------------

- Bold uppercase/lowercase symbols denote matrices/vectors (e.g., :math:`\mathbf{A}, \mathbf{B}, \mathbf{x}`).
- Gradients use :math:`\partial` notation, e.g., :math:`\partial L/\partial \mathbf{A}`.
- For batched inputs, leading batch dimensions are omitted in the math for clarity; operations are broadcast across batches.
- For sparse inputs, gradients are defined and returned only at the indices where the input has nonzeros (same sparsity pattern), unless explicitly noted.

Sparse Matrix Representations
------------------------------

Coordinate Format (COO)
~~~~~~~~~~~~~~~~~~~~~~~

The coordinate format stores sparse matrices using three arrays:

- **indices**: A 2×nnz tensor containing row and column indices
- **values**: A 1D tensor of length nnz containing the non-zero values
- **size**: The shape of the full matrix

For a sparse matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` with nnz non-zero elements:

.. math::

   \mathbf{A} = \sum_{k=1}^{\text{nnz}} v_k \cdot e_{i_k} e_{j_k}^T

where :math:`v_k` are the values, :math:`(i_k, j_k)` are the indices, and :math:`e_i` are standard basis vectors.

Compressed Sparse Row (CSR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSR format uses three arrays:

- **indices**: Column indices of non-zero elements
- **indptr**: Row pointers indicating start of each row
- **values**: Non-zero values

The CSR format allows efficient row-wise operations and is preferred for matrix-vector multiplication.

Autograd for Sparse Inputs (Sparsity-Preserving Gradients)
----------------------------------------------------------

When an operation takes a sparse input (e.g., :math:`\mathbf{A}`) and a dense input (e.g., :math:`\mathbf{B}`), we compute
gradients that respect the sparsity of the sparse operand:

- :math:`\partial L/\partial \mathbf{A}` is returned with the same sparsity pattern as :math:`\mathbf{A}` (nonzeros only).
- :math:`\partial L/\partial \mathbf{B}` is dense as usual.

Practically, this means we evaluate the dense gradient formula for :math:`\partial L/\partial \mathbf{A}` and then sample it
only at the nonzero coordinates of :math:`\mathbf{A}`.

Sparse Matrix Operations
-------------------------

Sparse Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For sparse matrix :math:`\mathbf{A}` and dense matrix :math:`\mathbf{B}`, the multiplication :math:`\mathbf{C} = \mathbf{A}\mathbf{B}` is computed as:

.. math::

   C_{ij} = \sum_{k} A_{ik} B_{kj}

The key insight in our implementation is preserving sparsity in gradients. When computing :math:`\frac{\partial L}{\partial \mathbf{A}}` for some loss :math:`L`, we ensure the gradient maintains the same sparsity pattern as :math:`\mathbf{A}`.

**Gradient Preservation**

For the backward pass, we compute:

.. math::

   \frac{\partial L}{\partial \mathbf{A}} = \frac{\partial L}{\partial \mathbf{C}} \, \mathbf{B}^\top

The gradient :math:`\frac{\partial L}{\partial \mathbf{A}}` has the same sparsity pattern as :math:`\mathbf{A}`, which is crucial for memory efficiency in sparse learning scenarios.

We make the above more explicit:

- Let :math:`\mathbf{C} = \mathbf{A}\mathbf{B}` and :math:`\mathbf{G} = \partial L/\partial \mathbf{C}`.
- Dense gradients are

   .. math::

       \frac{\partial L}{\partial \mathbf{B}} \;=\; \mathbf{A}^\top \mathbf{G},
       \qquad
       \left(\frac{\partial L}{\partial \mathbf{A}}\right)_{\text{dense}} \;=\; \mathbf{G}\,\mathbf{B}^\top.

- Sparse-aware gradient: evaluate :math:`(\partial L/\partial \mathbf{A})_{\text{dense}}` and extract the values only at
   nonzero locations of :math:`\mathbf{A}`:

   .. math::

       \left(\frac{\partial L}{\partial \mathbf{A}}\right)_{ij}
       \;=\; \sum_{k} G_{ik} B_{jk}
       \quad \text{for } (i,j) \in \operatorname{supp}(\mathbf{A}).

Linear System Solvers
----------------------

Triangular Systems
~~~~~~~~~~~~~~~~~~

For lower triangular systems :math:`\mathbf{L}\mathbf{x} = \mathbf{b}`, we use forward substitution:

.. math::

   x_i = \frac{1}{L_{ii}} \left( b_i - \sum_{j=1}^{i-1} L_{ij} x_j \right)

For upper triangular systems :math:`\mathbf{U}\mathbf{x} = \mathbf{b}`, we use backward substitution:

.. math::

   x_i = \frac{1}{U_{ii}} \left( b_i - \sum_{j=i+1}^{n} U_{ij} x_j \right)

**Batch Processing**

For batched systems :math:`\mathbf{L}\mathbf{X} = \mathbf{B}` where :math:`\mathbf{X}, \mathbf{B} \in \mathbb{R}^{n \times k}`, we solve :math:`k` systems simultaneously, leveraging vectorized operations for efficiency.

**Gradients**

Let :math:`\mathbf{L}\mathbf{x} = \mathbf{b}` with :math:`\mathbf{L}` lower triangular (upper-triangular is analogous).
Given upstream gradient :math:`\mathbf{G} = \partial L/\partial \mathbf{x}`, the gradients are

.. math::

   \frac{\partial L}{\partial \mathbf{b}} \;=\; \mathbf{L}^{-\top} \mathbf{G},
   \qquad
   \frac{\partial L}{\partial \mathbf{L}} \;=\; -\,\left(\mathbf{L}^{-\top} \mathbf{G}\right)\,\mathbf{x}^\top,

evaluated only at nonzeros of :math:`\mathbf{L}` in the sparse-aware case.

Generic Linear Systems
~~~~~~~~~~~~~~~~~~~~~~~

For general systems :math:`\mathbf{A}\mathbf{x} = \mathbf{b}`, we provide several iterative solvers:

Conjugate Gradient (CG)
^^^^^^^^^^^^^^^^^^^^^^^

For symmetric positive definite matrices, CG generates a sequence of approximations:

.. math::

   \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \, \mathbf{p}_k

.. math::

   \mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \, \mathbf{A} \, \mathbf{p}_k

.. math::

   \mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \, \mathbf{p}_k

where:

.. math::

   \alpha_k = \frac{\mathbf{r}_k^\top \, \mathbf{r}_k}{\mathbf{p}_k^\top \, \mathbf{A} \, \mathbf{p}_k}, \quad \beta_k = \frac{\mathbf{r}_{k+1}^\top \, \mathbf{r}_{k+1}}{\mathbf{r}_k^\top \, \mathbf{r}_k}

**Convergence**: CG converges in at most :math:`n` steps for exact arithmetic, but practical convergence depends on the condition number :math:`\kappa(\mathbf{A}) = \frac{\lambda_{\max}}{\lambda_{\min}}`.

BiConjugate Gradient Stabilized (BiCGSTAB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For non-symmetric systems, BiCGSTAB combines BiCG with stabilization:

.. math::

   \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \, \mathbf{p}_k + \omega_k \, \mathbf{s}_k

where :math:`\mathbf{s}_k = \mathbf{r}_k - \alpha_k \, \mathbf{v}_k` and the parameters are chosen to minimize residual norms.

Least Squares Minimal Residual (LSMR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LSMR solves the least squares problem :math:`\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2` using a variant of the conjugate gradient method applied to the normal equations:

.. math::

   \mathbf{A}^\top \mathbf{A} \, \mathbf{x} = \mathbf{A}^\top \mathbf{b}

The algorithm maintains numerical stability better than forming :math:`A^T A` explicitly.

Minimal Residual (MINRES)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For symmetric but indefinite matrices, MINRES minimizes the residual norm:

.. math::

   \|\mathbf{r}_k\|_2 = \|\mathbf{b} - \mathbf{A} \, \mathbf{x}_k\|_2

MINRES is based on the three-term recurrence relation and is particularly effective for saddle-point systems.

Gradients for Generic Solves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider :math:`\mathbf{A}\,\mathbf{x} = \mathbf{B}` with solution :math:`\mathbf{x} = \mathbf{A}^{-1}\mathbf{B}` (or using a suitable
solver if :math:`\mathbf{A}` is not directly inverted). With :math:`\mathbf{G} = \partial L/\partial \mathbf{x}`:

.. math::

   \frac{\partial L}{\partial \mathbf{B}} \;=\; \mathbf{A}^{-\top} \mathbf{G},
   \qquad
   \left(\frac{\partial L}{\partial \mathbf{A}}\right)_{\text{dense}} \;=\; -\,\left(\mathbf{A}^{-\top} \mathbf{G}\right)\,\mathbf{x}^\top.

For sparse-aware gradients, we evaluate the dense expression and take entries only at the nonzeros of :math:`\mathbf{A}`:

.. math::

   \left(\frac{\partial L}{\partial \mathbf{A}}\right)_{ij}
   \;=\; -\,\big(\mathbf{A}^{-\top} \mathbf{G}\big)_{i,:}\,\mathbf{x}_{j,:}^{\top}
   \quad \text{for } (i,j) \in \operatorname{supp}(\mathbf{A}).

These formulas follow from differentiating :math:`\mathbf{A}\mathbf{x}=\mathbf{B}` and applying the implicit function theorem.

Sparse Multivariate Normal Distributions
-----------------------------------------

Parameterizations
~~~~~~~~~~~~~~~~~

We support multiple parameterizations of multivariate normal distributions with sparse precision/covariance matrices.

**Covariance Parameterization**

.. math::

   \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \boldsymbol{\Sigma} = \mathbf{L}\,\mathbf{L}^T

where :math:`L` is the Cholesky factor.

**Precision Parameterization**

.. math::

   \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1}), \quad \boldsymbol{\Lambda} = \mathbf{Q}^T \, \mathbf{Q}

where :math:`\Lambda` is the precision matrix and :math:`Q` is its Cholesky factor.

**LDL Parameterization**

For numerical stability without positive definiteness constraints:

.. math::

   \boldsymbol{\Lambda} = \mathbf{L} \, \mathbf{D} \, \mathbf{L}^T

where :math:`L` is unit lower triangular and :math:`D` is diagonal.

Sampling and Probability Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Reparameterized Sampling**

For gradient-based optimization, we use reparameterization:

.. math::

   \mathbf{x} = \boldsymbol{\mu} + \mathbf{L} \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})

where :math:`\mathbf{L}` satisfies :math:`\mathbf{L}\,\mathbf{L}^T = \boldsymbol{\Sigma}`.

**Log Probability**

The log probability density is:

.. math::

   \log p(\mathbf{x}) = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \, \boldsymbol{\Lambda} \, (\mathbf{x}-\boldsymbol{\mu}) - \frac{1}{2}\log |2\pi \, \boldsymbol{\Lambda}^{-1}|

.. math::

   = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \, \boldsymbol{\Lambda} \, (\mathbf{x}-\boldsymbol{\mu}) + \frac{1}{2}\log |\boldsymbol{\Lambda}| - \frac{p}{2}\log(2\pi)

**Efficient Computation**

For sparse precision matrices, we avoid computing :math:`\boldsymbol{\Lambda}^{-1}` explicitly. Instead, we:

1. Solve :math:`\boldsymbol{\Lambda} \, \mathbf{z} = (\mathbf{x} - \boldsymbol{\mu})` for :math:`\mathbf{z}`
2. Compute the quadratic form as :math:`(\mathbf{x}-\boldsymbol{\mu})^T \, \mathbf{z}`
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

For differentiating through linear system solutions, we use the implicit function theorem. Given :math:`\mathbf{A}\mathbf{x} = \mathbf{b}` with solution :math:`\mathbf{x}^*`, the gradient with respect to parameters :math:`\theta` is:

.. math::

   \frac{d\mathbf{x}^*}{d\theta} = -\mathbf{A}^{-1} \left( \frac{d\mathbf{A}}{d\theta} \, \mathbf{x}^* - \frac{d\mathbf{b}}{d\theta} \right)

**Efficient Implementation**

Rather than computing :math:`A^{-1}` explicitly, we solve:

.. math::

   \mathbf{A}^T \, \boldsymbol{\lambda} = \frac{dL}{d\mathbf{x}^*}

and then compute:

.. math::

   \frac{dL}{d\theta} = -\boldsymbol{\lambda}^T \, \frac{d\mathbf{A}}{d\theta} \, \mathbf{x}^* + \boldsymbol{\lambda}^T \, \frac{d\mathbf{b}}{d\theta}

This approach preserves sparsity and avoids expensive matrix inversions.

Least Squares Problems
----------------------

Given :math:`\min_{\mathbf{x}} \lVert \mathbf{A}\mathbf{x} - \mathbf{B} \rVert_2^2` with (possibly rectangular) :math:`\mathbf{A}`,
the normal equations are :math:`\mathbf{A}^\top \mathbf{A}\,\mathbf{x} = \mathbf{A}^\top \mathbf{B}`. In practice we avoid forming
:math:`\mathbf{A}^\top \mathbf{A}` explicitly and use stable iterative solvers.

Let :math:`\mathbf{G} = \partial L/\partial \mathbf{x}`. Then

.. math::

   	ext{Solve}\; (\mathbf{A}^\top\mathbf{A})\,\mathbf{Y} = \mathbf{A}^\top \, \mathbf{G} \;\Rightarrow\; \frac{\partial L}{\partial \mathbf{B}} = \mathbf{Y},

which coincides with :math:`\mathbf{A}^+ \mathbf{G}` using the (left) pseudoinverse when :math:`\mathbf{A}` has full column rank.

For the gradient with respect to :math:`\mathbf{A}`, a convenient dense form is

.. math::

   \left(\frac{\partial L}{\partial \mathbf{A}}\right)_{\text{dense}}
   \;=\; -\,\big(\mathbf{R}\,\mathbf{x}^\top\big)\;+
   \;\big(\mathbf{G}\,\mathbf{B}^\top\big)\,\big(\mathbf{A}^\top\mathbf{A}\big)^{-1}\mathbf{A}^\top,\quad \mathbf{R} = \mathbf{A}\mathbf{x} - \mathbf{B},

with sparse-aware evaluation performed at :math:`\operatorname{supp}(\mathbf{A})`. Implementations typically compute the two
terms via solves, not by forming :math:`(\mathbf{A}^\top\mathbf{A})^{-1}`.

Indexed and Segmented Matrix Multiplication
-------------------------------------------

Some operations (e.g., ``gather_mm``, ``segment_mm``) perform matrix multiplication on indexed rows/segments.
Let :math:`\mathcal{I}` denote an index set mapping output rows to inputs. Forward maps follow standard matmul restricted to
indices, and gradients accumulate over repeated indices:

.. math::

   \frac{\partial L}{\partial \mathbf{B}} \;=\; \sum_{i\in\mathcal{I}} \mathbf{A}_{i,:}^\top\,\mathbf{G}_{i,:},
   \qquad
   \left(\frac{\partial L}{\partial \mathbf{A}}\right)_{i,:} \;=\; \mathbf{G}_{i,:}\,\mathbf{B}^\top.

When :math:`\mathbf{A}` is sparse, the same sparsity-preserving rule applies to :math:`\partial L/\partial \mathbf{A}`.

Iterative Solvers Overview
--------------------------

We provide stable iterative methods suitable for large sparse systems:

- CG: symmetric positive definite systems; minimizes :math:`\lVert \mathbf{r}_k\rVert_2` over Krylov subspaces; sensitive to conditioning.
- MINRES: symmetric (possibly indefinite); minimizes residual in :math:`\ell_2`; robust for saddle-point problems.
- BiCGSTAB: nonsymmetric systems; combines BiCG with stabilization; ~2 matvecs per iteration; can break down on exceptional inner products.
- LSMR: least-squares via Golub–Kahan; solves :math:`\min\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert_2`; avoids forming :math:`\mathbf{A}^\top\mathbf{A}`.

Preconditioning (diagonal/Jacobi, incomplete Cholesky/LU, problem-specific) typically reduces iteration counts.

Statistical Validation via Confidence Regions
---------------------------------------------

For validating batched multivariate Gaussian models, we include two tests:

Hotelling's T-squared for means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given sample mean :math:`\bar{\mathbf{x}}`, hypothesized mean :math:`\boldsymbol{\mu}_0`, sample covariance :math:`\hat{\boldsymbol{\Sigma}}`,
and sample size :math:`n`, the test statistic

.. math::

   T^2 \;=\; n\,(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)^\top\,\hat{\boldsymbol{\Sigma}}^{-1}\,(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)

has the relationship :math:`T^2 \sim \tfrac{p(n-1)}{n-p}\,F_{p,\,n-p}` under unknown covariance. At a chosen confidence level,
accept :math:`\boldsymbol{\mu}_0` if :math:`T^2 \le \tfrac{p(n-1)}{n-p}\,F_{p,\,n-p;\,\text{confidence level}}`.

Nagao's test for covariances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With hypothesized covariance :math:`\boldsymbol{\Sigma}_0`, sample covariance :math:`\hat{\boldsymbol{\Sigma}}`, and
whitened matrix :math:`\mathbf{W} = \boldsymbol{\Sigma}_0^{-1/2} \hat{\boldsymbol{\Sigma}} \, \boldsymbol{\Sigma}_0^{-1/2}`,

.. math::

   T_N \;=\; \tfrac{n}{2}\,\lVert \mathbf{W}-\mathbf{I} \rVert_F^2\;\sim\;\chi^2_{\,\nu},\qquad \nu = \tfrac{p(p+1)}{2}.

Accept :math:`\boldsymbol{\Sigma}_0` if :math:`T_N \le \chi^2_{\nu;\,\text{confidence level}}`.

Backends and Implementations
----------------------------

All formulas above are backend-agnostic. We provide bindings for PyTorch (primary), with optional JAX and CuPy integrations
that implement the same mathematics while leveraging their respective array libraries and sparse backends.

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
