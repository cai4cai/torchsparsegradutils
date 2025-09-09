# torchsparsegradutils: Sparsity-preserving gradient utility tools for PyTorch

[![Python tests](https://github.com/cai4cai/torchsparsegradutils/actions/workflows/python-package.yml/badge.svg)](https://github.com/cai4cai/torchsparsegradutils/actions/workflows/python-package.yml) [![License](https://img.shields.io/github/license/cai4cai/torchsparsegradutils)](https://github.com/cai4cai/torchsparsegradutils?tab=Apache-2.0-1-ov-file#readme) [![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive collection of utility functions to work with PyTorch sparse tensors, ensuring memory efficiency and supporting various sparsity-preserving tensor operations with automatic differentiation. This package addresses fundamental gaps in PyTorch's sparse tensor ecosystem, providing essential operations that preserve sparsity in gradients during backpropagation.

## 🚀 Key Features

### Core Sparse Operations with Sparse Gradient Support

**Memory-Efficient Sparse Matrix Multiplication**
- `sparse_mm`: Memory-efficient sparse matrix multiplication with batch support
- Preserves sparsity in gradients during backpropagation
- Workaround for [PyTorch issue #41128](https://github.com/pytorch/pytorch/issues/41128)
- Supports both COO and CSR formats with optional batching

**Sparse Linear System Solvers**
- `sparse_triangular_solve`: Sparse triangular solver with batch support
  -  Discussion reference: [PyTorch issue #87358](https://github.com/pytorch/pytorch/issues/87358)
- `sparse_generic_solve`: Generic sparse linear solver with pluggable backends
  - Tested and benchmarked with CG, BICGSTAB, LSMR and MINRES solvers

- `sparse_solve_c4t`: Wrappers around [cupy sparse solvers](https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html#solving-linear-problems)
  -  Discussion reference: [Pytorch issue #69538](https://github.com/pytorch/pytorch/issues/69538)
  - Tested and benchmarked with: [CG](https://docs.cupy.dev/en/v9.6.0/reference/generated/cupyx.scipy.sparse.linalg.cg.html), [CGS](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.cgs.html#cupyx.scipy.sparse.linalg.cgs), [MINRES](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.minres.html#cupyx.scipy.sparse.linalg.minres), [GMRES](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.gmres.html#cupyx.scipy.sparse.linalg.gmres), [spsolve](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spsolve.html#cupyx.scipy.sparse.linalg.spsolve) and [spsolve_triangular](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spsolve_triangular.html#cupyx.scipy.sparse.linalg.spsolve_triangular) CuPy solvers
- `tsgujax.sparse_solve_j4t`: Wrappers around [jax sparse solvers](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.sparse.linalg)
  - Tested with: CG and BICGSTAB JAX solvers
- `sparse_generic_lstsq`: Generic sparse linear least-squares solver

### Built-in Iterative Solvers (No External Dependencies)

**Pure PyTorch Implementations**
- **BICGSTAB**: Biconjugate Gradient Stabilized method (ported from [pykrylov](https://github.com/PythonOptimizers/pykrylov))
- **CG**: Conjugate Gradient method (ported from [cornellius-gp/linear_operator](https://github.com/cornellius-gp/linear_operator))
- **LSMR**: Least Squares Minimal Residual method (ported from [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize))
- **MINRES**: Minimal Residual method (ported from [cornellius-gp/linear_operator](https://github.com/cornellius-gp/linear_operator))

### Sparse Multivariate Normal Distributions

- **SparseMultivariateNormal**: Structured Gaussian Distribution
  - Implements reparameterised sampling (rsample)
  - Supports leading batch dimension
  - Supports COO and CSR sparse tensors
  - Covariance or precision matrices with LL^T or LDL^T parameterisations.
  - LDL^T parameterization offers numerical stability without SPD constraints
- **SparseMultivariateNormalNative**:
  - Implements reparameterised sampling (rsample)
  - Uses native `torch.sparse.mm` only
  - Only supports ubatched CSR tensors
  - Covariance LL^T parameterization

### Spatial Encoding Tools

**Pairwise Encoder**
- Encode local neighborhood relationships in nD spatial volumes
- Multi-channel/class support
- Configurable neighborhood radius and sparsity patterns
- Outputs sparse unbatched/batched COO or CSR matrices for downstream processing
- Optimised for medical imaging and volumetric data applications

### Graph Neural Network Operations

**Indexed Matrix Multiplication**
- `segment_mm`: Segmented matrix multiplication compatible with DGL/PyG
- `gather_mm`: Gather-based matrix multiplication for graph operations
- Pure PyTorch implementations as alternatives to [`dgl.ops.segment_mm`](https://docs.dgl.ai/generated/dgl.ops.segment_mm.html), [`pyg_lib.ops.segment_matmul`](https://pyg-lib.readthedocs.io/en/latest/modules/ops.html#pyg_lib.ops.segment_matmul), and [`dgl.ops.gather_mm`](https://docs.dgl.ai/generated/dgl.ops.gather_mm.html)
- Supports PyTorch >= 2.4 with nested tensor operations



## 🛠️ Installation

### Basic Installation

The package can be installed using pip:

```bash
pip install torchsparsegradutils
```

### Development Installation

For the latest features and development work:

```bash
pip install git+https://github.com/cai4cai/torchsparsegradutils
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For CuPy sparse solver support (GPU acceleration)
pip install cupy-cuda12x  # Replace with your CUDA version

# For JAX sparse solver support
pip install "jax[cpu]"     # CPU version
pip install "jax[cuda12]"  # GPU version (replace with your CUDA version)

# For benchmarking and testing
pip install scipy matplotlib pandas tqdm pytest
```

### Requirements

- **Python**: ≥ 3.10
- **PyTorch**: ≥ 2.5 (≥ 2.4 for indexed operations)
- **Operating Systems**: Linux, macOS, Windows
- **Hardware**: CPU and CUDA GPU support


## 📊 Performance Benchmarks

Our comprehensive benchmark suite demonstrates significant performance improvements across various sparse operations. All benchmarks were conducted on an NVIDIA GeForce RTX 4090 with PyTorch 2.8.0+cu128. Benchmarks are performed using [Rothberg/cfd2](https://suitesparse-collection-website.herokuapp.com/Rothberg/cfd2) matrix from [SuiteSparse Matrix Collection](https://suitesparse-collection-website.herokuapp.com/)

![Sparse MM Suite Performance (int32/float32 COO)](torchsparsegradutils/benchmarks/benchmark_visualizations/sparse_mm_suite_performance_int32_float32_coo.png)

![Sparse Triangular Solve Suite Performance (int32/float32 COO)](torchsparsegradutils/benchmarks/benchmark_visualizations/triangular_solve_suitesparse_performance_int32_float32_coo.png)

![Sparse Genertic Solve Suite Performance (int32/float32 COO)](torchsparsegradutils/benchmarks/benchmark_visualizations/sparse_solve_suite_performance_int32_float32_coo.png)

## 🚀 Quick Start

### Basic Sparse Matrix Multiplication

```python
import torch
from torchsparsegradutils import sparse_mm

# Create sparse matrix in COO format
indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
values = torch.tensor([3., 4., 5.], requires_grad=True)
A = torch.sparse_coo_tensor(indices, values, (2, 3))

# Dense matrix
B = torch.randn(3, 4, requires_grad=True)

# Memory-efficient sparse matrix multiplication with gradient support
C = sparse_mm(A, B)
loss = C.sum()
loss.backward()  # Gradients preserved in sparse format

print(f"A.grad: {A.grad}")  # Sparse gradient
print(f"B.grad: {B.grad}")  # Dense gradient
```

### Sparse Linear System Solving

```python
import torch
from torchsparsegradutils import sparse_triangular_solve, sparse_generic_solve
from torchsparsegradutils.utils import linear_cg

# Create sparse triangular matrix
A = create_sparse_triangular_matrix()  # Your sparse CSR matrix
b = torch.randn(A.shape[0], requires_grad=True)

# Triangular solve (fast for triangular systems)
x1 = sparse_triangular_solve(A, b, upper=False)

# Generic solve with different backends
x2 = sparse_generic_solve(A, b, solve=linear_cg, tol=1e-6)

# Using CuPy backend (if available)
from torchsparsegradutils.cupy import sparse_solve_c4t
x3 = sparse_solve_c4t(A, b, solve="cg", tol=1e-6)
```

### Sparse Multivariate Normal Distribution

```python
import torch
from torchsparsegradutils.distributions import SparseMultivariateNormal
from torchsparsegradutils.utils.random_sparse import rand_sparse_tri

# Create parameters
batch_size, event_size = 2, 1000
loc = torch.zeros(batch_size, event_size)

# Example 1: LDL^T parameterization (numerically stable for precision matrices)
# Create sparse lower triangular matrix (unit triangular, no diagonal)
scale_tril = rand_sparse_tri(
    (batch_size, event_size, event_size),
    nnz=5000,  # 5000 non-zeros for 1M parameters (0.5% sparsity)
    layout=torch.sparse_csr,
    upper=False,
    unit_triangular=True  # Unit triangular for LDL^T
)

# Diagonal component for LDL^T parameterization
diagonal = torch.ones(batch_size, event_size) * 0.5

# Create distribution with LDL^T parameterization
dist_ldlt = SparseMultivariateNormal(
    loc=loc,
    diagonal=diagonal,
    scale_tril=scale_tril  # Unit lower triangular
)

# Example 2: LL^T parameterization (standard Cholesky)
scale_tril_chol = rand_sparse_tri(
    (batch_size, event_size, event_size),
    nnz=5000,
    layout=torch.sparse_csr,
    upper=False,
    unit_triangular=False  # Include diagonal for LL^T
)

# Create distribution with LL^T parameterization
dist_chol = SparseMultivariateNormal(
    loc=loc,
    scale_tril=scale_tril_chol  # Lower triangular with diagonal
)

# Example 3: Precision matrix parameterization (more stable with LDL^T)
precision_tril = rand_sparse_tri(
    (batch_size, event_size, event_size),
    nnz=5000,
    layout=torch.sparse_csr,
    upper=False,
    unit_triangular=True
)

precision_diagonal = torch.ones(batch_size, event_size) * 2.0

dist_precision = SparseMultivariateNormal(
    loc=loc,
    diagonal=precision_diagonal,
    precision_tril=precision_tril  # Unit triangular precision factor
)

# Sample with gradient support
samples = dist_ldlt.rsample((100,))  # 100 samples

# Gradient computation preserves sparsity
loss = samples.sum()
loss.backward()
print(f"Sparse gradient shape: {scale_tril.grad.shape}")
print(f"Sparse gradient nnz: {scale_tril.grad._nnz()}")
print(f"Using LDL^T parameterization: {dist_ldlt.is_ldlt_parameterization}")
```

### Pairwise Voxel Encoding

```python
import torch
from torchsparsegradutils.encoders import PairwiseEncoder

# Create 3D volume encoder (channels, height, depth, width)
volume_shape = (4, 64, 64, 64)  # 4 channels, 64x64x64 spatial
encoder = PairwiseEncoder(
    radius=2.0,
    volume_shape=volume_shape,
    layout=torch.sparse_csr
)

# Generate values for each spatial relationship offset
num_offsets = len(encoder.offsets)
values = torch.randn(num_offsets, *volume_shape)

# Generate sparse encoding matrix
sparse_matrix = encoder(values)

print(f"Encoded volume shape: {sparse_matrix.shape}")
print(f"Sparsity: {sparse_matrix._nnz() / sparse_matrix.numel():.3%}")
print(f"Number of spatial offsets: {num_offsets}")

# Use in sparse multivariate normal
flat_size = 4 * 64 * 64 * 64  # Total flattened size
dist = SparseMultivariateNormal(
    loc=torch.zeros(flat_size),
    scale_tril=sparse_matrix
)
```

#### Spatial Relationship Visualization

The encoder creates sparse matrices that encode pairwise spatial relationships within a specified radius. Different channel relationship types affect how channels interact:

- **`indep`**: Independent channels (only spatial neighbors within same channel)
- **`intra`**: Intra-channel relationships (spatial neighbors within same channel)
- **`inter`**: Inter-channel relationships (spatial neighbors across all channels)

**3D Spatial Grid (3×3×3×3) with Different Channel Relations:**

<div align="center">

**Radius = 1.0**
![Spatial Encodings Radius 1](torchsparsegradutils/tests/test_outputs/sparse_encodings_radius_1.png)

**Radius = 2.0**
![Spatial Encodings Radius 2](torchsparsegradutils/tests/test_outputs/sparse_encodings_radius_2.png)
<!--
**Legend for Spatial Offsets:**
<table>
<tr>
<td><img src="torchsparsegradutils/tests/test_outputs/legend_radius_1.png" width="150"/></td>
<td><img src="torchsparsegradutils/tests/test_outputs/legend_radius_2.png" width="150"/></td>
</tr>
<tr>
<td align="center">Radius 1.0 Offsets</td>
<td align="center">Radius 2.0 Offsets</td>
</tr>
</table> -->

</div>

Each color represents a different spatial offset (relative position) in the 3D neighborhood. The sparse matrix encodes these relationships efficiently, enabling:

- **Local spatial modeling** for volumetric data (medical imaging, 3D computer vision)
- **Multi-channel feature interaction** in convolutional architectures
- **Sparse graph construction** from regular grids
- **Memory-efficient neighborhood encoding** for large volumes

**Key Parameters:**
- `radius`: Spatial neighborhood radius (1.0 = immediate neighbors, 2.0 = extended neighborhood)
- `volume_shape`: `(channels, height, depth, width)` for 4D volumes
- `channel_voxel_relation`: Controls cross-channel connectivity patterns
- `layout`: Output sparse format (`torch.sparse_coo` or `torch.sparse_csr`)

### Indexed Matrix Operations (Graph Neural Networks)

```python
import torch
from torchsparsegradutils import segment_mm, gather_mm

# Segment matrix multiplication (compatible with DGL/PyG)
a = torch.randn(15, 10, requires_grad=True)  # Node features
b = torch.randn(3, 10, 5, requires_grad=True)  # Edge type embeddings
seglen_a = torch.tensor([5, 6, 4])  # Segment lengths

# Performs: a[0:5] @ b[0], a[5:11] @ b[1], a[11:15] @ b[2]
result = segment_mm(a, b, seglen_a)

# Gather matrix multiplication
indices = torch.tensor([0, 0, 1, 1, 2])
a_gathered = torch.randn(5, 10, requires_grad=True)
result = gather_mm(a_gathered, b, indices)
```

### Statistical Distribution Validation

```python
import torch
from torch.distributions import MultivariateNormal
from torchsparsegradutils.utils import mean_hotelling_t2_test, cov_nagao_test

# Generate sample data from known distribution
torch.manual_seed(42)
true_mean = torch.tensor([[0.0, 0.0]])
true_cov = torch.eye(2).unsqueeze(0)
n = 1000

# Generate samples and compute statistics
dist = MultivariateNormal(true_mean.squeeze(0), true_cov.squeeze(0))
samples = dist.sample((n,)).unsqueeze(1)
sample_mean = samples.mean(0)
sample_cov = torch.cov(samples.squeeze(1).T).unsqueeze(0)

# Test if sample mean is consistent with hypothesized mean (should pass)
result, t2_stat, threshold = mean_hotelling_t2_test(
    sample_mean, true_mean, sample_cov, n, confidence_level=0.95
)
print(f"Mean test passed: {result.item()}")  # True

# Test if sample covariance is consistent with hypothesized covariance (should pass)
result, t_n_stat, threshold = cov_nagao_test(
    sample_cov, true_cov, n, confidence_level=0.95
)
print(f"Covariance test passed: {result.item()}")  # True

# Test against wrong parameters (should fail)
wrong_mean = true_mean + 1.0  # Significantly different mean
result, _, _ = mean_hotelling_t2_test(
    sample_mean, wrong_mean, sample_cov, n, confidence_level=0.95
)
print(f"Wrong mean test passed: {result.item()}")  # False
```

## 🧪 Testing and Benchmarks

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test modules
python -m pytest torchsparsegradutils/tests/test_sparse_matmul.py
python -m pytest torchsparsegradutils/tests/test_distributions.py

# Run with coverage
python -m pytest --cov=torchsparsegradutils
```

### Running Benchmarks

The package includes comprehensive benchmarks for performance evaluation:

```bash
# Sparse matrix multiplication benchmarks
python -m torchsparsegradutils.benchmarks.sparse_mm_rand
python -m torchsparsegradutils.benchmarks.batched_sparse_mm_rand

# Triangular solver benchmarks
python -m torchsparsegradutils.benchmarks.sparse_triangular_solve_rand

# Generic solver benchmarks
python -m torchsparsegradutils.benchmarks.sparse_generic_solve_suite

# SuiteSparse matrix benchmarks
python -m torchsparsegradutils.benchmarks.sparse_mm_suite
```

Results are automatically saved to `torchsparsegradutils/benchmarks/results/` as CSV files.

### Utility Functions

#### `torchsparsegradutils.utils.random_sparse`

**Sparse Random Matrix Generators**
- **`rand_sparse(size, nnz, layout=torch.sparse_coo, **kwargs)`**: Generate random sparse matrices with specified layout and properties
  - Supports COO and CSR
  - Supports batch dimension
- **`rand_sparse_tri(size, nnz, layout=torch.sparse_coo, upper=True, strict=False, **kwargs)`**: Generate random sparse triangular matrices
  - Supports COO and CSR
  - Supports batch dimension
  - Strict triangular (no diagonal) or non-strict (with diagonal values)
  - Option to produce well conditioned matrices and regulate diagonal values

- **`make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio=0.5, nz=None)`**: Generate sparse symmetric positive definite (SPD) matrices

#### `torchsparsegradutils.utils.utils`

**Sparse Matrix Operations**
- **`sparse_block_diag(*sparse_tensors)`**: Create block diagonal sparse matrix from multiple sparse tensors
- **`sparse_block_diag_split(sparse_block_diag_tensor, *shapes)`**: Split block diagonal sparse matrix into original sparse tensors
- **`sparse_eye(size, layout=torch.sparse_coo, **kwargs)`**: Create batched or unbatched sparse identity matrices
- **`stack_csr(tensors, dim=0)`**: Stack CSR tensors along batch dimension (like torch.stack for CSR)

**Sparse Format Conversion**
- **`convert_coo_to_csr_indices_values(coo_indices, num_rows, values=None)`**: Convert COO indices and values to CSR format, with support for batch dimension
- **`convert_coo_to_csr(sparse_coo_tensor)`**: Convert COO sparse tensor to CSR format with batch support

#### `torchsparsegradutils.utils.dist_stats_helpers`

**Statistical Distribution Validation**
- **`mean_hotelling_t2_test(sample_mean, true_mean, sample_cov, n, confidence_level=0.95)`**: One-sample Hotelling T² test for multivariate mean equality using confidence regions
  - Tests whether hypothesized mean vector lies within confidence region around sample mean
  - Uses F-distribution for threshold calculation with proper degrees of freedom
  - Higher confidence levels create larger (more permissive) acceptance regions
- **`cov_nagao_test(emp_cov, ref_cov, n, confidence_level=0.95)`**: Nagao's test for covariance matrix equality using confidence regions
  - Tests whether hypothesized covariance matrix is consistent with empirical covariance
  - Uses χ² distribution with appropriate degrees of freedom
  - Standardizes covariance matrices for improved numerical stability


## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/cai4cai/torchsparsegradutils/issues)
2. **Pull Requests**: Submit improvements via GitHub PRs
3. **Testing**: Ensure all tests pass and add tests for new functionality
4. **Documentation**: Update docstrings and examples for new features
5. **Benchmarks**: Include performance benchmarks for new operations

### Development Setup

#### Option 1: Local Development

```bash
git clone https://github.com/cai4cai/torchsparsegradutils
cd torchsparsegradutils
pip install -e ".[dev]"  # Install in development mode
pre-commit install       # Install pre-commit hooks
```

#### Option 2: Development Containers (Recommended)

For a consistent development environment with GPU support and all dependencies pre-installed, use VS Code Dev Containers:

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) with NVIDIA Container Toolkit (for GPU support)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**Quick Start:**
1. Clone the repository and open in VS Code:
   ```bash
   git clone https://github.com/cai4cai/torchsparsegradutils
   cd torchsparsegradutils
   code .
   ```

2. When prompted, click **"Reopen in Container"** or use the Command Palette:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "Dev Containers: Reopen in Container"

**Available Configurations:**

- **`.devcontainer/Dockerfile.stable`** (default): Uses stable PyTorch with CUDA 12.8 support
- **`.devcontainer/Dockerfile.nightly`**: Uses nightly PyTorch builds for latest features

To switch configurations, modify the `dockerfile` field in `.devcontainer/devcontainer.json`:
```json
"build": {
    "dockerfile": "./Dockerfile.nightly",  // or "./Dockerfile.stable"
    "context": "."
}
```

**What's Included:**
- **CUDA 12.8**: Full GPU development support with NVIDIA drivers
- **Pre-installed Dependencies**: PyTorch, CuPy, JAX, SciPy, and all development tools
- **VS Code Extensions**: Python, Pylance, Jupyter, GitHub Copilot, and code formatting tools
- **Development Tools**: pytest, black, flake8, pre-commit hooks
- **Python Environment**: Python 3.10+ with all optional dependencies

**Benefits:**
- ✅ **Consistent Environment**: Same setup across different machines
- ✅ **GPU Support**: Pre-configured CUDA environment
- ✅ **Zero Setup**: All dependencies and tools pre-installed
- ✅ **Isolated**: No conflicts with host system packages
- ✅ **VS Code Integration**: Seamless debugging, IntelliSense, and testing

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the foundational sparse tensor implementations
- **SciPy/CuPy Teams**: For high-performance sparse linear algebra routines
- **JAX Team**: For cross-platform sparse operations and XLA compilation
- **Open Source Libraries**: We port and adapt algorithms from:
  - [pykrylov](https://github.com/PythonOptimizers/pykrylov) (BICGSTAB)
  - [cornellius-gp/linear_operator](https://github.com/cornellius-gp/linear_operator) (CG, MINRES)
  - [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) (LSMR)

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{torchsparsegradutils,
  title={torchsparsegradutils: Sparsity-preserving gradient utility tools for PyTorch},
  author={Barfoot, Theodore and Glocker, Ben and Vercauteren, Tom},
  url={https://github.com/cai4cai/torchsparsegradutils},
  year={2024}
}
```

## ⚠️ Known Issues

### PyTorch Sparse COO Index Dtype Conversion

**Issue**: PyTorch automatically converts `int32` indices to `int64` when creating sparse COO tensors, but preserves `int32` for sparse CSR tensors. This affects memory usage and performance for algorithms that benefit from `int32` indices (such as `sparse_mm`).

**Impact**:
- **Memory**: `int64` indices use 2× more memory than `int32`
- **Performance**: Some sparse operations may run faster with `int32` indices
- **Cross-format consistency**: Different behavior between COO and CSR formats

**Example**:
```python
import torch

# Demonstrate the issue
indices_int32 = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
values = torch.tensor([1.0, 2.0])

print(f"Original indices dtype: {indices_int32.dtype}")  # torch.int32

# COO: int32 -> int64 conversion happens
coo_tensor = torch.sparse_coo_tensor(indices_int32, values, (2, 2)).coalesce()
print(f"COO indices dtype: {coo_tensor.indices().dtype}")  # torch.int64 (converted!)

# CSR: int32 is preserved
crow_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
col_indices = torch.tensor([1, 0], dtype=torch.int32)
csr_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, (2, 2))
print(f"CSR crow_indices dtype: {csr_tensor.crow_indices().dtype}")  # torch.int32 (preserved!)
print(f"CSR col_indices dtype: {csr_tensor.col_indices().dtype}")    # torch.int32 (preserved!)
```

**Workarounds**:
1. **Use CSR format** when `int32` indices are important for performance
2. **Account for extra memory** when using COO format with large sparse matrices
3. **Test performance** with both dtypes to determine if the conversion impacts your use case

**Status**: This is a known PyTorch behavior. Our test suite documents and validates this behavior to catch any future changes in PyTorch's handling of sparse tensor index dtypes.

### PairwiseEncoder CSR Memory Usage Issue

**Issue**: CSR sparse tensors generated by `PairwiseEncoder` consume significantly more memory during backward passes compared to COO format, particularly in integration tests with `SparseMultivariateNormal`.

**Impact**:
- **Memory Consumption**: CSR integration tests can use 2-3x more memory than equivalent COO tests during `.backward()`
- **Training Stability**: May cause out-of-memory errors during training with large spatial volumes
- **Development**: Affects integration testing with large tensor configurations

**Suspected Cause**: The issue may be related to CSR permutation operations within `PairwiseEncoder` that create additional intermediate tensors during gradient computation.

**Current Status**: Under investigation. The memory spike occurs specifically during backpropagation through the sparse matrix operations.

**Workarounds**:
1. **Use COO format** for `PairwiseEncoder` when memory is constrained during training
2. **Reduce batch sizes** or spatial dimensions when using CSR format
3. **Monitor memory usage** carefully when integrating `PairwiseEncoder` with gradient-based optimization

**Example**:
```python
# More memory-efficient approach for large tensors
encoder = PairwiseEncoder(
    radius=2.0,
    volume_shape=(4, 64, 64, 64),
    layout=torch.sparse_coo  # Use COO instead of CSR for memory efficiency
)
```

### SparseMultivariateNormal LL^T Precision Parameterization Gradient Issues

**Issue**: Large gradient magnitudes can occur when using LL^T parameterization with precision matrices in `SparseMultivariateNormal`, leading to training instability.

**Impact**:
- **Gradient Explosion**: Gradients can become extremely large (>1e6) during backpropagation
- **Training Instability**: May cause NaN values or divergent optimization
- **Numerical Issues**: Poor conditioning of the precision matrix can amplify gradient problems

**Affected Configurations**:
- LL^T parameterization (`scale_tril` parameter) combined with precision matrix formulation
- Both 2D and 3D spatial configurations show this behavior
- More pronounced with larger spatial dimensions and higher sparsity

**Root Cause**: The LL^T precision parameterization can lead to poor numerical conditioning, especially when the triangular matrix has small diagonal values or high condition number.

**Recommended Solution**: Use LDL^T parameterization instead, which provides better numerical stability:

```python
# Problematic: LL^T precision parameterization
dist_unstable = SparseMultivariateNormal(
    loc=loc,
    precision_tril=scale_tril  # LL^T with precision - can cause large gradients
)

# Better: LDL^T parameterization with separate diagonal
dist_stable = SparseMultivariateNormal(
    loc=loc,
    diagonal=diagonal,  # Separate diagonal component for stability
    precision_tril=unit_triangular_matrix  # Unit triangular (LDL^T)
)
```

**Benefits of LDL^T Parameterization**:
- **Numerical Stability**: Separates diagonal scaling from triangular structure
- **Gradient Stability**: More stable gradients during backpropagation
- **No SPD Constraints**: Doesn't require strict positive definiteness
- **Better Conditioning**: Diagonal component can be controlled independently

**Status**: This is a known limitation of the LL^T precision formulation. LDL^T parameterization is the recommended approach for precision matrices.