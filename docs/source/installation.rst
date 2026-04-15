Installation
============

PyPI Installation
------------------

The easiest way to install torchsparsegradutils is from PyPI:

.. code-block:: bash

   pip install torchsparsegradutils

This will install the package with its core dependencies.

Extra Dependencies
------------------

For additional functionality, you can install optional dependencies:

.. code-block:: bash

   # Install with CuPy support (GPU acceleration, requires CUDA 12.x)
   pip install torchsparsegradutils[cupy]

   # Install with JAX support
   pip install torchsparsegradutils[jax]

   # Install all optional dependencies
   pip install torchsparsegradutils[all]

.. note::

   The CuPy extra installs ``cupy-cuda12x>=13.0``. If you are using a different
   CUDA version, install the appropriate CuPy package manually
   (e.g. ``pip install cupy-cuda11x``).

Reviewer Test Environment
-------------------------

For a clean Python 3.12 environment with all optional dependencies after the
0.2.2 release:

.. code-block:: bash

   uv venv --python 3.12 --seed --managed-python
   pip install "torchsparsegradutils[all]==0.2.2"
   python -m pytest

The default pytest suite includes CPU tests and lightweight CUDA functional
tests when CUDA is available. CUDA memory, performance, and OOM experiments are
preserved for manual validation and can be run explicitly:

.. code-block:: bash

   python -m pytest --run-manual-cuda -m manual_cuda -s

Requirements
------------

Core Requirements
~~~~~~~~~~~~~~~~~

- Python >= 3.10
- PyTorch >= 2.5

Optional Requirements
~~~~~~~~~~~~~~~~~~~~~

- JAX (for JAX backend integration): ``pip install torchsparsegradutils[jax]``
- CuPy >= 13.0 (for CuPy backend integration): ``pip install torchsparsegradutils[cupy]``

Development Installation
------------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils.git
   cd torchsparsegradutils
   pip install -e .

To install development dependencies:

.. code-block:: bash

   pip install -e .[all]
   pip install -r requirements-ci.txt

Verification
------------

To verify your installation:

.. code-block:: python

   import torchsparsegradutils as tsgu
   print(tsgu.__version__)

   # Test basic functionality
   import torch
   from torchsparsegradutils import sparse_mm

   # Create a simple sparse matrix
   indices = torch.tensor([[0, 1], [1, 0]])
   values = torch.tensor([1.0, 2.0])
   A = torch.sparse_coo_tensor(indices, values, (2, 2))
   B = torch.randn(2, 3)

   result = sparse_mm(A, B)
   print("Installation successful!")

Docker Installation
-------------------

You can also use the package in a Docker container. Here's a simple Dockerfile:

.. code-block:: dockerfile

   FROM pytorch/pytorch:latest

   RUN pip install torchsparsegradutils[all]

   # Your application code
   COPY . /app
   WORKDIR /app

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**CUDA compatibility issues**

Make sure your PyTorch installation is compatible with your CUDA version:

.. code-block:: bash

   python -c "import torch; print(torch.version.cuda)"

**CuPy installation issues**

CuPy installation can be tricky. Refer to the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_.

**JAX installation issues**

For JAX installation issues, refer to the `JAX installation guide <https://github.com/google/jax#installation>`_.

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/cai4cai/torchsparsegradutils/issues>`_
2. Create a new issue with a minimal reproducible example
3. Include your Python version, PyTorch version, and operating system
