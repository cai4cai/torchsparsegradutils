Installation
============

PyPI Installation
------------------

The easiest way to install torchsparsegradutils is from PyPI:

.. code-block:: bash

   pip install torchsparsegradutils

This will install the package with its core dependencies.

Requirements
------------

Core Requirements
~~~~~~~~~~~~~~~~~

- Python >= 3.10
- PyTorch >= 2.5

Development Installation
------------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils.git
   cd torchsparsegradutils
   pip install -e .

To install development dependencies (via `uv <https://docs.astral.sh/uv/>`_):

.. code-block:: bash

   uv sync --group dev

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

   RUN pip install torchsparsegradutils

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

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/cai4cai/torchsparsegradutils/issues>`_
2. Create a new issue with a minimal reproducible example
3. Include your Python version, PyTorch version, and operating system
