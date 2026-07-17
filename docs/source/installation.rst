Installation
============

torchsparsegradutils ships as two packages: the pure-Python front package
(``torchsparsegradutils`` — public API, op schemas, dispatch, host-side
composites) and the CUDA backend (``torchsparsegradutils_cuda`` — the
compiled ``tsgu::`` kernels). The front package installs anywhere; **the
backend, an NVIDIA GPU, and CUDA are required to actually run the ops**.

Front package
-------------

Install from PyPI:

.. code-block:: bash

   pip install torchsparsegradutils

This installs a pure-Python wheel with its core dependencies — no compiler
involved.

CUDA backend (required at runtime)
----------------------------------

The core ops are CUDA-only by design: no CPU implementation ships, and
calling an op without a backend (or with CPU tensors) raises a clear error.
That error is intentional, not an installation failure.

The backend is built with `Hugging Face kernel-builder
<https://huggingface.co/docs/kernels>`_ from the ``cuda/`` directory of the
repository and distributed through the Hugging Face Hub (kernel-builder's Nix
build produces the binaries; its ``build-and-upload`` flow publishes them as
a Hub kernel repository). Prebuilt backend wheels are a post-migration item
and do not exist yet — until then, either pull the published kernel from the
Hub or build ``cuda/`` locally with kernel-builder.

At import, the front package probes for ``torchsparsegradutils_cuda`` and
validates a version handshake (``__backend_api_version__``); mismatched
packages refuse to wire together rather than failing inside a kernel.

Import-only workflows
~~~~~~~~~~~~~~~~~~~~~

For workflows that only need to *import* the package — docs builds, CPU-only
CI that collects tests, introspection — set:

.. code-block:: bash

   export TSGU_DISABLE_CUDA_BACKEND=1

This skips the backend probe. ``import torchsparsegradutils`` then works
without a GPU; calling any kernel-backed op still raises.

Requirements
------------

- Python >= 3.10
- PyTorch >= 2.5
- NVIDIA GPU + CUDA for anything beyond importing the package

Development Installation
------------------------

The repository is managed with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/cai4cai/torchsparsegradutils.git
   cd torchsparsegradutils
   uv sync --group dev

Verification
------------

To verify the front package and backend are wired together:

.. code-block:: python

   import torch
   import torchsparsegradutils
   from torchsparsegradutils._dispatch import backend_available

   print(backend_available())  # True iff a compatible CUDA backend is loaded

   # Exercise an op end-to-end (requires the backend and a CUDA device)
   from torchsparsegradutils import sparse_mm

   indices = torch.tensor([[0, 1], [1, 0]])
   values = torch.tensor([1.0, 2.0])
   A = torch.sparse_coo_tensor(indices, values, (2, 2)).cuda()
   B = torch.randn(2, 3, device="cuda")

   result = sparse_mm(A, B)
   print("Installation successful!")

Troubleshooting
---------------

**"CUDA backend not available" errors**

The front package imported but the backend probe failed. Check that
``torchsparsegradutils_cuda`` is installed, that its
``__backend_api_version__`` matches the front package's expectation (a
mismatch warning is emitted at import), and that
``TSGU_DISABLE_CUDA_BACKEND`` is not set in your environment.

**CUDA compatibility issues**

Make sure your PyTorch installation is compatible with your CUDA version:

.. code-block:: bash

   python -c "import torch; print(torch.version.cuda)"

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/cai4cai/torchsparsegradutils/issues>`_
2. Create a new issue with a minimal reproducible example
3. Include your Python version, PyTorch version, CUDA version, and GPU model
