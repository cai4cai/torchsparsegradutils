Experimental sparse shape checking
==================================

Pyrefly can check both sparse layouts and tensor dimensions at supported public API boundaries. This feature is opt-in. The default check continues to use the layout-only annotations and the package's Python 3.10 compatibility target.

The runtime layout types are ``NewType`` brands. They have no generic shape parameter, so wrapping them directly in a jaxtyping annotation loses the shape. The optional stub overlay describes each brand as a generic tensor type instead. It gives Pyrefly somewhere to store the shape while keeping the original tensor objects, validators, and autograd implementations at runtime.

Setup in a checkout
-------------------

Use Python 3.12 or later for the shape-checking tools. Install the regular development dependencies first, then install the shape-only dependencies into a separate directory. This keeps Pyrefly's partial PyTorch stubs out of the normal typing environment:

.. code-block:: bash

   python -m pip install --no-deps --target .cache/pyrefly-shapes -r requirements-shapes.txt
   python -m pyrefly check -c pyrefly-shapes.toml
   python -m pyrefly check -c pyrefly-shapes.toml --expectations typing_tests/shapes/negative.py
   PYTHONPATH=. python typing_tests/shapes/runtime.py

The checker and shape stubs are pinned to version 1.3.1. With this version, resolving ``shape_extensions`` enables shape checking, including jaxtyping; ``jaxtyping = true`` is not a recognized configuration key. The shape check uses a Python 3.13 *static* target because upstream stubs use newer typing syntax. This does not change the Python version needed to run torchsparsegradutils.

What is checked
---------------

* Tuple sizes flow through the public random sparse factories.
* Layout validators and guards preserve shapes already known to the checker.
* ``sparse_mm`` checks exactly two-dimensional or three-dimensional operands, matching batch sizes and inner dimensions, and infers the result shape.
* ``sparse_triangular_solve`` checks square matrices and matching matrix right hand sides, including its supported three-dimensional batching.
* ``sparse_generic_solve`` checks square, unbatched systems with vector or matrix right hand sides.
* ``sparse_generic_lstsq`` checks unbatched row compatibility and infers the solution shape for vector and matrix right hand sides.
* ``clone``, ``detach``, and the covered sparse layout conversions preserve dimensions. Conversions update the layout brand; ``to_dense`` removes it. PyTorch's runtime restrictions on which conversions are supported still apply.

For example, the following inference requires no shape annotation at the call:

.. code-block:: python

   import torch
   from torchsparsegradutils import sparse_mm
   from torchsparsegradutils.utils.random_sparse import generate_random_sparse_csr_matrix

   a = generate_random_sparse_csr_matrix((3, 4), 5)
   b = torch.randn(4, 2)
   result = sparse_mm(a, b)  # shape (3, 2)
   sparse_mm(a, torch.randn(5, 2))  # static error: inner dimensions differ

Using the installed overlay
---------------------------

Wheels include the overlay as data under ``torchsparsegradutils/_shape_stubs``. To check consumer code, add that directory and the isolated shape dependencies directory to Pyrefly's ``search-path``, and use ``python-version = "3.13"`` with Pyrefly 1.3.1. The installed overlay path can be found without importing the shape extensions:

.. code-block:: bash

   python -c 'from pathlib import Path; import torchsparsegradutils; print(Path(torchsparsegradutils.__file__).parent / "_shape_stubs")'

Use absolute paths when the consumer project is elsewhere. Keep the overlay enabled for a separate check rather than adding the shape stub packages to the ordinary environment. CI checks both a source checkout and an installed package.

Limits and maintenance
----------------------

These are static contracts for supported interfaces, not a proof of every internal tensor operation. Runtime checks remain necessary for unknown shapes, layout validation, dtypes, devices, sparse index invariants, triangularity, rank conditions, and data-dependent dimensions. Sparse layouts are checked for the sparse operands; a plain ``Tensor`` annotation does not certify a dense layout. List sizes remain gradual, and unmodeled PyTorch operations can lose shape information. In particular, converting a plain PyTorch tensor to sparse layout is not covered by the upstream stubs; prefer a typed factory or a known-shape function boundary followed by a validator.

Use ``transpose`` followed by a layout validator to retain inferred dimensions. The overlay conservatively types ``.T`` as an unbranded tensor because upstream's ``Self`` annotation incorrectly preserves the original shape and sparse layout. Other tensor methods and sparse utilities are outside the initial coverage.

Within stubs, apply ``Shaped`` to each layout separately before forming a union. Pyrefly 1.3.1 does not preserve shapes through ``Shaped[COO | CSR, ...]``. Likewise, guards must be generic in the incoming shape. Do not add broad fallback overloads to consumers: they can accept calls with known incompatible dimensions.

The generic classes exist only in stubs. Do not instantiate shape parameters at runtime, subclass the runtime brands, or use them in ``isinstance`` checks. The overlay does not provide jaxtyping runtime checking or runtime-evaluable generic annotations. Use postponed annotations and the provided layout validators.

The positive contracts use ``assert_type`` to detect information loss. The negative contracts use Pyrefly's ``--expectations`` mode, so missing diagnostics fail the check. Runtime counterparts verify tensor identity, outputs, sparse gradients, and public stub parameter compatibility. When upgrading Pyrefly, upgrade its shape stubs together and rerun all three checks.

Background: `Pyrefly tensor shapes <https://pyrefly.org/en/docs/tensor-shapes/>`_, `Pyrefly jaxtyping compatibility <https://pyrefly.org/en/docs/tensor-shapes-reference/#jaxtyping-compatibility>`_, and `the generic NewType limitation <https://github.com/microsoft/pyright/issues/6284>`_.
