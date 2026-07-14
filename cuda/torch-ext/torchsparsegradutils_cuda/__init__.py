"""torchsparsegradutils_cuda — the CUDA backend package (architecture.md §1).

Importing this module loads the compiled CUDA extension; as a side effect of
that shared-library load, the extension's static initializers
(``STABLE_TORCH_LIBRARY_IMPL`` in ``csrc/registration.cpp``) register CUDA
dispatch-key implementations for the ``tsgu::`` ops it ships, into the same
``tsgu`` ``torch.library`` namespace the front package
(``torchsparsegradutils``) defines schemas for (architecture.md §2). No
explicit call from Python is needed to reach that registration.

Phase 2 (spec/commit.md commit 10) ships exactly one op end-to-end for
bring-up: ``tsgu::_smoke``. The nine real ops (map.md routing table) get no
CUDA implementation here — only their own kernel commits (Phase 3) register
them.
"""

from ._ops import ops  # noqa: F401  (import triggers extension load + registration)

# Bumped in lockstep with torchsparsegradutils._dispatch.BACKEND_API_VERSION
# (architecture.md §1: "-cuda exposes __backend_api_version__; the front
# package probes at import (_dispatch.py), refuses on mismatch").
__backend_api_version__ = 1

__all__ = ["__backend_api_version__"]
