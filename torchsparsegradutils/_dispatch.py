"""Backend dispatch: the ``tsgu`` op namespace, CUDA backend probe, version
handshake, disable escape hatch. Architecture: architecture.md §1-2.

Every kernel-backed op is a ``torch.library`` custom op defined under the
``tsgu`` namespace (``torch.library.custom_op`` + ``register_fake`` +
``register_autograd`` — architecture.md §2). ``ops/`` modules (and
``utils/convert.py`` for ``tsgu::coo2csr``) define the nine ops from
map.md's "Kernel routing" table directly via that decorator, which implicitly
registers each op under this namespace; nothing needs to import this module
to *define* an op.

This module owns the cross-cutting concerns that don't belong to any single
op:

- the shared ``tsgu`` library namespace (``_library`` below — a place for
  any future direct registrations; op modules normally never need it,
  since ``torch.library.custom_op("tsgu::name", ...)`` defines the op
  itself),
- the CUDA backend probe + ``__backend_api_version__`` handshake
  (architecture.md §1: "-cuda exposes ``__backend_api_version__``; the
  front package probes at import (``_dispatch.py``), refuses on
  mismatch"),
- the ``TSGU_DISABLE_CUDA_BACKEND=1`` escape hatch,
- ``backend_available()`` for callers/tests that want to know whether real
  CUDA implementations are registered for the ``tsgu::`` ops (as opposed to
  only schema + fake).

No CUDA/CPU implementation is registered here or anywhere else in this
commit (spec/commit.md Phase 1 #9) — ``torchsparsegradutils_cuda`` registers
the CUDA dispatch key from C++ starting in Phase 2 (commit 10). Until then,
``backend_available()`` is always ``False`` (no such package to probe), and
calling any ``tsgu::`` op raises ``NotImplementedError`` from the op's own
Python body (see each op's definition) rather than through this module.
"""

from __future__ import annotations

import importlib
import os
import warnings

import torch

# The namespace every tsgu:: op is registered under. Op modules use the
# "tsgu::<name>" string directly with torch.library.custom_op; this constant
# exists so nothing hardcodes the literal string a second time.
TSGU_NAMESPACE = "tsgu"

# Bumped whenever the Python-front-package <-> torchsparsegradutils_cuda
# contract changes incompatibly (op set, schemas, semantics). The CUDA
# package exposes the same constant's value as its own
# __backend_api_version__; a mismatch means the two packages were built
# against different contracts and must not be wired together.
BACKEND_API_VERSION = 1

# A fragment of the `tsgu` library other code could register into directly
# (torch.library.Library) if a future commit ever needs to define something
# under the namespace outside the custom_op decorator path. Op modules do
# not use this today — torch.library.custom_op("tsgu::...", ...) already
# defines the op under the same namespace on first use.
_library = torch.library.Library(TSGU_NAMESPACE, "FRAGMENT")

_DISABLE_ENV_VAR = "TSGU_DISABLE_CUDA_BACKEND"


def _disabled_by_env() -> bool:
    """TSGU_DISABLE_CUDA_BACKEND=1 (or true/yes/on) escape hatch
    (architecture.md §1): forces backend_available() to False even if a
    compatible torchsparsegradutils_cuda is installed."""
    return os.getenv(_DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _probe_backend() -> bool:
    """Import ``torchsparsegradutils_cuda`` (if installed) and validate the
    ``__backend_api_version__`` handshake.

    Returns ``True`` iff a compatible CUDA backend is loaded and (per its own
    ``__init__``) has registered CUDA-key implementations for the ``tsgu::``
    ops. Never raises: an absent or incompatible backend degrades to
    "unavailable" (with a warning on version *mismatch* specifically, since
    that's the one case a plain ``ImportError`` wouldn't explain) — the front
    package must stay importable without the backend (architecture.md §4 /
    spec/commit.md commit 10's verify step).
    """
    if _disabled_by_env():
        return False
    try:
        # importlib (not a static `import torchsparsegradutils_cuda`
        # statement): the package doesn't exist until Phase 2 (commit 10),
        # and a static import of a genuinely-optional, not-yet-built sibling
        # package isn't something a type checker should flag as an error on
        # every commit until then.
        _backend = importlib.import_module("torchsparsegradutils_cuda")
    except ImportError:
        return False

    backend_version = getattr(_backend, "__backend_api_version__", None)
    if backend_version != BACKEND_API_VERSION:
        warnings.warn(
            "torchsparsegradutils_cuda is installed but its "
            f"__backend_api_version__ ({backend_version!r}) does not match "
            f"the front package's expected BACKEND_API_VERSION "
            f"({BACKEND_API_VERSION!r}). The CUDA backend will not be used; "
            "tsgu:: ops that need a CUDA implementation will raise at "
            "dispatch time. Reinstall matching torchsparsegradutils / "
            "torchsparsegradutils-cuda versions.",
            stacklevel=2,
        )
        return False
    return True


_backend_available = _probe_backend()


def backend_available() -> bool:
    """Whether a compatible ``torchsparsegradutils_cuda`` backend is loaded
    and registered for the CUDA dispatch key of the ``tsgu::`` ops."""
    return _backend_available


# ---------------------------------------------------------------------------
# tsgu::_smoke — CUDA bring-up smoke op (spec/commit.md Phase 2 #10).
#
# Not one of map.md's nine routed ops: this is the trivial `x + 1` op that
# proves the whole toolchain (kernel-builder scaffold -> compiled extension
# -> CUDA dispatch-key registration -> torch.ops.tsgu dispatch) end to end,
# before any real kernel commit (Phase 3) exists. Defined here rather than in
# ops/ (which is reserved for the nine public-API-routed ops) — private
# (leading underscore), never exported, never wired to a public wrapper.
#
# Schema + fake kernel only, as for every other op in this commit: the CUDA
# implementation is registered from C++ by torchsparsegradutils_cuda
# (cuda/csrc/registration.cpp, STABLE_TORCH_LIBRARY_IMPL(tsgu, CUDA, m)) —
# nothing here ever executes on CPU.
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::_smoke", mutates_args=())
def _smoke(x: torch.Tensor) -> torch.Tensor:
    """Bring-up smoke test: ``x + 1``, computed by an actual CUDA kernel
    launch in the ``torchsparsegradutils_cuda`` backend (never on CPU) —
    proves compile + load + dispatch end-to-end (spec/commit.md commit 10).
    """
    raise NotImplementedError(
        "tsgu::_smoke has no CPU implementation — it exists only to smoke-test the "
        "torchsparsegradutils_cuda backend (spec/commit.md Phase 2 #10). Install a "
        "compatible torchsparsegradutils_cuda and call this on a CUDA tensor."
    )


@_smoke.register_fake
def _smoke_fake(x: torch.Tensor) -> torch.Tensor:
    # Value-independent: same shape/dtype/device as the input.
    return torch.empty_like(x)
