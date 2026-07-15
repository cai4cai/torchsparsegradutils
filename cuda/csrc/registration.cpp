// tsgu CUDA backend registration (architecture.md §5: cuda/csrc/registration.cpp).
//
// STABLE_TORCH_LIBRARY_IMPL registers a CUDA dispatch-key implementation for
// an existing tsgu:: op *schema* — schemas live in the front package
// (torchsparsegradutils, via torch.library.custom_op — architecture.md §2).
// This translation unit never calls torch::library::def()/m.def(): every
// tsgu:: op's schema (including _smoke's) is defined in Python.
//
// Phase 2 (commit 10) registers exactly one op for end-to-end bring-up:
// tsgu::_smoke. The nine real ops (map.md routing table) get no
// implementation here — that starts in Phase 3 (commits 12-19). This file is
// append-only across kernel commits (spec/commit.md Phase 3 template, T2).
//
// Importing torchsparsegradutils_cuda (torch-ext/torchsparsegradutils_cuda/
// __init__.py) loads this file's compiled shared library; the static
// initializers STABLE_TORCH_LIBRARY_IMPL generates (below) run at load time,
// so no explicit call from Python is needed to reach this registration.

#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

// Declared, not defined, here: this is the only translation unit that needs
// the launcher's signature. Defined in csrc/kernels/_smoke/_smoke.cu.
torch::stable::Tensor tsgu_smoke_launch(torch::stable::Tensor const &x);

// Commit 12 (spec/commit.md Phase 3): tsgu::seglse + tsgu::seglse_bwd —
// Family 2 "Segmented logsumexp" (kernels.md). Defined in
// csrc/kernels/logsumexp/seglse.cu.
torch::stable::Tensor tsgu_seglse_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                          int64_t B, int64_t n, int64_t m, bool include_zeros);
torch::stable::Tensor tsgu_seglse_bwd_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                              torch::stable::Tensor const &lse, torch::stable::Tensor const &gout,
                                              int64_t B, int64_t n);

STABLE_TORCH_LIBRARY_IMPL(tsgu, CUDA, m) {
  m.impl("_smoke", TORCH_BOX(&tsgu_smoke_launch));
  m.impl("seglse", TORCH_BOX(&tsgu_seglse_launch));
  m.impl("seglse_bwd", TORCH_BOX(&tsgu_seglse_bwd_launch));
}

// --- Python extension module entry point ------------------------------------
// kernel-builder's own REGISTER_EXTENSION helper (its generated
// `registration.h` template) always lands at a fixed torch-ext/ path — see
// this commit's reported layout delta (build.toml) — which this file, living
// under csrc/ per architecture.md §5, cannot reach via a plain #include. The
// two-line boilerplate it wraps — a PyInit_<TORCH_EXTENSION_NAME> so Python
// can import the compiled .so as a module — is inlined directly instead.
#define TSGU_CONCAT_(a, b) a##b
#define TSGU_CONCAT(a, b) TSGU_CONCAT_(a, b)
#define TSGU_STRINGIFY_(a) #a
#define TSGU_STRINGIFY(a) TSGU_STRINGIFY_(a)

PyMODINIT_FUNC TSGU_CONCAT(PyInit_, TORCH_EXTENSION_NAME)() {
  static struct PyModuleDef module = {
      PyModuleDef_HEAD_INIT, TSGU_STRINGIFY(TORCH_EXTENSION_NAME), nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
