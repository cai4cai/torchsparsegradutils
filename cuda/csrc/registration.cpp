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

// Commit 13 (spec/commit.md Phase 3): tsgu::seglse_bidir + tsgu::seglse_bidir_bwd
// — fused row+column Family 2 variant (kernels.md). Defined in
// csrc/kernels/logsumexp/seglse_bidir.cu.
torch::stable::Tensor tsgu_seglse_bidir_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                                torch::stable::Tensor const &col, int64_t B, int64_t n, int64_t m,
                                                bool include_zeros);
torch::stable::Tensor tsgu_seglse_bidir_bwd_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                                    torch::stable::Tensor const &col, torch::stable::Tensor const &padded,
                                                    torch::stable::Tensor const &gout, int64_t B, int64_t n, int64_t m);

// Commit 14 (spec/commit.md Phase 3): tsgu::sddmm — Family 1 "SDDMM (the
// shared backward)" (kernels.md). Defined in csrc/kernels/sddmm/sddmm.cu.
// No wrapper routes to it yet (nothing in ops/ is switched over this
// commit) -- registered here so it is callable directly (torch.ops.tsgu.sddmm)
// for its own gates and as the future backward primitive commits 15-17 wire in.
torch::stable::Tensor tsgu_sddmm_launch(torch::stable::Tensor const &rowptr, torch::stable::Tensor const &col,
                                         torch::stable::Tensor const &g, torch::stable::Tensor const &mat, int64_t B,
                                         int64_t n, int64_t m, bool negate);

// Commit 15 (spec/commit.md Phase 3): tsgu::spmm — Family 3 "Vendor-baseline
// forwards" SpMM row (kernels.md). Defined in csrc/kernels/spmm/spmm.cu.
// Serves sparse_mm's forward, its gradB (called on the cached CSC transpose),
// and spmv (p = 1, no separate op).
torch::stable::Tensor tsgu_spmm_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                        torch::stable::Tensor const &col, torch::stable::Tensor const &dense,
                                        int64_t B, int64_t n, int64_t m);

// Commit 16 (spec/commit.md Phase 3): tsgu::spsm — Family 3 "Vendor-baseline
// forwards" Triangular SpSM row (kernels.md), architecture.md §3's SpSM
// analysis-plan cache. Defined in csrc/kernels/spsm/spsm.cu (plan cache:
// csrc/kernels/spsm/plan.cpp). Serves sparse_triangular_solve's forward and
// (via its own `transpose` flag, no separate op) its gradB.
torch::stable::Tensor tsgu_spsm_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                        torch::stable::Tensor const &col, torch::stable::Tensor const &rhs,
                                        int64_t B, int64_t n, bool upper, bool unitriangular, bool transpose);
// Test/introspection-only (spsm.cu's own comment): plan-cache (builds, hits)
// counters, spec/commit.md Phase 3 commit 16 T5.
torch::stable::Tensor tsgu_spsm_plan_cache_stats_launch(torch::stable::Tensor const &anchor);

STABLE_TORCH_LIBRARY_IMPL(tsgu, CUDA, m) {
  m.impl("_smoke", TORCH_BOX(&tsgu_smoke_launch));
  m.impl("seglse", TORCH_BOX(&tsgu_seglse_launch));
  m.impl("seglse_bwd", TORCH_BOX(&tsgu_seglse_bwd_launch));
  m.impl("seglse_bidir", TORCH_BOX(&tsgu_seglse_bidir_launch));
  m.impl("seglse_bidir_bwd", TORCH_BOX(&tsgu_seglse_bidir_bwd_launch));
  m.impl("sddmm", TORCH_BOX(&tsgu_sddmm_launch));
  m.impl("spmm", TORCH_BOX(&tsgu_spmm_launch));
  m.impl("spsm", TORCH_BOX(&tsgu_spsm_launch));
  m.impl("_spsm_plan_cache_stats", TORCH_BOX(&tsgu_spsm_plan_cache_stats_launch));
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
