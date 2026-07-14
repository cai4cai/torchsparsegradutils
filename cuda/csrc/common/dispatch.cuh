#pragma once
// TSGU_DISPATCH_VALUE / TSGU_DISPATCH_INDEX — kernel-template dispatch macros
// (architecture.md §5, naming.md §2 short names: vals/rowptr/col dtype
// dispatch). Header-only; parsed (not yet exercised beyond the smoke path)
// by cuda/csrc/kernels/_smoke/_smoke.cu until kernel commits (Phase 3) use
// them for real.
//
// Both switch on torch::headeronly::ScalarType (the stable-ABI scalar-type
// enum — architecture.md §2's op schemas are plain dense tensors, so kernels
// read dtype off torch::stable::Tensor::scalar_type()), instantiating NAME's
// body with `scalar_t` / `index_t` bound in scope.
//
// Supported value types: float32, float64 (architecture.md §3: "kernels
// templated over {f32,f64} x {i32,i64}"). Supported index types: int32,
// int64.

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#define TSGU_DISPATCH_VALUE(SCALAR_TYPE, NAME, ...)                   \
  [&] {                                                               \
    switch (SCALAR_TYPE) {                                            \
      case torch::headeronly::ScalarType::Float: {                   \
        using scalar_t = float;                                      \
        return __VA_ARGS__();                                        \
      }                                                               \
      case torch::headeronly::ScalarType::Double: {                  \
        using scalar_t = double;                                     \
        return __VA_ARGS__();                                        \
      }                                                               \
      default:                                                       \
        STD_TORCH_CHECK(false, NAME " does not support this dtype"); \
    }                                                                 \
  }()

#define TSGU_DISPATCH_INDEX(SCALAR_TYPE, NAME, ...)                         \
  [&] {                                                                    \
    switch (SCALAR_TYPE) {                                                 \
      case torch::headeronly::ScalarType::Int: {                         \
        using index_t = int32_t;                                         \
        return __VA_ARGS__();                                            \
      }                                                                   \
      case torch::headeronly::ScalarType::Long: {                       \
        using index_t = int64_t;                                        \
        return __VA_ARGS__();                                           \
      }                                                                   \
      default:                                                           \
        STD_TORCH_CHECK(false, NAME " does not support this index dtype"); \
    }                                                                      \
  }()
