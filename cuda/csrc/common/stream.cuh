#pragma once
// CUDAGuard + current-stream launch plumbing (architecture.md §5). Wraps the
// stable-ABI accelerator device guard (torch/csrc/stable/accelerator.h) and
// current-CUDA-stream lookup so kernel launch sites do not repeat the
// boilerplate. Mirrors the pattern used by kernel-builder's own
// relu-torch-stable-abi example (aoti_torch_get_current_cuda_stream is
// guarded behind USE_CUDA in torch's own headers and not otherwise exposed
// through a public stable-ABI header at this torch version, so it is
// forward-declared exactly as that example does). Header-only; the smoke
// launcher (cuda/csrc/kernels/_smoke/_smoke.cu) is this commit's only user.

#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void **ret_stream);

namespace tsgu {

// RAII: sets the current device to `reference`'s device for its lifetime and
// exposes that device's current CUDA stream, ready to pass to a kernel
// launch's `<<<..., stream>>>` slot.
class StreamGuard {
 public:
  explicit StreamGuard(torch::stable::Tensor const &reference) : device_guard_(reference.get_device_index()) {
    void *stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(reference.get_device_index(), &stream_ptr));
    stream_ = static_cast<cudaStream_t>(stream_ptr);
  }

  cudaStream_t stream() const { return stream_; }

 private:
  torch::stable::accelerator::DeviceGuard device_guard_;
  cudaStream_t stream_;
};

}  // namespace tsgu
