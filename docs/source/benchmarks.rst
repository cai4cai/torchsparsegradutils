Benchmarks
==========

The CUDA rewrite replaced the pre-rewrite benchmark suite wholesale. The full
record — timing protocol, corpus, acceptance bars, result tables, memory
companion data, and charts — lives in the repository at ``spec/benchmarks.md``,
with every measurement persisted as a provenance-stamped JSON file under
``benchmarks/results/``. This page summarises the headline results.

Measurement setup
-----------------

All numbers below were measured 2026-07-15 to 2026-07-17 on the migration
development machine: **NVIDIA RTX A1000 Laptop GPU, PyTorch 2.13.0+cu130,
CUDA 13.0**. Clocks were *not* locked (the harness runs without root); the
per-run headers record the queried clocks instead, so cross-run comparisons
inherit some clock jitter. This is a modest laptop GPU — treat absolute times
accordingly; every speedup is like-for-like on the same machine. The corpus is
the synthetic batched/ragged tier (plus a banded/SPD sweep for the solves);
real-matrix corpora (SuiteSparse, DLMC) are post-migration work.

The protocol (specified in ``spec/benchmarks.md``): CUDA-event timing with
warmup and windowed measurement reporting medians, an L2-cache flush between
iterations, geometric-mean aggregation, TF32 off for parity-relevant numbers,
peak-memory measurement alongside every timing, and a ``custom`` vs
``vendor-scaffold`` provenance label on every row. All rows below are
``custom`` — no vendor-scaffold rows enter any claim.

Headline results
----------------

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Claim
     - Result
   * - vs. like-for-like vendor calls (cuSPARSE SDDMM unbatched; cuSPARSE
       SpSM cold and warm)
     - geo-mean **1.52×** faster
   * - batched / fused / legacy-composite paths (batched SpMM, SDDMM, and
       SpSM; fused bidirectional logsumexp; grouped GEMM vs a per-segment
       cuBLAS loop; coo2csr; logsumexp vs ``pytorch_scatter`` and the old
       pure-PyTorch path; e2e rsample CSR vs COO)
     - geo-mean **8.11×** faster
   * - end-to-end ``SparseMultivariateNormal.rsample``, encoder-CSR vs
       encoder-COO (2×64³ volume)
     - **4.1×** faster, **0.56×** the backward peak memory
   * - end-to-end CG solve vs dense ``torch.linalg.solve`` (n = 4096, CSR)
     - **~16×** faster
   * - backward peak memory vs the dense-gradient counterfactual
       (encoder-CSR rsample, 2×64³: 524k² system)
     - 402 MB measured vs ~1.1 TB dense (**~2700×** saved)

Selected per-op rows (medians; baselines named per row):

.. list-table::
   :header-rows: 1
   :widths: 22 30 34 14

   * - Op
     - Variant
     - Baseline
     - Speedup
   * - ``sparse_logsumexp``
     - CSR, 4096², 1.68M nse
     - ``pytorch_scatter.scatter_logsumexp``
     - 98×
   * - ``sparse_bidir_logsumexp``
     - CSR, 4096², 1.68M nse
     - two single-dim ``tsgu::seglse`` calls
     - 10.4×
   * - ``tsgu::sddmm`` (the shared sparse-gradient kernel)
     - CSR, unbatched, p=128
     - cuSPARSE SDDMM
     - 2.41×
   * - ``sparse_mm``
     - CSR, B=8 ragged
     - block-diag + cuSPARSE
     - 4.1×
   * - ``sparse_triangular_solve``
     - CSR, B=8
     - looped cuSPARSE SpSM per item
     - 10.3×
   * - ``gather_mm``
     - 4096×64, 4 groups
     - per-segment cuBLAS GEMM loop
     - 2.6×
   * - ``convert_coo_to_csr``
     - COO, B=16, 5M nse
     - pure-torch argsort + compress
     - 1.9×

The open bar
------------

One acceptance bar is honestly still open: ``grouped_gemm`` on *segment
shapes* reaches **0.5–0.8×** of cuBLAS ``cublasGemmGroupedBatched`` after the
register-blocked performance pass. The fused gather path and both backward
passes beat their baselines; the tracked follow-up is vectorised 128-bit
shared-memory loads plus double buffering. Every other acceptance bar in
``spec/benchmarks.md`` is met.

Memory behaviour
----------------

Memory is measured alongside time in every run, because the library's core
claim is a memory claim: gradients w.r.t. a sparse input stay at the input's
sparsity pattern instead of materialising a dense ``n × m`` gradient.

The historically documented encoder-CSR backward memory blow-up (see older
release notes) is fixed and pinned as a regression case: the CSR path's
backward peak is now roughly **half** the COO path's (0.555–0.570× across
2×32³, 2×48³, and 2×64³ volumes), against a pinned bar of ≤ 1.2×.

Reproducing benchmarks
----------------------

The op-level harness lives in ``benchmarks/`` at the repository root (one
``bench_*.py`` per op; JSON results with machine fingerprint and provenance
written to ``benchmarks/results/``), and per-kernel NVBench microbenchmarks
live in ``cuda/bench/``. A CUDA GPU and the ``torchsparsegradutils_cuda``
backend are required. See ``spec/benchmarks.md`` for the protocol every
number must follow.
