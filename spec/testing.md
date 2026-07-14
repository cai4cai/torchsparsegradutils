# Testing — parity and legitimacy

Two pillars. **Parity**: our output matches a trusted reference. **Legitimacy**:
the op is *correct in itself* — right autograd formula, right metadata, right
behaviour on hostile input — independent of any reference. Parity without
legitimacy is "bug-compatible"; legitimacy without parity can't catch a wrong
constant. Both, always.

Shares fixtures with [benchmarks.md](benchmarks.md) (corpus loaders, variant
matrix, synthetic generators, INDEX_DTYPES) — no timing infra here. Correctness
gates run **before** any benchmark number is recorded.

## Pillar 1 — Parity

- **Oracle A: the old pure-PyTorch implementations.** Extracted from git
  history (`main` @ f19d7b4) into `tests/oracle/` — test-only code, never
  shipped in a wheel. This resolves architecture.md §4: option 1 + 3
  (CUDA-required runtime, legacy demoted to oracle).
- **Oracle B: fp64 dense reference.** Compute the op densely in fp64, cast to
  the test dtype. Catches bugs Oracle A shares with us (it's the origin of our
  adjoint formulas — a wrong formula inherited from it would pass A and fail B).
- **Tolerance is a named, per-dtype policy** — never ad-hoc `atol` per test:
  f64 → tight rel (1e-12); f32 → rel 1e-5 with accumulation-order allowance
  scaled by reduction length; bf16/fp16 rows → ULP bounds vs the fp32-accum
  reference. The tolerance table lives in one fixture module; a test that needs
  a looser bound documents why inline or fails review.
- **Coverage = the full variant matrix** from map.md's Contract column:
  layout × batched/unbatched (incl. ragged nse, empty batch items) ×
  value dtype × index dtype (INDEX_DTYPES fixture) × p sweep — for forward
  *and* both gradients of every §1 op, plus composite parity for
  distributions/encoder (rsample statistics via the existing Hotelling T² /
  Nagao helpers, which stay).

## Pillar 2 — Legitimacy

- **`torch.library.opcheck` on every `tsgu::` op** — schema, fake-kernel
  consistency, autograd registration correctness. Run with a representative
  input set: requires_grad on/off, every supported device, strided *and*
  non-contiguous dense operands. Fake kernels are additionally tested for
  value-independence (shapes derive from index lengths + shape args only).
- **`torch.autograd.gradcheck` in f64** on the values tensors — the adjoint
  formula proven numerically, per op, per layout, batched and not. Higher-order
  (gradgradcheck) where the current package supports it (solve ops — see PR #85).
- **Property-based structural invariants** (hypothesis): for generated random
  inputs — gradA indices identical to A's (pattern preservation is an equality,
  not a tolerance); output layout == input layout; index dtype preserved;
  `seglse_bidir` ≡ two single-dim calls; `coo2csr∘csr2coo` = identity.
- **Adversarial numerics** (seglse family especially): ±inf values, all-max
  ties, magnitudes near dtype max (max-shift must prevent overflow), empty
  segments with/without `include_zeros`, single-entry segments. Deviations
  here are bugs, not tolerance cases.
- **Determinism:** run-twice bitwise equality for every kernel that claims
  determinism; kernels with an atomic fast path are tested under
  `torch.use_deterministic_algorithms(True)` to confirm the slow path engages
  (policy: kernels.md open Q1).
- **Contract conformance:** error-message tests per map.md invariant 7 (raise,
  never silently accept; message states accepted logical shapes + received
  shape per naming.md template); rejection of unsupported layouts/dtypes;
  batch-size mismatch; `_nnz()` semantics assertions from naming.rst.
- **Workspace assertion** (shared with benchmarks): peak memory minus
  tensors ≤ O(nse) bound — a legitimacy property, enforced in both suites.

## Gates & ordering

GPU CI stages, in order — all must pass before benchmarks run or a PR merges:

1. smoke (one tiny input per op — fail fast)
2. opcheck (all ops × representative inputs)
3. parity (Oracle A + B, full matrix)
4. gradcheck / gradgradcheck (f64)
5. property + adversarial + determinism
6. contract conformance + workspace assertion

CPU CI (no GPU): fake-kernel/opcheck subset, contract conformance, oracle
self-consistency (A vs B on CPU), pyrefly/ruff. This is the merge gate for
non-kernel changes.

## Open questions

None.

Resolved:

- **No multi-GPU for now** — single-GPU only; kernels still honour the
  current-stream rule (kernels.md) and take a `CUDAGuard` on the input's
  device, so non-default single-device works, but no multi-device tests or
  claims until there's a user need.
- **Hypothesis budget (norm-checked):** registered profiles selected via
  `HYPOTHESIS_PROFILE` — `pr` profile `max_examples=50` (CI stays quick),
  `nightly` profile `max_examples=1000`; `deadline=None` in CI (Hypothesis's
  own CI default). Default profile for local dev = 100 (library default).
- **opcheck × compile (norm-checked):** trust opcheck — its
  `test_aot_dispatch_dynamic` already compares outputs *and* gradients between
  eager and torch.compile. We add one `torch.compile` smoke test per **public
  wrapper** (not per op), because the unwrap/rewrap logic in `ops/` lives
  outside the `tsgu::` ops and opcheck never sees it.

Sources: [PyTorch custom-op testing tutorial](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html),
[torch.library docs](https://docs.pytorch.org/docs/2.9/library.html),
[Correctness Illusion in LLM-Generated GPU Kernels](https://arxiv.org/pdf/2606.20128),
[Reliable Kernel Correctness Check (DeepReinforce)](https://deep-reinforce.com/correctness_check.html),
[Kernel Contracts](https://arxiv.org/html/2604.22032).
