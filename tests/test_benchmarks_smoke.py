"""CPU smoke tests for the benchmarks/ harness (spec/commit.md Phase 2 #11).

Not part of the six gates (those are for `tsgu::` op correctness/perf) --
this just proves the harness/memory/env/corpus/results plumbing itself
works in degraded (no-GPU) mode, independent of any op.
"""

from __future__ import annotations

import json
import warnings

import torch

from benchmarks import corpus, harness, memory
from benchmarks.env import fingerprint
from benchmarks.results import SCHEMA_FIELDS, Result, write_result


def test_do_bench_degrades_cleanly_without_cuda(tmp_path):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    result = harness.do_bench(fn, device=torch.device("cpu"))
    assert result.degraded is True
    assert result.n_iters > 0
    assert result.median_ms >= 0
    assert result.p10_ms <= result.median_ms <= result.p90_ms or result.n_iters == 1
    assert calls["n"] > 0


def test_memory_measure_degrades_cleanly_without_cuda():
    out = {}

    def fwd():
        out["x"] = torch.randn(4)
        return out["x"]

    result = memory.measure(fwd, device=torch.device("cpu"))
    assert result.degraded is True
    assert result.peak_fwd_mb is None
    assert result.peak_bwd_mb is None
    assert result.workspace_mb is None
    assert "x" in out  # forward_fn still actually ran


def test_corpus_vram_fallback_and_scaling():
    # With no CUDA, available_vram_gb() must fall back to the documented
    # constant rather than raising.
    assert corpus.available_vram_gb() == corpus.THIS_MACHINE_VRAM_GB_FALLBACK
    budget = corpus.peak_budget_gb(4.0)
    assert budget <= corpus.ABSOLUTE_PEAK_BUDGET_GB
    assert budget <= 4.0 * corpus.PEAK_BUDGET_FRACTION


def test_corpus_sweep_configs_shape_matches_spec():
    configs = corpus.sweep_configs(p_values=(8,))
    seen_b = {c.B for c in configs}
    seen_n = {c.n for c in configs}
    assert seen_b == set(corpus.SWEEP_B)
    assert seen_n == set(corpus.SWEEP_N)
    for c in configs:
        assert 1 <= c.nse_per_item <= c.n * c.m


def test_corpus_smoke_config_builds_a_real_sparse_tensor():
    cfg = corpus.smoke_config()
    a = corpus.make_sparse_batch(cfg, device=torch.device("cpu"))
    assert a.shape == (cfg.B, cfg.n, cfg.m)
    assert a.is_sparse  # COO


def test_fingerprint_never_raises_and_notes_degraded_mode():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp = fingerprint()
    d = fp.as_dict()
    assert d["torch"] == torch.__version__
    if not torch.cuda.is_available():
        assert d["degraded_no_gpu"] is True
        assert d["gpu"] is None


def test_result_roundtrip_has_every_schema_v5_field(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.results.RESULTS_DIR", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp = fingerprint()
    result = Result.build(
        op="sparse_mm",
        family="SpMM",
        backend=None,
        variant="COO·B=1·f32/i64",
        matrix="unit-test",
        n=10,
        m=10,
        nse=20,
        p=4,
        fp=fp,
    )
    path = write_result(result)
    assert path.exists()
    data = json.loads(path.read_text())
    for field in SCHEMA_FIELDS:
        assert field in data, f"missing benchmarks.md §5 field: {field}"
