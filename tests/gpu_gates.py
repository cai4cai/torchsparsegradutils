"""tests/gpu_gates.py — the six-stage GPU gate runner.

spec/testing.md "Gates & ordering", spec/commit.md Phase 2 #11. Invoked by
``tox -e gpu`` (``python -m tests.gpu_gates``). Runs pytest markers
``gate1``..``gate6``, **in order**, **fail-fast**: any stage whose pytest run
exits non-zero (and isn't just "no tests collected", see below) stops the
run — later stages don't execute.

Stage -> marker -> spec/testing.md stage name::

    1  gate1  smoke      (one tiny input per op — fail fast)
    2  gate2  opcheck    (all ops x representative inputs)
    3  gate3  parity     (Oracle A + B, full matrix)
    4  gate4  gradcheck  (f64, + gradgradcheck where supported)
    5  gate5  property + adversarial + determinism
    6  gate6  contract conformance + workspace assertion

Stages 2, 4, 5, 6 are EMPTY registries right now — the kernels those gates
exercise land in Phase 3 (spec/commit.md commits 12-19). "No tests collected
for this marker" (pytest exit code 5) is treated as a pass, not a failure:
an empty gate is vacuously green, and the point of this commit is the
runner + marker mechanism, not populated stages 2/4/5/6.

Populated today:
  - stage 1 (``tests/test_gate1_smoke.py``): the ``tsgu::_smoke`` bring-up
    op (commit 10). Its CUDA-executing test SKIPS cleanly (not an error)
    without a GPU + backend — that skip is this whole script's expected
    degraded-mode outcome for gate 1 on a machine with no GPU.
  - stage 3 (``tests/test_gate3_parity.py``): the Oracle A vs Oracle B
    self-consistency check testing.md's "CPU CI (no GPU)" bullet calls out
    by name — runs (and must pass) on CPU, no GPU needed.

Full GPU validation of all six stages happens once real kernels exist
(commit 12 onward) on a machine with a working GPU.
"""

from __future__ import annotations

import subprocess
import sys

# pytest's exit code for "the collected test count was zero" (e.g. no test
# in the repo carries this marker yet). Distinguishing this from a genuine
# failure is what lets stages 2/4/5/6 be legitimately empty right now.
_NO_TESTS_COLLECTED = 5

STAGES: tuple[tuple[int, str, str], ...] = (
    (1, "gate1", "smoke"),
    (2, "gate2", "opcheck"),
    (3, "gate3", "parity"),
    (4, "gate4", "gradcheck"),
    (5, "gate5", "property + adversarial + determinism"),
    (6, "gate6", "contract conformance + workspace assertion"),
)


def run_stage(num: int, marker: str, label: str) -> int:
    print(f"\n=== gate {num}/6: {label}  (pytest -m {marker}) ===", flush=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-m", marker])
    if result.returncode == _NO_TESTS_COLLECTED:
        print(
            f"--- gate {num} ({label}): no tests registered for -m {marker} yet "
            "-- empty registry, treated as a pass (spec/commit.md #11: stages "
            "2-6 are mostly empty until Phase 3's kernel commits populate them)."
        )
        return 0
    return result.returncode


def main() -> int:
    for num, marker, label in STAGES:
        code = run_stage(num, marker, label)
        if code != 0:
            print(
                f"\nGATE FAILURE at stage {num} ({label}) -- fail-fast per "
                "spec/testing.md's ordering; later stages did not run.",
                file=sys.stderr,
            )
            return code
    print("\nAll six gates passed (populated stages green, empty stages vacuously so).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
