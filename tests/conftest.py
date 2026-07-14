import os
import random

import numpy as np
import pytest
import torch
from hypothesis import settings

SEED = 42

# ---------------------------------------------------------------------------
# Hypothesis profiles (spec/testing.md "Resolved: Hypothesis budget",
# spec/commit.md #11).
#
# Three registered profiles, selected via the HYPOTHESIS_PROFILE env var:
#   - "pr"      max_examples=50   (CI on every PR — stays quick)
#   - "nightly" max_examples=1000 (deeper nightly run)
#   - "local"   max_examples=100  (library default; used when
#                                  HYPOTHESIS_PROFILE is unset)
#
# deadline=None in CI (Hypothesis's own CI default) — detected here via the
# CI env var (set by GitHub Actions and most other CI systems) rather than
# assumed, so a local run keeps Hypothesis's normal per-example deadline.
# ---------------------------------------------------------------------------


def _in_ci() -> bool:
    return os.getenv("CI", "").strip().lower() in {"1", "true", "yes", "on"}


_deadline_kwargs = {"deadline": None} if _in_ci() else {}

settings.register_profile("pr", max_examples=50, **_deadline_kwargs)
settings.register_profile("nightly", max_examples=1000, **_deadline_kwargs)
settings.register_profile("local", max_examples=100, **_deadline_kwargs)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "local"))


def _seed_unlocked() -> bool:
    unlock = os.getenv("TSGU_UNLOCK_SEED", os.getenv("UNLOCK_SEED", "false"))
    return unlock.lower() in {"1", "true", "yes", "on"}


def _seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pytest_configure(config):
    if not _seed_unlocked():
        _seed_all()


@pytest.fixture(autouse=True)
def seed_rng():
    if _seed_unlocked():
        yield
        return

    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    _seed_all()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        random.setstate(random_state)
        np.random.set_state(numpy_state)
