import os
import random

import numpy as np
import pytest
import torch

SEED = 42


def _seed_unlocked() -> bool:
    unlock = os.getenv("TSGU_UNLOCK_SEED", os.getenv("UNLOCK_SEED", "false"))
    return unlock.lower() in {"1", "true", "yes", "on"}


def _seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pytest_addoption(parser):
    parser.addoption(
        "--run-manual-cuda",
        action="store_true",
        default=False,
        help="Run CUDA memory/performance/OOM tests marked manual_cuda.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "manual_cuda: CUDA memory/performance/OOM tests excluded by default")
    if not _seed_unlocked():
        _seed_all()


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-manual-cuda"):
        return

    skip_manual_cuda = pytest.mark.skip(reason="manual CUDA test; use --run-manual-cuda to run")
    for item in items:
        if "manual_cuda" in item.keywords:
            item.add_marker(skip_manual_cuda)


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
