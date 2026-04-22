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
