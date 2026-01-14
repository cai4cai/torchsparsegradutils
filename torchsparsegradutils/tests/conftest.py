import os
import random

import pytest
import torch


@pytest.fixture(autouse=True)
def seed_rng():
    unlock = os.getenv("UNLOCK_SEED")
    if unlock is None or unlock.lower() == "false":
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        random.seed(42)
        yield
        torch.set_rng_state(rng_state)
    else:
        yield
