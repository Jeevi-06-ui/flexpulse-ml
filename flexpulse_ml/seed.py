"""
Seed management module for reproducible stochastic simulation and modeling.
"""

from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np


class SeedManager:
    """Manages pseudo-random number generator states across Python and NumPy."""

    def __init__(self, master_seed: int = 42) -> None:
        self._master_seed = master_seed
        self.set_seed(master_seed)

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def set_seed(self, seed: Optional[int] = None) -> int:
        chosen_seed = self._master_seed if seed is None else int(seed)
        random.seed(chosen_seed)
        np.random.seed(chosen_seed)
        os.environ["PYTHONHASHSEED"] = str(chosen_seed)
        return chosen_seed

    def spawn_run_seed(self, run_index: int) -> int:
        rng = np.random.default_rng(self._master_seed + run_index)
        return int(rng.integers(0, 2**31 - 1))

    def get_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        effective_seed = self._master_seed if seed is None else int(seed)
        return np.random.default_rng(effective_seed)
