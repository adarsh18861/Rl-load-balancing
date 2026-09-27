"""Seeding utilities. All randomness in this project flows through an
explicitly-passed numpy Generator; no global RNG state is ever used."""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a fresh, independent numpy Generator from an integer seed."""
    return np.random.default_rng(seed)
