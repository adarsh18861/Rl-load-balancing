"""Mixed-radix encoding of per-server queue-length states into a single
integer index, for array-indexed tabular Q-tables and the exact VI/PI
transition model."""

from typing import Sequence

import numpy as np


def state_count(radices: Sequence[int]) -> int:
    """Total number of distinct states for the given per-feature radices."""
    count = 1
    for r in radices:
        count *= r
    return count


def assert_state_count(radices: Sequence[int], max_states: int) -> int:
    """Raise ValueError if the state space exceeds max_states; else return it."""
    n = state_count(radices)
    if n > max_states:
        raise ValueError(
            f"State space has {n} states, exceeding the configured limit "
            f"of {max_states}. Reduce servers, Q_MAX, or bucket more coarsely."
        )
    return n


def encode(values: Sequence[int], radices: Sequence[int]) -> int:
    """Mixed-radix encode a per-server feature vector into a single integer.

    values[i] must be in [0, radices[i]).
    """
    index = 0
    for v, r in zip(values, radices):
        if not (0 <= v < r):
            raise ValueError(f"value {v} out of range for radix {r}")
        index = index * r + v
    return index


def decode(index: int, radices: Sequence[int]) -> np.ndarray:
    """Inverse of encode: integer index -> per-server feature vector."""
    n = len(radices)
    values = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        r = radices[i]
        values[i] = index % r
        index //= r
    return values
