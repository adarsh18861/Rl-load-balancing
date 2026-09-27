"""Non-learning baseline dispatch policies.

Common interface: act(obs, rng) -> server index. `update` is a no-op for
these but present so the runner can call it uniformly across policy types.
"""

from typing import Sequence

import numpy as np


class RoundRobin:
    """Cycles through servers in order, ignoring observed state entirely.

    Naturally immune to observation noise/delay since it never looks at obs.
    """

    def __init__(self, n_servers: int):
        self.n_servers = n_servers
        self._next = 0

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        server = self._next
        self._next = (self._next + 1) % self.n_servers
        return server

    def update(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        self._next = 0


class LeastConnections:
    """Joins the shortest observed queue; ties broken randomly."""

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        obs = np.asarray(obs)
        best = np.flatnonzero(obs == obs.min())
        return int(rng.choice(best))

    def update(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass


class ShortestExpectedDelay:
    """Joins the server minimizing (observed_queue + 1) / mu_i; ties random."""

    def __init__(self, mu: Sequence[float]):
        self.mu = np.asarray(mu, dtype=np.float64)

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        obs = np.asarray(obs)
        expected_delay = (obs + 1.0) / self.mu
        best = np.flatnonzero(expected_delay == expected_delay.min())
        return int(rng.choice(best))

    def update(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass
