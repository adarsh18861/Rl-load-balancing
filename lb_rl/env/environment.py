"""Slotted-time load balancing environment.

Each slot: (1) at most one job arrives, (2) if it arrived, the agent's
chosen action dispatches it to a server (or it is dropped if that server's
queue is full), (3) each server completes at most one queued job with
probability mu_i.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .servers import ServerPool


@dataclass
class StepInfo:
    arrived: bool
    dropped: bool
    response_times: List[int] = field(default_factory=list)


class LoadBalancingEnv:
    """Gym-like but dependency-free environment. All randomness passed in."""

    def __init__(self, n_servers: int, mu, q_max: int, traffic, overload_penalty: float):
        if len(mu) != n_servers:
            raise ValueError("len(mu) must equal n_servers")
        self.n_servers = n_servers
        self.q_max = q_max
        self.traffic = traffic
        self.overload_penalty = overload_penalty
        self.servers = ServerPool(mu, q_max)
        self.slot = 0

    def reset(self) -> np.ndarray:
        self.servers.reset()
        self.traffic.reset()
        self.slot = 0
        return self.servers.queue_lengths()

    def step(self, action: int, rng: np.random.Generator) -> Tuple[np.ndarray, float, StepInfo]:
        """Advance one slot. `action` selects the server for this slot's
        arrival (ignored if there is no arrival). Returns
        (next_state, reward, info)."""
        arrived = self.traffic.sample(rng, self.slot)
        dropped = False
        if arrived:
            accepted = self.servers.dispatch(action, self.slot)
            dropped = not accepted

        response_times = self.servers.serve(rng, self.slot)

        next_state = self.servers.queue_lengths()
        reward = -float(next_state.sum()) - self.overload_penalty * (1.0 if dropped else 0.0)

        info = StepInfo(arrived=arrived, dropped=dropped, response_times=response_times)
        self.slot += 1
        return next_state, reward, info

    def true_state(self) -> np.ndarray:
        return self.servers.queue_lengths()
