"""Server pool: per-server FIFO queues with heterogeneous service rates."""

from collections import deque
from typing import List, Sequence

import numpy as np


class ServerPool:
    """N servers, each with a bounded FIFO queue and a completion probability.

    Queue contents are tracked as arrival-slot timestamps so response time
    (completion_slot - arrival_slot) can be measured exactly.
    """

    def __init__(self, mu: Sequence[float], q_max: int):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.n = len(self.mu)
        self.q_max = q_max
        self.queues: List[deque] = [deque() for _ in range(self.n)]

    def reset(self) -> None:
        for q in self.queues:
            q.clear()

    def queue_lengths(self) -> np.ndarray:
        return np.array([len(q) for q in self.queues], dtype=np.int64)

    def dispatch(self, server: int, slot: int) -> bool:
        """Attempt to enqueue a job arriving at `slot` onto `server`.

        Returns True if accepted, False if dropped (queue was full).
        """
        if len(self.queues[server]) >= self.q_max:
            return False
        self.queues[server].append(slot)
        return True

    def serve(self, rng: np.random.Generator, current_slot: int) -> List[int]:
        """Each non-empty server completes one job with probability mu_i.

        Returns the list of response times (current_slot - arrival_slot) for
        jobs completed this slot.
        """
        response_times = []
        for i, q in enumerate(self.queues):
            if q and rng.random() < self.mu[i]:
                arrival_slot = q.popleft()
                response_times.append(current_slot - arrival_slot + 1)
        return response_times
