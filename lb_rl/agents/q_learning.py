"""Tabular Q-learning with epsilon-greedy exploration and linear epsilon
decay. State = whatever observation the observation pipeline produces
(full state for M3; noisy/delayed/estimated state from M4 on)."""

from typing import Sequence

import numpy as np

from lb_rl.utils.state_encoding import encode, state_count


class QLearningAgent:
    def __init__(
        self,
        radices: Sequence[int],
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
    ):
        self.radices = list(radices)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.Q = np.zeros((state_count(self.radices), n_actions))
        self.t = 0
        self.greedy = False

    def epsilon(self) -> float:
        if self.greedy:
            return 0.0
        frac = min(1.0, self.t / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        s = encode(obs, self.radices)
        if rng.random() < self.epsilon():
            return int(rng.integers(self.n_actions))
        row = self.Q[s]
        best = np.flatnonzero(row == row.max())
        return int(rng.choice(best))

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray) -> None:
        s = encode(obs, self.radices)
        s2 = encode(next_obs, self.radices)
        target = reward + self.gamma * self.Q[s2].max()
        self.Q[s, action] += self.alpha * (target - self.Q[s, action])
        self.t += 1

    def set_greedy(self, flag: bool) -> None:
        self.greedy = flag

    def reset(self) -> None:
        pass

    def greedy_policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)
