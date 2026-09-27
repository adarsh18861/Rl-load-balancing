"""EXTENSION: confidence-gated hybrid dispatcher.

Uses the trained Q-learning policy when the belief filter is confident
(largest posterior std across servers < tau) and falls back to Shortest
Expected Delay on the belief's expected queue lengths otherwise.
During training the gate is off, so Q-learning trains exactly like C11."""

import numpy as np


class ConfidenceGatedPolicy:
    def __init__(self, agent, belief, mu, tau: float = 0.5):
        self.agent = agent          # a QLearningAgent
        self.belief = belief        # the BeliefStateWrapper feeding this policy
        self.mu = np.asarray(mu, dtype=np.float64)
        self.tau = tau
        self.use_gate = False       # switched on for evaluation by set_greedy(True)

    def bind(self, belief) -> None:
        """Point the gate at a fresh BeliefStateWrapper (used for evaluation)."""
        self.belief = belief

    def act(self, obs, rng) -> int:
        std = getattr(self.belief, "std", None)
        if self.use_gate and std is not None and std.max() >= self.tau:
            expected_delay = (self.belief.mean + 1.0) / self.mu
            best = np.flatnonzero(expected_delay == expected_delay.min())
            return int(rng.choice(best))
        return self.agent.act(obs, rng)

    def update(self, *args, **kwargs) -> None:
        self.agent.update(*args, **kwargs)

    def set_greedy(self, flag: bool) -> None:
        self.agent.set_greedy(flag)
        self.use_gate = flag

    def reset(self) -> None:
        pass
