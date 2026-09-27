"""Value Iteration on the exact transition model (full observability,
steady traffic only)."""

from typing import Tuple

import numpy as np

from lb_rl.models.transition_model import TransitionModel
from lb_rl.utils.state_encoding import encode


def value_iteration(
    model: TransitionModel, gamma: float, theta: float = 1e-6, max_iters: int = 10_000
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (V, policy): V[s] optimal value, policy[s] optimal action."""
    V = np.zeros(model.num_states)
    for _ in range(max_iters):
        Q = _compute_q(model, V, gamma)
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    Q = _compute_q(model, V, gamma)
    policy = Q.argmax(axis=1)
    return V, policy


def _compute_q(model: TransitionModel, V: np.ndarray, gamma: float) -> np.ndarray:
    Q = np.zeros((model.num_states, model.num_actions))
    for s in range(model.num_states):
        for a in range(model.num_actions):
            succ_idx, succ_p = model.P_succ[s][a]
            Q[s, a] = model.R[s, a] + gamma * np.dot(succ_p, V[succ_idx])
    return Q


class ValueIterationPolicy:
    """Greedy policy from a precomputed VI table. Full observability only."""

    def __init__(self, policy: np.ndarray, radices):
        self.policy = policy
        self.radices = radices

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> int:
        s = encode(obs, self.radices)
        return int(self.policy[s])

    def update(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass
