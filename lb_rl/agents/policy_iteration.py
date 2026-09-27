"""Policy Iteration on the exact transition model (full observability,
steady traffic only)."""

from typing import Tuple

import numpy as np

from lb_rl.models.transition_model import TransitionModel
from lb_rl.utils.state_encoding import encode


def _evaluate_policy(
    model: TransitionModel, policy: np.ndarray, gamma: float, theta: float, max_iters: int
) -> np.ndarray:
    V = np.zeros(model.num_states)
    for _ in range(max_iters):
        V_new = np.zeros_like(V)
        for s in range(model.num_states):
            a = policy[s]
            succ_idx, succ_p = model.P_succ[s][a]
            V_new[s] = model.R[s, a] + gamma * np.dot(succ_p, V[succ_idx])
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    return V


def policy_iteration(
    model: TransitionModel,
    gamma: float,
    theta: float = 1e-6,
    max_eval_iters: int = 10_000,
    max_improve_iters: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (V, policy) after policy iteration converges to a stable policy."""
    policy = np.zeros(model.num_states, dtype=np.int64)
    for _ in range(max_improve_iters):
        V = _evaluate_policy(model, policy, gamma, theta, max_eval_iters)
        new_policy = np.zeros_like(policy)
        for s in range(model.num_states):
            q_s = np.array(
                [
                    model.R[s, a] + gamma * np.dot(model.P_succ[s][a][1], V[model.P_succ[s][a][0]])
                    for a in range(model.num_actions)
                ]
            )
            new_policy[s] = q_s.argmax()
        if np.array_equal(new_policy, policy):
            policy = new_policy
            break
        policy = new_policy
    V = _evaluate_policy(model, policy, gamma, theta, max_eval_iters)
    return V, policy


class PolicyIterationPolicy:
    """Greedy policy from a precomputed PI table. Full observability only."""

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
