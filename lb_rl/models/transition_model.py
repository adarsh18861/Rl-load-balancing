"""Exact transition model P[s,a,s'] and expected reward R[s,a] for the
fully-observed, steady-Bernoulli-traffic load balancing MDP.

Used only by Value Iteration / Policy Iteration (design decision 5): this
is the model-based optimum for that specific setting, not a universal
upper bound over all traffic patterns or partial-observability configs.

Per slot: with probability `lam` one job arrives and is dispatched to the
action's server (dropped if that server's queue is already full); then
each server independently completes one queued job with probability
mu_i. Each (s, a) therefore has at most 2 (arrival branches) * 2^N
(per-server completion outcomes) successor states.
"""

from dataclasses import dataclass
from itertools import product
from typing import List, Tuple

import numpy as np

from lb_rl.utils.state_encoding import decode, encode, state_count


@dataclass
class TransitionModel:
    n_servers: int
    q_max: int
    mu: np.ndarray
    lam: float
    overload_penalty: float
    num_states: int
    num_actions: int
    # P_succ[s][a] = (successor_state_indices, probabilities)
    P_succ: List[List[Tuple[np.ndarray, np.ndarray]]]
    R: np.ndarray  # shape (num_states, num_actions)

    @property
    def radices(self) -> List[int]:
        return [self.q_max + 1] * self.n_servers

    def check_rows_sum_to_one(self, atol: float = 1e-9) -> None:
        for s in range(self.num_states):
            for a in range(self.num_actions):
                _, probs = self.P_succ[s][a]
                total = probs.sum()
                if abs(total - 1.0) > atol:
                    raise AssertionError(
                        f"P[{s},{a},:] sums to {total}, not 1 (within {atol})"
                    )


def _service_outcomes(mid: np.ndarray, mu: np.ndarray):
    """Yield (next_state_vector, probability) for every completion-mask
    combination given a post-arrival mid-slot state."""
    busy = np.flatnonzero(mid > 0)
    if busy.size == 0:
        yield mid.copy(), 1.0
        return
    for combo in product([0, 1], repeat=busy.size):
        next_state = mid.copy()
        p = 1.0
        for idx, complete in zip(busy, combo):
            if complete:
                next_state[idx] -= 1
                p *= mu[idx]
            else:
                p *= 1.0 - mu[idx]
        yield next_state, p


def build_transition_model(
    n_servers: int,
    mu,
    q_max: int,
    lam: float,
    overload_penalty: float,
    max_states: int = 50_000,
) -> TransitionModel:
    """Build the exact P and R tables for the steady-Bernoulli, full-obs MDP."""
    mu = np.asarray(mu, dtype=np.float64)
    radices = [q_max + 1] * n_servers
    num_states = state_count(radices)
    if num_states > max_states:
        raise ValueError(
            f"State space has {num_states} states, exceeding max_states={max_states}."
        )
    num_actions = n_servers

    P_succ: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    R = np.zeros((num_states, num_actions), dtype=np.float64)

    for s in range(num_states):
        state_vec = decode(s, radices)
        row: List[Tuple[np.ndarray, np.ndarray]] = []
        for a in range(num_actions):
            succ_probs: dict = {}

            # Branch: no arrival (prob 1 - lam)
            for next_state, p in _service_outcomes(state_vec, mu):
                idx = int(encode(next_state, radices))
                succ_probs[idx] = succ_probs.get(idx, 0.0) + (1.0 - lam) * p

            # Branch: arrival dispatched to server a (prob lam)
            mid = state_vec.copy()
            dropped = mid[a] >= q_max
            if not dropped:
                mid[a] += 1
            for next_state, p in _service_outcomes(mid, mu):
                idx = int(encode(next_state, radices))
                succ_probs[idx] = succ_probs.get(idx, 0.0) + lam * p

            succ_idx = np.array(list(succ_probs.keys()), dtype=np.int64)
            succ_p = np.array(list(succ_probs.values()), dtype=np.float64)
            row.append((succ_idx, succ_p))

            expected_next_sum = float(np.dot(succ_p, [decode(i, radices).sum() for i in succ_idx]))
            drop_prob = lam if state_vec[a] >= q_max else 0.0
            R[s, a] = -expected_next_sum - overload_penalty * drop_prob

        P_succ.append(row)

    return TransitionModel(
        n_servers=n_servers,
        q_max=q_max,
        mu=mu,
        lam=lam,
        overload_penalty=overload_penalty,
        num_states=num_states,
        num_actions=num_actions,
        P_succ=P_succ,
        R=R,
    )
