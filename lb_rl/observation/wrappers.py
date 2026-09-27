"""Observation wrappers, applied to every state-dependent policy (Least
Connections, SED, and all RL agents). Round Robin ignores its `obs`
argument entirely and is therefore naturally immune.

Common interface:
    reset(true_state, rng) -> obs
    step(true_state, action, arrived, rng) -> obs
        `true_state`: the environment's true post-step queue lengths.
        `action`: the server the policy just dispatched to (the dispatcher
            always knows its own actions and whether an arrival occurred
            that slot -- that is not hidden information).
        `arrived`: whether a job arrived this slot.
    obs_radices(base_radices) -> radices for the produced observation
        (identity for every wrapper in this module).

Callers must drive the environment's dynamics RNG and any wrapper noise
RNG from independent streams (see eval.runner.rollout) so that noise
consumption never perturbs the environment's own randomness -- otherwise
policies that ignore `obs` (Round Robin) would stop being reproducibly
immune to the observation mode.
"""

from collections import deque
from typing import Sequence

import numpy as np


class FullObservation:
    """Passthrough: the agent sees the true queue lengths."""

    def reset(self, true_state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return true_state.copy()

    def step(self, true_state, action, arrived, rng) -> np.ndarray:
        return true_state.copy()

    def obs_radices(self, base_radices: Sequence[int]) -> list:
        return list(base_radices)


class NoiseObservation:
    """Each observed queue is independently perturbed by -1/0/+1: with
    probability p_noise the perturbation is +/-1 (chosen uniformly),
    otherwise 0. Clipped to [0, q_max]."""

    def __init__(self, p_noise: float, q_max: int):
        self.p_noise = p_noise
        self.q_max = q_max

    def _perturb(self, true_state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(true_state)
        changed = rng.random(n) < self.p_noise
        signs = rng.choice([-1, 1], size=n)
        delta = np.where(changed, signs, 0)
        return np.clip(true_state + delta, 0, self.q_max).astype(np.int64)

    def reset(self, true_state, rng) -> np.ndarray:
        return self._perturb(true_state, rng)

    def step(self, true_state, action, arrived, rng) -> np.ndarray:
        return self._perturb(true_state, rng)

    def obs_radices(self, base_radices: Sequence[int]) -> list:
        return list(base_radices)


class DelayObservation:
    """The agent sees the true-state snapshot from k slots ago."""

    def __init__(self, k: int):
        self.k = k
        self.history: deque = deque(maxlen=k + 1)

    def reset(self, true_state, rng) -> np.ndarray:
        self.history = deque([true_state.copy() for _ in range(self.k + 1)], maxlen=self.k + 1)
        return self.history[0].copy()

    def step(self, true_state, action, arrived, rng) -> np.ndarray:
        self.history.append(true_state.copy())
        return self.history[0].copy()

    def obs_radices(self, base_radices: Sequence[int]) -> list:
        return list(base_radices)


class NoiseDelayObservation:
    """Composes DelayObservation then NoiseObservation: the agent sees a
    noisy version of the k-slot-old snapshot."""

    def __init__(self, k: int, p_noise: float, q_max: int):
        self.delay = DelayObservation(k)
        self.noise = NoiseObservation(p_noise, q_max)

    def reset(self, true_state, rng) -> np.ndarray:
        return self.noise._perturb(self.delay.reset(true_state, rng), rng)

    def step(self, true_state, action, arrived, rng) -> np.ndarray:
        return self.noise._perturb(self.delay.step(true_state, action, arrived, rng), rng)

    def obs_radices(self, base_radices: Sequence[int]) -> list:
        return list(base_radices)


class DispatchCorrectorWrapper:
    """Belief-state approximation: corrects a k-slot-old snapshot using the
    dispatcher's own knowledge of what it has sent since that snapshot was
    taken. queue_i_est = clip(stale_q_i + dispatched_i_since_snapshot -
    mu_i * k, 0, q_max), rounded. At k=0 with no noise this is exactly the
    true state.

    `base` is the observation wrapper that produces the (possibly noisy)
    k-slot-old snapshot. If None, a noise-free DelayObservation(k) is used."""

    def __init__(self, k: int, mu: Sequence[float], q_max: int, n_servers: int, base=None):
        self.k = k
        self.mu = np.asarray(mu, dtype=np.float64)
        self.q_max = q_max
        self.n_servers = n_servers
        self.delay = base if base is not None else DelayObservation(k)
        self.action_window: deque = deque(maxlen=max(k, 1))

    def _dispatched_since_snapshot(self) -> np.ndarray:
        counts = np.zeros(self.n_servers, dtype=np.float64)
        for action, arrived in self.action_window:
            if arrived:
                counts[action] += 1
        return counts

    def _estimate(self, stale: np.ndarray) -> np.ndarray:
        dispatched = self._dispatched_since_snapshot()
        est = stale + dispatched - self.mu * self.k
        return np.clip(np.round(est), 0, self.q_max).astype(np.int64)

    def reset(self, true_state, rng) -> np.ndarray:
        stale = self.delay.reset(true_state, rng)
        self.action_window = deque(maxlen=max(self.k, 1))
        return self._estimate(stale)

    def step(self, true_state, action, arrived, rng) -> np.ndarray:
        stale = self.delay.step(true_state, action, arrived, rng)
        if self.k > 0:
            self.action_window.append((action, arrived))
        return self._estimate(stale)

    def obs_radices(self, base_radices: Sequence[int]) -> list:
        return list(base_radices)


class BeliefStateWrapper:
    """EXTENSION: exact per-server Bayesian belief filter.

    Given the dispatcher's own (action, arrived) log, each server's queue
    evolves independently, so the POMDP belief factorises exactly into one
    small distribution b_i(q), q in {0..q_max}, per server.

    Each slot:
      predict: b_i <- b_i @ A (only if a job was sent to i) @ S_i
      update : b_i(q) proportional to b_i(q) * P(observed o_i | true q)
    With delay k we keep the belief at the snapshot time, update it with the
    delayed (possibly noisy) reading, then roll it forward through the last
    k logged dispatches. Output = rounded posterior mean per server, so the
    Q-table size is unchanged (5^5 = 3125 states).

    `base` must be the same observation wrapper the experiment uses
    (Delay / Noise / NoiseDelay); k and p_noise must match its settings."""

    def __init__(self, base, k: int, p_noise: float, mu: Sequence[float], q_max: int, n_servers: int):
        self.base = base
        self.k = k
        self.q_max = q_max
        self.n_servers = n_servers
        Q = q_max + 1
        mu = np.asarray(mu, dtype=np.float64)

        # Service matrices: S[i, q, q'] = P(q -> q') for server i in one slot.
        self.S = np.zeros((n_servers, Q, Q))
        for i in range(n_servers):
            self.S[i, 0, 0] = 1.0
            for q in range(1, Q):
                self.S[i, q, q - 1] = mu[i]
                self.S[i, q, q] = 1.0 - mu[i]

        # Arrival matrix: q -> q+1, but stays at q_max (job dropped).
        self.A = np.zeros((Q, Q))
        for q in range(Q):
            self.A[q, min(q + 1, q_max)] = 1.0

        # Noise likelihood: L[o, q] = P(observed o | true q), matches NoiseObservation (+/-1, clipped).
        self.L = np.zeros((Q, Q))
        for q in range(Q):
            self.L[q, q] += 1.0 - p_noise
            self.L[max(q - 1, 0), q] += p_noise / 2.0
            self.L[min(q + 1, q_max), q] += p_noise / 2.0

        self.window: deque = deque()
        self.b_snap = None
        self.mean = None  # posterior mean per server (useful for analysis)
        self.std = None   # posterior std per server (a "confidence" signal)

    def _predict(self, b: np.ndarray, action: int, arrived: bool) -> np.ndarray:
        b = b.copy()
        if arrived:
            b[action] = b[action] @ self.A
        return np.einsum("iq,iqr->ir", b, self.S)

    def _update(self, b: np.ndarray, obs: np.ndarray) -> np.ndarray:
        b = b * self.L[obs, :]
        total = b.sum(axis=1, keepdims=True)
        bad = total[:, 0] == 0
        if bad.any():  # safety net: fall back to the likelihood alone
            b[bad] = self.L[obs[bad], :]
            total[bad] = b[bad].sum(axis=1, keepdims=True)
        return b / total

    def _estimate(self) -> np.ndarray:
        b = self.b_snap
        for action, arrived in self.window:
            b = self._predict(b, action, arrived)
        qs = np.arange(self.q_max + 1)
        self.mean = b @ qs
        self.std = np.sqrt(np.maximum(b @ (qs ** 2) - self.mean ** 2, 0.0))
        return np.clip(np.round(self.mean), 0, self.q_max).astype(np.int64)

    def reset(self, true_state, rng) -> np.ndarray:
        self.base.reset(true_state, rng)
        Q = self.q_max + 1
        # The system starts empty, which the dispatcher knows.
        self.b_snap = np.zeros((self.n_servers, Q))
        self.b_snap[np.arange(self.n_servers), np.asarray(true_state, dtype=np.int64)] = 1.0
        self.window = deque()
        return self._estimate()

    def step(self, true_state, action, arrived, rng) -> np.ndarray:
        obs = np.asarray(self.base.step(true_state, action, arrived, rng), dtype=np.int64)
        self.window.append((action, arrived))
        if len(self.window) > self.k:
            old_action, old_arrived = self.window.popleft()
            self.b_snap = self._predict(self.b_snap, old_action, old_arrived)
        self.b_snap = self._update(self.b_snap, obs)
        return self._estimate()

    def obs_radices(self, base_radices: Sequence[int]) -> list:
        return list(base_radices)
