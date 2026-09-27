"""M4 sanity check: degradation under partial obs, and whether the two
dispatch corrector helps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lb_rl.agents.baselines import LeastConnections
from lb_rl.agents.q_learning import QLearningAgent
from lb_rl.env.environment import LoadBalancingEnv
from lb_rl.env.traffic import SteadyTraffic
from lb_rl.eval.runner import rollout
from lb_rl.observation.wrappers import (
    DelayObservation,
    DispatchCorrectorWrapper,
    FullObservation,
)

MU = [0.9, 0.7, 0.5, 0.4, 0.3]
Q_MAX = 4
LAM = 0.8
K = 5
TRAIN_STEPS = 150_000
EVAL_STEPS = 20_000


def make_env():
    return LoadBalancingEnv(n_servers=5, mu=MU, q_max=Q_MAX, traffic=SteadyTraffic(lam=LAM), overload_penalty=5.0)


def eval_lc(wrapper, seed):
    tracker, _ = rollout(make_env(), LeastConnections(), wrapper, seed=seed, n_steps=EVAL_STEPS, time_bound=5)
    return tracker.summary()


def train_and_eval_q(wrapper, radices, train_seed, eval_seed):
    agent = QLearningAgent(
        radices=radices, n_actions=5, alpha=0.1, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=TRAIN_STEPS // 2,
    )
    rollout(make_env(), agent, wrapper, seed=train_seed, n_steps=TRAIN_STEPS, learn=True)
    agent.set_greedy(True)
    tracker, _ = rollout(make_env(), agent, wrapper, seed=eval_seed, n_steps=EVAL_STEPS, time_bound=5)
    return tracker.summary()


if __name__ == "__main__":
    base_radices = [Q_MAX + 1] * 5

    print("--- Least Connections ---")
    print("full obs        :", eval_lc(FullObservation(), seed=1))
    print(f"delay k={K}       :", eval_lc(DelayObservation(k=K), seed=1))
    print(f"delay+corrector  :", eval_lc(DispatchCorrectorWrapper(k=K, mu=MU, q_max=Q_MAX, n_servers=5), seed=1))

    print("--- Q-learning ---")
    print("full obs        :", train_and_eval_q(FullObservation(), base_radices, train_seed=0, eval_seed=1))
    print(f"delay k={K}       :", train_and_eval_q(DelayObservation(k=K), base_radices, train_seed=0, eval_seed=1))
    corr = DispatchCorrectorWrapper(k=K, mu=MU, q_max=Q_MAX, n_servers=5)
    print(f"delay+corrector  :", train_and_eval_q(corr, base_radices, train_seed=0, eval_seed=1))
