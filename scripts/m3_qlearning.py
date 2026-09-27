"""M3: Q-learning under full observability. Learning curve + comparison to VI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lb_rl.agents.q_learning import QLearningAgent
from lb_rl.agents.value_iteration import ValueIterationPolicy, value_iteration
from lb_rl.env.environment import LoadBalancingEnv
from lb_rl.env.traffic import SteadyTraffic
from lb_rl.eval.metrics import MetricsTracker
from lb_rl.models.transition_model import build_transition_model
from lb_rl.utils.seeding import make_rng

MU = [0.9, 0.7, 0.5, 0.4, 0.3]
Q_MAX = 4
LAM = 0.8
GAMMA = 0.95
TRAIN_STEPS = 300_000
EVAL_STEPS = 20_000


def train_q_learning(seed):
    env = LoadBalancingEnv(n_servers=5, mu=MU, q_max=Q_MAX, traffic=SteadyTraffic(lam=LAM), overload_penalty=5.0)
    agent = QLearningAgent(
        radices=[Q_MAX + 1] * 5, n_actions=5, alpha=0.1, gamma=GAMMA,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=TRAIN_STEPS // 2,
    )
    rng = make_rng(seed)
    state = env.reset()
    window = 500
    rewards = []
    moving_avg = []
    for t in range(TRAIN_STEPS):
        action = agent.act(state, rng)
        next_state, reward, info = env.step(action, rng)
        agent.update(state, action, reward, next_state)
        state = next_state
        rewards.append(reward)
        if len(rewards) >= window:
            moving_avg.append(np.mean(rewards[-window:]))
    return agent, moving_avg


def evaluate(policy, seed):
    env = LoadBalancingEnv(n_servers=5, mu=MU, q_max=Q_MAX, traffic=SteadyTraffic(lam=LAM), overload_penalty=5.0)
    rng = make_rng(seed)
    tracker = MetricsTracker(time_bound=5)
    state = env.reset()
    for _ in range(EVAL_STEPS):
        action = policy.act(state, rng)
        state, reward, info = env.step(action, rng)
        tracker.record(info, state)
    return tracker.summary()


if __name__ == "__main__":
    model = build_transition_model(n_servers=5, mu=MU, q_max=Q_MAX, lam=LAM, overload_penalty=5.0)
    _, vi_policy = value_iteration(model, gamma=GAMMA)
    vi = ValueIterationPolicy(vi_policy, model.radices)
    print("VI (eval):", evaluate(vi, seed=100))

    agent, moving_avg = train_q_learning(seed=0)
    final = moving_avg[-1]
    converged_at = next(
        (i for i, v in enumerate(moving_avg) if abs(v - final) / abs(final) < 0.05), None
    )
    print(f"Q-learning moving-avg reward converged (within 5% of final) at step ~{converged_at}")
    print(f"final moving-avg reward: {final:.3f}")

    agent.set_greedy(True)
    print("Q-learning (eval, greedy):", evaluate(agent, seed=100))

    match = np.mean(vi_policy == agent.greedy_policy())
    print(f"Q-learning greedy policy agrees with VI on {match*100:.1f}% of all states")
