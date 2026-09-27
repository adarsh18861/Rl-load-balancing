import numpy as np

from lb_rl.agents.policy_iteration import policy_iteration
from lb_rl.agents.q_learning import QLearningAgent
from lb_rl.env.environment import LoadBalancingEnv
from lb_rl.env.traffic import SteadyTraffic
from lb_rl.models.transition_model import build_transition_model
from lb_rl.utils.seeding import make_rng

MU = (0.6, 0.3)
Q_MAX = 2
LAM = 0.5


def train(seed, n_steps=60_000):
    env = LoadBalancingEnv(n_servers=2, mu=MU, q_max=Q_MAX, traffic=SteadyTraffic(lam=LAM), overload_penalty=5.0)
    agent = QLearningAgent(
        radices=[Q_MAX + 1, Q_MAX + 1], n_actions=2, alpha=0.1, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=n_steps // 2,
    )
    rng = make_rng(seed)
    state = env.reset()
    for _ in range(n_steps):
        action = agent.act(state, rng)
        next_state, reward, info = env.step(action, rng)
        agent.update(state, action, reward, next_state)
        state = next_state
    return agent, env


def test_q_learning_agrees_with_vi_on_visited_states():
    model = build_transition_model(n_servers=2, mu=MU, q_max=Q_MAX, lam=LAM, overload_penalty=5.0)
    _, vi_policy = policy_iteration(model, gamma=0.95)

    agent, env = train(seed=0)

    # Evaluate greedy policy while counting state visits.
    agent.set_greedy(True)
    rng = make_rng(1)
    state = env.reset()
    visits = np.zeros(model.num_states, dtype=np.int64)
    from lb_rl.utils.state_encoding import encode

    agree = 0
    total = 0
    for _ in range(20_000):
        s_idx = encode(state, model.radices)
        visits[s_idx] += 1
        action = agent.act(state, rng)
        if vi_policy[s_idx] == action:
            agree += 1
        total += 1
        state, reward, info = env.step(action, rng)

    # Restrict comparison to states visited at least a handful of times.
    frequently_visited = visits >= 20
    assert frequently_visited.sum() > 0
    assert agree / total > 0.85


def test_same_seed_reproducible_training():
    agent1, _ = train(seed=5, n_steps=5000)
    agent2, _ = train(seed=5, n_steps=5000)
    assert np.array_equal(agent1.Q, agent2.Q)
