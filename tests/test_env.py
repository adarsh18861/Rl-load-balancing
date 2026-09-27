import numpy as np

from lb_rl.env.environment import LoadBalancingEnv
from lb_rl.env.traffic import SteadyTraffic
from lb_rl.agents.baselines import LeastConnections, RoundRobin
from lb_rl.utils.seeding import make_rng


def make_env(lam=0.8, mu=(0.9, 0.7, 0.5, 0.4, 0.3), q_max=4, overload_penalty=5.0):
    traffic = SteadyTraffic(lam=lam)
    return LoadBalancingEnv(n_servers=len(mu), mu=mu, q_max=q_max, traffic=traffic, overload_penalty=overload_penalty)


def test_job_conservation():
    env = make_env()
    policy = LeastConnections()
    rng = make_rng(0)
    state = env.reset()
    arrivals = completions = drops = 0
    for _ in range(2000):
        action = policy.act(state, rng)
        state, reward, info = env.step(action, rng)
        if info.arrived:
            arrivals += 1
        if info.dropped:
            drops += 1
        completions += len(info.response_times)
    in_queue = int(state.sum())
    assert arrivals == completions + drops + in_queue


def test_same_seed_identical_results():
    def run():
        env = make_env()
        policy = LeastConnections()
        rng = make_rng(42)
        state = env.reset()
        rewards = []
        for _ in range(200):
            action = policy.act(state, rng)
            state, reward, info = env.step(action, rng)
            rewards.append(reward)
        return rewards

    assert run() == run()


def test_round_robin_ignores_state():
    env = make_env()
    policy = RoundRobin(n_servers=5)
    rng = make_rng(1)
    state = env.reset()
    actions = []
    for _ in range(20):
        action = policy.act(state, rng)
        actions.append(action)
        state, reward, info = env.step(action, rng)
    assert actions == [i % 5 for i in range(20)]


def test_queue_never_exceeds_cap():
    env = make_env(q_max=4)
    policy = LeastConnections()
    rng = make_rng(7)
    state = env.reset()
    for _ in range(2000):
        action = policy.act(state, rng)
        state, reward, info = env.step(action, rng)
        assert state.max() <= 4
