"""M1 sanity check: Round Robin vs Least Connections vs SED on steady traffic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lb_rl.agents.baselines import LeastConnections, RoundRobin, ShortestExpectedDelay
from lb_rl.env.environment import LoadBalancingEnv
from lb_rl.env.traffic import SteadyTraffic
from lb_rl.eval.metrics import MetricsTracker
from lb_rl.utils.seeding import make_rng

MU = [0.9, 0.7, 0.5, 0.4, 0.3]
Q_MAX = 4
LAM = 0.8  # arrival probability per slot (see README note on the load knob)
N_SLOTS = 20000


def run(policy_factory, seed):
    env = LoadBalancingEnv(
        n_servers=len(MU), mu=MU, q_max=Q_MAX,
        traffic=SteadyTraffic(lam=LAM), overload_penalty=5.0,
    )
    policy = policy_factory()
    rng = make_rng(seed)
    tracker = MetricsTracker(time_bound=5)
    state = env.reset()
    for _ in range(N_SLOTS):
        action = policy.act(state, rng)
        state, reward, info = env.step(action, rng)
        tracker.record(info, state)
    return tracker.summary()


if __name__ == "__main__":
    print(f"load lambda = {LAM:.3f} (arrival prob per slot; total load / sum(mu) = {LAM / sum(MU):.3f})")
    for name, factory in [
        ("RoundRobin", lambda: RoundRobin(n_servers=len(MU))),
        ("LeastConnections", LeastConnections),
        ("ShortestExpectedDelay", lambda: ShortestExpectedDelay(mu=MU)),
    ]:
        summary = run(factory, seed=0)
        print(f"{name:22s} {summary}")
