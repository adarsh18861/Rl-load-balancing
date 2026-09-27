"""M2 check: build the full N=5, Q_MAX=4 transition model, run VI/PI, and
compare against SED/LC on steady traffic."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lb_rl.agents.baselines import LeastConnections, ShortestExpectedDelay
from lb_rl.agents.policy_iteration import PolicyIterationPolicy, policy_iteration
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
N_SLOTS = 20000


def run(policy, seed):
    env = LoadBalancingEnv(
        n_servers=len(MU), mu=MU, q_max=Q_MAX,
        traffic=SteadyTraffic(lam=LAM), overload_penalty=5.0,
    )
    rng = make_rng(seed)
    tracker = MetricsTracker(time_bound=5)
    state = env.reset()
    for _ in range(N_SLOTS):
        action = policy.act(state, rng)
        state, reward, info = env.step(action, rng)
        tracker.record(info, state)
    return tracker.summary()


if __name__ == "__main__":
    t0 = time.time()
    model = build_transition_model(n_servers=5, mu=MU, q_max=Q_MAX, lam=LAM, overload_penalty=5.0)
    print(f"built model with {model.num_states} states in {time.time()-t0:.1f}s")
    model.check_rows_sum_to_one()
    print("P rows sum to 1: OK")

    t0 = time.time()
    _, vi_policy = value_iteration(model, gamma=GAMMA)
    print(f"VI converged in {time.time()-t0:.1f}s")

    t0 = time.time()
    _, pi_policy = policy_iteration(model, gamma=GAMMA)
    print(f"PI converged in {time.time()-t0:.1f}s")
    print(f"VI and PI agree on {np.mean(vi_policy == pi_policy)*100:.1f}% of states")

    for name, policy in [
        ("LeastConnections", LeastConnections()),
        ("ShortestExpectedDelay", ShortestExpectedDelay(mu=MU)),
        ("ValueIteration", ValueIterationPolicy(vi_policy, model.radices)),
        ("PolicyIteration", PolicyIterationPolicy(pi_policy, model.radices)),
    ]:
        print(name, run(policy, seed=0))
