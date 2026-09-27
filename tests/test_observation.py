import numpy as np

from lb_rl.agents.baselines import LeastConnections, RoundRobin
from lb_rl.env.environment import LoadBalancingEnv
from lb_rl.env.traffic import SteadyTraffic
from lb_rl.eval.runner import rollout
from lb_rl.observation.wrappers import (
    BeliefStateWrapper,
    DelayObservation,
    DispatchCorrectorWrapper,
    FullObservation,
    NoiseObservation,
)
from lb_rl.utils.seeding import make_rng

MU = [0.9, 0.7, 0.5, 0.4, 0.3]
Q_MAX = 4


def make_env():
    return LoadBalancingEnv(n_servers=5, mu=MU, q_max=Q_MAX, traffic=SteadyTraffic(lam=0.8), overload_penalty=5.0)


def test_delay_k0_equals_full_observation():
    env1, env2 = make_env(), make_env()
    m1, _ = rollout(env1, LeastConnections(), DelayObservation(k=0), seed=1, n_steps=500)
    m2, _ = rollout(env2, LeastConnections(), FullObservation(), seed=1, n_steps=500)
    assert m1.summary() == m2.summary()


def test_noise_p0_equals_full_observation():
    env1, env2 = make_env(), make_env()
    m1, _ = rollout(env1, LeastConnections(), NoiseObservation(p_noise=0.0, q_max=Q_MAX), seed=1, n_steps=500)
    m2, _ = rollout(env2, LeastConnections(), FullObservation(), seed=1, n_steps=500)
    assert m1.summary() == m2.summary()


def test_dispatch_corrector_age0_equals_true_state():
    env = make_env()
    corrector = DispatchCorrectorWrapper(k=0, mu=MU, q_max=Q_MAX, n_servers=5)
    rng = make_rng(0)
    true_state = env.reset()
    obs = corrector.reset(true_state, rng)
    assert np.array_equal(obs, true_state)
    for _ in range(50):
        action = 0
        next_true, reward, info = env.step(action, rng)
        obs = corrector.step(next_true, action, info.arrived, rng)
        assert np.array_equal(obs, next_true)


def test_round_robin_identical_across_observation_modes():
    wrappers = [
        FullObservation(),
        NoiseObservation(p_noise=0.3, q_max=Q_MAX),
        DelayObservation(k=5),
        DispatchCorrectorWrapper(k=3, mu=MU, q_max=Q_MAX, n_servers=5),
    ]
    summaries = []
    for w in wrappers:
        env = make_env()
        tracker, _ = rollout(env, RoundRobin(n_servers=5), w, seed=42, n_steps=1000)
        summaries.append(tracker.summary())
    for s in summaries[1:]:
        assert s == summaries[0]


def test_dispatch_corrector_clips_to_q_max():
    env = make_env()
    corrector = DispatchCorrectorWrapper(k=3, mu=MU, q_max=Q_MAX, n_servers=5)
    rng = make_rng(0)
    true_state = env.reset()
    obs = corrector.reset(true_state, rng)
    for _ in range(200):
        action = 0
        next_true, reward, info = env.step(action, rng)
        obs = corrector.step(next_true, action, info.arrived, rng)
        assert obs.max() <= Q_MAX
        assert obs.min() >= 0


def test_corrector_uses_noisy_base_observation():
    env = make_env()
    corrector = DispatchCorrectorWrapper(k=0, mu=MU, q_max=Q_MAX, n_servers=5,
                                         base=NoiseObservation(p_noise=0.3, q_max=Q_MAX))
    rng = make_rng(0)
    true_state = env.reset()
    corrector.reset(true_state, rng)
    differs = 0
    for _ in range(500):
        action = int(rng.integers(5))
        next_true, _, info = env.step(action, rng)
        obs = corrector.step(next_true, action, info.arrived, rng)
        differs += int(not np.array_equal(obs, next_true))
    assert differs > 0  # it must NOT see the true state when the monitor is noisy


def test_belief_k0_no_noise_equals_true_state():
    env = make_env()
    w = BeliefStateWrapper(FullObservation(), k=0, p_noise=0.0, mu=MU, q_max=Q_MAX, n_servers=5)
    rng = make_rng(0)
    true_state = env.reset()
    assert np.array_equal(w.reset(true_state, rng), true_state)
    for _ in range(300):
        action = int(rng.integers(5))
        next_true, _, info = env.step(action, rng)
        obs = w.step(next_true, action, info.arrived, rng)
        assert np.array_equal(obs, next_true)


def test_belief_with_delay_is_valid_and_beats_stale_snapshot():
    env = make_env()
    k = 5
    w = BeliefStateWrapper(DelayObservation(k), k=k, p_noise=0.0, mu=MU, q_max=Q_MAX, n_servers=5)
    stale = DelayObservation(k)
    rng = make_rng(1)
    true_state = env.reset()
    w.reset(true_state, rng)
    stale.reset(true_state, rng)
    err_belief = err_stale = 0.0
    for _ in range(3000):
        action = int(rng.integers(5))
        next_true, _, info = env.step(action, rng)
        est = w.step(next_true, action, info.arrived, rng)
        old = stale.step(next_true, action, info.arrived, rng)
        assert est.min() >= 0 and est.max() <= Q_MAX
        assert np.allclose(w.b_snap.sum(axis=1), 1.0)
        err_belief += np.abs(w.mean - next_true).sum()
        err_stale += np.abs(old - next_true).sum()
    assert err_belief < err_stale


def test_hybrid_uses_q_when_confident_and_sed_when_not():
    from lb_rl.agents.hybrid import ConfidenceGatedPolicy

    class FakeAgent:
        def act(self, obs, rng):
            return 4
        def set_greedy(self, flag):
            pass

    class FakeBelief:
        mean = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
        std = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    belief = FakeBelief()
    policy = ConfidenceGatedPolicy(FakeAgent(), belief, mu=MU, tau=0.5)
    policy.set_greedy(True)
    rng = make_rng(0)
    assert policy.act(np.zeros(5, dtype=int), rng) == 4      # confident -> learned policy
    belief.std = np.array([0.0, 0.9, 0.0, 0.0, 0.0])
    # uncertain -> SED on the mean: (0+1)/0.7 is the smallest expected delay -> server 1
    assert policy.act(np.zeros(5, dtype=int), rng) == 1
