import numpy as np

from lb_rl.agents.baselines import LeastConnections, RoundRobin, ShortestExpectedDelay
from lb_rl.utils.seeding import make_rng


def test_least_connections_picks_min():
    lc = LeastConnections()
    rng = make_rng(0)
    obs = np.array([3, 1, 4, 1, 5])
    action = lc.act(obs, rng)
    assert obs[action] == 1


def test_least_connections_tie_break_uses_rng():
    lc = LeastConnections()
    obs = np.array([0, 0, 5, 5, 5])
    seen = set()
    for seed in range(20):
        rng = make_rng(seed)
        seen.add(lc.act(obs, rng))
    assert seen == {0, 1}


def test_sed_picks_min_expected_delay():
    sed = ShortestExpectedDelay(mu=[0.9, 0.1])
    rng = make_rng(0)
    obs = np.array([2, 2])
    # server 0: (2+1)/0.9 = 3.33, server 1: (2+1)/0.1 = 30
    assert sed.act(obs, rng) == 0


def test_round_robin_cycles():
    rr = RoundRobin(n_servers=3)
    rng = make_rng(0)
    obs = np.zeros(3)
    actions = [rr.act(obs, rng) for _ in range(7)]
    assert actions == [0, 1, 2, 0, 1, 2, 0]
