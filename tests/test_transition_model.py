import numpy as np
import pytest

from lb_rl.agents.policy_iteration import policy_iteration
from lb_rl.agents.value_iteration import value_iteration
from lb_rl.models.transition_model import build_transition_model
from lb_rl.utils.state_encoding import decode, encode


def small_model(lam=0.5, mu=(0.6, 0.3), q_max=2, penalty=5.0):
    return build_transition_model(n_servers=2, mu=mu, q_max=q_max, lam=lam, overload_penalty=penalty)


def test_rows_sum_to_one():
    model = small_model()
    model.check_rows_sum_to_one()


def test_hand_checked_entries_two_server():
    # s = (0, 0), action 0: arrival (p=lam) goes to server 0 -> mid=(1,0);
    # no arrival (p=1-lam) -> mid=(0,0).
    lam, mu0, mu1 = 0.5, 0.6, 0.3
    model = small_model(lam=lam, mu=(mu0, mu1), q_max=2)
    radices = model.radices
    s = encode([0, 0], radices)
    succ_idx, succ_p = model.P_succ[s][0]
    probs = dict(zip(succ_idx.tolist(), succ_p.tolist()))

    # No-arrival branch: mid=(0,0) is terminal (no busy servers) -> next=(0,0) w.p. 1.
    # Arrival branch: mid=(1,0), server 0 busy -> completes w.p. mu0 -> (0,0); stays w.p. 1-mu0 -> (1,0)
    expected = {
        encode([0, 0], radices): (1 - lam) * 1.0 + lam * mu0,
        encode([1, 0], radices): lam * (1 - mu0),
    }
    for idx, p in expected.items():
        assert probs[idx] == pytest.approx(p)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_drop_case_hand_checked():
    # s = (q_max, 0), action 0: server 0 is full -> arrival is dropped, mid = s either way.
    lam, mu0, mu1 = 0.5, 0.6, 0.3
    q_max = 2
    model = small_model(lam=lam, mu=(mu0, mu1), q_max=q_max)
    radices = model.radices
    s = encode([q_max, 0], radices)
    succ_idx, succ_p = model.P_succ[s][0]
    probs = dict(zip(succ_idx.tolist(), succ_p.tolist()))
    # mid = (2, 0) regardless of arrival -> server 0 busy, completes w.p. mu0
    expected = {
        encode([q_max - 1, 0], radices): mu0,
        encode([q_max, 0], radices): 1 - mu0,
    }
    for idx, p in expected.items():
        assert probs[idx] == pytest.approx(p)
    # expected reward includes the overload penalty weighted by P(drop) = lam
    expected_next_sum = mu0 * (q_max - 1) + (1 - mu0) * q_max
    expected_R = -expected_next_sum - model.overload_penalty * lam
    assert model.R[s, 0] == pytest.approx(expected_R)


def test_vi_pi_agree_on_small_case():
    model = small_model()
    _, vi_policy = value_iteration(model, gamma=0.95)
    _, pi_policy = policy_iteration(model, gamma=0.95)
    assert np.array_equal(vi_policy, pi_policy)


def test_vi_pi_values_close():
    model = small_model()
    vi_V, _ = value_iteration(model, gamma=0.95)
    pi_V, _ = policy_iteration(model, gamma=0.95)
    assert np.allclose(vi_V, pi_V, atol=1e-4)
