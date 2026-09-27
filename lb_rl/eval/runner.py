"""Rollout helper: drives an environment + observation wrapper + policy for
a fixed number of slots, tracking metrics.

Environment dynamics, observation-wrapper noise, and policy tie-breaking
each get their own independent RNG stream (spawned from one seed) so that
a policy which ignores its observation (Round Robin) is bit-for-bit
reproducible regardless of the observation mode in use.
"""

from typing import Optional

import numpy as np

from lb_rl.eval.metrics import MetricsTracker


def make_streams(seed: int):
    """Spawn three independent generators (env, obs, policy) from one seed."""
    ss = np.random.SeedSequence(seed)
    env_seed, obs_seed, policy_seed = ss.spawn(3)
    return (
        np.random.default_rng(env_seed),
        np.random.default_rng(obs_seed),
        np.random.default_rng(policy_seed),
    )


def rollout(
    env,
    policy,
    obs_wrapper,
    seed: int,
    n_steps: int,
    learn: bool = False,
    time_bound: Optional[int] = None,
    track_rewards: bool = False,
):
    """Run one rollout. If `learn`, calls policy.update(...) each step
    (Q-learning); otherwise the policy is treated as frozen. Returns
    (MetricsTracker, rewards_or_None)."""
    env_rng, obs_rng, policy_rng = make_streams(seed)
    tracker = MetricsTracker(time_bound=time_bound)
    rewards = [] if track_rewards else None

    true_state = env.reset()
    obs = obs_wrapper.reset(true_state, obs_rng)
    for _ in range(n_steps):
        action = policy.act(obs, policy_rng)
        next_true_state, reward, info = env.step(action, env_rng)
        next_obs = obs_wrapper.step(next_true_state, action, info.arrived, obs_rng)
        if learn:
            policy.update(obs, action, reward, next_obs)
        tracker.record(info, next_true_state)
        if track_rewards:
            rewards.append(reward)
        obs = next_obs

    return tracker, rewards
