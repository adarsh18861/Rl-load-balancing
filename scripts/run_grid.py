"""Full experiment grid runner.

Usage:
    python scripts/run_grid.py --config configs/quick.yaml
    python scripts/run_grid.py --config configs/full_grid.yaml

Writes results/raw/results.csv (one row per config/traffic/obs_mode/seed)
and results/summary.csv (aggregated mean/std). Prints an estimated runtime
before starting the run.
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from lb_rl.agents.baselines import LeastConnections, RoundRobin, ShortestExpectedDelay
from lb_rl.agents.policy_iteration import PolicyIterationPolicy, policy_iteration
from lb_rl.agents.q_learning import QLearningAgent
from lb_rl.agents.hybrid import ConfidenceGatedPolicy
from lb_rl.agents.value_iteration import ValueIterationPolicy, value_iteration
from lb_rl.env.environment import LoadBalancingEnv
from lb_rl.env.traffic import make_traffic
from lb_rl.eval.metrics import convergence_step
from lb_rl.eval.runner import rollout
from lb_rl.models.transition_model import build_transition_model
from lb_rl.observation.wrappers import (
    BeliefStateWrapper,
    DelayObservation,
    DispatchCorrectorWrapper,
    FullObservation,
    NoiseDelayObservation,
    NoiseObservation,
)
from lb_rl.utils.state_encoding import assert_state_count

QLEARNING_CONFIGS = {"C5", "C6", "C8", "C11", "C13"}
NON_FULL_ONLY = {"C6", "C8", "C11", "C12", "C13"}
FULL_ONLY = {"C5"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def base_obs_wrapper(obs_cfg: dict, q_max: int):
    kind = obs_cfg["kind"]
    if kind == "full":
        return FullObservation()
    if kind == "noise":
        return NoiseObservation(p_noise=obs_cfg["p_noise"], q_max=q_max)
    if kind == "delay":
        return DelayObservation(k=obs_cfg["k"])
    if kind == "noise_delay":
        return NoiseDelayObservation(k=obs_cfg["k"], p_noise=obs_cfg["p_noise"], q_max=q_max)
    raise ValueError(f"unknown observation kind: {kind}")


def applicable(config_id: str, obs_name: str, traffic_name: str) -> bool:
    if config_id == "C4":
        return obs_name == "full" and traffic_name == "steady"
    if config_id in FULL_ONLY:
        return obs_name == "full"
    if config_id in NON_FULL_ONLY:
        return obs_name != "full"
    return True  # C1, C2, C3: every obs mode


def make_env(cfg: dict, traffic_name: str) -> LoadBalancingEnv:
    env_cfg = cfg["env"]
    traffic = make_traffic(traffic_name, **cfg["traffic"][traffic_name])
    # "mu" = what the dispatcher believes; "mu_true" (optional) = the real server speeds
    true_mu = env_cfg.get("mu_true", env_cfg["mu"])
    return LoadBalancingEnv(
        n_servers=env_cfg["n_servers"], mu=true_mu, q_max=env_cfg["q_max"],
        traffic=traffic, overload_penalty=env_cfg["overload_penalty"],
    )


def tune_qlearning_hyperparams(cfg: dict, traffic_name: str) -> tuple:
    """Grid search alpha/gamma using full-obs Q-learning on tuning seeds
    only; reused for every partial-obs Q-learning config under this
    traffic pattern (kept tractable given from-scratch tabular training
    cost -- see README)."""
    env_cfg = cfg["env"]
    ql_cfg = cfg["qlearning"]
    radices = [env_cfg["q_max"] + 1] * env_cfg["n_servers"]
    best = None
    for alpha, gamma in itertools.product(ql_cfg["alpha_grid"], ql_cfg["gamma_grid"]):
        scores = []
        for seed in ql_cfg["tuning_seeds"]:
            agent = QLearningAgent(
                radices=radices, n_actions=env_cfg["n_servers"], alpha=alpha, gamma=gamma,
                epsilon_start=ql_cfg["epsilon_start"], epsilon_end=ql_cfg["epsilon_end"],
                epsilon_decay_steps=max(1, ql_cfg["train_steps"] // 2),
            )
            rollout(make_env(cfg, traffic_name), agent, FullObservation(), seed=seed,
                    n_steps=ql_cfg["train_steps"], learn=True)
            agent.set_greedy(True)
            tracker, _ = rollout(make_env(cfg, traffic_name), agent, FullObservation(),
                                  seed=seed + 500, n_steps=ql_cfg["eval_steps"], time_bound=cfg["time_bound"])
            s = tracker.summary()
            scores.append(-s["avg_response_time"] if not np.isnan(s["avg_response_time"]) else -1e9)
        mean_score = float(np.mean(scores))
        if best is None or mean_score > best[0]:
            best = (mean_score, alpha, gamma)
    return best[1], best[2]


def policy_and_obs_for_config(
    config_id: str, cfg: dict, obs_name: str, obs_cfg: dict, traffic_name: str,
    alpha: float, gamma: float, vi_pi_cache: dict,
):
    """Returns (policy, obs_wrapper, needs_training)."""
    env_cfg = cfg["env"]
    mu, q_max, n = env_cfg["mu"], env_cfg["q_max"], env_cfg["n_servers"]
    radices = [q_max + 1] * n
    k = obs_cfg.get("k", 0)

    if config_id == "C1":
        return RoundRobin(n_servers=n), base_obs_wrapper(obs_cfg, q_max), False
    if config_id == "C2":
        return LeastConnections(), base_obs_wrapper(obs_cfg, q_max), False
    if config_id == "C3":
        return ShortestExpectedDelay(mu=mu), base_obs_wrapper(obs_cfg, q_max), False
    if config_id == "C4":
        policy = vi_pi_cache[traffic_name]
        return policy, FullObservation(), False
    if config_id == "C5":
        return (
            QLearningAgent(radices=radices, n_actions=n, alpha=alpha, gamma=gamma,
                            epsilon_start=cfg["qlearning"]["epsilon_start"],
                            epsilon_end=cfg["qlearning"]["epsilon_end"],
                            epsilon_decay_steps=max(1, cfg["qlearning"]["train_steps"] // 2)),
            FullObservation(), True,
        )
    if config_id == "C6":
        return (
            QLearningAgent(radices=radices, n_actions=n, alpha=alpha, gamma=gamma,
                            epsilon_start=cfg["qlearning"]["epsilon_start"],
                            epsilon_end=cfg["qlearning"]["epsilon_end"],
                            epsilon_decay_steps=max(1, cfg["qlearning"]["train_steps"] // 2)),
            base_obs_wrapper(obs_cfg, q_max), True,
        )
    if config_id == "C8":
        corrector = DispatchCorrectorWrapper(k=k, mu=mu, q_max=q_max, n_servers=n,
                                             base=base_obs_wrapper(obs_cfg, q_max))
        return (
            QLearningAgent(radices=radices, n_actions=n, alpha=alpha, gamma=gamma,
                            epsilon_start=cfg["qlearning"]["epsilon_start"],
                            epsilon_end=cfg["qlearning"]["epsilon_end"],
                            epsilon_decay_steps=max(1, cfg["qlearning"]["train_steps"] // 2)),
            corrector, True,
        )
    if config_id in ("C11", "C12", "C13"):
        belief = BeliefStateWrapper(
            base_obs_wrapper(obs_cfg, q_max), k=k, p_noise=obs_cfg.get("p_noise", 0.0),
            mu=mu, q_max=q_max, n_servers=n,
        )
        if config_id == "C12":
            return ShortestExpectedDelay(mu=mu), belief, False
        agent = QLearningAgent(radices=radices, n_actions=n, alpha=alpha, gamma=gamma,
                               epsilon_start=cfg["qlearning"]["epsilon_start"],
                               epsilon_end=cfg["qlearning"]["epsilon_end"],
                               epsilon_decay_steps=max(1, cfg["qlearning"]["train_steps"] // 2))
        if config_id == "C11":
            return agent, belief, True
        tau = cfg.get("hybrid", {}).get("tau", 0.5)
        return ConfidenceGatedPolicy(agent, belief, mu=mu, tau=tau), belief, True
    raise ValueError(config_id)


def build_run_list(cfg: dict):
    """Enumerate every (config, traffic, obs_name) combo that applies,
    independent of seed."""
    runs = []
    for traffic_name in cfg["traffic"]:
        for obs_name in cfg["observation"]:
            for config_id in cfg["configs"]:
                if applicable(config_id, obs_name, traffic_name):
                    runs.append((config_id, traffic_name, obs_name))
    return runs


def estimate_runtime(cfg: dict, runs: list) -> float:
    """Calibrate steps/sec for a baseline rollout and a Q-learning training
    rollout, then extrapolate total wall time across the whole grid."""
    n_seeds = len(cfg["seeds"])
    eval_steps = cfg["qlearning"]["eval_steps"]
    train_steps = cfg["qlearning"]["train_steps"]

    calib_steps = 2000
    t0 = time.time()
    rollout(make_env(cfg, "steady"), LeastConnections(), FullObservation(), seed=0, n_steps=calib_steps)
    baseline_sps = calib_steps / max(time.time() - t0, 1e-6)

    agent = QLearningAgent(radices=[cfg["env"]["q_max"] + 1] * cfg["env"]["n_servers"],
                            n_actions=cfg["env"]["n_servers"], alpha=0.1, gamma=0.95)
    t0 = time.time()
    rollout(make_env(cfg, "steady"), agent, FullObservation(), seed=0, n_steps=calib_steps, learn=True)
    ql_sps = calib_steps / max(time.time() - t0, 1e-6)

    total_seconds = 0.0
    for config_id, _, _ in runs:
        per_seed_steps = (train_steps + eval_steps) if config_id in QLEARNING_CONFIGS else eval_steps
        sps = ql_sps if config_id in QLEARNING_CONFIGS else baseline_sps
        total_seconds += n_seeds * per_seed_steps / sps

    n_tuning_combos = len(cfg["qlearning"]["alpha_grid"]) * len(cfg["qlearning"]["gamma_grid"])
    n_tuning_seeds = len(cfg["qlearning"]["tuning_seeds"])
    tuning_seconds = (
        len(cfg["traffic"]) * n_tuning_combos * n_tuning_seeds
        * (train_steps + eval_steps) / ql_sps
    )
    return total_seconds + tuning_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--estimate-only", action="store_true",
                         help="Print the runtime estimate and exit without running the grid.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert_state_count([cfg["env"]["q_max"] + 1] * cfg["env"]["n_servers"], cfg["max_states"])

    runs = build_run_list(cfg)
    est_seconds = estimate_runtime(cfg, runs)
    print(f"Grid: {len(runs)} (config, traffic, obs_mode) combos x {len(cfg['seeds'])} seeds")
    print(f"Estimated runtime: {est_seconds/60:.1f} minutes ({est_seconds/3600:.2f} hours)")
    if args.estimate_only:
        return

    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    # Precompute VI/PI once per traffic pattern where applicable (steady only).
    vi_pi_cache = {}
    if "C4" in cfg["configs"] and any(t == "steady" for t in cfg["traffic"]):
        env_cfg = cfg["env"]
        model = build_transition_model(
            n_servers=env_cfg["n_servers"], mu=env_cfg["mu"], q_max=env_cfg["q_max"],
            lam=cfg["traffic"]["steady"]["lam"], overload_penalty=env_cfg["overload_penalty"],
            max_states=cfg["max_states"],
        )
        _, vi_policy = value_iteration(model, gamma=0.95)
        vi_pi_cache["steady"] = ValueIterationPolicy(vi_policy, model.radices)

    # Hyperparameter tuning once per traffic pattern that has any Q-learning config.
    hyperparams = {}
    if QLEARNING_CONFIGS & set(cfg["configs"]):
        for traffic_name in cfg["traffic"]:
            alpha, gamma = tune_qlearning_hyperparams(cfg, traffic_name)
            hyperparams[traffic_name] = (alpha, gamma)
            print(f"tuned hyperparams for {traffic_name}: alpha={alpha}, gamma={gamma}")

    rows = []
    t_start = time.time()
    for i, (config_id, traffic_name, obs_name) in enumerate(runs):
        obs_cfg = cfg["observation"][obs_name]
        alpha, gamma = hyperparams.get(traffic_name, (None, None))
        for seed in cfg["seeds"]:
            policy, obs_wrapper, needs_training = policy_and_obs_for_config(
                config_id, cfg, obs_name, obs_cfg, traffic_name, alpha, gamma, vi_pi_cache,
            )
            conv_step = None
            if needs_training:
                _, rewards = rollout(
                    make_env(cfg, traffic_name), policy, obs_wrapper, seed=seed,
                    n_steps=cfg["qlearning"]["train_steps"], learn=True, track_rewards=True,
                )
                conv_step = convergence_step(rewards)
                policy.set_greedy(True)
                # Fresh wrapper instance for evaluation (wrappers carry history state).
                _, eval_obs_wrapper, _ = policy_and_obs_for_config(
                    config_id, cfg, obs_name, obs_cfg, traffic_name, alpha, gamma, vi_pi_cache,
                )
                if hasattr(policy, "bind"):
                    policy.bind(eval_obs_wrapper)
                tracker, _ = rollout(
                    make_env(cfg, traffic_name), policy, eval_obs_wrapper, seed=seed + 10_000,
                    n_steps=cfg["qlearning"]["eval_steps"], time_bound=cfg["time_bound"],
                )
            else:
                tracker, _ = rollout(
                    make_env(cfg, traffic_name), policy, obs_wrapper, seed=seed + 10_000,
                    n_steps=cfg["qlearning"]["eval_steps"], time_bound=cfg["time_bound"],
                )
            summary = tracker.summary()
            rows.append({
                "config": config_id, "traffic": traffic_name, "obs_mode": obs_name, "seed": seed,
                "alpha": alpha, "gamma": gamma, "convergence_step": conv_step, **summary,
            })
        elapsed = time.time() - t_start
        print(f"[{i+1}/{len(runs)}] {config_id} {traffic_name} {obs_name} done ({elapsed:.0f}s elapsed)")

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "raw" / "results.csv", index=False)

    metric_cols = ["avg_response_time", "drops", "drop_rate", "load_balance_variance",
                   "completion_rate_within_bound", "convergence_step"]
    summary_df = (
        raw_df.groupby(["config", "traffic", "obs_mode"])[metric_cols]
        .agg(["mean", "std"])
    )
    summary_df.columns = ["_".join(c) for c in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print(f"wrote {len(raw_df)} rows to {out_dir/'raw'/'results.csv'}")
    print(f"wrote summary to {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
