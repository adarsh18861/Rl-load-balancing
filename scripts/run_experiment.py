"""Run a single (config, traffic, obs_mode, seed) combination -- useful for
debugging or inspecting one cell of the grid without running the whole
thing. See scripts/run_grid.py for the full sweep.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml \\
        --experiment-config C6 --traffic steady --obs-mode delay_5 --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lb_rl.agents.value_iteration import ValueIterationPolicy, value_iteration
from lb_rl.eval.metrics import convergence_step
from lb_rl.eval.runner import rollout
from lb_rl.models.transition_model import build_transition_model
from lb_rl.utils.state_encoding import assert_state_count
from run_grid import load_config, make_env, policy_and_obs_for_config, tune_qlearning_hyperparams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment-config", required=True, help="C1-C6, C8, C11, C12, C13")
    parser.add_argument("--traffic", required=True)
    parser.add_argument("--obs-mode", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert_state_count([cfg["env"]["q_max"] + 1] * cfg["env"]["n_servers"], cfg["max_states"])
    obs_cfg = cfg["observation"][args.obs_mode]

    vi_pi_cache = {}
    if args.experiment_config == "C4":
        env_cfg = cfg["env"]
        model = build_transition_model(
            n_servers=env_cfg["n_servers"], mu=env_cfg["mu"], q_max=env_cfg["q_max"],
            lam=cfg["traffic"]["steady"]["lam"], overload_penalty=env_cfg["overload_penalty"],
            max_states=cfg["max_states"],
        )
        _, vi_policy = value_iteration(model, gamma=0.95)
        vi_pi_cache["steady"] = ValueIterationPolicy(vi_policy, model.radices)

    alpha = gamma = None
    if args.experiment_config in {"C5", "C6", "C8", "C11", "C13"}:
        alpha, gamma = tune_qlearning_hyperparams(cfg, args.traffic)
        print(f"tuned hyperparams: alpha={alpha}, gamma={gamma}")

    policy, obs_wrapper, needs_training = policy_and_obs_for_config(
        args.experiment_config, cfg, args.obs_mode, obs_cfg, args.traffic, alpha, gamma, vi_pi_cache,
    )

    if needs_training:
        _, rewards = rollout(
            make_env(cfg, args.traffic), policy, obs_wrapper, seed=args.seed,
            n_steps=cfg["qlearning"]["train_steps"], learn=True, track_rewards=True,
        )
        print(f"convergence step: {convergence_step(rewards)}")
        policy.set_greedy(True)
        _, obs_wrapper, _ = policy_and_obs_for_config(
            args.experiment_config, cfg, args.obs_mode, obs_cfg, args.traffic, alpha, gamma, vi_pi_cache,
        )
        if hasattr(policy, "bind"):
            policy.bind(obs_wrapper)
        tracker, _ = rollout(
            make_env(cfg, args.traffic), policy, obs_wrapper, seed=args.seed + 10_000,
            n_steps=cfg["qlearning"]["eval_steps"], time_bound=cfg["time_bound"],
        )
    else:
        tracker, _ = rollout(
            make_env(cfg, args.traffic), policy, obs_wrapper, seed=args.seed + 10_000,
            n_steps=cfg["qlearning"]["eval_steps"], time_bound=cfg["time_bound"],
        )

    print(tracker.summary())


if __name__ == "__main__":
    main()
