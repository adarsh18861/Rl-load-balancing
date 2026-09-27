"""Generate the required plots from results/raw/results.csv:
response time vs. delay per config, degradation bar chart, overload
events per config. (Learning curves are produced by scripts/m3_qlearning.py
and scripts/m4_partial_obs.py, which log per-step rewards not captured in
the grid's summary CSV.)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASELINE_CONFIG = {"C2": "C2", "C3": "C3", "C6": "C5", "C8": "C5",
                   "C11": "C5", "C12": "C3", "C13": "C5"}


def response_time_vs_delay(df: pd.DataFrame, out_dir: Path):
    delay_rows = df[df["obs_mode"].str.startswith("delay_")].copy()
    if delay_rows.empty:
        return
    delay_rows["k"] = delay_rows["obs_mode"].str.replace("delay_", "").astype(int)
    fig, ax = plt.subplots(figsize=(7, 5))
    for config, group in delay_rows.groupby("config"):
        agg = group.groupby("k")["avg_response_time"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["k"], agg["mean"], yerr=agg["std"], marker="o", label=config, capsize=3)
    ax.set_xlabel("delay k (slots)")
    ax.set_ylabel("avg response time")
    ax.set_title("Response time vs. observation delay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "response_time_vs_delay.png", dpi=150)
    plt.close(fig)


def degradation_bar_chart(df: pd.DataFrame, out_dir: Path):
    full = df[df["obs_mode"] == "full"].groupby(["config", "traffic"])["avg_response_time"].mean()
    rows = []
    for (config, traffic, obs_mode), group in df.groupby(["config", "traffic", "obs_mode"]):
        if obs_mode == "full":
            continue
        base_config = BASELINE_CONFIG.get(config)
        if base_config is None or (base_config, traffic) not in full.index:
            continue
        degradation = group["avg_response_time"].mean() - full[(base_config, traffic)]
        rows.append({"config": config, "traffic": traffic, "obs_mode": obs_mode, "degradation": degradation})
    if not rows:
        return
    deg_df = pd.DataFrame(rows)
    pivot = deg_df.groupby("config")["degradation"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(pivot.index, pivot.values)
    ax.set_xlabel("avg response time degradation vs. full obs (partial-obs mean - full-obs baseline)")
    ax.set_title("Degradation under partial observability")
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_bar_chart.png", dpi=150)
    plt.close(fig)
    deg_df.to_csv(out_dir.parent / "degradation.csv", index=False)


def overload_events_per_config(df: pd.DataFrame, out_dir: Path):
    agg = df.groupby("config")["drops"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(agg.index, agg.values)
    ax.set_xlabel("mean overload/drop events per run")
    ax.set_title("Overload events per config")
    fig.tight_layout()
    fig.savefig(out_dir / "overload_events_per_config.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "raw" / "results.csv")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    response_time_vs_delay(df, plots_dir)
    degradation_bar_chart(df, plots_dir)
    overload_events_per_config(df, plots_dir)
    print(f"wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
