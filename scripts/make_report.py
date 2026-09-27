"""Auto-generate results/REPORT.md: tables, plot references, and a short
factual findings section derived directly from the numbers. Run
make_plots.py first so the referenced images exist.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

CONFIG_NAMES = {
    "C1": "Round Robin", "C2": "Least Connections", "C3": "Shortest Expected Delay",
    "C4": "VI/PI (model-based, full obs, steady)", "C5": "Q-learning (full obs)",
    "C6": "Q-learning (partial obs)",
    "C8": "Q-learning (partial obs + dispatch corrector)",
    "C11": "Q-learning + belief filter (extension)",
    "C12": "SED + belief filter (extension)",
    "C13": "Hybrid: Q + belief, SED fallback (extension)",
}


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + ["" if pd.isna(v) else f"{v}" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_tables(df: pd.DataFrame) -> str:
    lines = []
    for traffic in sorted(df["traffic"].unique()):
        lines.append(f"### {traffic}\n")
        sub = df[df["traffic"] == traffic]
        pivot = sub.pivot_table(index="config", columns="obs_mode", values="avg_response_time", aggfunc="mean")
        pivot = pivot.rename(index=CONFIG_NAMES).round(3)
        lines.append(df_to_markdown(pivot))
        lines.append("")
    return "\n".join(lines)


def findings(df: pd.DataFrame) -> str:
    lines = []

    full = df[df["obs_mode"] == "full"]
    if not full.empty:
        best_full = full.groupby("config")["avg_response_time"].mean().idxmin()
        lines.append(
            f"- Under full observability, **{CONFIG_NAMES.get(best_full, best_full)}** achieves the "
            f"lowest mean response time across traffic patterns."
        )

    # Compare recovery mechanisms on DELAY modes only (where they actually have work to do).
    delay = df[df["obs_mode"].str.startswith("delay_")]
    compare = ["C3", "C6", "C8", "C11", "C12", "C13"]
    for traffic in sorted(df["traffic"].unique()):
        sub = delay[delay["traffic"] == traffic]
        parts = []
        for c in compare:
            v = sub[sub["config"] == c]["avg_response_time"].mean()
            d = sub[sub["config"] == c]["drops"].mean()
            if not pd.isna(v):
                parts.append(f"{CONFIG_NAMES.get(c, c)} = {v:.3f} ({d:.1f} drops)")
        if parts:
            lines.append(f"- **{traffic}** traffic, mean over delay modes only: " + "; ".join(parts))

    conv = df[df["convergence_step"].notna()]
    if not conv.empty:
        by_config = conv.groupby("config")["convergence_step"].mean()
        lines.append(
            "- Mean training convergence step (moving-avg reward within 5% of final): "
            + ", ".join(f"{CONFIG_NAMES.get(c, c)}={v:.0f}" for c, v in by_config.items())
        )

    drops = df.groupby("config")["drops"].mean()
    worst_drops = drops.idxmax()
    lines.append(
        f"- **{CONFIG_NAMES.get(worst_drops, worst_drops)}** has the highest mean overload/drop count "
        f"({drops[worst_drops]:.2f} per run) across the grid."
    )
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "raw" / "results.csv")

    report = ["# Experiment Report\n",
              "Auto-generated from results/raw/results.csv. Findings below are computed directly "
              "from the numbers; no conclusions beyond what the data shows.\n",
              "## Mean avg response time by config x obs_mode, per traffic pattern\n",
              build_tables(df),
              "## Plots\n",
              "![response time vs delay](plots/response_time_vs_delay.png)\n",
              "![degradation](plots/degradation_bar_chart.png)\n",
              "![overload events](plots/overload_events_per_config.png)\n",
              "## Findings\n",
              findings(df)]

    out_path = results_dir / "REPORT.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
