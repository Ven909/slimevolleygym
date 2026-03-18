"""
analyze_training.py
====================
Compare FrameStack, LSTM, and Exploit Agent training progress at
2M, 5M, and 10M step milestones.

Reads TensorBoard event files (.tfevents) directly — no manual CSV needed.

Supports multi-run agents (e.g. Exploit Agent whose training was split across
two runs): each run entry can carry a step_offset so that its logged steps
are shifted before merging with earlier runs into one continuous series.

Usage
-----
    python analyze_training.py

Outputs
-------
    milestone_summary.csv       — summary table
    delta_analysis.csv          — % improvement table
    bar_chart_milestones.png    — grouped bar chart
    learning_curves.png         — full learning curves with milestone markers
    delta_analysis.png          — % improvement horizontal bar chart
"""

import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# Each agent lists one or more run segments:
#   tfevents_glob : glob relative to BASE_DIR
#   step_offset   : add this to every logged step (for continuation runs whose
#                   internal counter restarted at 0)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

AGENT_CONFIGS = {
    "FrameStack": {
        "color": "#4C72B0",
        "runs": [
            {
                "tfevents_glob": "logs_framestack/run_20260223_202559/**/*.tfevents.*",
                "step_offset": 0,
            },
        ],
    },
    "LSTM": {
        "color": "#DD8452",
        "runs": [
            {
                "tfevents_glob": "logs_lstm/run_20260223_144038/**/*.tfevents.*",
                "step_offset": 0,
            },
        ],
    },
    "Exploit Agent": {
        "color": "#55A868",
        "runs": [
            # Original 0–5M run
            {
                "tfevents_glob": "logs_exploiter/run_20260303_204533/**/*.tfevents.*",
                "step_offset": 0,
            },
            # Continuation 5M–10M (SB3 restarted its counter from 0)
            {
                "tfevents_glob": "logs_exploiter/run_20260318_003219/**/*.tfevents.*",
                "step_offset": 5_000_000,
            },
        ],
    },
}

MILESTONES = [2_000_000, 5_000_000, 10_000_000]
MILESTONE_LABELS = ["2M", "5M", "10M"]

REWARD_TAG = "rollout/ep_rew_mean"
ALT_TAGS = ["train/ep_rew_mean", "eval/mean_reward", "rollout/mean_reward"]
# Accept the closest logged point within this many steps of a milestone
TOLERANCE = 400_000


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def load_tfevents_single(pattern: str, step_offset: int = 0) -> pd.DataFrame:
    """
    Parse all matching TensorBoard event files.
    Returns DataFrame [step, tag, value] with step_offset already applied.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
            STORE_EVERYTHING_SIZE_GUIDANCE,
        )
    except ImportError:
        raise ImportError("tensorboard is required: pip install tensorboard")

    files = sorted(glob.glob(str(BASE_DIR / pattern), recursive=True))
    if not files:
        return pd.DataFrame(columns=["step", "tag", "value"])

    rows = []
    for fpath in files:
        ea = EventAccumulator(fpath, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tag):
                rows.append({
                    "step": event.step + step_offset,
                    "tag": tag,
                    "value": event.value,
                })

    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def load_agent_series(runs: list) -> pd.DataFrame:
    """
    Load and merge all run segments for one agent into a single
    (step, mean_reward) DataFrame sorted by step.
    """
    frames = []
    for run in runs:
        df = load_tfevents_single(run["tfevents_glob"], run["step_offset"])
        if df.empty:
            continue

        # Pick the reward tag
        tag_used = None
        for candidate in [REWARD_TAG] + ALT_TAGS:
            if (df["tag"] == candidate).any():
                tag_used = candidate
                break

        if tag_used is None:
            continue

        sub = df[df["tag"] == tag_used][["step", "value"]].copy()
        sub.columns = ["step", "mean_reward"]
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["step", "mean_reward"])

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("step")
        .sort_values("step")
        .reset_index(drop=True)
    )
    return merged


def find_closest_value(series: pd.DataFrame, target_step: int):
    """
    Return (actual_step, mean_reward) for the point closest to target_step.
    Returns (None, None) if no point is within TOLERANCE.
    """
    if series.empty:
        return None, None
    diffs = (series["step"] - target_step).abs()
    idx = diffs.idxmin()
    if diffs[idx] > TOLERANCE:
        return None, None
    return int(series.loc[idx, "step"]), float(series.loc[idx, "mean_reward"])


def build_milestone_table(agent_series: dict) -> pd.DataFrame:
    rows = []
    for agent_name, series in agent_series.items():
        for milestone, label in zip(MILESTONES, MILESTONE_LABELS):
            actual_step, reward = find_closest_value(series, milestone)
            rows.append({
                "Agent": agent_name,
                "Training Steps": label,
                "Steps (exact)": actual_step,
                "Mean Reward": round(reward, 4) if reward is not None else np.nan,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DELTA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """
    % change = (new - old) / |old| * 100
    Uses abs(old) so negative-reward improvements show as positive %.
    """
    rows = []
    for agent in summary["Agent"].unique():
        sub = summary[summary["Agent"] == agent].set_index("Training Steps")
        for a_label, b_label in [("2M", "5M"), ("5M", "10M")]:
            r_a = sub.loc[a_label, "Mean Reward"] if a_label in sub.index else np.nan
            r_b = sub.loc[b_label, "Mean Reward"] if b_label in sub.index else np.nan
            if pd.notna(r_a) and pd.notna(r_b) and r_a != 0:
                delta = (r_b - r_a) / abs(r_a) * 100
            else:
                delta = np.nan
            rows.append({
                "Agent": agent,
                "Transition": f"{a_label}\u2192{b_label}",
                "Reward at Start": round(r_a, 4) if pd.notna(r_a) else np.nan,
                "Reward at End": round(r_b, 4) if pd.notna(r_b) else np.nan,
                "% Change": round(delta, 1) if pd.notna(delta) else np.nan,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_chart(summary: pd.DataFrame, out_path: str):
    """Grouped bar chart: agents x milestones, height = mean reward."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_order = list(AGENT_CONFIGS.keys())
    palette = [AGENT_CONFIGS[a]["color"] for a in agent_order]

    sns.barplot(
        data=summary.dropna(subset=["Mean Reward"]),
        x="Training Steps",
        y="Mean Reward",
        hue="Agent",
        hue_order=agent_order,
        palette=palette,
        order=MILESTONE_LABELS,
        ax=ax,
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate bar values
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title("Mean Episode Reward at Training Milestones", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.legend(title="Agent", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_learning_curves(agent_series: dict, summary: pd.DataFrame, out_path: str):
    """Full learning curves with vertical milestone lines and scatter dots."""
    fig, ax = plt.subplots(figsize=(13, 7))

    for agent_name, series in agent_series.items():
        if series.empty:
            continue
        color = AGENT_CONFIGS[agent_name]["color"]

        # Rolling average for readability
        smoothed = (
            series.set_index("step")["mean_reward"]
            .rolling(window=15, min_periods=1, center=True)
            .mean()
        )
        ax.plot(
            smoothed.index / 1e6,
            smoothed.values,
            color=color,
            linewidth=2,
            label=agent_name,
            alpha=0.9,
        )

        # Milestone scatter dots
        agent_rows = summary[summary["Agent"] == agent_name]
        for _, row in agent_rows.iterrows():
            if pd.notna(row["Mean Reward"]) and row["Steps (exact)"] is not None:
                ax.scatter(
                    row["Steps (exact)"] / 1e6,
                    row["Mean Reward"],
                    color=color,
                    s=90,
                    zorder=5,
                    edgecolors="white",
                    linewidths=1.5,
                )
                ax.annotate(
                    f"{row['Training Steps']}\n{row['Mean Reward']:.2f}",
                    xy=(row["Steps (exact)"] / 1e6, row["Mean Reward"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=color,
                )

    # Milestone vertical guide lines
    for milestone, label in zip(MILESTONES, MILESTONE_LABELS):
        ax.axvline(milestone / 1e6, color="grey", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(
            milestone / 1e6 + 0.05,
            ax.get_ylim()[0],
            label,
            fontsize=9,
            color="grey",
            va="bottom",
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.3)
    ax.set_title("Learning Curves with Milestone Markers (0 to 10M Steps)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Steps (Millions)", fontsize=12)
    ax.set_ylabel("Mean Episode Reward (rolling avg)", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.legend(title="Agent", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_delta_analysis(deltas: pd.DataFrame, out_path: str):
    """Horizontal bar chart of % improvement per agent per transition."""
    valid = deltas.dropna(subset=["% Change"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    transitions = ["2M\u21925M", "5M\u219210M"]
    titles = ["2M to 5M Improvement", "5M to 10M Improvement"]

    for ax, transition, title in zip(axes, transitions, titles):
        subset = valid[valid["Transition"] == transition].copy()
        colors = [AGENT_CONFIGS[a]["color"] for a in subset["Agent"]]
        bars = ax.barh(subset["Agent"], subset["% Change"], color=colors, edgecolor="white", height=0.5)

        for bar, val in zip(bars, subset["% Change"]):
            ax.text(
                bar.get_width() + (abs(bar.get_width()) * 0.02 + 0.5),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=10,
                fontweight="bold",
            )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("% Change in Mean Reward", fontsize=10)
        ax.tick_params(axis="y", labelsize=11)
        sns.despine(ax=ax)

    fig.suptitle(
        "% Change in Mean Reward Between Milestones\n(positive = reward improved, negative = reward declined)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RL Training Milestone Analysis  (0 to 10M steps)")
    print("=" * 60)

    # Load all agent data
    agent_series = {}
    for agent_name, cfg in AGENT_CONFIGS.items():
        print(f"\n[{agent_name}] Loading TensorBoard logs ...")
        try:
            series = load_agent_series(cfg["runs"])
            agent_series[agent_name] = series
            if series.empty:
                print(f"  WARNING: No reward data found.")
            else:
                print(
                    f"  {len(series):,} data points | "
                    f"steps {series['step'].min():,} -> {series['step'].max():,}"
                )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            agent_series[agent_name] = pd.DataFrame(columns=["step", "mean_reward"])

    # Summary table
    print("\n-- Milestone Summary Table ----------------------------------")
    summary = build_milestone_table(agent_series)
    print(summary[["Agent", "Training Steps", "Mean Reward"]].to_string(index=False))
    summary.to_csv(BASE_DIR / "milestone_summary.csv", index=False)
    print(f"\n  Saved: milestone_summary.csv")

    # Delta analysis
    print("\n-- Delta Analysis (% Change Between Milestones) -------------")
    deltas = compute_deltas(summary)
    print(deltas.to_string(index=False))
    deltas.to_csv(BASE_DIR / "delta_analysis.csv", index=False)
    print(f"\n  Saved: delta_analysis.csv")

    # Plots
    print("\n-- Generating Plots -----------------------------------------")
    plot_bar_chart(summary, str(BASE_DIR / "bar_chart_milestones.png"))
    plot_learning_curves(agent_series, summary, str(BASE_DIR / "learning_curves.png"))
    plot_delta_analysis(deltas, str(BASE_DIR / "delta_analysis.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
