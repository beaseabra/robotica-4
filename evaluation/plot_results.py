"""
plot_results.py — Generate all comparison plots and tables from evaluation JSON files.

Usage:
  python evaluation/plot_results.py

Expects results/*.json files produced by evaluate.py.
Outputs:
  results/plots/task_success_rate.png
  results/plots/non_collision_rate.png
  results/plots/lane_deviation.png
  results/plots/avg_steps.png
  results/plots/reward_distribution.png
  results/summary_table.csv
"""

import json
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "./results"
PLOTS_DIR = "./results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

CONDITION_ORDER = ["C1", "C2", "C3"]
ALGO_COLORS = {"DQN": "#185FA5", "PPO": "#D85A30"}
CONDITION_LABELS = {
    "C1": "C1\nBaseline",
    "C2": "C2\nSensor Noise",
    "C3": "C3\nDynamic Obs.",
}


def load_results():
    """Load all result JSON files into a dict keyed by (ALGO, CONDITION)."""
    data = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(path) as f:
            r = json.load(f)
        key = (r["algo"].upper(), r["condition"].upper())
        data[key] = r
        print(f"  Loaded: {key} — success {r['success_rate_pct']}%")
    return data


def get_metric(data, algo, condition, metric):
    key = (algo.upper(), condition.upper())
    if key not in data:
        return None
    return data[key].get(metric)


def get_episode_metric(data, algo, condition, metric):
    key = (algo.upper(), condition.upper())
    if key not in data:
        return []
    return [ep.get(metric) for ep in data[key].get("episodes", []) if ep.get(metric) is not None]


# ── Plot 1: Success rate bar chart ────────────────────────────────────────────

def plot_success_rate(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.35

    for i, algo in enumerate(["DQN", "PPO"]):
        vals = []
        for c in CONDITION_ORDER:
            v = get_metric(data, algo, c, "success_rate_pct")
            vals.append(v if v is not None else 0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=algo,
                      color=ALGO_COLORS[algo], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.2,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Task success rate by condition and algorithm")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "task_success_rate.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_non_collision_rate(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.35

    for i, algo in enumerate(["DQN", "PPO"]):
        vals = []
        for c in CONDITION_ORDER:
            v = get_metric(data, algo, c, "non_collision_rate_pct")
            vals.append(v if v is not None else 0)

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=algo,
            color=ALGO_COLORS[algo],
            alpha=0.85,
            edgecolor="white",
        )

        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{v:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Non-collision rate (%)")
    ax.set_title("Non-collision rate by condition and algorithm")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "non_collision_rate.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_reward_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    values = []
    colors = []

    for i, cond in enumerate(CONDITION_ORDER):
        for j, algo in enumerate(["DQN", "PPO"]):
            vals = get_episode_metric(data, algo, cond, "total_reward")
            if vals:
                pos = i * 3 + j
                positions.append(pos)
                values.append(vals)
                colors.append(ALGO_COLORS[algo])

    if values:
        bp = ax.boxplot(values, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    group_pos = [i * 3 + 0.5 for i in range(len(CONDITION_ORDER))]
    ax.set_xticks(group_pos)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])

    legend_patches = [
        mpatches.Patch(color=ALGO_COLORS["DQN"], label="DQN"),
        mpatches.Patch(color=ALGO_COLORS["PPO"], label="PPO"),
    ]
    ax.legend(handles=legend_patches)
    ax.set_ylabel("Total reward per episode")
    ax.set_title("Reward distribution by condition and algorithm")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "reward_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 2: Lane deviation boxplot ────────────────────────────────────────────

def plot_lane_deviation(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    values = []
    colors = []
    labels = []

    for i, cond in enumerate(CONDITION_ORDER):
        for j, algo in enumerate(["DQN", "PPO"]):
            vals = get_episode_metric(data, algo, cond, "lane_deviation")
            if vals:
                pos = i * 3 + j
                positions.append(pos)
                values.append(vals)
                colors.append(ALGO_COLORS[algo])
                labels.append(f"{algo}\n{cond}")

    if values:
        bp = ax.boxplot(values, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    # x-tick at centre of each condition group
    group_pos = [i * 3 + 0.5 for i in range(len(CONDITION_ORDER))]
    ax.set_xticks(group_pos)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])

    legend_patches = [
        mpatches.Patch(color=ALGO_COLORS["DQN"], label="DQN"),
        mpatches.Patch(color=ALGO_COLORS["PPO"], label="PPO"),
    ]
    ax.legend(handles=legend_patches)
    ax.set_ylabel("Lane deviation (normalised, 0–1)")
    ax.set_title("Lane deviation distribution by condition and algorithm")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "lane_deviation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_failure_analysis(data):
    reasons = ["max_steps", "early_end", "collision"]
    reason_labels = {
        "max_steps": "Completed",
        "early_end": "Early end",
        "collision": "Collision",
    }

    labels = []
    values_by_reason = {r: [] for r in reasons}

    for cond in CONDITION_ORDER:
        for algo in ["DQN", "PPO"]:
            key = (algo, cond)
            if key not in data:
                continue

            r = data[key]
            counts = r.get("termination_counts", {})
            total = r.get("n_episodes", 1)

            labels.append(f"{algo}\n{cond}")

            for reason in reasons:
                values_by_reason[reason].append(
                    100 * counts.get(reason, 0) / total
                )

    if not labels:
        print("No termination data available for failure analysis.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for reason in reasons:
        vals = np.array(values_by_reason[reason])
        ax.bar(x, vals, bottom=bottom, label=reason_labels[reason])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Episodes (%)")
    ax.set_title("Episode termination analysis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "failure_analysis.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 3: Average steps ─────────────────────────────────────────────────────

def plot_avg_steps(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.35

    for i, algo in enumerate(["DQN", "PPO"]):
        vals = []
        for c in CONDITION_ORDER:
            v = get_metric(data, algo, c, "avg_steps")
            vals.append(v if v is not None else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=algo,
               color=ALGO_COLORS[algo], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Average steps per episode")
    ax.set_title("Episode length by condition and algorithm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "avg_steps.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Table: summary CSV ────────────────────────────────────────────────────────

def save_summary_table(data):
    import csv

    rows = []
    for algo in ["DQN", "PPO"]:
        for cond in CONDITION_ORDER:
            key = (algo, cond)
            if key not in data:
                continue
            r = data[key]
            rows.append({
                "Algorithm": algo,
                "Condition": cond,
                "Task success rate (%)": r.get("success_rate_pct", "-"),
                "Non-collision rate (%)": r.get("non_collision_rate_pct", "-"),
                "Collision rate (%)": r.get("collision_rate_pct", "-"),
                "Avg steps": r.get("avg_steps", "-"),
                "Avg lap steps": r.get("avg_lap_steps", "-"),
                "Avg reward": r.get("avg_reward", "-"),
                "Avg lane deviation": r.get("avg_lane_deviation", "-"),
                "N episodes": r.get("n_episodes", "-"),
            })

    if not rows:
        print("No data to build table — run evaluations first.")
        return

    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    # pretty-print to terminal
    col_w = 22
    header = rows[0].keys()
    print("\n" + " | ".join(h.ljust(col_w) for h in header))
    print("-" * (col_w * len(list(header))))
    for row in rows:
        print(" | ".join(str(row[h]).ljust(col_w) for h in header))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()

    if not data:
        print("No result files found in ./results/. Run evaluate.py first.")
    else:
        print(f"\nFound {len(data)} result file(s). Generating plots...\n")
        plot_success_rate(data)
        plot_non_collision_rate(data)
        plot_lane_deviation(data)
        plot_avg_steps(data)
        plot_reward_distribution(data)
        plot_failure_analysis(data)
        save_summary_table(data)
        print("\nAll plots saved to ./results/plots/")
