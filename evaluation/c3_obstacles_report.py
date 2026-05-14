import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("./results")
PLOTS_DIR = RESULTS_DIR / "plots" / "condition_reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALGO_COLORS = {
    "DQN": "#00BFEA",
    "PPO": "#F5A400",
}


def load_result(algo, condition):
    path = RESULTS_DIR / f"{algo.lower()}_{condition.lower()}.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_episode_values(result, metric):
    return [
        ep.get(metric)
        for ep in result.get("episodes", [])
        if ep.get(metric) is not None
    ]


def delta_pct(c1_value, c3_value):
    if c1_value is None or c3_value is None:
        return None
    return c3_value - c1_value


def setup_white_plot(fig, axes):
    fig.patch.set_facecolor("white")

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")

        for spine in ax.spines.values():
            spine.set_color("#222222")


def print_c3_cards(dqn_c1, dqn_c3, ppo_c1, ppo_c3):
    dqn_success_delta = delta_pct(
        dqn_c1.get("success_rate_pct"),
        dqn_c3.get("success_rate_pct"),
    )

    ppo_success_delta = delta_pct(
        ppo_c1.get("success_rate_pct"),
        ppo_c3.get("success_rate_pct"),
    )

    dqn_collision_delta = delta_pct(
        dqn_c1.get("collision_rate_pct"),
        dqn_c3.get("collision_rate_pct"),
    )

    ppo_collision_delta = delta_pct(
        ppo_c1.get("collision_rate_pct"),
        ppo_c3.get("collision_rate_pct"),
    )

    print("\n" + "=" * 70)
    print("C3 CRITICAL DYNAMIC OBSTACLES — C1 → C3")
    print("=" * 70)

    print("\nPowerPoint useful values:")
    print(f"PPO Success Δ C1→C3: {ppo_success_delta:+.1f}%")
    print(f"DQN Success Δ C1→C3: {dqn_success_delta:+.1f}%")
    print(f"PPO Collision Δ C1→C3: {ppo_collision_delta:+.1f}%")
    print(f"DQN Collision Δ C1→C3: {dqn_collision_delta:+.1f}%")

    print("\nC3 individual values:")
    print(f"PPO C3 Success: {ppo_c3.get('success_rate_pct')}%")
    print(f"DQN C3 Success: {dqn_c3.get('success_rate_pct')}%")
    print(f"PPO C3 Non-Collision: {ppo_c3.get('non_collision_rate_pct')}%")
    print(f"DQN C3 Non-Collision: {dqn_c3.get('non_collision_rate_pct')}%")
    print(f"PPO C3 Collision: {ppo_c3.get('collision_rate_pct')}%")
    print(f"DQN C3 Collision: {dqn_c3.get('collision_rate_pct')}%")
    print(f"PPO C3 Avg Steps: {ppo_c3.get('avg_steps')}")
    print(f"DQN C3 Avg Steps: {dqn_c3.get('avg_steps')}")
    print(f"PPO C3 Lane Deviation: {ppo_c3.get('avg_lane_deviation')}")
    print(f"DQN C3 Lane Deviation: {dqn_c3.get('avg_lane_deviation')}")


def plot_c3_success_and_failures(dqn_c1, dqn_c3, ppo_c1, ppo_c3):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    setup_white_plot(fig, axes)

    # --- Left: C1 vs C3 success rate
    ax = axes[0]

    conditions = ["C1", "C3"]
    x = np.arange(len(conditions))
    width = 0.35

    dqn_vals = [
        dqn_c1.get("success_rate_pct", 0),
        dqn_c3.get("success_rate_pct", 0),
    ]
    ppo_vals = [
        ppo_c1.get("success_rate_pct", 0),
        ppo_c3.get("success_rate_pct", 0),
    ]

    bars_dqn = ax.bar(
        x - width / 2,
        dqn_vals,
        width,
        label="DQN",
        color=ALGO_COLORS["DQN"],
        edgecolor="white",
        alpha=0.9,
    )

    bars_ppo = ax.bar(
        x + width / 2,
        ppo_vals,
        width,
        label="PPO",
        color=ALGO_COLORS["PPO"],
        edgecolor="white",
        alpha=0.9,
    )

    for bars in [bars_dqn, bars_ppo]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Task success (%)")
    ax.set_title("Task success degradation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    # --- Right: C3 termination breakdown
    ax = axes[1]

    entries = [
        ("DQN\nC3", dqn_c3),
        ("PPO\nC3", ppo_c3),
    ]

    reasons = ["max_steps", "early_end", "collision"]
    reason_labels = {
        "max_steps": "Completed",
        "early_end": "Early end",
        "collision": "Collision",
    }

    reason_colors = {
        "max_steps": "#22C55E",
        "early_end": "#F5A400",
        "collision": "#EF4444",
    }

    x = np.arange(len(entries))
    bottom = np.zeros(len(entries))

    for reason in reasons:
        vals = []

        for _, result in entries:
            counts = result.get("termination_counts", {})
            total = result.get("n_episodes", 1) or 1
            vals.append(100 * counts.get(reason, 0) / total)

        vals = np.array(vals)

        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=reason_labels[reason],
            color=reason_colors[reason],
            alpha=0.9,
            edgecolor="white",
        )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in entries])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Episodes (%)")
    ax.set_title("C3 termination breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "C3 Critical Dynamic Obstacles — Performance & Failure Modes",
        fontsize=15,
        fontweight="bold",
        color="black",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = PLOTS_DIR / "c3_success_and_failures.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved chart: {path}")


def plot_c3_reward_distribution(dqn_c3, ppo_c3):
    dqn_rewards = get_episode_values(dqn_c3, "total_reward")
    ppo_rewards = get_episode_values(ppo_c3, "total_reward")

    if not dqn_rewards:
        dqn_rewards = [0]
    if not ppo_rewards:
        ppo_rewards = [0]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    setup_white_plot(fig, ax)

    bp = ax.boxplot(
        [dqn_rewards, ppo_rewards],
        labels=["DQN", "PPO"],
        patch_artist=True,
        widths=0.55,
    )

    bp["boxes"][0].set_facecolor(ALGO_COLORS["DQN"])
    bp["boxes"][1].set_facecolor(ALGO_COLORS["PPO"])

    for patch in bp["boxes"]:
        patch.set_alpha(0.85)

    ax.set_ylabel("Total reward per episode")
    ax.set_title("C3 reward distribution under dynamic obstacles")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    path = PLOTS_DIR / "c3_reward_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved chart: {path}")


def plot_c3_lane_and_steps(dqn_c3, ppo_c3):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    setup_white_plot(fig, axes)

    # --- Lane deviation
    ax = axes[0]

    dqn_lane = get_episode_values(dqn_c3, "lane_deviation")
    ppo_lane = get_episode_values(ppo_c3, "lane_deviation")

    if not dqn_lane:
        dqn_lane = [0]
    if not ppo_lane:
        ppo_lane = [0]

    bp = ax.boxplot(
        [dqn_lane, ppo_lane],
        labels=["DQN", "PPO"],
        patch_artist=True,
        widths=0.55,
    )

    bp["boxes"][0].set_facecolor(ALGO_COLORS["DQN"])
    bp["boxes"][1].set_facecolor(ALGO_COLORS["PPO"])

    for patch in bp["boxes"]:
        patch.set_alpha(0.85)

    ax.set_ylabel("Lane deviation")
    ax.set_title("Lane stability in C3")
    ax.grid(axis="y", alpha=0.25)

    # --- Avg steps
    ax = axes[1]

    avg_steps = [
        dqn_c3.get("avg_steps", 0),
        ppo_c3.get("avg_steps", 0),
    ]

    bars = ax.bar(
        ["DQN", "PPO"],
        avg_steps,
        color=[ALGO_COLORS["DQN"], ALGO_COLORS["PPO"]],
        edgecolor="white",
        alpha=0.9,
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(avg_steps + [1]) * 0.02,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    ax.set_ylabel("Average steps")
    ax.set_title("Survival time in C3")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "C3 Driving Quality and Episode Duration",
        fontsize=15,
        fontweight="bold",
        color="black",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = PLOTS_DIR / "c3_lane_and_steps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved chart: {path}")


def main():
    dqn_c1 = load_result("dqn", "c1")
    dqn_c3 = load_result("dqn", "c3")
    ppo_c1 = load_result("ppo", "c1")
    ppo_c3 = load_result("ppo", "c3")

    print_c3_cards(dqn_c1, dqn_c3, ppo_c1, ppo_c3)

    plot_c3_success_and_failures(dqn_c1, dqn_c3, ppo_c1, ppo_c3)
    plot_c3_reward_distribution(dqn_c3, ppo_c3)
    plot_c3_lane_and_steps(dqn_c3, ppo_c3)

    print("\nDone.")
    print(f"Charts saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()