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


def delta_pct(c1_value, c2_value):
    if c1_value is None or c2_value is None:
        return None
    return c2_value - c1_value


def print_c2_cards(dqn_c1, dqn_c2, ppo_c1, ppo_c2):
    dqn_success_delta = delta_pct(
        dqn_c1.get("success_rate_pct"),
        dqn_c2.get("success_rate_pct"),
    )

    ppo_success_delta = delta_pct(
        ppo_c1.get("success_rate_pct"),
        ppo_c2.get("success_rate_pct"),
    )

    dqn_non_collision_delta = delta_pct(
        dqn_c1.get("non_collision_rate_pct"),
        dqn_c2.get("non_collision_rate_pct"),
    )

    ppo_non_collision_delta = delta_pct(
        ppo_c1.get("non_collision_rate_pct"),
        ppo_c2.get("non_collision_rate_pct"),
    )

    print("\n" + "=" * 70)
    print("C2 SENSOR NOISE IMPACT — C1 → C2")
    print("=" * 70)

    print("\nPowerPoint top cards:")
    print(f"PPO Success Δ: {ppo_success_delta:+.1f}%")
    print(f"DQN Success Δ: {dqn_success_delta:+.1f}%")
    print(f"PPO Non-Collision Δ: {ppo_non_collision_delta:+.1f}%")
    print(f"DQN Non-Collision Δ: {dqn_non_collision_delta:+.1f}%")

    print("\nDetailed values:")
    print(f"PPO C1 Success: {ppo_c1.get('success_rate_pct')}%")
    print(f"PPO C2 Success: {ppo_c2.get('success_rate_pct')}%")
    print(f"DQN C1 Success: {dqn_c1.get('success_rate_pct')}%")
    print(f"DQN C2 Success: {dqn_c2.get('success_rate_pct')}%")

    print(f"\nPPO C1 Lane Deviation: {ppo_c1.get('avg_lane_deviation')}")
    print(f"PPO C2 Lane Deviation: {ppo_c2.get('avg_lane_deviation')}")
    print(f"DQN C1 Lane Deviation: {dqn_c1.get('avg_lane_deviation')}")
    print(f"DQN C2 Lane Deviation: {dqn_c2.get('avg_lane_deviation')}")


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


def plot_c2_comparison(dqn_c1, dqn_c2, ppo_c1, ppo_c2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    setup_white_plot(fig, axes)

    # --- 1) Task success C1 vs C2
    ax = axes[0]
    conditions = ["C1", "C2"]
    x = np.arange(len(conditions))
    width = 0.35

    dqn_vals = [
        dqn_c1.get("success_rate_pct", 0),
        dqn_c2.get("success_rate_pct", 0),
    ]
    ppo_vals = [
        ppo_c1.get("success_rate_pct", 0),
        ppo_c2.get("success_rate_pct", 0),
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

    # --- 2) Lane deviation boxplot C1 vs C2
    ax = axes[1]

    lane_data = [
        get_episode_values(dqn_c1, "lane_deviation"),
        get_episode_values(dqn_c2, "lane_deviation"),
        get_episode_values(ppo_c1, "lane_deviation"),
        get_episode_values(ppo_c2, "lane_deviation"),
    ]

    # Avoid empty boxplot errors
    lane_data = [vals if vals else [0] for vals in lane_data]

    labels = ["DQN\nC1", "DQN\nC2", "PPO\nC1", "PPO\nC2"]
    colors = [
        ALGO_COLORS["DQN"],
        ALGO_COLORS["DQN"],
        ALGO_COLORS["PPO"],
        ALGO_COLORS["PPO"],
    ]

    bp = ax.boxplot(
        lane_data,
        labels=labels,
        patch_artist=True,
        widths=0.55,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Lane deviation")
    ax.set_title("Lane stability under noise")
    ax.grid(axis="y", alpha=0.25)

    # --- 3) Termination breakdown C1 vs C2
    ax = axes[2]

    entries = [
        ("DQN\nC1", dqn_c1),
        ("DQN\nC2", dqn_c2),
        ("PPO\nC1", ppo_c1),
        ("PPO\nC2", ppo_c2),
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
    ax.set_title("Failure pattern shift")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "C2 Sensor Noise Impact — C1 vs C2",
        fontsize=15,
        fontweight="bold",
        color="black",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = PLOTS_DIR / "c2_sensor_noise_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved chart: {path}")


def main():
    dqn_c1 = load_result("dqn", "c1")
    dqn_c2 = load_result("dqn", "c2")
    ppo_c1 = load_result("ppo", "c1")
    ppo_c2 = load_result("ppo", "c2")

    print_c2_cards(dqn_c1, dqn_c2, ppo_c1, ppo_c2)
    plot_c2_comparison(dqn_c1, dqn_c2, ppo_c1, ppo_c2)

    print("\nDone.")
    print(f"Chart saved in: {PLOTS_DIR / 'c2_sensor_noise_comparison.png'}")


if __name__ == "__main__":
    main()