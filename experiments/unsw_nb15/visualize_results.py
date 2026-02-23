#!/usr/bin/env python3
"""
visualize_results.py — Visualize UNSW-NB15 collective intrusion detection results.

Run AFTER run_experiment.py — reads saved results and generates:
  1. Heatmap: per-agent, per-attack detection rates (pre vs post share)
  2. Bar chart: overall accuracy improvement
  3. Delta chart: unseen attack detection improvement
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

# ── Hardcoded results from the experiment ─────────────────────────────────
# (Run run_experiment.py with --save-results to update, or edit directly)

AGENT_ATTACKS = {
    "A": ["dos", "fuzzers", "exploits"],
    "B": ["backdoor", "shellcode", "reconnaissance"],
    "C": ["worms", "analysis", "generic"],
}

ATTACK_TYPES = [
    "dos", "fuzzers", "exploits",
    "backdoor", "shellcode", "reconnaissance",
    "worms", "analysis", "generic",
]

PRE_SHARE = {
    "A": {"dos": 1.000, "fuzzers": 1.000, "exploits": 1.000, "analysis": 1.000, "backdoor": 1.000, "generic": 0.907, "reconnaissance": 1.000, "shellcode": 1.000, "worms": 0.994, "normal": 0.988},
    "B": {"backdoor": 1.000, "shellcode": 1.000, "reconnaissance": 0.997, "analysis": 0.900, "dos": 0.897, "exploits": 0.817, "fuzzers": 0.690, "generic": 0.233, "worms": 0.848, "normal": 0.995},
    "C": {"worms": 1.000, "analysis": 0.993, "generic": 0.997, "backdoor": 0.980, "dos": 0.967, "exploits": 0.970, "fuzzers": 0.630, "reconnaissance": 0.967, "shellcode": 0.943, "normal": 0.994},
}

POST_SHARE = {
    "A": {"dos": 0.990, "fuzzers": 0.990, "exploits": 0.997, "analysis": 1.000, "backdoor": 1.000, "generic": 1.000, "reconnaissance": 1.000, "shellcode": 1.000, "worms": 1.000, "normal": 0.976},
    "B": {"backdoor": 0.990, "shellcode": 1.000, "reconnaissance": 1.000, "analysis": 0.993, "dos": 0.993, "exploits": 0.997, "fuzzers": 0.997, "generic": 0.997, "worms": 1.000, "normal": 0.975},
    "C": {"worms": 1.000, "analysis": 1.000, "generic": 0.857, "backdoor": 1.000, "dos": 1.000, "exploits": 0.997, "fuzzers": 1.000, "reconnaissance": 1.000, "shellcode": 1.000, "normal": 0.982},
}

def plot_heatmap_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.patch.set_facecolor("white")

    agents = ["A", "B", "C"]
    attacks = ATTACK_TYPES

    for ax, data, title in [(ax1, PRE_SHARE, "Pre-Share"), (ax2, POST_SHARE, "Post-Share")]:
        matrix = np.array([[data[a].get(atk, 0) for atk in attacks] for a in agents])

        # Using standard scientific colormap (Blues) scaled appropriately
        cmap = plt.cm.Blues
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        for i in range(len(agents)):
            for j in range(len(attacks)):
                val = matrix[i, j]
                is_seen = attacks[j] in AGENT_ATTACKS[agents[i]]
                txt = f"{val:.2f}"
                fontweight = "bold" if is_seen else "normal"
                # Text visibility against Blues: mostly white text for dark blue, black for light blue
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight=fontweight, color=color)

                if is_seen:
                    # distinct border for 'seen' classes
                    rect = plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                         linewidth=2.5, edgecolor="red", facecolor="none")
                    ax.add_patch(rect)

        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels([a.capitalize() for a in attacks],
                           rotation=45, ha="right", fontsize=10, color="black")
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels([f"Agent {a}" for a in agents], fontsize=12, color="black")
        ax.set_title(title, fontsize=16, fontweight="bold", color="black", pad=15)
        ax.tick_params(colors="black")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")

    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, pad=0.02)
    cbar.set_label("Detection Rate", fontsize=12, color="black")
    cbar.ax.tick_params(colors="black")

    fig.suptitle("UNSW-NB15: Attack Detection Rates by Agent",
                 fontsize=18, fontweight="bold", color="black", y=1.02)

    fig.text(0.5, -0.02, "Red Box = Seen attacks (trained on)  |  Unboxed = Unseen attacks (zero-shot)",
             ha="center", fontsize=11, color="dimgray")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "detection_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def plot_improvement_bars():
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    agents = ["A", "B", "C"]
    attacks = ATTACK_TYPES
    n_attacks = len(attacks)
    bar_width = 0.25
    x = np.arange(n_attacks)

    colors_pre = ["#1f77b4", "#2ca02c", "#d62728"]
    
    for i, agent_id in enumerate(agents):
        pre_vals = [PRE_SHARE[agent_id].get(atk, 0) for atk in attacks]
        post_vals = [POST_SHARE[agent_id].get(atk, 0) for atk in attacks]

        offset = (i - 1) * bar_width

        ax.bar(x + offset - bar_width/4, pre_vals, bar_width/2,
               color=colors_pre[i], alpha=0.4, label=f"Agent {agent_id} (pre)" if i == 0 else "")
        ax.bar(x + offset + bar_width/4, post_vals, bar_width/2,
               color=colors_pre[i], alpha=1.0, label=f"Agent {agent_id} (post)" if i == 0 else "")

        # Highlight unseen attacks with markers
        for j, atk in enumerate(attacks):
            is_unseen = atk not in AGENT_ATTACKS[agent_id]
            if is_unseen:
                delta = post_vals[j] - pre_vals[j]
                if delta > 0.05:
                    ax.annotate(f"+{delta:.0%}",
                                xy=(x[j] + offset + bar_width/4, post_vals[j]),
                                xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=9, color=colors_pre[i],
                                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in attacks],
                       rotation=45, ha="right", fontsize=11, color="black")
    ax.set_ylabel("Detection Rate", fontsize=13, color="black")
    ax.set_ylim(0, 1.15)
    ax.set_title("Attack Detection: Pre-Share vs Post-Share",
                 fontsize=16, fontweight="bold", color="black", pad=15)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Oracle baseline line
    oracle_acc = 0.9799
    ax.axhline(y=oracle_acc, color='purple', linestyle='--', linewidth=2, zorder=0)

    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="gray", alpha=0.4, label="Pre-share isolated"),
        Patch(facecolor="gray", alpha=1.0, label="Post-share collective"),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label=f"Central Oracle Baseline ({oracle_acc:.1%})")
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="lower right",
              facecolor="white", edgecolor="black", labelcolor="black")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "improvement_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def plot_delta_chart():
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    agents = ["A", "B", "C"]
    agent_colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}

    all_bars = []
    all_labels = []
    all_colors = []

    for agent_id in agents:
        for atk in ATTACK_TYPES:
            if atk not in AGENT_ATTACKS[agent_id]:
                pre = PRE_SHARE[agent_id].get(atk, 0)
                post = POST_SHARE[agent_id].get(atk, 0)
                delta = post - pre
                all_bars.append(delta)
                all_labels.append(f"{agent_id}:{atk[:6]}")
                all_colors.append(agent_colors[agent_id])

    x = np.arange(len(all_bars))
    bars = ax.bar(x, all_bars, color=all_colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, all_bars):
        color = "darkgreen" if val > 0.05 else ("darkgoldenrod" if val > 0 else "darkred")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"+{val:.0%}" if val >= 0 else f"{val:.0%}",
                ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=65, ha="right", fontsize=10, color="black")
    ax.set_ylabel("Δ Detection Rate", fontsize=13, color="black")
    ax.set_title("Improvement on UNSEEN Attack Types After Knowledge Sharing",
                 fontsize=15, fontweight="bold", color="black", pad=15)
    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=agent_colors[a], alpha=0.85, label=f"Agent {a}: {', '.join(AGENT_ATTACKS[a])}")
        for a in agents
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left",
              facecolor="white", edgecolor="black", labelcolor="black")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "unseen_improvement.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def plot_summary_card():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.text(0.5, 0.95, "UNSW-NB15 Collective Intrusion Detection",
            ha="center", va="top", fontsize=20, fontweight="bold", color="black",
            transform=ax.transAxes)
    ax.text(0.5, 0.88, "3 TM Agents × 9 Attack Types × Knowledge Sharing",
            ha="center", va="top", fontsize=13, color="dimgray",
            transform=ax.transAxes)

    metrics = [
        ("Avg Unseen Detection", "87.5% → 99.9%", "+12.4pp", "darkgreen"),
        ("Agent B: Generic", "23.3% → 99.7%", "+76.3pp", "darkgreen"),
        ("Agent C: Fuzzers", "63.0% → 100%", "+37.0pp", "darkgreen"),
        ("Agent B Overall", "89.6% → 98.7%", "+9.1pp", "darkgreen"),
    ]

    for i, (label, before_after, delta, color) in enumerate(metrics):
        y = 0.70 - i * 0.17
        ax.text(0.08, y, label, ha="left", va="center", fontsize=14,
                color="black", transform=ax.transAxes)
        ax.text(0.55, y, before_after, ha="center", va="center", fontsize=14,
                color="dimgray", transform=ax.transAxes, family="monospace")
        ax.text(0.82, y, delta, ha="center", va="center", fontsize=16,
                fontweight="bold", color=color, transform=ax.transAxes)

    ax.text(0.5, 0.02,
            "Dataset: UNSW-NB15 (2M flows) | Features: 128 boolean | "
            "Method: Synthetic data sharing | TM Clauses: 2000",
            ha="center", va="bottom", fontsize=10, color="gray",
            transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating visualizations...")
    plot_heatmap_comparison()
    plot_improvement_bars()
    plot_delta_chart()
    plot_summary_card()
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
