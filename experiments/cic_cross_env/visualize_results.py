#!/usr/bin/env python3
"""
visualize_results.py — Visualize cross-environment experiment results.

Reads results.json and generates:
  1. Degradation bar chart (within→cross) per condition
  2. Per-attack heatmap (conditions × settings)
  3. Environment shift profile (KL divergence)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
META_PATH = os.path.join(SCRIPT_DIR, "data", "prepared", "feature_meta.json")

AGENT_IDS = ["A", "B", "C"]
AGENT_ATTACKS = {
    "A": ["ddos", "dos_hulk", "dos_goldeneye"],
    "B": ["dos_slowloris", "dos_slowhttptest", "bot"],
    "C": ["portscan", "web_bruteforce", "web_xss", "web_sqli", "infiltration"],
}
ALL_ATTACKS = sorted(set(a for attacks in AGENT_ATTACKS.values() for a in attacks))
CONDITIONS = ["baseline", "synthetic", "synthetic_graduated", "clause"]


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def avg_unseen_detection(results, condition, setting):
    """Average unseen attack detection rate across all seeds and agents."""
    vals = []
    for seed_results in results[condition].values():
        if setting in seed_results:
            for agent_id in AGENT_IDS:
                if agent_id in seed_results[setting]:
                    unseen = seed_results[setting][agent_id].get("unseen", {})
                    vals.extend(unseen.values())
    return np.mean(vals) if vals else 0, np.std(vals) if vals else 0


def avg_per_attack(results, condition, setting, attack):
    """Average detection rate for a specific attack type across seeds and agents."""
    vals = []
    for seed_results in results[condition].values():
        if setting in seed_results:
            for agent_id in AGENT_IDS:
                if agent_id in seed_results[setting]:
                    per_attack = seed_results[setting][agent_id].get("per_attack", {})
                    if attack in per_attack:
                        vals.append(per_attack[attack])
    return np.mean(vals) if vals else 0


def plot_degradation_bars(results):
    """Plot 1: Within vs Cross degradation by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    conditions = CONDITIONS
    bar_width = 0.3
    x = np.arange(len(conditions))

    within_vals = []
    cross_vals = []
    within_errs = []
    cross_errs = []

    for cond in conditions:
        w_mean, w_std = avg_unseen_detection(results, cond, "within")
        c_mean, c_std = avg_unseen_detection(results, cond, "cross")
        within_vals.append(w_mean)
        cross_vals.append(c_mean)
        within_errs.append(w_std)
        cross_errs.append(c_std)

    bars1 = ax.bar(x - bar_width/2, within_vals, bar_width, yerr=within_errs,
                   color="#2196F3", alpha=0.85, label="Within-env (2017→2017)",
                   capsize=4, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + bar_width/2, cross_vals, bar_width, yerr=cross_errs,
                   color="#FF5722", alpha=0.85, label="Cross-env (2017→2018)",
                   capsize=4, edgecolor="black", linewidth=0.5)

    # Add delta labels
    for i, (w, c) in enumerate(zip(within_vals, cross_vals)):
        delta = w - c
        color = "darkred" if delta > 0.05 else ("darkgoldenrod" if delta > 0 else "darkgreen")
        ax.annotate(f"Δ={delta:+.1%}",
                    xy=(x[i], max(w, c) + 0.05),
                    ha="center", fontsize=11, color=color, fontweight="bold")

    # Oracle reference
    if "oracle" in results:
        w_oracle, _ = avg_unseen_detection(results, "oracle", "within")
        c_oracle, _ = avg_unseen_detection(results, "oracle", "cross")
        ax.axhline(y=w_oracle, color="purple", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(y=c_oracle, color="purple", linestyle=":", linewidth=1.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in conditions], fontsize=12, color="black")
    ax.set_ylabel("Avg Unseen Attack Detection Rate", fontsize=13, color="black")
    ax.set_title("Cross-Environment Degradation by Sharing Strategy",
                 fontsize=15, fontweight="bold", color="black", pad=15)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(fontsize=11, loc="upper right", facecolor="white",
              edgecolor="black", labelcolor="black")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "degradation_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def plot_attack_heatmap(results):
    """Plot 2: Per-attack heatmap (conditions × settings on columns, attacks on rows)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    fig.patch.set_facecolor("white")

    settings = ["within", "cross"]
    setting_labels = ["Within-env (2017→2017)", "Cross-env (2017→2018)"]

    for ax, setting, slab in zip(axes, settings, setting_labels):
        # Matrix: attacks × conditions
        conds = CONDITIONS + (["oracle"] if "oracle" in results else [])
        matrix = np.zeros((len(ALL_ATTACKS), len(conds)))

        for j, cond in enumerate(conds):
            for i, attack in enumerate(ALL_ATTACKS):
                matrix[i, j] = avg_per_attack(results, cond, setting, attack)

        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        for i in range(len(ALL_ATTACKS)):
            for j in range(len(conds)):
                val = matrix[i, j]
                color = "white" if val < 0.4 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.capitalize() for c in conds],
                           rotation=45, ha="right", fontsize=11, color="black")
        ax.set_yticks(range(len(ALL_ATTACKS)))
        ax.set_yticklabels([a.replace("_", " ").title() for a in ALL_ATTACKS],
                           fontsize=10, color="black")
        ax.set_title(slab, fontsize=14, fontweight="bold", color="black", pad=10)
        ax.tick_params(colors="black")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Detection Rate", fontsize=12, color="black")
    cbar.ax.tick_params(colors="black")

    fig.suptitle("Per-Attack Detection: Conditions × Environments",
                 fontsize=16, fontweight="bold", color="black", y=1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "attack_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def plot_shift_profile(results):
    """Plot 3: Per-attack degradation correlated with feature distribution shift."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # For each attack, compute degradation from within→cross for each condition
    cond_colors = {"baseline": "#9E9E9E", "synthetic": "#2196F3",
                    "synthetic_graduated": "#4CAF50", "clause": "#FF9800"}
    bar_width = 0.2
    x = np.arange(len(ALL_ATTACKS))

    for ci, cond in enumerate(CONDITIONS):
        deltas = []
        for attack in ALL_ATTACKS:
            within_val = avg_per_attack(results, cond, "within", attack)
            cross_val = avg_per_attack(results, cond, "cross", attack)
            deltas.append(within_val - cross_val)

        offset = (ci - 1.5) * bar_width
        bars = ax.bar(x + offset, deltas, bar_width,
                      color=cond_colors[cond], alpha=0.85,
                      label=cond.capitalize(), edgecolor="black", linewidth=0.5)

        # Label significant degradations
        for i, d in enumerate(deltas):
            if abs(d) > 0.05:
                color = "darkred" if d > 0 else "darkgreen"
                ax.text(x[i] + offset, d + 0.02 * np.sign(d),
                        f"{d:+.0%}", ha="center", va="bottom" if d > 0 else "top",
                        fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ").title() for a in ALL_ATTACKS],
                       rotation=45, ha="right", fontsize=10, color="black")
    ax.set_ylabel("Δ Detection Rate (within - cross)", fontsize=13, color="black")
    ax.set_title("Per-Attack Cross-Environment Degradation by Strategy",
                 fontsize=15, fontweight="bold", color="black", pad=15)
    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(fontsize=11, loc="upper right", facecolor="white",
              edgecolor="black", labelcolor="black")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.02,
             "Positive = degradation when moving to new environment | "
             "Negative = improved detection in new environment",
             ha="center", fontsize=10, color="dimgray")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shift_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def plot_summary_card(results):
    """Summary card with key metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.text(0.5, 0.95, "CIC-IDS2017 → CSE-CIC-IDS2018",
            ha="center", va="top", fontsize=22, fontweight="bold", color="black",
            transform=ax.transAxes)
    ax.text(0.5, 0.88, "Cross-Environment Knowledge Transfer Experiment",
            ha="center", va="top", fontsize=14, color="dimgray",
            transform=ax.transAxes)

    metrics = []
    for cond in CONDITIONS:
        w_mean, _ = avg_unseen_detection(results, cond, "within")
        c_mean, _ = avg_unseen_detection(results, cond, "cross")
        delta = w_mean - c_mean
        metrics.append((
            f"{cond.capitalize()} degradation",
            f"Within: {w_mean:.1%} → Cross: {c_mean:.1%}",
            f"Δ={delta:+.1%}",
            "darkred" if delta > 0.05 else ("darkgoldenrod" if delta > 0 else "darkgreen"),
        ))

    for i, (label, before_after, delta, color) in enumerate(metrics):
        y = 0.72 - i * 0.15
        ax.text(0.05, y, label, ha="left", va="center", fontsize=14,
                color="black", transform=ax.transAxes)
        ax.text(0.50, y, before_after, ha="center", va="center", fontsize=13,
                color="dimgray", transform=ax.transAxes, family="monospace")
        ax.text(0.88, y, delta, ha="center", va="center", fontsize=16,
                fontweight="bold", color=color, transform=ax.transAxes)

    # Hypothesis result
    syn_w, _ = avg_unseen_detection(results, "synthetic", "within")
    syn_c, _ = avg_unseen_detection(results, "synthetic", "cross")
    cls_w, _ = avg_unseen_detection(results, "clause", "within")
    cls_c, _ = avg_unseen_detection(results, "clause", "cross")
    syn_delta = syn_w - syn_c
    cls_delta = cls_w - cls_c
    supported = syn_delta < cls_delta
    verdict = "SUPPORTED ✓" if supported else "NOT SUPPORTED ✗"
    verdict_color = "darkgreen" if supported else "darkred"

    ax.text(0.5, 0.12,
            f"Hypothesis: Δ_synthetic ({syn_delta:.1%}) < Δ_clause ({cls_delta:.1%})",
            ha="center", va="center", fontsize=14, color="black",
            transform=ax.transAxes)
    ax.text(0.5, 0.04, verdict, ha="center", va="center", fontsize=18,
            fontweight="bold", color=verdict_color, transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: No results file found at {RESULTS_PATH}")
        print("Run run_experiment.py first.")
        return

    results = load_results()
    print("Generating visualizations...")

    plot_degradation_bars(results)
    plot_attack_heatmap(results)
    plot_shift_profile(results)
    plot_summary_card(results)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
