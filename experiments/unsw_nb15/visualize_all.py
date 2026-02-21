#!/usr/bin/env python3
"""
visualize_all.py — Unified visualization for all UNSW-NB15 collective experiments.

Generates 3 comparison plots summarizing the results of our experiments:
1. Baseline TM (128 features) vs LLM-Driven (20 features) Improvement
2. Direct comparison of Zero-Shot Onboarding results (Baseline vs LLM)
3. Overall detection rate bar chart across all modes
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots_summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────
COLOR_BG = "white"
COLOR_PANEL = "white"
COLOR_TEXT = "black"
COLOR_MUTED = "dimgray"
COLOR_GRID = "lightgray"

# Scientific palette (tab10)
COLOR_BASELINE = "#1f77b4" # blue
COLOR_BASELINE_LIGHT = "#aec7e8"

COLOR_LLM = "#d62728" # red
COLOR_LLM_LIGHT = "#ff9896"

COLOR_ZERO = "#2ca02c" # green
COLOR_ZERO_LIGHT = "#98df8a"


def setup_plot(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_PANEL)
    ax.tick_params(colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_MUTED)
    ax.grid(axis='y', color=COLOR_GRID, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


# ── Plot 1: Standard Experiment (Pre vs Post Share) ─────────────────────────
def plot_pre_post_comparison():
    fig, ax = setup_plot((12, 7))

    # Data
    labels = ["Baseline TM\n(128 features, 200k samples)", "LLM-Driven TM\n(20 approx features, 30 samples)"]
    
    pre = [87.5, 28.0]
    post = [99.9, 57.3]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, pre, width, label='Pre-Share', color=[COLOR_BASELINE, COLOR_LLM], alpha=0.6)
    rects2 = ax.bar(x + width/2, post, width, label='Post-Share', color=[COLOR_BASELINE, COLOR_LLM], alpha=1.0)

    # Add labels
    for rects, vals in zip([rects1, rects2], [pre, post]):
        for i, (rect, val) in enumerate(zip(rects, vals)):
            ax.annotate(f'{val:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 5),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color=COLOR_TEXT, fontweight='bold', fontsize=12)

    # Add deltas
    for i, (p, po) in enumerate(zip(pre, post)):
        delta = po - p
        ax.annotate(f'+{delta:.1f}pp',
                    xy=(x[i] + width/2, po/2),
                    ha='center', va='center', color=COLOR_BG, fontweight='bold', fontsize=14)

    ax.set_ylabel('Accuracy (%)', color=COLOR_TEXT, fontsize=14)
    ax.set_title('Impact of Knowledge Sharing (Pre vs Post)', color=COLOR_TEXT, fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=COLOR_TEXT, fontsize=13)
    ax.set_ylim(0, 110)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.6, label="Pre-share (Isolated)"),
        Patch(facecolor='gray', alpha=1.0, label="Post-share (Collective)"),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor=COLOR_PANEL, edgecolor=COLOR_MUTED, labelcolor=COLOR_TEXT)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_knowledge_sharing_impact.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")


# ── Plot 2: Zero-Shot Onboarding ──────────────────────────────────────────
def plot_zero_shot_onboarding():
    fig, ax = setup_plot((10, 6))

    # Data
    labels = ["Baseline collective max", "Agent D (Baseline Zero-Shot)", "LLM collective max", "Agent D (LLM Zero-Shot)"]
    
    accuracies = [98.9, 97.7, 72.0, 72.0]
    colors = [COLOR_BASELINE, COLOR_BASELINE_LIGHT, COLOR_LLM, COLOR_LLM_LIGHT]

    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.9, width=0.6)

    # Highlight exact matches
    ax.plot([0, 1], [accuracies[0], accuracies[0]], color=COLOR_MUTED, linestyle='--', zorder=0)
    ax.plot([2, 3], [accuracies[2], accuracies[2]], color=COLOR_MUTED, linestyle='--', zorder=0)

    for bar, val, is_d in zip(bars, accuracies, [False, True, False, True]):
        weight = 'bold' if is_d else 'normal'
        size = 14 if is_d else 12
        ax.annotate(f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', color=COLOR_TEXT, fontweight=weight, fontsize=size)

        if is_d:
            ax.annotate("Zero data\ntraining",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()/2),
                        ha='center', va='center', color='black', fontweight='bold', fontsize=11)

    ax.set_ylabel('Accuracy on Test Set (%)', color=COLOR_TEXT, fontsize=14)
    ax.set_title('Zero-Shot Sensor Onboarding Performance', color=COLOR_TEXT, fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(["Trained\nCollective\n(Baseline)", "New Agent D\nAbsorbing Pkt\n(Baseline)", 
                        "Trained\nCollective\n(LLM-Driven)", "New Agent D\nAbsorbing Pkt\n(LLM-Driven)"], 
                       color=COLOR_TEXT, fontsize=11)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_zero_shot_onboarding.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")


# ── Plot 3: Summary Card ────────────────────────────────────────────────
def plot_final_summary():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.axis("off")

    ax.text(0.5, 0.9, "UNSW-NB15 Collective Intelligence Summary",
            ha="center", va="top", fontsize=22, fontweight="bold", color=COLOR_TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.8, "Tsetlin Machines sharing propositional logic without model limits",
            ha="center", va="top", fontsize=12, color=COLOR_MUTED, transform=ax.transAxes)

    metrics = [
        ("Centralized Oracle", "98.0%", "5-seed mean for monolithic TM (± 0.26 variance)"),
        ("Collective vs Oracle", "Parity", "Statistically indistinguishable (98.4% avg, 98.6% peak)"),
        ("Zero-Shot Onboarding", "97.7%", "Blank Agent D immediately operational via collective"),
        ("LLM-Driven TM (+29.3pp)", "57.3%", "Noisy text-extracted features still double accuracy"),
    ]

    for i, (title, highlight, desc) in enumerate(metrics):
        y = 0.6 - i*0.15
        
        ax.text(0.1, y, title, ha="left", va="center", fontsize=14, color=COLOR_TEXT, transform=ax.transAxes, fontweight="bold")
        ax.text(0.45, y, highlight, ha="left", va="center", fontsize=16, color=COLOR_ZERO, transform=ax.transAxes, fontweight="bold")
        ax.text(0.65, y, desc, ha="left", va="center", fontsize=11, color=COLOR_MUTED, transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_overall_summary.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")


def main():
    print("Generating comprehensive visualizations...")
    plot_pre_post_comparison()
    plot_zero_shot_onboarding()
    plot_final_summary()
    print("Done.")

if __name__ == "__main__":
    main()
