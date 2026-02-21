# tm_collective/evaluation.py
"""
Evaluator: accuracy tracking and matplotlib plot generation.
"""

from __future__ import annotations

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False


def print_summary_table(summary: dict, share_round: int | None = None):
    """Print a formatted accuracy summary table."""
    print("\n" + "=" * 65)
    print(f"{'Agent':>20}  {'Pre-share':>10}  {'Post-share':>10}  {'Final':>8}")
    print("-" * 65)
    for agent_id, stats in summary.items():
        pre  = f"{stats['pre_share_avg']:.3f}"  if stats['pre_share_avg']  is not None else "  —  "
        post = f"{stats['post_share_avg']:.3f}" if stats['post_share_avg'] is not None else "  —  "
        fin  = f"{stats['final']:.3f}"          if stats['final']          is not None else "  —  "
        print(f"{agent_id:>20}  {pre:>10}  {post:>10}  {fin:>8}")
    print("=" * 65)


def plot_accuracy(
    history: dict[str, list[float]],
    share_rounds: list[int],
    title: str = "Collective TM Learning",
    save_path: str = "accuracy_plot.png",
    agent_labels: dict[str, str] | None = None,
):
    """
    Plot accuracy over rounds with vertical lines at sharing events.

    Args:
        history:       {agent_id: [acc_round1, acc_round2, ...]}
        share_rounds:  list of round numbers where sharing occurred
        title:         plot title
        save_path:     output file path
        agent_labels:  optional {agent_id: display_label} overrides
    """
    if not _MATPLOTLIB:
        print("matplotlib not available — skipping plot")
        return

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#FF5722"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(11, 6))
    n_rounds = max(len(v) for v in history.values())
    rounds = list(range(1, n_rounds + 1))

    for i, (aid, accs) in enumerate(history.items()):
        label = (agent_labels or {}).get(aid, aid)
        ax.plot(rounds[:len(accs)], accs,
                marker=markers[i % len(markers)],
                linewidth=2, color=colors[i % len(colors)], label=label)

    for sr in share_rounds:
        ax.axvline(x=sr, color="red", linestyle=":", linewidth=1.8, alpha=0.7)
    if share_rounds:
        ax.axvline(x=share_rounds[0], color="red", linestyle=":", linewidth=1.8,
                   alpha=0.7, label="Sharing event")

    ax.set_xlabel("Observation Round", fontsize=12)
    ax.set_ylabel("Accuracy on Clean Test Set", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0.45, 1.05)
    ax.set_xticks(rounds)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()
