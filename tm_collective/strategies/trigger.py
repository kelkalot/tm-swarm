# tm_collective/strategies/trigger.py
"""
Trigger policies: when an agent decides to share.

FixedRound  — share every N rounds
Plateau     — share when accuracy improvement drops below a threshold
OnceOnly    — share exactly once, at or after a specified round
"""

from __future__ import annotations


class TriggerPolicy:
    def should_share(self, agent_id: str, round_i: int, accuracy_history: list[float]) -> bool:
        raise NotImplementedError


class FixedRoundTrigger(TriggerPolicy):
    """
    Share every `every_n_rounds` rounds.

    Args:
        every_n_rounds: sharing interval (default 5)
    """
    def __init__(self, every_n_rounds: int = 5):
        self.every_n = every_n_rounds

    def should_share(self, agent_id, round_i, accuracy_history):
        return round_i % self.every_n == 0


class PlateauTrigger(TriggerPolicy):
    """
    Share when accuracy has plateaued: the range of accuracy values in the
    last `window` rounds is below `min_improvement`.

    Useful for adaptive sharing — agents share only when they've stopped
    improving on their own, which is when peer knowledge is most valuable.

    Args:
        min_improvement: accuracy range threshold below which sharing triggers
                         (e.g. 0.005 means <0.5% improvement over last window)
        window:          number of recent rounds to look back
        min_rounds:      don't trigger before this many rounds (warm-up)
    """
    def __init__(self, min_improvement: float = 0.005, window: int = 3, min_rounds: int = 3):
        self.min_improvement = min_improvement
        self.window = window
        self.min_rounds = min_rounds

    def should_share(self, agent_id, round_i, accuracy_history):
        if round_i < self.min_rounds or len(accuracy_history) < self.window:
            return False
        recent = accuracy_history[-self.window:]
        return (max(recent) - min(recent)) < self.min_improvement


class OnceOnlyTrigger(TriggerPolicy):
    """
    Each agent shares exactly once, at or after `at_round`.
    After firing, subsequent calls return False for that agent.

    Args:
        at_round: earliest round at which sharing is triggered
    """
    def __init__(self, at_round: int):
        self.at_round = at_round
        self._fired: set[str] = set()

    def should_share(self, agent_id, round_i, accuracy_history):
        if agent_id not in self._fired and round_i >= self.at_round:
            self._fired.add(agent_id)
            return True
        return False
