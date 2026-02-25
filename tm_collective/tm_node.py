# tm_collective/tm_node.py
"""
TMNode: one agent's Tsetlin Machine plus all its associated state.

Responsibilities:
  - Maintain a buffer of all training data seen (own + absorbed)
  - Apply the agent's observation noise profile before buffering
  - Delegate generate/absorb to a SharingStrategy
  - Track accuracy history for PlateauTrigger
"""

from __future__ import annotations
import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

from tm_collective.world_schema import WorldSchema
from tm_collective.knowledge_packet import KnowledgePacket
from tm_collective.strategies.sharing import SyntheticDataStrategy, SharingStrategy


class TMNode:
    """
    One Tsetlin Machine agent.

    Args:
        agent_id:        unique string identifier
        schema:          WorldSchema instance (shared across all agents)
        noisy_features:  list of feature names this agent observes with noise
        noise_rate:      probability of bit-flip for each noisy feature bit (default 0.45)
        n_clauses:       TM hyperparameter (default 80)
        T:               TM threshold hyperparameter (default 20)
        s:               TM specificity hyperparameter (default 3.9)
        state_bits:      TM number of state bits (default 8)
        sharing:         SharingStrategy instance (default SyntheticDataStrategy())
        epochs_per_round: training epochs applied each time observe_batch() is called

    Attributes:
        agent_id:        string
        schema:          WorldSchema
        round_i:         current round counter
        last_accuracy:   float, most recent accuracy measurement
        X_buffer:        list of np.ndarray — all training inputs (own + absorbed)
        y_buffer:        list of np.ndarray — all training labels (own + absorbed)
        X_own_buffer:    list of np.ndarray — agent's own observed data only
        y_own_buffer:    list of np.ndarray — agent's own labels only
        _fitted:         bool
        _acc_history:    list of float
    """

    def __init__(
        self,
        agent_id: str,
        schema: WorldSchema,
        noisy_features: list[str] | None = None,
        noise_rate: float = 0.45,
        n_clauses: int = 80,
        T: int = 20,
        s: float = 3.9,
        state_bits: int = 8,
        sharing: SharingStrategy | None = None,
        epochs_per_round: int = 50,
    ):
        self.agent_id = agent_id
        self.schema = schema
        self._noisy_cols = schema.columns_for_features(noisy_features or [])
        self._noise_rate = noise_rate
        self._n_clauses = n_clauses
        self._T = T
        self._s = s
        self._state_bits = state_bits
        self.sharing = sharing or SyntheticDataStrategy()
        self.epochs_per_round = epochs_per_round

        self.round_i = 0
        self.last_accuracy: float = 0.5
        self.X_buffer: list[np.ndarray] = []
        self.y_buffer: list[np.ndarray] = []
        self.X_own_buffer: list[np.ndarray] = []
        self.y_own_buffer: list[np.ndarray] = []
        self._fitted = False
        self._acc_history: list[float] = []

        self.tm = self._make_tm()

    def _make_tm(self) -> MultiClassTsetlinMachine:
        return MultiClassTsetlinMachine(
            self._n_clauses, self._T, self._s,
            number_of_state_bits=self._state_bits,
            boost_true_positive_feedback=1,
        )

    def _reset_tm(self):
        """Replace TM with a fresh instance (called before full re-training)."""
        self.tm = self._make_tm()

    def _apply_noise(self, X_clean: np.ndarray) -> np.ndarray:
        """Apply bit-flip noise to this agent's noisy columns."""
        if not self._noisy_cols or self._noise_rate == 0.0:
            return X_clean.copy()
        Xn = X_clean.copy()
        mask = np.random.random((len(X_clean), len(self._noisy_cols))) < self._noise_rate
        Xn[:, self._noisy_cols] = np.where(
            mask, 1 - X_clean[:, self._noisy_cols], X_clean[:, self._noisy_cols]
        )
        return Xn

    @property
    def n_observations(self) -> int:
        return sum(len(b) for b in self.X_buffer)

    def observe_batch(self, X_clean: np.ndarray, y: np.ndarray):
        """
        Accept a new observation batch (pre-encoded binary vectors, clean),
        apply noise profile, add to buffer, and retrain.

        Args:
            X_clean: (n, schema.n_binary) uint32 — clean binary feature vectors
            y:       (n,) uint32 — ground truth labels
        """
        X_noisy = self._apply_noise(X_clean)
        self.X_buffer.append(X_noisy)
        self.y_buffer.append(y)
        self.X_own_buffer.append(X_noisy)
        self.y_own_buffer.append(y)

        X_all = np.vstack(self.X_buffer)
        y_all = np.concatenate(self.y_buffer)
        self.tm.fit(X_all, y_all, epochs=self.epochs_per_round)
        self._fitted = True
        self.round_i += 1

    def observe_dicts(self, obs_list: list[dict], y: np.ndarray):
        """
        Convenience: accept raw observation dicts, encode them, then observe.

        Args:
            obs_list: list of {feature_name: value} dicts
            y:        (n,) uint32 ground truth labels
        """
        X_clean = self.schema.encode_batch(obs_list)
        self.observe_batch(X_clean, y)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute accuracy on a clean test set.
        Updates last_accuracy and _acc_history.
        Returns 0.5 if not yet fitted.
        """
        if not self._fitted:
            return 0.5
        acc = float(np.mean(self.tm.predict(X_test) == y_test))
        self.last_accuracy = acc
        self._acc_history.append(acc)
        return acc

    def generate_knowledge(self, n: int | None = None) -> KnowledgePacket:
        """
        Generate a KnowledgePacket using this node's sharing strategy.
        Delegates entirely to self.sharing.generate().
        """
        return self.sharing.generate(self, n)

    def absorb_knowledge(self, packet: KnowledgePacket) -> dict:
        """
        Absorb a KnowledgePacket from a peer.
        Delegates entirely to self.sharing.absorb().
        Returns absorption metadata dict.
        """
        return self.sharing.absorb(self, packet)

    def accuracy_history(self) -> list[float]:
        return list(self._acc_history)

    def status(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "round_i": self.round_i,
            "n_observations": self.n_observations,
            "fitted": self._fitted,
            "last_accuracy": self.last_accuracy,
            "noisy_features_n_cols": len(self._noisy_cols),
        }

    def __repr__(self) -> str:
        return (f"TMNode(id={self.agent_id}, round={self.round_i}, "
                f"obs={self.n_observations}, acc={self.last_accuracy:.3f})")
