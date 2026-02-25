# tm_collective/strategies/sharing.py
"""
Sharing strategies: how agents export and absorb knowledge.

SyntheticDataStrategy (RECOMMENDED):
  - Generate synthetic data from a trained TM and share with peers.
  - Supports perturbation-based generation (perturbs real training data)
    or pure random inputs (legacy fallback).
  - Supports graduated or fixed flip rates for perturbation.
  - Supports full or hybrid absorption modes.
  - Validated: cross-env experiments (CIC-IDS2017 → CSE-CIC-IDS2018)
    show graduated perturbation achieves 51.3% cross-env detection,
    matching a centralized baseline (50.7%).

ClauseTransferStrategy (DEPRECATED):
  - Showed no improvement over baseline in cross-environment experiments.
  - Kept for backward compatibility only.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from tm_collective.knowledge_packet import KnowledgePacket


class SharingStrategy(ABC):
    """Base class for knowledge sharing strategies."""

    @abstractmethod
    def generate(self, tm_node, n: int) -> KnowledgePacket:
        """Generate a KnowledgePacket from this agent's current knowledge."""
        ...

    @abstractmethod
    def absorb(self, tm_node, packet: KnowledgePacket) -> dict:
        """
        Absorb a KnowledgePacket into this agent's TM.
        Returns a dict with absorption metadata (samples added, accuracy change, etc.).
        """
        ...


class SyntheticDataStrategy(SharingStrategy):
    """
    Generate synthetic data from a trained TM and share with peers.

    Generation modes (controls how synthetic inputs are created):
        "perturb"   — perturb real training data with bit flips (RECOMMENDED).
                      Better cross-environment generalization than random inputs.
                      Requires training data in tm_node.X_buffer or X_own_buffer.
        "random"    — pure random binary inputs (original behavior, fallback).

    Perturbation rate modes (controls flip probability):
        "graduated" — sample flip_rate uniformly from [flip_rate_min, flip_rate_max]
                      each iteration. Mixes close-to-real and exploratory samples.
                      Best cross-environment performance. (RECOMMENDED)
        "fixed"     — constant flip_rate = flip_rate_min throughout.

    Absorption modes (controls how receiver integrates peer knowledge):
        "full"      — absorb all peer samples (attacks + normals). Best within-env.
        "hybrid"    — absorb peer attack samples (y=1) only; generate local normals
                      from receiver's own TM using X_own_buffer. Best stability
                      (lowest variance across seeds).

    Note: all agents in a collective should use the same strategy configuration.

    Args:
        n_synthetic:    synthetic samples to generate per share event (default 500)
        retrain_epochs: epochs after absorption (default 150)
        mode:           "perturb" | "random" (default "perturb")
        rate_mode:      "graduated" | "fixed" (default "graduated")
        flip_rate_min:  minimum flip probability (default 0.05)
        flip_rate_max:  maximum flip probability (default 0.50)
        absorption:     "full" | "hybrid" (default "full")
    """

    def __init__(
        self,
        n_synthetic: int = 500,
        retrain_epochs: int = 150,
        mode: str = "perturb",
        rate_mode: str = "graduated",
        flip_rate_min: float = 0.05,
        flip_rate_max: float = 0.50,
        absorption: str = "full",
    ):
        self.n_synthetic = n_synthetic
        self.retrain_epochs = retrain_epochs
        self.mode = mode
        self.rate_mode = rate_mode
        self.flip_rate_min = flip_rate_min
        self.flip_rate_max = flip_rate_max
        self.absorption = absorption

    def _sample_flip_rate(self) -> float:
        """Sample a flip rate based on the configured rate_mode."""
        if self.rate_mode == "graduated":
            return float(np.random.uniform(self.flip_rate_min, self.flip_rate_max))
        return self.flip_rate_min

    def _get_training_pool(self, tm_node) -> np.ndarray | None:
        """Get the training data pool for perturbation-based generation.

        Prefers X_own_buffer (agent's own data only, no peer contamination).
        Falls back to X_buffer if X_own_buffer is not available.
        Returns None if no data is available.
        """
        own = getattr(tm_node, "X_own_buffer", None)
        if own:
            return np.vstack(own).astype(np.uint32)
        if tm_node.X_buffer:
            return np.vstack(tm_node.X_buffer).astype(np.uint32)
        return None

    def _generate_perturbed(self, tm_node, n: int, n_binary: int):
        """
        Class-balanced synthetic data via perturbation + rejection sampling.
        Uses graduated or fixed flip rates based on rate_mode.
        """
        target_per_class = n // 2
        class_0_X, class_1_X = [], []
        n0, n1 = 0, 0
        max_attempts = 50

        X_pool = self._get_training_pool(tm_node)
        if X_pool is None or len(X_pool) == 0:
            return None, None  # signal to caller: fall back to random

        for _ in range(max_attempts):
            if n0 >= target_per_class and n1 >= target_per_class:
                break

            flip_rate = self._sample_flip_rate()

            indices = np.random.randint(0, len(X_pool), size=n)
            X_batch = X_pool[indices].copy()
            flip_mask = np.random.random(X_batch.shape) < flip_rate
            X_batch = np.where(flip_mask, 1 - X_batch, X_batch).astype(np.uint32)
            y_batch = tm_node.tm.predict(X_batch)

            for cls, collector in [(0, class_0_X), (1, class_1_X)]:
                n_collected = sum(len(a) for a in collector)
                needed = target_per_class - n_collected
                mask = y_batch == cls
                if needed > 0 and mask.sum() > 0:
                    collector.append(X_batch[mask][:needed])

            n0 = sum(len(a) for a in class_0_X)
            n1 = sum(len(a) for a in class_1_X)

        X_0 = np.vstack(class_0_X)[:target_per_class] if class_0_X else None
        X_1 = np.vstack(class_1_X)[:target_per_class] if class_1_X else None

        if X_0 is None or X_1 is None:
            return None, None  # fall back to random

        X_syn = np.vstack([X_0, X_1])
        y_syn = np.array([0] * len(X_0) + [1] * len(X_1), dtype=np.uint32)
        perm = np.random.permutation(len(X_syn))
        return X_syn[perm], y_syn[perm]

    def _generate_local_normals(self, tm_node, n: int, n_binary: int):
        """Generate n normal-class (y=0) samples from receiver's own TM.

        Uses X_own_buffer (uncontaminated by peer data) for perturbation.
        Falls back to random inputs if no training pool is available.
        """
        collected = []
        n_collected = 0
        max_attempts = 30

        X_pool = self._get_training_pool(tm_node)
        if X_pool is not None and len(X_pool) > 0:
            for _ in range(max_attempts):
                if n_collected >= n:
                    break
                flip_rate = self._sample_flip_rate()
                indices = np.random.randint(0, len(X_pool), size=n)
                X_batch = X_pool[indices].copy()
                flip_mask = np.random.random(X_batch.shape) < flip_rate
                X_batch = np.where(flip_mask, 1 - X_batch, X_batch).astype(np.uint32)
                y_batch = tm_node.tm.predict(X_batch)
                normal_mask = y_batch == 0
                needed = n - n_collected
                if normal_mask.sum() > 0:
                    collected.append(X_batch[normal_mask][:needed])
                    n_collected += min(int(normal_mask.sum()), needed)

        if not collected:
            # Fallback: random inputs classified as normal
            X_rand = np.random.randint(0, 2, (n * 2, n_binary)).astype(np.uint32)
            y_rand = tm_node.tm.predict(X_rand)
            normal_mask = y_rand == 0
            X_rand = X_rand[normal_mask][:n]
            if len(X_rand) == 0:
                X_rand = np.random.randint(0, 2, (n, n_binary)).astype(np.uint32)
            return X_rand, np.zeros(len(X_rand), dtype=np.uint32)

        X_local = np.vstack(collected)[:n]
        return X_local, np.zeros(len(X_local), dtype=np.uint32)

    def generate(self, tm_node, n: int | None = None) -> KnowledgePacket:
        n = n or self.n_synthetic
        n_binary = tm_node.schema.n_binary

        if not tm_node._fitted:
            X_rand = np.random.randint(0, 2, (n, n_binary)).astype(np.uint32)
            y_rand = np.random.randint(0, 2, n).astype(np.uint32)
            return KnowledgePacket(tm_node.agent_id, tm_node.round_i, X_rand, y_rand,
                                   {"fitted": False})

        # Choose generation path
        X_syn, y_syn = None, None
        if self.mode == "perturb":
            X_syn, y_syn = self._generate_perturbed(tm_node, n, n_binary)

        if X_syn is None:
            # Fallback: pure random (original behavior)
            X_syn = np.random.randint(0, 2, (n, n_binary)).astype(np.uint32)
            y_syn = tm_node.tm.predict(X_syn).astype(np.uint32)

        meta = {
            "fitted": True,
            "mode": self.mode,
            "rate_mode": self.rate_mode,
            "accuracy_at_share": tm_node.last_accuracy,
            "n_train_samples": tm_node.n_observations,
        }
        return KnowledgePacket(tm_node.agent_id, tm_node.round_i, X_syn, y_syn, meta)

    def absorb(self, tm_node, packet: KnowledgePacket) -> dict:
        """
        Absorb a KnowledgePacket. Behavior depends on self.absorption:

        "full":   add all peer samples to buffer, reset, retrain.
                  Best within-environment performance.
        "hybrid": add only peer attack samples (y=1); generate local normal
                  samples from receiver's own TM. Best stability.
        """
        if self.absorption == "hybrid" and tm_node._fitted:
            attack_mask = packet.y == 1
            X_peer = packet.X[attack_mask]
            y_peer = packet.y[attack_mask]

            if len(X_peer) > 0:
                n_binary = tm_node.schema.n_binary
                X_local, y_local = self._generate_local_normals(
                    tm_node, len(X_peer), n_binary
                )
                tm_node.X_buffer.append(X_peer)
                tm_node.y_buffer.append(y_peer)
                tm_node.X_buffer.append(X_local)
                tm_node.y_buffer.append(y_local)
            else:
                # No attack samples — absorb everything
                tm_node.X_buffer.append(packet.X)
                tm_node.y_buffer.append(packet.y)
        else:
            # "full" mode: original behavior
            tm_node.X_buffer.append(packet.X)
            tm_node.y_buffer.append(packet.y)

        # Reset and retrain from scratch on all accumulated data
        tm_node._reset_tm()
        X_all = np.vstack(tm_node.X_buffer)
        y_all = np.concatenate(tm_node.y_buffer)
        tm_node.tm.fit(X_all, y_all, epochs=self.retrain_epochs)
        tm_node._fitted = True

        return {
            "absorbed_from": packet.sender_id,
            "absorption_mode": self.absorption,
            "packet_size": len(packet),
            "total_samples_now": len(X_all),
        }


class ClauseTransferStrategy(SharingStrategy):
    """
    DEPRECATED. Showed no improvement over baseline in cross-environment
    experiments (CIC-IDS2017 → CSE-CIC-IDS2018). Statistically indistinguishable
    from no sharing across all conditions and seeds.

    Use SyntheticDataStrategy instead.

    Directly injects confident TA states from source TM into target TM.

    Requirements:
        - Both agents must use identical TM hyperparameters (n_clauses, T, s, state_bits)
        - Both agents must observe the same WorldSchema (identical binary feature space)
    """

    def __init__(self, top_k: int = 10):
        import warnings
        warnings.warn(
            "ClauseTransferStrategy is deprecated and showed no improvement "
            "over baseline in validation experiments. Use SyntheticDataStrategy.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.top_k = top_k

    def _decode_ta_states(self, ta_packed, n_clauses, n_features, n_state_bits, n_ta_chunks):
        n_literals = 2 * n_features
        states = np.zeros((n_clauses, n_literals), dtype=np.int32)
        for ci in range(n_clauses):
            for chunk in range(n_ta_chunks):
                for bit in range(n_state_bits):
                    idx = ci * n_ta_chunks * n_state_bits + chunk * n_state_bits + bit
                    val = ta_packed[idx]
                    for j in range(32):
                        lit = chunk * 32 + j
                        if lit >= n_literals:
                            break
                        states[ci, lit] |= (((val >> j) & 1) << bit)
        return states

    def _encode_ta_states(self, states, n_clauses, n_features, n_state_bits, n_ta_chunks):
        n_literals = 2 * n_features
        packed = np.zeros(n_clauses * n_ta_chunks * n_state_bits, dtype=np.uint32)
        for ci in range(n_clauses):
            for chunk in range(n_ta_chunks):
                for bit in range(n_state_bits):
                    idx = ci * n_ta_chunks * n_state_bits + chunk * n_state_bits + bit
                    val = np.uint32(0)
                    for j in range(32):
                        lit = chunk * 32 + j
                        if lit >= n_literals:
                            break
                        val |= np.uint32(((int(states[ci, lit]) >> bit) & 1) << j)
                    packed[idx] = val
        return packed

    def _clause_confidence(self, decoded, n_state_bits):
        midpoint = 2 ** (n_state_bits - 1)
        return np.mean(np.abs(decoded.astype(float) - midpoint) / midpoint, axis=1)

    def generate(self, tm_node, n=None) -> KnowledgePacket:
        if not tm_node._fitted:
            return KnowledgePacket(tm_node.agent_id, tm_node.round_i,
                                   np.zeros((1, tm_node.schema.n_binary), dtype=np.uint32),
                                   np.zeros(1, dtype=np.uint32),
                                   {"fitted": False, "strategy": "clause_transfer"})

        n_clauses  = tm_node.tm.number_of_clauses
        n_features = tm_node.schema.n_binary
        state_bits = tm_node.tm.number_of_state_bits
        ta_chunks  = tm_node.tm.number_of_ta_chunks
        n_classes  = tm_node.tm.number_of_classes

        top_clauses_per_class = {}
        full_state = tm_node.tm.get_state()

        for class_i in range(n_classes):
            _, ta_packed = full_state[class_i]
            decoded = self._decode_ta_states(ta_packed, n_clauses, n_features,
                                             state_bits, ta_chunks)
            conf = self._clause_confidence(decoded, state_bits)
            top_idx = np.argsort(conf)[-self.top_k:]
            top_clauses_per_class[class_i] = decoded[top_idx].tolist()

        meta = {
            "strategy": "clause_transfer",
            "top_k": self.top_k,
            "n_clauses": n_clauses,
            "state_bits": state_bits,
            "ta_chunks": ta_chunks,
            "top_clauses_per_class": top_clauses_per_class,
        }
        # X and y are empty placeholders for clause transfer
        empty = np.zeros((1, n_features), dtype=np.uint32)
        return KnowledgePacket(tm_node.agent_id, tm_node.round_i, empty,
                               np.zeros(1, dtype=np.uint32), meta)

    def absorb(self, tm_node, packet: KnowledgePacket) -> dict:
        if not packet.metadata.get("fitted", True):
            return {"absorbed_from": packet.sender_id, "skipped": True, "reason": "sender not fitted"}

        imported = packet.metadata.get("top_clauses_per_class", {})
        if not imported:
            return {"absorbed_from": packet.sender_id, "skipped": True, "reason": "no clauses in packet"}

        n_clauses  = tm_node.tm.number_of_clauses
        n_features = tm_node.schema.n_binary
        state_bits = tm_node.tm.number_of_state_bits
        ta_chunks  = tm_node.tm.number_of_ta_chunks
        n_classes  = tm_node.tm.number_of_classes

        full_state = list(tm_node.tm.get_state())

        for class_i in range(n_classes):
            key = str(class_i)
            if key not in imported:
                continue
            imported_decoded = np.array(imported[key], dtype=np.int32)
            cw, ta_packed = full_state[class_i]
            decoded = self._decode_ta_states(ta_packed, n_clauses, n_features,
                                             state_bits, ta_chunks)
            conf = self._clause_confidence(decoded, state_bits)
            weakest = np.argsort(conf)[:len(imported_decoded)]
            for i, ci in enumerate(weakest):
                decoded[ci] = imported_decoded[i]
            new_packed = self._encode_ta_states(decoded, n_clauses, n_features,
                                                state_bits, ta_chunks)
            full_state[class_i] = (cw, new_packed)

        tm_node.tm.set_state(full_state)
        return {
            "absorbed_from": packet.sender_id,
            "clauses_injected": self.top_k * n_classes,
        }
