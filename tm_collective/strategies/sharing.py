# tm_collective/strategies/sharing.py
"""
Sharing strategies: how agents export and absorb knowledge.

SyntheticDataStrategy (RECOMMENDED):
  - Generating agent labels N random inputs with its TM
  - Receiving agent adds those (X, y_pred) to its training buffer and retrains
  - Validated: 2-agent AND: 83% → 99% post-share

ClauseTransferStrategy (EXPERIMENTAL):
  - Directly injects TA states from confident clauses into peer's TM
  - Requires both agents to use identical TM hyperparameters
  - Can degrade accuracy; only use if you need interpretable clause inspection
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
    Generate synthetic (random input → TM prediction) pairs.
    Receiving agent trains on them as additional data.

    Args:
        n_synthetic:    number of synthetic samples to generate per share event
        retrain_epochs: epochs to train after absorption (more = better integration)
    """

    def __init__(self, n_synthetic: int = 500, retrain_epochs: int = 150):
        self.n_synthetic = n_synthetic
        self.retrain_epochs = retrain_epochs

    def generate(self, tm_node, n: int | None = None) -> KnowledgePacket:
        n = n or self.n_synthetic
        n_binary = tm_node.schema.n_binary

        if not tm_node._fitted:
            # Return random noise if not trained yet
            X_rand = np.random.randint(0, 2, (n, n_binary)).astype(np.uint32)
            y_rand = np.random.randint(0, 2, n).astype(np.uint32)
            return KnowledgePacket(tm_node.agent_id, tm_node.round_i, X_rand, y_rand,
                                   {"fitted": False})

        X_rand = np.random.randint(0, 2, (n, n_binary)).astype(np.uint32)
        y_pred = tm_node.tm.predict(X_rand).astype(np.uint32)

        meta = {
            "fitted": True,
            "accuracy_at_share": tm_node.last_accuracy,
            "n_train_samples": tm_node.n_observations,
        }
        return KnowledgePacket(tm_node.agent_id, tm_node.round_i, X_rand, y_pred, meta)

    def absorb(self, tm_node, packet: KnowledgePacket) -> dict:
        """
        Add packet's (X, y) to tm_node's buffer, reset TM, retrain from scratch.
        Resetting ensures the imported knowledge is fully integrated.
        """
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
            "packet_size": len(packet),
            "total_samples_now": len(X_all),
        }


class ClauseTransferStrategy(SharingStrategy):
    """
    EXPERIMENTAL. Directly injects confident TA states from source TM into target TM.

    Requirements:
        - Both agents must use identical TM hyperparameters (n_clauses, T, s, state_bits)
        - Both agents must observe the same WorldSchema (identical binary feature space)

    The strategy:
        1. Source: decode all TA states from bit-packed representation
        2. Source: score each clause by avg |state - midpoint| / midpoint (confidence)
        3. Source: export top-K most confident clauses per class
        4. Target: find its K least-confident clauses per class
        5. Target: overwrite them with the source's confident clauses
        6. Target: re-encode back to bit-packed format

    Use SyntheticDataStrategy unless you specifically need to inspect transferred clauses.
    """

    def __init__(self, top_k: int = 10):
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
