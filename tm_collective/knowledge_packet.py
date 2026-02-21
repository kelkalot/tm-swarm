# tm_collective/knowledge_packet.py
"""
KnowledgePacket: a JSON-serializable container for agent knowledge.

When an agent calls `generate_synthetic()`, it labels N random binary vectors
using its trained TM. Those (X, y_pred) pairs encode the agent's learned
decision boundary. Any agent can absorb them as additional training data.
"""

from __future__ import annotations
import json
import numpy as np


class KnowledgePacket:
    """
    Args:
        sender_id:       agent ID string
        round_i:         which observation round this was generated at
        X:               (n, n_binary) uint32 synthetic feature vectors
        y:               (n,) uint32 predicted labels (not ground truth)
        metadata:        any extra info: accuracy_at_share, schema_version, etc.
    """

    def __init__(
        self,
        sender_id: str,
        round_i: int,
        X: np.ndarray,
        y: np.ndarray,
        metadata: dict | None = None,
    ):
        self.sender_id = sender_id
        self.round_i = round_i
        self.X = X.astype(np.uint32)
        self.y = y.astype(np.uint32)
        self.metadata = metadata or {}

    def to_json(self) -> str:
        """Serialize to a JSON string suitable for transport."""
        return json.dumps({
            "sender_id": self.sender_id,
            "round_i": self.round_i,
            "X": self.X.tolist(),
            "y": self.y.tolist(),
            "metadata": self.metadata,
        })

    def to_dict(self) -> dict:
        return json.loads(self.to_json())

    @classmethod
    def from_json(cls, s: str) -> "KnowledgePacket":
        d = json.loads(s)
        return cls(
            sender_id=d["sender_id"],
            round_i=d["round_i"],
            X=np.array(d["X"], dtype=np.uint32),
            y=np.array(d["y"], dtype=np.uint32),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgePacket":
        return cls.from_json(json.dumps(d))

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        return (f"KnowledgePacket(from={self.sender_id}, "
                f"round={self.round_i}, n={len(self.X)})")
