# tm_collective/world_schema.py
"""
WorldSchema: defines the shared binary feature space all agents use.

Supported encoders:
  boolean     — single bit: int(bool(value))
  thermometer — N bits, one per threshold: [int(value > t) for t in thresholds]
  onehot      — one-hot over a vocabulary list

All agents must load the same schema file. This is the only shared contract.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class WorldSchema:
    """
    Loads a WORLD_SCHEMA.json and provides encode methods.

    Schema format:
      {
        "version": 1,
        "description": "optional human-readable description",
        "features": [
          {"id": 0, "name": "temp",   "encoder": "thermometer", "thresholds": [10,20,30,40]},
          {"id": 1, "name": "motion", "encoder": "boolean"},
          {"id": 2, "name": "status", "encoder": "onehot", "vocab": ["low","med","high"]}
        ]
      }

    Attributes:
        features:   list of feature dicts from the JSON
        n_binary:   total number of binary output dimensions
        _col_ranges: dict mapping feature name → (start_col, end_col) in binary vector
    """

    def __init__(self, schema_dict: dict):
        self.features = schema_dict["features"]
        self.version = schema_dict.get("version", 1)
        self.description = schema_dict.get("description", "")
        self.n_binary, self._col_ranges = self._build_index()

    @classmethod
    def from_file(cls, path: str | Path) -> "WorldSchema":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "WorldSchema":
        return cls(d)

    def _build_index(self) -> tuple[int, dict]:
        """
        Compute total binary width and a column-range index per feature name.
        Returns (n_binary, {name: (start, end)}).
        """
        col = 0
        ranges = {}
        for f in self.features:
            enc = f["encoder"]
            if enc == "thermometer":
                n = len(f["thresholds"])
            elif enc == "boolean":
                n = 1
            elif enc == "onehot":
                n = len(f["vocab"])
            else:
                raise ValueError(f"Unknown encoder '{enc}' for feature '{f['name']}'")
            ranges[f["name"]] = (col, col + n)
            col += n
        return col, ranges

    def encode_row(self, obs_dict: dict) -> np.ndarray:
        """
        Encode one observation dict → 1D uint32 array of length n_binary.

        Args:
            obs_dict: {feature_name: value, ...}
                      All feature names in the schema must be present.
        """
        bits = []
        for f in self.features:
            val = obs_dict[f["name"]]
            enc = f["encoder"]
            if enc == "thermometer":
                bits += [int(val > t) for t in f["thresholds"]]
            elif enc == "boolean":
                bits += [int(bool(val))]
            elif enc == "onehot":
                bits += [int(val == v) for v in f["vocab"]]
        return np.array(bits, dtype=np.uint32)

    def encode_batch(self, obs_list: list[dict]) -> np.ndarray:
        """
        Encode a list of observation dicts → 2D uint32 array (n, n_binary).
        """
        return np.vstack([self.encode_row(o) for o in obs_list]).astype(np.uint32)

    def columns_for_feature(self, feature_name: str) -> list[int]:
        """
        Return the list of binary column indices corresponding to a feature.
        Used by agents to apply noise only to their noisy-feature columns.
        """
        start, end = self._col_ranges[feature_name]
        return list(range(start, end))

    def columns_for_features(self, feature_names: list[str]) -> list[int]:
        """Return all binary column indices for a list of feature names."""
        cols = []
        for name in feature_names:
            cols.extend(self.columns_for_feature(name))
        return cols

    def __repr__(self) -> str:
        return (f"WorldSchema(n_features={len(self.features)}, "
                f"n_binary={self.n_binary}, "
                f"description='{self.description}')")
