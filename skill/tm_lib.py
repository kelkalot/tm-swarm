# tm_lib.py
"""
Shared helpers for the tm_world_model skill.
All scripts import this file. Do not call directly.

State persistence:
  TM state   → workspace/.tm_{agent_id}_state.json
  Buffer     → workspace/.tm_{agent_id}_buffer.json

Packet format (for sessions_send):
  X is stored as a list of bit-strings ("101001...") for compact transport (~8KB/500 samples).
  y is stored as a single bit-string.
"""

from __future__ import annotations
import json
import os
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
TM_N_CLAUSES = 80
TM_T         = 20
TM_S         = 3.9

# ── Path helpers ──────────────────────────────────────────────────────────────

def state_path(workspace: str, agent_id: str) -> str:
    return os.path.join(workspace, f".tm_{agent_id}_state.json")

def buffer_path(workspace: str, agent_id: str) -> str:
    return os.path.join(workspace, f".tm_{agent_id}_buffer.json")

def schema_path(workspace: str) -> str:
    return os.path.join(workspace, "world_schema.json")

# ── World schema ──────────────────────────────────────────────────────────────

def load_schema(workspace: str) -> dict:
    p = schema_path(workspace)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"world_schema.json not found at {p}. "
            "Create it before running any tm_* scripts."
        )
    with open(p) as f:
        return json.load(f)

def n_binary_features(schema: dict) -> int:
    total = 0
    for f in schema["features"]:
        enc = f["encoder"]
        if enc == "thermometer":
            total += len(f["thresholds"])
        elif enc == "boolean":
            total += 1
        elif enc == "onehot":
            total += len(f["vocab"])
    return total

def encode_observation(obs: dict, schema: dict) -> list[int]:
    bits = []
    for f in schema["features"]:
        val = obs[f["name"]]
        enc = f["encoder"]
        if enc == "thermometer":
            bits += [int(val > t) for t in f["thresholds"]]
        elif enc == "boolean":
            bits += [int(bool(val))]
        elif enc == "onehot":
            bits += [int(val == v) for v in f["vocab"]]
    return bits

def encode_batch(obs_list: list[dict], schema: dict) -> np.ndarray:
    return np.array([encode_observation(o, schema) for o in obs_list], dtype=np.uint32)

# ── TM state persistence ──────────────────────────────────────────────────────

def save_tm(tm, workspace: str, agent_id: str):
    state = tm.get_state()
    d = {
        "n_clauses": tm.number_of_clauses,
        "n_state_bits": tm.number_of_state_bits,
        "n_classes": len(state),
        "state": [(cw.tolist(), ta.tolist()) for cw, ta in state],
    }
    with open(state_path(workspace, agent_id), "w") as f:
        json.dump(d, f)

def load_or_create_tm(workspace: str, agent_id: str, n_features: int):
    """
    Load TM from saved state, or create a fresh one.
    Returns (tm, is_fitted: bool).
    """
    from pyTsetlinMachine.tm import MultiClassTsetlinMachine
    tm = MultiClassTsetlinMachine(TM_N_CLAUSES, TM_T, TM_S, boost_true_positive_feedback=1)
    p = state_path(workspace, agent_id)
    if not os.path.exists(p):
        return tm, False
    with open(p) as f:
        d = json.load(f)
    # Initialize TM structure to match saved state's class count before set_state
    n_classes = d.get("n_classes", len(d["state"]))
    X_dummy = np.zeros((n_classes, n_features), dtype=np.uint32)
    y_dummy = np.arange(n_classes, dtype=np.uint32)
    tm.fit(X_dummy, y_dummy, epochs=1)
    state = [
        (np.array(cw, dtype=np.uint32), np.array(ta, dtype=np.uint32))
        for cw, ta in d["state"]
    ]
    tm.set_state(state)
    return tm, True

# ── Training buffer persistence ───────────────────────────────────────────────

def load_buffer(workspace: str, agent_id: str) -> tuple[list, list]:
    p = buffer_path(workspace, agent_id)
    if not os.path.exists(p):
        return [], []
    with open(p) as f:
        d = json.load(f)
    return d["X"], d["y"]

def save_buffer(workspace: str, agent_id: str, X_list: list, y_list: list):
    with open(buffer_path(workspace, agent_id), "w") as f:
        json.dump({"X": X_list, "y": y_list}, f)

# ── Packet encoding (compact bit-string format for sessions_send) ─────────────

def pack_packet(sender_id: str, X: np.ndarray, y: np.ndarray, metadata: dict = None) -> str:
    """
    Encode a knowledge packet as a compact JSON string suitable for sessions_send.
    X stored as list of bit-strings (e.g. "101001"), y as one bit-string.
    ~8KB for 500 samples × 12 features.
    """
    return json.dumps({
        "v": 1,
        "type": "tm_knowledge_packet",
        "sender": sender_id,
        "X": ["".join(map(str, row)) for row in X.tolist()],
        "y": "".join(map(str, y.tolist())),
        "meta": metadata or {},
    })

def unpack_packet(packet_json: str, n_features: int) -> tuple[str, np.ndarray, np.ndarray, dict]:
    """
    Decode a knowledge packet JSON string.
    Returns (sender_id, X, y, metadata).
    """
    d = json.loads(packet_json)
    X = np.array([[int(b) for b in row] for row in d["X"]], dtype=np.uint32)
    y = np.array([int(b) for b in d["y"]], dtype=np.uint32)
    return d["sender"], X, y, d.get("meta", {})

# ── Noise application ─────────────────────────────────────────────────────────

def apply_noise(X: np.ndarray, noisy_cols: list[int], rate: float, seed: int = None) -> np.ndarray:
    if not noisy_cols or rate == 0.0:
        return X.copy()
    rng = np.random.RandomState(seed) if seed is not None else np.random
    Xn = X.copy()
    mask = rng.random((len(X), len(noisy_cols))) < rate
    Xn[:, noisy_cols] = np.where(mask, 1 - X[:, noisy_cols], X[:, noisy_cols])
    return Xn

def result_json(ok: bool, **kwargs) -> str:
    return json.dumps({"ok": ok, **kwargs})
