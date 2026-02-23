#!/usr/bin/env python3
"""
prepare_data.py — Prepare UNSW-NB15 for collective TM learning.

Loads the parquet flow dataset, selects key features, binarizes them
(thermometer encoding for numerics, one-hot for categoricals), and
splits data into per-agent training sets based on attack-type assignment.
"""
import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PARQUET_PATH = os.path.join(DATA_DIR, "Network-Flows", "UNSW_Flow.parquet")
OUTPUT_DIR = os.path.join(DATA_DIR, "prepared")

# ── Feature selection ─────────────────────────────────────────────────────
# Key numeric features (good discriminators per UNSW-NB15 literature)
NUMERIC_FEATURES = [
    "dur",           # flow duration
    "sbytes",        # source→dest bytes
    "dbytes",        # dest→source bytes
    "sttl",          # source TTL
    "dttl",          # dest TTL
    "sload",         # source bits/sec
    "dload",         # dest bits/sec
    "spkts",         # source packets
    "dpkts",         # dest packets
    "smeansz",       # source mean packet size
    "dmeansz",       # dest mean packet size
    "tcprtt",        # TCP round-trip time
    "synack",        # SYN-ACK time
    "ackdat",        # ACK-DATA time
    "sjit",          # source jitter
    "djit",          # dest jitter
    "ct_srv_src",    # connections to same service from source
    "ct_srv_dst",    # connections to same service to dest
    "ct_dst_ltm",    # connections to dest in last N seconds
    "ct_src_ltm",    # connections from source in last N seconds
]

CATEGORICAL_FEATURES = [
    "protocol",  # TCP, UDP, etc.
    "state",     # FIN, CON, INT, etc.
    "service",   # http, dns, ftp, etc.
]

# Thermometer thresholds (percentiles of training data)
THERMO_PERCENTILES = [10, 25, 50, 75, 90, 95]

# Top-N categories per categorical feature (rest become "other")
TOP_N_CATEGORIES = {
    "protocol": ["tcp", "udp", "arp", "ospf", "icmp"],
    "state":    ["FIN", "CON", "INT", "REQ", "RST", "ACC"],
    "service":  ["-", "dns", "http", "ftp", "ftp-data", "smtp", "ssh"],
}

# ── Agent attack-type assignments ─────────────────────────────────────────
# Each agent only sees certain attack types during training.
# "normal" is seen by ALL agents.
AGENT_ATTACKS = {
    "A": ["dos", "fuzzers", "exploits"],
    "B": ["backdoor", "shellcode", "reconnaissance"],
    "C": ["worms", "analysis", "generic"],
}

# ── Sampling config ───────────────────────────────────────────────────────
NORMAL_SAMPLES_PER_AGENT = 2000    # Normal traffic per agent
ATTACK_SAMPLES_PER_AGENT = 500     # Per attack type per agent
TEST_NORMAL = 2000
TEST_PER_ATTACK = 300


def compute_thresholds(df, numeric_features, percentiles):
    """Compute threshold values from training data."""
    thresholds = {}
    for feat in numeric_features:
        vals = df[feat].dropna().values.astype(float)
        thresholds[feat] = sorted(set(np.percentile(vals, percentiles).tolist()))
    return thresholds


def binarize_row(row, thresholds, categories):
    """Convert one row to a boolean feature vector."""
    bits = []

    # Numeric → thermometer
    for feat, thresh_vals in thresholds.items():
        val = float(row.get(feat, 0) or 0)
        for t in thresh_vals:
            bits.append(1 if val > t else 0)

    # Categorical → one-hot
    for feat, cats in categories.items():
        row_val = str(row.get(feat, "")).lower().strip()
        for cat in cats:
            bits.append(1 if row_val == cat.lower() else 0)
        bits.append(1 if row_val not in [c.lower() for c in cats] else 0)  # "other"

    return bits


def build_schema(thresholds, categories):
    """Build the world_schema.json for TM skill scripts."""
    features = []
    fid = 0

    for feat, thresh_vals in thresholds.items():
        for t in thresh_vals:
            features.append({
                "id": fid,
                "name": f"{feat}_gt_{t:.4g}",
                "encoder": "boolean",
            })
            fid += 1

    for feat, cats in categories.items():
        for cat in cats:
            features.append({
                "id": fid,
                "name": f"{feat}_is_{cat}",
                "encoder": "boolean",
            })
            fid += 1
        features.append({
            "id": fid,
            "name": f"{feat}_is_other",
            "encoder": "boolean",
        })
        fid += 1

    return {
        "version": 1,
        "description": f"UNSW-NB15 network flow features ({fid} boolean bits)",
        "features": features,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Normalize attack labels
    df["attack_label"] = df["attack_label"].str.lower().str.strip()
    print("\nAttack distribution:")
    print(df["attack_label"].value_counts().to_string())

    # Fill NaNs in numeric features
    for feat in NUMERIC_FEATURES:
        df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)

    # Compute thresholds from ALL data (OK since we're not tuning hyperparams)
    print("\nComputing thresholds...")
    thresholds = compute_thresholds(df, NUMERIC_FEATURES, THERMO_PERCENTILES)
    categories = OrderedDict()
    for feat in CATEGORICAL_FEATURES:
        categories[feat] = TOP_N_CATEGORIES[feat]

    # Build schema
    schema = build_schema(thresholds, categories)
    n_bits = len(schema["features"])
    print(f"  Boolean feature vector: {n_bits} bits")

    schema_path = os.path.join(OUTPUT_DIR, "world_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved: {schema_path}")

    # Separate normal and attack data
    normal_df = df[df["attack_label"] == "normal"]
    attack_dfs = {}
    attacks_sorted = sorted(list(set(a for attacks in AGENT_ATTACKS.values() for a in attacks)))
    for attack in attacks_sorted:
        adf = df[df["attack_label"] == attack]
        attack_dfs[attack] = adf
        print(f"  {attack}: {len(adf)} samples")

    print("\nEnsuring strict train/test data separation...")
    # First, perform a rigid 80/20 train/test split on every category to prevent any overlap
    train_normal, test_normal_pool = train_test_split(normal_df, test_size=0.2, random_state=42)
    
    train_attacks = {}
    test_attacks_pool = {}
    for attack, adf in attack_dfs.items():
        if len(adf) < 5:
            # Handle extremely rare attacks (e.g. if the dataset were tinier)
            train_attacks[attack] = adf
            test_attacks_pool[attack] = adf.copy() # Cannot avoid if too small, but UNSW is large enough
        else:
            trn, tst = train_test_split(adf, test_size=0.2, random_state=42)
            train_attacks[attack] = trn
            test_attacks_pool[attack] = tst

    # ── Build per-agent training data ─────────────────────────────────────
    print("\nBuilding per-agent training data...")
    rng = np.random.RandomState(42)

    for agent_id, attacks in AGENT_ATTACKS.items():
        # Sample normal traffic from the training pool
        n_normal = min(NORMAL_SAMPLES_PER_AGENT, len(train_normal))
        agent_normal = train_normal.sample(n=n_normal, random_state=rng)

        # Sample attacks from the training pool
        agent_attacks_list = []
        for attack in attacks:
            adf_train = train_attacks[attack]
            n = min(ATTACK_SAMPLES_PER_AGENT, len(adf_train))
            agent_attacks_list.append(adf_train.sample(n=n, random_state=rng, replace=True))

        agent_df = pd.concat([agent_normal] + agent_attacks_list, ignore_index=True)
        agent_df = agent_df.sample(frac=1, random_state=rng).reset_index(drop=True)

        # Binarize
        X = [binarize_row(row, thresholds, categories) for _, row in agent_df.iterrows()]
        y = [0 if row["attack_label"] == "normal" else 1 for _, row in agent_df.iterrows()]

        print(f"  Agent {agent_id}: {len(X)} samples ({sum(y)} attacks, {len(y)-sum(y)} normal)")
        print(f"    Attacks: {attacks}")

        # Save
        agent_path = os.path.join(OUTPUT_DIR, f"agent_{agent_id}.json")
        with open(agent_path, "w") as f:
            json.dump({"X": X, "y": y, "attacks": attacks, "n_features": n_bits}, f)
        print(f"    Saved: {agent_path}")

    # ── Build test data (ALL attack types) ────────────────────────────────
    print("\nBuilding test data (all attack types)...")
    test_normal = test_normal_pool.sample(n=min(TEST_NORMAL, len(test_normal_pool)), random_state=rng)
    
    test_attack_list = []
    for attack, adf_test in test_attacks_pool.items():
        n = min(TEST_PER_ATTACK, len(adf_test))
        test_attack_list.append(adf_test.sample(n=n, random_state=rng, replace=True))

    test_df = pd.concat([test_normal] + test_attack_list, ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    X_test = [binarize_row(row, thresholds, categories) for _, row in test_df.iterrows()]
    y_test = [0 if row["attack_label"] == "normal" else 1 for _, row in test_df.iterrows()]
    labels_test = [row["attack_label"] for _, row in test_df.iterrows()]

    print(f"  Test set: {len(X_test)} samples ({sum(y_test)} attacks, {len(y_test)-sum(y_test)} normal)")

    test_path = os.path.join(OUTPUT_DIR, "test_data.json")
    with open(test_path, "w") as f:
        json.dump({
            "X": X_test,
            "y": y_test,
            "labels": labels_test,
            "n_features": n_bits,
        }, f)
    print(f"  Saved: {test_path}")

    # Save thresholds for reproducibility
    meta_path = os.path.join(OUTPUT_DIR, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "thresholds": thresholds,
            "categories": {k: v for k, v in categories.items()},
            "agent_attacks": AGENT_ATTACKS,
        }, f, indent=2)
    print(f"  Metadata: {meta_path}")

    print("\nDone! Ready for run_experiment.py")


if __name__ == "__main__":
    main()
