#!/usr/bin/env python3
"""
prepare_data.py — Prepare CIC-IDS2017 + CSE-CIC-IDS2018 for cross-environment TM experiment.

Loads both datasets (separate CSV files), aligns column names, normalizes labels,
binarizes features (thresholds computed from 2017 ONLY), and produces:
  - train_agent_{A,B,C}.json  (from CIC-IDS2017)
  - test_within.json          (holdout from CIC-IDS2017)
  - test_cross.json           (from CSE-CIC-IDS2018)
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_2017_DIR = os.path.join(SCRIPT_DIR, "..", "..", "experiments_cic", "data", "2017")
DATA_2018_DIR = os.path.join(SCRIPT_DIR, "..", "..", "experiments_cic", "data", "2018")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")

# ── Column name mapping ──────────────────────────────────────────────────
# CIC-IDS2017 and CSE-CIC-IDS2018 use different column names for the same
# CICFlowMeter features. We map both to a common internal name.

# (internal_name, 2017_column_name, 2018_column_name)
COLUMN_MAP = [
    ("flow_duration",             "Flow Duration",                "Flow Duration"),
    ("total_fwd_packets",         "Total Fwd Packets",            "Tot Fwd Pkts"),
    ("total_bwd_packets",         "Total Backward Packets",       "Tot Bwd Pkts"),
    ("total_length_fwd_packets",  "Total Length of Fwd Packets",  "TotLen Fwd Pkts"),
    ("total_length_bwd_packets",  "Total Length of Bwd Packets",  "TotLen Bwd Pkts"),
    ("fwd_packet_length_max",     "Fwd Packet Length Max",        "Fwd Pkt Len Max"),
    ("fwd_packet_length_mean",    "Fwd Packet Length Mean",       "Fwd Pkt Len Mean"),
    ("bwd_packet_length_max",     "Bwd Packet Length Max",        "Bwd Pkt Len Max"),
    ("bwd_packet_length_mean",    "Bwd Packet Length Mean",       "Bwd Pkt Len Mean"),
    ("flow_bytes_per_s",          "Flow Bytes/s",                 "Flow Byts/s"),
    ("flow_packets_per_s",        "Flow Packets/s",               "Flow Pkts/s"),
    ("flow_iat_mean",             "Flow IAT Mean",                "Flow IAT Mean"),
    ("flow_iat_max",              "Flow IAT Max",                 "Flow IAT Max"),
    ("fwd_iat_mean",              "Fwd IAT Mean",                 "Fwd IAT Mean"),
    ("bwd_iat_mean",              "Bwd IAT Mean",                 "Bwd IAT Mean"),
    ("fwd_psh_flags",             "Fwd PSH Flags",                "Fwd PSH Flags"),
    ("fwd_urg_flags",             "Fwd URG Flags",                "Fwd URG Flags"),
    ("fin_flag_count",            "FIN Flag Count",               "FIN Flag Cnt"),
    ("syn_flag_count",            "SYN Flag Count",               "SYN Flag Cnt"),
    ("rst_flag_count",            "RST Flag Count",               "RST Flag Cnt"),
    ("psh_flag_count",            "PSH Flag Count",               "PSH Flag Cnt"),
    ("ack_flag_count",            "ACK Flag Count",               "ACK Flag Cnt"),
    ("avg_fwd_segment_size",      "Avg Fwd Segment Size",         "Fwd Seg Size Avg"),
    ("avg_bwd_segment_size",      "Avg Bwd Segment Size",         "Bwd Seg Size Avg"),
    ("init_win_bytes_forward",    "Init_Win_bytes_forward",       "Init Fwd Win Byts"),
    ("init_win_bytes_backward",   "Init_Win_bytes_backward",      "Init Bwd Win Byts"),
]

NUMERIC_FEATURES = [c[0] for c in COLUMN_MAP]

# ── Label normalization ───────────────────────────────────────────────────
# Use a function-based approach to handle encoding variants (the 2017 CSVs
# contain en-dash characters that decode differently depending on encoding).

def normalize_label_2017(raw):
    """Normalize CIC-IDS2017 label string to canonical form."""
    s = raw.strip().lower()
    if s == "benign":
        return "normal"
    if s == "ddos":
        return "ddos"
    if s == "dos hulk":
        return "dos_hulk"
    if s == "dos goldeneye":
        return "dos_goldeneye"
    if s == "dos slowloris":
        return "dos_slowloris"
    if s == "dos slowhttptest":
        return "dos_slowhttptest"
    if s == "bot":
        return "bot"
    if s == "portscan":
        return "portscan"
    if s == "infiltration":
        return "infiltration"
    # Web attacks: handle encoding variants by substring match
    if "brute force" in s:
        return "web_bruteforce"
    if "xss" in s:
        return "web_xss"
    if "sql injection" in s:
        return "web_sqli"
    # Excluded attacks (only in 2017)
    if "patator" in s or "heartbleed" in s:
        return None
    return None  # unknown label → will be dropped


def normalize_label_2018(raw):
    """Normalize CSE-CIC-IDS2018 label string to canonical form."""
    s = raw.strip().lower()
    if s == "benign":
        return "normal"
    if "ddos" in s or "hoic" in s or "loic" in s:
        return "ddos"
    if "hulk" in s:
        return "dos_hulk"
    if "goldeneye" in s:
        return "dos_goldeneye"
    if "slowloris" in s:
        return "dos_slowloris"
    if "slowhttptest" in s:
        return "dos_slowhttptest"
    if s == "bot":
        return "bot"
    if "brute force -web" in s:
        return "web_bruteforce"
    if "brute force -xss" in s or s == "xss":
        return "web_xss"
    if "sql injection" in s:
        return "web_sqli"
    if "infilter" in s or "infiltrat" in s:
        return "infiltration"
    # Excluded attacks (only in 2018) or header rows
    if "bruteforce" in s or s == "label":
        return None
    return None


# ── Agent attack-type assignments ─────────────────────────────────────────
AGENT_ATTACKS = {
    "A": ["ddos", "dos_hulk", "dos_goldeneye"],                  # Volumetric / flood
    "B": ["dos_slowloris", "dos_slowhttptest", "bot"],           # Slow & stealthy
    "C": ["portscan", "web_bruteforce", "web_xss", "web_sqli", "infiltration"],  # Recon / web
}

ALL_ATTACKS = sorted(set(a for attacks in AGENT_ATTACKS.values() for a in attacks))

# ── Sampling config ───────────────────────────────────────────────────────
NORMAL_SAMPLES_PER_AGENT = 2000
ATTACK_SAMPLES_PER_AGENT = 500
TEST_NORMAL = 2000
TEST_PER_ATTACK = 300

# Thermometer thresholds (percentiles of 2017 training data)
THERMO_PERCENTILES = [10, 25, 50, 75, 90, 95]


def load_dataset(data_dir, col_map_idx, normalize_fn, dataset_name):
    """Load all CSVs from a directory, rename columns, normalize labels.

    Args:
        data_dir: path to directory with CSVs
        col_map_idx: 1 for 2017 column names, 2 for 2018 column names
        normalize_fn: callable(raw_label) → normalized label or None
        dataset_name: for logging
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"\nLoading {dataset_name} ({len(csv_files)} files)...")

    frames = []
    for fpath in csv_files:
        print(f"  Reading {os.path.basename(fpath)}...", end=" ")
        try:
            df = pd.read_csv(fpath, low_memory=False, encoding='latin-1')
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        # Rename columns to internal names
        rename = {}
        for internal, c2017, c2018 in COLUMN_MAP:
            src = c2017 if col_map_idx == 1 else c2018
            if src in df.columns:
                rename[src] = internal

        # Also rename label column
        if "Label" in df.columns:
            rename["Label"] = "_label"
        elif " Label" in df.columns:
            rename[" Label"] = "_label"

        df = df.rename(columns=rename)

        # Keep only the columns we need
        keep_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + ["_label"]
        missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
        if missing:
            print(f"  WARNING: missing columns: {missing}")
        df = df[[c for c in keep_cols if c in df.columns]]

        # Normalize labels using the provided function
        if "_label" in df.columns:
            df["_label"] = df["_label"].astype(str).str.strip()
            df["attack_label"] = df["_label"].apply(normalize_fn)
            # Drop rows with unknown/excluded labels
            before = len(df)
            df = df.dropna(subset=["attack_label"])
            dropped = before - len(df)
            if dropped > 0:
                print(f"({dropped} excluded)", end=" ")
            df = df.drop(columns=["_label"])

        print(f"{len(df)} rows")
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    print(f"  Total {dataset_name}: {len(df_all)} rows")
    return df_all


def compute_thresholds(df, numeric_features, percentiles):
    """Compute threshold values from training data."""
    thresholds = {}
    for feat in numeric_features:
        if feat not in df.columns:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce").dropna().values.astype(float)
        # Replace inf with finite values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            thresholds[feat] = [0.0]
            continue
        thresholds[feat] = sorted(set(np.percentile(vals, percentiles).tolist()))
    return thresholds


def binarize_df(df, thresholds):
    """Convert dataframe rows to binary feature vectors using thermometer encoding."""
    n = len(df)
    n_bits = sum(len(t) for t in thresholds.values())
    X = np.zeros((n, n_bits), dtype=np.int32)

    col_idx = 0
    for feat, thresh_vals in thresholds.items():
        vals = pd.to_numeric(df[feat], errors="coerce").fillna(0).values.astype(float)
        vals = np.where(np.isfinite(vals), vals, 0)
        for t in thresh_vals:
            X[:, col_idx] = (vals > t).astype(np.int32)
            col_idx += 1

    return X


def build_schema(thresholds):
    """Build the world_schema.json for TM."""
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
    return {
        "version": 1,
        "description": f"CIC cross-env features ({fid} boolean bits)",
        "features": features,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load both datasets ────────────────────────────────────────────────
    df_2017 = load_dataset(DATA_2017_DIR, 1, normalize_label_2017, "CIC-IDS2017")
    df_2018 = load_dataset(DATA_2018_DIR, 2, normalize_label_2018, "CSE-CIC-IDS2018")

    print("\n── Label distribution (2017) ──")
    print(df_2017["attack_label"].value_counts().to_string())
    print("\n── Label distribution (2018) ──")
    print(df_2018["attack_label"].value_counts().to_string())

    # Clean numeric features
    for feat in NUMERIC_FEATURES:
        for df in [df_2017, df_2018]:
            if feat in df.columns:
                df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)

    # ── Split 2017 into train (80%) and within-test (20%) ─────────────────
    print("\nSplitting 2017 data into train/test...")
    df_2017_train, df_2017_test = train_test_split(
        df_2017, test_size=0.2, random_state=42, stratify=df_2017["attack_label"]
    )
    print(f"  2017 train: {len(df_2017_train)}, 2017 test: {len(df_2017_test)}")

    # ── Compute thresholds from 2017 training data ONLY ───────────────────
    print("\nComputing thresholds from 2017 training data...")
    thresholds = compute_thresholds(df_2017_train, NUMERIC_FEATURES, THERMO_PERCENTILES)
    n_bits = sum(len(t) for t in thresholds.values())
    print(f"  Boolean feature vector: {n_bits} bits")

    # Save schema
    schema = build_schema(thresholds)
    schema_path = os.path.join(OUTPUT_DIR, "world_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved: {schema_path}")

    # ── Build per-agent training data (from 2017 train split) ─────────────
    print("\n── Building per-agent training data ──")
    rng = np.random.RandomState(42)

    normal_train = df_2017_train[df_2017_train["attack_label"] == "normal"]

    for agent_id, attacks in AGENT_ATTACKS.items():
        # Sample normal traffic
        n_normal = min(NORMAL_SAMPLES_PER_AGENT, len(normal_train))
        agent_normal = normal_train.sample(n=n_normal, random_state=rng)

        # Sample attack traffic
        agent_attack_frames = []
        for attack in attacks:
            adf = df_2017_train[df_2017_train["attack_label"] == attack]
            n = min(ATTACK_SAMPLES_PER_AGENT, len(adf))
            if n == 0:
                print(f"  WARNING: no samples for {attack} in 2017 train")
                continue
            agent_attack_frames.append(
                adf.sample(n=n, random_state=rng, replace=(n > len(adf)))
            )

        agent_df = pd.concat([agent_normal] + agent_attack_frames, ignore_index=True)
        agent_df = agent_df.sample(frac=1, random_state=rng).reset_index(drop=True)

        # Binarize
        X = binarize_df(agent_df, thresholds)
        y = (agent_df["attack_label"] != "normal").astype(int).values.tolist()
        labels = agent_df["attack_label"].tolist()

        print(f"  Agent {agent_id}: {len(X)} samples ({sum(y)} attacks, {len(y)-sum(y)} normal)")
        print(f"    Attacks: {attacks}")

        agent_path = os.path.join(OUTPUT_DIR, f"train_agent_{agent_id}.json")
        with open(agent_path, "w") as f:
            json.dump({
                "X": X.tolist(), "y": y, "labels": labels,
                "attacks": attacks, "n_features": n_bits,
            }, f)
        print(f"    Saved: {agent_path}")

    # ── Build within-environment test (from 2017 test split) ──────────────
    print("\n── Building within-env test data ──")
    test_within_normal = df_2017_test[df_2017_test["attack_label"] == "normal"]
    test_within_n = min(TEST_NORMAL, len(test_within_normal))
    test_within_parts = [test_within_normal.sample(n=test_within_n, random_state=rng)]

    for attack in ALL_ATTACKS:
        adf = df_2017_test[df_2017_test["attack_label"] == attack]
        n = min(TEST_PER_ATTACK, len(adf))
        if n == 0:
            print(f"  WARNING: no {attack} in 2017 test set")
            continue
        test_within_parts.append(adf.sample(n=n, random_state=rng, replace=(n > len(adf))))

    test_within_df = pd.concat(test_within_parts, ignore_index=True)
    test_within_df = test_within_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    X_within = binarize_df(test_within_df, thresholds)
    y_within = (test_within_df["attack_label"] != "normal").astype(int).values.tolist()
    labels_within = test_within_df["attack_label"].tolist()

    print(f"  Within test: {len(X_within)} samples "
          f"({sum(y_within)} attacks, {len(y_within)-sum(y_within)} normal)")

    with open(os.path.join(OUTPUT_DIR, "test_within.json"), "w") as f:
        json.dump({
            "X": X_within.tolist(), "y": y_within, "labels": labels_within,
            "n_features": n_bits,
        }, f)

    # ── Build cross-environment test (from CSE-CIC-IDS2018) ───────────────
    print("\n── Building cross-env test data ──")
    test_cross_normal = df_2018[df_2018["attack_label"] == "normal"]
    test_cross_n = min(TEST_NORMAL, len(test_cross_normal))
    test_cross_parts = [test_cross_normal.sample(n=test_cross_n, random_state=rng)]

    for attack in ALL_ATTACKS:
        adf = df_2018[df_2018["attack_label"] == attack]
        n = min(TEST_PER_ATTACK, len(adf))
        if n == 0:
            print(f"  WARNING: no {attack} in 2018")
            continue
        test_cross_parts.append(adf.sample(n=n, random_state=rng, replace=(n > len(adf))))

    test_cross_df = pd.concat(test_cross_parts, ignore_index=True)
    test_cross_df = test_cross_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    X_cross = binarize_df(test_cross_df, thresholds)
    y_cross = (test_cross_df["attack_label"] != "normal").astype(int).values.tolist()
    labels_cross = test_cross_df["attack_label"].tolist()

    print(f"  Cross test: {len(X_cross)} samples "
          f"({sum(y_cross)} attacks, {len(y_cross)-sum(y_cross)} normal)")

    with open(os.path.join(OUTPUT_DIR, "test_cross.json"), "w") as f:
        json.dump({
            "X": X_cross.tolist(), "y": y_cross, "labels": labels_cross,
            "n_features": n_bits,
        }, f)

    # ── Save metadata ─────────────────────────────────────────────────────
    meta_path = os.path.join(OUTPUT_DIR, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "thresholds": {k: [float(v) for v in vs] for k, vs in thresholds.items()},
            "numeric_features": NUMERIC_FEATURES,
            "agent_attacks": AGENT_ATTACKS,
            "all_attacks": ALL_ATTACKS,
            "n_bits": n_bits,
            "n_2017_train": len(df_2017_train),
            "n_2017_test": len(df_2017_test),
            "n_2018": len(df_2018),
        }, f, indent=2)
    print(f"\n  Metadata: {meta_path}")

    print("\nDone! Ready for run_experiment.py")


if __name__ == "__main__":
    main()
