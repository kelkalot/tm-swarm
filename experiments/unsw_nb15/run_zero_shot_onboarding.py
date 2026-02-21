#!/usr/bin/env python3
"""
run_zero_shot_onboarding.py — Zero-Shot Onboarding Experiment

Tests whether a completely untrained agent (Agent D) can be onboarded 
by absorbing knowledge from a fully trained collective (Agent A + Agent B),
reaching high competency in a single sharing round without ever seeing
real training data.

Setup:
  Agent A trains on: normal + DoS, Fuzzers, Exploits
  Agent B trains on: normal + Backdoor, Shellcode, Reconnaissance
  Agent D starts blank (0 real training data)
  
  A and B generate synthetic knowledge packets.
  D absorbs A + B packets and trains on them.
  D is evaluated on the full test set.
"""
import os
import sys
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from pyTsetlinMachine.tm import MultiClassTsetlinMachine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")

# TM hyperparameters
TM_CLAUSES = 2000
TM_T = 50.0
TM_S = 10.0
TRAIN_EPOCHS = 100
ABSORB_EPOCHS = 100
N_SYNTHETIC = 5000

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def load_agent_data(agent_id):
    """Load prepared training data for one agent."""
    path = os.path.join(DATA_DIR, f"agent_{agent_id}.json")
    with open(path) as f:
        d = json.load(f)
    return np.array(d["X"], dtype=np.int32), np.array(d["y"], dtype=np.int32), d["attacks"]


def load_test_data():
    """Load test data with per-sample attack labels."""
    path = os.path.join(DATA_DIR, "test_data.json")
    with open(path) as f:
        d = json.load(f)
    return (
        np.array(d["X"], dtype=np.int32),
        np.array(d["y"], dtype=np.int32),
        d["labels"],
    )


def evaluate_agent(tm, X_test, y_test, labels, seen_attacks):
    """Evaluate agent: overall accuracy + per-attack-type recall."""
    preds = tm.predict(X_test)
    overall_acc = np.mean(preds == y_test)

    # Per-attack-type recall (attack = 1, so recall = fraction of attacks detected)
    attack_recalls = {}
    for attack_type in sorted(set(labels)):
        mask = np.array([l == attack_type for l in labels])
        if mask.sum() == 0:
            continue
        if attack_type == "normal":
            # For normal: specificity (true negative rate)
            attack_recalls[attack_type] = np.mean(preds[mask] == 0)
        else:
            attack_recalls[attack_type] = np.mean(preds[mask] == 1)

    # Split into seen vs unseen
    seen_recall = {k: v for k, v in attack_recalls.items()
                   if k in seen_attacks or k == "normal"}
    unseen_recall = {k: v for k, v in attack_recalls.items()
                     if k not in seen_attacks and k != "normal"}

    return overall_acc, seen_recall, unseen_recall


def print_eval(agent_id, overall, seen, unseen, label=""):
    """Pretty-print evaluation results."""
    print(f"\n  {BOLD}Agent {agent_id}{RESET} {label}")
    print(f"    Overall accuracy: {overall:.3f}")

    if seen:
        print(f"    {GREEN}Attacks (from incoming knowledge):{RESET}")
        for k, v in sorted(seen.items()):
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"      {k:20s} {bar} {v:.3f}")

    if unseen:
        print(f"    {YELLOW}Unseen attacks:{RESET}")
        for k, v in sorted(unseen.items()):
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            color = GREEN if v > 0.5 else (YELLOW if v > 0.2 else RED)
            print(f"      {color}{k:20s} {bar} {v:.3f}{RESET}")


def generate_synthetic(tm, n_bits, n_samples, rng, X_train=None, flip_rate=0.15):
    """Generate class-balanced synthetic data via perturbation + rejection sampling."""
    target_per_class = n_samples // 2
    class_0_X, class_1_X = [], []
    attempts = 0
    max_attempts = 50  # safety limit

    while (len(class_0_X) < target_per_class or len(class_1_X) < target_per_class) \
            and attempts < max_attempts:
        attempts += 1

        if X_train is not None and len(X_train) > 0:
            # Perturb real training data
            indices = rng.randint(0, len(X_train), size=n_samples)
            X_batch = X_train[indices].copy()
            # Flip ~flip_rate of bits randomly
            flip_mask = rng.random(X_batch.shape) < flip_rate
            X_batch = np.where(flip_mask, 1 - X_batch, X_batch).astype(np.int32)
        else:
            X_batch = rng.randint(0, 2, size=(n_samples, n_bits)).astype(np.int32)

        y_batch = tm.predict(X_batch)

        # Collect by class
        for cls, collector in [(0, class_0_X), (1, class_1_X)]:
            mask = y_batch == cls
            needed = target_per_class - len(collector)
            if needed > 0 and mask.sum() > 0:
                collector.append(X_batch[mask][:needed])

    # Build balanced dataset
    X_0 = np.vstack(class_0_X)[:target_per_class] if class_0_X else np.empty((0, n_bits), dtype=np.int32)
    X_1 = np.vstack(class_1_X)[:target_per_class] if class_1_X else np.empty((0, n_bits), dtype=np.int32)

    X_syn = np.vstack([X_0, X_1]) if len(X_0) > 0 and len(X_1) > 0 else \
            X_0 if len(X_1) == 0 else X_1
    y_syn = np.array([0] * len(X_0) + [1] * len(X_1), dtype=np.int32)

    # Shuffle
    perm = rng.permutation(len(X_syn))
    return X_syn[perm], y_syn[perm]


def main():
    print(f"\n{BOLD}{'='*60}")
    print(f"  Zero-Shot Onboarding Experiment on UNSW-NB15")
    print(f"{'='*60}{RESET}")
    print(f"  Agent A: DoS, Fuzzers, Exploits")
    print(f"  Agent B: Backdoor, Shellcode, Reconnaissance")
    print(f"  Agent D: starts BLANK (0 real training data)")
    print(f"  TM Clauses: {TM_CLAUSES} | Train Epochs: {TRAIN_EPOCHS} | Absorb epochs: {ABSORB_EPOCHS}\n")

    # 1. Load Data
    print(f"[{BOLD}1/4{RESET}] Loading data...")
    X_test, y_test, labels_test = load_test_data()
    X_a, y_a, attacks_a = load_agent_data("A")
    X_b, y_b, attacks_b = load_agent_data("B")
    n_features = X_test.shape[1]

    # 2. Train A and B
    print(f"\n[{BOLD}2/4{RESET}] Training base Collective (Agent A and Agent B)...")
    t0_train = time.time()
    
    print("  Training Agent A...")
    tm_a = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, boost_true_positive_feedback=1)
    tm_a.fit(X_a, y_a, epochs=TRAIN_EPOCHS)
    
    print("  Training Agent B...")
    tm_b = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, boost_true_positive_feedback=1)
    tm_b.fit(X_b, y_b, epochs=TRAIN_EPOCHS)
    
    print(f"  Training finished in {time.time()-t0_train:.1f}s")
    
    acc_a, seen_a, unseen_a = evaluate_agent(tm_a, X_test, y_test, labels_test, attacks_a)
    acc_b, seen_b, unseen_b = evaluate_agent(tm_b, X_test, y_test, labels_test, attacks_b)
    
    print_eval("A", acc_a, seen_a, None, "(Fully trained model)")
    print_eval("B", acc_b, seen_b, None, "(Fully trained model)")

    # 3. Generate Knowledge
    print(f"\n[{BOLD}3/4{RESET}] Generating Synthetic Knowledge Packets...")
    rng = np.random.RandomState(42)
    
    t0_gen = time.time()
    X_syn_a, y_syn_a = generate_synthetic(tm_a, n_features, N_SYNTHETIC, rng, X_train=X_a)
    print(f"  Agent A generated {len(X_syn_a)} synthetic samples")
    
    X_syn_b, y_syn_b = generate_synthetic(tm_b, n_features, N_SYNTHETIC, rng, X_train=X_b)
    print(f"  Agent B generated {len(X_syn_b)} synthetic samples")
    print(f"  Generation finished in {time.time()-t0_gen:.1f}s")

    # 4. Onboard Agent D
    print(f"\n[{BOLD}4/4{RESET}] Onboarding Agent D...")
    print("  Agent D is absorbing synthetic knowledge from A and B without any real data...")
    
    X_d_train = np.vstack([X_syn_a, X_syn_b])
    y_d_train = np.hstack([y_syn_a, y_syn_b])

    # Shuffle combined data
    perm = rng.permutation(len(X_d_train))
    X_d_train = X_d_train[perm]
    y_d_train = y_d_train[perm]

    t0_absorb = time.time()
    tm_d = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, boost_true_positive_feedback=1)
    tm_d.fit(X_d_train, y_d_train, epochs=ABSORB_EPOCHS)
    print(f"  Absorption finished in {time.time()-t0_absorb:.1f}s")

    # Evaluate D
    attacks_d = attacks_a + attacks_b
    acc_d, seen_d, unseen_d = evaluate_agent(tm_d, X_test, y_test, labels_test, attacks_d)
    
    print_eval("D", acc_d, seen_d, unseen_d, "(Zero-shot onboarded)")


if __name__ == "__main__":
    main()
