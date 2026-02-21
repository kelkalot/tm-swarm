#!/usr/bin/env python3
"""
run_experiment.py — Collective TM learning on UNSW-NB15 network intrusion data.

Tests whether agents can detect attack types they've NEVER trained on,
purely by absorbing knowledge from peers who have seen those attacks.

Setup:
  Agent A trains on: normal + DoS, Fuzzers, Exploits
  Agent B trains on: normal + Backdoor, Shellcode, Reconnaissance
  Agent C trains on: normal + Worms, Analysis, Generic

After sharing, we evaluate ALL agents on ALL 9 attack types.
The key metric: detection rate on UNSEEN attack types.
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

AGENT_IDS = ["A", "B", "C"]
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


def evaluate_agent(tm, X_test, y_test, labels, agent_id, seen_attacks):
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
        print(f"    {GREEN}Seen attacks:{RESET}")
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
    """Generate class-balanced synthetic data via perturbation + rejection sampling.

    Instead of random binary vectors (which are too far from real data distribution),
    we perturb actual training data with small noise. Then rejection-sample to
    collect balanced class-0 and class-1 samples.
    """
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
    print(f"  UNSW-NB15 Collective Intrusion Detection Experiment")
    print(f"{'='*60}{RESET}\n")

    # Load test data
    X_test, y_test, labels = load_test_data()
    n_bits = X_test.shape[1]
    print(f"Test data: {len(X_test)} samples, {n_bits} features")
    print(f"  Normal: {sum(1 for l in labels if l == 'normal')}")
    for attack in sorted(set(labels)):
        if attack != "normal":
            n = sum(1 for l in labels if l == attack)
            print(f"  {attack}: {n}")

    # ── Phase 1: Train each agent on its local data ──────────────────────
    print(f"\n{BOLD}{'='*60}")
    print(f"  PHASE 1: LOCAL TRAINING")
    print(f"{'='*60}{RESET}\n")

    agents = {}
    agent_attacks = {}
    agent_X_train = {}
    rng = np.random.RandomState(42)

    for agent_id in AGENT_IDS:
        X_train, y_train, attacks = load_agent_data(agent_id)
        agent_attacks[agent_id] = attacks
        print(f"  Agent {agent_id}: training on {len(X_train)} samples "
              f"(attacks: {attacks})")

        tm = MultiClassTsetlinMachine(
            TM_CLAUSES, TM_T, TM_S, number_of_state_bits=8
        )

        t0 = time.time()
        for epoch in range(TRAIN_EPOCHS):
            tm.fit(X_train, y_train, epochs=1, incremental=True)
        dt = time.time() - t0
        print(f"    Trained {TRAIN_EPOCHS} epochs in {dt:.1f}s")

        agents[agent_id] = tm
        agent_X_train[agent_id] = X_train

    # ── Phase 2: Pre-share evaluation ────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print(f"  PHASE 2: PRE-SHARE EVALUATION")
    print(f"{'='*60}{RESET}")

    pre_results = {}
    for agent_id in AGENT_IDS:
        overall, seen, unseen = evaluate_agent(
            agents[agent_id], X_test, y_test, labels,
            agent_id, agent_attacks[agent_id]
        )
        pre_results[agent_id] = (overall, seen, unseen)
        print_eval(agent_id, overall, seen, unseen, "(pre-share)")

    # ── Phase 3: Knowledge sharing ───────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print(f"  PHASE 3: KNOWLEDGE SHARING")
    print(f"{'='*60}{RESET}\n")

    # Each agent generates class-balanced synthetic data via perturbation
    packets = {}
    for agent_id in AGENT_IDS:
        X_syn, y_syn = generate_synthetic(
            agents[agent_id], n_bits, N_SYNTHETIC, rng,
            X_train=agent_X_train[agent_id]
        )
        packets[agent_id] = (X_syn, y_syn)
        print(f"  Agent {agent_id}: generated {N_SYNTHETIC} synthetic samples "
              f"(attack ratio: {y_syn.mean():.3f})")

    # All-to-all sharing: each agent absorbs all peers' synthetic data
    for agent_id in AGENT_IDS:
        peer_X = []
        peer_y = []
        for peer_id in AGENT_IDS:
            if peer_id != agent_id:
                X_syn, y_syn = packets[peer_id]
                peer_X.append(X_syn)
                peer_y.append(y_syn)
                print(f"  Agent {agent_id} ← absorbing from {peer_id}")

        X_absorb = np.vstack(peer_X)
        y_absorb = np.concatenate(peer_y)

        # Retrain on absorbed data
        t0 = time.time()
        for epoch in range(ABSORB_EPOCHS):
            agents[agent_id].fit(X_absorb, y_absorb, epochs=1, incremental=True)
        dt = time.time() - t0
        print(f"  Agent {agent_id}: absorbed + retrained ({ABSORB_EPOCHS} epochs, {dt:.1f}s)")

    # ── Phase 4: Post-share evaluation ───────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print(f"  PHASE 4: POST-SHARE EVALUATION")
    print(f"{'='*60}{RESET}")

    post_results = {}
    for agent_id in AGENT_IDS:
        overall, seen, unseen = evaluate_agent(
            agents[agent_id], X_test, y_test, labels,
            agent_id, agent_attacks[agent_id]
        )
        post_results[agent_id] = (overall, seen, unseen)
        print_eval(agent_id, overall, seen, unseen, "(post-share)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print(f"  SUMMARY: IMPROVEMENT FROM SHARING")
    print(f"{'='*60}{RESET}\n")

    for agent_id in AGENT_IDS:
        pre_overall = pre_results[agent_id][0]
        post_overall = post_results[agent_id][0]
        pre_unseen = pre_results[agent_id][2]
        post_unseen = post_results[agent_id][2]

        print(f"  {BOLD}Agent {agent_id}{RESET} (trained on: {agent_attacks[agent_id]})")
        print(f"    Overall:  {pre_overall:.3f} → {post_overall:.3f} "
              f"({'+' if post_overall >= pre_overall else ''}{post_overall - pre_overall:.3f})")

        if pre_unseen:
            print(f"    {YELLOW}Unseen attack detection improvements:{RESET}")
            for attack in sorted(pre_unseen.keys()):
                pre_v = pre_unseen.get(attack, 0)
                post_v = post_unseen.get(attack, 0)
                delta = post_v - pre_v
                color = GREEN if delta > 0.05 else (YELLOW if delta > 0 else RED)
                sign = "+" if delta >= 0 else ""
                print(f"      {color}{attack:20s} {pre_v:.3f} → {post_v:.3f} "
                      f"({sign}{delta:.3f}){RESET}")
        print()

    # Average unseen detection rate
    pre_avg_unseen = np.mean([v for aid in AGENT_IDS
                              for v in pre_results[aid][2].values()])
    post_avg_unseen = np.mean([v for aid in AGENT_IDS
                               for v in post_results[aid][2].values()])
    delta = post_avg_unseen - pre_avg_unseen

    print(f"  {BOLD}Average unseen attack detection:{RESET}")
    print(f"    Pre-share:  {pre_avg_unseen:.3f}")
    print(f"    Post-share: {post_avg_unseen:.3f}")
    color = GREEN if delta > 0.05 else (YELLOW if delta > 0 else RED)
    print(f"    {color}Δ = {'+' if delta >= 0 else ''}{delta:.3f}{RESET}")
    print()


if __name__ == "__main__":
    main()
