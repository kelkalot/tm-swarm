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
from tm_collective.strategies.sharing import SyntheticDataStrategy

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


class StandaloneAgent:
    """Wrapper to make standalone experiment code compatible with SyntheticDataStrategy."""
    def __init__(self, tm, X_train, y_train, schema_mock, agent_id):
        self.tm = tm
        self.X_buffer = [X_train]
        self.y_buffer = [y_train]
        self.X_own_buffer = [X_train]
        self.y_own_buffer = [y_train]
        self.schema = schema_mock
        self.agent_id = agent_id
        self.round_i = 1
        self.last_accuracy = 0.0
        self._fitted = True
        self.n_observations = len(X_train)

    def _reset_tm(self):
        """Recreate the TM to clear its state before retraining."""
        self.tm = MultiClassTsetlinMachine(
            TM_CLAUSES, TM_T, TM_S, number_of_state_bits=8
        )


def main():
    print(f"\n{BOLD}{'='*60}")
    print("  UNSW-NB15 Collective Intrusion Detection Experiment")
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
    print("  PHASE 1: LOCAL TRAINING")
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

        wrapper = StandaloneAgent(tm, X_train, y_train, type("MockSchema", (), {"n_binary": n_bits}), agent_id)
        agents[agent_id] = wrapper

    # ── Phase 2: Pre-share evaluation ────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  PHASE 2: PRE-SHARE EVALUATION")
    print(f"{'='*60}{RESET}")

    pre_results = {}
    for agent_id in AGENT_IDS:
        overall, seen, unseen = evaluate_agent(
            agents[agent_id].tm, X_test, y_test, labels,
            agent_id, agent_attacks[agent_id]
        )
        pre_results[agent_id] = (overall, seen, unseen)
        agents[agent_id].last_accuracy = overall
        print_eval(agent_id, overall, seen, unseen, "(pre-share)")

    # ── Phase 3: Knowledge sharing ───────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  PHASE 3: KNOWLEDGE SHARING")
    print(f"{'='*60}{RESET}\n")

    strategy = SyntheticDataStrategy(
        n_synthetic=N_SYNTHETIC, 
        retrain_epochs=ABSORB_EPOCHS,
        mode="perturb",
        rate_mode="graduated",
    )

    packets = {}
    for agent_id in AGENT_IDS:
        packet = strategy.generate(agents[agent_id])
        packets[agent_id] = packet
        print(f"  Agent {agent_id}: generated {len(packet.X)} synthetic samples "
              f"(attack ratio: {packet.y.mean():.3f})")

    # All-to-all sharing: each agent absorbs all peers' synthetic data
    for agent_id in AGENT_IDS:
        for peer_id in AGENT_IDS:
            if peer_id != agent_id:
                print(f"  Agent {agent_id} ← absorbing from {peer_id}")
                t0 = time.time()
                result = strategy.absorb(agents[agent_id], packets[peer_id])
                dt = time.time() - t0
                print(f"    Absorbed + retrained ({ABSORB_EPOCHS} epochs, {dt:.1f}s)")


    # ── Phase 4: Post-share evaluation ───────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  PHASE 4: POST-SHARE EVALUATION")
    print(f"{'='*60}{RESET}")

    post_results = {}
    for agent_id in AGENT_IDS:
        overall, seen, unseen = evaluate_agent(
            agents[agent_id].tm, X_test, y_test, labels,
            agent_id, agent_attacks[agent_id]
        )
        post_results[agent_id] = (overall, seen, unseen)
        print_eval(agent_id, overall, seen, unseen, "(post-share)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  SUMMARY: IMPROVEMENT FROM SHARING")
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
