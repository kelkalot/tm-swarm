#!/usr/bin/env python3
"""
run_experiment.py — Cross-environment collective TM experiment.

5 conditions × 2 eval settings × N seeds:
  Conditions: baseline, synthetic, synthetic_graduated, clause, oracle
  Settings:   within-env (CIC-IDS2017 holdout), cross-env (CSE-CIC-IDS2018)

Tests the hypothesis: synthetic sharing degrades less than clause transfer
when moving from within-environment to cross-environment evaluation.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from tm_collective.strategies.sharing import SyntheticDataStrategy, ClauseTransferStrategy
from tm_collective.knowledge_packet import KnowledgePacket
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")

# TM hyperparameters (same as UNSW-NB15 experiment)
TM_CLAUSES = 2000
TM_T = 50.0
TM_S = 10.0
TRAIN_EPOCHS = 100
ABSORB_EPOCHS = 100
N_SYNTHETIC = 5000
CLAUSE_TOP_K = 20
N_SEEDS = 5

# Graduated perturbation rates: low rates capture specific patterns,
# high rates capture general decision boundary structure
GRADUATED_RATES = [0.05, 0.15, 0.30, 0.50]

AGENT_IDS = ["A", "B", "C"]
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def load_agent_data(agent_id):
    """Load prepared training data for one agent."""
    path = os.path.join(DATA_DIR, f"train_agent_{agent_id}.json")
    with open(path) as f:
        d = json.load(f)
    return (np.array(d["X"], dtype=np.int32),
            np.array(d["y"], dtype=np.int32),
            d["attacks"], d["labels"])


def load_test_data(setting):
    """Load test data for within-env or cross-env setting."""
    fname = "test_within.json" if setting == "within" else "test_cross.json"
    path = os.path.join(DATA_DIR, fname)
    with open(path) as f:
        d = json.load(f)
    return (np.array(d["X"], dtype=np.int32),
            np.array(d["y"], dtype=np.int32),
            d["labels"])


def evaluate_agent(tm, X_test, y_test, labels, seen_attacks):
    """Evaluate: overall accuracy + per-attack recall + seen/unseen split."""
    preds = tm.predict(X_test)
    overall_acc = float(np.mean(preds == y_test))

    attack_recalls = {}
    for attack_type in sorted(set(labels)):
        mask = np.array([l == attack_type for l in labels])
        if mask.sum() == 0:
            continue
        if attack_type == "normal":
            attack_recalls[attack_type] = float(np.mean(preds[mask] == 0))
        else:
            attack_recalls[attack_type] = float(np.mean(preds[mask] == 1))

    seen_recall = {k: v for k, v in attack_recalls.items()
                   if k in seen_attacks or k == "normal"}
    unseen_recall = {k: v for k, v in attack_recalls.items()
                     if k not in seen_attacks and k != "normal"}

    return overall_acc, attack_recalls, seen_recall, unseen_recall


class StandaloneAgent:
    """Wrapper to make standalone experiment code compatible with Sharing Strategies."""
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


def run_condition(condition, seed, eval_settings, agent_data, n_bits):
    """Run one condition with one seed and both eval settings. Returns results dict."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    results = {}

    if condition == "oracle":
        # Single TM trained on ALL agents' data combined
        X_all = np.vstack([d[0] for d in agent_data.values()])
        y_all = np.concatenate([d[1] for d in agent_data.values()])
        all_attacks = sorted(set(a for d in agent_data.values() for a in d[2]))

        tm = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, number_of_state_bits=8)
        for _ in range(TRAIN_EPOCHS):
            tm.fit(X_all, y_all, epochs=1, incremental=True)

        for setting, (X_test, y_test, labels) in eval_settings.items():
            overall, per_attack, seen, unseen = evaluate_agent(
                tm, X_test, y_test, labels, all_attacks
            )
            results[setting] = {
                "oracle": {
                    "overall": overall, "per_attack": per_attack,
                    "seen": seen, "unseen": unseen,
                }
            }
        return results

    # Train individual agents
    agents = {}
    agent_attacks = {}
    for agent_id, (X_train, y_train, attacks, _labels) in agent_data.items():
        tm = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, number_of_state_bits=8)
        for _ in range(TRAIN_EPOCHS):
            tm.fit(X_train, y_train, epochs=1, incremental=True)
        wrapper = StandaloneAgent(tm, X_train, y_train, type("MockSchema", (), {"n_binary": n_bits}), agent_id)
        agents[agent_id] = wrapper
        agent_attacks[agent_id] = attacks

    # Pre-share evaluation
    pre_results = {}
    for setting, (X_test, y_test, labels) in eval_settings.items():
        pre_results[setting] = {}
        for agent_id in AGENT_IDS:
            overall, per_attack, seen, unseen = evaluate_agent(
                agents[agent_id].tm, X_test, y_test, labels, agent_attacks[agent_id]
            )
            agents[agent_id].last_accuracy = overall
            pre_results[setting][agent_id] = {
                "overall": overall, "per_attack": per_attack,
                "seen": seen, "unseen": unseen,
            }

    if condition == "baseline":
        return pre_results

    # Knowledge sharing
    if condition in ("synthetic", "synthetic_graduated",
                      "synthetic_attackonly", "synthetic_graduated_attackonly",
                      "synthetic_hybrid", "synthetic_graduated_hybrid"):
                      
        is_graduated = "graduated" in condition
        is_attackonly = "attackonly" in condition
        is_hybrid = "hybrid" in condition

        # Map to framework config
        rate_mode = "graduated" if is_graduated else "fixed"
        
        # We handle hybrid/attackonly customly through the framework
        # If hybrid, let framework handle it perfectly.
        # If attackonly or baseline, we will do manual absorption for ablation.
        absorption_mode = "hybrid" if is_hybrid else "full"
        
        # We need the random state set globally for the framework's numpy calls during generation
        np.random.seed(seed)
        
        # Instantiate strategy
        strategy = SyntheticDataStrategy(
            n_synthetic=N_SYNTHETIC, 
            retrain_epochs=ABSORB_EPOCHS,
            mode="perturb",
            rate_mode=rate_mode,
            absorption=absorption_mode
        )

        # Generate synthetic data from each agent
        packets = {}
        for agent_id in AGENT_IDS:
            # Setting seed ensures reproducibility within the framework's generator calls
            packet = strategy.generate(agents[agent_id])
            packets[agent_id] = packet

        # All-to-all sharing
        for agent_id in AGENT_IDS:
            if is_hybrid or (not is_attackonly and not is_hybrid):
                # We can use the framework's native absorb for hybrid and full modes
                for peer_id in AGENT_IDS:
                    if peer_id != agent_id:
                        strategy.absorb(agents[agent_id], packets[peer_id])
            else:
                # Manual absorption for attack-only ablation
                peer_X = []
                peer_y = []
                for peer_id in AGENT_IDS:
                    if peer_id != agent_id:
                        packet = packets[peer_id]
                        attack_mask = packet.y == 1
                        peer_X.append(packet.X[attack_mask])
                        peer_y.append(packet.y[attack_mask])
                
                X_peer = np.vstack(peer_X)
                y_peer = np.concatenate(peer_y)
                
                # Combine peer attacks with receiver's own training data (no local normals)
                X_own, y_own = agent_data[agent_id][0], agent_data[agent_id][1]
                X_absorb = np.vstack([X_own, X_peer])
                y_absorb = np.concatenate([y_own, y_peer])
                
                agents[agent_id]._reset_tm()
                agents[agent_id].tm.fit(X_absorb, y_absorb, epochs=ABSORB_EPOCHS)
                # Keep buffers updated for consistency even though script doesn't use them further
                agents[agent_id].X_buffer = [X_absorb]
                agents[agent_id].y_buffer = [y_absorb]

    elif condition == "clause":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Extract top-K clauses from each agent
            strategy = ClauseTransferStrategy(top_k=CLAUSE_TOP_K)
            packets = {}
            for agent_id in AGENT_IDS:
                packets[agent_id] = strategy.generate(agents[agent_id])

            # All-to-all injection
            for agent_id in AGENT_IDS:
                for peer_id in AGENT_IDS:
                    if peer_id != agent_id:
                        strategy.absorb(agents[agent_id], packets[peer_id])


    # Post-share evaluation
    post_results = {}
    for setting, (X_test, y_test, labels) in eval_settings.items():
        post_results[setting] = {}
        for agent_id in AGENT_IDS:
            overall, per_attack, seen, unseen = evaluate_agent(
                agents[agent_id].tm, X_test, y_test, labels, agent_attacks[agent_id]
            )
            post_results[setting][agent_id] = {
                "overall": overall, "per_attack": per_attack,
                "seen": seen, "unseen": unseen,
            }

    return post_results


def main():
    print(f"\n{BOLD}{'='*70}")
    print("  CIC-IDS2017 → CSE-CIC-IDS2018 Cross-Environment Experiment")
    print(f"{'='*70}{RESET}\n")

    # Load test data for both settings
    eval_settings = {}
    for setting in ["within", "cross"]:
        X_test, y_test, labels = load_test_data(setting)
        eval_settings[setting] = (X_test, y_test, labels)
        n_attacks = sum(1 for l in labels if l != "normal")
        print(f"  {setting}-env test: {len(X_test)} samples "
              f"({n_attacks} attacks, {len(X_test)-n_attacks} normal)")

    n_bits = eval_settings["within"][0].shape[1]
    print(f"  Features: {n_bits} bits")

    # Load agent training data
    agent_data = {}
    for agent_id in AGENT_IDS:
        X, y, attacks, labels = load_agent_data(agent_id)
        agent_data[agent_id] = (X, y, attacks, labels)
        print(f"  Agent {agent_id}: {len(X)} training samples, attacks: {attacks}")

    conditions = ["baseline", "synthetic", "synthetic_graduated",
                   "synthetic_attackonly", "synthetic_graduated_attackonly",
                   "synthetic_hybrid", "synthetic_graduated_hybrid",
                   "clause", "oracle"]
    all_results = {}

    for condition in conditions:
        print(f"\n{BOLD}{'='*70}")
        print(f"  CONDITION: {condition.upper()}")
        print(f"{'='*70}{RESET}")

        condition_results = {}
        for seed in range(N_SEEDS):
            print(f"\n  --- Seed {seed} ---")
            t0 = time.time()
            result = run_condition(
                condition, seed, eval_settings, agent_data, n_bits
            )
            dt = time.time() - t0
            print(f"  Completed in {dt:.1f}s")

            condition_results[str(seed)] = result

            # Print summary for this seed
            for setting in ["within", "cross"]:
                if setting in result:
                    print(f"\n  [{setting}-env]")
                    for agent_id, r in result[setting].items():
                        unseen_avg = np.mean(list(r["unseen"].values())) if r["unseen"] else 0
                        print(f"    {agent_id}: overall={r['overall']:.3f}, "
                              f"unseen_avg={unseen_avg:.3f}")

        all_results[condition] = condition_results

    # ── Save results ──────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{BOLD}Results saved to: {RESULTS_PATH}{RESET}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*70}")
    print("  SUMMARY: Cross-Environment Degradation")
    print(f"{'='*70}{RESET}\n")

    for condition in ["baseline", "synthetic", "synthetic_graduated",
                      "synthetic_attackonly", "synthetic_graduated_attackonly",
                      "synthetic_hybrid", "synthetic_graduated_hybrid", "clause"]:
        # Compute per-seed average unseen detection, then mean ± std across seeds
        seed_within_avgs = []
        seed_cross_avgs = []
        for seed_results in all_results[condition].values():
            within_r = seed_results.get("within", {})
            cross_r = seed_results.get("cross", {})
            w_agent_means = []
            c_agent_means = []
            for agent_id in AGENT_IDS:
                if agent_id in within_r and within_r[agent_id]["unseen"]:
                    w_agent_means.append(np.mean(list(within_r[agent_id]["unseen"].values())))
                if agent_id in cross_r and cross_r[agent_id]["unseen"]:
                    c_agent_means.append(np.mean(list(cross_r[agent_id]["unseen"].values())))
            seed_within_avgs.append(np.mean(w_agent_means) if w_agent_means else 0)
            seed_cross_avgs.append(np.mean(c_agent_means) if c_agent_means else 0)

        w_arr = np.array(seed_within_avgs)
        c_arr = np.array(seed_cross_avgs)
        d_arr = w_arr - c_arr
        print(f"  {condition:20s}: within={w_arr.mean():.1%} ± {w_arr.std():.1%}, "
              f"cross={c_arr.mean():.1%} ± {c_arr.std():.1%}, "
              f"Δ={d_arr.mean():.1%} ± {d_arr.std():.1%}")

    print()


if __name__ == "__main__":
    main()
