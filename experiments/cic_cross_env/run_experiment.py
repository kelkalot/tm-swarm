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


def generate_synthetic(tm, n_bits, n_samples, rng, X_train, flip_rate=0.15):
    """Generate class-balanced synthetic data via perturbation + rejection sampling."""
    target_per_class = n_samples // 2
    class_0_X, class_1_X = [], []
    attempts = 0
    max_attempts = 50

    while (len(class_0_X) < target_per_class or len(class_1_X) < target_per_class) \
            and attempts < max_attempts:
        attempts += 1
        indices = rng.randint(0, len(X_train), size=n_samples)
        X_batch = X_train[indices].copy()
        flip_mask = rng.random(X_batch.shape) < flip_rate
        X_batch = np.where(flip_mask, 1 - X_batch, X_batch).astype(np.int32)
        y_batch = tm.predict(X_batch)

        for cls, collector in [(0, class_0_X), (1, class_1_X)]:
            mask = y_batch == cls
            needed = target_per_class - len(collector)
            if needed > 0 and mask.sum() > 0:
                collector.append(X_batch[mask][:needed])

    X_0 = np.vstack(class_0_X)[:target_per_class] if class_0_X else np.empty((0, n_bits), dtype=np.int32)
    X_1 = np.vstack(class_1_X)[:target_per_class] if class_1_X else np.empty((0, n_bits), dtype=np.int32)
    X_syn = np.vstack([X_0, X_1]) if len(X_0) > 0 and len(X_1) > 0 else \
            X_0 if len(X_1) == 0 else X_1
    y_syn = np.array([0] * len(X_0) + [1] * len(X_1), dtype=np.int32)
    perm = rng.permutation(len(X_syn))
    return X_syn[perm], y_syn[perm]


def generate_synthetic_class(tm, n_bits, n_samples, rng, X_train, target_class, flip_rate=0.15):
    """Generate synthetic data for a single class via perturbation + rejection sampling."""
    collected = []
    for _ in range(50):
        if sum(len(c) for c in collected) >= n_samples:
            break
        indices = rng.randint(0, len(X_train), size=n_samples)
        X_batch = X_train[indices].copy()
        flip_mask = rng.random(X_batch.shape) < flip_rate
        X_batch = np.where(flip_mask, 1 - X_batch, X_batch).astype(np.int32)
        y_batch = tm.predict(X_batch)
        mask = y_batch == target_class
        needed = n_samples - sum(len(c) for c in collected)
        if needed > 0 and mask.sum() > 0:
            collected.append(X_batch[mask][:needed])
    if collected:
        X_out = np.vstack(collected)[:n_samples]
    else:
        X_out = np.empty((0, n_bits), dtype=np.int32)
    return X_out


# ── Clause transfer helpers (adapted from ClauseTransferStrategy) ─────────

def decode_ta_states(ta_packed, n_clauses, n_features, n_state_bits, n_ta_chunks):
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


def encode_ta_states(states, n_clauses, n_features, n_state_bits, n_ta_chunks):
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


def clause_confidence(decoded, n_state_bits):
    midpoint = 2 ** (n_state_bits - 1)
    return np.mean(np.abs(decoded.astype(float) - midpoint) / midpoint, axis=1)


def extract_top_clauses(tm, n_features, top_k):
    """Extract top-K most confident clauses per class."""
    n_clauses = tm.number_of_clauses
    state_bits = tm.number_of_state_bits
    ta_chunks = tm.number_of_ta_chunks
    n_classes = tm.number_of_classes

    top_clauses = {}
    full_state = tm.get_state()
    for class_i in range(n_classes):
        _, ta_packed = full_state[class_i]
        decoded = decode_ta_states(ta_packed, n_clauses, n_features, state_bits, ta_chunks)
        conf = clause_confidence(decoded, state_bits)
        top_idx = np.argsort(conf)[-top_k:]
        top_clauses[class_i] = decoded[top_idx]
    return top_clauses


def inject_clauses(tm, n_features, imported_clauses):
    """Inject imported clauses into TM, replacing least confident existing clauses."""
    n_clauses = tm.number_of_clauses
    state_bits = tm.number_of_state_bits
    ta_chunks = tm.number_of_ta_chunks
    n_classes = tm.number_of_classes

    full_state = list(tm.get_state())
    for class_i in range(n_classes):
        if class_i not in imported_clauses:
            continue
        imported_decoded = imported_clauses[class_i]
        cw, ta_packed = full_state[class_i]
        decoded = decode_ta_states(ta_packed, n_clauses, n_features, state_bits, ta_chunks)
        conf = clause_confidence(decoded, state_bits)
        weakest = np.argsort(conf)[:len(imported_decoded)]
        for i, ci in enumerate(weakest):
            decoded[ci] = imported_decoded[i]
        new_packed = encode_ta_states(decoded, n_clauses, n_features, state_bits, ta_chunks)
        full_state[class_i] = (cw, new_packed)
    tm.set_state(full_state)


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
        agents[agent_id] = tm
        agent_attacks[agent_id] = attacks

    # Pre-share evaluation
    pre_results = {}
    for setting, (X_test, y_test, labels) in eval_settings.items():
        pre_results[setting] = {}
        for agent_id in AGENT_IDS:
            overall, per_attack, seen, unseen = evaluate_agent(
                agents[agent_id], X_test, y_test, labels, agent_attacks[agent_id]
            )
            pre_results[setting][agent_id] = {
                "overall": overall, "per_attack": per_attack,
                "seen": seen, "unseen": unseen,
            }

    if condition == "baseline":
        return pre_results

    # Knowledge sharing
    if condition in ("synthetic", "synthetic_graduated",
                      "synthetic_attackonly", "synthetic_graduated_attackonly"):
        # Determine perturbation mode
        is_graduated = "graduated" in condition
        is_attackonly = "attackonly" in condition

        # Generate synthetic data from each agent
        packets = {}
        for agent_id in AGENT_IDS:
            if is_graduated:
                # Graduated: generate at multiple perturbation rates and combine
                samples_per_rate = N_SYNTHETIC // len(GRADUATED_RATES)
                rate_packets = []
                for rate in GRADUATED_RATES:
                    X_r, y_r = generate_synthetic(
                        agents[agent_id], n_bits, samples_per_rate, rng,
                        X_train=agent_data[agent_id][0], flip_rate=rate
                    )
                    rate_packets.append((X_r, y_r))
                X_syn = np.vstack([p[0] for p in rate_packets])
                y_syn = np.concatenate([p[1] for p in rate_packets])
                perm = rng.permutation(len(X_syn))
                X_syn, y_syn = X_syn[perm], y_syn[perm]
            else:
                # Fixed rate (original)
                X_syn, y_syn = generate_synthetic(
                    agents[agent_id], n_bits, N_SYNTHETIC, rng,
                    X_train=agent_data[agent_id][0]
                )
            packets[agent_id] = (X_syn, y_syn)

        # All-to-all sharing
        for agent_id in AGENT_IDS:
            peer_X = []
            peer_y = []
            for peer_id in AGENT_IDS:
                if peer_id != agent_id:
                    X_syn, y_syn = packets[peer_id]
                    if is_attackonly:
                        # Keep only attack-labeled samples (y=1)
                        attack_mask = y_syn == 1
                        peer_X.append(X_syn[attack_mask])
                        peer_y.append(y_syn[attack_mask])
                    else:
                        peer_X.append(X_syn)
                        peer_y.append(y_syn)

            X_peer = np.vstack(peer_X)
            y_peer = np.concatenate(peer_y)

            if is_attackonly:
                # Combine peer attacks with receiver's own training data
                X_own, y_own = agent_data[agent_id][0], agent_data[agent_id][1]
                X_absorb = np.vstack([X_own, X_peer])
                y_absorb = np.concatenate([y_own, y_peer])
            else:
                X_absorb = X_peer
                y_absorb = y_peer

            for _ in range(ABSORB_EPOCHS):
                agents[agent_id].fit(X_absorb, y_absorb, epochs=1, incremental=True)

    elif condition in ("synthetic_hybrid", "synthetic_graduated_hybrid"):
        is_graduated = "graduated" in condition

        # Generate synthetic data from each agent (same as before)
        packets = {}
        for agent_id in AGENT_IDS:
            if is_graduated:
                samples_per_rate = N_SYNTHETIC // len(GRADUATED_RATES)
                rate_packets = []
                for rate in GRADUATED_RATES:
                    X_r, y_r = generate_synthetic(
                        agents[agent_id], n_bits, samples_per_rate, rng,
                        X_train=agent_data[agent_id][0], flip_rate=rate
                    )
                    rate_packets.append((X_r, y_r))
                X_syn = np.vstack([p[0] for p in rate_packets])
                y_syn = np.concatenate([p[1] for p in rate_packets])
                perm = rng.permutation(len(X_syn))
                X_syn, y_syn = X_syn[perm], y_syn[perm]
            else:
                X_syn, y_syn = generate_synthetic(
                    agents[agent_id], n_bits, N_SYNTHETIC, rng,
                    X_train=agent_data[agent_id][0]
                )
            packets[agent_id] = (X_syn, y_syn)

        # Hybrid sharing: peer attacks + locally-generated normal
        for agent_id in AGENT_IDS:
            peer_attack_X = []
            for peer_id in AGENT_IDS:
                if peer_id != agent_id:
                    X_syn, y_syn = packets[peer_id]
                    attack_mask = y_syn == 1
                    peer_attack_X.append(X_syn[attack_mask])

            X_peer_attacks = np.vstack(peer_attack_X)
            y_peer_attacks = np.ones(len(X_peer_attacks), dtype=np.int32)

            # Generate normal samples locally from receiver's own TM
            # Use the same perturbation approach as the attack generation
            n_normal_target = len(X_peer_attacks)
            if is_graduated:
                # Match graduated rates used for attack generation
                normal_per_rate = n_normal_target // len(GRADUATED_RATES)
                normal_parts = []
                for rate in GRADUATED_RATES:
                    X_n = generate_synthetic_class(
                        agents[agent_id], n_bits, normal_per_rate, rng,
                        X_train=agent_data[agent_id][0], target_class=0,
                        flip_rate=rate
                    )
                    normal_parts.append(X_n)
                X_own_normal = np.vstack(normal_parts)[:n_normal_target]
            else:
                X_own_normal = generate_synthetic_class(
                    agents[agent_id], n_bits, n_normal_target, rng,
                    X_train=agent_data[agent_id][0], target_class=0,
                    flip_rate=0.15
                )
            y_own_normal = np.zeros(len(X_own_normal), dtype=np.int32)

            # Combine: own training + peer attacks + local normal
            X_own, y_own = agent_data[agent_id][0], agent_data[agent_id][1]
            X_absorb = np.vstack([X_own, X_peer_attacks, X_own_normal])
            y_absorb = np.concatenate([y_own, y_peer_attacks, y_own_normal])

            for _ in range(ABSORB_EPOCHS):
                agents[agent_id].fit(X_absorb, y_absorb, epochs=1, incremental=True)

    elif condition == "clause":
        # Extract top-K clauses from each agent
        clause_packets = {}
        for agent_id in AGENT_IDS:
            clause_packets[agent_id] = extract_top_clauses(
                agents[agent_id], n_bits, CLAUSE_TOP_K
            )

        # All-to-all injection
        for agent_id in AGENT_IDS:
            for peer_id in AGENT_IDS:
                if peer_id != agent_id:
                    inject_clauses(agents[agent_id], n_bits, clause_packets[peer_id])

    # Post-share evaluation
    post_results = {}
    for setting, (X_test, y_test, labels) in eval_settings.items():
        post_results[setting] = {}
        for agent_id in AGENT_IDS:
            overall, per_attack, seen, unseen = evaluate_agent(
                agents[agent_id], X_test, y_test, labels, agent_attacks[agent_id]
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
