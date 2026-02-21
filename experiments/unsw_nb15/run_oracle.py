#!/usr/bin/env python3
"""
run_oracle.py — Train a single centralized TM on the entire dataset.

This script trains a "Centralized Oracle" Tsetlin Machine on all training data
(all 9 attack types combined) to establish a baseline. We want to see if the
collective intelligence approach (where each agent specializes) can match or
even outperform a monolithic centralized model.
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

# Hyperparameters (same as collective nodes)
TM_CLAUSES = 2000
TM_T = 15000
TM_S = 2.0
TRAIN_EPOCHS = 10

def load_data():
    """Load and combine all agent training data."""
    X_train_pieces = []
    y_train_pieces = []
    
    for agent_id in ["A", "B", "C"]:
        path = os.path.join(DATA_DIR, f"agent_{agent_id}.json")
        with open(path) as f:
            d = json.load(f)
            X_train_pieces.append(np.array(d["X"], dtype=np.int32))
            y_train_pieces.append(np.array(d["y"], dtype=np.int32))
            
    X_train = np.vstack(X_train_pieces)
    y_train = np.concatenate(y_train_pieces)
    
    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    return X_train, y_train

def load_test_data():
    """Load the full test set."""
    path = os.path.join(DATA_DIR, "test_data.json")
    with open(path) as f:
        d = json.load(f)
    return np.array(d["X"], dtype=np.int32), np.array(d["y"], dtype=np.int32), d["labels"]

def main():
    print("Loading data...")
    X_train, y_train = load_data()
    X_test, y_test, labels_test = load_test_data()
    
    print(f"Loaded ALL combined training data: {X_train.shape[0]} samples")
    print(f"Loaded ALL test data: {X_test.shape[0]} samples")
    
    seeds = [42, 123, 456, 789, 1024]
    overall_accs = []
    
    print(f"\nTraining Centralized Oracle TM across {len(seeds)} seeds...")
    
    for seed in seeds:
        print(f"\n--- Run with Seed {seed} ---")
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X_train))
        X_trn = X_train[perm]
        y_trn = y_train[perm]
        
        t0 = time.time()
        tm = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, boost_true_positive_feedback=1)
        tm.fit(X_trn, y_trn, epochs=TRAIN_EPOCHS)
        
        preds = tm.predict(X_test)
        acc = np.mean(preds == y_test)
        overall_accs.append(acc)
        print(f"Seed {seed} finished in {time.time()-t0:.1f}s - Accuracy: {acc:.4f}")
    
    mean_acc = np.mean(overall_accs)
    std_acc = np.std(overall_accs)
    print("\n=== Oracle Variance Report ===")
    print(f"Accuracies: {', '.join(f'{a:.4f}' for a in overall_accs)}")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Detailed breakdown for the final seed run
    print("\nDetailed breakdown (last run):")
    attacks = sorted(list(set(labels_test)))
    for atk in attacks:
        indices = [i for i, l in enumerate(labels_test) if l == atk]
        if not indices:
            continue
        acc = np.mean(preds[indices] == y_test[indices])
        print(f"  {atk:<20} {acc:.4f}  ({len(indices)} samples)")

if __name__ == "__main__":
    main()
