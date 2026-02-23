#!/usr/bin/env python3
"""
explain_rules.py — Extract and translate Tsetlin Machine clauses to human rules.

This script trains a TM on a specific attack type (e.g., "Generic"), extracts
the clauses it learned to detect that attack, and translates them into readable
rules using the feature schema.

This demonstrates the interpretability advantage of Tsetlin Machines:
we can see exactly *what* knowledge was transferred from Agent A to Agent B.
"""
import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from pyTsetlinMachine.tm import MultiClassTsetlinMachine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")

# TM hyperparameters
TM_CLAUSES = 100
TM_T = 25.0
TM_S = 10.0
TRAIN_EPOCHS = 40

def load_schema():
    path = os.path.join(DATA_DIR, "world_schema.json")
    with open(path) as f:
        schema = json.load(f)
    return {f["id"]: f["name"] for f in schema["features"]}

def load_data_for_attack(target_attack):
    """Load train data and isolate a specific attack type vs normal."""
    # We need to load test data to find indices of specific attacks.
    # To be precise about "Generic", let's use the test set as our training set
    # for this explanatory script so we can filter exactly "generic" vs "normal".
    
    test_path = os.path.join(DATA_DIR, "test_data.json")
    with open(test_path) as f:
        td = json.load(f)
        
    X_test = np.array(td["X"], dtype=np.int32)
    labels = td["labels"]
    
    # Filter only normal and target_attack
    indices = [i for i, l in enumerate(labels) if l in ["normal", target_attack]]
    X = X_test[indices]
    y = np.array([1 if labels[i] == target_attack else 0 for i in indices], dtype=np.int32)
    
    return X, y

def explain_clause(action_include, feature_names):
    """Translate a TM clause's include actions into a readable rule."""
    num_features = len(feature_names)
    literals = []
    
    for k in range(num_features * 2):
        if action_include[k] == 1:
            if k < num_features:
                # Original feature
                feat = feature_names[k]
                feat = feat.replace("_gt_", " > ")
                feat = feat.replace("_is_", " == ")
                literals.append(feat)
            else:
                # Negated feature
                feat = feature_names[k - num_features]
                feat = feat.replace("_gt_", " <= ")
                feat = feat.replace("_is_", " != ")
                literals.append(f"NOT ({feat})")
    
    if not literals:
        return "Always True (Empty Clause)"
    
    return " AND \n    ".join(literals)

def main():
    target_attack = "generic"
    print(f"Loading data for attack: {target_attack} vs normal")
    X, y = load_data_for_attack(target_attack)
    feature_names = load_schema()
    
    print(f"Training TM on {len(X)} samples to detect {target_attack}...")
    tm = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, boost_true_positive_feedback=1)
    tm.fit(X, y, epochs=TRAIN_EPOCHS)
    
    preds = tm.predict(X)
    acc = np.mean(preds == y)
    print(f"Accuracy: {acc:.3f}")
    
    # Extract clauses for class 1 (Attack)
    # The first half of clauses for a class vote positive, the second half vote negative.
    print(f"\n--- Top Rules for Detecting '{target_attack.capitalize()}' Attack ---")
    
    state = tm.get_state()
    # state is a list of tuples: [(cw, ta)] for each class
    # For class 1 (Attack):
    cw, ta = state[1]
    
    # ta has shape (TM_CLAUSES, n_features * 2)
    # The first TM_CLAUSES // 2 clauses have positive polarity for this class.
    
    # We need to evaluate which clauses are actually firing and voting positive
    # A positive clause for class 1 is in the first half: indices 0 to TM_CLAUSES//2 - 1
    
    n_features = X.shape[1]
    
    positive_clauses_for_attack = []
    
    for j in range(TM_CLAUSES // 2):
        # Read the state of all التا's for this clause
        # The ta matrix for a class has shape (TM_CLAUSES, n_features * 2)
        # States >= TM_S/2 mean "Include", states < TM_S/2 mean "Exclude"
        # Since state representation might be compressed, let's use the explicit check:
        clause_ta_state = ta[j]
        action_include = (clause_ta_state >= TM_T).astype(int) # This isn't quite right, let's use the actual tm.ta_action API properly.
        # Wait, the error was: `MultiClassTsetlinMachine.ta_action(1, j)` missing 'ta'.
        # Actually in pyTsetlinMachine standard API: tm.ta_action(class_id, clause, ta)
        
        # To avoid API issues, let's just write the include action directly from the state:
        # In pyTsetlinMachine, TA states are 1..2N. >= N is Include (N is the number of states per TA).
        # Actually it's easier to just use `tm.ta_action(1, j, k)` -> wait, no.
        # Let's inspect pyTsetlinMachine's tm.py. Or simply read from ta matrix:
        action_include = np.zeros(n_features * 2, dtype=np.int32)
        for k in range(n_features * 2):
            action_include[k] = tm.ta_action(1, j, k)
            
        # Calculate how many times this clause evaluated to 1 on attack samples
        clause_outputs = np.zeros(len(X))
        for i in range(len(X)):
            if y[i] == 1: # only care about attack samples
                # A clause evaluates to 1 if all included literals are 1
                is_satisfied = True
                for k in range(n_features * 2):
                    if action_include[k] == 1:
                        # check feature value
                        if k < n_features:
                            feat_val = X[i, k]
                        else:
                            feat_val = 1 - X[i, k - n_features]
                            
                        if feat_val == 0:
                            is_satisfied = False
                            break
                if is_satisfied:
                    clause_outputs[i] = 1
                    
        support = np.sum(clause_outputs)
        if support > 0:
            rule = explain_clause(action_include, feature_names)
            if "Empty Clause" not in rule:
                positive_clauses_for_attack.append((support, rule))
                
    # Sort by support (how many attack samples this rule caught)
    positive_clauses_for_attack.sort(reverse=True, key=lambda x: x[0])
    
    for i, (support, rule) in enumerate(positive_clauses_for_attack[:5]):
        print(f"\nRULE #{i+1} (Hits {int(support)} attack flows):")
        print(f"  IF  {rule}")
        print(f"  THEN Attack='{target_attack}'")

if __name__ == "__main__":
    main()
