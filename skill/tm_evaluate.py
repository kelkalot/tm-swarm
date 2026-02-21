#!/usr/bin/env python3
# tm_evaluate.py
"""
Evaluate this agent's TM on a test set.

Usage:
  python tm_evaluate.py \\
    --workspace /path/to/workspace \\
    --agent agent_a \\
    --X '[[1,0,1,...],...]' \\
    --y '[0,1,0,...]'

Output (stdout, JSON):
  {"ok": true, "accuracy": 0.823, "fitted": true, "n_train_samples": 500}
  {"ok": true, "accuracy": 0.5,   "fitted": false}
"""

import argparse
import json
import sys
import numpy as np

try:
    import tm_lib
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import tm_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--agent",     required=True)
    parser.add_argument("--X",         required=True)
    parser.add_argument("--y",         required=True)
    args = parser.parse_args()

    try:
        X_test = np.array(json.loads(args.X), dtype=np.uint32)
        y_test = np.array(json.loads(args.y), dtype=np.uint32)
    except (json.JSONDecodeError, ValueError) as e:
        print(tm_lib.result_json(False, error=f"Failed to parse --X or --y: {e}"))
        sys.exit(1)

    n_features = X_test.shape[1]
    tm, fitted = tm_lib.load_or_create_tm(args.workspace, args.agent, n_features)

    if not fitted:
        print(tm_lib.result_json(True, accuracy=0.5, fitted=False, n_train_samples=0))
        return

    acc = float(np.mean(tm.predict(X_test) == y_test))
    X_buf, y_buf = tm_lib.load_buffer(args.workspace, args.agent)

    print(tm_lib.result_json(True, accuracy=acc, fitted=True, n_train_samples=len(X_buf)))


if __name__ == "__main__":
    main()
